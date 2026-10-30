//! NVIDIA's Parakeet TDT: a FastConformer encoder and a token-and-duration transducer (TDT)
//! decoder, run with greedy decoding.
//!
//! Weights are read from the Hugging Face `model.safetensors` (the `ParakeetForTDT` layout),
//! with `config.json` and `tokenizer.json` beside it.

use std::path::Path;
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use crate::features::{FrontEnd, FrontEndConfig};
use crate::gguf::{self, Gguf};
use crate::json::{self, Value};
use crate::kernels::{
    Act, AlignedBuf, BView, Epilogue, Isa, Kernels, PackedWeights, TILE_ROWS, for_rows,
};
use crate::pack::{MemorySink, Pack, Sink};
use crate::quant::Precision;
use crate::region::Vector;
use crate::safetensors::SafeTensors;
use crate::source::{GgufSource, Matrix, Source};
use crate::threads::{Pool, SharedMut};
use crate::tokenizer::Tokenizer;

/// The model name packs of Parakeet carry.
pub const PACK_MODEL: &str = "parakeet-tdt";

/// How a model runs.
#[derive(Debug, Clone)]
pub struct Options {
    /// Threads in all, the calling thread included.
    pub threads: usize,
    /// The instruction set, or `None` for the best this CPU has.
    pub isa: Option<Isa>,
    /// The precision weights are converted to when a model is loaded from its downloaded
    /// files. A pack keeps the precision it was made in.
    pub precision: Precision,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            isa: None,
            precision: Precision::F32,
        }
    }
}

/// The architecture, read from `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub d_model: usize,
    pub heads: usize,
    pub layers: usize,
    pub ff: usize,
    pub conv_kernel: usize,
    pub sub_channels: usize,
    pub n_mels: usize,
    pub pred_hidden: usize,
    pub pred_layers: usize,
    /// Tokens plus the blank, which is the last.
    pub vocab: usize,
    pub blank: usize,
    pub durations: Vec<usize>,
    pub max_symbols: usize,
}

impl Config {
    pub fn from_json(bytes: &[u8]) -> Result<Config> {
        let root = json::parse(bytes)?;
        let bad = |m: String| {
            Error::Model(format!(
                "config.json doesn't describe a supported Parakeet TDT model. {m}"
            ))
        };
        let enc = root
            .get("encoder_config")
            .ok_or_else(|| bad("It has no encoder_config.".into()))?;
        let num = |v: &json::Value, key: &str| -> Result<usize> {
            v.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| bad(format!("{key} is missing.")))
        };
        let expect = |v: &json::Value, key: &str, want: &str| -> Result<()> {
            let got = v.get(key).map(|x| match x {
                json::Value::String(s) => s.clone(),
                json::Value::Bool(b) => b.to_string(),
                json::Value::Number(n) => n.clone(),
                _ => String::new(),
            });
            if got.as_deref() == Some(want) {
                Ok(())
            } else {
                Err(bad(format!("{key} isn't {want}.")))
            }
        };
        expect(&root, "model_type", "parakeet_tdt")?;
        expect(&root, "hidden_act", "relu")?;
        expect(enc, "hidden_act", "silu")?;
        expect(enc, "attention_bias", "false")?;
        expect(enc, "convolution_bias", "false")?;
        expect(enc, "scale_input", "false")?;
        expect(enc, "subsampling_factor", "8")?;
        expect(enc, "subsampling_conv_kernel_size", "3")?;
        expect(enc, "subsampling_conv_stride", "2")?;
        let durations = root
            .get("durations")
            .and_then(|d| d.as_array())
            .and_then(|d| {
                d.iter()
                    .map(|x| x.as_u64().map(|x| x as usize))
                    .collect::<Option<Vec<_>>>()
            })
            .ok_or_else(|| bad("durations is missing.".into()))?;
        let config = Config {
            d_model: num(enc, "hidden_size")?,
            heads: num(enc, "num_attention_heads")?,
            layers: num(enc, "num_hidden_layers")?,
            ff: num(enc, "intermediate_size")?,
            conv_kernel: num(enc, "conv_kernel_size")?,
            sub_channels: num(enc, "subsampling_conv_channels")?,
            n_mels: num(enc, "num_mel_bins")?,
            pred_hidden: num(&root, "decoder_hidden_size")?,
            pred_layers: num(&root, "num_decoder_layers")?,
            vocab: num(&root, "vocab_size")?,
            blank: num(&root, "blank_token_id")?,
            durations,
            max_symbols: num(&root, "max_symbols_per_step")?,
        };
        if config.blank + 1 != config.vocab
            || !config.d_model.is_multiple_of(config.heads)
            || !config.n_mels.is_multiple_of(8)
            || config.conv_kernel.is_multiple_of(2)
            || config.durations.is_empty()
        {
            return Err(bad("Its sizes don't fit together.".into()));
        }
        Ok(config)
    }

    fn head_dim(&self) -> usize {
        self.d_model / self.heads
    }
}

/// Time spent per step, summed over a pass.
#[derive(Debug, Clone, Default)]
pub struct Timings {
    pub steps: Vec<(&'static str, Duration)>,
    /// How many times each step ran, in the order of `steps`.
    pub calls: Vec<usize>,
}

impl Timings {
    pub fn add(&mut self, name: &'static str, d: Duration) {
        match self.steps.iter().position(|(n, _)| *n == name) {
            Some(i) => {
                self.steps[i].1 += d;
                self.calls[i] += 1;
            }
            None => {
                self.steps.push((name, d));
                self.calls.push(1);
            }
        }
    }

    /// How many times a step ran.
    pub fn count(&self, name: &str) -> usize {
        self.steps
            .iter()
            .zip(&self.calls)
            .filter(|((n, _), _)| *n == name)
            .map(|(_, c)| c)
            .sum()
    }

    pub fn get(&self, name: &str) -> Duration {
        self.steps
            .iter()
            .filter(|(n, _)| *n == name)
            .map(|(_, d)| *d)
            .sum()
    }

    pub fn total(&self) -> Duration {
        self.steps.iter().map(|(_, d)| *d).sum()
    }

    fn time<T>(&mut self, name: &'static str, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let out = f();
        self.add(name, start.elapsed());
        out
    }
}

/// One emitted token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    pub id: u32,
    /// The encoder frame (80 ms each) the token was emitted at.
    pub frame: u32,
    /// The duration in frames the model predicted with it.
    pub duration: u32,
}

/// The result of transcribing one utterance.
#[derive(Debug, Clone)]
pub struct Transcript {
    pub text: String,
    pub tokens: Vec<Token>,
    pub timings: Timings,
}

/// Encoder output, row-major `[frames, width]`.
#[derive(Debug, Clone)]
pub struct Encoded {
    pub frames: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

/// What greedy decoding computed along the way, for comparing with a reference.
#[derive(Debug, Clone, Default)]
pub struct DecodeTrace {
    /// The joint's projection of each encoder frame, `[frames, pred_hidden]`.
    pub enc_proj: Vec<f32>,
    /// The prediction network's output for the start state and after each token.
    pub preds: Vec<Vec<f32>>,
    /// The joint's logits of the first steps, with the frame of each.
    pub joints: Vec<(usize, Vec<f32>)>,
}

struct Subsampling {
    /// `[9, channels]`, tap first.
    conv0_w: Vector,
    conv0_b: Vector,
    dw1_w: Vector,
    dw1_b: Vector,
    pw1: PackedWeights,
    pw1_b: Vector,
    dw2_w: Vector,
    dw2_b: Vector,
    pw2: PackedWeights,
    pw2_b: Vector,
    /// Reads the last convolution's output channel-major (`c * freqs + f`), as PyTorch flattens it.
    linear: PackedWeights,
    linear_b: Vector,
}

struct Norm {
    gamma: Vector,
    beta: Vector,
}

struct Layer {
    norm_ff1: Norm,
    ff1_in: PackedWeights,
    ff1_out: PackedWeights,
    norm_att: Norm,
    /// q, k and v stacked, `[3 * d_model, d_model]`.
    qkv: PackedWeights,
    pos: PackedWeights,
    bias_u: Vector,
    bias_v: Vector,
    out: PackedWeights,
    norm_conv: Norm,
    pw1: PackedWeights,
    /// Depthwise weights `[kernel, d_model]` with the BatchNorm folded in.
    dw_w: Vector,
    dw_b: Vector,
    pw2: PackedWeights,
    norm_ff2: Norm,
    ff2_in: PackedWeights,
    ff2_out: PackedWeights,
    norm_out: Norm,
}

struct LstmLayer {
    /// `[W_ih | W_hh]`, `[4 * hidden, 2 * hidden]`, gates in PyTorch's order i, f, g, o.
    w: PackedWeights,
    /// `b_ih + b_hh`.
    b: Vector,
}

struct Decoder {
    embedding: Vector,
    lstm: Vec<LstmLayer>,
    pred_proj: PackedWeights,
    pred_proj_b: Vector,
    enc_proj: PackedWeights,
    enc_proj_b: Vector,
    joint: PackedWeights,
    joint_b: Vector,
}

/// A loaded Parakeet TDT model.
pub struct Parakeet {
    config: Config,
    front_end: FrontEnd,
    subsampling: Subsampling,
    layers: Vec<Layer>,
    decoder: Decoder,
    tokenizer: Tokenizer,
    kernels: Kernels,
    pool: Pool,
    precision: Option<Precision>,
}

/// The precision a pack's weight matrices are in, if they share one.
fn pack_precision(pack: &Pack) -> Option<Precision> {
    let mut found = None;
    for entry in pack.entries() {
        if let crate::pack::Kind::Matrix { precision, .. } = entry.kind {
            match found {
                None => found = Some(precision),
                Some(p) if p != precision => return None,
                _ => {}
            }
        }
    }
    found
}

/// Transposes a row-major `[rows, cols]` matrix.
fn transpose(w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = w[r * cols + c];
        }
    }
    out
}

/// Converts Parakeet's weights from `src` (named as in the Hugging Face checkpoint) into pack
/// entries in `precision`. BatchNorm is folded into the depthwise convolutions, q, k and v are
/// stacked, and each LSTM layer's two matrices are joined.
pub fn convert(
    src: &mut dyn Source,
    config_json: &[u8],
    tokenizer_json: &[u8],
    precision: Precision,
    sink: &mut dyn Sink,
) -> Result<()> {
    let config = Config::from_json(config_json)?;
    let tokenizer = Tokenizer::from_json(tokenizer_json)?;
    if tokenizer.len() != config.vocab {
        return Err(Error::Model(format!(
            "The tokenizer has {} tokens but the model {}.",
            tokenizer.len(),
            config.vocab
        )));
    }
    sink.bytes("config.json", config_json)?;
    sink.bytes("tokenizer.json", tokenizer_json)?;
    let c = &config;
    let (d, ch, h) = (c.d_model, c.sub_channels, c.pred_hidden);
    let matrix = |sink: &mut dyn Sink, name: &str, m: Matrix, n: usize, k: usize| {
        sink.matrix(name, &m.rows(), n, k, precision)
    };

    let sub_f = c.n_mels / 8;
    for (i, name) in [(0, "conv0"), (2, "dw1"), (5, "dw2")] {
        let w = src.tensor(
            &format!("encoder.subsampling.layers.{i}.weight"),
            &[ch, 1, 3, 3],
        )?;
        sink.f32(&format!("sub.{name}.w"), &[9, ch], &transpose(&w, ch, 9))?;
        let b = src.tensor(&format!("encoder.subsampling.layers.{i}.bias"), &[ch])?;
        sink.f32(&format!("sub.{name}.b"), &[ch], &b)?;
    }
    for (i, name) in [(3, "pw1"), (6, "pw2")] {
        let w = src.matrix(
            &format!("encoder.subsampling.layers.{i}.weight"),
            &[ch, ch, 1, 1],
        )?;
        matrix(sink, &format!("sub.{name}"), w, ch, ch)?;
        let b = src.tensor(&format!("encoder.subsampling.layers.{i}.bias"), &[ch])?;
        sink.f32(&format!("sub.{name}.b"), &[ch], &b)?;
    }
    let w = src.matrix("encoder.subsampling.linear.weight", &[d, ch * sub_f])?;
    matrix(sink, "sub.linear", w, d, ch * sub_f)?;
    sink.f32(
        "sub.linear.b",
        &[d],
        &src.tensor("encoder.subsampling.linear.bias", &[d])?,
    )?;

    for i in 0..c.layers {
        let (p, o) = (format!("encoder.layers.{i}"), format!("layers.{i}"));
        let norm = |sink: &mut dyn Sink, src: &mut dyn Source, from: &str, to: &str| {
            sink.f32(
                &format!("{o}.{to}.w"),
                &[d],
                &src.tensor(&format!("{p}.{from}.weight"), &[d])?,
            )?;
            sink.f32(
                &format!("{o}.{to}.b"),
                &[d],
                &src.tensor(&format!("{p}.{from}.bias"), &[d])?,
            )
        };
        norm(sink, src, "norm_feed_forward1", "norm_ff1")?;
        norm(sink, src, "norm_self_att", "norm_att")?;
        norm(sink, src, "norm_conv", "norm_conv")?;
        norm(sink, src, "norm_feed_forward2", "norm_ff2")?;
        norm(sink, src, "norm_out", "norm_out")?;
        for (ff, to) in [("feed_forward1", "ff1"), ("feed_forward2", "ff2")] {
            let w = src.matrix(&format!("{p}.{ff}.linear1.weight"), &[c.ff, d])?;
            matrix(sink, &format!("{o}.{to}_in"), w, c.ff, d)?;
            let w = src.matrix(&format!("{p}.{ff}.linear2.weight"), &[d, c.ff])?;
            matrix(sink, &format!("{o}.{to}_out"), w, d, c.ff)?;
        }
        let qkv = ["q", "k", "v"]
            .iter()
            .map(|x| src.matrix(&format!("{p}.self_attn.{x}_proj.weight"), &[d, d]))
            .collect::<Result<Vec<_>>>()?;
        matrix(sink, &format!("{o}.qkv"), Matrix::stack(qkv), 3 * d, d)?;
        let w = src.matrix(&format!("{p}.self_attn.relative_k_proj.weight"), &[d, d])?;
        matrix(sink, &format!("{o}.pos"), w, d, d)?;
        let w = src.matrix(&format!("{p}.self_attn.o_proj.weight"), &[d, d])?;
        matrix(sink, &format!("{o}.out"), w, d, d)?;
        for x in ["u", "v"] {
            let shape = [c.heads, c.head_dim()];
            let b = src.tensor(&format!("{p}.self_attn.bias_{x}"), &shape)?;
            sink.f32(&format!("{o}.bias_{x}"), &shape, &b)?;
        }
        let w = src.matrix(&format!("{p}.conv.pointwise_conv1.weight"), &[2 * d, d, 1])?;
        matrix(sink, &format!("{o}.pw1"), w, 2 * d, d)?;
        let w = src.matrix(&format!("{p}.conv.pointwise_conv2.weight"), &[d, d, 1])?;
        matrix(sink, &format!("{o}.pw2"), w, d, d)?;
        let gamma = src.tensor(&format!("{p}.conv.norm.weight"), &[d])?;
        let beta = src.tensor(&format!("{p}.conv.norm.bias"), &[d])?;
        let mean = src.tensor(&format!("{p}.conv.norm.running_mean"), &[d])?;
        let var = src.tensor(&format!("{p}.conv.norm.running_var"), &[d])?;
        let dw = src.tensor(
            &format!("{p}.conv.depthwise_conv.weight"),
            &[d, 1, c.conv_kernel],
        )?;
        let mut dw_w = vec![0f32; c.conv_kernel * d];
        let mut dw_b = vec![0f32; d];
        for ch in 0..d {
            let scale = gamma[ch] as f64 / (var[ch] as f64 + 1e-5).sqrt();
            for k in 0..c.conv_kernel {
                dw_w[k * d + ch] = (dw[ch * c.conv_kernel + k] as f64 * scale) as f32;
            }
            dw_b[ch] = (beta[ch] as f64 - mean[ch] as f64 * scale) as f32;
        }
        sink.f32(&format!("{o}.dw.w"), &[c.conv_kernel, d], &dw_w)?;
        sink.f32(&format!("{o}.dw.b"), &[d], &dw_b)?;
    }

    for l in 0..c.pred_layers {
        let w_ih = src.matrix(&format!("decoder.lstm.weight_ih_l{l}"), &[4 * h, h])?;
        let w_hh = src.matrix(&format!("decoder.lstm.weight_hh_l{l}"), &[4 * h, h])?;
        let w = Matrix::concat_inputs(w_ih, h, w_hh, h, 4 * h);
        matrix(sink, &format!("dec.lstm.{l}"), w, 4 * h, 2 * h)?;
        let b_ih = src.tensor(&format!("decoder.lstm.bias_ih_l{l}"), &[4 * h])?;
        let b_hh = src.tensor(&format!("decoder.lstm.bias_hh_l{l}"), &[4 * h])?;
        let b: Vec<f32> = b_ih.iter().zip(&b_hh).map(|(a, b)| a + b).collect();
        sink.f32(&format!("dec.lstm.{l}.b"), &[4 * h], &b)?;
    }
    let outputs = c.vocab + c.durations.len();
    sink.f32(
        "dec.embedding",
        &[c.vocab, h],
        &src.tensor("decoder.embedding.weight", &[c.vocab, h])?,
    )?;
    for (from, to, n, k) in [
        ("decoder.decoder_projector", "dec.pred_proj", h, h),
        ("encoder_projector", "dec.enc_proj", h, d),
        ("joint.head", "dec.joint", outputs, h),
    ] {
        let w = src.matrix(&format!("{from}.weight"), &[n, k])?;
        matrix(sink, to, w, n, k)?;
        sink.f32(
            &format!("{to}.b"),
            &[n],
            &src.tensor(&format!("{from}.bias"), &[n])?,
        )?;
    }
    Ok(())
}

/// NVIDIA's GGUF names for the Hugging Face checkpoint's names.
fn nemo_name(name: &str) -> String {
    let fixed = [
        ("encoder.subsampling.layers.", "encoder.pre_encode.conv."),
        ("encoder.subsampling.linear.", "encoder.pre_encode.out."),
        ("decoder.embedding.", "decoder.prediction.embed."),
        ("decoder.decoder_projector.", "joint.pred."),
        ("encoder_projector.", "joint.enc."),
        ("joint.head.", "joint.joint_net.2."),
    ];
    for (from, to) in fixed {
        if let Some(rest) = name.strip_prefix(from) {
            return format!("{to}{rest}");
        }
    }
    if let Some(rest) = name.strip_prefix("decoder.lstm.") {
        // weight_ih_l0 -> ih_l0.weight
        if let Some((kind, which)) = rest.split_once('_') {
            return format!("decoder.prediction.dec_rnn.lstm.{which}.{kind}");
        }
    }
    let replacements = [
        (".self_attn.q_proj.", ".self_attn.linear_q."),
        (".self_attn.k_proj.", ".self_attn.linear_k."),
        (".self_attn.v_proj.", ".self_attn.linear_v."),
        (".self_attn.o_proj.", ".self_attn.linear_out."),
        (".self_attn.relative_k_proj.", ".self_attn.linear_pos."),
        (".self_attn.bias_u", ".self_attn.pos_bias_u"),
        (".self_attn.bias_v", ".self_attn.pos_bias_v"),
        (".conv.norm.", ".conv.batch_norm."),
    ];
    let mut out = name.to_string();
    for (from, to) in replacements {
        out = out.replace(from, to);
    }
    out
}

/// A `config.json` equivalent to the one beside the Hugging Face checkpoint, written from the
/// GGUF file's metadata after checking the architecture and front end are Parakeet's.
fn gguf_config(g: &Gguf) -> Result<String> {
    let bad = |m: String| {
        Error::Model(format!(
            "The GGUF file doesn't describe a supported Parakeet TDT model. {m}"
        ))
    };
    let num = |key: &str| -> Result<u64> {
        g.get(key)
            .and_then(gguf::Value::as_u64)
            .ok_or_else(|| bad(format!("{key} is missing.")))
    };
    let expect = |key: &str, want: gguf::Value| -> Result<()> {
        if g.get(key) == Some(&want) {
            Ok(())
        } else {
            Err(bad(format!("{key} isn't {want:?}.")))
        }
    };
    let near = |key: &str, want: f64| -> Result<()> {
        match g.get(key).and_then(gguf::Value::as_f64) {
            Some(v) if (v - want).abs() < 1e-6 => Ok(()),
            _ => Err(bad(format!("{key} isn't {want}."))),
        }
    };
    let s = |v: &str| gguf::Value::String(v.into());
    expect("general.architecture", s("asr"))?;
    expect("asr.head_type", s("tdt"))?;
    expect("asr.encoder.xscaling", gguf::Value::Bool(false))?;
    expect("asr.encoder.use_bias", gguf::Value::Bool(false))?;
    expect("asr.encoder.conv_norm", s("batch_norm"))?;
    expect("asr.encoder.att_context_style", s("regular"))?;
    for key in [
        "asr.encoder.offline_left_ctx",
        "asr.encoder.offline_right_ctx",
    ] {
        expect(key, gguf::Value::Int(-1))?;
    }
    if num("asr.encoder.subsampling_factor")? != 8 {
        return Err(bad("The subsampling factor isn't 8.".into()));
    }
    let n_mels = num("asr.encoder.feat_in")?;
    if num("asr.preprocessor.sample_rate")? != 16000
        || num("asr.preprocessor.n_fft")? != 512
        || num("asr.preprocessor.features")? != n_mels
    {
        return Err(bad("Its front end isn't Parakeet's.".into()));
    }
    near("asr.preprocessor.window_size", 0.025)?;
    near("asr.preprocessor.window_stride", 0.01)?;
    near("asr.preprocessor.preemph", 0.97)?;
    expect("asr.preprocessor.normalize", s("per_feature"))?;
    expect("asr.preprocessor.hann_periodic", gguf::Value::Bool(false))?;
    expect(
        "asr.preprocessor.stft_center_window",
        gguf::Value::Bool(true),
    )?;
    let pred_hidden = num("asr.rnnt.pred_hidden")?;
    if num("asr.rnnt.joint_dim")? != pred_hidden || num("asr.rnnt.pred_embed_dim")? != pred_hidden {
        return Err(bad(
            "The joint and the prediction network differ in width.".into()
        ));
    }
    let durations: Vec<Value> = g
        .get("asr.tdt.durations")
        .and_then(gguf::Value::as_array)
        .and_then(|d| d.iter().map(|v| v.as_u64().map(Value::number)).collect())
        .ok_or_else(|| bad("asr.tdt.durations is missing.".into()))?;
    let n = |key: &str| -> Result<Value> { Ok(Value::number(num(key)?)) };
    let b = |v: bool| Value::Bool(v);
    let st = |v: &str| Value::String(v.into());
    let encoder = Value::Object(vec![
        ("hidden_act".into(), st("silu")),
        ("attention_bias".into(), b(false)),
        ("convolution_bias".into(), b(false)),
        ("scale_input".into(), b(false)),
        ("subsampling_factor".into(), Value::number(8)),
        ("subsampling_conv_kernel_size".into(), Value::number(3)),
        ("subsampling_conv_stride".into(), Value::number(2)),
        ("hidden_size".into(), n("asr.encoder.d_model")?),
        ("num_attention_heads".into(), n("asr.encoder.n_heads")?),
        ("num_hidden_layers".into(), n("asr.encoder.n_layers")?),
        ("intermediate_size".into(), n("asr.encoder.d_ff")?),
        (
            "conv_kernel_size".into(),
            n("asr.encoder.conv_kernel_size")?,
        ),
        (
            "subsampling_conv_channels".into(),
            n("asr.encoder.subsampling_conv_channels")?,
        ),
        ("num_mel_bins".into(), Value::number(n_mels)),
    ]);
    let root = Value::Object(vec![
        ("model_type".into(), st("parakeet_tdt")),
        ("hidden_act".into(), st("relu")),
        ("encoder_config".into(), encoder),
        ("decoder_hidden_size".into(), Value::number(pred_hidden)),
        ("num_decoder_layers".into(), n("asr.rnnt.pred_num_layers")?),
        ("vocab_size".into(), n("asr.rnnt.vocab_size")?),
        ("blank_token_id".into(), n("asr.rnnt.blank_id")?),
        ("durations".into(), Value::Array(durations)),
        (
            "max_symbols_per_step".into(),
            n("asr.rnnt.max_symbols_per_step")?,
        ),
    ]);
    Ok(root.to_json())
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|e| Error::io(path, e))
}

/// Converts a Hugging Face checkpoint directory (`model.safetensors`, `config.json` and
/// `tokenizer.json`) or NVIDIA's GGUF file (with `tokenizer.json` beside it) into pack entries.
pub fn convert_path(path: &Path, precision: Precision, sink: &mut dyn Sink) -> Result<()> {
    if path.is_dir() {
        let config = read_file(&path.join("config.json"))?;
        let tokenizer = read_file(&path.join("tokenizer.json"))?;
        let mut st = SafeTensors::open(&path.join("model.safetensors"))?;
        return convert(&mut st, &config, &tokenizer, precision, sink);
    }
    let tokenizer_path = path.with_file_name("tokenizer.json");
    let tokenizer = read_file(&tokenizer_path)?;
    let mut g = Gguf::open(path)?;
    let config = gguf_config(&g)?;
    let pieces: Vec<&str> = g
        .get("asr.tokenizer.vocab")
        .and_then(gguf::Value::as_array)
        .map(|v| v.iter().filter_map(gguf::Value::as_str).collect())
        .unwrap_or_default();
    if !Tokenizer::from_json(&tokenizer)?.has_pieces(&pieces) {
        return Err(Error::Model(format!(
            "{} doesn't hold the GGUF file's vocabulary.",
            tokenizer_path.display()
        )));
    }
    let mut src = GgufSource {
        gguf: &mut g,
        rename: nemo_name,
    };
    convert(&mut src, config.as_bytes(), &tokenizer, precision, sink)
}

/// Pack metadata describing a conversion.
pub fn pack_metadata(path: &Path, precision: Precision) -> Vec<(String, String)> {
    let source = path
        .file_name()
        .map_or(String::new(), |n| n.to_string_lossy().into_owned());
    let bytes = if path.is_dir() {
        std::fs::metadata(path.join("model.safetensors")).map(|m| m.len())
    } else {
        std::fs::metadata(path).map(|m| m.len())
    };
    vec![
        ("precision".into(), precision.name().into()),
        ("source".into(), source),
        (
            "source_bytes".into(),
            bytes.map_or(String::new(), |b| b.to_string()),
        ),
        (
            "engine".into(),
            format!("notate {}", env!("CARGO_PKG_VERSION")),
        ),
    ]
}

impl Parakeet {
    /// Loads a model from a pack file, a Hugging Face checkpoint directory (`model.safetensors`,
    /// `config.json` and `tokenizer.json`) or NVIDIA's GGUF file (with `tokenizer.json` beside
    /// it). A pack is mapped as it is. The others are converted in memory to
    /// `options.precision` (a GGUF file's 8-bit blocks are kept unchanged for Q8_0).
    pub fn load(path: &Path, options: &Options) -> Result<Parakeet> {
        let is_pack = path.is_file() && path.extension().and_then(|e| e.to_str()) != Some("gguf");
        if is_pack {
            return Parakeet::from_pack(&Pack::open(path)?, options);
        }
        let mut sink = MemorySink::new(PACK_MODEL, pack_metadata(path, options.precision));
        convert_path(path, options.precision, &mut sink)?;
        Parakeet::from_pack(&sink.finish()?, options)
    }

    /// Builds the model from a pack's entries, which it keeps views of.
    pub fn from_pack(pack: &Pack, options: &Options) -> Result<Parakeet> {
        if pack.model() != PACK_MODEL {
            return Err(Error::Model(format!(
                "The pack holds {}, not Parakeet TDT.",
                pack.model()
            )));
        }
        let isa = options.isa.unwrap_or_else(Isa::detect);
        let kernels = Kernels::new(isa).ok_or_else(|| {
            Error::Model(format!("This CPU can't run the {} kernels.", isa.name()))
        })?;
        let config = Config::from_json(pack.bytes("config.json")?)?;
        let tokenizer = Tokenizer::from_json(pack.bytes("tokenizer.json")?)?;
        if tokenizer.len() != config.vocab {
            return Err(Error::Model(format!(
                "The tokenizer has {} tokens but the model {}.",
                tokenizer.len(),
                config.vocab
            )));
        }
        let c = &config;
        let (d, ch, h) = (c.d_model, c.sub_channels, c.pred_hidden);
        let v = |name: &str, shape: &[usize]| pack.vector(name, shape);
        let m = |name: &str, n: usize, k: usize| pack.matrix(name, n, k);
        let sub_f = c.n_mels / 8;
        let subsampling = Subsampling {
            conv0_w: v("sub.conv0.w", &[9, ch])?,
            conv0_b: v("sub.conv0.b", &[ch])?,
            dw1_w: v("sub.dw1.w", &[9, ch])?,
            dw1_b: v("sub.dw1.b", &[ch])?,
            pw1: m("sub.pw1", ch, ch)?,
            pw1_b: v("sub.pw1.b", &[ch])?,
            dw2_w: v("sub.dw2.w", &[9, ch])?,
            dw2_b: v("sub.dw2.b", &[ch])?,
            pw2: m("sub.pw2", ch, ch)?,
            pw2_b: v("sub.pw2.b", &[ch])?,
            linear: m("sub.linear", d, ch * sub_f)?,
            linear_b: v("sub.linear.b", &[d])?,
        };
        let mut layers = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            let p = format!("layers.{i}");
            let norm = |name: &str| -> Result<Norm> {
                Ok(Norm {
                    gamma: v(&format!("{p}.{name}.w"), &[d])?,
                    beta: v(&format!("{p}.{name}.b"), &[d])?,
                })
            };
            let hd = [c.heads, c.head_dim()];
            layers.push(Layer {
                norm_ff1: norm("norm_ff1")?,
                ff1_in: m(&format!("{p}.ff1_in"), c.ff, d)?,
                ff1_out: m(&format!("{p}.ff1_out"), d, c.ff)?,
                norm_att: norm("norm_att")?,
                qkv: m(&format!("{p}.qkv"), 3 * d, d)?,
                pos: m(&format!("{p}.pos"), d, d)?,
                bias_u: v(&format!("{p}.bias_u"), &hd)?,
                bias_v: v(&format!("{p}.bias_v"), &hd)?,
                out: m(&format!("{p}.out"), d, d)?,
                norm_conv: norm("norm_conv")?,
                pw1: m(&format!("{p}.pw1"), 2 * d, d)?,
                dw_w: v(&format!("{p}.dw.w"), &[c.conv_kernel, d])?,
                dw_b: v(&format!("{p}.dw.b"), &[d])?,
                pw2: m(&format!("{p}.pw2"), d, d)?,
                norm_ff2: norm("norm_ff2")?,
                ff2_in: m(&format!("{p}.ff2_in"), c.ff, d)?,
                ff2_out: m(&format!("{p}.ff2_out"), d, c.ff)?,
                norm_out: norm("norm_out")?,
            });
        }
        let lstm = (0..c.pred_layers)
            .map(|l| {
                Ok(LstmLayer {
                    w: m(&format!("dec.lstm.{l}"), 4 * h, 2 * h)?,
                    b: v(&format!("dec.lstm.{l}.b"), &[4 * h])?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let outputs = c.vocab + c.durations.len();
        let decoder = Decoder {
            embedding: v("dec.embedding", &[c.vocab, h])?,
            lstm,
            pred_proj: m("dec.pred_proj", h, h)?,
            pred_proj_b: v("dec.pred_proj.b", &[h])?,
            enc_proj: m("dec.enc_proj", h, d)?,
            enc_proj_b: v("dec.enc_proj.b", &[h])?,
            joint: m("dec.joint", outputs, h)?,
            joint_b: v("dec.joint.b", &[outputs])?,
        };
        let front_end = FrontEnd::new(FrontEndConfig {
            n_mels: c.n_mels,
            ..FrontEndConfig::parakeet()
        });
        Ok(Parakeet {
            config,
            front_end,
            subsampling,
            layers,
            decoder,
            tokenizer,
            kernels,
            pool: Pool::new(options.threads),
            precision: pack_precision(pack),
        })
    }

    /// The precision of the encoder's weights.
    pub fn precision(&self) -> Option<Precision> {
        self.precision
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn isa(&self) -> Isa {
        self.kernels.isa()
    }

    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Bytes of weights held in memory.
    pub fn weight_bytes(&self) -> usize {
        let packed = |w: &PackedWeights| w.bytes();
        let s = &self.subsampling;
        let mut total = packed(&s.pw1) + packed(&s.pw2) + packed(&s.linear);
        for l in &self.layers {
            total += [
                &l.ff1_in, &l.ff1_out, &l.qkv, &l.pos, &l.out, &l.pw1, &l.pw2, &l.ff2_in,
                &l.ff2_out,
            ]
            .iter()
            .map(|w| packed(w))
            .sum::<usize>();
        }
        let dec = &self.decoder;
        total += dec.embedding.len() * 4
            + packed(&dec.pred_proj)
            + packed(&dec.enc_proj)
            + packed(&dec.joint);
        total += dec.lstm.iter().map(|l| packed(&l.w)).sum::<usize>();
        total
    }

    /// Transcribes 16 kHz mono samples.
    pub fn transcribe(&self, samples: &[f32]) -> Result<Transcript> {
        let mut timings = Timings::default();
        let features = timings.time("features", || self.front_end.compute(samples))?;
        let encoded = self.encode(
            &features.data,
            features.frames,
            &mut timings,
            &mut |_, _, _| {},
        );
        let tokens = self.decode(&encoded, &mut timings, None);
        let ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
        let text = timings.time("detokenize", || self.tokenizer.decode(&ids));
        Ok(Transcript {
            text,
            tokens,
            timings,
        })
    }

    /// Computes the normalized log-mel features, `[frames, n_mels]`.
    pub fn features(&self, samples: &[f32]) -> Result<crate::features::Features> {
        self.front_end.compute(samples)
    }

    /// Runs the encoder over `frames` feature frames. `tap` sees the subsampling output and
    /// each layer's output (`"subsampling"`, `"layer.0"`, ...) as `[rows, d_model]`.
    pub fn encode(
        &self,
        features: &[f32],
        frames: usize,
        timings: &mut Timings,
        tap: &mut dyn FnMut(&str, usize, &[f32]),
    ) -> Encoded {
        let mut x = timings.time("subsampling", || self.subsample(features, frames));
        let d = self.config.d_model;
        let t = x.len() / d;
        tap("subsampling", t, &x);
        let pos = timings.time("positions", || positions(t, d));
        let mut scratch = Scratch::new(t, &self.config);
        for (i, layer) in self.layers.iter().enumerate() {
            self.layer(layer, &mut x, t, &pos, &mut scratch, timings);
            tap(&format!("layer.{i}"), t, &x);
        }
        Encoded {
            frames: t,
            width: d,
            data: x,
        }
    }

    fn subsample(&self, features: &[f32], frames: usize) -> Vec<f32> {
        let s = &self.subsampling;
        let k = &self.kernels;
        let ch = self.config.sub_channels;
        let out_len = |n: usize| (n - 1) / 2 + 1;
        let (t0, f0) = (frames, self.config.n_mels);
        let (t1, f1) = (out_len(t0), out_len(f0));
        let (t2, f2) = (out_len(t1), out_len(f1));
        let (t3, f3) = (out_len(t2), out_len(f2));

        let conv_rows = |input: &[f32],
                         in_t: usize,
                         in_f: usize,
                         depthwise: bool,
                         w: &[f32],
                         b: &[f32],
                         out_t: usize,
                         out_f: usize,
                         relu: bool| {
            let mut out = vec![0f32; out_t * out_f * ch];
            for_rows(
                &self.pool,
                &mut out,
                out_f * ch,
                out_t,
                4,
                &|first, part| {
                    for (r, row) in part.chunks_exact_mut(out_f * ch).enumerate() {
                        (k.conv3x3s2_row)(input, in_t, in_f, ch, depthwise, w, b, first + r, row);
                        if relu {
                            row.iter_mut().for_each(|v| *v = v.max(0.0));
                        }
                    }
                },
            );
            out
        };
        let relu = Epilogue {
            act: Act::Relu,
            ..Epilogue::PLAIN
        };
        let h1 = conv_rows(
            features, t0, f0, false, &s.conv0_w, &s.conv0_b, t1, f1, true,
        );
        let d1 = conv_rows(&h1, t1, f1, true, &s.dw1_w, &s.dw1_b, t2, f2, false);
        let mut h2 = vec![0f32; t2 * f2 * ch];
        k.linear(
            &self.pool,
            &d1,
            t2 * f2,
            &s.pw1,
            &mut h2,
            &Epilogue {
                bias: Some(&s.pw1_b),
                ..relu
            },
        );
        let d2 = conv_rows(&h2, t2, f2, true, &s.dw2_w, &s.dw2_b, t3, f3, false);
        let mut h3 = vec![0f32; t3 * f3 * ch];
        k.linear(
            &self.pool,
            &d2,
            t3 * f3,
            &s.pw2,
            &mut h3,
            &Epilogue {
                bias: Some(&s.pw2_b),
                ..relu
            },
        );
        // PyTorch flattens the last convolution's output channel by channel.
        let mut flat = vec![0f32; t3 * f3 * ch];
        for (src, dst) in h3.chunks_exact(f3 * ch).zip(flat.chunks_exact_mut(f3 * ch)) {
            for f in 0..f3 {
                for c in 0..ch {
                    dst[c * f3 + f] = src[f * ch + c];
                }
            }
        }
        let mut x = vec![0f32; t3 * self.config.d_model];
        k.linear(
            &self.pool,
            &flat,
            t3,
            &s.linear,
            &mut x,
            &Epilogue::bias(&s.linear_b),
        );
        x
    }

    fn norm(&self, norm: &Norm, x: &[f32], out: &mut [f32], t: usize) {
        let d = self.config.d_model;
        let k = &self.kernels;
        for_rows(&self.pool, out, d, t, 16, &|first, part| {
            for (r, row) in part.chunks_exact_mut(d).enumerate() {
                (k.layer_norm)(
                    &x[(first + r) * d..(first + r + 1) * d],
                    row,
                    &norm.gamma,
                    &norm.beta,
                    1e-5,
                );
            }
        });
    }

    fn layer(
        &self,
        l: &Layer,
        x: &mut [f32],
        t: usize,
        pos: &[f32],
        s: &mut Scratch,
        timings: &mut Timings,
    ) {
        let c = &self.config;
        let d = c.d_model;
        let k = &self.kernels;
        let pool = &self.pool;

        // Feed-forward 1, half-step residual.
        timings.time("layer_norm", || self.norm(&l.norm_ff1, x, &mut s.h, t));
        timings.time("ff_in", || {
            k.linear(
                pool,
                &s.h,
                t,
                &l.ff1_in,
                &mut s.big,
                &Epilogue {
                    act: Act::Silu,
                    ..Epilogue::PLAIN
                },
            )
        });
        timings.time("ff_out", || {
            k.linear(
                pool,
                &s.big,
                t,
                &l.ff1_out,
                x,
                &Epilogue {
                    scale: 0.5,
                    accumulate: true,
                    ..Epilogue::PLAIN
                },
            )
        });

        // Self-attention with relative positions.
        timings.time("layer_norm", || self.norm(&l.norm_att, x, &mut s.h, t));
        timings.time("qkv", || {
            k.linear(
                pool,
                &s.h,
                t,
                &l.qkv,
                &mut s.big[..t * 3 * d],
                &Epilogue::PLAIN,
            )
        });
        timings.time("pos_proj", || {
            k.linear(pool, pos, 2 * t - 1, &l.pos, &mut s.p, &Epilogue::PLAIN)
        });
        timings.time("attention", || self.attention(l, t, s));
        timings.time("att_out", || {
            k.linear(
                pool,
                &s.att,
                t,
                &l.out,
                x,
                &Epilogue {
                    accumulate: true,
                    ..Epilogue::PLAIN
                },
            )
        });

        // Convolution module.
        timings.time("layer_norm", || self.norm(&l.norm_conv, x, &mut s.h, t));
        timings.time("conv_pw1", || {
            k.linear(
                pool,
                &s.h,
                t,
                &l.pw1,
                &mut s.big[..t * 2 * d],
                &Epilogue::PLAIN,
            )
        });
        timings.time("glu", || {
            let big = &s.big;
            for_rows(pool, &mut s.h, d, t, 16, &|first, part| {
                for (r, row) in part.chunks_exact_mut(d).enumerate() {
                    let g = &big[(first + r) * 2 * d..(first + r + 1) * 2 * d];
                    (k.glu)(&g[..d], &g[d..], row);
                }
            });
        });
        timings.time("depthwise_conv", || {
            let input = &s.h[..t * d];
            for_rows(pool, &mut s.att, d, t, 8, &|first, part| {
                for (r, row) in part.chunks_exact_mut(d).enumerate() {
                    (k.dwconv1d_frame)(
                        input,
                        d,
                        t,
                        first + r,
                        &l.dw_w,
                        c.conv_kernel,
                        &l.dw_b,
                        row,
                    );
                    (k.silu)(row);
                }
            });
        });
        timings.time("conv_pw2", || {
            k.linear(
                pool,
                &s.att,
                t,
                &l.pw2,
                x,
                &Epilogue {
                    accumulate: true,
                    ..Epilogue::PLAIN
                },
            )
        });

        // Feed-forward 2, half-step residual.
        timings.time("layer_norm", || self.norm(&l.norm_ff2, x, &mut s.h, t));
        timings.time("ff_in", || {
            k.linear(
                pool,
                &s.h,
                t,
                &l.ff2_in,
                &mut s.big,
                &Epilogue {
                    act: Act::Silu,
                    ..Epilogue::PLAIN
                },
            )
        });
        timings.time("ff_out", || {
            k.linear(
                pool,
                &s.big,
                t,
                &l.ff2_out,
                x,
                &Epilogue {
                    scale: 0.5,
                    accumulate: true,
                    ..Epilogue::PLAIN
                },
            )
        });

        timings.time("layer_norm", || {
            self.norm(&l.norm_out, x, &mut s.h, t);
            x.copy_from_slice(&s.h[..t * d]);
        });
    }

    /// Multi-head attention with Transformer-XL relative positions. Reads q, k and v from
    /// `s.big` and the projected positions from `s.p`, writes the heads' outputs to `s.att`.
    fn attention(&self, l: &Layer, t: usize, s: &mut Scratch) {
        let c = &self.config;
        let (d, hd, heads) = (c.d_model, c.head_dim(), c.heads);
        let npos = 2 * t - 1;
        let (kt_ld, pt_ld) = (s.kt_ld, s.pt_ld);
        let k = &self.kernels;

        // Transposed keys and positions per head, `[hd, t]` and `[hd, 2t - 1]`, so the score
        // products read them as 16-column panels.
        {
            let qkv = &s.big;
            let p = &s.p;
            let kt = SharedMut(s.kt.as_mut_ptr());
            let pt = SharedMut(s.pt.as_mut_ptr());
            self.pool.run(heads, &|h| {
                // SAFETY: each head writes its own block of `kt` and `pt`.
                let kt = unsafe {
                    std::slice::from_raw_parts_mut(kt.get().add(h * hd * kt_ld), hd * kt_ld)
                };
                let pt = unsafe {
                    std::slice::from_raw_parts_mut(pt.get().add(h * hd * pt_ld), hd * pt_ld)
                };
                for j in 0..t {
                    let key = &qkv[j * 3 * d + d + h * hd..][..hd];
                    for (i, &v) in key.iter().enumerate() {
                        kt[i * kt_ld + j] = v;
                    }
                }
                for j in 0..npos {
                    let row = &p[j * d + h * hd..][..hd];
                    for (i, &v) in row.iter().enumerate() {
                        pt[i * pt_ld + j] = v;
                    }
                }
            });
        }

        let scale = 1.0 / (hd as f32).sqrt();
        let blocks = t.div_ceil(ATT_BLOCK);
        let qkv = &s.big;
        let (kt, pt) = (&s.kt, &s.pt);
        let att = SharedMut(s.att.as_mut_ptr());
        self.pool.run(heads * blocks, &|task| {
            let (h, blk) = (task / blocks, task % blocks);
            let (i0, i1) = (blk * ATT_BLOCK, ((blk + 1) * ATT_BLOCK).min(t));
            let rows = i1 - i0;
            let mut qu = vec![0f32; rows * hd];
            let mut qv = vec![0f32; rows * hd];
            for r in 0..rows {
                let q = &qkv[(i0 + r) * 3 * d + h * hd..][..hd];
                for i in 0..hd {
                    qu[r * hd + i] = q[i] + l.bias_u[h * hd + i];
                    qv[r * hd + i] = q[i] + l.bias_v[h * hd + i];
                }
            }
            // Content scores against every key.
            let mut scores = vec![0f32; rows * t];
            let kt_h = &kt[h * hd * kt_ld..(h + 1) * hd * kt_ld];
            k.matmul_serial(
                &qu,
                hd,
                rows,
                &BView::row_major(kt_h, hd, t, kt_ld),
                &mut scores,
                t,
                &Epilogue::PLAIN,
            );
            // Position scores. Row i needs position index t - 1 - i + j for key j, so the block
            // needs indices t - i1 to 2t - 2 - i0.
            let width = t + rows - 1;
            let mut bd = vec![0f32; rows * width];
            let pt_h = &pt[h * hd * pt_ld + (t - i1)..(h + 1) * hd * pt_ld];
            k.matmul_serial(
                &qv,
                hd,
                rows,
                &BView::row_major(pt_h, hd, width, pt_ld),
                &mut bd,
                width,
                &Epilogue::PLAIN,
            );
            for r in 0..rows {
                let i = i0 + r;
                let shift = i1 - 1 - i;
                let row = &mut scores[r * t..(r + 1) * t];
                let pos = &bd[r * width + shift..r * width + shift + t];
                for (sv, &pv) in row.iter_mut().zip(pos) {
                    *sv = (*sv + pv) * scale;
                }
                (k.softmax)(row);
            }
            let v = &qkv[2 * d + h * hd..];
            // SAFETY: each task writes rows i0..i1 of its own head's columns, inside `att`.
            unsafe {
                let out = SharedMut(att.get().add(i0 * d + h * hd));
                k.matmul_serial_raw(
                    &scores,
                    t,
                    rows,
                    &BView::row_major(v, t, hd, 3 * d),
                    out,
                    d,
                    &Epilogue::PLAIN,
                );
            }
        });
    }

    /// Greedy TDT decoding. With `trace`, it also records the intermediate values.
    pub fn decode(
        &self,
        encoded: &Encoded,
        timings: &mut Timings,
        mut trace: Option<&mut DecodeTrace>,
    ) -> Vec<Token> {
        let c = &self.config;
        let dec = &self.decoder;
        let k = &self.kernels;
        let pool = &self.pool;
        let h = c.pred_hidden;
        let frames = encoded.frames;

        let mut enc_proj = vec![0f32; frames * h];
        timings.time("enc_proj", || {
            k.linear(
                pool,
                &encoded.data,
                frames,
                &dec.enc_proj,
                &mut enc_proj,
                &Epilogue::bias(&dec.enc_proj_b),
            )
        });

        let mut state = vec![0f32; c.pred_layers * 2 * h];
        let mut pred = vec![0f32; h];
        // The start state is the blank token's input, as in NeMo.
        timings.time("predict", || self.predict(c.blank, &mut state, &mut pred));
        if let Some(trace) = trace.as_deref_mut() {
            trace.enc_proj = enc_proj.clone();
            trace.preds.push(pred.clone());
        }

        let outputs = c.vocab + c.durations.len();
        let mut z = vec![0f32; h];
        let mut logits = vec![0f32; outputs];
        let mut tokens = Vec::new();
        let mut t = 0;
        let mut on_frame = 0;
        while t < frames {
            timings.time("joint", || {
                for ((zi, &e), &p) in z.iter_mut().zip(&enc_proj[t * h..(t + 1) * h]).zip(&pred) {
                    *zi = (e + p).max(0.0);
                }
                k.linear(
                    pool,
                    &z,
                    1,
                    &dec.joint,
                    &mut logits,
                    &Epilogue::bias(&dec.joint_b),
                );
            });
            if let Some(trace) = trace.as_deref_mut()
                && trace.joints.len() < MAX_TRACED_JOINTS
            {
                trace.joints.push((t, logits.clone()));
            }
            let token = argmax(&logits[..c.vocab]);
            let duration = c.durations[argmax(&logits[c.vocab..])];
            if token == c.blank {
                t += duration.max(1);
                on_frame = 0;
                continue;
            }
            tokens.push(Token {
                id: token as u32,
                frame: t as u32,
                duration: duration as u32,
            });
            timings.time("predict", || self.predict(token, &mut state, &mut pred));
            if let Some(trace) = trace.as_deref_mut() {
                trace.preds.push(pred.clone());
            }
            if duration > 0 {
                t += duration;
                on_frame = 0;
            } else {
                on_frame += 1;
                if on_frame >= c.max_symbols {
                    t += 1;
                    on_frame = 0;
                }
            }
        }
        tokens
    }

    /// Runs the prediction network on one token, updating the LSTM state and writing the
    /// projected output to `pred`.
    fn predict(&self, token: usize, state: &mut [f32], pred: &mut [f32]) {
        let c = &self.config;
        let dec = &self.decoder;
        let k = &self.kernels;
        let h = c.pred_hidden;
        let mut input = vec![0f32; 2 * h];
        input[..h].copy_from_slice(&dec.embedding[token * h..(token + 1) * h]);
        let mut gates = vec![0f32; 4 * h];
        for (l, layer) in dec.lstm.iter().enumerate() {
            let (hs, cs) = state[l * 2 * h..(l + 1) * 2 * h].split_at_mut(h);
            input[h..].copy_from_slice(hs);
            k.linear(
                &self.pool,
                &input,
                1,
                &layer.w,
                &mut gates,
                &Epilogue::bias(&layer.b),
            );
            for j in 0..h {
                let i = sigmoid(gates[j]);
                let f = sigmoid(gates[h + j]);
                let g = gates[2 * h + j].tanh();
                let o = sigmoid(gates[3 * h + j]);
                cs[j] = f * cs[j] + i * g;
                hs[j] = o * cs[j].tanh();
            }
            input[..h].copy_from_slice(hs);
        }
        k.linear(
            &self.pool,
            &input[..h],
            1,
            &dec.pred_proj,
            pred,
            &Epilogue::bias(&dec.pred_proj_b),
        );
    }
}

/// Rows of queries per attention task.
const ATT_BLOCK: usize = 8 * TILE_ROWS;
/// Joint steps a trace keeps.
const MAX_TRACED_JOINTS: usize = 64;

/// Buffers reused across layers.
struct Scratch {
    h: Vec<f32>,
    big: Vec<f32>,
    att: Vec<f32>,
    p: Vec<f32>,
    kt: AlignedBuf,
    pt: AlignedBuf,
    kt_ld: usize,
    pt_ld: usize,
}

impl Scratch {
    fn new(t: usize, c: &Config) -> Scratch {
        let d = c.d_model;
        let kt_ld = t.next_multiple_of(16);
        // Room for a block's position window to start anywhere and still read whole panels.
        let pt_ld = (2 * t - 1).next_multiple_of(16) + 16;
        Scratch {
            h: vec![0f32; t * d],
            big: vec![0f32; t * c.ff.max(3 * d)],
            att: vec![0f32; t * d],
            p: vec![0f32; (2 * t - 1) * d],
            kt: AlignedBuf::zeroed(c.heads * c.head_dim() * kt_ld),
            pt: AlignedBuf::zeroed(c.heads * c.head_dim() * pt_ld + 16),
            kt_ld,
            pt_ld,
        }
    }
}

/// The sinusoidal relative position table, `[2t - 1, d]`, for positions t - 1 down to 1 - t,
/// sine and cosine interleaved.
fn positions(t: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0f32; (2 * t - 1) * d];
    for (r, row) in out.chunks_exact_mut(d).enumerate() {
        let p = t as f64 - 1.0 - r as f64;
        for i in 0..d / 2 {
            let freq = (-((2 * i) as f64) * (10000f64.ln() / d as f64)).exp();
            let angle = p * freq;
            row[2 * i] = angle.sin() as f32;
            row[2 * i + 1] = angle.cos() as f32;
        }
    }
    out
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The index of the first largest value.
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}
