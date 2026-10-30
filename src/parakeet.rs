//! NVIDIA's Parakeet TDT: a FastConformer encoder and a token-and-duration transducer (TDT)
//! decoder, run with greedy decoding.
//!
//! Weights are read from the Hugging Face `model.safetensors` (the `ParakeetForTDT` layout),
//! with `config.json` and `tokenizer.json` beside it.

use std::path::Path;
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use crate::features::{FrontEnd, FrontEndConfig};
use crate::json;
use crate::kernels::{
    Act, AlignedBuf, BView, Epilogue, Isa, Kernels, PackedWeights, TILE_ROWS, for_rows,
};
use crate::safetensors::SafeTensors;
use crate::threads::{Pool, SharedMut};
use crate::tokenizer::Tokenizer;

/// How a model runs.
#[derive(Debug, Clone)]
pub struct Options {
    /// Threads in all, the calling thread included.
    pub threads: usize,
    /// The instruction set, or `None` for the best this CPU has.
    pub isa: Option<Isa>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            isa: None,
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
}

impl Timings {
    pub fn add(&mut self, name: &'static str, d: Duration) {
        match self.steps.iter_mut().find(|(n, _)| *n == name) {
            Some((_, total)) => *total += d,
            None => self.steps.push((name, d)),
        }
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
    conv0_w: Vec<f32>,
    conv0_b: Vec<f32>,
    dw1_w: Vec<f32>,
    dw1_b: Vec<f32>,
    pw1: PackedWeights,
    pw1_b: Vec<f32>,
    dw2_w: Vec<f32>,
    dw2_b: Vec<f32>,
    pw2: PackedWeights,
    pw2_b: Vec<f32>,
    /// Inputs reordered to frequency-major (`f * channels + c`) to match the channels-last layout.
    linear: PackedWeights,
    linear_b: Vec<f32>,
}

struct Norm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

struct Layer {
    norm_ff1: Norm,
    ff1_in: PackedWeights,
    ff1_out: PackedWeights,
    norm_att: Norm,
    /// q, k and v stacked, `[3 * d_model, d_model]`.
    qkv: PackedWeights,
    pos: PackedWeights,
    bias_u: Vec<f32>,
    bias_v: Vec<f32>,
    out: PackedWeights,
    norm_conv: Norm,
    pw1: PackedWeights,
    /// Depthwise weights `[kernel, d_model]` with the BatchNorm folded in.
    dw_w: Vec<f32>,
    dw_b: Vec<f32>,
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
    b: Vec<f32>,
}

struct Decoder {
    embedding: Vec<f32>,
    lstm: Vec<LstmLayer>,
    pred_proj: PackedWeights,
    pred_proj_b: Vec<f32>,
    enc_proj: PackedWeights,
    enc_proj_b: Vec<f32>,
    joint: PackedWeights,
    joint_b: Vec<f32>,
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
}

struct Loader {
    st: SafeTensors,
}

impl Loader {
    fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        self.st.read_f32(name, Some(shape))
    }

    fn linear(&mut self, name: &str, n: usize, k: usize) -> Result<PackedWeights> {
        let w = self.take(name, &[n, k])?;
        Ok(PackedWeights::from_rows(&w, n, k))
    }

    fn norm(&mut self, prefix: &str, d: usize) -> Result<Norm> {
        Ok(Norm {
            gamma: self.take(&format!("{prefix}.weight"), &[d])?,
            beta: self.take(&format!("{prefix}.bias"), &[d])?,
        })
    }

    /// A `[channels, 1, 3, 3]` conv weight as `[9, channels]`.
    fn conv3x3(&mut self, name: &str, channels: usize) -> Result<Vec<f32>> {
        let w = self.take(name, &[channels, 1, 3, 3])?;
        Ok(transpose(&w, channels, 9))
    }
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

impl Parakeet {
    /// Loads `model.safetensors`, `config.json` and `tokenizer.json` from a directory.
    pub fn load(dir: &Path, options: &Options) -> Result<Parakeet> {
        let isa = options.isa.unwrap_or_else(Isa::detect);
        let kernels = Kernels::new(isa).ok_or_else(|| {
            Error::Model(format!("This CPU can't run the {} kernels.", isa.name()))
        })?;
        let config_path = dir.join("config.json");
        let config = Config::from_json(
            &std::fs::read(&config_path).map_err(|e| Error::io(&config_path, e))?,
        )?;
        let tokenizer = Tokenizer::load(&dir.join("tokenizer.json"))?;
        if tokenizer.len() != config.vocab {
            return Err(Error::Model(format!(
                "The tokenizer has {} tokens but the model {}.",
                tokenizer.len(),
                config.vocab
            )));
        }
        let mut ld = Loader {
            st: SafeTensors::open(&dir.join("model.safetensors"))?,
        };
        let c = &config;
        let (d, ch) = (c.d_model, c.sub_channels);

        let sub_f = c.n_mels / 8;
        let linear_w = ld.take("encoder.subsampling.linear.weight", &[d, ch * sub_f])?;
        // Column c * sub_f + f of the stored weight becomes column f * ch + c.
        let mut linear_perm = vec![0f32; d * ch * sub_f];
        for o in 0..d {
            for cc in 0..ch {
                for f in 0..sub_f {
                    linear_perm[o * ch * sub_f + f * ch + cc] =
                        linear_w[o * ch * sub_f + cc * sub_f + f];
                }
            }
        }
        let conv0 = ld.take("encoder.subsampling.layers.0.weight", &[ch, 1, 3, 3])?;
        let pw = |ld: &mut Loader, name: &str| -> Result<PackedWeights> {
            let w = ld.take(name, &[ch, ch, 1, 1])?;
            Ok(PackedWeights::from_rows(&w, ch, ch))
        };
        let subsampling = Subsampling {
            conv0_w: transpose(&conv0, ch, 9),
            conv0_b: ld.take("encoder.subsampling.layers.0.bias", &[ch])?,
            dw1_w: ld.conv3x3("encoder.subsampling.layers.2.weight", ch)?,
            dw1_b: ld.take("encoder.subsampling.layers.2.bias", &[ch])?,
            pw1: pw(&mut ld, "encoder.subsampling.layers.3.weight")?,
            pw1_b: ld.take("encoder.subsampling.layers.3.bias", &[ch])?,
            dw2_w: ld.conv3x3("encoder.subsampling.layers.5.weight", ch)?,
            dw2_b: ld.take("encoder.subsampling.layers.5.bias", &[ch])?,
            pw2: pw(&mut ld, "encoder.subsampling.layers.6.weight")?,
            pw2_b: ld.take("encoder.subsampling.layers.6.bias", &[ch])?,
            linear: PackedWeights::from_rows(&linear_perm, d, ch * sub_f),
            linear_b: ld.take("encoder.subsampling.linear.bias", &[d])?,
        };
        drop(linear_perm);

        let mut layers = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            let p = format!("encoder.layers.{i}");
            let mut qkv = ld.take(&format!("{p}.self_attn.q_proj.weight"), &[d, d])?;
            qkv.extend(ld.take(&format!("{p}.self_attn.k_proj.weight"), &[d, d])?);
            qkv.extend(ld.take(&format!("{p}.self_attn.v_proj.weight"), &[d, d])?);
            let gamma = ld.take(&format!("{p}.conv.norm.weight"), &[d])?;
            let beta = ld.take(&format!("{p}.conv.norm.bias"), &[d])?;
            let mean = ld.take(&format!("{p}.conv.norm.running_mean"), &[d])?;
            let var = ld.take(&format!("{p}.conv.norm.running_var"), &[d])?;
            let dw = ld.take(
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
            let pw1 = ld.take(&format!("{p}.conv.pointwise_conv1.weight"), &[2 * d, d, 1])?;
            let pw2 = ld.take(&format!("{p}.conv.pointwise_conv2.weight"), &[d, d, 1])?;
            layers.push(Layer {
                norm_ff1: ld.norm(&format!("{p}.norm_feed_forward1"), d)?,
                ff1_in: ld.linear(&format!("{p}.feed_forward1.linear1.weight"), c.ff, d)?,
                ff1_out: ld.linear(&format!("{p}.feed_forward1.linear2.weight"), d, c.ff)?,
                norm_att: ld.norm(&format!("{p}.norm_self_att"), d)?,
                qkv: PackedWeights::from_rows(&qkv, 3 * d, d),
                pos: ld.linear(&format!("{p}.self_attn.relative_k_proj.weight"), d, d)?,
                bias_u: ld.take(&format!("{p}.self_attn.bias_u"), &[c.heads, c.head_dim()])?,
                bias_v: ld.take(&format!("{p}.self_attn.bias_v"), &[c.heads, c.head_dim()])?,
                out: ld.linear(&format!("{p}.self_attn.o_proj.weight"), d, d)?,
                norm_conv: ld.norm(&format!("{p}.norm_conv"), d)?,
                pw1: PackedWeights::from_rows(&pw1, 2 * d, d),
                dw_w,
                dw_b,
                pw2: PackedWeights::from_rows(&pw2, d, d),
                norm_ff2: ld.norm(&format!("{p}.norm_feed_forward2"), d)?,
                ff2_in: ld.linear(&format!("{p}.feed_forward2.linear1.weight"), c.ff, d)?,
                ff2_out: ld.linear(&format!("{p}.feed_forward2.linear2.weight"), d, c.ff)?,
                norm_out: ld.norm(&format!("{p}.norm_out"), d)?,
            });
        }

        let h = c.pred_hidden;
        let mut lstm = Vec::with_capacity(c.pred_layers);
        for l in 0..c.pred_layers {
            let w_ih = ld.take(&format!("decoder.lstm.weight_ih_l{l}"), &[4 * h, h])?;
            let w_hh = ld.take(&format!("decoder.lstm.weight_hh_l{l}"), &[4 * h, h])?;
            let b_ih = ld.take(&format!("decoder.lstm.bias_ih_l{l}"), &[4 * h])?;
            let b_hh = ld.take(&format!("decoder.lstm.bias_hh_l{l}"), &[4 * h])?;
            let mut w = Vec::with_capacity(8 * h * h);
            for g in 0..4 * h {
                w.extend_from_slice(&w_ih[g * h..(g + 1) * h]);
                w.extend_from_slice(&w_hh[g * h..(g + 1) * h]);
            }
            lstm.push(LstmLayer {
                w: PackedWeights::from_rows(&w, 4 * h, 2 * h),
                b: b_ih.iter().zip(&b_hh).map(|(a, b)| a + b).collect(),
            });
        }
        let outputs = c.vocab + c.durations.len();
        let decoder = Decoder {
            embedding: ld.take("decoder.embedding.weight", &[c.vocab, h])?,
            lstm,
            pred_proj: ld.linear("decoder.decoder_projector.weight", h, h)?,
            pred_proj_b: ld.take("decoder.decoder_projector.bias", &[h])?,
            enc_proj: ld.linear("encoder_projector.weight", h, d)?,
            enc_proj_b: ld.take("encoder_projector.bias", &[h])?,
            joint: ld.linear("joint.head.weight", outputs, h)?,
            joint_b: ld.take("joint.head.bias", &[outputs])?,
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
        })
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
        let mut x = vec![0f32; t3 * self.config.d_model];
        k.linear(
            &self.pool,
            &h3,
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
        timings.time("predict", || self.predict(None, &mut state, &mut pred));
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
            timings.time("predict", || {
                self.predict(Some(token), &mut state, &mut pred)
            });
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

    /// Runs the prediction network on one token (or the start state with `None`), updating the
    /// LSTM state and writing the projected output to `pred`.
    fn predict(&self, token: Option<usize>, state: &mut [f32], pred: &mut [f32]) {
        let c = &self.config;
        let dec = &self.decoder;
        let k = &self.kernels;
        let h = c.pred_hidden;
        let mut input = vec![0f32; 2 * h];
        if let Some(token) = token {
            input[..h].copy_from_slice(&dec.embedding[token * h..(token + 1) * h]);
        }
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
