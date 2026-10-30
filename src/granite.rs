//! IBM's Granite Speech 5.0 TurboCTC: a Conformer encoder with attention inside blocks of
//! frames and a CTC head, run with greedy decoding.
//!
//! Weights come from a pack, or from the Hugging Face `model.safetensors` (the
//! `GraniteSpeech5ForCTC` layout) with `config.json` and `tokenizer.json` beside it, converted
//! to a pack in memory.
//!
//! The encoder takes rows of two stacked 10 ms frames (50 a second). Its first blocks halve the
//! frame rate twice, with a stride-2 depthwise convolution added to the input averaged over pairs
//! of frames, so the rest run at 12.5 frames a second. Each block's attention sees only the
//! frames of its own block of 128, with Shaw's learned relative positions. After the middle block
//! the CTC posteriors are projected back and added to the hidden states (self-conditioning), and
//! the same CTC head reads the last block's output.

use std::path::Path;
use std::time::Instant;

use crate::error::{Error, Result};
use crate::granite_features::GraniteFrontEnd;
use crate::json;
use crate::kernels::{Act, BView, Epilogue, Isa, Kernels, PackedWeights, for_rows};
use crate::pack::{MemorySink, Pack, Sink};
use crate::parakeet::{Options, Timings, Token, Transcript};
use crate::quant::{Precision, Rows};
use crate::region::Vector;
use crate::safetensors::SafeTensors;
use crate::source::{Matrix, Source};
use crate::threads::{Pool, SharedMut};
use crate::tokenizer::Tokenizer;

/// The model name packs of Granite Speech carry.
pub const PACK_MODEL: &str = "granite-speech-ctc";

/// The architecture, read from `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub d_model: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub ff: usize,
    pub conv_kernel: usize,
    /// Channels of the depthwise convolution, `d_model` times the expansion factor.
    pub conv_channels: usize,
    pub n_mels: usize,
    /// Frames per attention block.
    pub context: usize,
    /// The relative position table has `2 * max_relative + 1` rows, row `max_relative` for a
    /// distance of 0.
    pub max_relative: usize,
    /// The blocks that halve the frame rate.
    pub subsample_layers: Vec<usize>,
    pub vocab: usize,
    /// The CTC blank, which is also the padding token.
    pub blank: usize,
}

impl Config {
    pub fn from_json(bytes: &[u8]) -> Result<Config> {
        let root = json::parse(bytes)?;
        let bad = |m: String| {
            Error::Model(format!(
                "config.json doesn't describe a supported Granite Speech CTC model. {m}"
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
        let expect = |v: &json::Value, key: &str, want: &str, default: Option<&str>| {
            let got = match v.get(key) {
                Some(json::Value::String(s)) => Some(s.clone()),
                Some(json::Value::Bool(b)) => Some(b.to_string()),
                Some(json::Value::Number(n)) => Some(n.clone()),
                Some(_) => None,
                None => default.map(str::to_string),
            };
            if got.as_deref() == Some(want) {
                Ok(())
            } else {
                Err(bad(format!("{key} isn't {want}.")))
            }
        };
        expect(&root, "model_type", "granite_speech5_ctc", None)?;
        expect(enc, "model_type", "granite_speech5_encoder", None)?;
        expect(enc, "hidden_act", "silu", Some("silu"))?;
        expect(enc, "attention_bias", "true", Some("true"))?;
        expect(&root, "tie_word_embeddings", "true", Some("true"))?;
        let d_model = num(enc, "hidden_size")?;
        let heads = num(enc, "num_attention_heads")?;
        let head_dim = match enc.get("head_dim") {
            None | Some(json::Value::Null) => d_model / heads.max(1),
            Some(_) => num(enc, "head_dim")?,
        };
        let kv_heads = match enc.get("num_key_value_heads") {
            None | Some(json::Value::Null) => heads,
            Some(_) => num(enc, "num_key_value_heads")?,
        };
        let subsample_layers = match enc.get("subsample_layers") {
            None | Some(json::Value::Null) => vec![0, 1],
            Some(list) => list
                .as_array()
                .and_then(|l| {
                    l.iter()
                        .map(|x| x.as_u64().map(|x| x as usize))
                        .collect::<Option<Vec<_>>>()
                })
                .ok_or_else(|| bad("subsample_layers isn't a list of indices.".into()))?,
        };
        let config = Config {
            d_model,
            heads,
            head_dim,
            layers: num(enc, "num_hidden_layers")?,
            ff: num(enc, "intermediate_size")?,
            conv_kernel: num(enc, "conv_kernel_size")?,
            conv_channels: d_model * num(enc, "conv_expansion_factor")?,
            n_mels: num(enc, "num_mel_bins")?,
            context: num(enc, "context_size")?,
            max_relative: num(enc, "max_position_embeddings")?,
            subsample_layers,
            vocab: num(&root, "vocab_size")?,
            blank: num(&root, "pad_token_id")?,
        };
        let c = &config;
        if heads == 0
            || kv_heads != heads
            || heads * head_dim != d_model
            || !head_dim.is_multiple_of(16)
            || c.conv_kernel.is_multiple_of(2)
            || c.context == 0
            || c.context > c.max_relative
            || c.layers < 2
            || num(enc, "vocab_size")? != c.vocab
            || c.blank >= c.vocab
            || c.subsample_layers.windows(2).any(|w| w[0] >= w[1])
            || c.subsample_layers.iter().any(|&l| l >= c.layers)
        {
            return Err(bad("Its sizes don't fit together.".into()));
        }
        Ok(config)
    }

    /// The block after which the CTC posteriors are fed back.
    pub fn self_conditioning_layer(&self) -> usize {
        self.layers / 2 - 1
    }

    /// Encoder frames for `rows` rows of features.
    pub fn frames_for(&self, rows: usize) -> usize {
        self.subsample_layers.iter().fold(rows, |t, _| t / 2)
    }

    /// The attention's scale, `head_dim^-0.5` in f32.
    fn scale(&self) -> f32 {
        (self.head_dim as f64).powf(-0.5) as f32
    }
}

/// Encoder output, row-major `[frames, width]`.
#[derive(Debug, Clone)]
pub struct Encoded {
    pub frames: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

struct Norm {
    gamma: Vector,
    beta: Vector,
}

struct Linear {
    w: PackedWeights,
    b: Vector,
}

/// A feed-forward module and the norm before it.
struct FeedForward {
    norm: Norm,
    up: Linear,
    down: Linear,
}

pub(crate) struct Block {
    pub(crate) strided: bool,
    ff1: FeedForward,
    norm_att: Norm,
    /// q, k and v stacked, `[3 * d_model, d_model]`, without biases.
    qkv: PackedWeights,
    /// The rows of the relative position table that distances inside a block reach, from
    /// `context - 1` down to `1 - context`.
    rel: PackedWeights,
    out: Linear,
    norm_conv: Norm,
    pw1: Linear,
    /// Depthwise weights `[kernel, conv_channels]` with the BatchNorm folded in.
    dw_w: Vector,
    dw_b: Vector,
    pw2: Linear,
    ff2: FeedForward,
    norm_out: Norm,
}

struct Weights {
    input: Linear,
    blocks: Vec<Block>,
    /// `encoder.out`, the CTC head, which the self-conditioning shares.
    head: Linear,
    /// `encoder.out_mid`, from the CTC posteriors back to the hidden width.
    mid: Linear,
}

/// Converts Granite's weights from `src` (named as in the Hugging Face checkpoint) into pack
/// entries in `precision`. Every linear layer, the stacked q, k and v and the rows of each
/// relative position table that a block reaches become matrices, and BatchNorm is folded into
/// the depthwise convolutions. Biases, norms and the convolutions stay f32.
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
    let (d, cc, k, hd) = (c.d_model, c.conv_channels, c.conv_kernel, c.head_dim);
    let linear = |src: &mut dyn Source,
                  sink: &mut dyn Sink,
                  from: &str,
                  to: &str,
                  n: usize,
                  k: usize|
     -> Result<()> {
        let w = src.matrix(&format!("{from}.weight"), &[n, k])?;
        sink.matrix(to, &w.rows(), n, k, precision)?;
        let b = src.tensor(&format!("{from}.bias"), &[n])?;
        sink.f32(&format!("{to}.b"), &[n], &b)
    };
    let norm = |src: &mut dyn Source, sink: &mut dyn Sink, from: &str, to: &str| -> Result<()> {
        sink.f32(
            &format!("{to}.w"),
            &[d],
            &src.tensor(&format!("{from}.weight"), &[d])?,
        )?;
        sink.f32(
            &format!("{to}.b"),
            &[d],
            &src.tensor(&format!("{from}.bias"), &[d])?,
        )
    };

    linear(src, sink, "encoder.input_linear", "input", d, 4 * c.n_mels)?;
    for i in 0..c.layers {
        let (p, o) = (format!("encoder.layers.{i}"), format!("blocks.{i}"));
        for n in [1, 2] {
            let f = format!("{o}.ff{n}");
            norm(
                src,
                sink,
                &format!("{p}.norm_feed_forward{n}"),
                &format!("{f}.norm"),
            )?;
            let ff = format!("{p}.feed_forward{n}");
            linear(
                src,
                sink,
                &format!("{ff}.linear1"),
                &format!("{f}.up"),
                c.ff,
                d,
            )?;
            linear(
                src,
                sink,
                &format!("{ff}.linear2"),
                &format!("{f}.down"),
                d,
                c.ff,
            )?;
        }
        norm(
            src,
            sink,
            &format!("{p}.norm_self_att"),
            &format!("{o}.norm_att"),
        )?;
        let qkv = ["q", "k", "v"]
            .iter()
            .map(|x| src.matrix(&format!("{p}.self_attn.{x}_proj.weight"), &[d, d]))
            .collect::<Result<Vec<_>>>()?;
        sink.matrix(
            &format!("{o}.qkv"),
            &Matrix::stack(qkv).rows(),
            3 * d,
            d,
            precision,
        )?;
        // Distance i - j inside a block reads row max_relative + i - j. Keep the rows from
        // distance context - 1 down to 1 - context, so column m is distance context - 1 - m.
        let table = src.tensor(
            &format!("{p}.self_attn.rel_pos_emb.weight"),
            &[2 * c.max_relative + 1, hd],
        )?;
        let rows = 2 * c.context - 1;
        let mut rel = Vec::with_capacity(rows * hd);
        for m in 0..rows {
            let r = c.max_relative + c.context - 1 - m;
            rel.extend_from_slice(&table[r * hd..(r + 1) * hd]);
        }
        sink.matrix(&format!("{o}.rel"), &Rows::F32(&rel), rows, hd, precision)?;
        linear(
            src,
            sink,
            &format!("{p}.self_attn.o_proj"),
            &format!("{o}.out"),
            d,
            d,
        )?;
        norm(
            src,
            sink,
            &format!("{p}.norm_conv"),
            &format!("{o}.norm_conv"),
        )?;
        let pw1 = format!("{p}.conv.pointwise_lin1");
        linear(src, sink, &pw1, &format!("{o}.pw1"), 2 * cc, d)?;
        let gamma = src.tensor(&format!("{p}.conv.norm.weight"), &[cc])?;
        let beta = src.tensor(&format!("{p}.conv.norm.bias"), &[cc])?;
        let mean = src.tensor(&format!("{p}.conv.norm.running_mean"), &[cc])?;
        let var = src.tensor(&format!("{p}.conv.norm.running_var"), &[cc])?;
        let dw = src.tensor(&format!("{p}.conv.depthwise_conv.weight"), &[cc, 1, k])?;
        let mut dw_w = vec![0f32; k * cc];
        let mut dw_b = vec![0f32; cc];
        for ch in 0..cc {
            let s = gamma[ch] as f64 / (var[ch] as f64 + 1e-5).sqrt();
            for j in 0..k {
                dw_w[j * cc + ch] = (dw[ch * k + j] as f64 * s) as f32;
            }
            dw_b[ch] = (beta[ch] as f64 - mean[ch] as f64 * s) as f32;
        }
        sink.f32(&format!("{o}.dw.w"), &[k, cc], &dw_w)?;
        sink.f32(&format!("{o}.dw.b"), &[cc], &dw_b)?;
        let pw2 = format!("{p}.conv.pointwise_lin2");
        linear(src, sink, &pw2, &format!("{o}.pw2"), d, cc)?;
        norm(
            src,
            sink,
            &format!("{p}.norm_out"),
            &format!("{o}.norm_out"),
        )?;
    }
    linear(src, sink, "encoder.out", "head", c.vocab, d)?;
    linear(src, sink, "encoder.out_mid", "mid", d, c.vocab)
}

/// Converts a Hugging Face checkpoint directory (`model.safetensors`, `config.json` and
/// `tokenizer.json`) into pack entries.
pub fn convert_path(path: &Path, precision: Precision, sink: &mut dyn Sink) -> Result<()> {
    let read = |name: &str| {
        let p = path.join(name);
        std::fs::read(&p).map_err(|e| Error::io(&p, e))
    };
    let (config, tokenizer) = (read("config.json")?, read("tokenizer.json")?);
    let mut st = SafeTensors::open(&path.join("model.safetensors"))?;
    convert(&mut st, &config, &tokenizer, precision, sink)
}

/// A loaded Granite Speech CTC model.
pub struct Granite {
    config: Config,
    front_end: GraniteFrontEnd,
    weights: Weights,
    tokenizer: Tokenizer,
    kernels: Kernels,
    pool: Pool,
    precision: Option<Precision>,
}

/// Rows of CTC logits computed at once, which bounds the memory they take.
const LOGIT_ROWS: usize = 256;

pub(crate) fn timed<T>(timings: &mut Timings, name: &'static str, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let out = f();
    timings.add(name, start.elapsed());
    out
}

impl Granite {
    /// Loads a model from a pack file or a Hugging Face checkpoint directory
    /// (`model.safetensors`, `config.json` and `tokenizer.json`). A pack is mapped as it is, and
    /// a checkpoint is converted in memory to `options.precision`. The checkpoint's bf16 weights
    /// widen exactly, so f32 and bf16 give the same results.
    pub fn load(path: &Path, options: &Options) -> Result<Granite> {
        if path.is_file() {
            return Granite::from_pack(&Pack::open(path)?, options);
        }
        let mut sink = MemorySink::new(
            PACK_MODEL,
            crate::pack::metadata_for(path, options.precision),
        );
        convert_path(path, options.precision, &mut sink)?;
        Granite::from_pack(&sink.finish()?, options)
    }

    /// Builds the model from a pack's entries, which it keeps views of.
    pub fn from_pack(pack: &Pack, options: &Options) -> Result<Granite> {
        if pack.model() != PACK_MODEL {
            return Err(Error::Model(format!(
                "The pack holds {}, not Granite Speech.",
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
        let (d, cc) = (c.d_model, c.conv_channels);
        let v = |name: &str, shape: &[usize]| pack.vector(name, shape);
        let linear = |name: &str, n: usize, k: usize| -> Result<Linear> {
            Ok(Linear {
                w: pack.matrix(name, n, k)?,
                b: v(&format!("{name}.b"), &[n])?,
            })
        };
        let norm = |name: &str| -> Result<Norm> {
            Ok(Norm {
                gamma: v(&format!("{name}.w"), &[d])?,
                beta: v(&format!("{name}.b"), &[d])?,
            })
        };
        let feed_forward = |name: &str| -> Result<FeedForward> {
            Ok(FeedForward {
                norm: norm(&format!("{name}.norm"))?,
                up: linear(&format!("{name}.up"), c.ff, d)?,
                down: linear(&format!("{name}.down"), d, c.ff)?,
            })
        };
        let mut blocks = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            let o = format!("blocks.{i}");
            blocks.push(Block {
                strided: c.subsample_layers.contains(&i),
                ff1: feed_forward(&format!("{o}.ff1"))?,
                norm_att: norm(&format!("{o}.norm_att"))?,
                qkv: pack.matrix(&format!("{o}.qkv"), 3 * d, d)?,
                rel: pack.matrix(&format!("{o}.rel"), 2 * c.context - 1, c.head_dim)?,
                out: linear(&format!("{o}.out"), d, d)?,
                norm_conv: norm(&format!("{o}.norm_conv"))?,
                pw1: linear(&format!("{o}.pw1"), 2 * cc, d)?,
                dw_w: v(&format!("{o}.dw.w"), &[c.conv_kernel, cc])?,
                dw_b: v(&format!("{o}.dw.b"), &[cc])?,
                pw2: linear(&format!("{o}.pw2"), d, cc)?,
                ff2: feed_forward(&format!("{o}.ff2"))?,
                norm_out: norm(&format!("{o}.norm_out"))?,
            });
        }
        let weights = Weights {
            input: linear("input", d, 4 * c.n_mels)?,
            blocks,
            head: linear("head", c.vocab, d)?,
            mid: linear("mid", d, c.vocab)?,
        };
        Ok(Granite {
            front_end: GraniteFrontEnd::new(c.n_mels),
            config,
            weights,
            tokenizer,
            kernels,
            pool: Pool::new(options.threads),
            precision: pack.precision(),
        })
    }

    /// The precision of the weight matrices, if they share one.
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
        let linear = |l: &Linear| l.w.bytes() + l.b.len() * 4;
        let w = &self.weights;
        let mut total = linear(&w.input) + linear(&w.head) + linear(&w.mid);
        for b in &w.blocks {
            total += [
                &b.ff1.up,
                &b.ff1.down,
                &b.out,
                &b.pw1,
                &b.pw2,
                &b.ff2.up,
                &b.ff2.down,
            ]
            .iter()
            .map(|l| linear(l))
            .sum::<usize>();
            total += b.qkv.bytes() + b.rel.bytes() + (b.dw_w.len() + b.dw_b.len()) * 4;
        }
        total
    }

    /// Computes the stacked log-mel features, `[rows, 4 * n_mels]`.
    pub fn features(&self, samples: &[f32]) -> Result<crate::features::Features> {
        self.front_end.compute(samples)
    }

    /// Transcribes 16 kHz mono samples. Each token's frame is where its run of CTC frames starts
    /// (80 ms each), and its duration is the run's length.
    pub fn transcribe(&self, samples: &[f32]) -> Result<Transcript> {
        let mut timings = Timings::default();
        let features = timed(&mut timings, "features", || self.features(samples))?;
        let encoded = self.encode(
            &features.data,
            features.frames,
            &mut timings,
            &mut |_, _, _| {},
        )?;
        let ids = timed(&mut timings, "ctc_head", || self.frame_ids(&encoded));
        let tokens = ctc_tokens(&ids, self.config.blank);
        let text = timed(&mut timings, "detokenize", || {
            self.tokenizer
                .decode(&tokens.iter().map(|t| t.id).collect::<Vec<_>>())
        });
        Ok(Transcript {
            text,
            tokens,
            timings,
        })
    }

    /// Runs the encoder over `rows` rows of features. `tap` sees the input projection
    /// (`"input_linear"`), each block's output (`"layer.0"`, ...) and the self-conditioning
    /// injection (`"selfcond_injection"`) as `[frames, d_model]`.
    pub fn encode(
        &self,
        features: &[f32],
        rows: usize,
        timings: &mut Timings,
        tap: &mut dyn FnMut(&str, usize, &[f32]),
    ) -> Result<Encoded> {
        let c = &self.config;
        let d = c.d_model;
        if c.frames_for(rows) == 0 {
            return Err(Error::Model(format!(
                "The audio is too short ({rows} feature rows). The model needs at least {}.",
                1 << c.subsample_layers.len()
            )));
        }
        assert_eq!(features.len(), rows * 4 * c.n_mels);
        let mut x = vec![0f32; rows * d];
        timed(timings, "input_linear", || {
            self.input_projection(features, rows, &mut x)
        });
        tap("input_linear", rows, &x);
        let mut s = Scratch::new(rows, c);
        let mut t = rows;
        for (i, block) in self.weights.blocks.iter().enumerate() {
            let out_t = if block.strided { t / 2 } else { t };
            let mut out = vec![0f32; out_t * d];
            self.block_span(block, &mut x, 0, t, &[], 0, &mut out, &mut s, timings);
            x = out;
            t = out_t;
            tap(&format!("layer.{i}"), t, &x);
            if i == c.self_conditioning_layer() {
                timed(timings, "self_conditioning", || {
                    self.self_conditioning(&x, t, &mut s.att)
                });
                tap("selfcond_injection", t, &s.att[..t * d]);
                for (v, &a) in x.iter_mut().zip(&s.att[..t * d]) {
                    *v += a;
                }
            }
        }
        Ok(Encoded {
            frames: t,
            width: d,
            data: x,
        })
    }

    /// The CTC logits of frames `first..first + rows`, `[rows, vocab]`.
    pub fn ctc_logits(&self, encoded: &Encoded, first: usize, rows: usize) -> Vec<f32> {
        let d = self.config.d_model;
        self.logits_of(&encoded.data[first * d..(first + rows) * d], rows)
    }

    /// The CTC logits of `rows` frames of encoder output, `[rows, vocab]`.
    pub(crate) fn logits_of(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut logits = vec![0f32; rows * self.config.vocab];
        self.kernels.linear(
            &self.pool,
            x,
            rows,
            &self.weights.head.w,
            &mut logits,
            &Epilogue::bias(&self.weights.head.b),
        );
        logits
    }

    /// The most likely token of every frame, the first one where several tie.
    pub fn frame_ids(&self, encoded: &Encoded) -> Vec<u32> {
        self.ids_of(&encoded.data, encoded.frames)
    }

    /// The most likely token of each of `frames` frames of encoder output, computed a chunk of
    /// frames at a time.
    pub(crate) fn ids_of(&self, x: &[f32], frames: usize) -> Vec<u32> {
        let (d, vocab) = (self.config.d_model, self.config.vocab);
        let mut ids = Vec::with_capacity(frames);
        for first in (0..frames).step_by(LOGIT_ROWS) {
            let rows = (frames - first).min(LOGIT_ROWS);
            let logits = self.logits_of(&x[first * d..(first + rows) * d], rows);
            ids.extend(logits.chunks_exact(vocab).map(|row| argmax(row) as u32));
        }
        ids
    }

    /// The encoder's blocks.
    pub(crate) fn blocks(&self) -> &[Block] {
        &self.weights.blocks
    }

    /// Projects `rows` rows of features to the encoder's width, the first block's input.
    pub(crate) fn input_projection(&self, features: &[f32], rows: usize, out: &mut [f32]) {
        self.kernels.linear(
            &self.pool,
            features,
            rows,
            &self.weights.input.w,
            out,
            &Epilogue::bias(&self.weights.input.b),
        )
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

    /// A feed-forward module with its half-step residual, `x += 0.5 * ff(norm(x))`.
    fn feed_forward(
        &self,
        f: &FeedForward,
        x: &mut [f32],
        t: usize,
        s: &mut Scratch,
        timings: &mut Timings,
    ) {
        let (k, pool) = (&self.kernels, &self.pool);
        let d = self.config.d_model;
        let (ff_in, ff_out) = (&f.up, &f.down);
        timed(timings, "layer_norm", || self.norm(&f.norm, x, &mut s.h, t));
        timed(timings, "ff_in", || {
            k.linear(
                pool,
                &s.h[..t * d],
                t,
                &ff_in.w,
                &mut s.big,
                &Epilogue {
                    bias: Some(&ff_in.b),
                    act: Act::Silu,
                    ..Epilogue::PLAIN
                },
            )
        });
        timed(timings, "ff_out", || {
            k.linear(
                pool,
                &s.big[..t * self.config.ff],
                t,
                &ff_out.w,
                &mut x[..t * d],
                &Epilogue {
                    bias: Some(&ff_out.b),
                    scale: 0.5,
                    accumulate: true,
                    ..Epilogue::PLAIN
                },
            )
        });
    }

    /// Runs one block over input frames `start..end`, whose input `x` holds (`[end - start,
    /// d_model]`, rewritten in place), and writes the outputs of frames `out_start` up to the
    /// last one to `out` (`end / 2` frames for a strided block, `end` otherwise). `start` begins
    /// an attention block. `left` holds the post-attention states of input frames `lo..start`,
    /// where `lo` is what [`Granite::conv_context`] gives, which the convolution reads at the left
    /// edge. Afterwards `x` holds the post-attention states of `start..end`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn block_span(
        &self,
        b: &Block,
        x: &mut [f32],
        start: usize,
        end: usize,
        left: &[f32],
        out_start: usize,
        out: &mut [f32],
        s: &mut Scratch,
        timings: &mut Timings,
    ) {
        let c = &self.config;
        let (d, cc) = (c.d_model, c.conv_channels);
        let (k, pool) = (&self.kernels, &self.pool);
        let t = end - start;
        debug_assert_eq!(start % c.context, 0);

        self.feed_forward(&b.ff1, x, t, s, timings);

        timed(timings, "layer_norm", || {
            self.norm(&b.norm_att, x, &mut s.h, t)
        });
        timed(timings, "qkv", || {
            k.linear(
                pool,
                &s.h[..t * d],
                t,
                &b.qkv,
                &mut s.big[..t * 3 * d],
                &Epilogue::PLAIN,
            )
        });
        timed(timings, "attention", || {
            self.attention(b, &s.big[..t * 3 * d], t, &mut s.att)
        });
        timed(timings, "att_out", || {
            k.linear(
                pool,
                &s.att[..t * d],
                t,
                &b.out.w,
                &mut x[..t * d],
                &Epilogue {
                    bias: Some(&b.out.b),
                    accumulate: true,
                    ..Epilogue::PLAIN
                },
            )
        });

        // Convolution module over the post-attention states of lo..end. A strided block keeps
        // every second frame of the convolution and adds it to the input averaged over pairs of
        // frames, dropping a last odd frame.
        let lo = self.conv_context(b, out_start);
        assert_eq!(left.len(), (start - lo) * d);
        let m = end - lo;
        let state = |f: usize| {
            if f < start {
                &left[(f - lo) * d..(f - lo + 1) * d]
            } else {
                &x[(f - start) * d..(f - start + 1) * d]
            }
        };
        let (stride, out_end) = if b.strided { (2, end / 2) } else { (1, end) };
        let n = out_end - out_start;
        timed(timings, "layer_norm", || {
            for_rows(pool, &mut s.h, d, m, 16, &|first, part| {
                for (r, row) in part.chunks_exact_mut(d).enumerate() {
                    (k.layer_norm)(
                        state(lo + first + r),
                        row,
                        &b.norm_conv.gamma,
                        &b.norm_conv.beta,
                        1e-5,
                    );
                }
            });
        });
        timed(timings, "conv_pw1", || {
            k.linear(
                pool,
                &s.h[..m * d],
                m,
                &b.pw1.w,
                &mut s.big[..m * 2 * cc],
                &Epilogue::bias(&b.pw1.b),
            )
        });
        timed(timings, "glu", || {
            let big = &s.big;
            for_rows(pool, &mut s.glu, cc, m, 16, &|first, part| {
                for (r, row) in part.chunks_exact_mut(cc).enumerate() {
                    let g = &big[(first + r) * 2 * cc..(first + r + 1) * 2 * cc];
                    (k.glu)(&g[..cc], &g[cc..], row);
                }
            });
        });
        timed(timings, "depthwise_conv", || {
            let input = &s.glu[..m * cc];
            for_rows(pool, &mut s.big, cc, n, 8, &|first, part| {
                for (r, row) in part.chunks_exact_mut(cc).enumerate() {
                    (k.dwconv1d_frame)(
                        input,
                        cc,
                        m,
                        stride * (out_start + first + r) - lo,
                        &b.dw_w,
                        c.conv_kernel,
                        &b.dw_b,
                        row,
                    );
                    (k.silu)(row);
                }
            });
        });
        let out = &mut out[..n * d];
        timed(timings, "pool", || {
            for (r, row) in out.chunks_exact_mut(d).enumerate() {
                let f = stride * (out_start + r);
                if b.strided {
                    for ((o, &p), &q) in row.iter_mut().zip(state(f)).zip(state(f + 1)) {
                        *o = (p + q) / 2.0;
                    }
                } else {
                    row.copy_from_slice(state(f));
                }
            }
        });
        timed(timings, "conv_pw2", || {
            k.linear(
                pool,
                &s.big[..n * cc],
                n,
                &b.pw2.w,
                out,
                &Epilogue {
                    bias: Some(&b.pw2.b),
                    accumulate: true,
                    ..Epilogue::PLAIN
                },
            )
        });

        self.feed_forward(&b.ff2, out, n, s, timings);

        timed(timings, "layer_norm", || {
            self.norm(&b.norm_out, out, &mut s.h, n);
            out.copy_from_slice(&s.h[..n * d]);
        });
    }

    /// The first input frame the convolution of a block reads to compute output frames from
    /// `out_start` on.
    pub(crate) fn conv_context(&self, b: &Block, out_start: usize) -> usize {
        let half = self.config.conv_kernel / 2;
        let first = if b.strided { 2 * out_start } else { out_start };
        first.saturating_sub(half)
    }

    /// Attention inside blocks of `context` frames with Shaw's relative positions. Reads q, k
    /// and v from `qkv` (`[t, 3 * d_model]`) and writes the heads' outputs to `att`. Each task is
    /// one head of one block, and the last block runs at its true length.
    fn attention(&self, b: &Block, qkv: &[f32], t: usize, att: &mut [f32]) {
        let c = &self.config;
        let (d, hd, heads, ctx) = (c.d_model, c.head_dim, c.heads, c.context);
        let k = &self.kernels;
        let scale = c.scale();
        let blocks = t.div_ceil(ctx);
        let positions = 2 * ctx - 1;
        let att = SharedMut(att.as_mut_ptr());
        self.pool.run(heads * blocks, &|task| {
            let (h, blk) = (task / blocks, task % blocks);
            let i0 = blk * ctx;
            let len = (t - i0).min(ctx);
            let q = &qkv[i0 * 3 * d + h * hd..];
            // The block's keys transposed, `[hd, len]`, so the scores read them as panels.
            let ld = len.next_multiple_of(16);
            let mut kt = vec![0f32; hd * ld];
            for j in 0..len {
                let key = &qkv[(i0 + j) * 3 * d + d + h * hd..][..hd];
                for (i, &v) in key.iter().enumerate() {
                    kt[i * ld + j] = v;
                }
            }
            let mut scores = vec![0f32; len * len];
            k.matmul_serial(
                q,
                3 * d,
                len,
                &BView::row_major(&kt, hd, len, ld),
                &mut scores,
                len,
                &Epilogue {
                    scale,
                    ..Epilogue::PLAIN
                },
            );
            // Column m of `pos` is the query against distance ctx - 1 - m, so key j of row i
            // reads column ctx - 1 - i + j.
            let mut pos = vec![0f32; len * positions];
            k.matmul_serial(
                q,
                3 * d,
                len,
                &b.rel.view(),
                &mut pos,
                positions,
                &Epilogue {
                    scale,
                    ..Epilogue::PLAIN
                },
            );
            for i in 0..len {
                let row = &mut scores[i * len..(i + 1) * len];
                let p = &pos[i * positions + ctx - 1 - i..][..len];
                for (s, &pv) in row.iter_mut().zip(p) {
                    *s += pv;
                }
                (k.softmax)(row);
            }
            // SAFETY: each task writes rows i0..i0 + len of its own head's columns, inside `att`.
            unsafe {
                k.matmul_serial_raw(
                    &scores,
                    len,
                    len,
                    &BView::row_major(&qkv[i0 * 3 * d + 2 * d + h * hd..], len, hd, 3 * d),
                    SharedMut(att.get().add(i0 * d + h * hd)),
                    d,
                    &Epilogue::PLAIN,
                );
            }
        });
    }

    /// Writes `out_mid(softmax(out(x)))` for `t` frames to `injection`, a chunk of frames at a
    /// time.
    pub(crate) fn self_conditioning(&self, x: &[f32], t: usize, injection: &mut [f32]) {
        let (d, vocab) = (self.config.d_model, self.config.vocab);
        let (k, pool, w) = (&self.kernels, &self.pool, &self.weights);
        let mut logits = vec![0f32; t.min(LOGIT_ROWS) * vocab];
        for first in (0..t).step_by(LOGIT_ROWS) {
            let rows = (t - first).min(LOGIT_ROWS);
            let logits = &mut logits[..rows * vocab];
            k.linear(
                pool,
                &x[first * d..(first + rows) * d],
                rows,
                &w.head.w,
                logits,
                &Epilogue::bias(&w.head.b),
            );
            for_rows(pool, logits, vocab, rows, 4, &|_, part| {
                for row in part.chunks_exact_mut(vocab) {
                    (k.softmax)(row);
                }
            });
            k.linear(
                pool,
                logits,
                rows,
                &w.mid.w,
                &mut injection[first * d..(first + rows) * d],
                &Epilogue::bias(&w.mid.b),
            );
        }
    }
}

/// Collapses per-frame ids into tokens: repeats merge and blanks drop out. A blank between two
/// equal ids separates them.
pub fn ctc_tokens(ids: &[u32], blank: usize) -> Vec<Token> {
    let mut tokens: Vec<Token> = Vec::new();
    let mut previous = None;
    for (frame, &id) in ids.iter().enumerate() {
        if id as usize != blank {
            if previous == Some(id) {
                if let Some(last) = tokens.last_mut() {
                    last.duration += 1;
                }
            } else {
                tokens.push(Token {
                    id,
                    frame: frame as u32,
                    duration: 1,
                });
            }
        }
        previous = Some(id);
    }
    tokens
}

/// Buffers reused across blocks, sized for the most frames a block reads.
pub(crate) struct Scratch {
    h: Vec<f32>,
    big: Vec<f32>,
    att: Vec<f32>,
    glu: Vec<f32>,
}

impl Scratch {
    pub(crate) fn new(t: usize, c: &Config) -> Scratch {
        let d = c.d_model;
        Scratch {
            h: vec![0f32; t * d],
            big: vec![0f32; t * c.ff.max(3 * d).max(2 * c.conv_channels)],
            att: vec![0f32; t * d],
            glu: vec![0f32; t * c.conv_channels],
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"{"architectures":["GraniteSpeech5ForCTC"],"model_type":"granite_speech5_ctc",
        "pad_token_id":0,"tie_word_embeddings":true,"vocab_size":16384,
        "encoder_config":{"attention_bias":true,"context_size":128,"conv_expansion_factor":2,
        "conv_kernel_size":7,"head_dim":128,"hidden_act":"silu","hidden_size":1024,
        "intermediate_size":4096,"max_position_embeddings":512,"model_type":"granite_speech5_encoder",
        "num_attention_heads":8,"num_hidden_layers":16,"num_key_value_heads":8,"num_mel_bins":80,
        "subsample_layers":[0,1],"vocab_size":16384}}"#;

    #[test]
    fn the_config_reads_and_checks_the_architecture() {
        let c = Config::from_json(CONFIG.as_bytes()).unwrap();
        assert_eq!(
            (c.d_model, c.heads, c.head_dim, c.layers, c.ff),
            (1024, 8, 128, 16, 4096)
        );
        assert_eq!((c.conv_kernel, c.conv_channels, c.n_mels), (7, 2048, 80));
        assert_eq!((c.context, c.max_relative), (128, 512));
        assert_eq!(c.subsample_layers, vec![0, 1]);
        assert_eq!(
            (c.vocab, c.blank, c.self_conditioning_layer()),
            (16384, 0, 7)
        );
        assert_eq!(c.frames_for(337), 84);
        assert_eq!(c.frames_for(3), 0);
        for (from, to) in [
            ("granite_speech5_ctc", "parakeet_tdt"),
            (r#""hidden_act":"silu""#, r#""hidden_act":"relu""#),
            (r#""context_size":128"#, r#""context_size":1024"#),
            (
                r#""tie_word_embeddings":true"#,
                r#""tie_word_embeddings":false"#,
            ),
            (r#""subsample_layers":[0,1]"#, r#""subsample_layers":[1,0]"#),
            (r#""num_key_value_heads":8"#, r#""num_key_value_heads":4"#),
        ] {
            assert!(
                Config::from_json(CONFIG.replace(from, to).as_bytes()).is_err(),
                "{to}"
            );
        }
    }

    #[test]
    fn ctc_merges_repeats_and_drops_blanks() {
        let tokens = ctc_tokens(&[0, 5, 5, 0, 5, 7, 7, 7, 0, 0, 2], 0);
        let got: Vec<(u32, u32, u32)> =
            tokens.iter().map(|t| (t.id, t.frame, t.duration)).collect();
        assert_eq!(got, vec![(5, 1, 2), (5, 4, 1), (7, 5, 3), (2, 10, 1)]);
        assert!(ctc_tokens(&[0, 0], 0).is_empty());
    }
}
