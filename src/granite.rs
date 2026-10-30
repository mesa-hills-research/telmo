//! IBM's Granite Speech 5.0 TurboCTC: a Conformer encoder with attention inside blocks of
//! frames and a CTC head, run with greedy decoding.
//!
//! Weights are read from the Hugging Face `model.safetensors` (the `GraniteSpeech5ForCTC`
//! layout, whose bf16 values widen to f32 exactly), with `config.json` and `tokenizer.json`
//! beside it.
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
use crate::parakeet::{Options, Timings, Token, Transcript};
use crate::safetensors::SafeTensors;
use crate::threads::{Pool, SharedMut};
use crate::tokenizer::Tokenizer;

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
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

struct Linear {
    w: PackedWeights,
    b: Vec<f32>,
}

/// A feed-forward module and the norm before it.
struct FeedForward {
    norm: Norm,
    up: Linear,
    down: Linear,
}

struct Block {
    strided: bool,
    ff1: FeedForward,
    norm_att: Norm,
    /// q, k and v stacked, `[3 * d_model, d_model]`, without biases.
    qkv: PackedWeights,
    /// The rows of the relative position table that distances inside a block reach, from
    /// `context - 1` down to `1 - context`, times the attention's scale.
    rel: PackedWeights,
    out: Linear,
    norm_conv: Norm,
    pw1: Linear,
    /// Depthwise weights `[kernel, conv_channels]` with the BatchNorm folded in.
    dw_w: Vec<f32>,
    dw_b: Vec<f32>,
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

/// Reads tensors by name, widened to f32.
struct Loader {
    st: SafeTensors,
}

impl Loader {
    fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        self.st.read_f32_widened(name, Some(shape))
    }
}

/// Reads every weight the model uses and puts it in the kernels' layout.
fn load_weights(ld: &mut Loader, c: &Config) -> Result<Weights> {
    let linear = |ld: &mut Loader, prefix: &str, n: usize, k: usize| -> Result<Linear> {
        let w = ld.take(&format!("{prefix}.weight"), &[n, k])?;
        Ok(Linear {
            w: PackedWeights::from_rows(&w, n, k),
            b: ld.take(&format!("{prefix}.bias"), &[n])?,
        })
    };
    let norm = |ld: &mut Loader, prefix: &str, d: usize| -> Result<Norm> {
        Ok(Norm {
            gamma: ld.take(&format!("{prefix}.weight"), &[d])?,
            beta: ld.take(&format!("{prefix}.bias"), &[d])?,
        })
    };
    let feed_forward = |ld: &mut Loader, block: &str, n: usize| -> Result<FeedForward> {
        Ok(FeedForward {
            norm: norm(ld, &format!("{block}.norm_feed_forward{n}"), c.d_model)?,
            up: linear(ld, &format!("{block}.feed_forward{n}.linear1"), c.ff, c.d_model)?,
            down: linear(ld, &format!("{block}.feed_forward{n}.linear2"), c.d_model, c.ff)?,
        })
    };
    let (d, cc, k) = (c.d_model, c.conv_channels, c.conv_kernel);
    let input = linear(ld, "encoder.input_linear", d, 4 * c.n_mels)?;
    let mut blocks = Vec::with_capacity(c.layers);
    for i in 0..c.layers {
        let p = format!("encoder.layers.{i}");
        let mut qkv = ld.take(&format!("{p}.self_attn.q_proj.weight"), &[d, d])?;
        qkv.extend(ld.take(&format!("{p}.self_attn.k_proj.weight"), &[d, d])?);
        qkv.extend(ld.take(&format!("{p}.self_attn.v_proj.weight"), &[d, d])?);

        let table = ld.take(
            &format!("{p}.self_attn.rel_pos_emb.weight"),
            &[2 * c.max_relative + 1, c.head_dim],
        )?;
        // Distance i - j inside a block reads row max_relative + i - j. Keep the rows from
        // distance context - 1 down to 1 - context, so column m is distance context - 1 - m,
        // scaled in f32 as the reference scales the looked-up rows.
        let scale = c.scale();
        let rows = 2 * c.context - 1;
        let mut rel = Vec::with_capacity(rows * c.head_dim);
        for m in 0..rows {
            let r = c.max_relative + c.context - 1 - m;
            rel.extend(
                table[r * c.head_dim..(r + 1) * c.head_dim]
                    .iter()
                    .map(|&v| v * scale),
            );
        }

        let gamma = ld.take(&format!("{p}.conv.norm.weight"), &[cc])?;
        let beta = ld.take(&format!("{p}.conv.norm.bias"), &[cc])?;
        let mean = ld.take(&format!("{p}.conv.norm.running_mean"), &[cc])?;
        let var = ld.take(&format!("{p}.conv.norm.running_var"), &[cc])?;
        let dw = ld.take(&format!("{p}.conv.depthwise_conv.weight"), &[cc, 1, k])?;
        let mut dw_w = vec![0f32; k * cc];
        let mut dw_b = vec![0f32; cc];
        for ch in 0..cc {
            let s = gamma[ch] as f64 / (var[ch] as f64 + 1e-5).sqrt();
            for j in 0..k {
                dw_w[j * cc + ch] = (dw[ch * k + j] as f64 * s) as f32;
            }
            dw_b[ch] = (beta[ch] as f64 - mean[ch] as f64 * s) as f32;
        }

        blocks.push(Block {
            strided: c.subsample_layers.contains(&i),
            ff1: feed_forward(ld, &p, 1)?,
            norm_att: norm(ld, &format!("{p}.norm_self_att"), d)?,
            qkv: PackedWeights::from_rows(&qkv, 3 * d, d),
            rel: PackedWeights::from_rows(&rel, rows, c.head_dim),
            out: linear(ld, &format!("{p}.self_attn.o_proj"), d, d)?,
            norm_conv: norm(ld, &format!("{p}.norm_conv"), d)?,
            pw1: linear(ld, &format!("{p}.conv.pointwise_lin1"), 2 * cc, d)?,
            dw_w,
            dw_b,
            pw2: linear(ld, &format!("{p}.conv.pointwise_lin2"), d, cc)?,
            ff2: feed_forward(ld, &p, 2)?,
            norm_out: norm(ld, &format!("{p}.norm_out"), d)?,
        });
    }
    Ok(Weights {
        input,
        blocks,
        head: linear(ld, "encoder.out", c.vocab, d)?,
        mid: linear(ld, "encoder.out_mid", d, c.vocab)?,
    })
}

/// A loaded Granite Speech CTC model.
pub struct Granite {
    config: Config,
    front_end: GraniteFrontEnd,
    weights: Weights,
    tokenizer: Tokenizer,
    kernels: Kernels,
    pool: Pool,
}

/// Rows of CTC logits computed at once, which bounds the memory they take.
const LOGIT_ROWS: usize = 256;

fn timed<T>(timings: &mut Timings, name: &'static str, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let out = f();
    timings.add(name, start.elapsed());
    out
}

impl Granite {
    /// Loads `model.safetensors`, `config.json` and `tokenizer.json` from a directory.
    pub fn load(dir: &Path, options: &Options) -> Result<Granite> {
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
        let weights = load_weights(&mut ld, &config)?;
        Ok(Granite {
            front_end: GraniteFrontEnd::new(config.n_mels),
            config,
            weights,
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
        let linear = |l: &Linear| l.w.bytes() + l.b.len() * 4;
        let w = &self.weights;
        let mut total = linear(&w.input) + linear(&w.head) + linear(&w.mid);
        for b in &w.blocks {
            total += [
                &b.ff1.up, &b.ff1.down, &b.out, &b.pw1, &b.pw2, &b.ff2.up, &b.ff2.down,
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
            self.kernels.linear(
                &self.pool,
                features,
                rows,
                &self.weights.input.w,
                &mut x,
                &Epilogue::bias(&self.weights.input.b),
            )
        });
        tap("input_linear", rows, &x);
        let mut s = Scratch::new(rows, c);
        let mut t = rows;
        for (i, block) in self.weights.blocks.iter().enumerate() {
            t = self.block(block, &mut x, t, &mut s, timings);
            tap(&format!("layer.{i}"), t, &x[..t * d]);
            if i == c.self_conditioning_layer() {
                timed(timings, "self_conditioning", || {
                    self.self_conditioning(&x[..t * d], t, &mut s)
                });
                tap("selfcond_injection", t, &s.att[..t * d]);
                for (v, &a) in x[..t * d].iter_mut().zip(&s.att[..t * d]) {
                    *v += a;
                }
            }
        }
        x.truncate(t * d);
        Ok(Encoded {
            frames: t,
            width: d,
            data: x,
        })
    }

    /// The CTC logits of frames `first..first + rows`, `[rows, vocab]`.
    pub fn ctc_logits(&self, encoded: &Encoded, first: usize, rows: usize) -> Vec<f32> {
        let d = self.config.d_model;
        let mut logits = vec![0f32; rows * self.config.vocab];
        self.kernels.linear(
            &self.pool,
            &encoded.data[first * d..(first + rows) * d],
            rows,
            &self.weights.head.w,
            &mut logits,
            &Epilogue::bias(&self.weights.head.b),
        );
        logits
    }

    /// The most likely token of every frame, the first one where several tie.
    pub fn frame_ids(&self, encoded: &Encoded) -> Vec<u32> {
        let vocab = self.config.vocab;
        let mut ids = Vec::with_capacity(encoded.frames);
        for first in (0..encoded.frames).step_by(LOGIT_ROWS) {
            let rows = (encoded.frames - first).min(LOGIT_ROWS);
            let logits = self.ctc_logits(encoded, first, rows);
            ids.extend(logits.chunks_exact(vocab).map(|row| argmax(row) as u32));
        }
        ids
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

    /// Runs one block over `t` frames of `x` and returns the frames it leaves, `t / 2` for a
    /// strided block.
    fn block(
        &self,
        b: &Block,
        x: &mut [f32],
        t: usize,
        s: &mut Scratch,
        timings: &mut Timings,
    ) -> usize {
        let c = &self.config;
        let (d, cc) = (c.d_model, c.conv_channels);
        let (k, pool) = (&self.kernels, &self.pool);

        self.feed_forward(&b.ff1, x, t, s, timings);

        timed(timings, "layer_norm", || self.norm(&b.norm_att, x, &mut s.h, t));
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

        // Convolution module. A strided block keeps every second frame of the convolution and
        // adds it to the input averaged over pairs of frames, dropping a last odd frame.
        let out_t = if b.strided { t / 2 } else { t };
        let stride = if b.strided { 2 } else { 1 };
        timed(timings, "layer_norm", || {
            self.norm(&b.norm_conv, x, &mut s.h, t)
        });
        timed(timings, "conv_pw1", || {
            k.linear(
                pool,
                &s.h[..t * d],
                t,
                &b.pw1.w,
                &mut s.big[..t * 2 * cc],
                &Epilogue::bias(&b.pw1.b),
            )
        });
        timed(timings, "glu", || {
            let big = &s.big;
            for_rows(pool, &mut s.glu, cc, t, 16, &|first, part| {
                for (r, row) in part.chunks_exact_mut(cc).enumerate() {
                    let g = &big[(first + r) * 2 * cc..(first + r + 1) * 2 * cc];
                    (k.glu)(&g[..cc], &g[cc..], row);
                }
            });
        });
        timed(timings, "depthwise_conv", || {
            let input = &s.glu[..t * cc];
            for_rows(pool, &mut s.big, cc, out_t, 8, &|first, part| {
                for (r, row) in part.chunks_exact_mut(cc).enumerate() {
                    (k.dwconv1d_frame)(
                        input,
                        cc,
                        t,
                        stride * (first + r),
                        &b.dw_w,
                        c.conv_kernel,
                        &b.dw_b,
                        row,
                    );
                    (k.silu)(row);
                }
            });
        });
        if b.strided {
            timed(timings, "pool", || {
                for (r, row) in s.h[..out_t * d].chunks_exact_mut(d).enumerate() {
                    let pair = &x[2 * r * d..(2 * r + 2) * d];
                    for ((o, &p), &q) in row.iter_mut().zip(&pair[..d]).zip(&pair[d..]) {
                        *o = (p + q) / 2.0;
                    }
                }
                x[..out_t * d].copy_from_slice(&s.h[..out_t * d]);
            });
        }
        let t = out_t;
        timed(timings, "conv_pw2", || {
            k.linear(
                pool,
                &s.big[..t * cc],
                t,
                &b.pw2.w,
                &mut x[..t * d],
                &Epilogue {
                    bias: Some(&b.pw2.b),
                    accumulate: true,
                    ..Epilogue::PLAIN
                },
            )
        });

        self.feed_forward(&b.ff2, x, t, s, timings);

        timed(timings, "layer_norm", || {
            self.norm(&b.norm_out, x, &mut s.h, t);
            x[..t * d].copy_from_slice(&s.h[..t * d]);
        });
        t
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
                &Epilogue::PLAIN,
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

    /// Writes `out_mid(softmax(out(x)))` to `s.att`, a chunk of frames at a time.
    fn self_conditioning(&self, x: &[f32], t: usize, s: &mut Scratch) {
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
                &mut s.att[first * d..(first + rows) * d],
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

/// Buffers reused across blocks, sized for the first block's frames.
struct Scratch {
    h: Vec<f32>,
    big: Vec<f32>,
    att: Vec<f32>,
    glu: Vec<f32>,
}

impl Scratch {
    fn new(t: usize, c: &Config) -> Scratch {
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
        assert_eq!((c.vocab, c.blank, c.self_conditioning_layer()), (16384, 0, 7));
        assert_eq!(c.frames_for(337), 84);
        assert_eq!(c.frames_for(3), 0);
        for (from, to) in [
            ("granite_speech5_ctc", "parakeet_tdt"),
            (r#""hidden_act":"silu""#, r#""hidden_act":"relu""#),
            (r#""context_size":128"#, r#""context_size":1024"#),
            (r#""tie_word_embeddings":true"#, r#""tie_word_embeddings":false"#),
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
        let got: Vec<(u32, u32, u32)> = tokens
            .iter()
            .map(|t| (t.id, t.frame, t.duration))
            .collect();
        assert_eq!(got, vec![(5, 1, 2), (5, 4, 1), (7, 5, 3), (2, 10, 1)]);
        assert!(ctc_tokens(&[0, 0], 0).is_empty());
    }
}
