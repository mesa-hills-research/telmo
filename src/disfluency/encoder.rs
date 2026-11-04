//! A ModernBERT encoder with a token classification head, as Hugging Face's
//! `ModernBertForTokenClassification` computes it.
//!
//! Each layer is pre-norm attention (the first layer has no norm before it) with fused q, k and
//! v, rotary positions and no biases, then a pre-norm GeGLU. Every `global_every`-th layer from
//! the first attends to all tokens, and the others to a band of `local_window / 2` tokens either
//! side. The head is the final norm, a dense layer, GELU and a norm, then the classifier, which
//! alone has a bias.

use crate::error::{Error, Result};
use crate::json;
use crate::kernels::{BView, Epilogue, Kernels, PackedWeights, for_rows};
use crate::pack::{Pack, Sink};
use crate::quant::Precision;
use crate::region::Vector;
use crate::source::Source;
use crate::threads::{Pool, SharedMut};

/// The architecture, read from `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub labels: usize,
    /// Layers `0, n, 2n, ...` attend globally.
    pub global_every: usize,
    /// The span of a local layer's attention, half on each side.
    pub local_window: usize,
    pub global_theta: f64,
    pub local_theta: f64,
    pub norm_eps: f32,
    pub cls: u32,
    pub sep: u32,
}

impl Config {
    pub fn from_json(bytes: &[u8]) -> Result<Config> {
        let root = json::parse(bytes)?;
        let bad = |m: String| Error::Model(format!("The tagger's config.json isn't usable. {m}"));
        let n = |key: &str| -> Result<usize> {
            root.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| bad(format!("It has no {key}.")))
        };
        if root.get("model_type").and_then(|t| t.as_str()) != Some("modernbert") {
            return Err(bad("It isn't a ModernBERT model.".into()));
        }
        for key in ["attention_bias", "mlp_bias", "norm_bias", "classifier_bias"] {
            if root.get(key).and_then(|v| v.as_bool()) != Some(false) {
                return Err(bad(format!("Telmo runs it only with {key} false.")));
            }
        }
        for key in ["hidden_activation", "classifier_activation"] {
            if root.get(key).and_then(|v| v.as_str()) != Some("gelu") {
                return Err(bad(format!("Telmo runs it only with {key} gelu.")));
            }
        }
        // Newer files give each attention type's rotary base, older ones a global and a local.
        let theta = |kind: &str, old: &str| -> Result<f64> {
            root.get("rope_parameters")
                .and_then(|r| r.get(kind))
                .and_then(|r| r.get("rope_theta"))
                .or_else(|| root.get(old))
                .and_then(|v| v.as_f64())
                .ok_or_else(|| bad(format!("It has no rotary base for {kind}.")))
        };
        let labels = root
            .get("id2label")
            .and_then(|l| l.as_object())
            .map(|l| l.len())
            .ok_or_else(|| bad("It has no labels.".into()))?;
        let config = Config {
            hidden: n("hidden_size")?,
            heads: n("num_attention_heads")?,
            layers: n("num_hidden_layers")?,
            intermediate: n("intermediate_size")?,
            vocab: n("vocab_size")?,
            labels,
            global_every: n("global_attn_every_n_layers")?,
            local_window: n("local_attention")?,
            global_theta: theta("full_attention", "global_rope_theta")?,
            local_theta: theta("sliding_attention", "local_rope_theta")?,
            norm_eps: root
                .get("norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-5) as f32,
            cls: n("cls_token_id")? as u32,
            sep: n("sep_token_id")? as u32,
        };
        let c = &config;
        if c.heads == 0
            || !c.hidden.is_multiple_of(c.heads)
            || !(c.hidden / c.heads).is_multiple_of(2)
            || c.global_every == 0
            || c.layers == 0
            || c.labels == 0
            || [c.hidden, c.intermediate, c.vocab, c.layers]
                .iter()
                .any(|&v| v > 1 << 20)
        {
            return Err(bad("Its sizes don't fit together.".into()));
        }
        if let Some(types) = root.get("layer_types").and_then(|t| t.as_array()) {
            let expected = (0..c.layers).map(|i| c.is_global(i));
            let given = types.iter().map(|t| t.as_str() == Some("full_attention"));
            if types.len() != c.layers || !expected.eq(given) {
                return Err(bad("Its layer types aren't every n-th layer global.".into()));
            }
        }
        Ok(config)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    pub fn is_global(&self, layer: usize) -> bool {
        layer.is_multiple_of(self.global_every)
    }
}

/// One layer's weights.
struct Layer {
    /// None for the first layer, which has no norm before attention.
    attn_norm: Option<Vector>,
    /// q, k and v stacked, `[3 * hidden, hidden]`.
    qkv: PackedWeights,
    out: PackedWeights,
    mlp_norm: Vector,
    /// The GeGLU's input and gate halves stacked, `[2 * intermediate, hidden]`.
    up: PackedWeights,
    down: PackedWeights,
}

/// One trained model, an ensemble's member.
pub(crate) struct Member {
    /// The token embedding as a matrix of `vocab` outputs by `hidden` inputs, so each row is one
    /// token's vector.
    embedding: PackedWeights,
    embedding_norm: Vector,
    layers: Vec<Layer>,
    final_norm: Vector,
    head_dense: PackedWeights,
    head_norm: Vector,
    classifier: PackedWeights,
    classifier_bias: Vector,
}

/// Converts a member's weights, named as in the Hugging Face checkpoint after `prefix`, into
/// pack entries named after `to`. Matrices take `precision`, and norms and the classifier's bias
/// stay f32.
pub(crate) fn convert_member(
    src: &mut dyn Source,
    c: &Config,
    prefix: &str,
    to: &str,
    precision: Precision,
    sink: &mut dyn Sink,
) -> Result<()> {
    let (d, ff) = (c.hidden, c.intermediate);
    let mut matrix = |src: &mut dyn Source, from: &str, name: &str, n: usize, k: usize| {
        let w = src.matrix(&format!("{prefix}{from}.weight"), &[n, k])?;
        sink.matrix(&format!("{to}{name}"), &w.rows(), n, k, precision)
    };
    matrix(
        src,
        "model.embeddings.tok_embeddings",
        "embedding",
        c.vocab,
        d,
    )?;
    for i in 0..c.layers {
        let (p, o) = (format!("model.layers.{i}"), format!("layers.{i}"));
        matrix(
            src,
            &format!("{p}.attn.Wqkv"),
            &format!("{o}.qkv"),
            3 * d,
            d,
        )?;
        matrix(src, &format!("{p}.attn.Wo"), &format!("{o}.out"), d, d)?;
        matrix(src, &format!("{p}.mlp.Wi"), &format!("{o}.up"), 2 * ff, d)?;
        matrix(src, &format!("{p}.mlp.Wo"), &format!("{o}.down"), d, ff)?;
    }
    matrix(src, "head.dense", "head.dense", d, d)?;
    matrix(src, "classifier", "classifier", c.labels, d)?;
    let mut norms: Vec<(String, String)> = vec![
        ("model.embeddings.norm".into(), "embedding_norm".into()),
        ("model.final_norm".into(), "final_norm".into()),
        ("head.norm".into(), "head.norm".into()),
    ];
    for i in 0..c.layers {
        let names = ["attn_norm", "mlp_norm"]
            .map(|n| (format!("model.layers.{i}.{n}"), format!("layers.{i}.{n}")));
        // The first layer has no norm before attention.
        norms.extend(names.into_iter().skip(usize::from(i == 0)));
    }
    for (from, name) in norms {
        let w = src.tensor(&format!("{prefix}{from}.weight"), &[d])?;
        sink.f32(&format!("{to}{name}"), &[d], &w)?;
    }
    let b = src.tensor(&format!("{prefix}classifier.bias"), &[c.labels])?;
    sink.f32(&format!("{to}classifier.b"), &[c.labels], &b)
}

impl Member {
    /// The member whose pack entries start with `prefix`.
    pub(crate) fn from_pack(pack: &Pack, c: &Config, prefix: &str) -> Result<Member> {
        let (d, ff) = (c.hidden, c.intermediate);
        let m = |name: &str, n: usize, k: usize| pack.matrix(&format!("{prefix}{name}"), n, k);
        let v = |name: &str, len: usize| pack.vector(&format!("{prefix}{name}"), &[len]);
        let layers = (0..c.layers)
            .map(|i| {
                let o = format!("layers.{i}");
                Ok(Layer {
                    attn_norm: (i > 0)
                        .then(|| v(&format!("{o}.attn_norm"), d))
                        .transpose()?,
                    qkv: m(&format!("{o}.qkv"), 3 * d, d)?,
                    out: m(&format!("{o}.out"), d, d)?,
                    mlp_norm: v(&format!("{o}.mlp_norm"), d)?,
                    up: m(&format!("{o}.up"), 2 * ff, d)?,
                    down: m(&format!("{o}.down"), d, ff)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Member {
            embedding: m("embedding", c.vocab, d)?,
            embedding_norm: v("embedding_norm", d)?,
            layers,
            final_norm: v("final_norm", d)?,
            head_dense: m("head.dense", d, d)?,
            head_norm: v("head.norm", d)?,
            classifier: m("classifier", c.labels, d)?,
            classifier_bias: v("classifier.b", c.labels)?,
        })
    }

    /// Bytes of weights held.
    pub(crate) fn weight_bytes(&self) -> usize {
        let mut total = self.embedding.bytes()
            + self.head_dense.bytes()
            + self.classifier.bytes()
            + 4 * (self.embedding_norm.len()
                + self.final_norm.len()
                + self.head_norm.len()
                + self.classifier_bias.len());
        for l in &self.layers {
            total += l.qkv.bytes() + l.out.bytes() + l.up.bytes() + l.down.bytes();
            total += 4 * (l.mlp_norm.len() + l.attn_norm.as_ref().map_or(0, |n| n.len()));
        }
        total
    }
}

/// What runs a member, the architecture, the kernels and the pool.
pub(crate) struct Runner<'a> {
    pub config: &'a Config,
    pub kernels: &'a Kernels,
    pub pool: &'a Pool,
}

impl Runner<'_> {
    /// The logits of every token of `ids`, `[ids.len(), labels]`. `tap` sees the embeddings'
    /// output (`"embeddings"`), each layer's (`"layer.0"`, ...), the final norm's
    /// (`"final_norm"`) and the head's before the classifier (`"head"`), each
    /// `[tokens, hidden]`.
    pub fn logits(
        &self,
        member: &Member,
        ids: &[u32],
        tap: &mut dyn FnMut(&str, &[f32]),
    ) -> Vec<f32> {
        let c = self.config;
        let (k, pool) = (self.kernels, self.pool);
        let (t, d, ff) = (ids.len(), c.hidden, c.intermediate);
        let zeros = vec![0f32; d];
        let mut x = vec![0f32; t * d];
        for (row, &id) in x.chunks_exact_mut(d).zip(ids) {
            member.embedding.decode_row(id as usize, row);
        }
        let mut h = vec![0f32; t * d];
        self.norm(&x, &mut h, &member.embedding_norm, &zeros);
        std::mem::swap(&mut x, &mut h);
        tap("embeddings", &x);
        let mut qkv = vec![0f32; t * 3 * d];
        let mut att = vec![0f32; t * d];
        let mut big = vec![0f32; t * 2 * ff];
        let mut gated = vec![0f32; t * ff];
        let accumulate = Epilogue {
            accumulate: true,
            ..Epilogue::PLAIN
        };
        for (i, layer) in member.layers.iter().enumerate() {
            let input = match &layer.attn_norm {
                Some(norm) => {
                    self.norm(&x, &mut h, norm, &zeros);
                    &h
                }
                None => &x,
            };
            k.linear(pool, input, t, &layer.qkv, &mut qkv, &Epilogue::PLAIN);
            let theta = if c.is_global(i) {
                c.global_theta
            } else {
                c.local_theta
            };
            self.rotate(&mut qkv, t, theta);
            let band = (!c.is_global(i)).then_some(c.local_window / 2);
            self.attention(&qkv, t, band, &mut att);
            k.linear(pool, &att, t, &layer.out, &mut x, &accumulate);
            self.norm(&x, &mut h, &layer.mlp_norm, &zeros);
            k.linear(pool, &h, t, &layer.up, &mut big, &Epilogue::PLAIN);
            for_rows(pool, &mut gated, ff, t, 8, &|first, part| {
                for (r, out) in part.chunks_exact_mut(ff).enumerate() {
                    let row = &big[(first + r) * 2 * ff..(first + r + 1) * 2 * ff];
                    for (j, o) in out.iter_mut().enumerate() {
                        *o = gelu(row[j]) * row[ff + j];
                    }
                }
            });
            k.linear(pool, &gated, t, &layer.down, &mut x, &accumulate);
            tap(&format!("layer.{i}"), &x);
        }
        self.norm(&x, &mut h, &member.final_norm, &zeros);
        tap("final_norm", &h);
        k.linear(pool, &h, t, &member.head_dense, &mut x, &Epilogue::PLAIN);
        for v in x.iter_mut() {
            *v = gelu(*v);
        }
        self.norm(&x, &mut h, &member.head_norm, &zeros);
        tap("head", &h);
        let mut logits = vec![0f32; t * c.labels];
        k.linear(
            pool,
            &h,
            t,
            &member.classifier,
            &mut logits,
            &Epilogue::bias(&member.classifier_bias),
        );
        logits
    }

    fn norm(&self, x: &[f32], out: &mut [f32], gamma: &[f32], beta: &[f32]) {
        let (d, eps) = (self.config.hidden, self.config.norm_eps);
        let k = self.kernels;
        let rows = x.len() / d;
        for_rows(self.pool, out, d, rows, 16, &|first, part| {
            for (r, row) in part.chunks_exact_mut(d).enumerate() {
                (k.layer_norm)(
                    &x[(first + r) * d..(first + r + 1) * d],
                    row,
                    gamma,
                    beta,
                    eps,
                );
            }
        });
    }

    /// Applies rotary positions to the queries and keys of `qkv`, `[t, 3 * hidden]`, rotating
    /// each head's first half against its second as Hugging Face's `rotate_half` does.
    fn rotate(&self, qkv: &mut [f32], t: usize, theta: f64) {
        let c = self.config;
        let (d, hd) = (c.hidden, c.head_dim());
        let half = hd / 2;
        // The frequencies in f32, as the reference computes them.
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / hd as f32))
            .collect();
        for (pos, row) in qkv.chunks_exact_mut(3 * d).enumerate().take(t) {
            for (i, &f) in inv.iter().enumerate() {
                let angle = pos as f32 * f;
                let (sin, cos) = (angle.sin(), angle.cos());
                for head in 0..2 * c.heads {
                    let base = head * hd;
                    let (a, b) = (row[base + i], row[base + half + i]);
                    row[base + i] = a * cos - b * sin;
                    row[base + half + i] = b * cos + a * sin;
                }
            }
        }
    }

    /// Scaled dot-product attention of each head over `qkv`, writing the heads side by side to
    /// `att`. With a `band`, token i attends only to tokens within `band` of it. Each task is one
    /// head, on the serial kernels, so no task waits on the pool.
    fn attention(&self, qkv: &[f32], t: usize, band: Option<usize>, att: &mut [f32]) {
        let c = self.config;
        let (d, hd) = (c.hidden, c.head_dim());
        let k = self.kernels;
        let scale = (hd as f64).powf(-0.5) as f32;
        let out = SharedMut(att.as_mut_ptr());
        self.pool.run(c.heads, &|h| {
            // The keys transposed, `[hd, t]`, so the scores read them as panels.
            let ld = t.next_multiple_of(16);
            let mut kt = vec![0f32; hd * ld];
            for j in 0..t {
                let key = &qkv[j * 3 * d + d + h * hd..][..hd];
                for (i, &v) in key.iter().enumerate() {
                    kt[i * ld + j] = v;
                }
            }
            let mut scores = vec![0f32; t * t];
            k.matmul_serial(
                &qkv[h * hd..],
                3 * d,
                t,
                &BView::row_major(&kt, hd, t, ld),
                &mut scores,
                t,
                &Epilogue {
                    scale,
                    ..Epilogue::PLAIN
                },
            );
            for i in 0..t {
                let row = &mut scores[i * t..(i + 1) * t];
                let (lo, hi) = match band {
                    Some(b) => (i.saturating_sub(b), (i + b + 1).min(t)),
                    None => (0, t),
                };
                row[..lo].fill(0.0);
                row[hi..].fill(0.0);
                (k.softmax)(&mut row[lo..hi]);
            }
            // SAFETY: each task writes the columns of its own head, inside `att`.
            unsafe {
                k.matmul_serial_raw(
                    &scores,
                    t,
                    t,
                    &BView::row_major(&qkv[2 * d + h * hd..], t, hd, 3 * d),
                    SharedMut(out.get().add(h * hd)),
                    d,
                    &Epilogue::PLAIN,
                );
            }
        });
    }
}

/// GELU with the exact error function, `x * (1 + erf(x / sqrt 2)) / 2`.
pub(crate) fn gelu(x: f32) -> f32 {
    let x = x as f64;
    (0.5 * x * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2))) as f32
}

/// The error function, by the Chebyshev fit of erfc in Numerical Recipes, whose relative error
/// is below 1.2e-7, under f32's rounding of GELU's `1 + erf`.
fn erf(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z - 1.26551223
        + t * (1.00002368
            + t * (0.37409196
                + t * (0.09678418
                    + t * (-0.18628806
                        + t * (0.27886807
                            + t * (-1.13520398
                                + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
    let erfc = t * poly.exp();
    if x >= 0.0 { 1.0 - erfc } else { erfc - 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_and_gelu_match_known_values() {
        for (x, want) in [
            (0.0, 0.0),
            (0.5, 0.520_499_877_8),
            (1.0, 0.842_700_792_9),
            (2.0, 0.995_322_265_0),
            (-1.0, -0.842_700_792_9),
            (3.5, 0.999_999_256_8),
        ] {
            assert!((erf(x) - want).abs() < 2e-7, "erf({x}) = {}", erf(x));
        }
        assert!((gelu(1.0) - 0.841_344_8).abs() < 1e-6);
        assert!((gelu(-3.0) + 0.004_049_7).abs() < 1e-6);
        assert_eq!(gelu(0.0), 0.0);
    }
}
