//! Plain Rust kernels. They run on any CPU and are the reference the others are tested against.

use super::{Act, Isa, Kernels, PANEL, TILE_ROWS};
use crate::quant::{Precision, Q8_BLOCK, bf16_to_f32, f16_to_f32};

pub(crate) fn kernels() -> Kernels {
    Kernels {
        isa: Isa::Scalar,
        tile_rows: TILE_ROWS,
        tile,
        widen,
        tile_narrow: None,
        store,
        layer_norm,
        softmax,
        glu,
        silu,
        dwconv1d_frame,
        conv3x3s2_row,
    }
}

/// # Safety
/// `a` must hold `rows` rows of `k` values `lda` apart, `b` must hold `k` runs of 16 values
/// `k_stride` apart, and `out` must hold `rows` rows of 16.
unsafe fn tile(
    k: usize,
    a: *const f32,
    lda: usize,
    rows: usize,
    b: *const f32,
    k_stride: usize,
    out: *mut f32,
    load: bool,
) {
    for r in 0..rows {
        // SAFETY: in bounds by the function's contract.
        let acc = unsafe { std::slice::from_raw_parts_mut(out.add(r * PANEL), PANEL) };
        if !load {
            acc.fill(0.0);
        }
        for i in 0..k {
            let (av, bp) = unsafe { (*a.add(r * lda + i), b.add(i * k_stride)) };
            for (j, slot) in acc.iter_mut().enumerate() {
                *slot += av * unsafe { *bp.add(j) };
            }
        }
    }
}

/// # Safety
/// `src` must hold `k` runs of 16 values in `precision`, `scales` the scales the kernel table
/// describes, and `out` room for `k * 16` floats.
pub(crate) unsafe fn widen(
    precision: Precision,
    k: usize,
    src: *const u8,
    scales: *const f32,
    out: *mut f32,
) {
    // SAFETY: in bounds by the function's contract.
    let out = unsafe { std::slice::from_raw_parts_mut(out, k * PANEL) };
    let n = k * PANEL;
    match precision {
        Precision::F32 => {
            let src = unsafe { std::slice::from_raw_parts(src, n * 4) };
            for (o, b) in out.iter_mut().zip(src.chunks_exact(4)) {
                *o = f32::from_le_bytes(b.try_into().unwrap());
            }
        }
        Precision::F16 | Precision::Bf16 => {
            let src = unsafe { std::slice::from_raw_parts(src, n * 2) };
            let convert = if precision == Precision::F16 {
                f16_to_f32
            } else {
                bf16_to_f32
            };
            for (o, b) in out.iter_mut().zip(src.chunks_exact(2)) {
                *o = convert(u16::from_le_bytes([b[0], b[1]]));
            }
        }
        Precision::Int8 => {
            let src = unsafe { std::slice::from_raw_parts(src as *const i8, n) };
            let scales = unsafe { std::slice::from_raw_parts(scales, PANEL) };
            for (o, q) in out.chunks_exact_mut(PANEL).zip(src.chunks_exact(PANEL)) {
                for j in 0..PANEL {
                    o[j] = q[j] as f32 * scales[j];
                }
            }
        }
        Precision::Q8_0 => {
            let src = unsafe { std::slice::from_raw_parts(src as *const i8, n) };
            let scales =
                unsafe { std::slice::from_raw_parts(scales, k.div_ceil(Q8_BLOCK) * PANEL) };
            for (i, (o, q)) in out
                .chunks_exact_mut(PANEL)
                .zip(src.chunks_exact(PANEL))
                .enumerate()
            {
                let s = &scales[i / Q8_BLOCK * PANEL..][..PANEL];
                for j in 0..PANEL {
                    o[j] = q[j] as f32 * s[j];
                }
            }
        }
    }
}

#[inline(always)]
pub(crate) fn activate(v: f32, act: Act) -> f32 {
    match act {
        Act::None => v,
        Act::Relu => v.max(0.0),
        Act::Silu => v / (1.0 + (-v).exp()),
    }
}

fn store(acc: &[f32], c: &mut [f32], bias: Option<&[f32]>, act: Act, scale: f32, accumulate: bool) {
    for (j, out) in c.iter_mut().enumerate() {
        let v = acc[j] + bias.map_or(0.0, |b| b[j]);
        let v = activate(v, act) * scale;
        *out = if accumulate { *out + v } else { v };
    }
}

fn layer_norm(x: &[f32], out: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let rstd = 1.0 / (var + eps as f64).sqrt();
    for i in 0..x.len() {
        out[i] = (((x[i] as f64 - mean) * rstd) as f32) * gamma[i] + beta[i];
    }
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let mut sum = 0f64;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v as f64;
    }
    let inv = (1.0 / sum) as f32;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

fn glu(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..out.len() {
        out[i] = a[i] / (1.0 + (-b[i]).exp());
    }
}

fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = activate(*v, Act::Silu);
    }
}

/// One output frame of a depthwise convolution with "same" zero padding: for each channel,
/// `bias + sum over j of w[j][c] * input[t + j - kernel/2][c]`. Weights are `[kernel, channels]`.
#[inline(always)]
pub(crate) fn dwconv1d_frame(
    input: &[f32],
    channels: usize,
    frames: usize,
    t: usize,
    w: &[f32],
    kernel: usize,
    bias: &[f32],
    out: &mut [f32],
) {
    let out = &mut out[..channels];
    out.copy_from_slice(&bias[..channels]);
    let half = kernel / 2;
    for j in 0..kernel {
        let Some(src) = (t + j).checked_sub(half).filter(|&s| s < frames) else {
            continue;
        };
        let row = &input[src * channels..(src + 1) * channels];
        let wj = &w[j * channels..(j + 1) * channels];
        for c in 0..channels {
            out[c] += wj[c] * row[c];
        }
    }
}

/// One output row (time step `t`) of a 3x3 convolution with stride 2 and zero padding 1 over
/// a channels-last `[in_t, in_f, channels]` input. Depthwise, each channel convolves its own
/// input. Otherwise the input has a single channel (`[in_t, in_f]`) that every output channel
/// reads. Weights are `[9, channels]`, tap `dt * 3 + df` first. Writes `[out_f, channels]`.
#[inline(always)]
pub(crate) fn conv3x3s2_row(
    input: &[f32],
    in_t: usize,
    in_f: usize,
    channels: usize,
    depthwise: bool,
    w: &[f32],
    bias: &[f32],
    t: usize,
    out: &mut [f32],
) {
    let out_f = (in_f - 1) / 2 + 1;
    for f in 0..out_f {
        let o = &mut out[f * channels..(f + 1) * channels];
        o.copy_from_slice(&bias[..channels]);
        for dt in 0..3 {
            let Some(st) = (2 * t + dt).checked_sub(1).filter(|&s| s < in_t) else {
                continue;
            };
            for df in 0..3 {
                let Some(sf) = (2 * f + df).checked_sub(1).filter(|&s| s < in_f) else {
                    continue;
                };
                let wk = &w[(dt * 3 + df) * channels..(dt * 3 + df + 1) * channels];
                if depthwise {
                    let src = &input[(st * in_f + sf) * channels..(st * in_f + sf + 1) * channels];
                    for c in 0..channels {
                        o[c] += wk[c] * src[c];
                    }
                } else {
                    let v = input[st * in_f + sf];
                    for c in 0..channels {
                        o[c] += wk[c] * v;
                    }
                }
            }
        }
    }
}
