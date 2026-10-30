//! Plain Rust kernels. They run on any CPU and are the reference the others are tested against.

use super::{Act, Isa, Kernels, PANEL};

pub(crate) fn kernels() -> Kernels {
    Kernels {
        isa: Isa::Scalar,
        tile,
        store,
        layer_norm,
        softmax,
        glu,
        silu,
        axpy,
        dwconv1d_frame,
    }
}

/// # Safety
/// `a` must hold `rows` rows of `k` values `lda` apart, and `b` must hold `k` runs of 16 values
/// `k_stride` apart.
unsafe fn tile(
    k: usize,
    a: *const f32,
    lda: usize,
    rows: usize,
    b: *const f32,
    k_stride: usize,
    out: &mut [f32; 96],
) {
    out.fill(0.0);
    for r in 0..rows {
        let acc = &mut out[r * PANEL..(r + 1) * PANEL];
        for i in 0..k {
            // SAFETY: in bounds by the function's contract.
            let (av, bp) = unsafe { (*a.add(r * lda + i), b.add(i * k_stride)) };
            for (j, slot) in acc.iter_mut().enumerate() {
                *slot += av * unsafe { *bp.add(j) };
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

#[inline(always)]
pub(crate) fn axpy(alpha: f32, y: &[f32], x: &mut [f32]) {
    for (xi, &yi) in x.iter_mut().zip(y) {
        *xi += alpha * yi;
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
