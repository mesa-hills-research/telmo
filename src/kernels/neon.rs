//! Kernels for aarch64 CPUs, which all have NEON.
//!
//! Every function here is compiled with NEON enabled. The table in [`kernels`] is only built
//! after detection confirms it, which makes the safe wrappers sound.
//!
//! These kernels compile, and the tests compare them with the scalar kernels on any aarch64
//! CPU, but they haven't run on real hardware yet.

use std::arch::aarch64::*;
use std::arch::asm;

use super::{Act, Isa, Kernels, PANEL, TILE_ROWS, scalar};
use crate::quant::{Precision, Q8_BLOCK};

pub(crate) fn kernels() -> Kernels {
    assert!(Isa::Neon.is_supported());
    Kernels {
        isa: Isa::Neon,
        tile_rows: TILE_ROWS,
        tile,
        widen,
        tile_narrow: Some(tile_narrow),
        store: |acc, c, bias, act, scale, accumulate| unsafe {
            store(acc, c, bias, act, scale, accumulate)
        },
        layer_norm: |x, out, gamma, beta, eps| unsafe { layer_norm(x, out, gamma, beta, eps) },
        softmax: |x| unsafe { softmax(x) },
        glu: |a, b, out| unsafe { glu(a, b, out) },
        silu: |x| unsafe { silu(x) },
        dwconv1d_frame: |input, channels, frames, t, w, kernel, bias, out| unsafe {
            dwconv1d_frame(input, channels, frames, t, w, kernel, bias, out)
        },
        conv3x3s2_row: |input, in_t, in_f, channels, depthwise, w, bias, t, out| unsafe {
            conv3x3s2_row(input, in_t, in_f, channels, depthwise, w, bias, t, out)
        },
    }
}

/// # Safety
/// As for the scalar tile, and the CPU must have NEON.
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
    unsafe {
        match rows {
            6 => tile_rows::<6>(k, a, lda, b, k_stride, out, load),
            5 => tile_rows::<5>(k, a, lda, b, k_stride, out, load),
            4 => tile_rows::<4>(k, a, lda, b, k_stride, out, load),
            3 => tile_rows::<3>(k, a, lda, b, k_stride, out, load),
            2 => tile_rows::<2>(k, a, lda, b, k_stride, out, load),
            1 => tile_rows::<1>(k, a, lda, b, k_stride, out, load),
            _ => unreachable!("tiles have 1 to 6 rows"),
        }
    }
}

/// An `R x 16` register tile. For each input, four 4-wide loads of the panel and one value per
/// row feed `4R` fused multiply-adds. Six rows take 24 of the 32 registers.
#[target_feature(enable = "neon")]
unsafe fn tile_rows<const R: usize>(
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    k_stride: usize,
    out: *mut f32,
    load: bool,
) {
    let mut acc = [[vdupq_n_f32(0.0); 4]; R];
    if load {
        for (r, acc) in acc.iter_mut().enumerate() {
            for (q, v) in acc.iter_mut().enumerate() {
                // SAFETY: `out` holds R rows of 16.
                *v = unsafe { vld1q_f32(out.add(r * PANEL + q * 4)) };
            }
        }
    }
    let mut rows = [a; R];
    for (r, row) in rows.iter_mut().enumerate() {
        *row = unsafe { a.add(r * lda) };
    }
    let mut bp = b;
    for i in 0..k {
        // SAFETY: the caller guarantees `k` panel steps and `k` values per row.
        unsafe {
            let bv = [
                vld1q_f32(bp),
                vld1q_f32(bp.add(4)),
                vld1q_f32(bp.add(8)),
                vld1q_f32(bp.add(12)),
            ];
            for r in 0..R {
                let av = *rows[r].add(i);
                for q in 0..4 {
                    acc[r][q] = vfmaq_n_f32(acc[r][q], bv[q], av);
                }
            }
            bp = bp.add(k_stride);
        }
    }
    for (r, acc) in acc.iter().enumerate() {
        for (q, v) in acc.iter().enumerate() {
            unsafe { vst1q_f32(out.add(r * PANEL + q * 4), *v) };
        }
    }
}

/// # Safety
/// As for `tile`, with `b` holding `k` runs of 16 narrow values and `scales` as `widen` takes
/// them, and the CPU must have NEON.
unsafe fn tile_narrow(
    precision: Precision,
    k: usize,
    a: *const f32,
    lda: usize,
    rows: usize,
    b: *const u8,
    scales: *const f32,
    out: *mut f32,
    load: bool,
) {
    macro_rules! rows {
        ($p:expr) => {
            match rows {
                6 => tile_narrow_rows::<6, { $p }>(k, a, lda, b, scales, out, load),
                5 => tile_narrow_rows::<5, { $p }>(k, a, lda, b, scales, out, load),
                4 => tile_narrow_rows::<4, { $p }>(k, a, lda, b, scales, out, load),
                3 => tile_narrow_rows::<3, { $p }>(k, a, lda, b, scales, out, load),
                2 => tile_narrow_rows::<2, { $p }>(k, a, lda, b, scales, out, load),
                1 => tile_narrow_rows::<1, { $p }>(k, a, lda, b, scales, out, load),
                _ => unreachable!("tiles have 1 to 6 rows"),
            }
        };
    }
    // SAFETY: passed on from the caller.
    unsafe {
        match precision {
            Precision::F32 => unreachable!("f32 weights use the plain tile"),
            Precision::F16 => rows!(F16),
            Precision::Bf16 => rows!(BF16),
            Precision::Int8 => rows!(INT8),
            Precision::Q8_0 => rows!(Q8_0),
        }
    }
}

const F16: u8 = 0;
const BF16: u8 = 1;
const INT8: u8 = 2;
const Q8_0: u8 = 3;

/// An `R x 16` register tile over narrow weights, each input's 16 values widened once in
/// registers (int8 times its scale) and shared by the `R` rows. The values and the order of the
/// fused multiply-adds are those of `widen` followed by `tile_rows`, so the sums are identical.
#[target_feature(enable = "neon")]
unsafe fn tile_narrow_rows<const R: usize, const P: u8>(
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const u8,
    scales: *const f32,
    out: *mut f32,
    load: bool,
) {
    let mut acc = [[vdupq_n_f32(0.0); 4]; R];
    // SAFETY: the caller guarantees `out` holds R rows of 16, `a` R rows of `k` values, `b`
    // `k` runs of 16 values, and the scales.
    unsafe {
        if load {
            for (r, acc) in acc.iter_mut().enumerate() {
                *acc = load16(out.add(r * PANEL));
            }
        }
        let mut rows = [a; R];
        for (r, row) in rows.iter_mut().enumerate() {
            *row = a.add(r * lda);
        }
        let step = if P == F16 || P == BF16 { 32 } else { 16 };
        let mut s = [vdupq_n_f32(1.0); 4];
        if P == INT8 {
            s = load16(scales);
        }
        let mut i = 0;
        while i < k {
            let end = if P == Q8_0 {
                s = load16(scales.add(i / Q8_BLOCK * PANEL));
                (i + Q8_BLOCK).min(k)
            } else {
                k
            };
            while i < end {
                let p = b.add(i * step);
                let w = if P == F16 {
                    let (lo, hi) = (
                        vld1q_u16(p as *const u16),
                        vld1q_u16(p.add(16) as *const u16),
                    );
                    [
                        f16_low_to_f32(lo),
                        f16_high_to_f32(lo),
                        f16_low_to_f32(hi),
                        f16_high_to_f32(hi),
                    ]
                } else if P == BF16 {
                    let (lo, hi) = (
                        vld1q_u16(p as *const u16),
                        vld1q_u16(p.add(16) as *const u16),
                    );
                    [
                        vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(lo))),
                        vreinterpretq_f32_u32(vshll_high_n_u16::<16>(lo)),
                        vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(hi))),
                        vreinterpretq_f32_u32(vshll_high_n_u16::<16>(hi)),
                    ]
                } else {
                    let q = vld1q_s8(p as *const i8);
                    let (lo, hi) = (vmovl_s8(vget_low_s8(q)), vmovl_high_s8(q));
                    [
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), s[0]),
                        vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(lo)), s[1]),
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), s[2]),
                        vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(hi)), s[3]),
                    ]
                };
                for r in 0..R {
                    let av = *rows[r].add(i);
                    for q in 0..4 {
                        acc[r][q] = vfmaq_n_f32(acc[r][q], w[q], av);
                    }
                }
                i += 1;
            }
        }
        for (r, acc) in acc.iter().enumerate() {
            for (q, v) in acc.iter().enumerate() {
                vst1q_f32(out.add(r * PANEL + q * 4), *v);
            }
        }
    }
}

/// The low four f16 values of `h` as f32. FCVTL is exact, and like `quant::f16_to_f32` it
/// quiets signalling NaNs and keeps their payload (with the default FPCR, which leaves the
/// default-NaN mode off). Rust's f16 vector intrinsics aren't stable, hence the assembly.
#[target_feature(enable = "neon")]
fn f16_low_to_f32(h: uint16x8_t) -> float32x4_t {
    let out: float32x4_t;
    // SAFETY: FCVTL reads and writes vector registers only.
    unsafe {
        asm!(
            "fcvtl {out:v}.4s, {h:v}.4h",
            h = in(vreg) h,
            out = lateout(vreg) out,
            options(pure, nomem, nostack, preserves_flags)
        )
    };
    out
}

/// The high four f16 values of `h` as f32, as in [`f16_low_to_f32`].
#[target_feature(enable = "neon")]
fn f16_high_to_f32(h: uint16x8_t) -> float32x4_t {
    let out: float32x4_t;
    // SAFETY: FCVTL2 reads and writes vector registers only.
    unsafe {
        asm!(
            "fcvtl2 {out:v}.4s, {h:v}.8h",
            h = in(vreg) h,
            out = lateout(vreg) out,
            options(pure, nomem, nostack, preserves_flags)
        )
    };
    out
}

/// Widens narrow weights to f32, 16 values per input. The conversions are exact (and the int8
/// products correctly rounded), so the values equal the scalar kernel's.
///
/// # Safety
/// As for the scalar `widen`, and the CPU must have NEON.
#[target_feature(enable = "neon")]
unsafe fn widen(precision: Precision, k: usize, src: *const u8, scales: *const f32, out: *mut f32) {
    // SAFETY: every pointer stays inside the ranges the caller guarantees.
    unsafe {
        match precision {
            Precision::F32 => scalar::widen(precision, k, src, scales, out),
            Precision::F16 => {
                for i in 0..k {
                    let p = src.add(i * 32) as *const u16;
                    let o = out.add(i * 16);
                    let (lo, hi) = (vld1q_u16(p), vld1q_u16(p.add(8)));
                    vst1q_f32(o, f16_low_to_f32(lo));
                    vst1q_f32(o.add(4), f16_high_to_f32(lo));
                    vst1q_f32(o.add(8), f16_low_to_f32(hi));
                    vst1q_f32(o.add(12), f16_high_to_f32(hi));
                }
            }
            Precision::Bf16 => {
                for i in 0..k {
                    let p = src.add(i * 32) as *const u16;
                    let o = out.add(i * 16);
                    for (half, h) in [vld1q_u16(p), vld1q_u16(p.add(8))].into_iter().enumerate() {
                        let lo = vshll_n_u16::<16>(vget_low_u16(h));
                        let hi = vshll_high_n_u16::<16>(h);
                        vst1q_f32(o.add(half * 8), vreinterpretq_f32_u32(lo));
                        vst1q_f32(o.add(half * 8 + 4), vreinterpretq_f32_u32(hi));
                    }
                }
            }
            Precision::Int8 => {
                let s = load16(scales);
                for i in 0..k {
                    widen_i8(src.add(i * 16), &s, out.add(i * 16));
                }
            }
            Precision::Q8_0 => {
                for b in 0..k.div_ceil(Q8_BLOCK) {
                    let s = load16(scales.add(b * PANEL));
                    for i in b * Q8_BLOCK..((b + 1) * Q8_BLOCK).min(k) {
                        widen_i8(src.add(i * 16), &s, out.add(i * 16));
                    }
                }
            }
        }
    }
}

/// 16 floats as four vectors.
#[target_feature(enable = "neon")]
unsafe fn load16(p: *const f32) -> [float32x4_t; 4] {
    // SAFETY: the caller passes 16 readable floats.
    unsafe {
        [
            vld1q_f32(p),
            vld1q_f32(p.add(4)),
            vld1q_f32(p.add(8)),
            vld1q_f32(p.add(12)),
        ]
    }
}

/// 16 int8 values times their scales.
#[target_feature(enable = "neon")]
unsafe fn widen_i8(src: *const u8, s: &[float32x4_t; 4], out: *mut f32) {
    // SAFETY: the caller passes 16 readable bytes and 16 writable floats.
    unsafe {
        let q = vld1q_s8(src as *const i8);
        let (lo, hi) = (vmovl_s8(vget_low_s8(q)), vmovl_high_s8(q));
        let ints = [
            vmovl_s16(vget_low_s16(lo)),
            vmovl_high_s16(lo),
            vmovl_s16(vget_low_s16(hi)),
            vmovl_high_s16(hi),
        ];
        for (j, v) in ints.into_iter().enumerate() {
            vst1q_f32(out.add(j * 4), vmulq_f32(vcvtq_f32_s32(v), s[j]));
        }
    }
}

/// e^x for 4 lanes, the same steps as the AVX2 kernels' exp, so within about 2 ulp of the
/// true value. Inputs are clamped to [-87.3, 88.4], so very negative inputs give about 1e-38
/// instead of 0.
#[target_feature(enable = "neon")]
fn exp4(x: float32x4_t) -> float32x4_t {
    let x = vminnmq_f32(
        vmaxnmq_f32(x, vdupq_n_f32(-87.336_54)),
        vdupq_n_f32(88.376_26),
    );
    let n = vrndnq_f32(vmulq_n_f32(x, std::f32::consts::LOG2_E));
    // r = x - n ln 2, with ln 2 split in two for precision.
    let r = vfmsq_f32(x, n, vdupq_n_f32(0.693_359_4));
    let r = vfmsq_f32(r, n, vdupq_n_f32(-2.121_944_4e-4));
    let mut p = vdupq_n_f32(1.987_569_1e-4);
    p = vfmaq_f32(vdupq_n_f32(1.398_2e-3), p, r);
    p = vfmaq_f32(vdupq_n_f32(8.333_452e-3), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.166_579_6e-2), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.166_666_65), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5), p, r);
    p = vfmaq_f32(r, p, vmulq_f32(r, r));
    p = vaddq_f32(p, vdupq_n_f32(1.0));
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(
        vcvtq_s32_f32(n),
        vdupq_n_s32(127),
    )));
    vmulq_f32(p, pow2n)
}

/// e^x of one value, through the vector code so it agrees with the other lanes.
#[target_feature(enable = "neon")]
fn exp1(x: f32) -> f32 {
    vgetq_lane_f32::<0>(exp4(vdupq_n_f32(x)))
}

/// x * sigmoid(x) = x / (1 + e^-x).
#[target_feature(enable = "neon")]
fn silu4(x: float32x4_t) -> float32x4_t {
    vdivq_f32(x, vaddq_f32(vdupq_n_f32(1.0), exp4(vnegq_f32(x))))
}

#[target_feature(enable = "neon")]
unsafe fn store(
    acc: &[f32],
    c: &mut [f32],
    bias: Option<&[f32]>,
    act: Act,
    scale: f32,
    accumulate: bool,
) {
    let cols = c.len();
    if cols < PANEL {
        // A partial panel at the right edge goes through full-width copies.
        let (mut a16, mut b16, mut c16) = ([0f32; PANEL], [0f32; PANEL], [0f32; PANEL]);
        a16[..cols].copy_from_slice(&acc[..cols]);
        if let Some(b) = bias {
            b16[..cols].copy_from_slice(&b[..cols]);
        }
        c16[..cols].copy_from_slice(c);
        let bias = bias.map(|_| &b16[..]);
        unsafe { store(&a16, &mut c16, bias, act, scale, accumulate) };
        c.copy_from_slice(&c16[..cols]);
        return;
    }
    assert!(acc.len() >= PANEL && bias.is_none_or(|b| b.len() >= PANEL));
    let vscale = vdupq_n_f32(scale);
    for o in (0..PANEL).step_by(4) {
        // SAFETY: `acc`, `bias` and `c` hold 16 values.
        unsafe {
            let mut v = vld1q_f32(acc.as_ptr().add(o));
            if let Some(b) = bias {
                v = vaddq_f32(v, vld1q_f32(b.as_ptr().add(o)));
            }
            v = match act {
                Act::None => v,
                Act::Relu => vmaxnmq_f32(v, vdupq_n_f32(0.0)),
                Act::Silu => silu4(v),
            };
            v = vmulq_f32(v, vscale);
            if accumulate {
                v = vaddq_f32(vld1q_f32(c.as_ptr().add(o)), v);
            }
            vst1q_f32(c.as_mut_ptr().add(o), v);
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn layer_norm(x: &[f32], out: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    let n = x.len();
    assert!(out.len() >= n && gamma.len() >= n && beta.len() >= n);
    let body = n / 8 * 8;
    let xp = x.as_ptr();
    // Two accumulators, so the additions don't wait on each other.
    let (mut s0, mut s1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    for i in (0..body).step_by(8) {
        // SAFETY: `i + 8 <= n`.
        unsafe {
            s0 = vaddq_f32(s0, vld1q_f32(xp.add(i)));
            s1 = vaddq_f32(s1, vld1q_f32(xp.add(i + 4)));
        }
    }
    let mut sum = vaddvq_f32(vaddq_f32(s0, s1));
    for &v in &x[body..] {
        sum += v;
    }
    let mean = sum / n as f32;
    let vmean = vdupq_n_f32(mean);
    let (mut q0, mut q1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    for i in (0..body).step_by(8) {
        unsafe {
            let d0 = vsubq_f32(vld1q_f32(xp.add(i)), vmean);
            let d1 = vsubq_f32(vld1q_f32(xp.add(i + 4)), vmean);
            q0 = vfmaq_f32(q0, d0, d0);
            q1 = vfmaq_f32(q1, d1, d1);
        }
    }
    let mut sq = vaddvq_f32(vaddq_f32(q0, q1));
    for &v in &x[body..] {
        sq += (v - mean) * (v - mean);
    }
    let rstd = 1.0 / (sq / n as f32 + eps).sqrt();
    let vrstd = vdupq_n_f32(rstd);
    for i in (0..n / 4 * 4).step_by(4) {
        // SAFETY: `i + 4 <= n`, and `out`, `gamma` and `beta` hold `n` values.
        unsafe {
            let d = vmulq_f32(vsubq_f32(vld1q_f32(xp.add(i)), vmean), vrstd);
            let y = vfmaq_f32(
                vld1q_f32(beta.as_ptr().add(i)),
                d,
                vld1q_f32(gamma.as_ptr().add(i)),
            );
            vst1q_f32(out.as_mut_ptr().add(i), y);
        }
    }
    for i in n / 4 * 4..n {
        out[i] = (x[i] - mean) * rstd * gamma[i] + beta[i];
    }
}

#[target_feature(enable = "neon")]
unsafe fn softmax(x: &mut [f32]) {
    let n = x.len();
    let body = n / 8 * 8;
    let low = vdupq_n_f32(f32::NEG_INFINITY);
    let (mut m0, mut m1) = (low, low);
    for i in (0..body).step_by(8) {
        // SAFETY: `i + 8 <= n`.
        unsafe {
            m0 = vmaxnmq_f32(m0, vld1q_f32(x.as_ptr().add(i)));
            m1 = vmaxnmq_f32(m1, vld1q_f32(x.as_ptr().add(i + 4)));
        }
    }
    let mut max = vmaxnmvq_f32(vmaxnmq_f32(m0, m1));
    for &v in &x[body..] {
        max = max.max(v);
    }
    let vmax = vdupq_n_f32(max);
    let (mut s0, mut s1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    for i in (0..body).step_by(8) {
        unsafe {
            let p = x.as_mut_ptr().add(i);
            let e0 = exp4(vsubq_f32(vld1q_f32(p), vmax));
            let e1 = exp4(vsubq_f32(vld1q_f32(p.add(4)), vmax));
            vst1q_f32(p, e0);
            vst1q_f32(p.add(4), e1);
            s0 = vaddq_f32(s0, e0);
            s1 = vaddq_f32(s1, e1);
        }
    }
    let mut sum = vaddvq_f32(vaddq_f32(s0, s1));
    for v in &mut x[body..] {
        *v = exp1(*v - max);
        sum += *v;
    }
    let inv = 1.0 / sum;
    for i in (0..n / 4 * 4).step_by(4) {
        unsafe {
            let p = x.as_mut_ptr().add(i);
            vst1q_f32(p, vmulq_n_f32(vld1q_f32(p), inv));
        }
    }
    for v in &mut x[n / 4 * 4..] {
        *v *= inv;
    }
}

#[target_feature(enable = "neon")]
unsafe fn glu(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = out.len();
    assert!(a.len() >= n && b.len() >= n);
    let one = vdupq_n_f32(1.0);
    let body = n / 4 * 4;
    for i in (0..body).step_by(4) {
        // SAFETY: `i + 4 <= n`, and `a` and `b` hold `n` values.
        unsafe {
            let e = exp4(vnegq_f32(vld1q_f32(b.as_ptr().add(i))));
            let y = vdivq_f32(vld1q_f32(a.as_ptr().add(i)), vaddq_f32(one, e));
            vst1q_f32(out.as_mut_ptr().add(i), y);
        }
    }
    for j in body..n {
        out[j] = a[j] / (1.0 + exp1(-b[j]));
    }
}

#[target_feature(enable = "neon")]
unsafe fn silu(x: &mut [f32]) {
    let n = x.len();
    let body = n / 4 * 4;
    for i in (0..body).step_by(4) {
        // SAFETY: `i + 4 <= n`.
        unsafe {
            let p = x.as_mut_ptr().add(i);
            vst1q_f32(p, silu4(vld1q_f32(p)));
        }
    }
    for v in &mut x[body..] {
        *v = vgetq_lane_f32::<0>(silu4(vdupq_n_f32(*v)));
    }
}

/// The scalar loop, compiled with NEON so it vectorizes over channels. Each output sums its
/// taps in the same order, so the results equal the scalar kernel's.
#[target_feature(enable = "neon")]
unsafe fn dwconv1d_frame(
    input: &[f32],
    channels: usize,
    frames: usize,
    t: usize,
    w: &[f32],
    kernel: usize,
    bias: &[f32],
    out: &mut [f32],
) {
    scalar::dwconv1d_frame(input, channels, frames, t, w, kernel, bias, out)
}

/// The scalar loop, compiled with NEON so it vectorizes over channels, with the same results.
#[target_feature(enable = "neon")]
unsafe fn conv3x3s2_row(
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
    scalar::conv3x3s2_row(input, in_t, in_f, channels, depthwise, w, bias, t, out)
}
