//! Kernels for x86-64 CPUs with AVX2 and FMA.
//!
//! Every function here is compiled with those features enabled. The table in [`kernels`] is
//! only built after detection confirms them, which makes the safe wrappers sound.

use std::arch::x86_64::*;

use super::{Act, Isa, Kernels, PANEL, TILE_ROWS, scalar};
use crate::quant::{Precision, Q8_BLOCK};

pub(crate) fn kernels() -> Kernels {
    assert!(Isa::Avx2.is_supported());
    Kernels {
        isa: Isa::Avx2,
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
/// As for the scalar tile, and the CPU must have AVX2 and FMA.
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

/// An `R x 16` register tile. For each input, two 8-wide loads of the panel and one broadcast
/// per row feed `2R` fused multiply-adds.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_rows<const R: usize>(
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    k_stride: usize,
    out: *mut f32,
    load: bool,
) {
    let mut acc0 = [_mm256_setzero_ps(); R];
    let mut acc1 = [_mm256_setzero_ps(); R];
    if load {
        for r in 0..R {
            // SAFETY: `out` holds R rows of 16.
            unsafe {
                acc0[r] = _mm256_loadu_ps(out.add(r * PANEL));
                acc1[r] = _mm256_loadu_ps(out.add(r * PANEL + 8));
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
            let b0 = _mm256_loadu_ps(bp);
            let b1 = _mm256_loadu_ps(bp.add(8));
            for r in 0..R {
                let av = _mm256_broadcast_ss(&*rows[r].add(i));
                acc0[r] = _mm256_fmadd_ps(av, b0, acc0[r]);
                acc1[r] = _mm256_fmadd_ps(av, b1, acc1[r]);
            }
            bp = bp.add(k_stride);
        }
    }
    for r in 0..R {
        unsafe {
            _mm256_storeu_ps(out.add(r * PANEL), acc0[r]);
            _mm256_storeu_ps(out.add(r * PANEL + 8), acc1[r]);
        }
    }
}

/// # Safety
/// As for `tile`, with `b` holding `k` runs of 16 narrow values and `scales` as `widen` takes
/// them, and the CPU must have AVX2, FMA and F16C.
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
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_narrow_rows<const R: usize, const P: u8>(
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const u8,
    scales: *const f32,
    out: *mut f32,
    load: bool,
) {
    let mut acc0 = [_mm256_setzero_ps(); R];
    let mut acc1 = [_mm256_setzero_ps(); R];
    // SAFETY: the caller guarantees `out` holds R rows of 16, `a` R rows of `k` values, `b`
    // `k` runs of 16 values, and the scales.
    unsafe {
        if load {
            for r in 0..R {
                acc0[r] = _mm256_loadu_ps(out.add(r * PANEL));
                acc1[r] = _mm256_loadu_ps(out.add(r * PANEL + 8));
            }
        }
        let mut rows = [a; R];
        for (r, row) in rows.iter_mut().enumerate() {
            *row = a.add(r * lda);
        }
        let step = if P == F16 || P == BF16 { 32 } else { 16 };
        let (mut s0, mut s1) = (_mm256_set1_ps(1.0), _mm256_set1_ps(1.0));
        if P == INT8 {
            s0 = _mm256_loadu_ps(scales);
            s1 = _mm256_loadu_ps(scales.add(8));
        }
        let mut i = 0;
        while i < k {
            let end = if P == Q8_0 {
                let s = scales.add(i / Q8_BLOCK * PANEL);
                s0 = _mm256_loadu_ps(s);
                s1 = _mm256_loadu_ps(s.add(8));
                (i + Q8_BLOCK).min(k)
            } else {
                k
            };
            while i < end {
                let p = b.add(i * step);
                let (w0, w1) = if P == F16 {
                    let p = p as *const __m128i;
                    (
                        _mm256_cvtph_ps(_mm_loadu_si128(p)),
                        _mm256_cvtph_ps(_mm_loadu_si128(p.add(1))),
                    )
                } else if P == BF16 {
                    let p = p as *const __m128i;
                    (
                        _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(
                            _mm_loadu_si128(p),
                        ))),
                        _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(
                            _mm_loadu_si128(p.add(1)),
                        ))),
                    )
                } else {
                    let q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                        p as *const __m128i,
                    )));
                    let q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                        p.add(8) as *const __m128i
                    )));
                    (_mm256_mul_ps(q0, s0), _mm256_mul_ps(q1, s1))
                };
                for r in 0..R {
                    let av = _mm256_broadcast_ss(&*rows[r].add(i));
                    acc0[r] = _mm256_fmadd_ps(av, w0, acc0[r]);
                    acc1[r] = _mm256_fmadd_ps(av, w1, acc1[r]);
                }
                i += 1;
            }
        }
        for r in 0..R {
            _mm256_storeu_ps(out.add(r * PANEL), acc0[r]);
            _mm256_storeu_ps(out.add(r * PANEL + 8), acc1[r]);
        }
    }
}

/// Widens narrow weights to f32, 16 values per input. The conversions are exact (and the int8
/// products correctly rounded), so the values equal the scalar kernel's.
///
/// # Safety
/// As for the scalar `widen`, and the CPU must have AVX2, FMA and F16C.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn widen(precision: Precision, k: usize, src: *const u8, scales: *const f32, out: *mut f32) {
    // SAFETY: every pointer stays inside the ranges the caller guarantees.
    unsafe {
        match precision {
            Precision::F32 => scalar::widen(precision, k, src, scales, out),
            Precision::F16 => {
                for i in 0..k {
                    let p = src.add(i * 32) as *const __m128i;
                    _mm256_storeu_ps(out.add(i * 16), _mm256_cvtph_ps(_mm_loadu_si128(p)));
                    _mm256_storeu_ps(
                        out.add(i * 16 + 8),
                        _mm256_cvtph_ps(_mm_loadu_si128(p.add(1))),
                    );
                }
            }
            Precision::Bf16 => {
                for i in 0..k {
                    let p = src.add(i * 32) as *const __m128i;
                    let lo = _mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(_mm_loadu_si128(p)));
                    let hi =
                        _mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(_mm_loadu_si128(p.add(1))));
                    _mm256_storeu_ps(out.add(i * 16), _mm256_castsi256_ps(lo));
                    _mm256_storeu_ps(out.add(i * 16 + 8), _mm256_castsi256_ps(hi));
                }
            }
            Precision::Int8 => {
                let (s0, s1) = (_mm256_loadu_ps(scales), _mm256_loadu_ps(scales.add(8)));
                for i in 0..k {
                    widen_i8(src.add(i * 16), s0, s1, out.add(i * 16));
                }
            }
            Precision::Q8_0 => {
                for b in 0..k.div_ceil(Q8_BLOCK) {
                    let s = scales.add(b * PANEL);
                    let (s0, s1) = (_mm256_loadu_ps(s), _mm256_loadu_ps(s.add(8)));
                    for i in b * Q8_BLOCK..((b + 1) * Q8_BLOCK).min(k) {
                        widen_i8(src.add(i * 16), s0, s1, out.add(i * 16));
                    }
                }
            }
        }
    }
}

/// 16 int8 values times their scales.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn widen_i8(src: *const u8, s0: __m256, s1: __m256, out: *mut f32) {
    // SAFETY: the caller passes 16 readable bytes and 16 writable floats.
    unsafe {
        let lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(src as *const __m128i)));
        let hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
            src.add(8) as *const __m128i
        )));
        _mm256_storeu_ps(out, _mm256_mul_ps(lo, s0));
        _mm256_storeu_ps(out.add(8), _mm256_mul_ps(hi, s1));
    }
}

/// e^x for 8 lanes, within about 2 ulp of the true value. Inputs are clamped to
/// [-87.3, 88.4], so very negative inputs give about 1e-38 instead of 0.
#[target_feature(enable = "avx2,fma,f16c")]
fn exp256(x: __m256) -> __m256 {
    let x = _mm256_min_ps(
        _mm256_max_ps(x, _mm256_set1_ps(-87.336_54)),
        _mm256_set1_ps(88.376_26),
    );
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        x,
        _mm256_set1_ps(std::f32::consts::LOG2_E),
    ));
    // r = x - n ln 2, with ln 2 split in two for precision.
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693_359_4), x);
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.121_944_4e-4), r);
    let mut p = _mm256_set1_ps(1.987_569_1e-4);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.398_2e-3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.333_452e-3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.166_579_6e-2));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.166_666_65));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5));
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), r);
    p = _mm256_add_ps(p, _mm256_set1_ps(1.0));
    let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(
        _mm256_cvtps_epi32(n),
        _mm256_set1_epi32(127),
    )));
    _mm256_mul_ps(p, pow2n)
}

/// x * sigmoid(x) = x / (1 + e^-x).
#[target_feature(enable = "avx2,fma,f16c")]
fn silu256(x: __m256) -> __m256 {
    let e = exp256(_mm256_sub_ps(_mm256_setzero_ps(), x));
    _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0), e))
}

#[target_feature(enable = "avx2,fma,f16c")]
fn hsum(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps::<1>(s, s));
    _mm_cvtss_f32(s)
}

#[target_feature(enable = "avx2,fma,f16c")]
fn hmax(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let s = _mm_max_ps(lo, hi);
    let s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_max_ss(s, _mm_shuffle_ps::<1>(s, s));
    _mm_cvtss_f32(s)
}

#[target_feature(enable = "avx2,fma,f16c")]
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
        // A partial panel at the right edge. No need for masked loads.
        for (j, out) in c.iter_mut().enumerate() {
            let v = acc[j] + bias.map_or(0.0, |b| b[j]);
            let v = match act {
                Act::Silu => {
                    let mut lane = [0f32; 8];
                    unsafe { _mm256_storeu_ps(lane.as_mut_ptr(), silu256(_mm256_set1_ps(v))) };
                    lane[0]
                }
                other => scalar::activate(v, other),
            } * scale;
            *out = if accumulate { *out + v } else { v };
        }
        return;
    }
    let vscale = _mm256_set1_ps(scale);
    for half in 0..2 {
        let o = half * 8;
        unsafe {
            let mut v = _mm256_loadu_ps(acc.as_ptr().add(o));
            if let Some(b) = bias {
                v = _mm256_add_ps(v, _mm256_loadu_ps(b.as_ptr().add(o)));
            }
            v = match act {
                Act::None => v,
                Act::Relu => _mm256_max_ps(v, _mm256_setzero_ps()),
                Act::Silu => silu256(v),
            };
            v = _mm256_mul_ps(v, vscale);
            if accumulate {
                v = _mm256_add_ps(_mm256_loadu_ps(c.as_ptr().add(o)), v);
            }
            _mm256_storeu_ps(c.as_mut_ptr().add(o), v);
        }
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn layer_norm(x: &[f32], out: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    let n = x.len();
    let body = n / 8 * 8;
    let mut s = _mm256_setzero_ps();
    for i in (0..body).step_by(8) {
        s = _mm256_add_ps(s, unsafe { _mm256_loadu_ps(x.as_ptr().add(i)) });
    }
    let mut sum = hsum(s);
    for &v in &x[body..] {
        sum += v;
    }
    let mean = sum / n as f32;
    let vmean = _mm256_set1_ps(mean);
    let mut q = _mm256_setzero_ps();
    for i in (0..body).step_by(8) {
        let d = _mm256_sub_ps(unsafe { _mm256_loadu_ps(x.as_ptr().add(i)) }, vmean);
        q = _mm256_fmadd_ps(d, d, q);
    }
    let mut sq = hsum(q);
    for &v in &x[body..] {
        sq += (v - mean) * (v - mean);
    }
    let rstd = 1.0 / (sq / n as f32 + eps).sqrt();
    let vrstd = _mm256_set1_ps(rstd);
    for i in (0..body).step_by(8) {
        unsafe {
            let d = _mm256_mul_ps(
                _mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vmean),
                vrstd,
            );
            let y = _mm256_fmadd_ps(
                d,
                _mm256_loadu_ps(gamma.as_ptr().add(i)),
                _mm256_loadu_ps(beta.as_ptr().add(i)),
            );
            _mm256_storeu_ps(out.as_mut_ptr().add(i), y);
        }
    }
    for i in body..n {
        out[i] = (x[i] - mean) * rstd * gamma[i] + beta[i];
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn softmax(x: &mut [f32]) {
    let n = x.len();
    let body = n / 8 * 8;
    let mut m = _mm256_set1_ps(f32::NEG_INFINITY);
    for i in (0..body).step_by(8) {
        m = _mm256_max_ps(m, unsafe { _mm256_loadu_ps(x.as_ptr().add(i)) });
    }
    let mut max = hmax(m);
    for &v in &x[body..] {
        max = max.max(v);
    }
    let vmax = _mm256_set1_ps(max);
    let mut s = _mm256_setzero_ps();
    for i in (0..body).step_by(8) {
        unsafe {
            let e = exp256(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vmax));
            _mm256_storeu_ps(x.as_mut_ptr().add(i), e);
            s = _mm256_add_ps(s, e);
        }
    }
    let mut sum = hsum(s);
    for v in &mut x[body..] {
        let mut lane = [0f32; 8];
        unsafe { _mm256_storeu_ps(lane.as_mut_ptr(), exp256(_mm256_set1_ps(*v - max))) };
        *v = lane[0];
        sum += *v;
    }
    let inv = _mm256_set1_ps(1.0 / sum);
    for i in (0..body).step_by(8) {
        unsafe {
            _mm256_storeu_ps(
                x.as_mut_ptr().add(i),
                _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), inv),
            )
        };
    }
    for v in &mut x[body..] {
        *v /= sum;
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn glu(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = out.len();
    assert!(a.len() >= n && b.len() >= n);
    let one = _mm256_set1_ps(1.0);
    let mut i = 0;
    while i + 8 <= n {
        unsafe {
            let e = exp256(_mm256_sub_ps(
                _mm256_setzero_ps(),
                _mm256_loadu_ps(b.as_ptr().add(i)),
            ));
            let y = _mm256_div_ps(_mm256_loadu_ps(a.as_ptr().add(i)), _mm256_add_ps(one, e));
            _mm256_storeu_ps(out.as_mut_ptr().add(i), y);
        }
        i += 8;
    }
    for j in i..n {
        let mut lane = [0f32; 8];
        unsafe { _mm256_storeu_ps(lane.as_mut_ptr(), exp256(_mm256_set1_ps(-b[j]))) };
        out[j] = a[j] / (1.0 + lane[0]);
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn silu(x: &mut [f32]) {
    let n = x.len();
    let mut i = 0;
    while i + 8 <= n {
        unsafe {
            _mm256_storeu_ps(
                x.as_mut_ptr().add(i),
                silu256(_mm256_loadu_ps(x.as_ptr().add(i))),
            )
        };
        i += 8;
    }
    for v in &mut x[i..] {
        let mut lane = [0f32; 8];
        unsafe { _mm256_storeu_ps(lane.as_mut_ptr(), silu256(_mm256_set1_ps(*v))) };
        *v = lane[0];
    }
}

/// The scalar loop, compiled with AVX2 so it vectorizes over channels. Each output sums its
/// taps in the same order, so the results equal the scalar kernel's.
#[target_feature(enable = "avx2,fma,f16c")]
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

/// The scalar loop, compiled with AVX2 so it vectorizes over channels, with the same results.
#[target_feature(enable = "avx2,fma,f16c")]
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
