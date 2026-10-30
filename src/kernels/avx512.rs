//! Kernels for x86-64 CPUs with AVX-512F, FMA and F16C.
//!
//! Every function here is compiled with those features enabled. The table in [`kernels`] is
//! only built after detection confirms them, which makes the safe wrappers sound.
//!
//! These kernels compile, and the tests compare them with the scalar kernels on any CPU with
//! AVX-512, but they haven't run on real AVX-512 hardware yet.

use std::arch::x86_64::*;

use super::{Act, Isa, Kernels, PANEL, scalar};
use crate::quant::{Precision, Q8_BLOCK};

/// Rows of the register tile. A row of the 16-wide panel is one register, so 12 rows use 12
/// accumulators of the 32 registers, and 12 divides the 48 rows of an attention block.
const TILE_ROWS: usize = 12;

pub(crate) fn kernels() -> Kernels {
    assert!(Isa::Avx512.is_supported());
    Kernels {
        isa: Isa::Avx512,
        tile_rows: TILE_ROWS,
        tile,
        widen,
        tile_narrow: None,
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
/// As for the scalar tile, and the CPU must have AVX-512F.
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
            12 => tile_rows::<12>(k, a, lda, b, k_stride, out, load),
            11 => tile_rows::<11>(k, a, lda, b, k_stride, out, load),
            10 => tile_rows::<10>(k, a, lda, b, k_stride, out, load),
            9 => tile_rows::<9>(k, a, lda, b, k_stride, out, load),
            8 => tile_rows::<8>(k, a, lda, b, k_stride, out, load),
            7 => tile_rows::<7>(k, a, lda, b, k_stride, out, load),
            6 => tile_rows::<6>(k, a, lda, b, k_stride, out, load),
            5 => tile_rows::<5>(k, a, lda, b, k_stride, out, load),
            4 => tile_rows::<4>(k, a, lda, b, k_stride, out, load),
            3 => tile_rows::<3>(k, a, lda, b, k_stride, out, load),
            2 => tile_rows::<2>(k, a, lda, b, k_stride, out, load),
            1 => tile_rows::<1>(k, a, lda, b, k_stride, out, load),
            _ => unreachable!("tiles have 1 to 12 rows"),
        }
    }
}

/// An `R x 16` register tile. For each input, one load of the panel and one broadcast per row
/// feed `R` fused multiply-adds.
#[target_feature(enable = "avx512f,fma,f16c")]
unsafe fn tile_rows<const R: usize>(
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    k_stride: usize,
    out: *mut f32,
    load: bool,
) {
    let mut acc = [_mm512_setzero_ps(); R];
    if load {
        for (r, acc) in acc.iter_mut().enumerate() {
            // SAFETY: `out` holds R rows of 16.
            *acc = unsafe { _mm512_loadu_ps(out.add(r * PANEL)) };
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
            let bv = _mm512_loadu_ps(bp);
            for r in 0..R {
                acc[r] = _mm512_fmadd_ps(_mm512_set1_ps(*rows[r].add(i)), bv, acc[r]);
            }
            bp = bp.add(k_stride);
        }
    }
    for (r, acc) in acc.iter().enumerate() {
        unsafe { _mm512_storeu_ps(out.add(r * PANEL), *acc) };
    }
}

/// Widens narrow weights to f32, 16 values per input. The conversions are exact (and the int8
/// products correctly rounded), so the values equal the scalar kernel's.
///
/// # Safety
/// As for the scalar `widen`, and the CPU must have AVX-512F.
#[target_feature(enable = "avx512f,fma,f16c")]
unsafe fn widen(precision: Precision, k: usize, src: *const u8, scales: *const f32, out: *mut f32) {
    // SAFETY: every pointer stays inside the ranges the caller guarantees.
    unsafe {
        match precision {
            Precision::F32 => scalar::widen(precision, k, src, scales, out),
            Precision::F16 => {
                for i in 0..k {
                    let h = _mm256_loadu_si256(src.add(i * 32) as *const __m256i);
                    _mm512_storeu_ps(out.add(i * 16), _mm512_cvtph_ps(h));
                }
            }
            Precision::Bf16 => {
                for i in 0..k {
                    let h = _mm256_loadu_si256(src.add(i * 32) as *const __m256i);
                    let bits = _mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(h));
                    _mm512_storeu_ps(out.add(i * 16), _mm512_castsi512_ps(bits));
                }
            }
            Precision::Int8 => {
                let s = _mm512_loadu_ps(scales);
                for i in 0..k {
                    widen_i8(src.add(i * 16), s, out.add(i * 16));
                }
            }
            Precision::Q8_0 => {
                for b in 0..k.div_ceil(Q8_BLOCK) {
                    let s = _mm512_loadu_ps(scales.add(b * PANEL));
                    for i in b * Q8_BLOCK..((b + 1) * Q8_BLOCK).min(k) {
                        widen_i8(src.add(i * 16), s, out.add(i * 16));
                    }
                }
            }
        }
    }
}

/// 16 int8 values times their scales.
#[target_feature(enable = "avx512f,fma,f16c")]
unsafe fn widen_i8(src: *const u8, s: __m512, out: *mut f32) {
    // SAFETY: the caller passes 16 readable bytes and 16 writable floats.
    unsafe {
        let q = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(src as *const __m128i)));
        _mm512_storeu_ps(out, _mm512_mul_ps(q, s));
    }
}

/// The lanes of a 16-wide step that fall inside `len` values.
fn lanes(len: usize) -> __mmask16 {
    if len >= 16 {
        0xffff
    } else {
        ((1u32 << len) - 1) as __mmask16
    }
}

/// e^x for 16 lanes, the same steps as the AVX2 kernels' exp, so within about 2 ulp of the
/// true value. Inputs are clamped to [-87.3, 88.4], so very negative inputs give about 1e-38
/// instead of 0.
#[target_feature(enable = "avx512f,fma,f16c")]
fn exp512(x: __m512) -> __m512 {
    let x = _mm512_min_ps(
        _mm512_max_ps(x, _mm512_set1_ps(-87.336_54)),
        _mm512_set1_ps(88.376_26),
    );
    let n = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
        _mm512_mul_ps(x, _mm512_set1_ps(std::f32::consts::LOG2_E)),
    );
    // r = x - n ln 2, with ln 2 split in two for precision.
    let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693_359_4), x);
    let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.121_944_4e-4), r);
    let mut p = _mm512_set1_ps(1.987_569_1e-4);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.398_2e-3));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.333_452e-3));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.166_579_6e-2));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.166_666_65));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5));
    p = _mm512_fmadd_ps(p, _mm512_mul_ps(r, r), r);
    p = _mm512_add_ps(p, _mm512_set1_ps(1.0));
    let pow2n = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(
        _mm512_cvtps_epi32(n),
        _mm512_set1_epi32(127),
    )));
    _mm512_mul_ps(p, pow2n)
}

/// x * sigmoid(x) = x / (1 + e^-x).
#[target_feature(enable = "avx512f,fma,f16c")]
fn silu512(x: __m512) -> __m512 {
    let e = exp512(_mm512_sub_ps(_mm512_setzero_ps(), x));
    _mm512_div_ps(x, _mm512_add_ps(_mm512_set1_ps(1.0), e))
}

#[target_feature(enable = "avx512f,fma,f16c")]
unsafe fn store(
    acc: &[f32],
    c: &mut [f32],
    bias: Option<&[f32]>,
    act: Act,
    scale: f32,
    accumulate: bool,
) {
    let cols = c.len();
    assert!(acc.len() >= cols && bias.is_none_or(|b| b.len() >= cols));
    // A partial panel at the right edge loads and stores only its own columns.
    let m = lanes(cols);
    // SAFETY: the masked loads and the store touch the first `cols` values alone.
    unsafe {
        let mut v = _mm512_maskz_loadu_ps(m, acc.as_ptr());
        if let Some(b) = bias {
            v = _mm512_add_ps(v, _mm512_maskz_loadu_ps(m, b.as_ptr()));
        }
        v = match act {
            Act::None => v,
            Act::Relu => _mm512_max_ps(v, _mm512_setzero_ps()),
            Act::Silu => silu512(v),
        };
        v = _mm512_mul_ps(v, _mm512_set1_ps(scale));
        if accumulate {
            v = _mm512_add_ps(_mm512_maskz_loadu_ps(m, c.as_ptr()), v);
        }
        _mm512_mask_storeu_ps(c.as_mut_ptr(), m, v);
    }
}

#[target_feature(enable = "avx512f,fma,f16c")]
unsafe fn layer_norm(x: &[f32], out: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    let n = x.len();
    assert!(out.len() >= n && gamma.len() >= n && beta.len() >= n);
    // SAFETY: every masked load and store stays inside the first `n` values.
    unsafe {
        let mut s = _mm512_setzero_ps();
        for i in (0..n).step_by(16) {
            s = _mm512_add_ps(s, _mm512_maskz_loadu_ps(lanes(n - i), x.as_ptr().add(i)));
        }
        let mean = _mm512_reduce_add_ps(s) / n as f32;
        let vmean = _mm512_set1_ps(mean);
        let mut q = _mm512_setzero_ps();
        for i in (0..n).step_by(16) {
            let m = lanes(n - i);
            let d = _mm512_maskz_sub_ps(m, _mm512_maskz_loadu_ps(m, x.as_ptr().add(i)), vmean);
            q = _mm512_fmadd_ps(d, d, q);
        }
        let rstd = 1.0 / (_mm512_reduce_add_ps(q) / n as f32 + eps).sqrt();
        let vrstd = _mm512_set1_ps(rstd);
        for i in (0..n).step_by(16) {
            let m = lanes(n - i);
            let d = _mm512_mul_ps(
                _mm512_sub_ps(_mm512_maskz_loadu_ps(m, x.as_ptr().add(i)), vmean),
                vrstd,
            );
            let y = _mm512_fmadd_ps(
                d,
                _mm512_maskz_loadu_ps(m, gamma.as_ptr().add(i)),
                _mm512_maskz_loadu_ps(m, beta.as_ptr().add(i)),
            );
            _mm512_mask_storeu_ps(out.as_mut_ptr().add(i), m, y);
        }
    }
}

#[target_feature(enable = "avx512f,fma,f16c")]
unsafe fn softmax(x: &mut [f32]) {
    let n = x.len();
    let p = x.as_mut_ptr();
    // SAFETY: every masked load and store stays inside the first `n` values.
    unsafe {
        let low = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut mx = low;
        for i in (0..n).step_by(16) {
            mx = _mm512_max_ps(mx, _mm512_mask_loadu_ps(low, lanes(n - i), p.add(i)));
        }
        let vmax = _mm512_set1_ps(_mm512_reduce_max_ps(mx));
        let mut s = _mm512_setzero_ps();
        for i in (0..n).step_by(16) {
            let m = lanes(n - i);
            let e = exp512(_mm512_sub_ps(_mm512_maskz_loadu_ps(m, p.add(i)), vmax));
            _mm512_mask_storeu_ps(p.add(i), m, e);
            s = _mm512_mask_add_ps(s, m, s, e);
        }
        let inv = _mm512_set1_ps(1.0 / _mm512_reduce_add_ps(s));
        for i in (0..n).step_by(16) {
            let m = lanes(n - i);
            let y = _mm512_mul_ps(_mm512_maskz_loadu_ps(m, p.add(i)), inv);
            _mm512_mask_storeu_ps(p.add(i), m, y);
        }
    }
}

#[target_feature(enable = "avx512f,fma,f16c")]
unsafe fn glu(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = out.len();
    assert!(a.len() >= n && b.len() >= n);
    let one = _mm512_set1_ps(1.0);
    for i in (0..n).step_by(16) {
        let m = lanes(n - i);
        // SAFETY: the masked loads and the store stay inside the first `n` values.
        unsafe {
            let e = exp512(_mm512_sub_ps(
                _mm512_setzero_ps(),
                _mm512_maskz_loadu_ps(m, b.as_ptr().add(i)),
            ));
            let y = _mm512_div_ps(
                _mm512_maskz_loadu_ps(m, a.as_ptr().add(i)),
                _mm512_add_ps(one, e),
            );
            _mm512_mask_storeu_ps(out.as_mut_ptr().add(i), m, y);
        }
    }
}

#[target_feature(enable = "avx512f,fma,f16c")]
unsafe fn silu(x: &mut [f32]) {
    let n = x.len();
    for i in (0..n).step_by(16) {
        let m = lanes(n - i);
        // SAFETY: the masked load and store stay inside the first `n` values.
        unsafe {
            let p = x.as_mut_ptr().add(i);
            _mm512_mask_storeu_ps(p, m, silu512(_mm512_maskz_loadu_ps(m, p)));
        }
    }
}

/// The scalar loop, compiled with AVX-512 so it vectorizes over channels. Each output sums its
/// taps in the same order, so the results equal the scalar kernel's.
#[target_feature(enable = "avx512f,fma,f16c")]
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

/// The scalar loop, compiled with AVX-512 so it vectorizes over channels, with the same results.
#[target_feature(enable = "avx512f,fma,f16c")]
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
