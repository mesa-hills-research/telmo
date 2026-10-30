//! Weight precisions: f16 and bf16 conversion, 8-bit quantization, and the panel layout the
//! matrix product reads.
//!
//! A weight matrix of `n` outputs and `k` inputs is stored as panels of 16 outputs, with the
//! inputs running inside each panel, so panel `p` holds `k` runs of 16 values. Outputs past `n`
//! in the last panel are zero. The values are f32, f16, bf16 or int8. Int8 comes in two kinds.
//!
//! - [`Precision::Int8`] has one f32 scale per output channel, `w = q * scale`.
//! - [`Precision::Q8_0`] has one scale per block of 32 inputs of each output, as in GGUF's
//!   `Q8_0` type. Its scales come from f16 values, so `q * scale` is exact in f32.
//!
//! Scales are stored apart from the values, as f32. Int8 keeps 16 per panel. Q8_0 keeps 16 per
//! block of 32 inputs per panel, in panel order then block order.

use crate::kernels::PANEL;

/// Inputs per block of the Q8_0 format.
pub const Q8_BLOCK: usize = 32;

/// How a weight matrix's values are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    F32,
    F16,
    Bf16,
    /// 8-bit integers with one scale per output channel.
    Int8,
    /// 8-bit integers with one scale per block of 32 inputs, GGUF's `Q8_0`.
    Q8_0,
}

impl Precision {
    pub const ALL: [Precision; 5] = [
        Precision::F32,
        Precision::F16,
        Precision::Bf16,
        Precision::Int8,
        Precision::Q8_0,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::F16 => "f16",
            Precision::Bf16 => "bf16",
            Precision::Int8 => "int8",
            Precision::Q8_0 => "q8_0",
        }
    }

    pub fn from_name(name: &str) -> Option<Precision> {
        Precision::ALL.into_iter().find(|p| p.name() == name)
    }

    /// Bytes per stored value.
    pub fn value_bytes(self) -> usize {
        match self {
            Precision::F32 => 4,
            Precision::F16 | Precision::Bf16 => 2,
            Precision::Int8 | Precision::Q8_0 => 1,
        }
    }

    /// Bytes of the values of an `n x k` matrix in panels.
    pub fn panel_bytes(self, n: usize, k: usize) -> usize {
        n.div_ceil(PANEL) * k * PANEL * self.value_bytes()
    }

    /// Scales of an `n x k` matrix in panels.
    pub fn scale_count(self, n: usize, k: usize) -> usize {
        let panels = n.div_ceil(PANEL);
        match self {
            Precision::Int8 => panels * PANEL,
            Precision::Q8_0 => panels * k.div_ceil(Q8_BLOCK) * PANEL,
            _ => 0,
        }
    }

    /// Whether a matrix with `k` inputs can be stored this way.
    pub fn fits(self, k: usize) -> bool {
        self != Precision::Q8_0 || k.is_multiple_of(Q8_BLOCK)
    }
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// f32 to IEEE half precision, rounding to nearest even. Values too large become infinity.
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    if exp == 0xff {
        let nan = if man != 0 {
            0x200 | (man >> 13) as u16
        } else {
            0
        };
        return sign | 0x7c00 | nan;
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // A subnormal half: the value is m * 2^-24.
        let shift = (126 - exp) as u32;
        if shift > 24 {
            return sign;
        }
        let full = man | 0x80_0000;
        let mut m = full >> shift;
        let rem = full & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        if rem > half || (rem == half && m & 1 == 1) {
            m += 1;
        }
        return sign | m as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        // A carry into the exponent is right, up to infinity.
        h += 1;
    }
    sign | h as u16
}

/// IEEE half precision to f32, exactly.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            let v = man as f32 * (1.0 / (1u32 << 24) as f32);
            return if sign != 0 { -v } else { v };
        }
        0x1f => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// f32 to bfloat16, rounding to nearest even. NaN stays NaN.
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// bfloat16 to f32, exactly.
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// A row-major `[n, k]` weight matrix before it is laid out in panels.
pub enum Rows<'a> {
    F32(&'a [f32]),
    /// Q8_0 blocks: `d` holds `n * k / 32` scales, `q` the `n * k` values.
    Q8_0 {
        d: &'a [f32],
        q: &'a [i8],
    },
}

/// Quantizes one block the way GGUF's reference `quantize_row_q8_0` does: the scale is the
/// largest magnitude over 127, stored as f16, and the values are rounded with the unrounded
/// scale's reciprocal.
pub fn quantize_q8_0_block(x: &[f32], q: &mut [i8]) -> f32 {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    for (q, &v) in q.iter_mut().zip(x) {
        *q = (v * id).round() as i8;
    }
    f16_to_f32(f32_to_f16(d))
}

/// Quantizes a row with one scale, the largest magnitude over 127.
fn quantize_row_int8(x: &[f32], q: &mut [i8]) -> f32 {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    for (q, &v) in q.iter_mut().zip(x) {
        *q = (v * id).round().clamp(-127.0, 127.0) as i8;
    }
    d
}

/// Lays out a matrix in panels in `precision`. `values` must hold
/// `precision.panel_bytes(n, k)` bytes and `scales` `precision.scale_count(n, k)` floats.
pub fn encode(
    rows: &Rows,
    n: usize,
    k: usize,
    precision: Precision,
    values: &mut [u8],
    scales: &mut [f32],
) {
    assert!(
        precision.fits(k),
        "{precision} needs inputs in multiples of 32"
    );
    assert_eq!(values.len(), precision.panel_bytes(n, k));
    assert_eq!(scales.len(), precision.scale_count(n, k));
    values.fill(0);
    scales.fill(0.0);
    let blocks = k.div_ceil(Q8_BLOCK);
    let vb = precision.value_bytes();
    let mut row = vec![0f32; k];
    let mut q = vec![0i8; k];
    for o in 0..n {
        let (p, j) = (o / PANEL, o % PANEL);
        let panel = &mut values[p * k * PANEL * vb..(p + 1) * k * PANEL * vb];
        // The row's int8 values and their scales, if the target is 8-bit.
        match (rows, precision) {
            (Rows::Q8_0 { d, q: src }, Precision::Q8_0) => {
                q.copy_from_slice(&src[o * k..(o + 1) * k]);
                for b in 0..blocks {
                    scales[(p * blocks + b) * PANEL + j] = d[o * blocks + b];
                }
            }
            _ => {
                match rows {
                    Rows::F32(w) => row.copy_from_slice(&w[o * k..(o + 1) * k]),
                    Rows::Q8_0 { d, q: src } => {
                        for (i, v) in row.iter_mut().enumerate() {
                            *v = src[o * k + i] as f32 * d[o * blocks + i / Q8_BLOCK];
                        }
                    }
                }
                match precision {
                    Precision::Int8 => scales[p * PANEL + j] = quantize_row_int8(&row, &mut q),
                    Precision::Q8_0 => {
                        for b in 0..blocks {
                            let r = b * Q8_BLOCK..((b + 1) * Q8_BLOCK).min(k);
                            scales[(p * blocks + b) * PANEL + j] =
                                quantize_q8_0_block(&row[r.clone()], &mut q[r]);
                        }
                    }
                    _ => {}
                }
            }
        }
        for i in 0..k {
            let at = (i * PANEL + j) * vb;
            match precision {
                Precision::F32 => panel[at..at + 4].copy_from_slice(&row[i].to_le_bytes()),
                Precision::F16 => {
                    panel[at..at + 2].copy_from_slice(&f32_to_f16(row[i]).to_le_bytes())
                }
                Precision::Bf16 => {
                    panel[at..at + 2].copy_from_slice(&f32_to_bf16(row[i]).to_le_bytes())
                }
                Precision::Int8 | Precision::Q8_0 => panel[at] = q[i] as u8,
            }
        }
    }
}

/// The values a stored matrix stands for, row-major `[n, k]`, for tests and comparisons.
pub fn decode(values: &[u8], scales: &[f32], n: usize, k: usize, precision: Precision) -> Vec<f32> {
    let vb = precision.value_bytes();
    let blocks = k.div_ceil(Q8_BLOCK);
    let mut out = vec![0f32; n * k];
    for o in 0..n {
        let (p, j) = (o / PANEL, o % PANEL);
        for i in 0..k {
            let at = ((p * k + i) * PANEL + j) * vb;
            let b = &values[at..at + vb];
            out[o * k + i] = match precision {
                Precision::F32 => f32::from_le_bytes(b.try_into().unwrap()),
                Precision::F16 => f16_to_f32(u16::from_le_bytes(b.try_into().unwrap())),
                Precision::Bf16 => bf16_to_f32(u16::from_le_bytes(b.try_into().unwrap())),
                Precision::Int8 => b[0] as i8 as f32 * scales[p * PANEL + j],
                Precision::Q8_0 => {
                    b[0] as i8 as f32 * scales[(p * blocks + i / Q8_BLOCK) * PANEL + j]
                }
            };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_half_round_trips_through_f32() {
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(f32_to_f16(f) & 0x7c00 == 0x7c00 && f32_to_f16(f) & 0x3ff != 0);
                continue;
            }
            assert_eq!(f32_to_f16(f), h, "{h:#06x} -> {f}");
        }
    }

    #[test]
    fn halves_round_to_nearest_even() {
        // Exactly between 1.0 and the next half (2^-10 apart): ties go to the even one.
        let one_and_half_ulp = 1.0 + 2f32.powi(-11);
        assert_eq!(f32_to_f16(one_and_half_ulp), 0x3c00);
        let next = 1.0 + 3.0 * 2f32.powi(-11);
        assert_eq!(f32_to_f16(next), 0x3c02);
        assert_eq!(f32_to_f16(65520.0), 0x7c00);
        assert_eq!(f32_to_f16(65519.0), 0x7bff);
        // The smallest subnormal half is 2^-24, and half of it rounds to zero (even).
        assert_eq!(f32_to_f16(2f32.powi(-24)), 1);
        assert_eq!(f32_to_f16(2f32.powi(-25)), 0);
        assert_eq!(f32_to_f16(1.5 * 2f32.powi(-25)), 1);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn halves_match_the_cpu_conversion() {
        if !std::arch::is_x86_feature_detected!("f16c") {
            return;
        }
        use std::arch::x86_64::*;
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..200_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let x = f32::from_bits(s as u32);
            if x.is_nan() {
                continue;
            }
            // SAFETY: F16C is present.
            let hw = unsafe {
                _mm_extract_epi16::<0>(_mm_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(_mm_set1_ps(
                    x,
                ))) as u16
            };
            assert_eq!(f32_to_f16(x), hw, "{x:e}");
        }
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8001)), 0x3f81);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
        for h in [0u16, 0x8000, 0x3f80, 0x7f7f, 0xff80] {
            assert_eq!(f32_to_bf16(bf16_to_f32(h)), h);
        }
    }

    #[test]
    fn q8_0_blocks_follow_the_reference_quantizer() {
        let x: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 0.37).collect();
        let mut q = [0i8; 32];
        let d = quantize_q8_0_block(&x, &mut q);
        assert_eq!(q[0], -127);
        assert_eq!(*q.iter().max().unwrap(), 127);
        for (v, &qi) in x.iter().zip(&q) {
            assert!((qi as f32 * d - v).abs() <= 0.5 * d * 1.01);
        }
        let mut zero = [1i8; 32];
        assert_eq!(quantize_q8_0_block(&[0.0; 32], &mut zero), 0.0);
        assert_eq!(zero, [0; 32]);
    }

    #[test]
    fn every_precision_round_trips_within_its_resolution() {
        let (n, k) = (37, 64);
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i * 7919 % 1000) as f32 / 500.0 - 1.0) * (1.0 + (i / k) as f32))
            .collect();
        for precision in Precision::ALL {
            let mut values = vec![0u8; precision.panel_bytes(n, k)];
            let mut scales = vec![0f32; precision.scale_count(n, k)];
            encode(&Rows::F32(&w), n, k, precision, &mut values, &mut scales);
            let back = decode(&values, &scales, n, k, precision);
            for o in 0..n {
                let row = &w[o * k..(o + 1) * k];
                let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
                for (b, v) in back[o * k..(o + 1) * k].iter().zip(row) {
                    let tol = match precision {
                        Precision::F32 => 0.0,
                        Precision::F16 => v.abs() * 2f32.powi(-11),
                        Precision::Bf16 => v.abs() * 2f32.powi(-8),
                        Precision::Int8 => amax / 127.0 * 0.51,
                        // Q8_0 rounds with the exact scale and stores it as f16.
                        Precision::Q8_0 => amax / 127.0 * 0.6,
                    };
                    assert!((b - v).abs() <= tol, "{precision} row {o}: {b} vs {v}");
                }
            }
        }
    }

    #[test]
    fn q8_0_blocks_are_laid_out_unchanged() {
        let (n, k) = (20, 64);
        let q: Vec<i8> = (0..n * k).map(|i| (i % 255) as i8).collect();
        let d: Vec<f32> = (0..n * k / 32)
            .map(|i| f16_to_f32(f32_to_f16(0.01 * (i + 1) as f32)))
            .collect();
        let precision = Precision::Q8_0;
        let mut values = vec![0u8; precision.panel_bytes(n, k)];
        let mut scales = vec![0f32; precision.scale_count(n, k)];
        encode(
            &Rows::Q8_0 { d: &d, q: &q },
            n,
            k,
            precision,
            &mut values,
            &mut scales,
        );
        let back = decode(&values, &scales, n, k, precision);
        for o in 0..n {
            for i in 0..k {
                assert_eq!(back[o * k + i], q[o * k + i] as f32 * d[o * 2 + i / 32]);
            }
        }
        // Converting the blocks to another precision dequantizes them first.
        let mut f32_values = vec![0u8; Precision::F32.panel_bytes(n, k)];
        encode(
            &Rows::Q8_0 { d: &d, q: &q },
            n,
            k,
            Precision::F32,
            &mut f32_values,
            &mut [],
        );
        assert_eq!(decode(&f32_values, &[], n, k, Precision::F32), back);
    }
}
