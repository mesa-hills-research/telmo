//! A real FFT for power-of-two lengths, in f64.
//!
//! A real signal of length n is packed into n/2 complex values, transformed with an iterative
//! radix-2 FFT and split into the n/2 + 1 bins of the real transform.

use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }

    fn mul(self, o: Complex) -> Complex {
        Complex::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }

    fn add(self, o: Complex) -> Complex {
        Complex::new(self.re + o.re, self.im + o.im)
    }

    fn sub(self, o: Complex) -> Complex {
        Complex::new(self.re - o.re, self.im - o.im)
    }

    fn conj(self) -> Complex {
        Complex::new(self.re, -self.im)
    }

    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

/// A precomputed plan for one length.
#[derive(Debug, Clone)]
pub struct RealFft {
    n: usize,
    /// e^(-2πij/(n/2)) for j < n/4, the twiddles of the half-length complex FFT.
    half_twiddles: Vec<Complex>,
    /// e^(-2πik/n) for k <= n/2, for the split into real bins.
    split_twiddles: Vec<Complex>,
    /// Bit-reversed index of each position of the half-length FFT.
    reversed: Vec<usize>,
}

impl RealFft {
    /// A plan for real inputs of length `n`, a power of two of at least 4.
    pub fn new(n: usize) -> RealFft {
        assert!(
            n >= 4 && n.is_power_of_two(),
            "the FFT length must be a power of two of at least 4"
        );
        let m = n / 2;
        let half_twiddles = (0..m / 2)
            .map(|j| unit(-2.0 * PI * j as f64 / m as f64))
            .collect();
        let split_twiddles = (0..=m)
            .map(|k| unit(-2.0 * PI * k as f64 / n as f64))
            .collect();
        let bits = m.trailing_zeros();
        let reversed = (0..m)
            .map(|i| {
                if bits == 0 {
                    0
                } else {
                    i.reverse_bits() >> (usize::BITS - bits)
                }
            })
            .collect();
        RealFft {
            n,
            half_twiddles,
            split_twiddles,
            reversed,
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// Transforms `input` (length n) into `output` (length n/2 + 1).
    /// `scratch` must hold n/2 values.
    pub fn forward(&self, input: &[f64], output: &mut [Complex], scratch: &mut [Complex]) {
        let n = self.n;
        let m = n / 2;
        assert_eq!(input.len(), n);
        assert_eq!(output.len(), m + 1);
        assert_eq!(scratch.len(), m);

        for (i, &r) in self.reversed.iter().enumerate() {
            scratch[r] = Complex::new(input[2 * i], input[2 * i + 1]);
        }
        let mut size = 2;
        while size <= m {
            let half = size / 2;
            let step = m / size;
            for start in (0..m).step_by(size) {
                for j in 0..half {
                    let w = self.half_twiddles[j * step];
                    let a = scratch[start + j];
                    let b = scratch[start + j + half].mul(w);
                    scratch[start + j] = a.add(b);
                    scratch[start + j + half] = a.sub(b);
                }
            }
            size *= 2;
        }

        // Z[k] = E[k] + i O[k] with E and O the transforms of the even and odd samples.
        // X[k] = E[k] + e^(-2πik/n) O[k].
        for k in 0..=m {
            let z = scratch[k % m];
            let zc = scratch[(m - k) % m].conj();
            let even = Complex::new((z.re + zc.re) * 0.5, (z.im + zc.im) * 0.5);
            let diff = z.sub(zc);
            // (Z[k] - conj(Z[m-k])) / 2i
            let odd = Complex::new(diff.im * 0.5, -diff.re * 0.5);
            output[k] = even.add(self.split_twiddles[k].mul(odd));
        }
    }
}

fn unit(angle: f64) -> Complex {
    Complex::new(angle.cos(), angle.sin())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct_dft(input: &[f64]) -> Vec<Complex> {
        let n = input.len();
        (0..=n / 2)
            .map(|k| {
                let mut acc = Complex::default();
                for (t, &x) in input.iter().enumerate() {
                    // Reduce k*t mod n first so the angle stays small and exact.
                    let angle = -2.0 * PI * ((k * t) % n) as f64 / n as f64;
                    acc = acc.add(unit(angle).mul(Complex::new(x, 0.0)));
                }
                acc
            })
            .collect()
    }

    #[test]
    fn matches_a_direct_dft() {
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut random = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        for n in [4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let fft = RealFft::new(n);
            let input: Vec<f64> = (0..n).map(|_| random()).collect();
            let mut output = vec![Complex::default(); n / 2 + 1];
            let mut scratch = vec![Complex::default(); n / 2];
            fft.forward(&input, &mut output, &mut scratch);
            let expected = direct_dft(&input);
            for (k, (got, want)) in output.iter().zip(&expected).enumerate() {
                let err = got.sub(*want).norm_sqr().sqrt();
                assert!(
                    err < 1e-11 * n as f64,
                    "n={n} bin {k}: got {got:?}, want {want:?}"
                );
            }
        }
    }

    #[test]
    fn a_cosine_lands_in_its_bin() {
        let n = 512;
        let fft = RealFft::new(n);
        let input: Vec<f64> = (0..n)
            .map(|t| (2.0 * PI * 5.0 * t as f64 / n as f64).cos())
            .collect();
        let mut output = vec![Complex::default(); n / 2 + 1];
        let mut scratch = vec![Complex::default(); n / 2];
        fft.forward(&input, &mut output, &mut scratch);
        for (k, bin) in output.iter().enumerate() {
            let expected = if k == 5 { n as f64 / 2.0 } else { 0.0 };
            assert!(
                (bin.re - expected).abs() < 1e-9 && bin.im.abs() < 1e-9,
                "bin {k}: {bin:?}"
            );
        }
    }
}
