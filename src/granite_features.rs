//! Log-mel features as Granite Speech's front end computes them.
//!
//! These are torchaudio's `MelSpectrogram` defaults: a centered STFT with reflect padding (FFT of
//! 512, a 400-sample periodic Hann window centered in it, hop 160), the power spectrum and 80 HTK
//! mel filters from 0 to 8 kHz without normalization. Then `log10(max(x, 1e-10))`, a floor 8 below
//! the utterance's largest value, `/ 4 + 1`, and deltas over 3 frames with the edge frames
//! repeated. Each frame's 80 values and 80 deltas make 160, and two frames side by side make one
//! row of 320, so rows come at 50 a second. The waveform is padded with zeros at the end when the
//! last pair would otherwise lack its second frame.
//!
//! The STFT, the mel sums and the logarithm run in f64. Every other step rounds to f32 where the
//! reference does.

use crate::error::{Error, Result};
use crate::features::Features;
use crate::fft::{Complex, RealFft};

const SAMPLE_RATE: u32 = 16000;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP: usize = 160;
/// The log-mel floor below the utterance's largest value, in log10 units (80 dB).
const FLOOR: f32 = 8.0;
/// Mel frames stacked into one row.
const STACK: usize = 2;

/// Mel filters stored sparsely, one run of nonzero weights per filter.
#[derive(Debug, Clone)]
struct MelFilter {
    first_bin: usize,
    weights: Vec<f32>,
}

/// Granite Speech's front end.
#[derive(Debug, Clone)]
pub struct GraniteFrontEnd {
    n_mels: usize,
    fft: RealFft,
    /// The window zero-padded to the FFT size and centered.
    window: Vec<f64>,
    filters: Vec<MelFilter>,
}

impl GraniteFrontEnd {
    /// A front end with `n_mels` mel filters (80 for Granite Speech 5.0).
    pub fn new(n_mels: usize) -> GraniteFrontEnd {
        let mut window = vec![0.0; N_FFT];
        let offset = (N_FFT - WIN_LENGTH) / 2;
        for (i, w) in periodic_hann(WIN_LENGTH).into_iter().enumerate() {
            window[offset + i] = w as f64;
        }
        GraniteFrontEnd {
            n_mels,
            fft: RealFft::new(N_FFT),
            window,
            filters: htk_mel_filters(SAMPLE_RATE, N_FFT, n_mels),
        }
    }

    /// Values per row: the mels and their deltas for two frames.
    pub fn width(&self) -> usize {
        2 * STACK * self.n_mels
    }

    /// The number of rows for `samples` samples. Each row holds two 10 ms frames, and a last odd
    /// frame gets a row of its own.
    pub fn rows_for(&self, samples: usize) -> usize {
        (samples / HOP).div_ceil(STACK)
    }

    /// Computes the stacked features of 16 kHz mono samples, `[rows, width]`.
    pub fn compute(&self, samples: &[f32]) -> Result<Features> {
        let rows = self.rows_for(samples.len());
        if samples.len() < 2 * HOP {
            return Err(Error::Model(format!(
                "The audio is too short ({} samples). It needs at least {}.",
                samples.len(),
                2 * HOP
            )));
        }
        let frames = STACK * rows;
        // Zeros at the end, so the last frame's center lies inside the signal.
        let len = samples.len().max((frames - 1) * HOP + 1);
        let signal = |i: usize| samples.get(i).map_or(0.0, |&s| s as f64);
        // Centered reflect padding of the zero-padded signal.
        let pad = N_FFT / 2;
        let padded: Vec<f64> = (0..len + 2 * pad)
            .map(|i| {
                let j = i as isize - pad as isize;
                let j = if j < 0 {
                    -j as usize
                } else if j as usize >= len {
                    2 * (len - 1) - j as usize
                } else {
                    j as usize
                };
                signal(j)
            })
            .collect();

        let n_mels = self.n_mels;
        let bins = N_FFT / 2 + 1;
        let mut segment = vec![0.0f64; N_FFT];
        let mut spectrum = vec![Complex::default(); bins];
        let mut scratch = vec![Complex::default(); N_FFT / 2];
        let mut power = vec![0.0f64; bins];
        let mut logs = vec![0f32; frames * n_mels];
        for f in 0..frames {
            let start = f * HOP;
            for (i, s) in segment.iter_mut().enumerate() {
                *s = padded[start + i] * self.window[i];
            }
            self.fft.forward(&segment, &mut spectrum, &mut scratch);
            for (p, s) in power.iter_mut().zip(&spectrum) {
                *p = s.norm_sqr();
            }
            for (m, filter) in self.filters.iter().enumerate() {
                let mut acc = 0.0;
                for (k, &w) in filter.weights.iter().enumerate() {
                    acc += w as f64 * power[filter.first_bin + k];
                }
                // The reference keeps the mel energies in f32 and clamps them at 1e-10.
                let energy = (acc as f32).max(1e-10);
                logs[f * n_mels + m] = (energy as f64).log10() as f32;
            }
        }
        let floor = logs.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)) - FLOOR;
        for v in &mut logs {
            *v = v.max(floor) / 4.0 + 1.0;
        }

        let width = self.width();
        let mut data = vec![0f32; rows * width];
        for f in 0..frames {
            let out = &mut data[f * 2 * n_mels..(f + 1) * 2 * n_mels];
            let (before, after) = (f.saturating_sub(1), (f + 1).min(frames - 1));
            for m in 0..n_mels {
                out[m] = logs[f * n_mels + m];
                out[n_mels + m] = (logs[after * n_mels + m] - logs[before * n_mels + m]) / 2.0;
            }
        }
        Ok(Features {
            frames: rows,
            n_mels: width,
            data,
        })
    }
}

/// `torch.hann_window(n, periodic=True)` in f32.
fn periodic_hann(n: usize) -> Vec<f32> {
    let step = (2.0 * std::f64::consts::PI / n as f64) as f32;
    (0..n)
        .map(|i| {
            let x = i as f32 * step;
            0.5 + -0.5 * ((x as f64).cos() as f32)
        })
        .collect()
}

/// `torch.linspace(start, end, steps)` in f32, which counts up from `start` for the first half
/// and down from `end` for the second.
fn linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
    let step = (end - start) / (steps - 1) as f32;
    (0..steps)
        .map(|i| {
            if i < steps / 2 {
                start + step * i as f32
            } else {
                end - step * (steps - 1 - i) as f32
            }
        })
        .collect()
}

/// torchaudio's `melscale_fbanks(norm=None, mel_scale="htk")` from 0 Hz to Nyquist, in its f32
/// rounding.
fn htk_mel_filters(sample_rate: u32, n_fft: usize, n_mels: usize) -> Vec<MelFilter> {
    let bins = n_fft / 2 + 1;
    let nyquist = (sample_rate / 2) as f64;
    let all_freqs = linspace(0.0, nyquist as f32, bins);
    let m_max = 2595.0 * (1.0 + nyquist / 700.0).log10();
    let mels = linspace(0.0, m_max as f32, n_mels + 2);
    let f_pts: Vec<f32> = mels
        .iter()
        .map(|&m| {
            let p = 10f64.powf((m / 2595.0) as f64) as f32;
            700.0 * (p - 1.0)
        })
        .collect();
    let f_diff: Vec<f32> = f_pts.windows(2).map(|w| w[1] - w[0]).collect();
    (0..n_mels)
        .map(|m| {
            let weights: Vec<f32> = all_freqs
                .iter()
                .map(|&f| {
                    let down = -(f_pts[m] - f) / f_diff[m];
                    let up = (f_pts[m + 2] - f) / f_diff[m + 1];
                    down.min(up).max(0.0)
                })
                .collect();
            let first = weights.iter().position(|&w| w != 0.0).unwrap_or(0);
            let last = weights
                .iter()
                .rposition(|&w| w != 0.0)
                .map_or(first, |l| l + 1);
            MelFilter {
                first_bin: first,
                weights: weights[first..last].to_vec(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_meets_both_ends() {
        let v = linspace(0.0, 2840.023, 82);
        assert_eq!(v.len(), 82);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[81], 2840.023);
        assert!(v.windows(2).all(|w| w[1] > w[0]));
        assert_eq!(linspace(0.0, 8000.0, 257)[256], 8000.0);
        assert_eq!(linspace(0.0, 8000.0, 257)[1], 31.25);
    }

    #[test]
    fn htk_filters_are_triangles_of_height_one() {
        let filters = htk_mel_filters(16000, 512, 80);
        assert_eq!(filters.len(), 80);
        for filter in &filters[10..] {
            let peak = filter.weights.iter().fold(0f32, |m, &w| m.max(w));
            assert!(peak > 0.5 && peak <= 1.0, "peak {peak}");
        }
        // The lowest filters are narrower than one FFT bin and may miss every bin.
        let last = filters.last().unwrap();
        assert!(last.first_bin + last.weights.len() <= 257);
    }

    #[test]
    fn rows_pair_frames_and_pad_the_last() {
        let fe = GraniteFrontEnd::new(80);
        assert_eq!(fe.width(), 320);
        assert_eq!(fe.rows_for(16000), 50);
        assert_eq!(fe.rows_for(16160), 51);
        let samples: Vec<f32> = (0..16160)
            .map(|i| (i as f32 * 0.05).sin() * 0.3 + ((i * 7919) % 101) as f32 / 1000.0)
            .collect();
        let feats = fe.compute(&samples).unwrap();
        assert_eq!((feats.frames, feats.n_mels), (51, 320));
        let logs = || feats.data.chunks_exact(160).flat_map(|frame| &frame[..80]);
        let max = logs().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        let min_log = logs().fold(f32::INFINITY, |m, &v| m.min(v));
        // The floor is 8 below the largest log10 value, so 2 below it after dividing by 4.
        assert!(min_log >= max - 2.0 - 1e-6, "{min_log} {max}");
        // The first delta repeats the first frame, so it halves the step to the next.
        let first = &feats.data[..160];
        let second = &feats.data[160..240];
        for m in 0..80 {
            assert_eq!(first[80 + m], (second[m] - first[m]) / 2.0);
        }
        assert!(fe.compute(&samples[..300]).is_err());
    }
}
