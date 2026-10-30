//! Log-mel features as Parakeet's front end computes them.
//!
//! Pre-emphasis 0.97, a centered STFT with zero padding (FFT of 512, a 400-sample symmetric Hann
//! window centered in it, hop 160), the power spectrum, 128 Slaney-normalized mel filters from 0
//! to 8 kHz, `ln(x + 2^-24)`, then each mel bin normalized to zero mean and unit standard deviation
//! over the utterance (standard deviation with n - 1, plus 1e-5).

use crate::error::{Error, Result};
use crate::fft::{Complex, RealFft};

/// Front-end settings. [`FrontEndConfig::parakeet`] gives Parakeet's.
#[derive(Debug, Clone, PartialEq)]
pub struct FrontEndConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop: usize,
    pub n_mels: usize,
    pub preemphasis: f32,
}

impl FrontEndConfig {
    pub fn parakeet() -> Self {
        FrontEndConfig {
            sample_rate: 16000,
            n_fft: 512,
            win_length: 400,
            hop: 160,
            n_mels: 128,
            preemphasis: 0.97,
        }
    }
}

/// Features of one utterance, row-major `[frames, n_mels]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Features {
    pub frames: usize,
    pub n_mels: usize,
    pub data: Vec<f32>,
}

impl Features {
    pub fn frame(&self, i: usize) -> &[f32] {
        &self.data[i * self.n_mels..(i + 1) * self.n_mels]
    }
}

/// Mel filters stored sparsely, one run of nonzero weights per filter.
#[derive(Debug, Clone)]
struct MelFilter {
    first_bin: usize,
    weights: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct FrontEnd {
    config: FrontEndConfig,
    fft: RealFft,
    /// The window zero-padded to n_fft and centered.
    window: Vec<f64>,
    filters: Vec<MelFilter>,
}

const LOG_GUARD: f64 = 1.0 / (1u64 << 24) as f64;
const STD_EPS: f64 = 1e-5;

impl FrontEnd {
    pub fn new(config: FrontEndConfig) -> FrontEnd {
        let fft = RealFft::new(config.n_fft);
        let mut window = vec![0.0; config.n_fft];
        let offset = (config.n_fft - config.win_length) / 2;
        let denom = (config.win_length - 1) as f64;
        for i in 0..config.win_length {
            // torch.hann_window(periodic=False), rounded to f32 as torch stores it.
            let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / denom).cos();
            window[offset + i] = w as f32 as f64;
        }
        let filters = slaney_mel_filters(config.sample_rate, config.n_fft, config.n_mels);
        FrontEnd {
            config,
            fft,
            window,
            filters,
        }
    }

    pub fn config(&self) -> &FrontEndConfig {
        &self.config
    }

    /// The number of feature frames for `samples` samples (the STFT's frames minus the last,
    /// which only covers padding).
    pub fn frames_for(&self, samples: usize) -> usize {
        samples / self.config.hop
    }

    /// Computes normalized log-mel features of 16 kHz mono samples.
    pub fn compute(&self, samples: &[f32]) -> Result<Features> {
        let c = &self.config;
        let frames = self.frames_for(samples.len());
        if frames < 2 {
            return Err(Error::Model(format!(
                "The audio is too short ({} samples). It needs at least {}.",
                samples.len(),
                2 * c.hop
            )));
        }
        let pad = c.n_fft / 2;
        // Pre-emphasis in f32 as the reference does, then centered zero padding.
        let mut padded = vec![0.0f64; samples.len() + 2 * pad];
        padded[pad] = samples[0] as f64;
        for i in 1..samples.len() {
            padded[pad + i] = (samples[i] - c.preemphasis * samples[i - 1]) as f64;
        }

        let bins = c.n_fft / 2 + 1;
        let mut segment = vec![0.0f64; c.n_fft];
        let mut spectrum = vec![Complex::default(); bins];
        let mut scratch = vec![Complex::default(); c.n_fft / 2];
        let mut power = vec![0.0f64; bins];
        let mut data = vec![0.0f32; frames * c.n_mels];
        let mut logs = vec![0.0f64; frames * c.n_mels];
        for f in 0..frames {
            let start = f * c.hop;
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
                logs[f * c.n_mels + m] = (acc + LOG_GUARD).ln();
            }
        }

        for m in 0..c.n_mels {
            let column = || (0..frames).map(|f| logs[f * c.n_mels + m]);
            let mean = column().sum::<f64>() / frames as f64;
            let var = column().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (frames - 1) as f64;
            let scale = 1.0 / (var.sqrt() + STD_EPS);
            for f in 0..frames {
                data[f * c.n_mels + m] = ((logs[f * c.n_mels + m] - mean) * scale) as f32;
            }
        }
        Ok(Features {
            frames,
            n_mels: c.n_mels,
            data,
        })
    }
}

fn hz_to_mel_slaney(hz: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        hz / f_sp
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_sp * mel
    }
}

/// librosa's `filters.mel(norm="slaney", htk=False)` from 0 Hz to Nyquist, in its f32 rounding.
fn slaney_mel_filters(sample_rate: u32, n_fft: usize, n_mels: usize) -> Vec<MelFilter> {
    let bins = n_fft / 2 + 1;
    let fft_freqs: Vec<f64> = (0..bins)
        .map(|k| k as f64 * sample_rate as f64 / n_fft as f64)
        .collect();
    let max_mel = hz_to_mel_slaney(sample_rate as f64 / 2.0);
    let mel_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz_slaney(max_mel * i as f64 / (n_mels + 1) as f64))
        .collect();
    (0..n_mels)
        .map(|m| {
            let (lo, center, hi) = (mel_points[m], mel_points[m + 1], mel_points[m + 2]);
            let enorm = 2.0 / (hi - lo);
            let weights: Vec<f32> = fft_freqs
                .iter()
                .map(|&f| {
                    let lower = (f - lo) / (center - lo);
                    let upper = (hi - f) / (hi - center);
                    let w = lower.min(upper).max(0.0) as f32;
                    (w as f64 * enorm) as f32
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
    fn slaney_scale_round_trips_and_is_linear_below_1khz() {
        for hz in [0.0, 100.0, 999.0, 1000.0, 4000.0, 8000.0] {
            assert!((mel_to_hz_slaney(hz_to_mel_slaney(hz)) - hz).abs() < 1e-9);
        }
        assert!((hz_to_mel_slaney(500.0) - 7.5).abs() < 1e-12);
        assert!((hz_to_mel_slaney(1000.0) - 15.0).abs() < 1e-12);
    }

    #[test]
    fn mel_filters_cover_the_band_with_slaney_area() {
        let filters = slaney_mel_filters(16000, 512, 128);
        assert_eq!(filters.len(), 128);
        // Slaney normalization makes each filter's area in Hz equal 1 (a triangle of
        // height 2/(hi-lo) and base hi-lo). The low filters are narrower than two bins,
        // so only the wide ones sample their triangle well enough to check.
        for filter in &filters[100..] {
            let area: f64 = filter.weights.iter().map(|&w| w as f64 * 31.25).sum();
            assert!((area - 1.0).abs() < 0.05, "area {area}");
        }
        assert!(filters.last().unwrap().first_bin + filters.last().unwrap().weights.len() <= 257);
    }

    #[test]
    fn features_are_normalized_per_bin() {
        let fe = FrontEnd::new(FrontEndConfig::parakeet());
        let samples: Vec<f32> = (0..16000)
            .map(|i| (i as f32 * 0.05).sin() * 0.3 + ((i * 7919) % 101) as f32 / 1000.0)
            .collect();
        let feats = fe.compute(&samples).unwrap();
        assert_eq!(feats.frames, 100);
        for m in [0, 40, 127] {
            let col: Vec<f64> = (0..feats.frames)
                .map(|f| feats.frame(f)[m] as f64)
                .collect();
            let mean = col.iter().sum::<f64>() / col.len() as f64;
            let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (col.len() - 1) as f64;
            assert!(mean.abs() < 1e-5, "bin {m} mean {mean}");
            assert!(
                (var.sqrt() - 1.0).abs() < 1e-3,
                "bin {m} std {}",
                var.sqrt()
            );
        }
        assert!(fe.compute(&samples[..300]).is_err());
    }
}
