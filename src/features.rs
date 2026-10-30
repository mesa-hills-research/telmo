//! Log-mel features as Parakeet's front end computes them.
//!
//! Pre-emphasis 0.97, a centered STFT with zero padding (FFT of 512, a 400-sample symmetric Hann
//! window centered in it, hop 160), the power spectrum, 128 Slaney-normalized mel filters from 0
//! to 8 kHz, `ln(x + 2^-24)`, then each mel bin normalized to zero mean and unit standard deviation
//! over the utterance (standard deviation with n - 1, plus 1e-5).

use std::ops::Range;

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
        Ok(normalize(&self.log_mels(samples), c.n_mels))
    }

    /// A stream that computes the log-mel features of one utterance as its audio comes in.
    pub fn stream(&self) -> LogMelStream {
        LogMelStream::new(self.clone())
    }

    /// The log-mel values of every frame, before normalization, `[frames, n_mels]`.
    fn log_mels(&self, samples: &[f32]) -> Vec<f64> {
        let c = &self.config;
        let frames = self.frames_for(samples.len());
        let pad = c.n_fft / 2;
        // Pre-emphasis in f32 as the reference does, then centered zero padding.
        let mut padded = vec![0.0f64; samples.len() + 2 * pad];
        if let Some(&first) = samples.first() {
            padded[pad] = first as f64;
        }
        for i in 1..samples.len() {
            padded[pad + i] = (samples[i] - c.preemphasis * samples[i - 1]) as f64;
        }

        let mut work = FrameWork::new(c);
        let mut logs = vec![0.0f64; frames * c.n_mels];
        let (lo, hi) = self.window_span(0);
        for (f, out) in logs.chunks_exact_mut(c.n_mels).enumerate() {
            let start = (f * c.hop) as isize + pad as isize;
            let span = (start + lo) as usize..(start + hi) as usize;
            self.log_mel_frame(&padded[span], &mut work, out);
        }
        logs
    }

    /// The samples under frame `f`'s window, relative to the utterance's first sample
    /// (negative ones fall in the zero padding). Outside it the window is zero.
    fn window_span(&self, f: usize) -> (isize, isize) {
        let c = &self.config;
        let lo =
            (f * c.hop) as isize + ((c.n_fft - c.win_length) / 2) as isize - (c.n_fft / 2) as isize;
        (lo, lo + c.win_length as isize)
    }

    /// The log-mel values of one frame from its pre-emphasized samples under the window.
    fn log_mel_frame(&self, samples: &[f64], work: &mut FrameWork, out: &mut [f64]) {
        let c = &self.config;
        let offset = (c.n_fft - c.win_length) / 2;
        let window = &self.window[offset..offset + c.win_length];
        for ((s, &x), &w) in work.segment[offset..].iter_mut().zip(samples).zip(window) {
            *s = x * w;
        }
        self.fft
            .forward(&work.segment, &mut work.spectrum, &mut work.scratch);
        for (p, s) in work.power.iter_mut().zip(&work.spectrum) {
            *p = s.norm_sqr();
        }
        for (o, filter) in out.iter_mut().zip(&self.filters) {
            let mut acc = 0.0;
            for (k, &w) in filter.weights.iter().enumerate() {
                acc += w as f64 * work.power[filter.first_bin + k];
            }
            *o = (acc + LOG_GUARD).ln();
        }
    }
}

/// Buffers for computing one frame. The segment's samples outside the window stay zero.
#[derive(Debug, Clone)]
struct FrameWork {
    segment: Vec<f64>,
    spectrum: Vec<Complex>,
    scratch: Vec<Complex>,
    power: Vec<f64>,
}

impl FrameWork {
    fn new(c: &FrontEndConfig) -> FrameWork {
        let bins = c.n_fft / 2 + 1;
        FrameWork {
            segment: vec![0.0; c.n_fft],
            spectrum: vec![Complex::default(); bins],
            scratch: vec![Complex::default(); c.n_fft / 2],
            power: vec![0.0; bins],
        }
    }
}

/// Normalizes each mel bin of `logs` (`[frames, n_mels]`, at least two frames) to zero mean and
/// unit standard deviation over its frames.
fn normalize(logs: &[f64], n_mels: usize) -> Features {
    let frames = logs.len() / n_mels;
    let mut data = vec![0.0f32; frames * n_mels];
    for m in 0..n_mels {
        let column = || (0..frames).map(|f| logs[f * n_mels + m]);
        let mean = column().sum::<f64>() / frames as f64;
        let var = column().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (frames - 1) as f64;
        let scale = 1.0 / (var.sqrt() + STD_EPS);
        for f in 0..frames {
            data[f * n_mels + m] = ((logs[f * n_mels + m] - mean) * scale) as f32;
        }
    }
    Features {
        frames,
        n_mels,
        data,
    }
}

/// Parakeet's log-mel features of one utterance, computed as its audio comes in.
///
/// [`LogMelStream::push`] takes the utterance's samples in order, in chunks of any size, and
/// computes each frame once its whole window has arrived, half a window after its center (200
/// samples or 12.5 ms for Parakeet). [`LogMelStream::finish`] ends the utterance with the same
/// zero padding as [`FrontEnd::compute`], which brings the count to `samples / hop`. Every
/// frame equals the offline computation's, bit for bit, and [`LogMelStream::normalize`] over
/// all frames equals [`FrontEnd::compute`].
#[derive(Debug, Clone)]
pub struct LogMelStream {
    front_end: FrontEnd,
    work: FrameWork,
    /// Pre-emphasized samples from index `buffer_start` of the utterance on. Negative indices
    /// are the zero padding before it.
    buffer: Vec<f64>,
    buffer_start: isize,
    last_sample: Option<f32>,
    samples: usize,
    /// The log-mel values of the frames so far, `[frames, n_mels]`.
    logs: Vec<f64>,
    finished: bool,
}

impl LogMelStream {
    pub fn new(front_end: FrontEnd) -> LogMelStream {
        // The first window reaches back into the padding before the utterance.
        let (lo, _) = front_end.window_span(0);
        LogMelStream {
            work: FrameWork::new(&front_end.config),
            buffer: vec![0.0; (-lo) as usize],
            buffer_start: lo,
            last_sample: None,
            samples: 0,
            logs: Vec::new(),
            finished: false,
            front_end,
        }
    }

    pub fn front_end(&self) -> &FrontEnd {
        &self.front_end
    }

    /// Takes the utterance's next samples and computes the frames they complete.
    ///
    /// Panics after [`LogMelStream::finish`].
    pub fn push(&mut self, samples: &[f32]) {
        assert!(!self.finished, "the utterance has finished");
        let a = self.front_end.config.preemphasis;
        self.buffer.reserve(samples.len());
        for &x in samples {
            let y = match self.last_sample {
                Some(previous) => x - a * previous,
                None => x,
            };
            self.buffer.push(y as f64);
            self.last_sample = Some(x);
        }
        self.samples += samples.len();
        self.compute(usize::MAX);
    }

    /// Ends the utterance, computing its last frames over zero padding.
    pub fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        let total = self.front_end.frames_for(self.samples);
        if total > self.frames() {
            let (_, hi) = self.front_end.window_span(total - 1);
            let end = self.buffer_start + self.buffer.len() as isize;
            let zeros = (hi - end).max(0) as usize;
            self.buffer.resize(self.buffer.len() + zeros, 0.0);
            self.compute(total);
        }
    }

    /// The frames computed so far.
    pub fn frames(&self) -> usize {
        self.logs.len() / self.front_end.config.n_mels
    }

    /// The samples pushed so far.
    pub fn samples(&self) -> usize {
        self.samples
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Frame `f`'s log-mel values before normalization, one for each mel bin.
    pub fn log_mel(&self, f: usize) -> &[f64] {
        let n = self.front_end.config.n_mels;
        &self.logs[f * n..(f + 1) * n]
    }

    /// The features of `frames`, each mel bin normalized by its mean and standard deviation
    /// over those frames, as [`FrontEnd::compute`] does over the whole utterance.
    ///
    /// Panics if the range isn't within `0..self.frames()` or holds fewer than two frames.
    pub fn normalize(&self, frames: Range<usize>) -> Features {
        assert!(
            frames.len() >= 2 && frames.end <= self.frames(),
            "normalizing frames {frames:?} of {}",
            self.frames()
        );
        let n = self.front_end.config.n_mels;
        normalize(&self.logs[frames.start * n..frames.end * n], n)
    }

    /// Computes the frames whose windows are in the buffer, up to `limit` frames in all, then
    /// drops the samples no later frame needs.
    fn compute(&mut self, limit: usize) {
        let n_mels = self.front_end.config.n_mels;
        let end = self.buffer_start + self.buffer.len() as isize;
        loop {
            let f = self.frames();
            let (lo, hi) = self.front_end.window_span(f);
            if f >= limit || hi > end {
                break;
            }
            let at = self.logs.len();
            self.logs.resize(at + n_mels, 0.0);
            let span = (lo - self.buffer_start) as usize..(hi - self.buffer_start) as usize;
            self.front_end
                .log_mel_frame(&self.buffer[span], &mut self.work, &mut self.logs[at..]);
        }
        let (lo, _) = self.front_end.window_span(self.frames());
        let drop = (lo - self.buffer_start).clamp(0, self.buffer.len() as isize);
        self.buffer.drain(..drop as usize);
        self.buffer_start += drop;
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

    fn golden_wavs() -> Vec<(String, Vec<f32>)> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/goldens/parakeet-tdt-0.6b-v3");
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".wav"))
            .collect();
        names.sort();
        assert!(names.len() >= 6);
        names
            .into_iter()
            .map(|n| {
                let audio = crate::wav::read(&dir.join(&n)).unwrap();
                (n, audio.samples)
            })
            .collect()
    }

    fn same_bits_f64(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// Frames whose 400-sample window, centered on `160 * f`, lies within `n` samples.
    fn ready(n: usize) -> usize {
        if n < 200 { 0 } else { (n - 200) / 160 + 1 }
    }

    #[test]
    fn the_stream_equals_the_offline_features_bit_for_bit() {
        let fe = FrontEnd::new(FrontEndConfig::parakeet());
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut random = move |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n) as usize
        };
        for (name, samples) in golden_wavs() {
            let logs = fe.log_mels(&samples);
            let offline = fe.compute(&samples).unwrap();
            let mut stream = fe.stream();
            let mut pos = 0;
            while pos < samples.len() {
                // Mostly chunks of up to 4000 samples, some of a few samples, some empty.
                let size = match random(4) {
                    0 => random(4),
                    _ => random(4000),
                }
                .min(samples.len() - pos);
                let before = stream.frames();
                stream.push(&samples[pos..pos + size]);
                pos += size;
                assert_eq!(stream.frames(), ready(pos), "{name} at {pos}");
                for f in before..stream.frames() {
                    assert!(
                        same_bits_f64(stream.log_mel(f), &logs[f * 128..(f + 1) * 128]),
                        "{name} frame {f}"
                    );
                }
            }
            let before = stream.frames();
            stream.finish();
            assert_eq!(stream.frames(), samples.len() / 160);
            for f in before..stream.frames() {
                assert!(same_bits_f64(
                    stream.log_mel(f),
                    &logs[f * 128..(f + 1) * 128]
                ));
            }
            let streamed = stream.normalize(0..stream.frames());
            assert_eq!(streamed.frames, offline.frames);
            assert!(
                streamed
                    .data
                    .iter()
                    .zip(&offline.data)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{name}"
            );
        }
    }

    #[test]
    fn frames_are_ready_once_their_window_is_in() {
        let fe = FrontEnd::new(FrontEndConfig::parakeet());
        let samples: Vec<f32> = (0..2000).map(|i| (i as f32 * 0.03).sin() * 0.5).collect();
        let mut stream = fe.stream();
        for (i, &x) in samples.iter().enumerate() {
            stream.push(&[x]);
            assert_eq!(stream.frames(), ready(i + 1));
        }
        assert_eq!(stream.samples(), 2000);
        stream.finish();
        assert!(stream.is_finished());
        assert_eq!(stream.frames(), 12);
        // A window of the stream is normalized over its own frames.
        let window = stream.normalize(3..10);
        assert_eq!(window.frames, 7);
        for m in [0, 64, 127] {
            let column: Vec<f64> = (0..7).map(|f| window.frame(f)[m] as f64).collect();
            let mean = column.iter().sum::<f64>() / 7.0;
            assert!(mean.abs() < 1e-5, "bin {m}: mean {mean}");
        }
        // The first sample isn't pre-emphasized, and later ones are across pushes.
        let mut a = fe.stream();
        a.push(&samples[..1]);
        a.push(&samples[1..500]);
        a.finish();
        let offline = fe.log_mels(&samples[..500]);
        assert!(same_bits_f64(&a.logs, &offline));
    }
}
