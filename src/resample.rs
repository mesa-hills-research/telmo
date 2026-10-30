//! Sample-rate conversion with a polyphase windowed-sinc filter.
//!
//! The ratio of the two rates is reduced by their greatest common divisor to `up / down`, so
//! 48 kHz to 16 kHz is 1 to 3 and 44.1 kHz to 16 kHz is 160 to 441. Each output sample is the dot
//! product of the latest input samples with one of `up` phases of a sinc lowpass filter shaped
//! by a Kaiser window. The filter keeps 90% of the band below the lower rate's Nyquist frequency
//! and attenuates the rest by at least 80 dB. For a 16 kHz output that keeps 0 to 7.2 kHz and
//! removes 8 kHz and above, and for 8 kHz upsampled to 16 kHz it keeps 0 to 3.6 kHz and removes
//! the images from 4 kHz up.

use std::f64::consts::PI;

/// The band kept, as a fraction of the lower rate's Nyquist frequency. The transition band
/// runs from there to the Nyquist frequency.
const PASSBAND: f64 = 0.9;
/// The stopband attenuation the filter is designed for, in dB. It leaves a margin above the
/// 80 dB the filter has to reach, since Kaiser's formulas are approximations.
const DESIGN_ATTENUATION_DB: f64 = 90.0;
/// The taps of each phase are padded with zeros to a multiple of this, the number of
/// independent sums in the dot product.
const LANES: usize = 8;

/// A streaming resampler from one integer rate to another.
///
/// The output is the same whatever the sizes of the chunks given to [`Resampler::process`].
/// Output sample `n` stands for the input at time `(n - delay) / to_hz`, where
/// [`Resampler::delay`] is the filter's delay in output samples.
#[derive(Debug, Clone)]
pub struct Resampler {
    from_hz: u32,
    to_hz: u32,
    up: usize,
    down: usize,
    /// Taps of each phase, a multiple of `LANES`.
    taps: usize,
    /// `up` rows of `taps` coefficients. Row `p` holds the phase for outputs whose position
    /// is `p / up` input samples past an input sample, oldest input first.
    coefs: Vec<f32>,
    delay: usize,
    /// Input samples from absolute index `history_start` on. The stream starts with `taps - 1`
    /// zeros before index 0.
    history: Vec<f32>,
    history_start: i64,
    /// The next output `n` has `n * down = index * up + phase`, so it is the first output that
    /// needs input `index`.
    index: i64,
    phase: usize,
    pushed: u64,
    emitted: u64,
}

impl Resampler {
    /// A resampler from `from_hz` to `to_hz`. Equal rates pass samples through unchanged.
    ///
    /// Panics if either rate is zero.
    pub fn new(from_hz: u32, to_hz: u32) -> Resampler {
        assert!(from_hz > 0 && to_hz > 0, "sample rates must be above zero");
        let g = gcd(from_hz, to_hz);
        let up = (to_hz / g) as usize;
        let down = (from_hz / g) as usize;
        let (taps, delay, coefs) = if up == down {
            (0, 0, Vec::new())
        } else {
            let filter = Filter::new(from_hz, to_hz, up);
            let wide = filter.half_width_units;
            // Output n is centered on input (n - delay) * down / up. The delay is the smallest that
            // keeps every tap at or before input floor(n * down / up), and `taps` covers the
            // window from there back for every phase.
            let wide = wide as u64;
            let delay = (wide - 1).div_ceil(down as u64) as usize;
            let span = (delay as u64 * down as u64 + wide).div_ceil(up as u64) as usize;
            let taps = span.div_ceil(LANES) * LANES;
            let mut coefs = vec![0.0f32; up * taps];
            for (p, row) in coefs.chunks_exact_mut(taps).enumerate() {
                for (q, c) in row.iter_mut().enumerate() {
                    let back = (taps - 1 - q) as i64;
                    let offset = back * up as i64 + p as i64 - delay as i64 * down as i64;
                    *c = filter.at(offset, up) as f32;
                }
            }
            (taps, delay, coefs)
        };
        let mut resampler = Resampler {
            from_hz,
            to_hz,
            up,
            down,
            taps,
            coefs,
            delay,
            history: Vec::new(),
            history_start: 0,
            index: 0,
            phase: 0,
            pushed: 0,
            emitted: 0,
        };
        resampler.reset();
        resampler
    }

    pub fn from_hz(&self) -> u32 {
        self.from_hz
    }

    pub fn to_hz(&self) -> u32 {
        self.to_hz
    }

    /// The filter's delay in output samples. Output sample `n` stands for the input at time
    /// `(n - delay) / to_hz`, so the first `delay` outputs come before the input's start.
    pub fn delay(&self) -> usize {
        self.delay
    }

    /// Resamples the next part of the stream and appends what's ready to `out`. After `n`
    /// input samples in all, `ceil(n * to_hz / from_hz)` output samples are out.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        self.pushed += input.len() as u64;
        if self.up == self.down {
            out.extend_from_slice(input);
            return;
        }
        self.history.extend_from_slice(input);
        self.produce(out, u64::MAX);
    }

    /// Ends the stream, appending the last `delay` output samples (the input is taken as zero
    /// past its end), and resets the resampler for a new stream.
    pub fn flush(&mut self, out: &mut Vec<f32>) {
        if self.up != self.down {
            let target = self.ready_for(self.pushed) + self.delay as u64;
            if self.emitted < target {
                // The last output's newest tap is input floor((target - 1) * down / up).
                let last = (target - 1) as u128 * self.down as u128 / self.up as u128;
                let end = self.history_start + self.history.len() as i64;
                let zeros = (last as i64 + 1 - end).max(0) as usize;
                self.history.resize(self.history.len() + zeros, 0.0);
                self.produce(out, target);
            }
        }
        self.reset();
    }

    /// The number of outputs ready once `inputs` input samples have come in.
    fn ready_for(&self, inputs: u64) -> u64 {
        (inputs as u128 * self.up as u128).div_ceil(self.down as u128) as u64
    }

    fn reset(&mut self) {
        self.history.clear();
        self.history.resize(self.taps.saturating_sub(1), 0.0);
        self.history_start = -(self.history.len() as i64);
        self.index = 0;
        self.phase = 0;
        self.pushed = 0;
        self.emitted = 0;
    }

    /// Computes outputs while their inputs are in the history, up to `limit` outputs in all,
    /// then drops the history no later output needs.
    fn produce(&mut self, out: &mut Vec<f32>, limit: u64) {
        let end = self.history_start + self.history.len() as i64;
        while self.index < end && self.emitted < limit {
            let first = (self.index - self.history_start) as usize + 1 - self.taps;
            let row = &self.coefs[self.phase * self.taps..(self.phase + 1) * self.taps];
            out.push(dot(&self.history[first..first + self.taps], row));
            self.emitted += 1;
            self.phase += self.down;
            self.index += (self.phase / self.up) as i64;
            self.phase %= self.up;
        }
        let keep_from = self.index + 1 - self.taps as i64;
        let drop = (keep_from - self.history_start).clamp(0, self.history.len() as i64);
        self.history.drain(..drop as usize);
        self.history_start += drop;
    }
}

/// Resamples a whole signal. The output is aligned with the input in time (the filter's delay
/// is removed) and has `len * to_hz / from_hz` samples, rounded to the nearest.
pub fn resample(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz {
        return samples.to_vec();
    }
    let mut resampler = Resampler::new(from_hz, to_hz);
    let mut out = Vec::new();
    resampler.process(samples, &mut out);
    let delay = resampler.delay();
    resampler.flush(&mut out);
    let len = (2 * samples.len() as u128 * to_hz as u128 + from_hz as u128) / (2 * from_hz as u128);
    out.drain(..delay);
    out.truncate(len as usize);
    out
}

/// A dot product in `LANES` independent sums, so the compiler can vectorize it. The order of
/// the additions is fixed, which keeps the output independent of how the input is chunked.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; LANES];
    for (x, y) in a.chunks_exact(LANES).zip(b.chunks_exact(LANES)) {
        for ((s, &x), &y) in acc.iter_mut().zip(x).zip(y) {
            *s += x * y;
        }
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]))
}

/// The continuous lowpass kernel, in units of input samples.
#[derive(Debug, Clone)]
struct Filter {
    /// The cutoff in cycles per input sample, halfway through the transition band.
    cutoff: f64,
    /// Half the window's length in units of `1 / up` input samples.
    half_width_units: i64,
    half_width: f64,
    beta: f64,
    i0_beta: f64,
}

impl Filter {
    fn new(from_hz: u32, to_hz: u32, up: usize) -> Filter {
        let nyquist = from_hz.min(to_hz) as f64 / 2.0;
        let from = from_hz as f64;
        let cutoff = (1.0 + PASSBAND) / 2.0 * nyquist / from;
        let transition = 2.0 * PI * (1.0 - PASSBAND) * nyquist / from;
        // Kaiser's estimates of the filter order and the window's shape for an attenuation.
        let a = DESIGN_ATTENUATION_DB;
        let order = (a - 7.95) / (2.285 * transition);
        let beta = 0.1102 * (a - 8.7);
        let half_width_units = (order / 2.0 * up as f64).ceil() as i64;
        Filter {
            cutoff,
            half_width_units,
            half_width: half_width_units as f64 / up as f64,
            beta,
            i0_beta: bessel_i0(beta),
        }
    }

    /// The kernel at `offset / up` input samples from its center.
    fn at(&self, offset: i64, up: usize) -> f64 {
        if offset.abs() >= self.half_width_units {
            return 0.0;
        }
        let t = offset as f64 / up as f64;
        let x = t / self.half_width;
        let window = bessel_i0(self.beta * (1.0 - x * x).sqrt()) / self.i0_beta;
        2.0 * self.cutoff * sinc(2.0 * self.cutoff * t) * window
    }
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

/// The modified Bessel function of the first kind and order zero, by its power series.
fn bessel_i0(x: f64) -> f64 {
    let half = x / 2.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..200 {
        term *= half / k as f64;
        let t = term * term;
        sum += t;
        if t < sum * 1e-17 {
            break;
        }
    }
    sum
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A complex exponential's response, as the cosine's and the sine's resampled outputs.
    struct Analytic {
        re: Vec<f32>,
        im: Vec<f32>,
    }

    fn resample_analytic(phase: impl Fn(usize) -> f64, len: usize, from: u32, to: u32) -> Analytic {
        let cos: Vec<f32> = (0..len).map(|k| phase(k).cos() as f32).collect();
        let sin: Vec<f32> = (0..len).map(|k| phase(k).sin() as f32).collect();
        Analytic {
            re: resample(&cos, from, to),
            im: resample(&sin, from, to),
        }
    }

    /// The amplitude, relative to the input's, of the component at `out_hz` (signed, an
    /// integer multiple of 10 Hz) in the output for a complex exponential at `in_hz`
    /// (a multiple of 10 Hz too). The output is measured over 0.1 s after a 20 ms margin, which
    /// holds a whole number of periods of every component, so each lands in one DFT bin.
    fn gain(from: u32, to: u32, in_hz: i64, out_hz: i64) -> f64 {
        let len = from as usize * 14 / 100;
        let from_i = from as i64;
        let signal = resample_analytic(
            |k| 2.0 * PI * ((in_hz * k as i64).rem_euclid(from_i)) as f64 / from as f64,
            len,
            from,
            to,
        );
        let start = to as usize / 50;
        let n = to as usize / 10;
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for i in 0..n {
            let angle = -2.0 * PI * ((out_hz * i as i64).rem_euclid(to as i64)) as f64 / to as f64;
            let (yr, yi) = (signal.re[start + i] as f64, signal.im[start + i] as f64);
            let (c, s) = (angle.cos(), angle.sin());
            re += yr * c - yi * s;
            im += yr * s + yi * c;
        }
        (re * re + im * im).sqrt() / n as f64
    }

    fn db(x: f64) -> f64 {
        20.0 * x.log10()
    }

    /// The output frequency an input frequency lands on, in [-to/2, to/2).
    fn alias(hz: i64, to: u32) -> i64 {
        let to = to as i64;
        (hz + to / 2).rem_euclid(to) - to / 2
    }

    struct Response {
        ripple_db: f64,
        stopband_db: f64,
    }

    /// The passband's ripple (largest gain minus smallest, in dB) from 10 Hz to 90% of the
    /// lower Nyquist frequency, and the stopband's attenuation (the smallest, in dB) of
    /// everything at or above that Nyquist frequency, on a 10 Hz grid.
    fn measure(from: u32, to: u32) -> Response {
        let nyquist = from.min(to) as i64 / 2;
        let pass_edge = nyquist * 9 / 10;
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        let mut worst = f64::MIN;
        let mut f = 10;
        while f <= pass_edge {
            let g = db(gain(from, to, f, alias(f, to)));
            lo = lo.min(g);
            hi = hi.max(g);
            if from < to {
                // Upsampling leaves an image of the passband at f - from.
                worst = worst.max(db(gain(from, to, f, alias(f - from as i64, to))));
            }
            f += if f < 200 { 10 } else { 50 };
        }
        if from > to {
            // Everything from the output's Nyquist frequency up to the input's aliases.
            let mut f = nyquist;
            while f < from as i64 / 2 {
                worst = worst.max(db(gain(from, to, f, alias(f, to))));
                f += 50;
            }
        } else {
            // The images of the transition band, which land from the input's Nyquist up.
            let mut f = pass_edge;
            while f < nyquist {
                worst = worst.max(db(gain(from, to, f, alias(f - from as i64, to))));
                f += 10;
            }
        }
        Response {
            ripple_db: hi - lo,
            stopband_db: -worst,
        }
    }

    #[test]
    fn the_passband_is_flat_and_the_stopband_is_80_db_down() {
        for (from, to) in [(48000, 16000), (44100, 16000), (8000, 16000)] {
            let r = measure(from, to);
            eprintln!(
                "{from} to {to}: passband ripple {:.5} dB, stopband {:.1} dB down",
                r.ripple_db, r.stopband_db
            );
            assert!(
                r.ripple_db < 0.01,
                "{from} to {to}: ripple {} dB",
                r.ripple_db
            );
            assert!(
                r.stopband_db >= 80.0,
                "{from} to {to}: stopband {} dB",
                r.stopband_db
            );
        }
    }

    /// The magnitude of each output sample of a resampled complex chirp.
    fn sweep(from: u32, to: u32, f0: f64, f1: f64, seconds: f64) -> Vec<f64> {
        let len = (from as f64 * seconds) as usize;
        let rate = (f1 - f0) / seconds;
        let fs = from as f64;
        let signal = resample_analytic(
            |k| {
                let t = k as f64 / fs;
                2.0 * PI * (f0 * t + rate * t * t / 2.0)
            },
            len,
            from,
            to,
        );
        signal
            .re
            .iter()
            .zip(&signal.im)
            .map(|(&r, &i)| ((r as f64).powi(2) + (i as f64).powi(2)).sqrt())
            .collect()
    }

    #[test]
    fn a_sweep_keeps_the_passband_and_removes_the_stopband() {
        for (from, to) in [(48000u32, 16000u32), (44100, 16000)] {
            let nyquist = to as f64 / 2.0;
            let edge = to as usize / 50;
            // 50 Hz to 7.2 kHz in 2 s.
            let pass = sweep(from, to, 50.0, 0.9 * nyquist, 2.0);
            let pass = &pass[edge..pass.len() - edge];
            let (lo, hi) = pass
                .iter()
                .fold((f64::MAX, f64::MIN), |(lo, hi), &m| (lo.min(m), hi.max(m)));
            assert!(
                db(hi) - db(lo) < 0.01,
                "{from} to {to}: {} to {} dB",
                db(lo),
                db(hi)
            );
            // 8.05 kHz up to the input's Nyquist frequency in 4 s. The sweep moves less than
            // 50 Hz across the filter's window.
            let stop = sweep(from, to, nyquist + 50.0, from as f64 / 2.0, 4.0);
            let peak = stop[edge..stop.len() - edge]
                .iter()
                .fold(0.0f64, |m, &x| m.max(x));
            eprintln!(
                "{from} to {to} swept: passband {:.5} dB, stopband {:.1} dB down",
                db(hi) - db(lo),
                -db(peak)
            );
            assert!(db(peak) < -80.0, "{from} to {to}: {} dB", db(peak));
        }
    }

    fn noise(len: usize, mut seed: u64) -> Vec<f32> {
        (0..len)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn the_output_is_the_same_whatever_the_chunks() {
        for (from, to) in [
            (48000, 16000),
            (44100, 16000),
            (8000, 16000),
            (22050, 16000),
            (16000, 48000),
        ] {
            let input = noise(from as usize / 3 + 17, 0x9E3779B97F4A7C15);
            let mut whole = Resampler::new(from, to);
            let mut expected = Vec::new();
            whole.process(&input, &mut expected);
            whole.flush(&mut expected);

            let mut seed = 0x2545F4914F6CDD1Du64 ^ from as u64;
            let mut chunked = Resampler::new(from, to);
            for round in 0..2 {
                let mut got = Vec::new();
                let mut pos = 0;
                while pos < input.len() {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let size = ((seed % 700) as usize).min(input.len() - pos);
                    chunked.process(&input[pos..pos + size], &mut got);
                    assert_eq!(got.len() as u64, chunked.ready_for((pos + size) as u64));
                    pos += size;
                }
                chunked.flush(&mut got);
                // The second round checks that flush starts a fresh stream.
                assert!(got == expected, "{from} to {to}, round {round}");
            }
        }
    }

    #[test]
    fn the_delay_is_where_an_impulse_lands() {
        for (from, to) in [(48000, 16000), (44100, 16000), (8000, 16000)] {
            // An input sample that falls exactly on an output sample.
            let g = gcd(from, to) as usize;
            let k = 7 * from as usize / g;
            let mut input = vec![0.0f32; 2 * k];
            input[k] = 1.0;
            let mut r = Resampler::new(from, to);
            let mut out = Vec::new();
            r.process(&input, &mut out);
            r.flush(&mut out);
            let peak = (0..out.len())
                .max_by(|&a, &b| out[a].total_cmp(&out[b]))
                .unwrap();
            assert_eq!(
                peak,
                r.delay() + k * to as usize / from as usize,
                "{from} to {to}"
            );
        }
        assert_eq!(Resampler::new(48000, 16000).delay(), 57);
        assert_eq!(Resampler::new(16000, 16000).delay(), 0);
    }

    #[test]
    fn the_one_shot_output_is_aligned_with_its_input() {
        for (from, to) in [
            (48000u32, 16000u32),
            (44100, 16000),
            (8000, 16000),
            (16000, 16000),
        ] {
            for len in [0, 1, 2, 3, 1000, 44101] {
                let out = resample(&vec![0.0; len], from, to);
                let expected = ((len as f64 * to as f64 / from as f64) + 0.5).floor() as usize;
                assert_eq!(out.len(), expected, "{from} to {to}, {len} samples");
            }
            // A 1 kHz sine comes out as the same sine at the new rate.
            let len = from as usize / 2;
            let sine = |i: usize, rate: u32| (2.0 * PI * 1000.0 * i as f64 / rate as f64).sin();
            let input: Vec<f32> = (0..len).map(|i| sine(i, from) as f32).collect();
            let out = resample(&input, from, to);
            let edge = to as usize / 50;
            for (i, &y) in out.iter().enumerate().take(out.len() - edge).skip(edge) {
                assert!(
                    (y as f64 - sine(i, to)).abs() < 1e-4,
                    "{from} to {to} at {i}: {y}"
                );
            }
        }
    }

    #[test]
    fn rates_reduce_to_their_ratio() {
        let r = Resampler::new(44100, 16000);
        assert_eq!((r.up, r.down), (160, 441));
        assert_eq!(r.coefs.len(), 160 * r.taps);
        assert_eq!(resample(&vec![0.0; 44100], 44100, 16000).len(), 16000);
        let r = Resampler::new(48000, 16000);
        assert_eq!((r.up, r.down), (1, 3));
        let r = Resampler::new(8000, 16000);
        assert_eq!((r.up, r.down), (2, 1));
    }

    #[test]
    fn equal_rates_pass_samples_through() {
        let input = noise(1000, 7);
        let mut r = Resampler::new(16000, 16000);
        let mut out = Vec::new();
        r.process(&input[..300], &mut out);
        r.process(&input[300..], &mut out);
        r.flush(&mut out);
        assert!(out == input);
        assert!(resample(&input, 16000, 16000) == input);
    }

    #[test]
    fn the_polyphase_output_equals_a_direct_windowed_sinc() {
        for (from, to) in [(48000, 16000), (44100, 16000), (8000, 16000)] {
            let input = noise(from as usize / 10, 99);
            let mut r = Resampler::new(from, to);
            let mut out = Vec::new();
            r.process(&input, &mut out);
            r.flush(&mut out);
            // Each output straight from the kernel: output n is centered on input
            // (n - delay) * down / up, and the kernel is evaluated at every input's offset.
            let filter = Filter::new(from, to, r.up);
            let (up, down, delay) = (r.up as i64, r.down as i64, r.delay as i64);
            let mut worst = 0.0f64;
            for (n, &y) in out.iter().enumerate() {
                let center = (n as i64 - delay) * down;
                let lo = (center - filter.half_width_units).div_euclid(up).max(0);
                let hi =
                    ((center + filter.half_width_units).div_euclid(up) + 1).min(input.len() as i64);
                let direct: f64 = (lo..hi)
                    .map(|k| input[k as usize] as f64 * filter.at(center - k * up, r.up))
                    .sum();
                worst = worst.max((y as f64 - direct).abs());
            }
            assert!(worst < 2e-6, "{from} to {to}: {worst}");
        }
    }
}
