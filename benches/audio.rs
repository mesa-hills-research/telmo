//! Time per second of audio of the streaming audio steps, each fed 20 ms blocks of a minute of
//! noise, best of three runs.
//!
//! ```text
//! cargo bench --bench audio
//! ```

use std::time::Instant;

use notate::resample::Resampler;
use notate::vad::{Vad, VadConfig};

const SECONDS: usize = 60;
const RUNS: usize = 3;

fn main() {
    for (from, to) in [(48000, 16000), (44100, 16000), (8000, 16000)] {
        let input = noise(from as usize * SECONDS);
        let mut out = Vec::with_capacity(to as usize * SECONDS + 1000);
        let per_second = best_of(|| {
            out.clear();
            let mut resampler = Resampler::new(from, to);
            for block in input.chunks(from as usize / 50) {
                resampler.process(block, &mut out);
            }
            resampler.flush(&mut out);
        });
        report(&format!("resample {from} Hz to {to} Hz"), per_second);
    }

    let audio = noise(16000 * SECONDS);
    let per_second = best_of(|| {
        let mut vad = Vad::new(VadConfig::default());
        for block in audio.chunks(320) {
            std::hint::black_box(vad.push(block));
        }
    });
    report("voice activity", per_second);
}

/// The best time of `RUNS` runs of `f`, in seconds per second of audio.
fn best_of(mut f: impl FnMut()) -> f64 {
    (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed().as_secs_f64() / SECONDS as f64
        })
        .fold(f64::MAX, f64::min)
}

fn report(what: &str, per_second: f64) {
    println!(
        "{what:<32} {:>8.3} ms per second of audio",
        per_second * 1000.0
    );
}

/// Uniform noise in [-0.5, 0.5).
fn noise(len: usize) -> Vec<f32> {
    let mut seed = 0x9E3779B97F4A7C15u64;
    (0..len)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}
