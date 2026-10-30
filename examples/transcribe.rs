//! Transcribes WAV files and prints the text with timings.
//!
//! ```text
//! cargo run --release --example transcribe -- MODEL_DIR FILE.wav... [--threads N] [--isa scalar|avx2] [--steps] [--repeat N]
//! ```
//!
//! MODEL_DIR holds `model.safetensors`, `config.json` and `tokenizer.json` of
//! nvidia/parakeet-tdt-0.6b-v3 or ibm-granite/granite-speech-5.0-470m-turboctc. The audio must be
//! 16 kHz.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use notate::kernels::Isa;
use notate::{Model, Options};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut options = Options::default();
    let mut steps = false;
    let mut repeat = 1;
    let mut paths: Vec<PathBuf> = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--threads" => {
                options.threads = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--threads takes a number")?
            }
            "--repeat" => {
                repeat = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--repeat takes a number")?
            }
            "--isa" => {
                options.isa = Some(match args.next().as_deref() {
                    Some("scalar") => Isa::Scalar,
                    Some("avx2") => Isa::Avx2,
                    _ => return Err("--isa takes scalar or avx2".into()),
                })
            }
            "--steps" => steps = true,
            _ => paths.push(arg.into()),
        }
    }
    if paths.len() < 2 {
        return Err("Usage: transcribe MODEL_DIR FILE.wav... [--threads N] [--isa scalar|avx2] [--steps] [--repeat N]".into());
    }
    let model_dir = paths.remove(0);

    let start = Instant::now();
    let model = Model::load(&model_dir, &options).map_err(|e| e.to_string())?;
    let load = start.elapsed();
    eprintln!(
        "Loaded {} in {:.2} s, {} threads, {} kernels, {:.0} MB of weights.",
        model.name(),
        load.as_secs_f64(),
        model.threads(),
        model.isa().name(),
        model.weight_bytes() as f64 / 1e6
    );

    for path in &paths {
        let audio = notate::wav::read(path).map_err(|e| e.to_string())?;
        if audio.sample_rate != 16000 {
            return Err(format!(
                "{} is {} Hz. Resample it to 16 kHz first.",
                path.display(),
                audio.sample_rate
            ));
        }
        let mut best: Option<(Duration, notate::Transcript)> = None;
        for _ in 0..repeat {
            let start = Instant::now();
            let transcript = model
                .transcribe(&audio.samples)
                .map_err(|e| e.to_string())?;
            let elapsed = start.elapsed();
            if best.as_ref().is_none_or(|(b, _)| elapsed < *b) {
                best = Some((elapsed, transcript));
            }
        }
        let (elapsed, transcript) = best.expect("at least one run");
        println!("{}", transcript.text);
        let seconds = audio.seconds();
        eprintln!(
            "{}: {:.2} s of audio in {:.3} s, real-time factor {:.4} ({:.1}x real time)",
            path.display(),
            seconds,
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() / seconds,
            seconds / elapsed.as_secs_f64()
        );
        if steps {
            let total = transcript.timings.total().as_secs_f64();
            for (name, d) in &transcript.timings.steps {
                eprintln!(
                    "  {name:<16} {:>9.2} ms {:>5.1}%",
                    d.as_secs_f64() * 1e3,
                    100.0 * d.as_secs_f64() / total
                );
            }
        }
    }
    if let Some(peak) = peak_memory() {
        eprintln!("Peak memory {:.0} MB.", peak as f64 / 1e6);
    }
    Ok(())
}

/// The process's peak resident memory in bytes, on Linux.
fn peak_memory() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}
