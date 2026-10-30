//! Real-time factor of single passes over clips of 2 to 60 seconds, at several thread counts,
//! with the time of each step and a check that every thread count gives bit-identical output.
//!
//! ```text
//! cargo bench --bench rtf -- MODEL_DIR LIST.tsv [--threads 1,6] [--seconds 2,5,10,20,30,60] [--runs 3] [--isa scalar|avx2]
//! ```
//!
//! The clips are consecutive utterances of LIST.tsv (`id<TAB>wav path<TAB>text`, 16 kHz audio)
//! joined end to end and cut to length.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use notate::kernels::Isa;
use notate::parakeet::Timings;
use notate::{Model, Options};

fn main() {
    let mut args = std::env::args().skip(1).filter(|a| a != "--bench");
    let mut threads = vec![1, 6];
    let mut seconds = vec![2.0, 5.0, 10.0, 20.0, 30.0, 60.0];
    let mut runs = 3;
    let mut isa = None;
    let mut positional: Vec<String> = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--threads" => {
                threads = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|t| t.parse().unwrap())
                    .collect()
            }
            "--seconds" => {
                seconds = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|t| t.parse().unwrap())
                    .collect()
            }
            "--runs" => runs = args.next().unwrap().parse().unwrap(),
            "--isa" => {
                isa = Some(if args.next().as_deref() == Some("scalar") {
                    Isa::Scalar
                } else {
                    Isa::Avx2
                })
            }
            _ => positional.push(arg),
        }
    }
    let [model_dir, list] = positional.as_slice() else {
        eprintln!(
            "Usage: rtf MODEL_DIR LIST.tsv [--threads 1,6] [--seconds 2,5,...] [--runs 3] [--isa scalar|avx2]"
        );
        std::process::exit(2);
    };

    // Enough audio for the longest clip.
    let longest = seconds.iter().cloned().fold(0.0, f64::max);
    let mut audio = Vec::new();
    for line in std::fs::read_to_string(list).expect("the list").lines() {
        if audio.len() as f64 >= longest * 16000.0 {
            break;
        }
        let path = PathBuf::from(line.split('\t').nth(1).expect("a WAV path"));
        let wav = notate::wav::read(&path).expect("a WAV file");
        assert_eq!(wav.sample_rate, 16000);
        audio.extend_from_slice(&wav.samples);
    }

    let mut outputs: Vec<Vec<Vec<f32>>> = vec![Vec::new(); seconds.len()];
    for &t in &threads {
        let start = Instant::now();
        let model = Model::load(&PathBuf::from(model_dir), &Options { threads: t, isa })
            .expect("the model loads");
        println!(
            "{}, {t} threads, {} kernels, loaded in {:.2} s",
            model.name(),
            model.isa().name(),
            start.elapsed().as_secs_f64()
        );
        println!(
            "{:>8} {:>10} {:>8} {:>10} {:>10} {:>10}",
            "seconds", "best ms", "RTF", "encoder ms", "decoder ms", "x realtime"
        );
        for (i, &s) in seconds.iter().enumerate() {
            let samples = &audio[..((s * 16000.0) as usize).min(audio.len())];
            let mut best: Option<(Duration, Timings)> = None;
            for _ in 0..runs {
                let start = Instant::now();
                let transcript = model.transcribe(samples).expect("a transcript");
                let elapsed = start.elapsed();
                if best.as_ref().is_none_or(|(b, _)| elapsed < *b) {
                    best = Some((elapsed, transcript.timings));
                }
            }
            let (elapsed, timings) = best.unwrap();
            let decoder = timings.get("enc_proj")
                + timings.get("predict")
                + timings.get("joint")
                + timings.get("ctc_head");
            let encoder =
                timings.total() - decoder - timings.get("features") - timings.get("detokenize");
            println!(
                "{s:>8.0} {:>10.1} {:>8.4} {:>10.1} {:>10.1} {:>10.1}",
                elapsed.as_secs_f64() * 1e3,
                elapsed.as_secs_f64() / s,
                encoder.as_secs_f64() * 1e3,
                decoder.as_secs_f64() * 1e3,
                s / elapsed.as_secs_f64()
            );
            if i + 1 == seconds.len() {
                let total = timings.total().as_secs_f64();
                println!("  steps of the {s:.0} s pass:");
                for (name, d) in &timings.steps {
                    println!(
                        "  {name:<16} {:>9.2} ms {:>5.1}%",
                        d.as_secs_f64() * 1e3,
                        100.0 * d.as_secs_f64() / total
                    );
                }
            }
            // The encoder output, to compare across thread counts.
            let encoded = match &model {
                Model::Parakeet(m) => {
                    let features = m.features(samples).unwrap();
                    let mut timings = Timings::default();
                    m.encode(&features.data, features.frames, &mut timings, &mut |_, _, _| {})
                        .data
                }
                Model::Granite(m) => {
                    let features = m.features(samples).unwrap();
                    let mut timings = Timings::default();
                    m.encode(&features.data, features.frames, &mut timings, &mut |_, _, _| {})
                        .unwrap()
                        .data
                }
            };
            outputs[i].push(encoded);
        }
    }
    let identical = outputs.iter().all(|per_threads| {
        per_threads.windows(2).all(|w| {
            w[0].len() == w[1].len()
                && w[0]
                    .iter()
                    .zip(&w[1])
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        })
    });
    println!("Encoder output bit-identical across {threads:?} threads: {identical}");
    if let Some(peak) = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<u64>().ok())
        })
    {
        println!("Peak memory {:.0} MB", peak as f64 / 1024.0);
    }
}
