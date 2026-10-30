//! Time per streaming update of Granite Speech, against a full offline pass over the audio so
//! far.
//!
//! ```text
//! cargo bench --bench stream -- MODEL LIST.tsv [--threads 1,6] [--seconds 60] [--every 0.5,1] [--lag 0,16] [--precision P]
//! ```
//!
//! The audio is consecutive utterances of LIST.tsv (`id<TAB>wav path<TAB>text`, 16 kHz) joined
//! end to end and cut to length. Each stream pushes `--every` seconds at a time and updates after
//! each push, the last update finished. At about a dozen of those points an offline pass runs
//! over all the audio so far, for comparison. MODEL is what `Granite::load` takes, converted to
//! `--precision` (int8 by default) when it isn't a pack.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use notate::granite::Granite;
use notate::parakeet::Timings;
use notate::{GraniteStream, Options, Precision};

fn list<T: std::str::FromStr>(v: &str) -> Vec<T> {
    v.split(',')
        .map(|x| x.parse().ok().expect("a number"))
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1).filter(|a| a != "--bench");
    let mut threads = vec![1usize, 6];
    let mut seconds = 60.0f64;
    let mut everies = vec![0.5f64, 1.0];
    let mut lags = vec![0usize];
    let mut precision = Precision::Int8;
    let mut positional: Vec<String> = Vec::new();
    while let Some(arg) = args.next() {
        let mut value = || args.next().expect("a value");
        match arg.as_str() {
            "--threads" => threads = list(&value()),
            "--seconds" => seconds = value().parse().unwrap(),
            "--every" => everies = list(&value()),
            "--lag" => lags = list(&value()),
            "--precision" => {
                precision = Precision::from_name(&value()).expect("f32, f16, bf16, int8 or q8_0")
            }
            _ => positional.push(arg),
        }
    }
    let [model_dir, list_path] = positional.as_slice() else {
        eprintln!(
            "Usage: stream MODEL LIST.tsv [--threads 1,6] [--seconds 60] [--every 0.5,1] [--lag 0,16] [--precision P]"
        );
        std::process::exit(2);
    };

    let mut audio = Vec::new();
    for line in std::fs::read_to_string(list_path)
        .expect("the list")
        .lines()
    {
        if audio.len() as f64 >= seconds * 16000.0 {
            break;
        }
        let path = PathBuf::from(line.split('\t').nth(1).expect("a WAV path"));
        let wav = notate::wav::read(&path).expect("a WAV file");
        assert_eq!(wav.sample_rate, 16000);
        audio.extend_from_slice(&wav.samples);
    }
    audio.truncate((seconds * 16000.0) as usize);
    let seconds = audio.len() as f64 / 16000.0;

    for &t in &threads {
        let options = Options {
            threads: t,
            isa: None,
            precision,
        };
        let model = Granite::load(&PathBuf::from(model_dir), &options).expect("the model loads");
        println!(
            "{t} threads, {} kernels, {} weights, {seconds:.1} s of audio",
            model.isa().name(),
            model.precision().map_or("mixed", |p| p.name()),
        );
        println!(
            "{:>7} {:>5} {:>8} {:>10} {:>10} {:>10} {:>12} {:>14} {:>10}",
            "every s",
            "lag",
            "updates",
            "mean ms",
            "p90 ms",
            "max ms",
            "x real time",
            "offline ms",
            "update ms"
        );
        for &every in &everies {
            for &lag in &lags {
                let mut stream = GraniteStream::new(&model);
                stream.set_freeze_lag(lag);
                let step = (every * 16000.0) as usize;
                let mut times: Vec<Duration> = Vec::new();
                let pieces = audio.len().div_ceil(step);
                for (i, piece) in audio.chunks(step).enumerate() {
                    stream.push(piece);
                    let start = Instant::now();
                    stream.update(&model, i + 1 == pieces, &mut Timings::default());
                    times.push(start.elapsed());
                }
                let total: Duration = times.iter().sum();
                let mut sorted = times.clone();
                sorted.sort();
                let ms = |d: Duration| d.as_secs_f64() * 1e3;
                // An offline pass over the audio so far at about a dozen update points, against
                // the update there.
                let stride = (pieces / 12).max(1);
                let (mut offline, mut updates) = (Duration::ZERO, Duration::ZERO);
                for i in (stride - 1..pieces).step_by(stride) {
                    let so_far = ((i + 1) * step).min(audio.len());
                    let start = Instant::now();
                    model.transcribe(&audio[..so_far]).expect("a transcript");
                    offline += start.elapsed();
                    updates += times[i];
                }
                let points = (stride - 1..pieces).step_by(stride).count() as f64;
                println!(
                    "{every:>7.2} {lag:>5} {:>8} {:>10.1} {:>10.1} {:>10.1} {:>12.1} {:>14.1} {:>10.1}",
                    times.len(),
                    ms(total) / times.len() as f64,
                    ms(sorted[sorted.len() * 9 / 10]),
                    ms(*sorted.last().unwrap()),
                    seconds / total.as_secs_f64(),
                    ms(offline) / points,
                    ms(updates) / points,
                );
            }
        }
        println!("{:>7} {:>10}", "audio s", "offline ms");
        for length in [5.0, 10.24, 20.0, 30.0, 45.0, 60.0, 90.0] {
            if length > seconds + 1e-9 {
                continue;
            }
            let samples = &audio[..(length * 16000.0) as usize];
            let best = (0..2)
                .map(|_| {
                    let start = Instant::now();
                    model.transcribe(samples).expect("a transcript");
                    start.elapsed()
                })
                .min()
                .unwrap();
            println!("{length:>7.2} {:>10.1}", best.as_secs_f64() * 1e3);
        }
    }
}
