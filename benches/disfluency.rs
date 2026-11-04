//! Time the disfluency tagger takes over a text of a given number of words, rules and every
//! member included.
//!
//! ```text
//! cargo bench --bench disfluency -- TAGGER ROWS.jsonl [--threads 4] [--words 40] [--windows 200] [--runs 3]
//! ```
//!
//! The words of ROWS.jsonl (one object with `words` per line) are joined in order and cut into
//! `--windows` texts of `--words` words. Each is decided `--runs` times after one pass to warm
//! up, and the median, the 95th percentile and the slowest pass are reported, with the tokens
//! per text and the size of TAGGER when it's a pack.

use std::path::PathBuf;
use std::time::Instant;

use telmo::disfluency::Tagger;
use telmo::json::{self, Value};
use telmo::{Options, Precision};

fn main() {
    let mut args = std::env::args().skip(1).filter(|a| a != "--bench");
    let mut options = Options {
        threads: 4,
        precision: Precision::F32,
        ..Options::default()
    };
    let (mut words, mut windows, mut runs) = (40usize, 200usize, 3usize);
    let mut positional: Vec<PathBuf> = Vec::new();
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next()
                .expect("a value")
                .parse::<usize>()
                .expect("a number")
        };
        match arg.as_str() {
            "--threads" => options.threads = value(),
            "--words" => words = value(),
            "--windows" => windows = value(),
            "--runs" => runs = value(),
            _ => positional.push(arg.into()),
        }
    }
    let [tagger_path, rows] = positional.as_slice() else {
        eprintln!(
            "Usage: disfluency TAGGER ROWS.jsonl [--threads 4] [--words 40] [--windows 200] [--runs 3]"
        );
        std::process::exit(2);
    };
    let tagger = Tagger::load(tagger_path, &options).expect("the tagger");
    let text = std::fs::read_to_string(rows).expect("the rows");
    let mut all: Vec<String> = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let row = json::parse(line.as_bytes()).expect("a row");
        let list = row.get("words").and_then(Value::as_array).expect("words");
        all.extend(list.iter().map(|w| w.as_str().expect("a word").to_string()));
    }
    let texts: Vec<Vec<&str>> = (all.chunks_exact(words).take(windows))
        .map(|c| c.iter().map(String::as_str).collect())
        .collect();
    assert!(!texts.is_empty(), "the rows hold fewer than {words} words");
    for t in &texts {
        tagger.decide(t);
    }
    let mut times: Vec<f64> = Vec::new();
    for _ in 0..runs {
        for t in &texts {
            let start = Instant::now();
            std::hint::black_box(tagger.decide(t));
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }
    times.sort_by(f64::total_cmp);
    let at = |q: f64| times[((times.len() - 1) as f64 * q).round() as usize];
    let tokens: Vec<usize> = texts.iter().map(|t| tagger.token_ids(t).0.len()).collect();
    let mean_tokens = tokens.iter().sum::<usize>() as f64 / tokens.len() as f64;
    let size = (tagger_path.is_file())
        .then(|| std::fs::metadata(tagger_path).map(|m| m.len()).ok())
        .flatten();
    println!(
        "{} texts of {words} words ({mean_tokens:.1} tokens on average, {} at most), {} members, \
         {:?}, {} threads, {}",
        texts.len(),
        tokens.iter().max().unwrap_or(&0),
        tagger.members(),
        tagger.precision(),
        tagger.threads(),
        tagger.isa().name(),
    );
    println!(
        "median {:.1} ms, 95th percentile {:.1} ms, slowest {:.1} ms over {} passes",
        at(0.5),
        at(0.95),
        times[times.len() - 1],
        times.len()
    );
    if let Some(bytes) = size {
        println!("pack {:.1} MB ({bytes} bytes)", bytes as f64 / 1e6);
    }
}
