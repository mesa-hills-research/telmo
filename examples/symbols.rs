//! Writes the spoken symbols of transcripts as the symbols pass does, and says what changed.
//!
//! ```text
//! cargo run --release --example symbols -- [--code] [--paths DIR] [--quiet] FILE.tsv...
//! ```
//!
//! Each line of a file is a transcript, its text in the last tab-separated column. The example
//! prints each line it changes, before and after, then for each file its lines, words, spoken
//! symbol names and the runs written as code. With `--paths` it lists the names under `DIR`
//! first and runs snap to them, and it reports how long the listing and the lines took.

use std::process::ExitCode;
use std::time::Instant;

use telmo::{PathLimits, Paths, Symbols};

const NAMES: [&str; 8] = [
    "slash",
    "backslash",
    "dot",
    "underscore",
    "dash",
    "hyphen",
    "colon",
    "tilde",
];

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
    let mut symbols = Symbols::for_language("en").expect("English has symbols");
    let mut paths: Option<Paths> = None;
    let mut quiet = false;
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--code" => symbols = symbols.for_code(),
            "--quiet" => quiet = true,
            "--paths" => {
                let dir = args.next().ok_or("--paths takes a folder")?;
                let start = Instant::now();
                let listed =
                    Paths::scan(dir.as_ref(), &PathLimits::default()).map_err(|e| e.to_string())?;
                eprintln!(
                    "Listed {} names under {dir} in {:.1} ms{}.",
                    listed.len(),
                    start.elapsed().as_secs_f64() * 1e3,
                    if listed.is_complete() {
                        ""
                    } else {
                        ", up to the limit"
                    }
                );
                paths = Some(listed);
            }
            _ => files.push(arg),
        }
    }
    if files.is_empty() {
        return Err("Usage: symbols [--code] [--paths DIR] [--quiet] FILE.tsv...".into());
    }
    for file in &files {
        let text = std::fs::read_to_string(file).map_err(|e| format!("{file}: {e}"))?;
        let (mut lines, mut words, mut names, mut changed) = (0, 0, 0, 0);
        let start = Instant::now();
        for line in text.lines() {
            let said = line.rsplit('\t').next().unwrap_or("");
            let list: Vec<&str> = said.split_whitespace().collect();
            lines += 1;
            words += list.len();
            names += (list.iter().flat_map(|w| w.split('-')))
                .filter(|w| {
                    let bare: String = w.chars().filter(|c| c.is_alphabetic()).collect();
                    NAMES.contains(&bare.to_lowercase().as_str())
                })
                .count();
            let written = symbols.write(said, paths.as_ref());
            let before = list.join(" ");
            if written != before {
                changed += runs(&before, &written);
                if !quiet {
                    println!("  {before}\n  {written}");
                }
            }
        }
        println!(
            "{file}: {lines} lines, {words} words, {names} symbol names, {changed} runs written \
             as code, {:.1} ms",
            start.elapsed().as_secs_f64() * 1e3
        );
    }
    Ok(())
}

/// The runs written as code, each a word of `after` that `before` doesn't have.
fn runs(before: &str, after: &str) -> usize {
    let mut left: Vec<&str> = before.split(' ').collect();
    (after.split(' '))
        .filter(|w| match left.iter().position(|b| b == w) {
            Some(k) => {
                left.remove(k);
                false
            }
            None => true,
        })
        .count()
}
