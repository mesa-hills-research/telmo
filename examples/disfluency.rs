//! Writes what the disfluency tagger decides for each utterance of a words file, to compare with
//! another implementation.
//!
//! ```text
//! cargo run --release --example disfluency -- TAGGER ROWS.jsonl [--threads N] > OUT.jsonl
//! ```
//!
//! TAGGER is a pack or an export directory (converted to f32). Each line of ROWS.jsonl is an
//! object with an `id` and its `words`, as the speech model wrote them. Each output line holds
//! the `id`, whether each word is `deleted`, the tagger's probability `p` (1 where the filler
//! rules delete the word), its most likely deleting tag's first letter in `letters`, and whether
//! the filler rules (`filler`) or the filler rules and the other rules (`rules`) delete it.

use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use telmo::disfluency::Tagger;
use telmo::json::{self, Value};
use telmo::{Options, Precision};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn flags(values: impl Iterator<Item = bool>) -> String {
    let list: Vec<&str> = values.map(|v| if v { "true" } else { "false" }).collect();
    format!("[{}]", list.join(","))
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut options = Options {
        precision: Precision::F32,
        ..Options::default()
    };
    let mut paths: Vec<PathBuf> = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--threads" => {
                options.threads =
                    (args.next().and_then(|t| t.parse().ok())).ok_or("--threads takes a number")?
            }
            _ => paths.push(arg.into()),
        }
    }
    let [tagger, rows] = paths.as_slice() else {
        return Err("Usage: disfluency TAGGER ROWS.jsonl [--threads N]".into());
    };
    let tagger = Tagger::load(tagger, &options).map_err(|e| e.to_string())?;
    let text = std::fs::read_to_string(rows).map_err(|e| e.to_string())?;
    let mut out = BufWriter::new(std::io::stdout().lock());
    let (start, mut count) = (Instant::now(), 0);
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let row = json::parse(line.as_bytes()).map_err(|e| e.to_string())?;
        let id = row.get("id").ok_or("a row has no id")?;
        let words: Vec<&str> = (row.get("words").and_then(Value::as_array))
            .ok_or("a row has no words")?
            .iter()
            .map(|w| w.as_str().ok_or("a word isn't a string"))
            .collect::<Result<_, _>>()?;
        let decided = tagger.decide(&words);
        let p: Vec<String> = (decided.iter())
            .map(|d| format!("{:?}", if d.filler { 1.0 } else { d.probability }))
            .collect();
        writeln!(
            out,
            r#"{{"id":{},"deleted":{},"p":[{}],"letters":"{}","filler":{},"rules":{}}}"#,
            id.to_json(),
            flags(decided.iter().map(|d| d.deleted)),
            p.join(","),
            decided.iter().map(|d| d.tag).collect::<String>(),
            flags(decided.iter().map(|d| d.filler)),
            flags(decided.iter().map(|d| d.filler || d.rule)),
        )
        .map_err(|e| e.to_string())?;
        count += 1;
    }
    out.flush().map_err(|e| e.to_string())?;
    eprintln!(
        "{count} utterances in {:.1} s with {} threads ({:?}).",
        start.elapsed().as_secs_f64(),
        tagger.threads(),
        tagger.precision()
    );
    Ok(())
}
