//! Transcribes a test set and scores it the way the Open ASR leaderboard does: both texts go
//! through Whisper's English normalizer, then the word errors of all utterances are summed and
//! divided by the reference words.
//!
//! ```text
//! cargo run --release --example wer -- MODEL_DIR LIST.tsv... [--threads N] [--limit N] [--seed S] [--out DIR] [--compare DIR]
//! cargo run --release --example wer -- --hyp HYP.tsv LIST.tsv [--compare DIR]
//! ```
//!
//! A list has one utterance per line, `id<TAB>path to a 16 kHz WAV<TAB>reference text`.
//! `--limit` scores a fixed random subset of each list (`--seed` picks it). `--out` writes the
//! raw hypotheses to `DIR/<list name>.tsv` as `id<TAB>text` and resumes from that file if it
//! exists. `--hyp` scores a hypothesis file of that format without running a model. With
//! `--compare DIR`, either form also reports how often the transcripts equal those in
//! `DIR/<list name>.tsv`.

mod normalize;
mod score;
#[cfg(test)]
mod vectors;

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use notate::{Options, Parakeet};
use score::Errors;

struct Utterance {
    id: String,
    path: PathBuf,
    reference: String,
}

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
    let mut limit = None;
    let mut seed = 0u64;
    let mut out = None;
    let mut hyp = None;
    let mut compare = None;
    let mut positional: Vec<PathBuf> = Vec::new();
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--threads" => {
                options.threads = value("--threads")?
                    .parse()
                    .map_err(|_| "--threads takes a number")?
            }
            "--limit" => {
                limit = Some(
                    value("--limit")?
                        .parse()
                        .map_err(|_| "--limit takes a number")?,
                )
            }
            "--seed" => {
                seed = value("--seed")?
                    .parse()
                    .map_err(|_| "--seed takes a number")?
            }
            "--out" => out = Some(PathBuf::from(value("--out")?)),
            "--hyp" => hyp = Some(PathBuf::from(value("--hyp")?)),
            "--compare" => compare = Some(PathBuf::from(value("--compare")?)),
            _ => positional.push(arg.into()),
        }
    }

    if let Some(hyp) = hyp {
        let [list] = positional.as_slice() else {
            return Err("--hyp takes one list".into());
        };
        let utterances = select(read_list(list)?, limit, seed);
        let hyps = read_hyps(&hyp)?;
        let pairs: Vec<(&Utterance, &str)> = utterances
            .iter()
            .filter_map(|u| hyps.get(&u.id).map(|h| (u, h.as_str())))
            .collect();
        if pairs.len() < utterances.len() {
            eprintln!(
                "{} of {} utterances have no hypothesis and are skipped.",
                utterances.len() - pairs.len(),
                utterances.len()
            );
        }
        report(&name_of(list), &pairs, None);
        if let Some(dir) = &compare {
            let other = read_hyps(&dir.join(format!("{}.tsv", name_of(list))))?;
            agreement(&pairs, &other);
        }
        return Ok(());
    }

    if positional.len() < 2 {
        return Err("Usage: wer MODEL_DIR LIST.tsv... [--threads N] [--limit N] [--seed S] [--out DIR] [--compare DIR], or wer --hyp HYP.tsv LIST.tsv".into());
    }
    let model_dir = positional.remove(0);
    let start = Instant::now();
    let model = Parakeet::load(&model_dir, &options).map_err(|e| e.to_string())?;
    eprintln!(
        "Loaded in {:.1} s, {} threads, {} kernels.",
        start.elapsed().as_secs_f64(),
        model.threads(),
        model.isa().name()
    );

    for list in &positional {
        let name = name_of(list);
        let utterances = select(read_list(list)?, limit, seed);
        let out_path = out.as_ref().map(|dir| dir.join(format!("{name}.tsv")));
        let mut hyps = match &out_path {
            Some(p) if p.exists() => read_hyps(p)?,
            _ => HashMap::new(),
        };
        let mut out_file = match &out_path {
            Some(p) => {
                std::fs::create_dir_all(p.parent().unwrap()).map_err(|e| e.to_string())?;
                Some(
                    std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(p)
                        .map_err(|e| format!("{}: {e}", p.display()))?,
                )
            }
            None => None,
        };
        let mut compute = Duration::ZERO;
        let mut audio_seconds = 0.0;
        let todo: Vec<&Utterance> = utterances
            .iter()
            .filter(|u| !hyps.contains_key(&u.id))
            .collect();
        for (done, u) in todo.iter().enumerate() {
            let audio = notate::wav::read(&u.path).map_err(|e| e.to_string())?;
            if audio.sample_rate != 16000 {
                return Err(format!(
                    "{} is {} Hz, not 16 kHz.",
                    u.path.display(),
                    audio.sample_rate
                ));
            }
            let start = Instant::now();
            let text = model
                .transcribe(&audio.samples)
                .map_err(|e| format!("{}: {e}", u.id))?
                .text;
            compute += start.elapsed();
            audio_seconds += audio.seconds();
            if let Some(f) = &mut out_file {
                writeln!(f, "{}\t{}", u.id, text.replace(['\t', '\n'], " "))
                    .map_err(|e| e.to_string())?;
            }
            hyps.insert(u.id.clone(), text);
            if (done + 1) % 200 == 0 {
                eprintln!(
                    "{name}: {} of {}, real-time factor {:.4}",
                    done + 1,
                    todo.len(),
                    compute.as_secs_f64() / audio_seconds
                );
            }
        }
        let pairs: Vec<(&Utterance, &str)> = utterances
            .iter()
            .map(|u| (u, hyps[&u.id].as_str()))
            .collect();
        let speed = (audio_seconds > 0.0).then(|| (compute, audio_seconds, model.threads()));
        report(&name, &pairs, speed);
        if let Some(dir) = &compare {
            let other = read_hyps(&dir.join(format!("{name}.tsv")))?;
            agreement(&pairs, &other);
        }
    }
    Ok(())
}

fn name_of(list: &Path) -> String {
    list.file_stem()
        .map_or_else(|| "list".into(), |s| s.to_string_lossy().into_owned())
}

fn read_list(path: &Path) -> Result<Vec<Utterance>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let mut parts = line.splitn(3, '\t');
            match (parts.next(), parts.next(), parts.next()) {
                (Some(id), Some(p), Some(reference)) => Ok(Utterance {
                    id: id.into(),
                    path: p.into(),
                    reference: reference.into(),
                }),
                _ => Err(format!(
                    "{}: a line doesn't have three tab-separated fields",
                    path.display()
                )),
            }
        })
        .collect()
}

fn read_hyps(path: &Path) -> Result<HashMap<String, String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(text
        .lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            let (id, hyp) = line.split_once('\t').unwrap_or((line, ""));
            (id.to_string(), hyp.to_string())
        })
        .collect())
}

/// A fixed random subset of `limit` utterances (all of them without a limit), in id order.
fn select(mut utterances: Vec<Utterance>, limit: Option<usize>, seed: u64) -> Vec<Utterance> {
    utterances.sort_by(|a, b| a.id.cmp(&b.id));
    if let Some(limit) = limit.filter(|&l| l < utterances.len()) {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for i in (1..utterances.len()).rev() {
            let j = (next() % (i as u64 + 1)) as usize;
            utterances.swap(i, j);
        }
        utterances.truncate(limit);
        utterances.sort_by(|a, b| a.id.cmp(&b.id));
    }
    utterances
}

fn report(name: &str, pairs: &[(&Utterance, &str)], speed: Option<(Duration, f64, usize)>) {
    let normalized: Vec<(String, String)> = pairs
        .iter()
        .map(|(u, hyp)| {
            (
                normalize::normalize(&u.reference),
                normalize::normalize(hyp),
            )
        })
        .collect();
    let scored = normalized
        .iter()
        .filter(|(r, _)| score::is_scored(r))
        .count();
    let total: Errors =
        score::corpus_errors(normalized.iter().map(|(r, h)| (r.as_str(), h.as_str())));
    println!(
        "{name}: WER {:.2}% over {scored} utterances and {} words (substitutions {}, deletions {}, insertions {})",
        100.0 * total.wer(),
        total.reference_words,
        total.substitutions,
        total.deletions,
        total.insertions
    );
    if let Some((compute, audio, threads)) = speed {
        println!(
            "{name}: {:.2} h of audio in {:.0} s at {threads} threads, real-time factor {:.4}",
            audio / 3600.0,
            compute.as_secs_f64(),
            compute.as_secs_f64() / audio
        );
    }
}

fn agreement(pairs: &[(&Utterance, &str)], other: &HashMap<String, String>) {
    let (mut both, mut same, mut same_normalized) = (0, 0, 0);
    for (u, hyp) in pairs {
        let Some(theirs) = other.get(&u.id) else {
            continue;
        };
        both += 1;
        same += (theirs.trim() == hyp.trim()) as usize;
        same_normalized += (normalize::normalize(theirs) == normalize::normalize(hyp)) as usize;
    }
    println!(
        "Compared with {both} transcripts of the other system, {same} are identical and {same_normalized} identical after normalization."
    );
}
