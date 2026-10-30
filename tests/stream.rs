//! Streams the golden clips in steps of half a second and checks that the settled text matches
//! the offline transcript. The tests need the model files, so they are ignored by default.
//!
//! ```text
//! PARAKEET_TDT_DIR=... GRANITE_SPEECH_DIR=... cargo test --release --test stream -- --include-ignored
//! ```

use std::path::{Path, PathBuf};

use notate::json::{self, Value};
use notate::stream::{PassConfig, SAMPLE_RATE, UtteranceStream};
use notate::{Model, Options};

fn clips() -> Vec<(String, Vec<f32>)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/parakeet-tdt-0.6b-v3");
    let manifest = json::parse(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    manifest
        .get("clips")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|c| {
            let field = |k: &str| c.get(k).and_then(Value::as_str).unwrap().to_string();
            let audio = notate::wav::read(&dir.join(field("wav"))).unwrap();
            (field("id"), audio.samples)
        })
        .collect()
}

fn model(var: &str) -> Model {
    let dir: PathBuf = std::env::var_os(var)
        .unwrap_or_else(|| panic!("set {var} to the model's directory"))
        .into();
    Model::load(&dir, &Options::default()).expect("the model loads")
}

/// Streams `samples` in steps of `step` seconds with a pass after each, then a final pass.
/// Returns the settled text and how many tentative words were replaced before settling.
fn stream(model: &Model, samples: &[f32], step: f64) -> (String, usize, usize) {
    let config = PassConfig::default();
    let mut s = UtteranceStream::new(model);
    let step = (step * SAMPLE_RATE as f64) as usize;
    let mut settled = String::new();
    let mut tentative = String::new();
    let mut changed = 0;
    for chunk in samples.chunks(step) {
        s.push(chunk);
        let out = s.pass(model, &config, false).unwrap();
        if std::env::var_os("STREAM_TRACE").is_some() {
            eprintln!(
                "  {:5.2} s, frame {}: settled [{}] tentative [{}]",
                s.seconds(),
                s.settled_frame(),
                out.settled,
                out.tentative
            );
        }
        settled.push_str(&out.settled);
        // Words shown before that are neither settled now nor still shown were replaced.
        let shown: Vec<&str> = tentative.split_whitespace().collect();
        let now: Vec<&str> = out
            .settled
            .split_whitespace()
            .chain(out.tentative.split_whitespace())
            .collect();
        changed += shown
            .iter()
            .zip(now.iter().chain(std::iter::repeat(&"")))
            .filter(|(a, b)| a != b)
            .count();
        tentative = out.tentative;
    }
    let out = s.pass(model, &config, true).unwrap();
    settled.push_str(&out.settled);
    (settled.trim_start().to_string(), changed, s.passes())
}

/// Lower-case words without punctuation.
fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Word-level edit distance.
fn distance(a: &[String], b: &[String]) -> usize {
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for x in a {
        let mut diag = row[0];
        row[0] += 1;
        for (j, y) in b.iter().enumerate() {
            let next = (row[j + 1] + 1)
                .min(row[j] + 1)
                .min(diag + (x != y) as usize);
            diag = row[j + 1];
            row[j + 1] = next;
        }
    }
    row[b.len()]
}

fn check(var: &str) {
    let model = model(var);
    let (mut errors, mut total) = (0, 0);
    for (id, samples) in &clips() {
        let offline = model.transcribe(samples).unwrap().text;
        let (streamed, changed, passes) = stream(&model, samples, 0.5);
        let (o, s) = (words(&offline), words(&streamed));
        let e = distance(&o, &s);
        eprintln!(
            "{id}: {passes} passes, {changed} tentative words replaced, {e} words differ\n  offline  {offline}\n  streamed {streamed}"
        );
        errors += e;
        total += o.len();
    }
    eprintln!("{errors} of {total} words differ from offline");
    // Streaming re-encodes windows with less context, which changes a few words.
    assert!(errors * 20 <= total, "{errors} of {total} words differ");
}

#[test]
#[ignore = "needs the model files"]
fn parakeet_streams_the_golden_clips_like_offline() {
    check("PARAKEET_TDT_DIR");
}

#[test]
#[ignore = "needs the model files"]
fn granite_streams_the_golden_clips_like_offline() {
    check("GRANITE_SPEECH_DIR");
}
