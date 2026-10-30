//! Streams the golden clips in steps of half a second and checks that the settled text matches
//! the offline transcript. The tests need the model files, so they are ignored by default.
//!
//! ```text
//! PARAKEET_TDT_DIR=... GRANITE_SPEECH_DIR=... cargo test --release --test stream -- --include-ignored
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use notate::json::{self, Value};
use notate::stream::{
    Event, PassConfig, PassMode, SAMPLE_RATE, Stream, StreamConfig, Streamer, UtteranceStream,
};
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
        let out = s.pass(model, &config, PassMode::Tentative).unwrap();
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
    let out = s.pass(model, &config, PassMode::Final).unwrap();
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

/// The golden clips under 10 s, joined with 1.5 s pauses over quiet noise, at 48 kHz.
fn session() -> (Vec<f32>, Vec<Vec<f32>>) {
    let clips: Vec<Vec<f32>> = clips()
        .into_iter()
        .map(|(_, s)| s)
        .filter(|s| s.len() < 10 * SAMPLE_RATE)
        .collect();
    let mut audio = vec![0f32; SAMPLE_RATE];
    for clip in &clips {
        audio.extend_from_slice(clip);
        audio.extend(std::iter::repeat_n(0.0, 3 * SAMPLE_RATE / 2));
    }
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    for s in &mut audio {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        *s += ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.002;
    }
    (notate::resample::resample(&audio, 16000, 48000), clips)
}

/// The settled text and the number of utterances in a stream's events.
fn settled(events: &[Event]) -> (String, usize) {
    let mut text = String::new();
    let mut ends = 0;
    for e in events {
        match e {
            Event::Settled { text: t } => text.push_str(t),
            Event::UtteranceEnd { .. } => ends += 1,
            _ => {}
        }
    }
    (text, ends)
}

fn check_session(var: &str) {
    let model = Arc::new(model(var));
    let (audio, clips) = session();
    let offline: Vec<String> = clips
        .iter()
        .map(|c| model.transcribe(c).unwrap().text)
        .collect();
    let offline = words(&offline.join(" "));
    let config = StreamConfig {
        input_rate: 48000,
        ..StreamConfig::default()
    };

    // The synchronous core, with a pass after every half second of audio.
    let mut streamer = Streamer::new(model.clone(), config.clone());
    let mut events = Vec::new();
    for piece in audio.chunks(24000) {
        events.extend(streamer.push(piece));
        while streamer.wants_final() || streamer.has_new_audio() {
            events.extend(streamer.pass().unwrap());
            if !streamer.wants_final() {
                break;
            }
        }
    }
    events.extend(streamer.finish().unwrap());
    let (text, ends) = settled(&events);
    let e = distance(&offline, &words(&text));
    eprintln!(
        "{ends} utterances, {e} of {} words differ\n  {text}",
        offline.len()
    );
    assert_eq!(ends, clips.len(), "one utterance per clip");
    // The streamed audio carries noise, a round trip through 48 kHz and the quiet around each
    // utterance, so a few words come out differently from the clean clips. A lost or repeated
    // sentence would differ by far more.
    assert!(e * 8 <= offline.len());

    // The same audio through a stream on its own thread, as fast as it takes it.
    let log: Arc<Mutex<Vec<Event>>> = Arc::default();
    let sink = log.clone();
    let stream = Stream::start(model, config, move |e| sink.lock().unwrap().push(e));
    for piece in audio.chunks(960) {
        stream.push(piece);
    }
    let stats = stream.finish().unwrap();
    let (text, ends) = settled(&log.lock().unwrap());
    let e = distance(&offline, &words(&text));
    eprintln!(
        "threaded: {} passes, {ends} utterances, {e} words differ",
        stats.passes
    );
    assert_eq!(ends, clips.len());
    assert!(e * 8 <= offline.len());
}

#[test]
#[ignore = "needs the model files"]
fn parakeet_streams_a_session_of_utterances() {
    check_session("PARAKEET_TDT_DIR");
}

#[test]
#[ignore = "needs the model files"]
fn granite_streams_a_session_of_utterances() {
    check_session("GRANITE_SPEECH_DIR");
}
