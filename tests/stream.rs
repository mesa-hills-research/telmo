//! Streams the golden clips in steps of half a second and checks that the settled text matches
//! the offline transcript. The tests need the model files, so they are ignored by default.
//!
//! ```text
//! PARAKEET_TDT_DIR=... GRANITE_SPEECH_DIR=... cargo test --release --test stream -- --include-ignored
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use telmo::json::{self, Value};
use telmo::stream::{
    Event, PassConfig, PassMode, SAMPLE_RATE, Stream, StreamConfig, Streamer, UtteranceStream,
};
use telmo::{Model, Options, SpecialWords};

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
            let audio = telmo::wav::read(&dir.join(field("wav"))).unwrap();
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
fn stream(
    model: &Model,
    samples: &[f32],
    step: f64,
    words: &SpecialWords,
) -> (String, usize, usize) {
    let config = PassConfig::default();
    let mut s = UtteranceStream::with_special_words(model, words);
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

fn check(var: &str, special: &SpecialWords) {
    let model = model(var);
    let (mut errors, mut total) = (0, 0);
    for (id, samples) in &clips() {
        let offline = model.transcribe_biased(samples, special).unwrap().text;
        let (streamed, changed, passes) = stream(&model, samples, 0.5, special);
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

/// The golden clips' rarer words, and words that don't occur in them.
fn special_words() -> SpecialWords {
    {
        let mut words = SpecialWords::new();
        for word in [
            "Southey",
            "Montesquieu",
            "Bertie",
            "Hamburg",
            "kubectl",
            "nginx",
            "serde",
            "tokio",
            "Fauchelevent",
        ] {
            words.add(word).unwrap();
        }
        words
    }
}

#[test]
#[ignore = "needs the model files"]
fn parakeet_streams_the_golden_clips_like_offline() {
    check("PARAKEET_TDT_DIR", &SpecialWords::new());
}

#[test]
#[ignore = "needs the model files"]
fn granite_streams_the_golden_clips_like_offline() {
    check("GRANITE_SPEECH_DIR", &SpecialWords::new());
}

#[test]
#[ignore = "needs the model files"]
fn parakeet_streams_with_special_words_like_offline() {
    check("PARAKEET_TDT_DIR", &special_words());
}

#[test]
#[ignore = "needs the model files"]
fn granite_streams_with_special_words_like_offline() {
    check("GRANITE_SPEECH_DIR", &special_words());
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
    (telmo::resample::resample(&audio, 16000, 48000), clips)
}

/// The settled text and the number of utterances in a stream's events.
fn settled(events: &[Event]) -> (String, usize) {
    let mut text = String::new();
    let mut ends = 0;
    for e in events {
        match e {
            Event::Settled { text: t, .. } => text.push_str(t),
            Event::UtteranceEnd { .. } => ends += 1,
            _ => {}
        }
    }
    (text, ends)
}

fn check_session(var: &str, special: &SpecialWords) {
    let model = Arc::new(model(var));
    let (audio, clips) = session();
    let offline: Vec<String> = clips
        .iter()
        .map(|c| model.transcribe_biased(c, special).unwrap().text)
        .collect();
    let offline = words(&offline.join(" "));
    let config = StreamConfig {
        input_rate: 48000,
        ..StreamConfig::default()
    };

    // The synchronous core, with a pass after every half second of audio.
    let mut streamer = Streamer::new(model.clone(), config.clone());
    streamer.set_special_words(special);
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
    // A sentence end that closed an utterance waits as tentative text until the next one's
    // first words settle, and the last one settles when the stream finishes.
    let held = |e: &&Event| matches!(e, Event::Tentative { text, .. } if [".", "?", "!"].contains(&text.as_str()));
    assert!(events.iter().any(|e| held(&e)), "no sentence end was held");
    assert!(text.trim_end().ends_with(['.', '?', '!']), "{text}");
    // The streamed audio carries noise, a round trip through 48 kHz and the quiet around each
    // utterance, so a few words come out differently from the clean clips. A lost or repeated
    // sentence would differ by far more.
    assert!(e * 8 <= offline.len());

    // The same audio through a stream on its own thread, as fast as it takes it.
    let log: Arc<Mutex<Vec<Event>>> = Arc::default();
    let sink = log.clone();
    let stream = Stream::start(model, config, move |e| sink.lock().unwrap().push(e));
    stream.set_special_words(special.clone());
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
    check_session("PARAKEET_TDT_DIR", &SpecialWords::new());
}

#[test]
#[ignore = "needs the model files"]
fn granite_streams_a_session_of_utterances() {
    check_session("GRANITE_SPEECH_DIR", &SpecialWords::new());
}

#[test]
#[ignore = "needs the model files"]
fn parakeet_streams_a_session_with_special_words() {
    check_session("PARAKEET_TDT_DIR", &special_words());
}

#[test]
#[ignore = "needs the model files"]
fn granite_streams_a_session_with_special_words() {
    check_session("GRANITE_SPEECH_DIR", &special_words());
}

/// Streams a golden clip's sentence with a pause of 1 s put in after the words `before`, at the
/// quietest 10 ms between them and the next word as offline transcription places them, and
/// returns the settled text.
fn split_by_a_pause(var: &str, clip: &str, before: &str) -> String {
    split_by_a_pause_with(var, var, clip, before, None, None)
}

/// [`split_by_a_pause`] with the words placed by the model `timing` names (Granite's CTC
/// frames fall at the words' starts, where Parakeet's come late in a word) and special `words`
/// set on the stream. With `again`, the audio after the pause starts over from the word after
/// `again`, so the words from there to the pause are said twice ("...and he recalled the | he
/// recalled the words").
fn split_by_a_pause_with(
    var: &str,
    timing: &str,
    clip: &str,
    before: &str,
    again: Option<&str>,
    words: Option<&SpecialWords>,
) -> String {
    let model = Arc::new(model(var));
    let (_, samples) = clips().into_iter().find(|(id, _)| id == clip).unwrap();
    let timing = if timing == var {
        model.clone()
    } else {
        Arc::new(self::model(timing))
    };
    let tokens = timing.transcribe(&samples).unwrap().tokens;
    let tok = timing.tokenizer();
    let ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
    let token_after = |text: &str| {
        (1..tokens.len())
            .find(|&i| {
                (tok.decode(&ids[..i]).trim_end().to_lowercase()).ends_with(&text.to_lowercase())
            })
            .expect("the words are heard")
    };
    // The quietest 10 ms between the token before `at` and `at`.
    let quietest_before = |at: usize| {
        let frame = 1280;
        let from = (tokens[at - 1].frame + tokens[at - 1].duration.max(1)) as usize * frame;
        let to = (tokens[at].frame as usize + 1) * frame;
        (from..to.max(from + 160).min(samples.len() - 160))
            .step_by(160)
            .min_by(|&a, &b| {
                let energy = |s: usize| samples[s..s + 160].iter().map(|x| x * x).sum::<f32>();
                energy(a).total_cmp(&energy(b))
            })
            .unwrap()
    };
    let quietest = quietest_before(token_after(before));
    let resume = again.map_or(quietest, |again| quietest_before(token_after(again)));
    let mut audio = vec![0f32; SAMPLE_RATE / 2];
    audio.extend_from_slice(&samples[..quietest]);
    audio.extend(std::iter::repeat_n(0.0, SAMPLE_RATE));
    audio.extend_from_slice(&samples[resume..]);
    audio.extend(std::iter::repeat_n(0.0, SAMPLE_RATE));
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    for s in &mut audio {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        *s += ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.002;
    }
    let mut streamer = Streamer::new(model, StreamConfig::default());
    if let Some(words) = words {
        streamer.set_special_words(words);
    }
    let mut events = Vec::new();
    for piece in audio.chunks(SAMPLE_RATE / 2) {
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
    eprintln!("{ends} utterances: {text}");
    assert!(ends >= 2, "the pause ends an utterance");
    text
}

#[test]
#[ignore = "needs the model files"]
fn granite_keeps_a_sentence_end_across_a_short_pause() {
    // "... Good night, husband. The music came nearer, ..."
    let text = split_by_a_pause("GRANITE_SPEECH_DIR", "clean-joined-40s", "husband");
    assert!(text.contains("husband. The music"), "{text}");
}

/// Whether `text` goes on from `before` to `after`, with no sentence end between them and
/// `after` in lower case.
fn goes_on(text: &str, before: &str, after: &str) -> bool {
    let lower = text.to_lowercase();
    let Some(from) = lower.find(before).map(|i| i + before.len()) else {
        return false;
    };
    let Some(to) = lower[from..].find(after).map(|i| from + i) else {
        return false;
    };
    !text[from..to].contains(['.', '?', '!']) && text[to..].starts_with(after)
}

#[test]
#[ignore = "needs the model files"]
fn granite_keeps_a_sentence_going_across_a_short_pause() {
    // "... and he recalled the words, the words of Shelley's fragment ..."
    let text = split_by_a_pause("GRANITE_SPEECH_DIR", "clean-joined-40s", "recalled");
    assert!(goes_on(&text, "recalled", "the words"), "{text}");
}

#[test]
#[ignore = "needs the model files"]
fn parakeet_keeps_a_sentence_going_across_a_short_pause() {
    // "Find some Cresswells there. Big plantations, rated at two hundred and fifty thousand ..."
    let text = split_by_a_pause("PARAKEET_TDT_DIR", "clean-1995-1826-0007", "plantations,");
    assert!(goes_on(&text, "plantations", "rated"), "{text}");
}

/// Special words that leave out the hand-written cleanup's disfluencies.
fn hand_written_cleanup() -> SpecialWords {
    let mut words = SpecialWords::new();
    words.set_disfluencies(Some(Arc::new(telmo::disfluency::Tagger::hand_written())));
    words
}

/// Words said again after a short pause are a restart. The sentence goes on, and with the
/// cleanup the copy goes.
fn check_restart_across_a_short_pause(var: &str) {
    // "... and he recalled the | he recalled the words, the words of Shelley's fragment ..."
    let (clip, timing) = ("clean-joined-40s", "GRANITE_SPEECH_DIR");
    let text = split_by_a_pause_with(var, timing, clip, "recalled the", Some("nearer, and"), None);
    assert!(
        goes_on(&text, "recalled the", "he recalled the words"),
        "{text}"
    );
    let cleanup = hand_written_cleanup();
    let text = split_by_a_pause_with(
        var,
        timing,
        clip,
        "recalled the",
        Some("nearer, and"),
        Some(&cleanup),
    );
    let lower = text.to_lowercase();
    assert!(
        lower.contains("and he recalled the words") && lower.matches("recalled").count() == 1,
        "{text}"
    );
}

#[test]
#[ignore = "needs the model files"]
fn granite_cuts_a_restart_across_a_short_pause() {
    check_restart_across_a_short_pause("GRANITE_SPEECH_DIR");
}

#[test]
#[ignore = "needs the model files"]
fn parakeet_cuts_a_restart_across_a_short_pause() {
    check_restart_across_a_short_pause("PARAKEET_TDT_DIR");
}

/// Granite's probability of a sentence end at this pause spliced into read speech is 0.06 to
/// 0.19, far above the 0.003 it takes to keep the mark, so the period stays where the sentence
/// goes on. A threshold that joined it would also join real sentences.
#[test]
#[ignore = "needs the model files, and shows the sentence-end threshold's trade-off"]
fn granite_keeps_a_period_at_a_pause_spliced_after_a_phrase() {
    // "I saw at the Hamburg Museum the skeleton of one of these creatures, thirty feet in length."
    let text = split_by_a_pause("GRANITE_SPEECH_DIR", "clean-260-123286-0018", "Museum");
    assert!(text.contains("Museum. The skeleton"), "{text}");
}
