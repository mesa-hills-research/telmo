//! Checks streaming transcription with Granite Speech against the offline encoder.
//!
//! The tests need the model's files, so they are ignored by default. Run them with
//!
//! ```text
//! GRANITE_SPEECH_DIR=/path/to/granite-speech-5.0-470m-turboctc cargo test --release --test granite_stream -- --include-ignored --nocapture
//! ```
//!
//! `long_joined_audio_streams_close_to_offline` also needs a LibriSpeech list
//! (`id<TAB>wav<TAB>text`) in `GRANITE_STREAM_LIST`, whose consecutive utterances it joins into
//! 30 to 90 s of audio.

use std::path::{Path, PathBuf};

use notate::granite::Granite;
use notate::json::{self, Value};
use notate::parakeet::Timings;
use notate::{GraniteStream, Options, Precision};

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/granite-speech-5.0-470m-turboctc")
}

fn model_in(precision: Precision) -> Granite {
    let dir = std::env::var_os("GRANITE_SPEECH_DIR")
        .expect("set GRANITE_SPEECH_DIR to the model's directory to run this test");
    let options = Options {
        threads: 6,
        isa: None,
        precision,
    };
    Granite::load(Path::new(&dir), &options).expect("the model loads")
}

/// The golden clips, by id and samples.
fn clips() -> Vec<(String, Vec<f32>)> {
    let dir = golden_dir();
    let manifest = std::fs::read(dir.join("manifest.json")).expect("the goldens' manifest.json");
    let manifest = json::parse(&manifest).expect("manifest.json parses");
    let list = manifest
        .get("clips")
        .and_then(Value::as_array)
        .expect("manifest.json lists clips");
    list.iter()
        .map(|clip| {
            let field = |key: &str| clip.get(key).and_then(Value::as_str).unwrap().to_string();
            let audio = notate::wav::read(&dir.join(field("wav"))).expect("the clip's WAV");
            assert_eq!(audio.sample_rate, 16000);
            (field("id"), audio.samples)
        })
        .collect()
}

/// The offline encoder's output, every block's output (the middle one with the
/// self-conditioning injection added) and the frame ids.
struct Offline {
    layers: Vec<Vec<f32>>,
    encoded: Vec<f32>,
    ids: Vec<u32>,
    text: String,
}

fn offline(model: &Granite, samples: &[f32]) -> Offline {
    let features = model.features(samples).unwrap();
    let mut layers: Vec<Vec<f32>> = Vec::new();
    let encoded = model
        .encode(
            &features.data,
            features.frames,
            &mut Timings::default(),
            &mut |name, _, x| {
                if name.starts_with("layer.") {
                    layers.push(x.to_vec());
                } else if name == "selfcond_injection" {
                    let last = layers.last_mut().unwrap();
                    for (v, &a) in last.iter_mut().zip(x) {
                        *v += a;
                    }
                }
            },
        )
        .unwrap();
    let ids = model.frame_ids(&encoded);
    let tokens = notate::granite::ctc_tokens(&ids, model.config().blank);
    let text = model
        .tokenizer()
        .decode(&tokens.iter().map(|t| t.id).collect::<Vec<_>>());
    Offline {
        layers,
        encoded: encoded.data,
        ids,
        text,
    }
}

fn relative_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        num += (g as f64 - w as f64).powi(2);
        den += (w as f64).powi(2);
    }
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
}

/// Pushes `samples` in pieces of 0.25 to 1 s, a fixed sequence from `seed`, with an update after
/// each, the last one finished.
fn stream_in_pieces(model: &Granite, samples: &[f32], seed: u64, lag: usize) -> GraniteStream {
    let mut stream = GraniteStream::new(model);
    stream.set_freeze_lag(lag);
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut at = 0;
    while at < samples.len() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let piece = 4000 + (state % 12001) as usize;
        let end = (at + piece).min(samples.len());
        stream.push(&samples[at..end]);
        at = end;
        let before = stream.frame_ids().to_vec();
        stream.update(model, at == samples.len(), &mut Timings::default());
        // Frames before the first changed one keep their ids.
        let first = stream.first_changed();
        assert!(first <= stream.frames() && first <= before.len());
        assert_eq!(stream.frame_ids()[..first], before[..first]);
    }
    stream
}

/// How a streamed utterance compares with the offline pass.
struct Comparison {
    frames: usize,
    same_ids: usize,
    same_text: bool,
    /// Relative L2 of each block's output against offline, over the frames the stream froze
    /// (all frames for the strided blocks).
    frozen_l2: Vec<f64>,
    /// Relative L2 of the encoder output over all frames.
    encoder_l2: f64,
}

fn compare(model: &Granite, stream: &GraniteStream, reference: &Offline) -> Comparison {
    let d = model.config().d_model;
    let frames = reference.ids.len();
    assert_eq!(stream.frames(), frames, "frame count");
    let frozen = stream.frozen();
    let frozen_l2 = (0..model.config().layers)
        .map(|layer| {
            let got = stream.layer_output(layer);
            let want = &reference.layers[layer];
            assert_eq!(got.len(), want.len(), "block {layer} frames");
            let rows = if layer < model.config().subsample_layers.len() {
                want.len() / d
            } else {
                frozen
            };
            relative_l2(&got[..rows * d], &want[..rows * d])
        })
        .collect();
    let same_ids = stream
        .frame_ids()
        .iter()
        .zip(&reference.ids)
        .filter(|(a, b)| a == b)
        .count();
    Comparison {
        frames,
        same_ids,
        same_text: stream.text(model) == reference.text,
        frozen_l2,
        encoder_l2: relative_l2(
            stream.encoder_output(model, 0, frames),
            &reference.encoded,
        ),
    }
}

fn describe(label: &str, c: &Comparison) -> String {
    let worst = c
        .frozen_l2
        .iter()
        .enumerate()
        .skip(2)
        .fold((0, 0f64), |b, (i, &e)| if e > b.1 { (i, e) } else { b });
    format!(
        "{label}: {} of {} frame ids equal ({:.2}%), text {}, encoder L2 {:.2e}, strided blocks L2 {:.2e} {:.2e}, frozen blocks worst L2 {:.2e} (block {}), block 15 {:.2e}",
        c.same_ids,
        c.frames,
        100.0 * c.same_ids as f64 / c.frames.max(1) as f64,
        if c.same_text { "equal" } else { "differs" },
        c.encoder_l2,
        c.frozen_l2[0],
        c.frozen_l2[1],
        worst.1,
        worst.0,
        c.frozen_l2[c.frozen_l2.len() - 1],
    )
}

#[test]
#[ignore = "needs the model files, see the top of this file"]
fn one_finished_update_equals_the_offline_encoder() {
    for precision in [Precision::F32, Precision::Int8] {
        let model = model_in(precision);
        for (id, samples) in clips() {
            let reference = offline(&model, &samples);
            let mut stream = GraniteStream::new(&model);
            stream.push(&samples);
            stream.update(&model, true, &mut Timings::default());
            let frames = reference.ids.len();
            assert_eq!(stream.frames(), frames, "{id}");
            assert_eq!(stream.first_changed(), 0, "{id}");
            let same = stream
                .encoder_output(&model, 0, frames)
                .iter()
                .zip(&reference.encoded)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "{id} {precision}: the encoder output differs");
            assert_eq!(stream.frame_ids(), &reference.ids[..], "{id} {precision}");
        }
    }
}

#[test]
#[ignore = "needs the model files, see the top of this file"]
fn streaming_in_pieces_stays_close_to_offline() {
    let model = model_in(Precision::F32);
    let lags: Vec<usize> = std::env::var("GRANITE_STREAM_LAGS")
        .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_default();
    for (id, samples) in clips() {
        let reference = offline(&model, &samples);
        for &lag in &lags {
            let stream = stream_in_pieces(&model, &samples, 1, lag);
            let c = compare(&model, &stream, &reference);
            eprintln!("{}", describe(&format!("{id} lag {lag}"), &c));
        }
        let stream = stream_in_pieces(&model, &samples, 1, 0);
        let c = compare(&model, &stream, &reference);
        eprintln!("{}", describe(&id, &c));
        if stream.frozen() == 0 {
            // Within one 10.24 s block nothing freezes, and only the running peak of the
            // features differs from offline.
            assert_eq!(stream.frame_ids(), &reference.ids[..], "{id}");
        }
    }
}

#[test]
#[ignore = "needs the model files and GRANITE_STREAM_LIST, see the top of this file"]
fn long_joined_audio_streams_close_to_offline() {
    let list = std::env::var_os("GRANITE_STREAM_LIST")
        .expect("set GRANITE_STREAM_LIST to a LibriSpeech list");
    let text = std::fs::read_to_string(&list).unwrap();
    let mut rows: Vec<(&str, &str)> = text
        .lines()
        .filter_map(|l| {
            let mut f = l.split('\t');
            Some((f.next()?, f.next()?))
        })
        .collect();
    rows.sort();
    let model = model_in(Precision::F32);
    let mut next = 0;
    for (n, seconds) in [30usize, 45, 60, 75, 90].into_iter().enumerate() {
        let mut samples = Vec::new();
        while samples.len() < seconds * 16000 {
            let audio = notate::wav::read(Path::new(rows[next].1)).unwrap();
            samples.extend_from_slice(&audio.samples);
            next += 1;
        }
        samples.truncate(seconds * 16000);
        let reference = offline(&model, &samples);
        for lag in [0, 16] {
            let stream = stream_in_pieces(&model, &samples, n as u64 + 7, lag);
            let c = compare(&model, &stream, &reference);
            eprintln!("{}", describe(&format!("{seconds} s, lag {lag}"), &c));
        }
    }
}
