//! Compares Granite Speech with tensors saved from the reference implementation (transformers'
//! `GraniteSpeech5ForCTC`) on LibriSpeech and FLEURS clips.
//! `tests/goldens/granite-speech-5.0-470m-turboctc/README.md` says where they come from.
//!
//! The front-end test needs only the files in this repository. The others need the model's
//! files, so they are ignored by default. Run them with
//!
//! ```text
//! GRANITE_SPEECH_DIR=/path/to/granite-speech-5.0-470m-turboctc cargo test --release --test granite_goldens -- --include-ignored
//! ```

use std::path::{Path, PathBuf};

use notate::granite_features::GraniteFrontEnd;
use notate::json::{self, Value};
use notate::safetensors::SafeTensors;
use notate::tokenizer::Tokenizer;

/// Features may differ by this much. The reference computes its STFT and mel sums in f32.
const FEATURES_MAX_ABS: f32 = 5e-5;
const FEATURES_MEAN_ABS: f64 = 5e-7;

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/granite-speech-5.0-470m-turboctc")
}

fn model_dir() -> PathBuf {
    std::env::var_os("GRANITE_SPEECH_DIR")
        .expect("set GRANITE_SPEECH_DIR to the model's directory to run this test")
        .into()
}

struct Clip {
    id: String,
    samples: Vec<f32>,
    tensors: SafeTensors,
}

fn clips() -> Vec<Clip> {
    let dir = golden_dir();
    let manifest = std::fs::read(dir.join("manifest.json")).expect("the goldens' manifest.json");
    let manifest = json::parse(&manifest).expect("manifest.json parses");
    let list = manifest
        .get("clips")
        .and_then(Value::as_array)
        .expect("manifest.json lists clips");
    list.iter()
        .map(|clip| {
            let field = |key: &str| {
                clip.get(key)
                    .and_then(Value::as_str)
                    .unwrap_or_else(|| panic!("a clip has no {key}"))
                    .to_string()
            };
            let audio = notate::wav::read(&dir.join(field("wav"))).expect("the clip's WAV");
            assert_eq!(audio.sample_rate, 16000);
            let tensors = SafeTensors::from_bytes(
                std::fs::read(dir.join(field("tensors"))).expect("the clip's tensors"),
            )
            .expect("the clip's tensors parse");
            Clip {
                id: field("id"),
                samples: audio.samples,
                tensors,
            }
        })
        .collect()
}

fn f32s(st: &mut SafeTensors, name: &str) -> (Vec<usize>, Vec<f32>) {
    let shape = st
        .info(name)
        .unwrap_or_else(|| panic!("{name} is missing"))
        .shape
        .clone();
    (shape, st.read_f32(name, None).unwrap())
}

#[test]
fn features_match_the_reference() {
    let front_end = GraniteFrontEnd::new(80);
    let mut checked = 0;
    for mut clip in clips() {
        if clip.tensors.info("features").is_none() {
            continue;
        }
        checked += 1;
        let got = front_end.compute(&clip.samples).unwrap();
        let (shape, want) = f32s(&mut clip.tensors, "features");
        assert_eq!(
            (got.frames, got.n_mels),
            (shape[0], shape[1]),
            "{}: shape",
            clip.id
        );
        let errors: Vec<f32> = got
            .data
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .collect();
        let (at, max) = errors
            .iter()
            .enumerate()
            .fold((0, 0f32), |b, (i, &e)| if e > b.1 { (i, e) } else { b });
        let mean = errors.iter().map(|&e| e as f64).sum::<f64>() / errors.len() as f64;
        eprintln!(
            "{}: features max abs error {max:.2e} (row {}, column {}, value {:.4}), mean {mean:.2e}",
            clip.id,
            at / 320,
            at % 320,
            want[at]
        );
        assert!(
            mean <= FEATURES_MEAN_ABS,
            "{}: features differ by {mean} on average",
            clip.id
        );
        assert!(
            max <= FEATURES_MAX_ABS,
            "{}: features differ by up to {max}",
            clip.id
        );
    }
    assert!(checked >= 5, "only {checked} clips have features");
}

#[test]
#[ignore = "needs the model files, see the top of this file"]
fn the_tokenizer_decodes_like_the_reference() {
    let tokenizer = Tokenizer::load(&model_dir().join("tokenizer.json")).unwrap();
    let vectors =
        std::fs::read(golden_dir().join("tokenizer-vectors.json")).expect("tokenizer-vectors.json");
    let vectors = json::parse(&vectors).unwrap();
    let single = vectors.get("single").and_then(Value::as_array).unwrap();
    assert_eq!(single.len(), tokenizer.len());
    for (id, want) in single.iter().enumerate() {
        assert_eq!(
            tokenizer.decode(&[id as u32]),
            want.as_str().unwrap(),
            "id {id}"
        );
    }
    let sequences = vectors.get("sequences").and_then(Value::as_array).unwrap();
    assert!(sequences.len() >= 400);
    for sequence in sequences {
        let ids: Vec<u32> = sequence
            .get("ids")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .map(|i| i.as_u64().unwrap() as u32)
            .collect();
        let want = sequence.get("text").and_then(Value::as_str).unwrap();
        assert_eq!(tokenizer.decode(&ids), want, "ids {ids:?}");
    }
}
