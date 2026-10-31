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

use telmo::granite::{Granite, ctc_tokens};
use telmo::granite_features::GraniteFrontEnd;
use telmo::json::{self, Value};
use telmo::kernels::Isa;
use telmo::parakeet::Timings;
use telmo::safetensors::SafeTensors;
use telmo::tokenizer::Tokenizer;
use telmo::{Options, Precision, SpecialWords};

/// Features may differ by this much. The reference computes its STFT and mel sums in f32.
const FEATURES_MAX_ABS: f32 = 5e-5;
const FEATURES_MEAN_ABS: f64 = 5e-7;

/// Relative L2 error allowed for a block's output, growing with depth. The self-conditioning
/// injection follows block 7, and the logits the last block.
fn layer_tolerance(name: &str) -> f64 {
    let depth = match name {
        "selfcond_injection" => 8.0,
        "logits" => 16.0,
        _ => 1.0 + name.strip_prefix("layer.").unwrap().parse::<f64>().unwrap(),
    };
    1e-4 * (1.0 + depth / 8.0)
}

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
    text: String,
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
            let audio = telmo::wav::read(&dir.join(field("wav"))).expect("the clip's WAV");
            assert_eq!(audio.sample_rate, 16000);
            let tensors = SafeTensors::from_bytes(
                std::fs::read(dir.join(field("tensors"))).expect("the clip's tensors"),
            )
            .expect("the clip's tensors parse");
            Clip {
                id: field("id"),
                samples: audio.samples,
                tensors,
                text: field("text"),
            }
        })
        .collect()
}

fn model(isa: Option<Isa>, threads: usize) -> Granite {
    let options = Options {
        threads,
        isa,
        ..Options::default()
    };
    Granite::load(&model_dir(), &options).expect("the model loads")
}

fn model_in(precision: Precision) -> Granite {
    let options = Options {
        threads: 6,
        isa: None,
        precision,
    };
    let model = Granite::load(&model_dir(), &options).expect("the model loads");
    assert_eq!(model.precision(), Some(precision));
    model
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        ab += x as f64 * y as f64;
        aa += x as f64 * x as f64;
        bb += y as f64 * y as f64;
    }
    ab / (aa.sqrt() * bb.sqrt())
}

/// ||got - want|| / ||want||.
fn relative_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (mut num, mut den) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        num += (g as f64 - w as f64).powi(2);
        den += (w as f64).powi(2);
    }
    (num / den).sqrt()
}

fn i64s(st: &mut SafeTensors, name: &str) -> Vec<i64> {
    st.read_i64(name, None)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
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
    // The published package leaves the golden files out.
    if !golden_dir().exists() {
        eprintln!("The golden files aren't in this copy of the source, so this test is skipped.");
        return;
    }
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

#[test]
#[ignore = "needs the model files, see the top of this file"]
fn the_tokenizer_encodes_like_the_reference() {
    let tokenizer = Tokenizer::load(&model_dir().join("tokenizer.json")).unwrap();
    let vectors =
        std::fs::read(golden_dir().join("encode-vectors.json")).expect("encode-vectors.json");
    let vectors = json::parse(&vectors).unwrap();
    let vectors = vectors.get("vectors").and_then(Value::as_array).unwrap();
    assert!(vectors.len() >= 200);
    for vector in vectors {
        let text = vector.get("text").and_then(Value::as_str).unwrap();
        let want: Vec<u32> = vector
            .get("ids")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .map(|i| i.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(tokenizer.encode(text), want, "{text}");
    }
}

#[test]
#[ignore = "needs the model files, see the top of this file"]
fn special_words_leave_right_transcripts_as_they_are() {
    let model = model(None, 6);
    // The golden clips' rarer words, and words that don't occur in them.
    let words: SpecialWords = {
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
    };
    for clip in clips() {
        let plain = model.transcribe(&clip.samples).unwrap().text;
        let biased = model.transcribe_biased(&clip.samples, &words).unwrap().text;
        // A beam search may find a likelier path that differs in case or punctuation.
        let normalized = |text: &str| -> String {
            text.chars()
                .filter(|c| c.is_alphanumeric() || *c == ' ')
                .flat_map(char::to_lowercase)
                .collect()
        };
        assert_eq!(normalized(&biased), normalized(&plain), "{}", clip.id);
    }
}

/// Runs a clip through the encoder and the CTC head, checking every saved tensor.
fn check_clip(model: &Granite, clip: &mut Clip) {
    let id = clip.id.clone();
    let st = &mut clip.tensors;
    let d = model.config().d_model;
    let features = model.features(&clip.samples).unwrap();
    let mut errors: Vec<(String, f64)> = Vec::new();
    let mut compared = Vec::new();
    let encoded = model
        .encode(
            &features.data,
            features.frames,
            &mut Timings::default(),
            &mut |name, rows, x| {
                if st.info(name).is_none() {
                    return;
                }
                let (shape, want) = f32s(st, name);
                assert_eq!(shape, [rows, d], "{id}: {name} shape");
                let err = relative_l2(x, &want);
                eprintln!("{id}: {name} relative L2 error {err:.2e}");
                compared.push(name.to_string());
                if err > layer_tolerance(name) {
                    errors.push((name.to_string(), err));
                }
            },
        )
        .unwrap();
    let saved = [
        "layer.0",
        "layer.1",
        "layer.7",
        "selfcond_injection",
        "layer.15",
    ];
    for name in saved.iter().filter(|n| st.info(n).is_some()) {
        assert!(
            compared.iter().any(|c| c == name),
            "{id}: {name} wasn't reached"
        );
    }

    let vocab = model.config().vocab;
    let logits = model.ctc_logits(&encoded, 0, encoded.frames);
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .fold(0, |best, (i, &v)| if v > row[best] { i } else { best })
    };
    let ids: Vec<u32> = logits
        .chunks_exact(vocab)
        .map(|r| argmax(r) as u32)
        .collect();
    let want_ids = i64s(st, "ctc_ids");
    let (_, margins) = f32s(st, "logit_margin");
    let differ: Vec<(usize, f32)> = (0..want_ids.len())
        .filter(|&f| ids.get(f).map(|&i| i as i64) != Some(want_ids[f]))
        .map(|f| (f, margins[f]))
        .collect();
    assert_eq!(ids.len(), want_ids.len(), "{id}: CTC frames");
    assert!(
        differ.is_empty(),
        "{id}: the argmax differs at (frame, reference margin) {differ:?}"
    );
    let (_, want_max) = f32s(st, "logit_max");
    let got_max: Vec<f32> = logits
        .chunks_exact(vocab)
        .map(|r| r.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)))
        .collect();
    let err = relative_l2(&got_max, &want_max);
    eprintln!("{id}: largest logit per frame relative L2 error {err:.2e}");
    if err > layer_tolerance("logits") {
        errors.push(("logit_max".into(), err));
    }
    let frames = i64s(st, "logit_frames");
    let (_, want_logits) = f32s(st, "logits");
    for (k, &f) in frames.iter().enumerate() {
        let f = f as usize;
        let err = relative_l2(
            &logits[f * vocab..(f + 1) * vocab],
            &want_logits[k * vocab..(k + 1) * vocab],
        );
        eprintln!("{id}: logits of frame {f} relative L2 error {err:.2e}");
        if err > layer_tolerance("logits") {
            errors.push((format!("logits of frame {f}"), err));
        }
    }

    let tokens: Vec<i64> = ctc_tokens(&ids, model.config().blank)
        .iter()
        .map(|t| t.id as i64)
        .collect();
    assert_eq!(tokens, i64s(st, "tokens"), "{id}: tokens");
    let transcript = model.transcribe(&clip.samples).unwrap();
    let transcribed: Vec<i64> = transcript.tokens.iter().map(|t| t.id as i64).collect();
    assert_eq!(transcribed, tokens, "{id}: tokens of transcribe");
    assert_eq!(transcript.text, clip.text, "{id}: text");
    assert!(errors.is_empty(), "{id}: errors over tolerance {errors:?}");
}

#[test]
#[ignore = "needs the model files, see the top of this file"]
fn every_tensor_matches_the_reference() {
    let model = model(None, 6);
    for mut clip in clips() {
        check_clip(&model, &mut clip);
    }
}

#[test]
#[ignore = "needs the model files, see the top of this file"]
fn encoder_output_is_bit_identical_at_every_thread_count() {
    // The 40 s clip spans several attention blocks in every layer.
    let clip = clips().into_iter().max_by_key(|c| c.samples.len()).unwrap();
    let mut first: Option<Vec<f32>> = None;
    for threads in [1, 2, 3, 6] {
        let model = model(None, threads);
        let features = model.features(&clip.samples).unwrap();
        let encoded = model
            .encode(
                &features.data,
                features.frames,
                &mut Timings::default(),
                &mut |_, _, _| {},
            )
            .unwrap();
        match &first {
            None => first = Some(encoded.data),
            Some(f) => {
                let same = f
                    .iter()
                    .zip(&encoded.data)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "{threads} threads give a different encoder output");
            }
        }
    }
}

#[test]
#[ignore = "needs the model files, see the top of this file"]
fn the_scalar_kernels_give_the_same_text() {
    let model = model(Some(Isa::Scalar), 6);
    for clip in clips().iter().filter(|c| c.samples.len() < 16000 * 10) {
        let transcript = model.transcribe(&clip.samples).unwrap();
        assert_eq!(transcript.text, clip.text, "{}", clip.id);
    }
}

/// Runs every clip through a model with narrow weights, comparing each saved block's output by
/// cosine similarity and the CTC tokens with the reference. Returns the worst cosine, the clips
/// with identical tokens and the number of clips.
fn check_narrow(label: &str, model: &Granite) -> (f64, usize, usize) {
    let (mut worst, mut same, mut total) = (1f64, 0, 0);
    for mut clip in clips() {
        total += 1;
        let st = &mut clip.tensors;
        let features = model.features(&clip.samples).unwrap();
        let mut line = format!("{label} {}:", clip.id);
        let encoded = model
            .encode(
                &features.data,
                features.frames,
                &mut Timings::default(),
                &mut |name, _, x| {
                    if st.info(name).is_none() {
                        return;
                    }
                    let (_, want) = f32s(st, name);
                    let cos = cosine(x, &want);
                    line += &format!(" {name} {cos:.6}");
                    worst = worst.min(cos);
                },
            )
            .unwrap();
        let tokens: Vec<i64> = ctc_tokens(&model.frame_ids(&encoded), model.config().blank)
            .iter()
            .map(|t| t.id as i64)
            .collect();
        let identical = tokens == i64s(st, "tokens");
        same += identical as usize;
        eprintln!(
            "{line} tokens {}",
            if identical { "identical" } else { "differ" }
        );
        if !identical {
            let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            eprintln!(
                "  got  {}\n  want {}",
                model.tokenizer().decode(&ids),
                clip.text
            );
        }
    }
    eprintln!(
        "{label}: worst layer cosine {worst:.6}, identical tokens on {same} of {total} clips"
    );
    (worst, same, total)
}

#[test]
#[ignore = "needs the model files, see the top of this file"]
fn bf16_weights_give_the_same_output_as_f32() {
    // The checkpoint is bf16, so every weight matrix keeps its exact values in bf16.
    let clip = clips().into_iter().max_by_key(|c| c.samples.len()).unwrap();
    let outputs: Vec<Vec<f32>> = [Precision::F32, Precision::Bf16]
        .into_iter()
        .map(|precision| {
            let model = model_in(precision);
            let features = model.features(&clip.samples).unwrap();
            let mut timings = Timings::default();
            model
                .encode(
                    &features.data,
                    features.frames,
                    &mut timings,
                    &mut |_, _, _| {},
                )
                .unwrap()
                .data
        })
        .collect();
    let same = outputs[0]
        .iter()
        .zip(&outputs[1])
        .all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(same, "bf16 weights give a different encoder output");
}

#[test]
#[ignore = "needs the model files, see the top of this file"]
fn narrow_weights_stay_close_to_the_reference() {
    let mut failures = Vec::new();
    for precision in [Precision::F16, Precision::Int8, Precision::Q8_0] {
        let model = model_in(precision);
        let (worst, same, total) = check_narrow(precision.name(), &model);
        let floor = if precision.value_bytes() == 2 {
            0.999
        } else {
            0.995
        };
        if worst < floor {
            failures.push(format!(
                "{}: a layer's cosine similarity is {worst}",
                precision.name()
            ));
        }
        if 2 * same < total {
            failures.push(format!("{}: tokens differ on most clips", precision.name()));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}
