//! Compares the engine with tensors saved from the reference implementation (NVIDIA NeMo) on
//! LibriSpeech clips. `tests/goldens/parakeet-tdt-0.6b-v3/README.md` says where they come from.
//!
//! The front-end test needs only the files in this repository. The others need the model's
//! files, so they are ignored by default. Run them with
//!
//! ```text
//! PARAKEET_TDT_DIR=/path/to/parakeet-tdt-0.6b-v3 cargo test --release --test goldens -- --include-ignored
//! ```

use std::path::{Path, PathBuf};

use notate::features::{FrontEnd, FrontEndConfig};
use notate::json::{self, Value};
use notate::kernels::Isa;
use notate::parakeet::{DecodeTrace, Timings};
use notate::safetensors::SafeTensors;
use notate::{Options, Parakeet};

/// Features may differ by this much (after normalization to unit variance). The reference
/// computes its STFT in f32, and the rounding shows most in the lowest mel filters, which sum
/// only one or two FFT bins, so the largest differences sit there while the mean stays near 1e-6.
const FEATURES_MAX_ABS: f32 = 5e-4;
const FEATURES_MEAN_ABS: f64 = 2e-6;
/// Relative L2 error allowed for the subsampling output and each layer, growing with depth.
fn layer_tolerance(layer: Option<usize>) -> f64 {
    match layer {
        None => 1e-5,
        Some(i) => 1e-4 * (1.0 + i as f64 / 8.0),
    }
}

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/parakeet-tdt-0.6b-v3")
}

struct Clip {
    id: String,
    samples: Vec<f32>,
    tensors: SafeTensors,
    reference_text: String,
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
            let id = field("id");
            let audio = notate::wav::read(&dir.join(field("wav"))).expect("the clip's WAV");
            assert_eq!(audio.sample_rate, 16000);
            let tensors = SafeTensors::from_bytes(
                std::fs::read(dir.join(field("tensors"))).expect("the clip's tensors"),
            )
            .expect("the clip's tensors parse");
            Clip {
                id,
                samples: audio.samples,
                tensors,
                reference_text: field("nemo_text"),
            }
        })
        .collect()
}

fn model(isa: Option<Isa>, threads: usize) -> Parakeet {
    let dir = std::env::var_os("PARAKEET_TDT_DIR")
        .expect("set PARAKEET_TDT_DIR to the model's directory to run this test");
    Parakeet::load(Path::new(&dir), &Options { threads, isa }).expect("the model loads")
}

/// ||got - want|| / ||want|| over the first `rows` rows of `width`.
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
    let front_end = FrontEnd::new(FrontEndConfig::parakeet());
    for mut clip in clips() {
        let got = front_end.compute(&clip.samples).unwrap();
        let valid = i64s(&mut clip.tensors, "features_len")[0] as usize;
        let (shape, want) = f32s(&mut clip.tensors, "features");
        assert_eq!(shape[1], 128);
        assert_eq!(got.frames, valid, "{}: frame count", clip.id);
        let errors: Vec<f32> = got
            .data
            .iter()
            .zip(&want[..valid * 128])
            .map(|(g, w)| (g - w).abs())
            .collect();
        let (at, max) = errors
            .iter()
            .enumerate()
            .fold((0, 0f32), |b, (i, &e)| if e > b.1 { (i, e) } else { b });
        let mean = errors.iter().map(|&e| e as f64).sum::<f64>() / errors.len() as f64;
        eprintln!(
            "{}: features max abs error {max:.2e} (frame {}, bin {}, value {:.3}), mean {mean:.2e}",
            clip.id,
            at / 128,
            at % 128,
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
}

/// Runs a clip through the encoder and decoder, checking every saved tensor.
fn check_clip(model: &Parakeet, clip: &mut Clip) {
    let id = clip.id.clone();
    let st = &mut clip.tensors;
    let features = model.features(&clip.samples).unwrap();
    let mut errors: Vec<(String, f64)> = Vec::new();
    let mut timings = Timings::default();
    let encoded = model.encode(
        &features.data,
        features.frames,
        &mut timings,
        &mut |name, rows, x| {
            if st.info(name).is_none() {
                return;
            }
            // The reference may run one frame past the valid length, which only padding feeds.
            let (shape, want) = f32s(st, name);
            assert!(
                shape[0] >= rows && shape[0] <= rows + 1,
                "{id}: {name} has {} rows, the engine {rows}",
                shape[0]
            );
            let err = relative_l2(x, &want[..x.len()]);
            let layer = name.strip_prefix("layer.").map(|n| n.parse().unwrap());
            eprintln!("{id}: {name} relative L2 error {err:.2e}");
            if err > layer_tolerance(layer) {
                errors.push((name.to_string(), err));
            }
        },
    );
    assert_eq!(
        i64s(st, "encoded_len")[0] as usize,
        encoded.frames,
        "{id}: encoder frames"
    );

    let mut trace = DecodeTrace::default();
    let tokens = model.decode(&encoded, &mut timings, Some(&mut trace));
    let (_, enc_proj) = f32s(st, "enc_proj");
    let err = relative_l2(&trace.enc_proj, &enc_proj[..trace.enc_proj.len()]);
    eprintln!("{id}: enc_proj relative L2 error {err:.2e}");
    if err > layer_tolerance(Some(24)) {
        errors.push(("enc_proj".into(), err));
    }

    let ids: Vec<i64> = tokens.iter().map(|t| t.id as i64).collect();
    let want_ids = i64s(st, "tokens");
    assert_eq!(ids, want_ids, "{id}: token ids");
    let frames: Vec<i64> = tokens.iter().map(|t| t.frame as i64).collect();
    assert_eq!(frames, i64s(st, "token_frame"), "{id}: token frames");
    let durations: Vec<i64> = tokens.iter().map(|t| t.duration as i64).collect();
    assert_eq!(
        durations,
        i64s(st, "token_duration"),
        "{id}: token durations"
    );

    let (pred_shape, preds) = f32s(st, "pred");
    let width = pred_shape[1];
    let got: Vec<f32> = trace.preds.iter().flatten().copied().collect();
    let err = relative_l2(&got, &preds[..got.len()]);
    eprintln!(
        "{id}: prediction network relative L2 error {err:.2e} over {} states",
        trace.preds.len()
    );
    assert_eq!(got.len(), preds.len(), "{id}: prediction network states");
    assert_eq!(width, model.config().pred_hidden);
    if err > layer_tolerance(Some(24)) {
        errors.push(("pred".into(), err));
    }

    let (joint_shape, joints) = f32s(st, "joint");
    let joint_frames = i64s(st, "joint_frame");
    let steps = joint_shape[0];
    let outputs = joint_shape[1];
    let mut worst = 0f64;
    for s in 0..steps {
        let (frame, logits) = &trace.joints[s];
        assert_eq!(*frame as i64, joint_frames[s], "{id}: joint step {s} frame");
        worst = worst.max(relative_l2(logits, &joints[s * outputs..(s + 1) * outputs]));
    }
    eprintln!("{id}: joint logits worst relative L2 error {worst:.2e} over {steps} steps");
    if worst > layer_tolerance(Some(24)) {
        errors.push(("joint".into(), worst));
    }

    let text = model
        .tokenizer()
        .decode(&tokens.iter().map(|t| t.id).collect::<Vec<_>>());
    assert_eq!(text, clip.reference_text, "{id}: text");
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
    let clip = clips().into_iter().next().unwrap();
    let mut first: Option<Vec<f32>> = None;
    for threads in [1, 2, 3, 6] {
        let model = model(None, threads);
        let features = model.features(&clip.samples).unwrap();
        let encoded = model.encode(
            &features.data,
            features.frames,
            &mut Timings::default(),
            &mut |_, _, _| {},
        );
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
fn the_scalar_kernels_give_the_same_tokens() {
    let model = model(Some(Isa::Scalar), 6);
    for clip in clips().iter().filter(|c| c.samples.len() < 16000 * 10) {
        let transcript = model.transcribe(&clip.samples).unwrap();
        assert_eq!(transcript.text, clip.reference_text, "{}", clip.id);
    }
}
