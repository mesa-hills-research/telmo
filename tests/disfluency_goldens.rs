//! Compares the disfluency tagger with tensors saved from the reference implementation
//! (transformers' `ModernBertForTokenClassification`, each member) on a few sentences, one of
//! them past the local layers' band, so the attention mask is checked at its edge.
//!
//! The tests need the tagger's export (its checkpoint and goldens), so they are ignored by
//! default. Run them with
//!
//! ```text
//! DISFLUENCY_TAGGER_DIR=/path/to/export cargo test --release --test disfluency_goldens -- --include-ignored
//! ```

use std::path::PathBuf;

use telmo::disfluency::{ByteLevelTokenizer, Tagger};
use telmo::json::{self, Value};
use telmo::safetensors::SafeTensors;
use telmo::{Options, Precision};

/// Relative L2 error allowed for a layer's output, growing with depth, as for Granite.
fn tolerance(depth: usize) -> f64 {
    1e-4 * (1.0 + depth as f64 / 8.0)
}

fn export_dir() -> PathBuf {
    std::env::var_os("DISFLUENCY_TAGGER_DIR")
        .expect("set DISFLUENCY_TAGGER_DIR to the tagger's export to run this test")
        .into()
}

fn relative_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (mut diff, mut norm) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        diff += (g as f64 - w as f64).powi(2);
        norm += (w as f64).powi(2);
    }
    (diff / norm.max(1e-30)).sqrt()
}

struct Sentence {
    words: Vec<String>,
    ids: Vec<u32>,
    positions: Vec<usize>,
}

fn sentences(dir: &std::path::Path) -> Vec<Sentence> {
    let bytes = std::fs::read(dir.join("goldens.json")).expect("goldens.json");
    let list = json::parse(&bytes).expect("goldens.json parses");
    let numbers = |v: &Value, key: &str| -> Vec<u64> {
        (v.get(key).and_then(Value::as_array).expect(key).iter())
            .map(|n| n.as_u64().expect("a number"))
            .collect()
    };
    (list.as_array().expect("a list of sentences").iter())
        .map(|s| Sentence {
            words: (s.get("text").and_then(Value::as_str).expect("text"))
                .split_whitespace()
                .map(str::to_string)
                .collect(),
            ids: numbers(s, "input_ids")
                .into_iter()
                .map(|n| n as u32)
                .collect(),
            positions: numbers(s, "positions")
                .into_iter()
                .map(|n| n as usize)
                .collect(),
        })
        .collect()
}

/// Strings and their token ids by Hugging Face's tokenizer, from
/// `tests/goldens/disfluency/tokenizer-vectors.json` or the file `DISFLUENCY_TOKENIZER_VECTORS`
/// names.
#[test]
#[ignore = "needs the tagger's export"]
fn the_tokenizer_splits_text_as_the_reference_does() {
    let path = std::env::var_os("DISFLUENCY_TOKENIZER_VECTORS").map_or_else(
        || {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/goldens/disfluency/tokenizer-vectors.json")
        },
        PathBuf::from,
    );
    let bytes = std::fs::read(export_dir().join("tokenizer.json")).expect("tokenizer.json");
    let tokenizer = ByteLevelTokenizer::from_json(&bytes).expect("the tokenizer loads");
    let vectors = json::parse(&std::fs::read(&path).expect("the vectors")).expect("they parse");
    let list = vectors.as_array().expect("a list");
    for v in list {
        let text = v.get("text").and_then(Value::as_str).expect("text");
        let want: Vec<u32> = (v.get("ids").and_then(Value::as_array).expect("ids").iter())
            .map(|i| i.as_u64().expect("an id") as u32)
            .collect();
        assert_eq!(tokenizer.encode(text), want, "{text:?}");
    }
    eprintln!("{} strings", list.len());
}

#[test]
#[ignore = "needs the tagger's export"]
fn each_member_matches_the_reference_layer_by_layer() {
    let dir = export_dir();
    let options = Options {
        precision: Precision::F32,
        ..Options::default()
    };
    let tagger = Tagger::load(&dir, &options).expect("the tagger loads");
    let mut goldens = SafeTensors::open(&dir.join("goldens.safetensors")).expect("the goldens");
    let layers = tagger.config().layers;
    let tags = tagger.config().labels;
    let mut worst = 0f64;
    let mut checked = 0;
    for (n, s) in sentences(&dir).iter().enumerate() {
        let words: Vec<&str> = s.words.iter().map(String::as_str).collect();
        let (ids, positions) = tagger.token_ids(&words);
        assert_eq!(ids, s.ids, "sentence {n}'s tokens");
        assert_eq!(positions, s.positions, "sentence {n}'s word starts");
        for m in 0..tagger.members() {
            let mut taps: Vec<(String, Vec<f32>)> = Vec::new();
            let logits =
                tagger.logits(m, &ids, &mut |name, x| taps.push((name.into(), x.to_vec())));
            // Some sentences have every layer's output, in tap order, so each sits one deeper.
            for (depth, (name, got)) in taps.iter().enumerate() {
                let Ok(want) = goldens.read_f32(&format!("hidden.{n}.{m}.{name}"), None) else {
                    continue;
                };
                let err = relative_l2(got, &want);
                worst = worst.max(err / tolerance(depth));
                checked += 1;
                assert!(
                    err < tolerance(depth),
                    "sentence {n}, member {m}, {name}: {err:e}"
                );
            }
            // The logits are kept at the words' first tokens.
            let want = goldens
                .read_f32(&format!("logits.{n}.{m}"), None)
                .expect("the logits");
            let got: Vec<f32> = (positions.iter())
                .flat_map(|&p| logits[p * tags..(p + 1) * tags].to_vec())
                .collect();
            let err = relative_l2(&got, &want);
            assert!(
                err < tolerance(layers + 3),
                "sentence {n}, member {m}, logits: {err:e}"
            );
        }
        // The members' average with the keep bias, as each word's probability of a deleting tag.
        let want = goldens
            .read_f32(&format!("delete_p.{n}"), None)
            .expect("delete_p");
        for (i, (d, w)) in tagger.decide(&words).iter().zip(&want).enumerate() {
            assert!(
                (d.probability - w).abs() < 1e-4,
                "sentence {n}, word {i}: {} against {w}",
                d.probability
            );
        }
    }
    assert!(
        checked >= 2 * (layers + 3),
        "the goldens hold too few hidden states"
    );
    eprintln!("{checked} hidden states, the worst at {worst:.3} of its tolerance");
}
