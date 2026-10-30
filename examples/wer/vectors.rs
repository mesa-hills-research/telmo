//! Checks the normalizers and the scorer against the output of the Python code they port
//! (`scripts/gen_vectors.py` and `scripts/gen_basic.py` write the vector files).

use crate::normalize::{Version, normalize_version};
use crate::score::{Errors, corpus_errors, count_errors, is_scored};

/// Undoes the backslash escapes of the vector files.
fn unescape(field: &str) -> String {
    let mut out = String::with_capacity(field.len());
    let mut chars = field.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('\\') => out.push('\\'),
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('u') => {
                assert_eq!(chars.next(), Some('{'), "escape in {field:?}");
                let hex: String = chars.by_ref().take_while(|&c| c != '}').collect();
                let code = u32::from_str_radix(&hex, 16).expect("hex escape");
                out.push(char::from_u32(code).expect("code point"));
            }
            other => panic!("unknown escape {other:?} in {field:?}"),
        }
    }
    out
}

#[test]
fn normalize_matches_python() {
    check_vectors(include_str!("normalize-vectors.tsv"), |text| {
        normalize_version(text, Version::V2025)
    });
}

#[test]
fn normalize_2026_matches_python() {
    check_vectors(include_str!("normalize-2026-vectors.tsv"), |text| {
        normalize_version(text, Version::V2026)
    });
}

#[test]
fn basic_normalize_matches_python() {
    check_vectors(include_str!("basic-vectors.tsv"), crate::basic::normalize);
}

/// Runs a normalizer over a vector file's inputs and compares with its outputs.
fn check_vectors(vectors: &str, normalize: fn(&str) -> String) {
    let mut failures = Vec::new();
    let mut count = 0;
    for (index, line) in vectors.lines().enumerate() {
        if line.starts_with('#') {
            continue;
        }
        let (input, expected) = line
            .split_once('\t')
            .unwrap_or_else(|| panic!("line {} has no tab", index + 1));
        let (input, expected) = (unescape(input), unescape(expected));
        let got = normalize(&input);
        count += 1;
        if got != expected {
            failures.push(format!(
                "line {} {input:?}\n  python {expected:?}\n  rust   {got:?}",
                index + 1
            ));
        }
    }
    assert!(count > 0);
    assert!(
        failures.is_empty(),
        "{} of {count} vectors differ\n{}",
        failures.len(),
        failures[..failures.len().min(40)].join("\n")
    );
}

/// splitmix64, as gen_vectors.py runs it.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// One reference and hypothesis, the same as `pair` in gen_vectors.py.
fn score_pair(r: &mut SplitMix) -> (String, String) {
    let kind = r.below(10);
    let vocabulary = 2 + r.below(8);
    let (reference, hypothesis) = if kind < 5 {
        let n = r.below(20);
        let reference: Vec<u64> = (0..n).map(|_| r.below(vocabulary)).collect();
        let n = r.below(20);
        let hypothesis: Vec<u64> = (0..n).map(|_| r.below(vocabulary)).collect();
        (reference, hypothesis)
    } else {
        let length = if kind < 9 {
            r.below(100)
        } else {
            65 + r.below(400)
        };
        let rate = r.below(if kind < 9 { 30 } else { 8 });
        let reference: Vec<u64> = (0..length).map(|_| r.below(vocabulary * 4)).collect();
        let mut hypothesis = Vec::new();
        for &word in &reference {
            if r.below(100) < rate {
                match r.below(3) {
                    0 => hypothesis.push(r.below(vocabulary * 4)),
                    1 => {
                        hypothesis.push(word);
                        hypothesis.push(r.below(vocabulary * 4));
                    }
                    _ => {}
                }
            } else {
                hypothesis.push(word);
            }
        }
        (reference, hypothesis)
    };
    let text = |words: Vec<u64>| {
        words
            .iter()
            .map(|w| format!("w{w}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    (text(reference), text(hypothesis))
}

#[test]
fn score_matches_jiwer() {
    let mut lines = include_str!("score-vectors.txt")
        .lines()
        .filter(|line| !line.starts_with('#'));
    let wer: f64 = lines
        .next()
        .and_then(|line| line.strip_prefix("wer "))
        .and_then(|wer| wer.parse().ok())
        .expect("wer line");
    let mut r = SplitMix(42);
    let mut total = Errors::default();
    let mut failures = Vec::new();
    let mut count = 0;
    for (index, line) in lines.enumerate() {
        let expected: Vec<usize> = line.split(' ').map(|n| n.parse().expect("count")).collect();
        let (reference, hypothesis) = score_pair(&mut r);
        let errors = count_errors(&reference, &hypothesis);
        let got = [
            errors.substitutions,
            errors.deletions,
            errors.insertions,
            errors.reference_words,
        ];
        if got[..] != expected[..] {
            failures.push(format!(
                "pair {index}: jiwer {expected:?}, rust {got:?}\n  {reference:?}\n  {hypothesis:?}"
            ));
        }
        total += errors;
        count += 1;
    }
    assert!(count >= 5000, "only {count} pairs");
    assert!(
        failures.is_empty(),
        "{} of {count} pairs differ\n{}",
        failures.len(),
        failures[..failures.len().min(20)].join("\n")
    );
    assert_eq!(
        total.wer().to_bits(),
        wer.to_bits(),
        "{} against {wer}",
        total.wer()
    );
}

#[test]
fn empty_texts() {
    let errors = |s, d, i, n| Errors {
        substitutions: s,
        deletions: d,
        insertions: i,
        reference_words: n,
    };
    assert_eq!(count_errors("", ""), errors(0, 0, 0, 0));
    assert_eq!(count_errors("a b", ""), errors(0, 2, 0, 2));
    assert_eq!(count_errors(" ", "a b"), errors(0, 0, 2, 0));
    assert_eq!(count_errors("a  b ", " a\tb"), errors(1, 1, 0, 2));
    assert!(!is_scored(""));
    assert!(!is_scored("  "));
    assert!(!is_scored(" ignore time segment in scoring "));
    assert!(is_scored("ignore time segment"));
    let total = corpus_errors([("a b", "a c"), ("", "x y z"), ("c", "")]);
    assert_eq!(total, errors(1, 1, 0, 3));
    assert_eq!(total.wer(), 2.0 / 3.0);
    assert_eq!(format!("{:.2}", 100.0 * total.wer()), "66.67");
}
