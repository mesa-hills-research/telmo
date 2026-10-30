//! Word error rate, counted as the Open ASR leaderboard counted it in 2025.
//!
//! The leaderboard scored with the "wer" metric of Hugging Face `evaluate`. That runs jiwer's
//! `process_words` on each pair of normalized reference and hypothesis, then divides the
//! substitutions, deletions and insertions of all pairs by the reference words of all pairs.
//! jiwer aligns the words with rapidfuzz's Levenshtein `opcodes`. This module aligns the same way,
//! down to which of several shortest alignments rapidfuzz picks, so the three counts match jiwer's
//! and not only their sum. For very long pairs rapidfuzz switches to Hirschberg's algorithm (when
//! the parts left after the common prefix and suffix have lengths that multiply to 2^22 or more,
//! such as 2,048 words each). There the split between the counts can differ, and their sum still
//! matches.
//!
//! The leaderboard drops a sample before scoring when its normalized reference is empty or reads
//! "ignore time segment in scoring" (`is_target_text_in_range` in `normalizer/data_utils.py`).
//! [`is_scored`] makes that test, and [`corpus_errors`] skips those pairs. An empty hypothesis
//! counts every reference word as deleted.

use std::ops::AddAssign;

/// The edit operations that turn a reference into a hypothesis, and the reference's length.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Errors {
    pub substitutions: usize,
    pub deletions: usize,
    pub insertions: usize,
    pub reference_words: usize,
}

impl Errors {
    pub fn total(&self) -> usize {
        self.substitutions + self.deletions + self.insertions
    }

    /// Errors over reference words, the same `f64` that `evaluate` returns. With no reference
    /// words it's NaN or infinity, where Python raises `ZeroDivisionError`. The leaderboard shows
    /// `round(100 * wer, 2)`, which `format!("{:.2}", 100.0 * wer)` matches.
    pub fn wer(&self) -> f64 {
        self.total() as f64 / self.reference_words as f64
    }
}

impl AddAssign for Errors {
    fn add_assign(&mut self, other: Errors) {
        self.substitutions += other.substitutions;
        self.deletions += other.deletions;
        self.insertions += other.insertions;
        self.reference_words += other.reference_words;
    }
}

/// Whether the leaderboard scores a sample with this normalized reference.
pub fn is_scored(normalized_reference: &str) -> bool {
    let reference = normalized_reference.trim_matches(is_python_space);
    !reference.is_empty() && reference != "ignore time segment in scoring"
}

/// The errors of one normalized reference and hypothesis, as `jiwer.process_words` counts them.
pub fn count_errors(reference: &str, hypothesis: &str) -> Errors {
    let reference = words(reference);
    let hypothesis = words(hypothesis);
    let mut errors = align(&reference, &hypothesis);
    errors.reference_words = reference.len();
    errors
}

/// The errors of a corpus of normalized (reference, hypothesis) pairs, without the pairs whose
/// reference the leaderboard drops. Their [`Errors::wer`] is the corpus word error rate.
pub fn corpus_errors<'a>(pairs: impl IntoIterator<Item = (&'a str, &'a str)>) -> Errors {
    let mut total = Errors::default();
    for (reference, hypothesis) in pairs {
        if is_scored(reference) {
            total += count_errors(reference, hypothesis);
        }
    }
    total
}

/// Python's `str.isspace()` for one character, so `\s` and `str.strip()` too.
fn is_python_space(c: char) -> bool {
    matches!(
        c,
        '\t'..='\r'
            | '\u{1c}'..='\u{20}'
            | '\u{85}'
            | '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
    )
}

/// jiwer's default transform for words. `RemoveMultipleSpaces` turns each run of two or more
/// whitespace characters into a space, `Strip` trims the ends, and `ReduceToListOfListOfWords`
/// splits on spaces.
fn words(text: &str) -> Vec<String> {
    let mut collapsed = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if is_python_space(c) && chars.peek().copied().is_some_and(is_python_space) {
            while chars.peek().copied().is_some_and(is_python_space) {
                chars.next();
            }
            collapsed.push(' ');
        } else {
            collapsed.push(c);
        }
    }
    collapsed
        .trim_matches(is_python_space)
        .split(' ')
        .filter(|word| !word.is_empty())
        .map(str::to_string)
        .collect()
}

/// rapidfuzz's `Levenshtein.editops(reference, hypothesis)`, counted. It drops the common prefix
/// and suffix, fills the edit distance matrix, and walks back from the end. At each cell it takes
/// a deletion when that is on a shortest path, else an insertion, else the diagonal.
fn align(reference: &[String], hypothesis: &[String]) -> Errors {
    let prefix = reference
        .iter()
        .zip(hypothesis)
        .take_while(|(a, b)| a == b)
        .count();
    let (reference, hypothesis) = (&reference[prefix..], &hypothesis[prefix..]);
    let suffix = reference
        .iter()
        .rev()
        .zip(hypothesis.iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let s1 = &reference[..reference.len() - suffix];
    let s2 = &hypothesis[..hypothesis.len() - suffix];

    // D[i][j] is the distance between s1[..i] and s2[..j]. Like rapidfuzz, keep only whether
    // D[i][j] - D[i - 1][j] is +1 (vp) or -1 (vn), one bit per cell.
    let stride = s1.len().div_ceil(64);
    let mut vp = vec![0u64; s2.len() * stride];
    let mut vn = vec![0u64; s2.len() * stride];
    let bit = |row: usize, col: usize| {
        (
            (row - 1) * stride + (col - 1) / 64,
            1u64 << ((col - 1) % 64),
        )
    };
    let mut column: Vec<usize> = (0..=s1.len()).collect();
    for row in 1..=s2.len() {
        let mut diagonal = column[0];
        column[0] = row;
        for col in 1..=s1.len() {
            let above = column[col];
            let cost = usize::from(s1[col - 1] != s2[row - 1]);
            let value = (diagonal + cost).min(above + 1).min(column[col - 1] + 1);
            diagonal = above;
            column[col] = value;
            let (word, mask) = bit(row, col);
            if value > column[col - 1] {
                vp[word] |= mask;
            } else if value < column[col - 1] {
                vn[word] |= mask;
            }
        }
    }

    let mut errors = Errors::default();
    let (mut col, mut row) = (s1.len(), s2.len());
    while row > 0 && col > 0 {
        let (word, mask) = bit(row, col);
        if vp[word] & mask != 0 {
            errors.deletions += 1;
            col -= 1;
            continue;
        }
        row -= 1;
        if row > 0 {
            let (word, mask) = bit(row, col);
            if vn[word] & mask != 0 {
                errors.insertions += 1;
                continue;
            }
        }
        col -= 1;
        if s1[col] != s2[row] {
            errors.substitutions += 1;
        }
    }
    errors.deletions += col;
    errors.insertions += row;
    errors
}
