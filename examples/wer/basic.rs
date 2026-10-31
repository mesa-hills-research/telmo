//! Whisper's multilingual text normalizer, `BasicTextNormalizer()` in
//! `whisper/normalizers/basic.py` of github.com/openai/whisper (MIT), ported from Python.
//!
//! It lowercases the text, removes words in brackets and parentheses, applies NFKC, turns every
//! mark, symbol and punctuation character into a space, lowercases again and collapses runs of
//! whitespace into one space. The NFKC tables in `unicode-nfkc.tsv` and the character classes come
//! from the `unicodedata` of Python 3.14 (Unicode 16.0), so the output matches that Python's.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::normalize::{
    Class, class, collapse_spaces, lowercase, remove_bracketed, remove_parenthesized,
};

/// Normalizes a transcript the way Whisper's basic normalizer does. The result can start or end
/// with a space.
pub fn normalize(text: &str) -> String {
    let s = lowercase(text);
    let s = remove_bracketed(&s);
    let s = remove_parenthesized(&s);
    let s: String = nfkc(&s)
        .into_iter()
        .map(|c| match class(c) {
            Class::Nonspacing | Class::Mark | Class::Symbol | Class::Punctuation => ' ',
            _ => c,
        })
        .collect();
    let s = lowercase(&s);
    collapse_spaces(&s).into_iter().collect()
}

struct Tables {
    decompositions: HashMap<u32, Vec<char>>,
    /// Runs of characters with a nonzero canonical combining class, as (first, last, class).
    classes: Vec<(u32, u32, u8)>,
    compositions: HashMap<(u32, u32), char>,
}

fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let hex = |s: &str| u32::from_str_radix(s, 16).expect("hex code point in unicode-nfkc.tsv");
        let ch = |s: &str| char::from_u32(hex(s)).expect("code point in unicode-nfkc.tsv");
        let mut tables = Tables {
            decompositions: HashMap::new(),
            classes: Vec::new(),
            compositions: HashMap::new(),
        };
        for line in include_str!("unicode-nfkc.tsv").lines() {
            let fields: Vec<&str> = line.split(' ').collect();
            match fields.as_slice() {
                ["d", c, parts @ ..] => {
                    tables
                        .decompositions
                        .insert(hex(c), parts.iter().map(|p| ch(p)).collect());
                }
                ["k", run, class] => {
                    let (first, last) = run.split_once('-').unwrap_or((run, run));
                    let class = class.parse().expect("combining class in unicode-nfkc.tsv");
                    tables.classes.push((hex(first), hex(last), class));
                }
                ["p", a, b, c] => {
                    tables.compositions.insert((hex(a), hex(b)), ch(c));
                }
                _ => {}
            }
        }
        tables
    })
}

const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const S_COUNT: u32 = L_COUNT * V_COUNT * T_COUNT;

fn combining_class(c: char) -> u8 {
    let classes = &tables().classes;
    let index = classes.partition_point(|&(first, _, _)| first <= c as u32);
    match index.checked_sub(1).map(|i| classes[i]) {
        Some((_, last, class)) if c as u32 <= last => class,
        _ => 0,
    }
}

fn decompose(c: char, out: &mut Vec<char>) {
    let s = (c as u32).wrapping_sub(S_BASE);
    if s < S_COUNT {
        let jamo = |cp: u32| char::from_u32(cp).expect("a Hangul jamo");
        out.push(jamo(L_BASE + s / (V_COUNT * T_COUNT)));
        out.push(jamo(V_BASE + (s % (V_COUNT * T_COUNT)) / T_COUNT));
        if !s.is_multiple_of(T_COUNT) {
            out.push(jamo(T_BASE + s % T_COUNT));
        }
        return;
    }
    match tables().decompositions.get(&(c as u32)) {
        Some(parts) => out.extend_from_slice(parts),
        None => out.push(c),
    }
}

fn compose(a: char, b: char) -> Option<char> {
    let (a, b) = (a as u32, b as u32);
    let (l, v) = (a.wrapping_sub(L_BASE), b.wrapping_sub(V_BASE));
    if l < L_COUNT && v < V_COUNT {
        return char::from_u32(S_BASE + (l * V_COUNT + v) * T_COUNT);
    }
    let (s, t) = (a.wrapping_sub(S_BASE), b.wrapping_sub(T_BASE));
    if s < S_COUNT && s.is_multiple_of(T_COUNT) && t > 0 && t < T_COUNT {
        return char::from_u32(a + t);
    }
    tables().compositions.get(&(a, b)).copied()
}

/// Unicode normalization form KC, which is full compatibility decomposition, canonical ordering
/// of combining marks, then canonical composition.
pub fn nfkc(s: &[char]) -> Vec<char> {
    let mut chars = Vec::with_capacity(s.len());
    for &c in s {
        decompose(c, &mut chars);
    }
    let mut i = 0;
    while i < chars.len() {
        if combining_class(chars[i]) == 0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len() && combining_class(chars[i]) != 0 {
            i += 1;
        }
        chars[start..i].sort_by_key(|&c| combining_class(c));
    }

    let mut out: Vec<char> = Vec::with_capacity(chars.len());
    // The last starter's index in `out`, and the class of the last character after it.
    let mut starter: Option<usize> = None;
    let mut last_class: Option<u8> = None;
    for c in chars {
        let class = combining_class(c);
        if let Some(index) = starter {
            let blocked = last_class.is_some_and(|last| last == 0 || last >= class);
            if !blocked && let Some(composite) = compose(out[index], c) {
                out[index] = composite;
                continue;
            }
        }
        if class == 0 {
            starter = Some(out.len());
            last_class = None;
        } else {
            last_class = Some(class);
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nfkc_str(s: &str) -> String {
        nfkc(&s.chars().collect::<Vec<_>>()).into_iter().collect()
    }

    #[test]
    fn nfkc_composes_reorders_and_maps_compatibility_forms() {
        assert_eq!(nfkc_str("e\u{301}"), "\u{e9}");
        assert_eq!(nfkc_str("e\u{301}\u{327}"), "\u{229}\u{301}");
        assert_eq!(nfkc_str("\u{212b}"), "\u{c5}");
        assert_eq!(nfkc_str("\u{fb01}"), "fi");
        assert_eq!(nfkc_str("\u{ff76}\u{ff9e}"), "\u{30ac}");
        assert_eq!(nfkc_str("\u{1100}\u{1161}\u{11a8}"), "\u{ac01}");
        assert_eq!(nfkc_str("\u{ac00}\u{11a8}"), "\u{ac01}");
        assert_eq!(nfkc_str("\u{ac01}"), "\u{ac01}");
        assert_eq!(nfkc_str("\u{301}e"), "\u{301}e");
        assert_eq!(nfkc_str("\u{ff21}\u{2460}"), "A1");
    }

    #[test]
    fn basic_drops_brackets_and_turns_symbols_into_spaces() {
        assert_eq!(normalize("Hello, World!"), "hello world ");
        assert_eq!(normalize("[noise] Ça (va) bien?"), " ça bien ");
        assert_eq!(normalize("日本語、テスト。"), "日本語 テスト ");
    }
}
