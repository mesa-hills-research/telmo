//! The Open ASR leaderboard's English text normalizer, ported from Python.
//!
//! The leaderboard normalizes both the reference and the hypothesis before it counts word errors.
//! This is a port of `EnglishTextNormalizer` in `normalizer/normalizer.py` (Whisper's normalizer)
//! with the spelling map of `normalizer/english_abbreviations.py`, from
//! github.com/huggingface/open_asr_leaderboard at commit 6a3e6d4e3ec28b81aa5d9a15a5467e13e2b98622
//! (Apache-2.0). That is the code behind the leaderboard's scores of 2025, before it changed its
//! normalizer in April 2026.
//!
//! [`Version::V2026`] follows the same file at commit 2aaa07a5ea7c5b18ea53c511426d2079b35d1200
//! (2026), which the leaderboard scores with now. It removes more filler words, drops the ":00" of
//! whole hours, expands "'s" only after a few pronouns and adverbs, joins compounds such as
//! "wi fi", reads "oh" as a zero only inside a run of digits, maps name variants to one spelling
//! and joins runs of single letters such as "b b c". Its tables of spelling changes and names are
//! in `english-2026.tsv`.
//!
//! The output matches the Python code character for character, Python's Unicode rules included.
//! Each regular expression of the original is a hand-written scan here, and the comment above it
//! quotes the expression. Character classes come from `unicode-classes.tsv`, generated from the
//! `unicodedata` of Python 3.14 (Unicode 16.0), and the spelling map from `english-spelling.tsv`.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Which version of the leaderboard's normalizer to follow.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Version {
    /// Commit 6a3e6d4e, behind the leaderboard's scores of 2025.
    #[default]
    V2025,
    /// Commit 2aaa07a5 of 2026.
    V2026,
}

/// Normalizes a transcript the way a version of the leaderboard's normalizer does before it
/// computes the word error rate.
///
/// The result can start or end with a space, as the Python code's does. Split it on whitespace
/// to get the words.
pub fn normalize_version(text: &str, version: Version) -> String {
    let s = lowercase(text);
    let s = remove_bracketed(&s);
    let s = remove_parenthesized(&s);
    let s = match version {
        Version::V2025 => remove_fillers(&s),
        Version::V2026 => remove_fillers_2026(&s),
    };
    let s = join_apostrophes(&s);
    let s = expand_contractions(s, version);
    let s = remove_digit_commas(&s);
    let s = remove_periods(&s);
    let s = remove_symbols_and_diacritics(&s);
    let s = match version {
        Version::V2025 => s,
        Version::V2026 => join_compounds(s),
    };
    let s = standardize_numbers(&s, version);
    let s = standardize_spellings(&s, version);
    let s = match version {
        Version::V2025 => s,
        Version::V2026 => collapse_acronyms(&standardize_names(&s)),
    };
    let s = remove_lone_symbols(&s);
    collapse_spaces(&s).into_iter().collect()
}

// Character data.

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Class {
    /// `str.isspace()`, so `\s` and `str.split()`.
    Space,
    /// `str.isdecimal()`, so `\d`.
    Decimal,
    /// The other characters of `str.isalnum()`, letters and numbers.
    Word,
    /// Category Mn, which the symbol removal drops.
    Nonspacing,
    /// The other marks, which the symbol removal turns into spaces like symbols and punctuation.
    Mark,
    Symbol,
    Punctuation,
    Other,
}

/// What the symbol removal turns a run of characters into.
enum Replacement {
    /// Every character of the run becomes this text.
    Fixed(String),
    /// The run's characters become this character, the next one and so on.
    CountUp(u32),
}

struct Tables {
    ascii: [Class; 128],
    /// The first code point of each run of one class, in order.
    classes: Vec<(u32, Class)>,
    /// The first and last code point of each run, in order.
    replacements: Vec<(u32, u32, Replacement)>,
    /// Runs of the characters that Python's `str.lower()` changes, first and last code point.
    lowered: Vec<(u32, u32)>,
    spellings: HashMap<&'static str, &'static str>,
    /// The spelling map of 2026.
    spellings_2026: HashMap<&'static str, &'static str>,
    /// Name variants and the spelling they map to, from 2026.
    names: HashMap<&'static str, &'static str>,
}

fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(load_tables)
}

fn load_tables() -> Tables {
    let hex = |s: &str| u32::from_str_radix(s, 16).expect("hex code point in unicode-classes.tsv");
    let range = |field: &str| match field.split_once('-') {
        Some((a, b)) => (hex(a), hex(b)),
        None => (hex(field), hex(field)),
    };
    let mut classes = Vec::new();
    let mut replacements = Vec::new();
    let mut lowered = Vec::new();
    for line in include_str!("unicode-classes.tsv").lines() {
        let mut fields = line.split(' ');
        match fields.next() {
            Some("l") => lowered.push(range(fields.next().expect("code point"))),
            Some("c") => {
                let start = hex(fields.next().expect("start"));
                let class = match fields.next() {
                    Some("z") => Class::Space,
                    Some("d") => Class::Decimal,
                    Some("w") => Class::Word,
                    Some("n") => Class::Nonspacing,
                    Some("m") => Class::Mark,
                    Some("s") => Class::Symbol,
                    Some("p") => Class::Punctuation,
                    Some("o") => Class::Other,
                    other => panic!("unknown class {other:?} in unicode-classes.tsv"),
                };
                classes.push((start, class));
            }
            Some("r") => {
                let (first, last) = range(fields.next().expect("code point"));
                let rest: Vec<&str> = fields.collect();
                let replacement = match rest.as_slice() {
                    [up] if up.starts_with('+') => Replacement::CountUp(hex(&up[1..])),
                    _ => Replacement::Fixed(
                        rest.iter()
                            .map(|h| char::from_u32(hex(h)).expect("code point"))
                            .collect(),
                    ),
                };
                replacements.push((first, last, replacement));
            }
            _ => {}
        }
    }
    let mut ascii = [Class::Other; 128];
    for (c, class) in ascii.iter_mut().enumerate() {
        *class = lookup_class(&classes, c as u32);
    }
    let spellings: HashMap<&'static str, &'static str> = include_str!("english-spelling.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| line.split_once('\t'))
        .collect();
    let mut spellings_2026 = spellings.clone();
    let mut names = HashMap::new();
    for line in include_str!("english-2026.tsv").lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        match fields.as_slice() {
            ["s", word, spelling] => {
                spellings_2026.insert(*word, *spelling);
            }
            ["x", word] => {
                spellings_2026.remove(word);
            }
            ["n", name, canonical] => {
                names.insert(*name, *canonical);
            }
            _ => {}
        }
    }
    Tables {
        ascii,
        classes,
        replacements,
        lowered,
        spellings,
        spellings_2026,
        names,
    }
}

/// Python's `str.lower()`. Rust's `str::to_lowercase` follows the same rules, the final sigma
/// included, but its Unicode version can be newer than Python's. So only the characters that
/// Python changes take Rust's lowercase, and the others stay.
pub(crate) fn lowercase(text: &str) -> Vec<char> {
    let lowered = text.to_lowercase();
    let runs = &tables().lowered;
    let mut out = Vec::with_capacity(text.len());
    let mut cursor = 0;
    for c in text.chars() {
        // str::to_lowercase gives each character's own lowercase, and σ or ς for Σ.
        let length = if c == 'Σ' {
            'σ'.len_utf8()
        } else {
            c.to_lowercase().map(char::len_utf8).sum()
        };
        let index = runs.partition_point(|&(first, _)| first <= c as u32);
        if index > 0 && c as u32 <= runs[index - 1].1 {
            out.extend(lowered[cursor..cursor + length].chars());
        } else {
            out.push(c);
        }
        cursor += length;
    }
    out
}

fn lookup_class(classes: &[(u32, Class)], c: u32) -> Class {
    let index = classes.partition_point(|&(start, _)| start <= c);
    classes[index - 1].1
}

pub(crate) fn class(c: char) -> Class {
    let tables = tables();
    match tables.ascii.get(c as usize) {
        Some(&class) => class,
        None => lookup_class(&tables.classes, c as u32),
    }
}

fn is_space(c: char) -> bool {
    class(c) == Class::Space
}

/// `\w`, which also decides where `\b` matches.
fn is_word(c: char) -> bool {
    c == '_' || matches!(class(c), Class::Decimal | Class::Word)
}

fn is_decimal(c: char) -> bool {
    class(c) == Class::Decimal
}

/// The value of a decimal digit of any script. Each script's digits run from 0 to 9 in order,
/// and its run of the class table starts at a zero.
fn digit_value(c: char) -> u32 {
    let classes = &tables().classes;
    let index = classes.partition_point(|&(start, _)| start <= c as u32);
    (c as u32 - classes[index - 1].0) % 10
}

/// `\b` at position `i`, between `s[i - 1]` and `s[i]`.
fn boundary(s: &[char], i: usize) -> bool {
    let before = i > 0 && is_word(s[i - 1]);
    let after = i < s.len() && is_word(s[i]);
    before != after
}

fn starts_with(s: &[char], i: usize, literal: &str) -> bool {
    let mut rest = s.get(i..).unwrap_or_default().iter();
    literal.chars().all(|c| rest.next() == Some(&c))
}

/// Python's `str.split()`.
fn split_words(s: &[char]) -> Vec<String> {
    s.split(|&c| is_space(c))
        .filter(|word| !word.is_empty())
        .map(|word| word.iter().collect())
        .collect()
}

// The steps of `EnglishTextNormalizer.__call__`, in order.

/// `[<\[][^>\]]*[>\]]` becomes nothing.
pub(crate) fn remove_bracketed(s: &[char]) -> Vec<char> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if matches!(s[i], '<' | '[') {
            match s[i + 1..].iter().position(|&c| matches!(c, '>' | ']')) {
                Some(k) => {
                    i += k + 2;
                    continue;
                }
                None => {
                    // No closing bracket follows, so no later opening one can match either.
                    out.extend_from_slice(&s[i..]);
                    break;
                }
            }
        }
        out.push(s[i]);
        i += 1;
    }
    out
}

/// `\(([^)]+?)\)` becomes nothing. An empty pair "()" stays.
pub(crate) fn remove_parenthesized(s: &[char]) -> Vec<char> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if s[i] == '(' && i + 1 < s.len() && s[i + 1] != ')' {
            match s[i + 2..].iter().position(|&c| c == ')') {
                Some(k) => {
                    i += k + 3;
                    continue;
                }
                None => {
                    out.extend_from_slice(&s[i..]);
                    break;
                }
            }
        }
        out.push(s[i]);
        i += 1;
    }
    out
}

/// `\b(hmm|mm|mhm|mmm|uh|um)\b` becomes nothing. Every alternative is all word characters, so a
/// match is a whole run of word characters.
fn remove_fillers(s: &[char]) -> Vec<char> {
    const FILLERS: [&str; 6] = ["hmm", "mm", "mhm", "mmm", "uh", "um"];
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if is_word(s[i]) {
            let end = s[i..]
                .iter()
                .position(|&c| !is_word(c))
                .map_or(s.len(), |k| i + k);
            let word = &s[i..end];
            if !FILLERS.iter().any(|f| f.chars().eq(word.iter().copied())) {
                out.extend_from_slice(word);
            }
            i = end;
        } else {
            out.push(s[i]);
            i += 1;
        }
    }
    out
}

/// The run of word characters that starts at `i`, as an end index.
fn word_end(s: &[char], i: usize) -> usize {
    s[i..]
        .iter()
        .position(|&c| !is_word(c))
        .map_or(s.len(), |k| i + k)
}

/// The filler patterns of 2026 after "ah-ha", each a sequence of a character or a set in
/// brackets with an optional `+` or `*`. Python writes the tenth as `m{2,}`.
const FILLERS_2026: [&str; 25] = [
    "a+h+m*",
    "a+h+a+",
    "e+h+m*",
    "e+m+",
    "e+r+m*",
    "h+a+h+",
    "h+e+h+",
    "h+m+",
    "h+u+h+",
    "mm+",
    "m+h+m*",
    "t+s+k+",
    "u+g+h+",
    "u+h+m*",
    "u+h+u+[hm]*",
    "u+m+h*",
    "ahem",
    "eheh",
    "ehehe",
    "ehr",
    "hmmph",
    "hum",
    "hunh",
    "mhum",
    "mmkay",
];

/// Whether a whole word matches a filler pattern. Each pattern's neighbouring parts take
/// different characters, except `mm+`, whose first part takes exactly one, so matching each part
/// greedily finds every match.
fn matches_filler(word: &[char], pattern: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let (mut p, mut w) = (0, 0);
    while p < pattern.len() {
        let set: &[char] = if pattern[p] == '[' {
            let close = pattern[p..].iter().position(|&c| c == ']').unwrap() + p;
            let set = &pattern[p + 1..close];
            p = close + 1;
            set
        } else {
            p += 1;
            &pattern[p - 1..p]
        };
        let (min, more) = match pattern.get(p) {
            Some('+') => (1, true),
            Some('*') => (0, true),
            _ => (1, false),
        };
        if more {
            p += 1;
        }
        let max = if more { usize::MAX } else { 1 };
        let mut count = 0;
        while w < word.len() && count < max && set.contains(&word[w]) {
            w += 1;
            count += 1;
        }
        if count < min {
            return false;
        }
    }
    w == word.len()
}

/// The 2026 fillers, `\b(ah-ha|a+h+m*|a+h+a+|...|mhum|mmkay)\b`, become nothing. Every
/// alternative after "ah-ha" is all letters, so it matches a whole run of word characters. "ah-ha"
/// comes first, so the run "ah" followed by "-" and the run "ha" goes as a whole.
fn remove_fillers_2026(s: &[char]) -> Vec<char> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if !is_word(s[i]) {
            out.push(s[i]);
            i += 1;
            continue;
        }
        let end = word_end(s, i);
        let word = &s[i..end];
        if word == ['a', 'h'] && starts_with(s, end, "-ha") && boundary(s, end + 3) {
            i = end + 3;
            continue;
        }
        if !FILLERS_2026.iter().any(|p| matches_filler(word, p)) {
            out.extend_from_slice(word);
        }
        i = end;
    }
    out
}

/// `\s+'` becomes `'`.
fn join_apostrophes(s: &[char]) -> Vec<char> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if is_space(s[i]) {
            let end = s[i..]
                .iter()
                .position(|&c| !is_space(c))
                .map_or(s.len(), |k| i + k);
            if end < s.len() && s[end] == '\'' {
                out.push('\'');
                i = end + 1;
            } else {
                out.extend_from_slice(&s[i..end]);
                i = end;
            }
        } else {
            out.push(s[i]);
            i += 1;
        }
    }
    out
}

/// The `replacers` of `EnglishTextNormalizer`, applied one after another. Each is a literal that
/// ends with `\b`, and the ones marked `true` also start with `\b`.
const REPLACERS: [(&str, bool, &str); 49] = [
    // Common contractions.
    ("won't", true, "will not"),
    ("can't", true, "can not"),
    ("let's", true, "let us"),
    ("ain't", true, "aint"),
    ("y'all", true, "you all"),
    ("wanna", true, "want to"),
    ("gotta", true, "got to"),
    ("gonna", true, "going to"),
    ("i'ma", true, "i am going to"),
    ("imma", true, "i am going to"),
    ("woulda", true, "would have"),
    ("coulda", true, "could have"),
    ("shoulda", true, "should have"),
    ("ma'am", true, "madam"),
    // Titles.
    ("mr", true, "mister "),
    ("mrs", true, "missus "),
    ("st", true, "saint "),
    ("dr", true, "doctor "),
    ("prof", true, "professor "),
    ("capt", true, "captain "),
    ("gov", true, "governor "),
    ("ald", true, "alderman "),
    ("gen", true, "general "),
    ("sen", true, "senator "),
    ("rep", true, "representative "),
    ("pres", true, "president "),
    ("rev", true, "reverend "),
    ("hon", true, "honorable "),
    ("asst", true, "assistant "),
    ("assoc", true, "associate "),
    ("lt", true, "lieutenant "),
    ("col", true, "colonel "),
    ("jr", true, "junior "),
    ("sr", true, "senior "),
    ("esq", true, "esquire "),
    // Perfect tenses.
    ("'d been", false, " had been"),
    ("'s been", false, " has been"),
    ("'d gone", false, " had gone"),
    ("'s gone", false, " has gone"),
    ("'d done", false, " had done"),
    ("'s got", false, " has got"),
    // General contractions.
    ("n't", false, " not"),
    ("'re", false, " are"),
    ("'s", false, " is"),
    ("'d", false, " would"),
    ("'ll", false, " will"),
    ("'t", false, " not"),
    ("'ve", false, " have"),
    ("'m", false, " am"),
];

/// The words that "'s" expands after in 2026,
/// `\b(it|he|she|what|that|who|here|there|how|when|where|why|this)'s\b`.
const IS_WORDS: [&str; 13] = [
    "it", "he", "she", "what", "that", "who", "here", "there", "how", "when", "where", "why",
    "this",
];

/// The 2026 replacer `\b(it|he|...|this)'s\b`, which becomes `\1 is`.
fn expand_is_2026(s: &[char]) -> Vec<char> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    'scan: while i < s.len() {
        if boundary(s, i) {
            for word in IS_WORDS {
                let length = word.chars().count() + 2;
                if starts_with(s, i, word)
                    && starts_with(s, i + length - 2, "'s")
                    && boundary(s, i + length)
                {
                    out.extend(word.chars());
                    out.extend(" is".chars());
                    i += length;
                    continue 'scan;
                }
            }
        }
        out.push(s[i]);
        i += 1;
    }
    out
}

/// The first 2026 replacer, `\b(\d{1,2}):00\b`, which becomes `\1`.
fn drop_whole_hours(s: &[char]) -> Vec<char> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if boundary(s, i) && is_decimal(s[i]) {
            let digits = s[i..].iter().take_while(|&&c| is_decimal(c)).count();
            if digits <= 2 && starts_with(s, i + digits, ":00") && boundary(s, i + digits + 3) {
                out.extend_from_slice(&s[i..i + digits]);
                i += digits + 3;
                continue;
            }
        }
        out.push(s[i]);
        i += 1;
    }
    out
}

fn expand_contractions(mut s: Vec<char>, version: Version) -> Vec<char> {
    if version == Version::V2026 {
        s = drop_whole_hours(&s);
    }
    for (literal, leading_boundary, replacement) in REPLACERS {
        if version == Version::V2026 && literal == "'s" {
            s = expand_is_2026(&s);
            continue;
        }
        let length = literal.chars().count();
        let mut out = Vec::with_capacity(s.len());
        let mut i = 0;
        while i < s.len() {
            if starts_with(&s, i, literal)
                && (!leading_boundary || boundary(&s, i))
                && boundary(&s, i + length)
            {
                out.extend(replacement.chars());
                i += length;
            } else {
                out.push(s[i]);
                i += 1;
            }
        }
        s = out;
    }
    s
}

/// `(\d),(\d)` becomes `\1\2`.
fn remove_digit_commas(s: &[char]) -> Vec<char> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if i + 2 < s.len() && is_decimal(s[i]) && s[i + 1] == ',' && is_decimal(s[i + 2]) {
            out.push(s[i]);
            out.push(s[i + 2]);
            i += 3;
        } else {
            out.push(s[i]);
            i += 1;
        }
    }
    out
}

/// `\.([^0-9]|$)` becomes ` \1`. The character after the period is part of the match, so a
/// period right after a removed one stays.
fn remove_periods(s: &[char]) -> Vec<char> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if s[i] == '.' && i + 1 == s.len() {
            out.push(' ');
            i += 1;
        } else if s[i] == '.' && !s[i + 1].is_ascii_digit() {
            out.push(' ');
            out.push(s[i + 1]);
            i += 2;
        } else {
            out.push(s[i]);
            i += 1;
        }
    }
    out
}

/// `remove_symbols_and_diacritics(s, keep=".%$¢€£")`. It decomposes the text (NFKD), drops
/// nonspacing marks, turns other marks, symbols and punctuation into spaces, and spells a few
/// letters with plain ones ("ø" becomes "o"). Canonical reordering only moves marks, which all
/// become nothing or a space, so the result is the same done one character at a time.
fn remove_symbols_and_diacritics(s: &[char]) -> Vec<char> {
    let tables = tables();
    let mut out = Vec::with_capacity(s.len());
    for &c in s {
        let code = c as u32;
        if (0xAC00..=0xD7A3).contains(&code) {
            // A Hangul syllable decomposes into its letters (jamo), which are kept.
            let index = code - 0xAC00;
            let jamo = [
                0x1100 + index / 588,
                0x1161 + (index % 588) / 28,
                0x11A7 + index % 28,
            ];
            let count = if index.is_multiple_of(28) { 2 } else { 3 };
            out.extend(jamo[..count].iter().filter_map(|&j| char::from_u32(j)));
            continue;
        }
        let index = tables
            .replacements
            .partition_point(|&(first, _, _)| first <= code);
        if index > 0 {
            let (first, last, replacement) = &tables.replacements[index - 1];
            if code <= *last {
                match replacement {
                    Replacement::Fixed(text) => out.extend(text.chars()),
                    Replacement::CountUp(start) => {
                        out.extend(char::from_u32(start + (code - first)));
                    }
                }
                continue;
            }
        }
        if matches!(c, '.' | '%' | '$' | '¢' | '€' | '£') {
            out.push(c);
            continue;
        }
        match class(c) {
            Class::Nonspacing => {}
            Class::Mark | Class::Symbol | Class::Punctuation => out.push(' '),
            _ => out.push(c),
        }
    }
    out
}

/// The compounds of 2026, each `\b<first>\s+<second>\b` becoming one word, in order.
const COMPOUNDS: [(&str, &str, &str); 16] = [
    ("et", "cetera", "etc"),
    ("al", "right", "alright"),
    ("all", "right", "alright"),
    ("how", "ever", "however"),
    ("wi", "fi", "wifi"),
    ("hi", "fi", "hifi"),
    ("lo", "fi", "lofi"),
    ("sci", "fi", "scifi"),
    ("e", "mail", "email"),
    ("e", "book", "ebook"),
    ("e", "commerce", "ecommerce"),
    ("x", "ray", "xray"),
    ("t", "shirt", "tshirt"),
    ("a", "m", "am"),
    ("p", "m", "pm"),
    ("o", "k", "okay"),
];

/// The 2026 compound rules, applied one after another after symbol removal, then `\bo+h+\b` and
/// `\booo+\b`, which both become "oh".
fn join_compounds(mut s: Vec<char>) -> Vec<char> {
    for (first, second, joined) in COMPOUNDS {
        let mut out = Vec::with_capacity(s.len());
        let mut i = 0;
        while i < s.len() {
            if boundary(&s, i) && starts_with(&s, i, first) {
                let gap = i + first.chars().count();
                let next = s[gap..]
                    .iter()
                    .position(|&c| !is_space(c))
                    .map_or(s.len(), |k| gap + k);
                let end = next + second.chars().count();
                if next > gap && starts_with(&s, next, second) && boundary(&s, end) {
                    out.extend(joined.chars());
                    i = end;
                    continue;
                }
            }
            out.push(s[i]);
            i += 1;
        }
        s = out;
    }
    for pattern in ["o+h+", "ooo+"] {
        let mut out = Vec::with_capacity(s.len());
        let mut i = 0;
        while i < s.len() {
            if is_word(s[i]) && boundary(&s, i) {
                let end = word_end(&s, i);
                if matches_filler(&s[i..end], pattern) {
                    out.extend("oh".chars());
                } else {
                    out.extend_from_slice(&s[i..end]);
                }
                i = end;
                continue;
            }
            out.push(s[i]);
            i += 1;
        }
        s = out;
    }
    s
}

fn standardize_spellings(s: &[char], version: Version) -> Vec<char> {
    let spellings = match version {
        Version::V2025 => &tables().spellings,
        Version::V2026 => &tables().spellings_2026,
    };
    map_words(s, spellings)
}

/// `" ".join(mapping.get(word, word) for word in s.split())`.
fn map_words(s: &[char], mapping: &HashMap<&str, &str>) -> Vec<char> {
    let mut out = String::with_capacity(s.len());
    for (i, word) in split_words(s).iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(mapping.get(word.as_str()).copied().unwrap_or(word));
    }
    out.chars().collect()
}

/// `EnglishNameNormalizer` of 2026.
fn standardize_names(s: &[char]) -> Vec<char> {
    map_words(s, &tables().names)
}

/// `EnglishAcronymNormalizer` of 2026. A run of words that are one letter or digit each joins
/// into one word when it has at least two, or three if one of them is "a" or "i".
fn collapse_acronyms(s: &[char]) -> Vec<char> {
    let words = split_words(s);
    let single = |w: &str| {
        let mut chars = w.chars();
        matches!((chars.next(), chars.next()), (Some(c), None) if is_word(c) && c != '_')
    };
    let mut out: Vec<String> = Vec::with_capacity(words.len());
    let mut i = 0;
    while i < words.len() {
        if !single(&words[i]) {
            out.push(words[i].clone());
            i += 1;
            continue;
        }
        let end = words[i..]
            .iter()
            .position(|w| !single(w))
            .map_or(words.len(), |k| i + k);
        let run = &words[i..end];
        let min = if run.iter().any(|w| w == "a" || w == "i") {
            3
        } else {
            2
        };
        if run.len() >= min {
            out.push(run.concat());
        } else {
            out.extend(run.iter().cloned());
        }
        i = end;
    }
    out.join(" ").chars().collect()
}

/// `[.$¢€£]([^0-9])` becomes ` \1`, then `([^0-9])%` becomes `\1 `. These remove the symbols
/// that kept their place for the numbers but ended up with no number beside them.
fn remove_lone_symbols(s: &[char]) -> Vec<char> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if i + 1 < s.len()
            && matches!(s[i], '.' | '$' | '¢' | '€' | '£')
            && !s[i + 1].is_ascii_digit()
        {
            out.push(' ');
            out.push(s[i + 1]);
            i += 2;
        } else {
            out.push(s[i]);
            i += 1;
        }
    }
    let s = out;
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if i + 1 < s.len() && !s[i].is_ascii_digit() && s[i + 1] == '%' {
            out.push(s[i]);
            out.push(' ');
            i += 2;
        } else {
            out.push(s[i]);
            i += 1;
        }
    }
    out
}

/// `\s+` becomes a single space.
pub(crate) fn collapse_spaces(s: &[char]) -> Vec<char> {
    let mut out = Vec::with_capacity(s.len());
    let mut in_run = false;
    for &c in s {
        if !is_space(c) {
            out.push(c);
        } else if !in_run {
            out.push(' ');
        }
        in_run = is_space(c);
    }
    out
}

// `EnglishNumberNormalizer`, which writes spelled-out numbers as digits.

fn standardize_numbers(s: &[char], version: Version) -> Vec<char> {
    let s = preprocess_numbers(s);
    let words = split_words(&s);
    let joined: Vec<char> = process_words(&words, version).join(" ").chars().collect();
    postprocess_numbers(&joined)
}

fn zero(word: &str) -> bool {
    matches!(word, "o" | "oh" | "zero")
}

const ONES: [&str; 19] = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];

fn ones(word: &str) -> Option<u32> {
    let index = ONES.iter().position(|&name| name == word)?;
    Some(index as u32 + 1)
}

/// Plurals ("sixes") and ordinals ("fifth") of the ones. The ordinals are built as Whisper builds
/// them, so "nineth" is one and "ninth" isn't.
fn ones_suffixed(word: &str) -> Option<(u32, &'static str)> {
    match word {
        "zeroth" => return Some((0, "th")),
        "first" => return Some((1, "st")),
        "second" => return Some((2, "nd")),
        "third" => return Some((3, "rd")),
        "fifth" => return Some((5, "th")),
        "twelfth" => return Some((12, "th")),
        _ => {}
    }
    for (index, name) in ONES.iter().enumerate() {
        let value = index as u32 + 1;
        let Some(rest) = word.strip_prefix(name) else {
            continue;
        };
        let plural = if *name == "six" { "es" } else { "s" };
        let ordinal = if name.ends_with('t') { "h" } else { "th" };
        if rest == plural {
            return Some((value, "s"));
        }
        if rest == ordinal && value > 3 && value != 5 && value != 12 {
            return Some((value, "th"));
        }
    }
    None
}

fn tens(word: &str) -> Option<u32> {
    Some(match word {
        "twenty" => 20,
        "thirty" => 30,
        "forty" => 40,
        "fifty" => 50,
        "sixty" => 60,
        "seventy" => 70,
        "eighty" => 80,
        "ninety" => 90,
        _ => return None,
    })
}

/// "twenties" and "twentieth" and so on.
fn tens_suffixed(word: &str) -> Option<(u32, &'static str)> {
    if let Some(stem) = word.strip_suffix("ies") {
        return tens(&format!("{stem}y")).map(|value| (value, "s"));
    }
    let stem = word.strip_suffix("ieth")?;
    tens(&format!("{stem}y")).map(|value| (value, "th"))
}

/// A multiplier's power of ten.
fn multiplier(word: &str) -> Option<usize> {
    Some(match word {
        "hundred" => 2,
        "thousand" => 3,
        "million" => 6,
        "billion" => 9,
        "trillion" => 12,
        "quadrillion" => 15,
        "quintillion" => 18,
        "sextillion" => 21,
        "septillion" => 24,
        "octillion" => 27,
        "nonillion" => 30,
        "decillion" => 33,
        _ => return None,
    })
}

fn multiplier_suffixed(word: &str) -> Option<(usize, &'static str)> {
    if let Some(power) = word.strip_suffix('s').and_then(multiplier) {
        return Some((power, "s"));
    }
    word.strip_suffix("th")
        .and_then(multiplier)
        .map(|power| (power, "th"))
}

fn preceding_prefixer(word: &str) -> Option<char> {
    match word {
        "minus" | "negative" => Some('-'),
        "plus" | "positive" => Some('+'),
        _ => None,
    }
}

fn following_prefixer(word: &str) -> Option<char> {
    match word {
        "pound" | "pounds" => Some('£'),
        "euro" | "euros" => Some('€'),
        "dollar" | "dollars" => Some('$'),
        "cent" | "cents" => Some('¢'),
        _ => None,
    }
}

fn suffixer(word: &str) -> bool {
    matches!(word, "per" | "percent")
}

fn special(word: &str) -> bool {
    matches!(word, "and" | "double" | "triple" | "point")
}

/// `self.decimals`, the words that can follow "point".
fn decimal_word(word: &str) -> bool {
    ones(word).is_some() || tens(word).is_some() || zero(word)
}

/// `self.words`, every word the number normalizer acts on.
fn number_word(word: &str) -> bool {
    zero(word)
        || ones(word).is_some()
        || ones_suffixed(word).is_some()
        || tens(word).is_some()
        || tens_suffixed(word).is_some()
        || multiplier(word).is_some()
        || multiplier_suffixed(word).is_some()
        || preceding_prefixer(word).is_some()
        || following_prefixer(word).is_some()
        || suffixer(word)
        || special(word)
}

/// `^\d+(\.\d+)?$`.
fn numeric(word: &str) -> bool {
    let digits = |part: &str| !part.is_empty() && part.chars().all(is_decimal);
    match word.split_once('.') {
        Some((whole, fraction)) => digits(whole) && digits(fraction),
        None => digits(word),
    }
}

/// `EnglishNumberNormalizer.preprocess`.
fn preprocess_numbers(s: &[char]) -> Vec<char> {
    // re.split(r"\band\s+a\s+half\b", s), then "<number> and a half" becomes
    // "<number> point five".
    let mut segments: Vec<&[char]> = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < s.len() {
        match and_a_half(s, i) {
            Some(end) => {
                segments.push(&s[start..i]);
                start = end;
                i = end;
            }
            None => i += 1,
        }
    }
    segments.push(&s[start..]);
    let mut results: Vec<String> = Vec::new();
    for (index, segment) in segments.iter().enumerate() {
        if segment.iter().all(|&c| is_space(c)) {
            // Python skips the segment and the "and a half" after it.
            continue;
        }
        results.push(segment.iter().collect());
        if index + 1 < segments.len() {
            let words = split_words(segment);
            let last = words.last().map_or("", String::as_str);
            if decimal_word(last) || multiplier(last).is_some() {
                results.push("point five".to_string());
            } else {
                results.push("and a half".to_string());
            }
        }
    }
    let s: Vec<char> = results.join(" ").chars().collect();

    // ([a-z])([0-9]) becomes "\1 \2", then ([0-9])([a-z]) does.
    let s = space_between(&s, |c| c.is_ascii_lowercase(), |c| c.is_ascii_digit());
    let s = space_between(&s, |c| c.is_ascii_digit(), |c| c.is_ascii_lowercase());

    // ([0-9])\s+(st|nd|rd|th|s)\b becomes \1\2.
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if s[i].is_ascii_digit() && i + 1 < s.len() && is_space(s[i + 1]) {
            let after = s[i + 1..]
                .iter()
                .position(|&c| !is_space(c))
                .map_or(s.len(), |k| i + 1 + k);
            let suffix = ["st", "nd", "rd", "th", "s"].into_iter().find(|suffix| {
                starts_with(&s, after, suffix) && boundary(&s, after + suffix.len())
            });
            if let Some(suffix) = suffix {
                out.push(s[i]);
                out.extend(suffix.chars());
                i = after + suffix.len();
                continue;
            }
        }
        out.push(s[i]);
        i += 1;
    }
    out
}

/// The end of a match of `\band\s+a\s+half\b` at `i`.
fn and_a_half(s: &[char], i: usize) -> Option<usize> {
    if !boundary(s, i) || !starts_with(s, i, "and") {
        return None;
    }
    let spaces = |from: usize| {
        let end = s[from.min(s.len())..]
            .iter()
            .position(|&c| !is_space(c))
            .map_or(s.len(), |k| from + k);
        (end > from).then_some(end)
    };
    let j = spaces(i + 3)?;
    if !starts_with(s, j, "a") {
        return None;
    }
    let j = spaces(j + 1)?;
    (starts_with(s, j, "half") && boundary(s, j + 4)).then_some(j + 4)
}

/// `(x)(y)` becomes `\1 \2` for two single-character classes.
fn space_between(
    s: &[char],
    first: impl Fn(char) -> bool,
    second: impl Fn(char) -> bool,
) -> Vec<char> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if i + 1 < s.len() && first(s[i]) && second(s[i + 1]) {
            out.extend([s[i], ' ', s[i + 1]]);
            i += 2;
        } else {
            out.push(s[i]);
            i += 1;
        }
    }
    out
}

/// A number being read. Python keeps it as an int, or as a str once it stops being a plain
/// integer ("1.5", or digits read one by one such as "0 7").
#[derive(Clone)]
enum Value {
    /// An integer, as ASCII decimal digits without leading zeros.
    Int(String),
    Str(String),
}

impl Value {
    /// `str(value)`.
    fn text(&self) -> &str {
        match self {
            Value::Int(digits) | Value::Str(digits) => digits,
        }
    }

    /// `str(value or "")`, where 0 and "" count as false.
    fn text_or_empty(value: &Option<Value>) -> &str {
        match value {
            Some(Value::Int(digits)) if digits == "0" => "",
            Some(value) => value.text(),
            None => "",
        }
    }
}

/// The state of `process_words`, and its `output`.
#[derive(Default)]
struct Reader {
    out: Vec<String>,
    prefix: Option<char>,
    value: Option<Value>,
}

impl Reader {
    /// `yield output(result)`.
    fn emit(&mut self, result: &str) {
        let mut text = String::new();
        text.extend(self.prefix.take());
        text.push_str(result);
        self.value = None;
        self.out.push(text);
    }

    /// `if value is not None: yield output(value)`.
    fn emit_value(&mut self) {
        if let Some(value) = self.value.take() {
            self.emit(value.text());
        }
    }
}

/// `is_digit_token` of 2026, words that continue a run of digits ("four", "oh", "20").
fn is_digit_token(word: Option<&str>) -> bool {
    word.is_some_and(|w| {
        (!w.is_empty() && w.chars().all(is_decimal))
            || zero(w)
            || ones(w).is_some()
            || tens(w).is_some()
    })
}

/// `EnglishNumberNormalizer.process_words`.
fn process_words(words: &[String], version: Version) -> Vec<String> {
    let mut r = Reader::default();
    let mut skip = false;
    for (i, current) in words.iter().enumerate() {
        let current = current.as_str();
        let prev = if i > 0 {
            Some(words[i - 1].as_str())
        } else {
            None
        };
        let next = words.get(i + 1).map(String::as_str);
        if skip {
            skip = false;
            continue;
        }
        let next_is_numeric = next.is_some_and(numeric);
        let next_is_word = next.is_some_and(number_word);
        let prev_is_ones = prev.is_some_and(|p| ones(p).is_some());
        let prev_is_tens = prev.is_some_and(|p| tens(p).is_some());
        let first = current.chars().next().expect("split gives no empty words");
        let has_prefix = matches!(first, '-' | '+' | '£' | '€' | '$' | '¢');
        let without_prefix = if has_prefix {
            &current[first.len_utf8()..]
        } else {
            current
        };

        if numeric(without_prefix) {
            // Arabic numbers, possibly with a sign and a fraction.
            let number = Decimal::parse(without_prefix).expect("a numeric word parses");
            if let Some(value) = &r.value {
                if let Value::Str(text) = value
                    && text.ends_with('.')
                {
                    // Concatenate decimals or the parts of an IP address.
                    r.value = Some(Value::Str(format!("{text}{current}")));
                    continue;
                }
                r.emit_value();
            }
            if has_prefix {
                r.prefix = Some(first);
            }
            r.value = Some(match number.times_power_of_ten(0) {
                Some(integer) => Value::Int(integer),
                None => Value::Str(without_prefix.to_string()),
            });
        } else if !number_word(current) {
            r.emit_value();
            r.emit(current);
        } else if zero(current) {
            // In 2026 "oh" is a zero only inside a run of digits. One must follow, and with no
            // number pending a second must follow too ("oh seven nine").
            let in_number = is_digit_token(next)
                && (r.value.is_some() || is_digit_token(words.get(i + 2).map(String::as_str)));
            if version == Version::V2026 && current == "oh" && !in_number {
                r.emit_value();
                r.emit(current);
            } else {
                r.value = Some(Value::Str(format!("{}0", Value::text_or_empty(&r.value))));
            }
        } else if let Some(ones) = ones(current) {
            r.value = Some(match &r.value {
                None => Value::Int(ones.to_string()),
                Some(Value::Int(n)) if !prev_is_ones => add_or_append(n, ones),
                // A str, or an int right after another of the ones ("one two").
                Some(value) if prev_is_tens && ones < 10 => {
                    // Replace the zero of the tens with the digit.
                    Value::Str(format!("{}{ones}", drop_last(value.text())))
                }
                Some(value) => Value::Str(format!("{}{ones}", value.text())),
            });
        } else if let Some((ones, suffix)) = ones_suffixed(current) {
            let text = match &r.value {
                None => format!("{ones}{suffix}"),
                Some(Value::Int(n)) if !prev_is_ones => {
                    format!("{}{suffix}", add_or_append(n, ones).text())
                }
                Some(value) if prev_is_tens && ones < 10 => {
                    format!("{}{ones}{suffix}", drop_last(value.text()))
                }
                Some(value) => format!("{}{ones}{suffix}", value.text()),
            };
            r.emit(&text);
        } else if let Some(tens) = tens(current) {
            r.value = Some(match &r.value {
                None => Value::Int(tens.to_string()),
                Some(Value::Str(text)) => Value::Str(format!("{text}{tens}")),
                Some(Value::Int(n)) if divisible_by_100(n) => Value::Int(add(n, &tens.to_string())),
                Some(Value::Int(n)) => Value::Str(format!("{n}{tens}")),
            });
        } else if let Some((tens, suffix)) = tens_suffixed(current) {
            let text = match &r.value {
                None => format!("{tens}{suffix}"),
                Some(Value::Str(text)) => format!("{text}{tens}{suffix}"),
                Some(Value::Int(n)) if divisible_by_100(n) => {
                    format!("{}{suffix}", add(n, &tens.to_string()))
                }
                Some(Value::Int(n)) => format!("{n}{tens}{suffix}"),
            };
            r.emit(&text);
        } else if let Some(power) = multiplier(current) {
            match r.value.clone() {
                None => r.value = Some(Value::Int(power_of_ten(power))),
                Some(Value::Int(n)) if n != "0" => {
                    r.value = Some(Value::Int(scale_last_group(&n, power)));
                }
                Some(value) => {
                    match Decimal::parse(value.text()).and_then(|f| f.times_power_of_ten(power)) {
                        Some(product) => r.value = Some(Value::Int(product)),
                        None => {
                            r.emit_value();
                            r.value = Some(Value::Int(power_of_ten(power)));
                        }
                    }
                }
            }
        } else if let Some((power, suffix)) = multiplier_suffixed(current) {
            match r.value.clone() {
                None => r.emit(&format!("{}{suffix}", power_of_ten(power))),
                Some(Value::Str(text)) => {
                    match Decimal::parse(&text).and_then(|f| f.times_power_of_ten(power)) {
                        Some(product) => r.emit(&format!("{product}{suffix}")),
                        None => {
                            r.emit_value();
                            r.emit(&format!("{}{suffix}", power_of_ten(power)));
                        }
                    }
                }
                Some(Value::Int(n)) => {
                    r.emit(&format!("{}{suffix}", scale_last_group(&n, power)));
                }
            }
            r.value = None;
        } else if let Some(sign) = preceding_prefixer(current) {
            // A sign ("minus", "plus") applies when a number follows.
            r.emit_value();
            if next_is_word || next_is_numeric {
                r.prefix = Some(sign);
            } else {
                r.emit(current);
            }
        } else if let Some(currency) = following_prefixer(current) {
            // A currency ("dollars", "cents") applies after a number.
            if r.value.is_some() {
                r.prefix = Some(currency);
                r.emit_value();
            } else {
                r.emit(current);
            }
        } else if suffixer(current) {
            // "percent" and "per cent" become "%".
            match r.value.clone() {
                Some(value) if current == "per" => {
                    if next == Some("cent") {
                        r.emit(&format!("{}%", value.text()));
                        skip = true;
                    } else {
                        r.emit_value();
                        r.emit(current);
                    }
                }
                Some(value) => r.emit(&format!("{}%", value.text())),
                None => r.emit(current),
            }
        } else if special(current) {
            if !next_is_word && !next_is_numeric {
                // Only special when the next word can be part of a number.
                r.emit_value();
                r.emit(current);
            } else if current == "and" {
                // "and" after "hundred", "thousand" and so on disappears.
                if !prev.is_some_and(|p| multiplier(p).is_some()) {
                    r.emit_value();
                    r.emit(current);
                }
            } else if current == "double" || current == "triple" {
                let next = next.unwrap_or("");
                if ones(next).is_some() || zero(next) {
                    let repeats = if current == "double" { 2 } else { 3 };
                    let digit = ones(next).unwrap_or(0).to_string();
                    let text = format!(
                        "{}{}",
                        Value::text_or_empty(&r.value),
                        digit.repeat(repeats)
                    );
                    r.value = Some(Value::Str(text));
                    skip = true;
                } else {
                    r.emit_value();
                    r.emit(current);
                }
            } else if decimal_word(next.unwrap_or("")) || next_is_numeric {
                // "point" before a digit. Before any other number word it disappears.
                r.value = Some(Value::Str(format!("{}.", Value::text_or_empty(&r.value))));
            }
        }
    }
    r.emit_value();
    r.out
}

/// `value[:-1]`.
fn drop_last(text: &str) -> &str {
    text.char_indices().last().map_or(text, |(i, _)| &text[..i])
}

/// The Python for ones after an integer. Below ten, `value += ones` when `value % 10 == 0`.
/// From ten up, when `value % 100 == 0`. Otherwise the digits are appended.
fn add_or_append(n: &str, ones: u32) -> Value {
    let fits = if ones < 10 {
        n.ends_with('0')
    } else {
        divisible_by_100(n)
    };
    if fits {
        Value::Int(add(n, &ones.to_string()))
    } else {
        Value::Str(format!("{n}{ones}"))
    }
}

fn divisible_by_100(n: &str) -> bool {
    n == "0" || n.ends_with("00")
}

fn power_of_ten(power: usize) -> String {
    format!("1{}", "0".repeat(power))
}

/// `value // 1000 * 1000 + value % 1000 * multiplier`.
fn scale_last_group(n: &str, power: usize) -> String {
    let split = n.len().saturating_sub(3);
    let (before, residual) = n.split_at(split);
    let residual = canonical(residual);
    let scaled = if residual == "0" {
        residual
    } else {
        format!("{residual}{}", "0".repeat(power))
    };
    let thousands = if before.is_empty() {
        "0".to_string()
    } else {
        format!("{before}000")
    };
    add(&thousands, &scaled)
}

/// The sum of two integers written as ASCII digits.
fn add(a: &str, b: &str) -> String {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut digits = Vec::with_capacity(a.len().max(b.len()) + 1);
    let mut carry = 0;
    for k in 0..a.len().max(b.len()) {
        let x = if k < a.len() {
            a[a.len() - 1 - k] - b'0'
        } else {
            0
        };
        let y = if k < b.len() {
            b[b.len() - 1 - k] - b'0'
        } else {
            0
        };
        let sum = x + y + carry;
        digits.push(b'0' + sum % 10);
        carry = sum / 10;
    }
    if carry > 0 {
        digits.push(b'0' + carry);
    }
    digits.reverse();
    canonical(&String::from_utf8(digits).expect("ASCII digits"))
}

/// Digits without leading zeros, and "0" for none.
fn canonical(digits: &str) -> String {
    let trimmed = digits.trim_start_matches('0');
    if trimmed.is_empty() {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

/// What `Fraction(text)` makes of a run of decimal digits of any script with at most one period,
/// such as "12", "1.5", "1." or ".5".
struct Decimal {
    /// The integer part as ASCII digits.
    whole: String,
    /// The digits after the period, as ASCII digits.
    fraction: String,
}

impl Decimal {
    fn parse(text: &str) -> Option<Decimal> {
        let (whole, fraction) = text.split_once('.').unwrap_or((text, ""));
        let ascii = |part: &str| -> Option<String> {
            part.chars()
                .map(|c| {
                    is_decimal(c).then(|| char::from_digit(digit_value(c), 10).expect("digit"))
                })
                .collect()
        };
        let (whole, fraction) = (ascii(whole)?, ascii(fraction)?);
        if whole.is_empty() && fraction.is_empty() {
            return None;
        }
        Some(Decimal { whole, fraction })
    }

    /// The number times 10 to the power, when that is an integer, as `Fraction` would give its
    /// numerator.
    fn times_power_of_ten(&self, power: usize) -> Option<String> {
        let (kept, rest) = self.fraction.split_at(power.min(self.fraction.len()));
        if rest.bytes().any(|d| d != b'0') {
            return None;
        }
        let padding = "0".repeat(power - kept.len());
        Some(canonical(&format!("{}{kept}{padding}", self.whole)))
    }
}

/// `EnglishNumberNormalizer.postprocess`.
fn postprocess_numbers(s: &[char]) -> Vec<char> {
    // ([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b becomes "<currency><integer>.<cents:02d>", so
    // "$2 and ¢7" becomes "$2.07".
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if let Some((end, text)) = combine_cents(s, i) {
            out.extend(text.chars());
            i = end;
        } else {
            out.push(s[i]);
            i += 1;
        }
    }

    // [€£$]0.([0-9]{1,2})\b becomes "¢<cents>". The "." is any character but a newline.
    let s = out;
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if matches!(s[i], '€' | '£' | '$')
            && i + 2 < s.len()
            && s[i + 1] == '0'
            && s[i + 2] != '\n'
            && let Some((end, cents)) = cents(&s, i + 3)
        {
            out.push('¢');
            out.extend(cents.to_string().chars());
            i = end;
        } else {
            out.push(s[i]);
            i += 1;
        }
    }

    // \b1(s?)\b becomes "one\1".
    let s = out;
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if s[i] == '1' && boundary(&s, i) {
            if i + 1 < s.len() && s[i + 1] == 's' && boundary(&s, i + 2) {
                out.extend("ones".chars());
                i += 2;
                continue;
            }
            if boundary(&s, i + 1) {
                out.extend("one".chars());
                i += 1;
                continue;
            }
        }
        out.push(s[i]);
        i += 1;
    }
    out
}

/// The end of a match of `([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b` at `i`, and its replacement.
fn combine_cents(s: &[char], i: usize) -> Option<(usize, String)> {
    if !matches!(s[i], '€' | '£' | '$') {
        return None;
    }
    let digits_end = s[i + 1..]
        .iter()
        .position(|c| !c.is_ascii_digit())
        .map_or(s.len(), |k| i + 1 + k);
    if digits_end == i + 1 || digits_end >= s.len() || s[digits_end] != ' ' {
        return None;
    }
    let mut j = digits_end + 1;
    if starts_with(s, j, "and ") {
        j += 4;
    }
    if j >= s.len() || s[j] != '¢' {
        return None;
    }
    let (end, cents) = cents(s, j + 1)?;
    let integer: String = s[i + 1..digits_end].iter().collect();
    Some((end, format!("{}{integer}.{cents:02}", s[i])))
}

/// The end of a match of `([0-9]{1,2})\b` at `i`, and the number it holds.
fn cents(s: &[char], i: usize) -> Option<(usize, u32)> {
    let digit = |k: usize| {
        s.get(k)
            .filter(|c| c.is_ascii_digit())
            .map(|c| c.to_digit(10).unwrap())
    };
    let first = digit(i)?;
    if let Some(second) = digit(i + 1)
        && boundary(s, i + 2)
    {
        return Some((i + 2, first * 10 + second));
    }
    boundary(s, i + 1).then_some((i + 1, first))
}
