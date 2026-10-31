//! Session-word biasing: decoding that favours a list of special words, such as the names and
//! technical terms of a conversation, with little change to ordinary words.
//!
//! Each special word is split into the model's tokens in its usual spelling and capitalized.
//!
//! - Parakeet decodes with a beam search over a prefix tree of those tokens with failure links
//!   (Aho-Corasick). A token of a special word under way earns a provisional bonus, which becomes
//!   final only for words completed and followed by a word boundary or the end of the audio.
//! - Granite decodes greedily, then spots each special word with a CTC alignment of free start
//!   and end. Where the alignment plus a bonus beats the greedy words over the same frames by a
//!   margin, the special word replaces them.
//!
//! Special words are words written with spaces between them. Scripts written without spaces,
//! such as Japanese, aren't supported yet. With no special words, both models decode greedily,
//! exactly as without biasing.

mod spotting;
pub(crate) mod tdt_beam;

use std::collections::VecDeque;
use std::fmt;

use crate::error::{Error, Result};
use crate::tokenizer::Tokenizer;

pub(crate) use spotting::{score_row, spot};

/// How strongly the decoders favour special words. The defaults were tuned on LibriSpeech.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Boost {
    bonus: f32,
    beam: usize,
    spotting_bonus: f32,
    spotting_margin: f32,
    short_word_margin: f32,
}

impl Default for Boost {
    fn default() -> Self {
        Boost {
            bonus: 1.5,
            beam: 4,
            spotting_bonus: 4.0,
            spotting_margin: 2.0,
            short_word_margin: SHORT_WORD_MARGIN,
        }
    }
}

/// The default extra spotting margin for each letter a word has below
/// [`Boost::SHORT_WORD_LETTERS`].
const SHORT_WORD_MARGIN: f32 = 3.0;

impl Boost {
    /// The largest bonus a token may earn, in log probability.
    pub const MAX_BONUS: f32 = 10.0;
    /// The most hypotheses Parakeet's beam search may keep.
    pub const MAX_BEAM: usize = 16;
    /// The largest spotting margin, in log probability.
    pub const MAX_MARGIN: f32 = 50.0;
    /// Spotted words with fewer letters or digits than this need a larger margin.
    pub const SHORT_WORD_LETTERS: usize = 5;

    /// Settings for both models. `bonus` is Parakeet's bonus for each token of a special word and
    /// `beam` the hypotheses its beam search keeps. `spotting_bonus` is Granite's bonus for each
    /// token of a spotted word, and `spotting_margin` how much a spotted word with its bonus must
    /// beat the greedy words it replaces. Bonuses and margins are in log probability, from 0 to
    /// [`Boost::MAX_BONUS`] and [`Boost::MAX_MARGIN`], and the beam from 1 to [`Boost::MAX_BEAM`].
    pub fn new(
        bonus: f32,
        beam: usize,
        spotting_bonus: f32,
        spotting_margin: f32,
    ) -> Result<Boost> {
        let in_range = |x: f32, max: f32| (0.0..=max).contains(&x);
        if !in_range(bonus, Boost::MAX_BONUS) || !in_range(spotting_bonus, Boost::MAX_BONUS) {
            return Err(Error::Setting(format!(
                "A bonus must be from 0 to {}.",
                Boost::MAX_BONUS
            )));
        }
        if !in_range(spotting_margin, Boost::MAX_MARGIN) {
            return Err(Error::Setting(format!(
                "The spotting margin must be from 0 to {}.",
                Boost::MAX_MARGIN
            )));
        }
        if !(1..=Boost::MAX_BEAM).contains(&beam) {
            return Err(Error::Setting(format!(
                "The beam must be from 1 to {}.",
                Boost::MAX_BEAM
            )));
        }
        Ok(Boost {
            bonus,
            beam,
            spotting_bonus,
            spotting_margin,
            ..Boost::default()
        })
    }

    /// Sets how much more a spotted word must beat the greedy words for each letter or digit it
    /// has below [`Boost::SHORT_WORD_LETTERS`], from 0 to [`Boost::MAX_MARGIN`]. A short word
    /// carries little sound, so it needs a stronger case to replace a common word.
    pub fn with_short_word_margin(self, margin: f32) -> Result<Boost> {
        if !(0.0..=Boost::MAX_MARGIN).contains(&margin) {
            return Err(Error::Setting(format!(
                "The short-word margin must be from 0 to {}.",
                Boost::MAX_MARGIN
            )));
        }
        Ok(Boost {
            short_word_margin: margin,
            ..self
        })
    }

    pub fn bonus(&self) -> f32 {
        self.bonus
    }

    pub fn beam(&self) -> usize {
        self.beam
    }

    pub fn spotting_bonus(&self) -> f32 {
        self.spotting_bonus
    }

    pub fn spotting_margin(&self) -> f32 {
        self.spotting_margin
    }

    pub fn short_word_margin(&self) -> f32 {
        self.short_word_margin
    }
}

/// Why a special word was left out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejection {
    /// It has no letter or digit.
    Empty,
    /// It has one or two letters or digits, and wasn't added with
    /// [`SpecialWords::add_code_word`].
    TooShort,
    /// It has more than [`SpecialWords::MAX_CHARS`] characters or [`SpecialWords::MAX_PARTS`]
    /// words.
    TooLong,
    /// It has control characters, combining marks (give it composed, in NFC) or characters of a
    /// script written without spaces, such as Japanese.
    UnsupportedCharacters,
    /// It's in the list already.
    Duplicate,
    /// The list holds [`SpecialWords::MAX_WORDS`] words already.
    ListFull,
    /// The model's vocabulary can't spell it as written.
    NotInVocabulary,
    /// It takes more than [`SpecialWords::MAX_TOKENS`] of the model's tokens.
    TooManyTokens,
}

impl fmt::Display for Rejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Rejection::Empty => "It has no letter or digit.",
            Rejection::TooShort => "It has fewer than three letters or digits.",
            Rejection::TooLong => "It's too long.",
            Rejection::UnsupportedCharacters => "It has characters biasing doesn't support.",
            Rejection::Duplicate => "It's in the list already.",
            Rejection::ListFull => "The list is full.",
            Rejection::NotInVocabulary => "The model can't spell it as written.",
            Rejection::TooManyTokens => "It takes too many of the model's tokens.",
        })
    }
}

impl std::error::Error for Rejection {}

/// Words to favour while decoding, each in its usual spelling.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpecialWords {
    words: Vec<String>,
    boost: Boost,
}

impl SpecialWords {
    /// The most words a list holds.
    pub const MAX_WORDS: usize = 1000;
    /// The most characters a word may have, spaces included.
    pub const MAX_CHARS: usize = 64;
    /// The most words separated by spaces an entry may have.
    pub const MAX_PARTS: usize = 5;
    /// The most tokens a spelling may take in a model's vocabulary. Longer ones are left out.
    pub const MAX_TOKENS: usize = 32;

    pub fn new() -> SpecialWords {
        SpecialWords::default()
    }

    /// Adds a word, or several separated by spaces that are favoured together ("Visual Studio
    /// Code"). Words of one or two letters or digits are refused, since favouring them would
    /// turn short ordinary words into them.
    pub fn add(&mut self, word: &str) -> Result<(), Rejection> {
        let word = normalize(word)?;
        match word.chars().filter(|c| c.is_alphanumeric()).count() {
            0 => Err(Rejection::Empty),
            1 | 2 => Err(Rejection::TooShort),
            _ => self.insert(word),
        }
    }

    /// Adds a word seen in code, a path or an identifier, which keeps it even when it has only
    /// one or two letters.
    pub fn add_code_word(&mut self, word: &str) -> Result<(), Rejection> {
        self.insert(normalize(word)?)
    }

    fn insert(&mut self, word: String) -> Result<(), Rejection> {
        if !word.chars().any(char::is_alphanumeric) {
            return Err(Rejection::Empty);
        }
        if word
            .chars()
            .any(|c| c.is_control() || is_combining(c) || is_unspaced_script(c))
        {
            return Err(Rejection::UnsupportedCharacters);
        }
        if self.words.contains(&word) {
            return Err(Rejection::Duplicate);
        }
        if self.words.len() == SpecialWords::MAX_WORDS {
            return Err(Rejection::ListFull);
        }
        self.words.push(word);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    pub fn words(&self) -> impl Iterator<Item = &str> {
        self.words.iter().map(String::as_str)
    }

    pub fn boost(&self) -> Boost {
        self.boost
    }

    pub fn set_boost(&mut self, boost: Boost) {
        self.boost = boost;
    }
}

/// `word` with each run of whitespace as one space and none at its ends. It stops at the first
/// part or character past [`SpecialWords::MAX_PARTS`] and [`SpecialWords::MAX_CHARS`], so a long
/// input costs no more than a short one.
fn normalize(word: &str) -> Result<String, Rejection> {
    let mut out = String::new();
    let (mut chars, mut parts, mut space) = (0, 0, false);
    for c in word.chars() {
        if c.is_whitespace() {
            space = !out.is_empty();
            continue;
        }
        if space || out.is_empty() {
            if space {
                out.push(' ');
                chars += 1;
                space = false;
            }
            parts += 1;
            if parts > SpecialWords::MAX_PARTS {
                return Err(Rejection::TooLong);
            }
        }
        out.push(c);
        chars += 1;
        if chars > SpecialWords::MAX_CHARS {
            return Err(Rejection::TooLong);
        }
    }
    Ok(out)
}

/// Combining marks, which a word should hold composed instead (NFC), as the models write it.
fn is_combining(c: char) -> bool {
    matches!(c, '\u{0300}'..='\u{036f}' | '\u{1ab0}'..='\u{1aff}' | '\u{1dc0}'..='\u{1dff}' | '\u{20d0}'..='\u{20ff}' | '\u{fe20}'..='\u{fe2f}')
}

/// Characters of scripts written without spaces between words: Chinese, Japanese, Thai, Lao,
/// Khmer and Myanmar, and full-width forms.
fn is_unspaced_script(c: char) -> bool {
    matches!(c, '\u{0e00}'..='\u{0eff}' | '\u{1000}'..='\u{109f}' | '\u{1780}'..='\u{17ff}' | '\u{3000}'..='\u{30ff}' | '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}' | '\u{ff00}'..='\u{ffef}')
}

/// A special word that tokenizes to a single piece is one the model knows well, so it earns
/// this share of the bonus.
const SINGLE_PIECE_SHARE: f32 = 0.5;

/// The tree's root, where no special word is under way.
const ROOT: u32 = 0;
/// No node.
const NONE: u32 = u32::MAX;

#[derive(Debug, Clone)]
struct Node {
    /// (token, node), sorted by token.
    children: Vec<(u32, u32)>,
    /// The node of the longest proper suffix of this node's tokens that is in the tree.
    fail: u32,
    /// Tokens from the root.
    depth: u32,
    /// The deepest node among this one and those its failure links lead to where a special word
    /// ends, or `NONE`.
    output: u32,
}

/// Where a hypothesis stands in the matcher. Its last `depth(node)` tokens are the longest
/// special word under way, and `covered` marks those already in a completed word (bit `i` is
/// the token `i` places from the end), `half` those paid at a single-token word's share.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct MatchState {
    node: u32,
    covered: u64,
    half: u64,
}

impl MatchState {
    /// No special word under way.
    pub(crate) const START: MatchState = MatchState {
        node: ROOT,
        covered: 0,
        half: 0,
    };
}

/// Matches special words in a growing token sequence and keeps two kinds of bonus apart. The
/// committed bonus pays each token of a completed word once, at the word's weight, so
/// overlapping words ("New" and "New York") never pay a token twice. The provisional bonus pays
/// the tokens of the longest word under way that no completed word covers, and is given back
/// when that word dies.
#[derive(Debug, Clone)]
pub(crate) struct Matcher {
    nodes: Vec<Node>,
    /// Whether a word may end before each token of the vocabulary.
    boundary: Vec<bool>,
    /// The bonus for each token.
    bonus: f32,
}

fn mask(depth: u32) -> u64 {
    if depth >= 64 {
        u64::MAX
    } else {
        (1u64 << depth) - 1
    }
}

impl Matcher {
    /// A matcher of the token sequences `forms`, each starting with a word-start token and at
    /// most [`SpecialWords::MAX_TOKENS`] long.
    fn new(forms: &[Vec<u32>], tokenizer: &Tokenizer, bonus: f32) -> Matcher {
        let mut nodes = vec![Node {
            children: Vec::new(),
            fail: ROOT,
            depth: 0,
            output: NONE,
        }];
        for form in forms {
            debug_assert!(form.len() <= SpecialWords::MAX_TOKENS);
            let mut node = ROOT;
            for &token in form {
                node = match child(&nodes, node, token) {
                    Some(c) => c,
                    None => {
                        let new = nodes.len() as u32;
                        let depth = nodes[node as usize].depth + 1;
                        nodes.push(Node {
                            children: Vec::new(),
                            fail: ROOT,
                            depth,
                            output: NONE,
                        });
                        let children = &mut nodes[node as usize].children;
                        let at = children.partition_point(|&(t, _)| t < token);
                        children.insert(at, (token, new));
                        new
                    }
                };
            }
            nodes[node as usize].output = node;
        }
        // Failure and output links, breadth first, as Aho-Corasick builds them.
        let mut queue: VecDeque<u32> = nodes[ROOT as usize]
            .children
            .iter()
            .map(|&(_, c)| c)
            .collect();
        while let Some(node) = queue.pop_front() {
            let fail_output = nodes[nodes[node as usize].fail as usize].output;
            if nodes[node as usize].output == NONE {
                nodes[node as usize].output = fail_output;
            }
            for (token, c) in nodes[node as usize].children.clone() {
                let mut f = nodes[node as usize].fail;
                let fail = loop {
                    if let Some(next) = child(&nodes, f, token).filter(|&n| n != c) {
                        break next;
                    }
                    if f == ROOT {
                        break ROOT;
                    }
                    f = nodes[f as usize].fail;
                };
                nodes[c as usize].fail = fail;
                queue.push_back(c);
            }
        }
        Matcher {
            nodes,
            boundary: (0..tokenizer.len() as u32)
                .map(|id| tokenizer.is_word_boundary(id))
                .collect(),
            bonus,
        }
    }

    fn is_boundary(&self, token: u32) -> bool {
        self.boundary.get(token as usize).copied().unwrap_or(true)
    }

    fn depth(&self, node: u32) -> u32 {
        self.nodes[node as usize].depth
    }

    /// The provisional bonus a hypothesis in state `s` holds.
    fn provisional(&self, s: &MatchState) -> f32 {
        self.bonus * (self.depth(s.node) - s.covered.count_ones()) as f32
    }

    /// Completes the deepest word that ends where `s` stands, and returns what it adds to the
    /// committed bonus: its weight for each of its tokens not covered yet, and the rest of it for
    /// tokens a single-token word covered at its share.
    fn commit(&self, s: &mut MatchState) -> f32 {
        let output = self.nodes[s.node as usize].output;
        if output == NONE {
            return 0.0;
        }
        let depth = self.depth(output);
        let window = mask(depth);
        let new = window & !s.covered;
        s.covered |= window;
        if depth == 1 {
            s.half |= new;
            new.count_ones() as f32 * self.bonus * SINGLE_PIECE_SHARE
        } else {
            let upgraded = window & s.half;
            s.half &= !window;
            (new.count_ones() as f32 + upgraded.count_ones() as f32 * (1.0 - SINGLE_PIECE_SHARE))
                * self.bonus
        }
    }

    /// The node after `token` from `node`, following failure links.
    fn goto(&self, mut node: u32, token: u32) -> u32 {
        loop {
            if let Some(c) = child(&self.nodes, node, token) {
                return c;
            }
            if node == ROOT {
                return ROOT;
            }
            node = self.nodes[node as usize].fail;
        }
    }

    /// The state after `token`, and the change in the bonus the hypothesis holds: what words
    /// completed before a word boundary commit, plus the change in provisional bonus.
    pub(crate) fn step(&self, s: MatchState, token: u32) -> (MatchState, f32) {
        let before = self.provisional(&s);
        let mut s = s;
        let committed = if self.is_boundary(token) {
            self.commit(&mut s)
        } else {
            0.0
        };
        let node = self.goto(s.node, token);
        let window = mask(self.depth(node));
        let next = MatchState {
            node,
            covered: (s.covered << 1) & window,
            half: (s.half << 1) & window,
        };
        (next, committed + self.provisional(&next) - before)
    }

    /// The change in bonus when the audio ends in state `s`. A word that ends there completes,
    /// and the rest of the provisional bonus goes.
    pub(crate) fn finish(&self, s: MatchState) -> f32 {
        let before = self.provisional(&s);
        let mut s = s;
        self.commit(&mut s) - before
    }

    /// The change in bonus of every token that neither continues nor starts a special word from
    /// `s`, for word boundaries and for other tokens. [`Matcher::special_tokens`] lists the rest.
    pub(crate) fn default_steps(&self, s: MatchState) -> (f32, f32) {
        let before = self.provisional(&s);
        let mut committed = s;
        (self.commit(&mut committed) - before, -before)
    }

    /// The tokens that continue a special word from `s` or start one, perhaps with repeats.
    pub(crate) fn special_tokens(&self, s: MatchState) -> impl Iterator<Item = u32> + '_ {
        let mut node = Some(s.node);
        std::iter::from_fn(move || {
            let n = node?;
            node = (n != ROOT).then(|| self.nodes[n as usize].fail);
            Some(self.nodes[n as usize].children.iter().map(|&(t, _)| t))
        })
        .flatten()
    }
}

fn child(nodes: &[Node], node: u32, token: u32) -> Option<u32> {
    let children = &nodes[node as usize].children;
    children
        .binary_search_by_key(&token, |&(t, _)| t)
        .ok()
        .map(|i| children[i].1)
}

/// One spelling of a special word, for word spotting.
#[derive(Debug, Clone)]
pub(crate) struct Form {
    /// The spelling's tokens.
    pub tokens: Vec<u32>,
    /// The column of each token in [`score_row`]'s rows.
    columns: Vec<usize>,
    /// The bonus for spotting it.
    bonus: f32,
    /// How much a spotting of it must beat the greedy words by.
    margin: f32,
}

impl Form {
    /// The column in [`score_row`]'s rows of alignment state `s`. The even states are blanks
    /// (before, between and after the tokens), the odd ones the tokens.
    fn column(&self, s: usize) -> usize {
        if s.is_multiple_of(2) {
            1
        } else {
            self.columns[s / 2]
        }
    }

    /// Whether state `s` may follow state `s - 2`, skipping the blank between two different
    /// tokens.
    fn may_skip(&self, s: usize) -> bool {
        !s.is_multiple_of(2) && s >= 3 && self.tokens[s / 2] != self.tokens[s / 2 - 1]
    }
}

/// Special words split into one model's tokens, ready for decoding.
#[derive(Debug, Clone)]
pub(crate) struct Compiled {
    pub matcher: Matcher,
    pub forms: Vec<Form>,
    /// The tokens of all forms, sorted. Word spotting reads their logits.
    pub tokens: Vec<u32>,
    pub boost: Boost,
}

impl Compiled {
    /// Splits the words into the tokenizer's tokens, and lists the words with no spelling the
    /// model can use.
    pub fn new(
        words: &SpecialWords,
        tokenizer: &Tokenizer,
    ) -> (Compiled, Vec<(String, Rejection)>) {
        let boost = words.boost;
        let mut forms: Vec<Vec<u32>> = Vec::new();
        let mut letters: Vec<usize> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut unusable = Vec::new();
        for word in words.words() {
            let mut chars = word.chars();
            let capitalized: String = chars
                .next()
                .map(|c| c.to_uppercase().chain(chars).collect())
                .unwrap_or_default();
            let mut problem = None;
            let mut usable = false;
            for spelling in [word, capitalized.as_str()] {
                let tokens = tokenizer.encode(spelling);
                if tokens.is_empty() || tokenizer.decode(&tokens) != spelling {
                    problem.get_or_insert(Rejection::NotInVocabulary);
                } else if tokens.len() > SpecialWords::MAX_TOKENS {
                    problem = Some(Rejection::TooManyTokens);
                } else {
                    usable = true;
                    if seen.insert(tokens.clone()) {
                        forms.push(tokens);
                        letters.push(spelling.chars().filter(|c| c.is_alphanumeric()).count());
                    }
                }
            }
            if !usable {
                unusable.push((
                    word.to_string(),
                    problem.unwrap_or(Rejection::NotInVocabulary),
                ));
            }
        }
        let matcher = if forms.is_empty() {
            Matcher {
                nodes: Vec::new(),
                boundary: Vec::new(),
                bonus: boost.bonus,
            }
        } else {
            Matcher::new(&forms, tokenizer, boost.bonus)
        };
        let mut tokens: Vec<u32> = forms.iter().flatten().copied().collect();
        tokens.sort_unstable();
        tokens.dedup();
        let forms = forms
            .into_iter()
            .zip(letters)
            .map(|(t, letters)| {
                let share = if t.len() == 1 {
                    SINGLE_PIECE_SHARE
                } else {
                    1.0
                };
                let missing = Boost::SHORT_WORD_LETTERS.saturating_sub(letters);
                Form {
                    margin: boost.spotting_margin + boost.short_word_margin * missing as f32,
                    columns: t
                        .iter()
                        .map(|id| 2 + tokens.binary_search(id).expect("a form's token is listed"))
                        .collect(),
                    bonus: boost.spotting_bonus * t.len() as f32 * share,
                    tokens: t,
                }
            })
            .collect();
        (
            Compiled {
                matcher,
                forms,
                tokens,
                boost,
            },
            unusable,
        )
    }

    pub fn is_empty(&self) -> bool {
        self.forms.is_empty()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// "▁ku" 1, "be" 2, "ctl" 3, "▁cube" 4, "▁control" 5, "▁the" 6, "," 7, "▁Ku" 8, "▁ng" 9,
    /// "inx" 10, "▁" 11, "▁New" 42, "▁York" 43, "▁city" 44, "s" 47, plus letters and merges.
    pub(crate) fn tokenizer() -> Tokenizer {
        let json = r#"{"added_tokens":[{"id":0,"content":"<blank>","special":true}],
            "model":{"type":"BPE","vocab":{"<blank>":0,"▁ku":1,"be":2,"ctl":3,"▁cube":4,"▁control":5,"▁the":6,",":7,"▁Ku":8,"▁ng":9,"inx":10,"▁":11,
              "k":12,"u":13,"b":14,"e":15,"c":16,"t":17,"l":18,"K":19,"n":20,"g":21,"i":22,"x":23,"o":24,"r":25,"h":26,"▁k":27,"▁K":28,"▁n":29,"▁c":30,"▁t":31,"ct":32,"▁co":33,"▁con":34,"ro":35,"▁cu":36,"nt":37,"ntro":38,"ntrol":39,"in":40,"▁th":41,
              "▁New":42,"▁York":43,"▁city":44,"▁Ne":45,"w":46,"s":47,"N":48,"Y":49,"▁N":50,"▁Y":51,"▁Yo":52,"▁Yor":53,"▁ci":54,"▁cit":55,"y":56},
              "merges":[["▁","k"],["▁k","u"],["b","e"],["c","t"],["ct","l"],["▁","K"],["▁K","u"],["▁","n"],["▁n","g"],["i","n"],["in","x"],
                ["▁","c"],["▁c","u"],["▁cu","be"],["▁c","o"],["▁co","n"],["n","t"],["r","o"],["nt","ro"],["ntro","l"],["▁con","trol"],["▁","t"],["▁t","h"],["▁th","e"],
                ["▁","N"],["▁N","e"],["▁Ne","w"],["▁","Y"],["▁Y","o"],["▁Yo","r"],["▁Yor","k"],["▁c","i"],["▁ci","t"],["▁cit","y"]]}}"#;
        Tokenizer::from_json(json.as_bytes()).unwrap()
    }

    pub(crate) fn special(words: &[&str]) -> SpecialWords {
        let mut list = SpecialWords::new();
        for w in words {
            list.add_code_word(w).unwrap();
        }
        list
    }

    #[test]
    fn words_split_into_tokens_in_both_spellings() {
        let tok = tokenizer();
        assert_eq!(tok.encode("kubectl"), vec![1, 2, 3]);
        assert_eq!(tok.encode("Kubectl nginx"), vec![8, 2, 3, 9, 10]);
        let mut words = SpecialWords::new();
        assert_eq!(words.add("kubectl"), Ok(()));
        assert_eq!(words.add("ng"), Err(Rejection::TooShort));
        assert_eq!(words.add("nginx"), Ok(()));
        assert_eq!(words.add_code_word("Zing "), Ok(()));
        assert_eq!(words.len(), 3);
        let (c, unusable) = Compiled::new(&words, &tok);
        let forms: Vec<&[u32]> = c.forms.iter().map(|f| f.tokens.as_slice()).collect();
        assert_eq!(
            forms,
            vec![&[1, 2, 3][..], &[8, 2, 3], &[9, 10], &[50, 21, 10]]
        );
        // The vocabulary has no "Z", so "Zing" has no spelling the model can use.
        assert_eq!(
            unusable,
            vec![("Zing".to_string(), Rejection::NotInVocabulary)]
        );
        assert_eq!(c.tokens, vec![1, 2, 3, 8, 9, 10, 21, 50]);
    }

    #[test]
    fn lists_and_settings_refuse_what_they_cant_hold() {
        let mut words = SpecialWords::new();
        assert_eq!(words.add("  "), Err(Rejection::Empty));
        assert_eq!(words.add("--"), Err(Rejection::Empty));
        assert_eq!(words.add_code_word("rg"), Ok(()));
        assert_eq!(words.add_code_word("rg"), Err(Rejection::Duplicate));
        assert_eq!(words.add(&"x".repeat(65)), Err(Rejection::TooLong));
        assert_eq!(words.add("a b c d e f"), Err(Rejection::TooLong));
        assert_eq!(
            words.add(&"word ".repeat(1_000_000)),
            Err(Rejection::TooLong)
        );
        assert_eq!(
            words.add_code_word(&"x".repeat(10_000_000)),
            Err(Rejection::TooLong)
        );
        assert_eq!(
            normalize("  Visual \t Studio\nCode  "),
            Ok("Visual Studio Code".to_string())
        );
        assert_eq!(normalize(&"x".repeat(64)).map(|w| w.len()), Ok(64));
        assert_eq!(normalize("a b c d e"), Ok("a b c d e".to_string()));
        assert_eq!(
            words.add("cafe\u{301}"),
            Err(Rejection::UnsupportedCharacters)
        );
        assert_eq!(words.add("東京都"), Err(Rejection::UnsupportedCharacters));
        assert_eq!(
            words.add("tab\tbed\u{7}"),
            Err(Rejection::UnsupportedCharacters)
        );
        assert_eq!(words.add("café"), Ok(()));
        for i in words.len()..SpecialWords::MAX_WORDS {
            words.add(&format!("word{i}")).unwrap();
        }
        assert_eq!(words.add("another"), Err(Rejection::ListFull));

        assert!(Boost::new(1.5, 4, 4.0, 2.0).is_ok());
        assert!(Boost::new(f32::NAN, 4, 4.0, 2.0).is_err());
        assert!(Boost::new(1.5, 4, f32::INFINITY, 2.0).is_err());
        assert!(Boost::new(1.5, 4, 4.0, -1.0).is_err());
        assert!(Boost::new(-0.5, 4, 4.0, 2.0).is_err());
        assert!(Boost::new(1.5, 0, 4.0, 2.0).is_err());
        assert!(Boost::new(1.5, usize::MAX, 4.0, 2.0).is_err());
        assert_eq!(Boost::new(1.5, 4, 4.0, 2.0).unwrap(), Boost::default());
    }

    /// The committed bonus after `tokens` and the end of the audio, and the bonus held before the
    /// end, in units of the bonus per token.
    fn walk(m: &Matcher, tokens: &[u32]) -> (f32, f32) {
        let mut s = MatchState::START;
        let mut held = 0.0;
        for &t in tokens {
            let (next, delta) = m.step(s, t);
            s = next;
            held += delta;
        }
        ((held + m.finish(s)) / m.bonus, held / m.bonus)
    }

    #[test]
    fn the_matcher_pays_each_token_of_a_completed_word_once() {
        let tok = tokenizer();
        let half = SINGLE_PIECE_SHARE;
        // (entries, tokens, final bonus, bonus held before the end), in bonuses per token.
        let cases: &[(&[&str], &[u32], f32, f32)] = &[
            (&["kubectl"], &[1, 2, 3, 6], 3.0, 3.0),
            (&["kubectl"], &[1, 2, 3], 3.0, 3.0),
            (&["kubectl"], &[1, 2, 3, 7, 6], 3.0, 3.0),
            // Left before the end, or the word goes on with letters: taken back.
            (&["kubectl"], &[1, 2, 6], 0.0, 0.0),
            (&["kubectl"], &[1, 2], 0.0, 2.0),
            (&["kubectl"], &[1, 2, 3, 47], 0.0, 0.0),
            // A new special word right where another was left.
            (&["kubectl"], &[1, 2, 8, 2, 3], 3.0, 3.0),
            // "New" doesn't block "New York", and the shared token is paid once, fully.
            (&["New", "New York"], &[42, 43, 6], 2.0, 2.0),
            (&["New", "New York"], &[42, 6], half, half),
            (&["New", "New York"], &[42, 43], 2.0, 1.0 + half),
            // A word that ends inside a longer one that dies keeps its own bonus.
            (&["New York city", "York"], &[42, 43, 6], half, half),
            (&["New York city", "York"], &[42, 43, 44], 3.0, 2.0 + half),
            (&["New York city", "York"], &[42, 43], half, 2.0),
            // Two words in a row, and a single-token word.
            (
                &["kubectl", "the"],
                &[1, 2, 3, 6, 7],
                3.0 + half,
                3.0 + half,
            ),
            (&["the"], &[6, 6, 47], half, half),
        ];
        for (entries, tokens, done, held) in cases {
            let (c, unusable) = Compiled::new(&special(entries), &tok);
            assert!(unusable.is_empty(), "{entries:?}");
            let (got_done, got_held) = walk(&c.matcher, tokens);
            assert!(
                (got_done - done).abs() < 1e-5 && (got_held - held).abs() < 1e-5,
                "{entries:?} {tokens:?}: final {got_done} held {got_held}, want {done} and {held}"
            );
        }
    }

    #[test]
    fn default_steps_match_the_steps_of_ordinary_tokens() {
        let tok = tokenizer();
        let (c, _) = Compiled::new(&special(&["kubectl", "New York"]), &tok);
        let m = &c.matcher;
        for prefix in [&[][..], &[1], &[1, 2, 3], &[42], &[42, 43]] {
            let mut s = MatchState::START;
            for &t in prefix {
                s = m.step(s, t).0;
            }
            let special: Vec<u32> = m.special_tokens(s).collect();
            let (boundary, other) = m.default_steps(s);
            for t in 0..tok.len() as u32 {
                if special.contains(&t) {
                    continue;
                }
                let want = if m.is_boundary(t) { boundary } else { other };
                assert_eq!(m.step(s, t).1, want, "{prefix:?} then {t}");
            }
        }
    }
}
