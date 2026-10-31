//! Session-word biasing: decoding that favours a list of special words, such as the names and
//! technical terms of a conversation, while ordinary words come out as before.
//!
//! Each special word is split into the model's tokens in its usual spelling and capitalized.
//!
//! - Parakeet decodes with a beam search over a prefix tree of those tokens with failure links
//!   (Aho-Corasick). A token that continues a special word earns a bonus, which is taken back
//!   unless the word is completed and followed by a word boundary or the end of the audio.
//! - Granite decodes greedily, then spots each special word with a CTC alignment of free start
//!   and end. Where the alignment plus a bonus beats the greedy words over the same frames by a
//!   margin, the special word replaces them.
//!
//! With no special words, both models decode greedily, exactly as without biasing.

use crate::parakeet::Token;
use crate::tokenizer::Tokenizer;

/// How strongly the decoders favour special words. The defaults were tuned on LibriSpeech.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Boost {
    /// Parakeet's bonus for each token of a special word, in log probability.
    pub bonus: f32,
    /// The hypotheses Parakeet's beam search keeps.
    pub beam: usize,
    /// Granite's bonus for each token of a spotted word, in log probability.
    pub spotting_bonus: f32,
    /// How much a spotted word with its bonus must beat the greedy words it replaces, in log
    /// probability.
    pub spotting_margin: f32,
}

impl Default for Boost {
    fn default() -> Self {
        Boost {
            bonus: 2.0,
            beam: 4,
            spotting_bonus: 4.0,
            spotting_margin: 2.0,
        }
    }
}

/// Words to favour while decoding, each in its usual spelling, at most a few hundred.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpecialWords {
    words: Vec<String>,
    pub boost: Boost,
}

impl SpecialWords {
    pub fn new() -> SpecialWords {
        SpecialWords::default()
    }

    /// Adds a word, or several separated by spaces that are favoured together ("Visual Studio
    /// Code"). Words of one or two letters or digits are left out, since favouring them would
    /// turn short ordinary words into them.
    pub fn add(&mut self, word: &str) {
        if word.chars().filter(|c| c.is_alphanumeric()).count() > 2 {
            self.add_code_word(word);
        }
    }

    /// Adds a word seen in code, a path or an identifier, which keeps it even when it has only
    /// one or two letters.
    pub fn add_code_word(&mut self, word: &str) {
        let word = word.split_whitespace().collect::<Vec<_>>().join(" ");
        if !word.is_empty() && !self.words.contains(&word) {
            self.words.push(word);
        }
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
}

impl<S: AsRef<str>> FromIterator<S> for SpecialWords {
    fn from_iter<I: IntoIterator<Item = S>>(iter: I) -> Self {
        let mut words = SpecialWords::new();
        for word in iter {
            words.add(word.as_ref());
        }
        words
    }
}

/// A special word that tokenizes to a single piece is one the model knows well, so it earns
/// this share of the bonus.
const SINGLE_PIECE_SHARE: f32 = 0.5;

/// The tree's root, where no special word is under way.
pub(crate) const ROOT: u32 = 0;

#[derive(Debug, Clone)]
struct Node {
    /// (token, node), sorted by token.
    children: Vec<(u32, u32)>,
    /// The node of the longest proper suffix of this node's tokens that is in the tree.
    fail: u32,
    /// The bonus a hypothesis holds here, summed over the tokens from the root.
    bonus: f32,
    /// Whether a special word ends here.
    is_end: bool,
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
    nodes: Vec<Node>,
    /// Whether a word may end before each token of the vocabulary.
    boundary: Vec<bool>,
    pub forms: Vec<Form>,
    /// The tokens of all forms, sorted. Word spotting reads their logits.
    pub tokens: Vec<u32>,
    pub boost: Boost,
}

impl Compiled {
    /// Splits the words into the tokenizer's tokens. Spellings with tokens the vocabulary lacks
    /// are left out.
    pub fn new(words: &SpecialWords, tokenizer: &Tokenizer) -> Compiled {
        let boost = words.boost;
        let mut forms: Vec<Vec<u32>> = Vec::new();
        if words.is_empty() {
            return Compiled {
                nodes: Vec::new(),
                boundary: Vec::new(),
                forms: Vec::new(),
                tokens: Vec::new(),
                boost,
            };
        }
        for word in words.words() {
            let mut chars = word.chars();
            let capitalized: String = chars
                .next()
                .map(|c| c.to_uppercase().chain(chars).collect())
                .unwrap_or_default();
            for spelling in [word, capitalized.as_str()] {
                let tokens = tokenizer.encode(spelling);
                // A spelling with characters the vocabulary lacks can't be emitted as written.
                let exact = tokenizer.decode(&tokens) == spelling;
                if !tokens.is_empty() && exact && !forms.contains(&tokens) {
                    forms.push(tokens);
                }
            }
        }

        let mut nodes = vec![Node {
            children: Vec::new(),
            fail: ROOT,
            bonus: 0.0,
            is_end: false,
        }];
        for form in &forms {
            let mut node = ROOT;
            for &token in form {
                node = match child(&nodes, node, token) {
                    Some(c) => c,
                    None => {
                        let new = nodes.len() as u32;
                        let bonus = nodes[node as usize].bonus + boost.bonus;
                        nodes.push(Node {
                            children: Vec::new(),
                            fail: ROOT,
                            bonus,
                            is_end: false,
                        });
                        let children = &mut nodes[node as usize].children;
                        let at = children.partition_point(|&(t, _)| t < token);
                        children.insert(at, (token, new));
                        new
                    }
                };
            }
            nodes[node as usize].is_end = true;
        }
        for &(_, node) in &nodes[ROOT as usize].children.clone() {
            let n = &mut nodes[node as usize];
            if n.is_end {
                n.bonus *= SINGLE_PIECE_SHARE;
                let share = n.bonus;
                // The words that go on from a single-piece word start from its smaller bonus.
                let mut stack: Vec<(u32, f32)> =
                    n.children.iter().map(|&(_, c)| (c, share)).collect();
                while let Some((c, base)) = stack.pop() {
                    let node = &mut nodes[c as usize];
                    node.bonus = base + boost.bonus;
                    let base = node.bonus;
                    stack.extend(node.children.iter().map(|&(_, g)| (g, base)));
                }
            }
        }
        // Failure links, breadth first, as Aho-Corasick builds them.
        let mut queue: std::collections::VecDeque<u32> = nodes[ROOT as usize]
            .children
            .iter()
            .map(|&(_, c)| c)
            .collect();
        while let Some(node) = queue.pop_front() {
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

        let mut tokens: Vec<u32> = forms.iter().flatten().copied().collect();
        tokens.sort_unstable();
        tokens.dedup();
        let forms = forms
            .into_iter()
            .map(|t| {
                let share = if t.len() == 1 {
                    SINGLE_PIECE_SHARE
                } else {
                    1.0
                };
                Form {
                    columns: t
                        .iter()
                        .map(|id| 2 + tokens.binary_search(id).expect("a form's token is listed"))
                        .collect(),
                    bonus: boost.spotting_bonus * t.len() as f32 * share,
                    tokens: t,
                }
            })
            .collect();
        Compiled {
            nodes,
            boundary: (0..tokenizer.len() as u32)
                .map(|id| tokenizer.is_word_boundary(id))
                .collect(),
            forms,
            tokens,
            boost,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.forms.is_empty()
    }

    fn is_boundary(&self, token: u32) -> bool {
        self.boundary.get(token as usize).copied().unwrap_or(true)
    }

    /// The tree node after `token` from `node`, and the change in the bonus the hypothesis
    /// holds. A special word that ended at `node` keeps its bonus when `token` starts a new word
    /// or is punctuation. Otherwise leaving a word takes back what its tokens earned.
    pub fn step(&self, node: u32, token: u32) -> (u32, f32) {
        let here = &self.nodes[node as usize];
        let (mut from, held) = if here.is_end && self.is_boundary(token) {
            (ROOT, 0.0)
        } else {
            (node, here.bonus)
        };
        let next = loop {
            if let Some(c) = child(&self.nodes, from, token) {
                break c;
            }
            if from == ROOT {
                break ROOT;
            }
            from = self.nodes[from as usize].fail;
        };
        (next, self.nodes[next as usize].bonus - held)
    }

    /// The change in bonus when the audio ends at `node`: a completed word keeps its bonus and
    /// a partial one loses it.
    pub fn finish(&self, node: u32) -> f32 {
        let n = &self.nodes[node as usize];
        if n.is_end { 0.0 } else { -n.bonus }
    }

    /// Tokens that would continue a special word from `node` or start one.
    pub fn next_tokens(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        let here = &self.nodes[node as usize].children;
        let root = if node == ROOT {
            &[][..]
        } else {
            &self.nodes[ROOT as usize].children[..]
        };
        here.iter().chain(root).map(|&(t, _)| t)
    }
}

fn child(nodes: &[Node], node: u32, token: u32) -> Option<u32> {
    let children = &nodes[node as usize].children;
    children
        .binary_search_by_key(&token, |&(t, _)| t)
        .ok()
        .map(|i| children[i].1)
}

/// The scores word spotting reads, for each frame: its best logit, the blank's logit and the
/// logits of [`Compiled::tokens`], in that order.
pub(crate) fn score_row(logits: &[f32], blank: usize, tokens: &[u32], out: &mut Vec<f32>) {
    out.push(logits.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x)));
    out.push(logits[blank]);
    out.extend(tokens.iter().map(|&t| logits[t as usize]));
}

/// A stretch of greedy tokens that spotting replaces whole or not at all: a word, or a
/// punctuation token.
#[derive(Debug, Clone, Copy)]
struct Span {
    /// The span's first and last frame.
    first: usize,
    last: usize,
    has_word: bool,
}

/// A replacement word spotting found.
#[derive(Debug, Clone, Copy)]
struct Spot {
    form: usize,
    first: usize,
    last: usize,
    gain: f32,
}

/// Replaces greedy CTC tokens with special words where an alignment of a word, plus its bonus,
/// beats the greedy path over the same frames by the margin. `greedy` holds the greedy tokens of
/// frames `first..` (each a run of frames), `scores` the rows of [`score_row`] for every frame
/// from 0, `width` long each. A spotted word covers whole greedy words, at least one, and the
/// frames between them. Its last token lasts to the end of those frames.
pub(crate) fn spot(
    words: &Compiled,
    tokenizer: &Tokenizer,
    greedy: &[Token],
    scores: &[f32],
    first: usize,
) -> Vec<Token> {
    let width = 2 + words.tokens.len();
    let frames = scores.len() / width;
    if words.is_empty() || first >= frames {
        return greedy.to_vec();
    }
    // Words and punctuation of the greedy text, as spans of frames.
    let mut spans: Vec<Span> = Vec::new();
    for t in greedy {
        let last = (t.frame + t.duration.max(1) - 1) as usize;
        match spans.last_mut() {
            Some(s) if !tokenizer.is_word_boundary(t.id) => s.last = last,
            _ => spans.push(Span {
                first: t.frame as usize,
                last,
                has_word: false,
            }),
        }
        let s = spans.last_mut().expect("just pushed");
        s.has_word |= !tokenizer
            .decode(&[t.id])
            .chars()
            .all(|c| !c.is_alphanumeric());
    }
    // An alignment may start where no span is under way and end where none goes on, and needs
    // to know how many words it covers.
    let n = frames - first;
    let mut may_start = vec![true; n];
    let mut may_end = vec![true; n];
    let mut words_before = vec![0usize; n + 1];
    for s in spans.iter().filter(|s| s.first >= first) {
        for f in s.first + 1..=s.last.min(frames - 1) {
            may_start[f - first] = false;
        }
        for f in s.first..s.last.min(frames - 1) {
            may_end[f - first] = false;
        }
        if s.has_word && s.last < frames {
            words_before[s.last + 1 - first] += 1;
        }
    }
    for i in 0..n {
        words_before[i + 1] += words_before[i];
    }
    let cost = |f: usize, column: usize| {
        let row = &scores[f * width..(f + 1) * width];
        row[0] - row[column]
    };

    let mut spots = Vec::new();
    let margin = words.boost.spotting_margin;
    for (index, form) in words.forms.iter().enumerate() {
        let states = 2 * form.tokens.len() + 1;
        let mut cost_of = vec![f32::INFINITY; states];
        let mut start = vec![0usize; states];
        let mut next_cost = cost_of.clone();
        let mut next_start = start.clone();
        for f in first..frames {
            let i = f - first;
            for s in 0..states {
                let (mut best, mut from) = (cost_of[s], start[s]);
                if s >= 1 && cost_of[s - 1] < best {
                    (best, from) = (cost_of[s - 1], start[s - 1]);
                }
                if form.may_skip(s) && cost_of[s - 2] < best {
                    (best, from) = (cost_of[s - 2], start[s - 2]);
                }
                if s <= 1 && may_start[i] && best > 0.0 {
                    (best, from) = (0.0, f);
                }
                next_cost[s] = best + cost(f, form.column(s));
                next_start[s] = from;
            }
            std::mem::swap(&mut cost_of, &mut next_cost);
            std::mem::swap(&mut start, &mut next_start);
            if !may_end[i] {
                continue;
            }
            for s in [states - 2, states - 1] {
                let gain = form.bonus - cost_of[s] - margin;
                if gain > 0.0 && words_before[i + 1] > words_before[start[s] - first] {
                    spots.push(Spot {
                        form: index,
                        first: start[s],
                        last: f,
                        gain,
                    });
                }
            }
        }
    }
    if spots.is_empty() {
        return greedy.to_vec();
    }
    // The best spots that don't overlap win.
    spots.sort_by(|a, b| b.gain.total_cmp(&a.gain));
    let mut chosen: Vec<Spot> = Vec::new();
    for s in spots {
        if chosen.iter().all(|c| s.last < c.first || s.first > c.last) {
            chosen.push(s);
        }
    }
    chosen.sort_by_key(|s| s.first);
    let mut out = Vec::with_capacity(greedy.len());
    let mut next = chosen.iter().peekable();
    let mut i = 0;
    while i < greedy.len() {
        let t = greedy[i];
        match next.peek() {
            Some(s) if t.frame as usize >= s.first => {
                out.extend(align(&words.forms[s.form], s.first, s.last, &cost));
                while i < greedy.len() && greedy[i].frame as usize <= s.last {
                    i += 1;
                }
                next.next();
            }
            _ => {
                out.push(t);
                i += 1;
            }
        }
    }
    // A spot after the last greedy token (it covers at least one word, so it can't be).
    debug_assert!(next.peek().is_none());
    out
}

/// The tokens of the best CTC alignment of `form` over frames `first..=last`, each at the frame
/// its run starts, the last lasting to `last`.
fn align(form: &Form, first: usize, last: usize, cost: &dyn Fn(usize, usize) -> f32) -> Vec<Token> {
    let states = 2 * form.tokens.len() + 1;
    let n = last + 1 - first;
    let mut cost_of = vec![f32::INFINITY; states];
    cost_of[0] = cost(first, form.column(0));
    cost_of[1] = cost(first, form.column(1));
    // back[i][s]: the state at frame first + i - 1 the best path to s at first + i came from.
    let mut back = vec![vec![0u8; states]; n];
    for (i, row) in back.iter_mut().enumerate().skip(1) {
        let mut next = vec![f32::INFINITY; states];
        for s in 0..states {
            let mut best = (cost_of[s], 0u8);
            if s >= 1 && cost_of[s - 1] < best.0 {
                best = (cost_of[s - 1], 1);
            }
            if form.may_skip(s) && cost_of[s - 2] < best.0 {
                best = (cost_of[s - 2], 2);
            }
            next[s] = best.0 + cost(first + i, form.column(s));
            row[s] = best.1;
        }
        cost_of = next;
    }
    let mut s = if cost_of[states - 2] <= cost_of[states - 1] {
        states - 2
    } else {
        states - 1
    };
    let mut path = vec![0usize; n];
    for i in (0..n).rev() {
        path[i] = s;
        s -= back[i][s] as usize;
    }
    let mut tokens: Vec<Token> = Vec::new();
    for (i, &s) in path.iter().enumerate() {
        let new_run = i == 0 || path[i - 1] != s;
        if !s.is_multiple_of(2) && new_run {
            tokens.push(Token {
                id: form.tokens[s / 2],
                frame: (first + i) as u32,
                duration: 1,
            });
        } else if !s.is_multiple_of(2) {
            tokens.last_mut().expect("a run is open").duration += 1;
        }
    }
    if let Some(t) = tokens.last_mut() {
        t.duration = last as u32 + 1 - t.frame;
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "▁ku" 1, "be" 2, "ctl" 3, "▁cube" 4, "▁control" 5, "▁the" 6, "," 7, "▁Ku" 8, "▁ng" 9,
    /// "inx" 10, "▁" 11, plus single letters.
    fn tokenizer() -> Tokenizer {
        let json = r#"{"added_tokens":[{"id":0,"content":"<blank>","special":true}],
            "model":{"type":"BPE","vocab":{"<blank>":0,"▁ku":1,"be":2,"ctl":3,"▁cube":4,"▁control":5,"▁the":6,",":7,"▁Ku":8,"▁ng":9,"inx":10,"▁":11,
              "k":12,"u":13,"b":14,"e":15,"c":16,"t":17,"l":18,"K":19,"n":20,"g":21,"i":22,"x":23,"o":24,"r":25,"h":26,"▁k":27,"▁K":28,"▁n":29,"▁c":30,"▁t":31,"ct":32,"▁co":33,"▁con":34,"ro":35,"▁cu":36,"nt":37,"ntro":38,"ntrol":39,"in":40,"▁th":41},
              "merges":[["▁","k"],["▁k","u"],["b","e"],["c","t"],["ct","l"],["▁","K"],["▁K","u"],["▁","n"],["▁n","g"],["i","n"],["in","x"],
                ["▁","c"],["▁c","u"],["▁cu","be"],["▁c","o"],["▁co","n"],["n","t"],["r","o"],["nt","ro"],["ntro","l"],["▁con","trol"],["▁","t"],["▁t","h"],["▁th","e"]]}}"#;
        Tokenizer::from_json(json.as_bytes()).unwrap()
    }

    #[test]
    fn words_split_into_tokens_in_both_spellings() {
        let tok = tokenizer();
        assert_eq!(tok.encode("kubectl"), vec![1, 2, 3]);
        assert_eq!(tok.encode("Kubectl nginx"), vec![8, 2, 3, 9, 10]);
        let words: SpecialWords = ["kubectl", "ng", "nginx"].into_iter().collect();
        // "ng" has two letters, so it's left out.
        assert_eq!(words.len(), 2);
        let c = Compiled::new(&words, &tok);
        let forms: Vec<&[u32]> = c.forms.iter().map(|f| f.tokens.as_slice()).collect();
        // The vocabulary has no "N", so "Nginx" is left out.
        assert_eq!(forms, vec![&[1, 2, 3][..], &[8, 2, 3], &[9, 10]]);
        assert_eq!(c.tokens, vec![1, 2, 3, 8, 9, 10]);
    }

    #[test]
    fn a_bonus_holds_only_for_words_completed_at_a_boundary() {
        let tok = tokenizer();
        let words: SpecialWords = ["kubectl"].into_iter().collect();
        let c = Compiled::new(&words, &tok);
        let b = c.boost.bonus;
        let walk = |tokens: &[u32]| {
            let (mut node, mut total) = (ROOT, 0.0);
            for &t in tokens {
                let (next, delta) = c.step(node, t);
                node = next;
                total += delta;
            }
            (node, total, total + c.finish(node))
        };
        // Completed, then a new word: the bonus stays.
        let (_, held, done) = walk(&[1, 2, 3, 6]);
        assert_eq!((held, done), (3.0 * b, 3.0 * b));
        // Completed at the end, or before punctuation.
        assert_eq!(walk(&[1, 2, 3]).2, 3.0 * b);
        assert_eq!(walk(&[1, 2, 3, 7, 6]).2, 3.0 * b);
        // Left before the end: taken back.
        let (node, held, done) = walk(&[1, 2, 6]);
        assert_eq!((node, held, done), (ROOT, 0.0, 0.0));
        assert_eq!(walk(&[1, 2]).2, 0.0);
        // Completed but the word goes on with letters: taken back.
        assert_eq!(walk(&[1, 2, 3, 15]).2, 0.0);
        // A new special word starts right where another was left.
        let (_, held, _) = walk(&[1, 2, 8, 2]);
        assert_eq!(held, 2.0 * b);
        let next: Vec<u32> = c.next_tokens(walk(&[1]).0).collect();
        assert_eq!(next, vec![2, 1, 8]);
    }

    #[test]
    fn single_piece_words_earn_less() {
        let tok = tokenizer();
        let words: SpecialWords = ["cube", "cubectl"].into_iter().collect();
        let c = Compiled::new(&words, &tok);
        let b = c.boost.bonus;
        let (node, delta) = c.step(ROOT, 4);
        assert_eq!(delta, SINGLE_PIECE_SHARE * b);
        assert_eq!(c.step(node, 3).1, b);
        assert_eq!(
            c.forms[0].bonus,
            SINGLE_PIECE_SHARE * c.boost.spotting_bonus
        );
    }

    /// Scores for `frames` frames where each frame's best token (or the blank) has logit 0 and
    /// every other token -10, except the logits in `set` (frame, token, logit).
    fn scores(c: &Compiled, best: &[u32], set: &[(usize, u32, f32)]) -> Vec<f32> {
        let mut out = Vec::new();
        for (f, &b) in best.iter().enumerate() {
            let mut logits = vec![-10f32; 42];
            logits[b as usize] = 0.0;
            for &(sf, t, l) in set {
                if sf == f {
                    logits[t as usize] = l;
                }
            }
            score_row(&logits, 0, &c.tokens, &mut out);
        }
        out
    }

    fn tokens(ids: &[u32]) -> Vec<Token> {
        ctc(ids)
    }

    fn ctc(ids: &[u32]) -> Vec<Token> {
        crate::granite::ctc_tokens(ids, 0)
    }

    #[test]
    fn spotting_replaces_whole_greedy_words_when_the_word_is_close() {
        let tok = tokenizer();
        let words: SpecialWords = ["kubectl"].into_iter().collect();
        let c = Compiled::new(&words, &tok);
        // Greedy: "▁the ▁cube ▁control ,".
        let best = [6, 0, 4, 4, 0, 5, 5, 0, 7, 0];
        let greedy = tokens(&best);
        // The word's tokens are a little less likely than the greedy ones on the same frames.
        let close = [(2, 1, -0.5), (3, 2, -0.5), (5, 3, -0.5), (6, 3, -0.5)];
        let out = spot(&c, &tok, &greedy, &scores(&c, &best, &close), 0);
        let ids: Vec<u32> = out.iter().map(|t| t.id).collect();
        assert_eq!(ids, vec![6, 1, 2, 3, 7]);
        assert_eq!((out[3].frame, out[3].duration), (5, 2));
        // Far from it, the greedy words stay.
        let far = [(2, 1, -6.0), (3, 2, -6.0), (5, 3, -6.0), (6, 3, -6.0)];
        assert_eq!(spot(&c, &tok, &greedy, &scores(&c, &best, &far), 0), greedy);
        // Frames before `first` aren't touched.
        assert_eq!(
            spot(&c, &tok, &greedy[2..], &scores(&c, &best, &close), 5),
            greedy[2..]
        );
    }

    #[test]
    fn spotting_never_inserts_a_word_into_silence() {
        let tok = tokenizer();
        let words: SpecialWords = ["kubectl"].into_iter().collect();
        let c = Compiled::new(&words, &tok);
        let best = [6, 0, 0, 0, 0, 0];
        let greedy = tokens(&best);
        let strong = [(2, 1, 0.0), (3, 2, 0.0), (4, 3, 0.0)];
        assert_eq!(
            spot(&c, &tok, &greedy, &scores(&c, &best, &strong), 0),
            greedy
        );
    }
}
