//! Spoken forms written as their words. After decoding, a whole-word match of a spoken form
//! ("engine x") in the transcript's text, whatever the case of its ASCII letters, is replaced by
//! the tokens of its word ("Nginx") on the same frames, so the transcript's text stays the
//! decoding of its tokens.

use std::ops::Range;

use crate::parakeet::Token;
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Default)]
struct Node {
    /// (byte, node), sorted by byte.
    children: Vec<(u8, u32)>,
    /// The entry a spoken form ending here stands for.
    form: Option<EntryId>,
    /// The entries whose word ends here, by their place in `alike`.
    written: Option<u32>,
}

/// An entry of [`Spoken`], a word as written with its spoken forms, by its place in the list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EntryId(u32);

/// The entries a stretch of a transcript stands for. That's one, or for a word heard as written
/// in other capitals than any of the entries whose words differ only in case, each of those.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Entries {
    One(EntryId),
    Alike(u32),
}

/// Spoken forms and the words they stand for, in a tree of their bytes with ASCII letters in
/// lower case.
#[derive(Debug, Clone, Default)]
pub(crate) struct Spoken {
    /// The tree, empty without forms. The root is node 0.
    nodes: Vec<Node>,
    entries: Vec<Entry>,
    /// Entries whose words differ only in case, in the order they were added.
    alike: Vec<Vec<EntryId>>,
}

#[derive(Debug, Clone)]
struct Entry {
    /// The word as written.
    word: String,
    tokens: Vec<u32>,
    forms: Vec<String>,
}

/// A spoken form found in a transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Replacement {
    /// The tokens it takes.
    pub heard: Range<usize>,
    /// The entry whose word is written in its place.
    pub entry: EntryId,
    /// The tokens of what follows the form in its last token, written after the word, such as
    /// the period of a token "5." where the form ends in "5".
    pub rest: Vec<u32>,
    /// The form's length in bytes.
    pub len: usize,
}

/// An entry's spoken form or word found in a transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Unit {
    /// A spoken form, written as its entry's word.
    Form(Replacement),
    /// An entry's word heard as written, which stays as heard.
    Written {
        heard: Range<usize>,
        entries: Entries,
        /// The word's length in bytes.
        len: usize,
    },
}

impl Unit {
    /// The tokens it takes.
    pub fn heard(&self) -> Range<usize> {
        match self {
            Unit::Form(r) => r.heard.clone(),
            Unit::Written { heard, .. } => heard.clone(),
        }
    }

    pub fn entries(&self) -> Entries {
        match self {
            Unit::Form(r) => Entries::One(r.entry),
            Unit::Written { entries, .. } => *entries,
        }
    }

    /// The length in bytes of the form or word as heard.
    pub fn len(&self) -> usize {
        match self {
            Unit::Form(r) => r.len,
            Unit::Written { len, .. } => *len,
        }
    }
}

impl Spoken {
    /// Adds an entry for the word `word`, with tokens `tokens`, unless it has one already, and
    /// returns it for [`Spoken::insert`].
    pub fn add_word(&mut self, word: &str, tokens: Vec<u32>) -> EntryId {
        let node = self.node_for(word);
        let group = match self.nodes[node].written {
            Some(group) => group as usize,
            None => {
                self.alike.push(Vec::new());
                self.nodes[node].written = Some(self.alike.len() as u32 - 1);
                self.alike.len() - 1
            }
        };
        let entries = &self.entries;
        if let Some(&id) = self.alike[group]
            .iter()
            .find(|id| entries[id.0 as usize].word == word)
        {
            return id;
        }
        let id = EntryId(self.entries.len() as u32);
        self.alike[group].push(id);
        self.entries.push(Entry {
            word: word.to_string(),
            tokens,
            forms: Vec::new(),
        });
        id
    }

    /// Writes the word of `entry` for `form`, unless an earlier entry has the form.
    pub fn insert(&mut self, form: &str, entry: EntryId) {
        let forms = &mut self.entries[entry.0 as usize].forms;
        if !forms.iter().any(|f| f == form) {
            forms.push(form.to_string());
        }
        let node = self.node_for(form);
        self.nodes[node].form.get_or_insert(entry);
    }

    /// The node where `text` ends in the tree, added with the nodes before it if need be.
    fn node_for(&mut self, text: &str) -> usize {
        if self.nodes.is_empty() {
            self.nodes.push(Node::default());
        }
        let mut node = 0;
        for b in text.bytes().map(|b| b.to_ascii_lowercase()) {
            node = match self.child(node, b) {
                Some(c) => c,
                None => {
                    let new = self.nodes.len() as u32;
                    self.nodes.push(Node::default());
                    let children = &mut self.nodes[node as usize].children;
                    let at = children.partition_point(|&(x, _)| x < b);
                    children.insert(at, (b, new));
                    new
                }
            };
        }
        node as usize
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The tokens of the word of `entry`.
    pub fn tokens(&self, entry: EntryId) -> &[u32] {
        &self.entries[entry.0 as usize].tokens
    }

    fn ids(&self, entries: Entries) -> Vec<EntryId> {
        match entries {
            Entries::One(id) => vec![id],
            Entries::Alike(group) => self.alike[group as usize].clone(),
        }
    }

    /// The length in bytes of the word of `entries`.
    pub fn word_len(&self, entries: Entries) -> usize {
        self.entries[self.ids(entries)[0].0 as usize].word.len()
    }

    /// Each way `entries` are said, as written and as each of their forms, followed by `rest`
    /// and without it, as [`normalized_words`].
    pub fn spellings(&self, entries: Entries, rest: &str) -> impl Iterator<Item = Vec<String>> {
        self.ids(entries).into_iter().flat_map(move |id| {
            let entry = &self.entries[id.0 as usize];
            std::iter::once(&entry.word)
                .chain(&entry.forms)
                .flat_map(move |s| {
                    let with = normalized_words(&format!("{s}{rest}"));
                    let without = normalized_words(s);
                    let both = with != without;
                    std::iter::once(with).chain(both.then_some(without))
                })
        })
    }

    /// The entries whose word starts `text`, in any case of its ASCII letters, and its length,
    /// where it ends before a character that isn't a letter or digit, or at the end, and isn't
    /// longer than `most` bytes. The longest wins. Of entries whose words differ only in case,
    /// it's the one spelled as in `text`, or else each of them.
    fn written_at(&self, text: &[u8], most: usize) -> Option<(Entries, usize)> {
        if self.is_empty() {
            return None;
        }
        let mut node = 0;
        let mut found = None;
        for (p, &b) in text.iter().enumerate().take(most) {
            match self.child(node, b.to_ascii_lowercase()) {
                Some(c) => node = c,
                None => break,
            }
            if let Some(group) = self.nodes[node as usize].written
                && ends_word(&text[p + 1..])
            {
                found = Some((group, p + 1));
            }
        }
        let (group, len) = found?;
        let exact = self.alike[group as usize]
            .iter()
            .find(|id| self.entries[id.0 as usize].word.as_bytes() == &text[..len]);
        let entries = exact.map_or(Entries::Alike(group), |&id| Entries::One(id));
        Some((entries, len))
    }

    /// The entries' spoken forms and words in `ids`, in any case of their ASCII letters. The
    /// forms are those of [`Spoken::find`], and the words, leftmost first and the longest of
    /// those starting at one token, are found in the stretches between them by the same rules.
    pub fn units(&self, tokenizer: &Tokenizer, ids: &[u32]) -> Vec<Unit> {
        let mut units = Vec::new();
        if self.is_empty() {
            return units;
        }
        let (text, ends) = text_of(tokenizer, ids);
        let mut forms = self.find(tokenizer, ids).into_iter().peekable();
        let mut i = 0;
        while i < ids.len() {
            if let Some(r) = forms.next_if(|r| r.heard.start == i) {
                i = r.heard.end;
                units.push(Unit::Form(r));
                continue;
            }
            let start = if i == 0 { 0 } else { ends[i - 1] };
            // A word ends before the next form.
            let limit = forms.peek().map_or(text.len(), |r| match r.heard.start {
                0 => 0,
                k => ends[k - 1],
            });
            let found = (text.get(start) == Some(&b' ') && start < ends[i])
                .then(|| self.written_at(&text[start + 1..], limit.saturating_sub(start + 1)))
                .flatten();
            let Some((entries, len)) = found else {
                i += 1;
                continue;
            };
            let end = ends.partition_point(|&e| e < start + 1 + len) + 1;
            units.push(Unit::Written {
                heard: i..end,
                entries,
                len,
            });
            i = end;
        }
        units
    }

    fn child(&self, node: u32, b: u8) -> Option<u32> {
        let children = &self.nodes[node as usize].children;
        children
            .binary_search_by_key(&b, |&(x, _)| x)
            .ok()
            .map(|i| children[i].1)
    }

    /// The spoken forms in `ids`, in any case of their ASCII letters, leftmost first and the
    /// longest of those starting at one token. A form starts at a token that starts a word, and
    /// ends before a character that isn't a letter or digit, or the end, which may be inside a
    /// token.
    pub fn find(&self, tokenizer: &Tokenizer, ids: &[u32]) -> Vec<Replacement> {
        let mut found = Vec::new();
        if self.is_empty() {
            return found;
        }
        let (text, ends) = text_of(tokenizer, ids);
        let mut i = 0;
        while i < ids.len() {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            if text.get(start) != Some(&b' ') || start == ends[i] {
                i += 1;
                continue;
            }
            // Walk the tree from the byte after the word's space.
            let (mut node, mut token) = (0, i);
            let mut longest = None;
            for (p, &b) in text.iter().enumerate().skip(start + 1) {
                match self.child(node, b.to_ascii_lowercase()) {
                    Some(c) => node = c,
                    None => break,
                }
                while ends[token] < p + 1 {
                    token += 1;
                }
                if let Some(entry) = self.nodes[node as usize].form
                    && ends_word(&text[p + 1..])
                    && let Some(rest) = continuation(tokenizer, &text[p + 1..ends[token]])
                {
                    longest = Some(Replacement {
                        heard: i..token + 1,
                        entry,
                        rest,
                        len: p - start,
                    });
                }
            }
            match longest {
                Some(r) => {
                    i = r.heard.end;
                    found.push(r);
                }
                None => i += 1,
            }
        }
        found
    }

    /// The first token of `ids` before token `frontier` from which a spoken form or an entry's
    /// word could still be under way, given that the tokens from `frontier` on may still change,
    /// or none. That's a token starting a word whose text up to the frontier is the start of a
    /// form or word that goes on with another word, since the frontier starts one.
    pub fn open_before(
        &self,
        tokenizer: &Tokenizer,
        ids: &[u32],
        frontier: usize,
    ) -> Option<usize> {
        if self.is_empty() || frontier == 0 {
            return None;
        }
        let (text, ends) = text_of(tokenizer, &ids[..frontier]);
        (0..frontier).find(|&i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            if text.get(start) != Some(&b' ') || start == ends[i] {
                return false;
            }
            let mut node = 0;
            for &b in &text[start + 1..] {
                match self.child(node, b.to_ascii_lowercase()) {
                    Some(c) => node = c,
                    None => return false,
                }
            }
            self.child(node, b' ').is_some()
        })
    }

    /// The tokens written for `r` with their frames, given the frames of the first and last token
    /// heard for it. The word's tokens spread over those frames as [`place`] puts them, and the
    /// rest of the last heard token keeps that token's frame.
    pub fn written(&self, r: &Replacement, first: usize, last: usize) -> Vec<(u32, usize)> {
        let word = self.tokens(r.entry);
        let spread = word
            .iter()
            .enumerate()
            .map(|(k, &id)| (id, place(first, last, word.len(), k)));
        spread.chain(r.rest.iter().map(|&id| (id, last))).collect()
    }

    /// `tokens` with each spoken form written as its word, on the frames given by
    /// [`Spoken::written`]. The last token lasts to the end of the form.
    pub fn rewrite(&self, tokenizer: &Tokenizer, tokens: Vec<Token>) -> Vec<Token> {
        if self.is_empty() {
            return tokens;
        }
        let ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
        let found = self.find(tokenizer, &ids);
        if found.is_empty() {
            return tokens;
        }
        let mut out = Vec::with_capacity(tokens.len());
        let mut next = 0;
        for r in &found {
            out.extend_from_slice(&tokens[next..r.heard.start]);
            let (first, last) = (tokens[r.heard.start], tokens[r.heard.end - 1]);
            let written = self.written(r, first.frame as usize, last.frame as usize);
            for (k, &(id, frame)) in written.iter().enumerate() {
                let end = written
                    .get(k + 1)
                    .map_or(last.frame + last.duration, |&(_, f)| f as u32);
                out.push(Token {
                    id,
                    frame: frame as u32,
                    duration: end - frame as u32,
                });
            }
            next = r.heard.end;
        }
        out.extend_from_slice(&tokens[next..]);
        out
    }
}

/// Whether a word ends at the start of `text`, meaning it starts with a character that isn't a
/// letter or digit, or is empty.
fn ends_word(text: &[u8]) -> bool {
    String::from_utf8_lossy(&text[..text.len().min(4)])
        .chars()
        .next()
        .is_none_or(|c| !c.is_alphanumeric())
}

/// The text of `ids`, with a space for each `▁`, and where each token's bytes end in it.
fn text_of(tokenizer: &Tokenizer, ids: &[u32]) -> (Vec<u8>, Vec<usize>) {
    let mut text = Vec::new();
    let mut ends = Vec::with_capacity(ids.len());
    for &id in ids {
        text.extend_from_slice(tokenizer.piece_bytes(id));
        ends.push(text.len());
    }
    (text, ends)
}

/// The tokens of `rest`, the end of a token after a form, written as part of the word before
/// it, if the vocabulary spells it.
fn continuation(tokenizer: &Tokenizer, rest: &[u8]) -> Option<Vec<u32>> {
    let rest = std::str::from_utf8(rest).ok()?;
    let tokens = tokenizer.encode_continuation(rest);
    (tokenizer.decode_raw(&tokens) == rest).then_some(tokens)
}

/// The words of `text` in lower case, each with only its letters and digits, as streams compare
/// words.
pub(crate) fn normalized_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// The frame of token `k` of `n` written in place of a form whose tokens start at frames
/// `first` to `last`. The first takes the form's first frame, the last the frame of its last
/// token, and those between spread evenly over the frames between.
pub(crate) fn place(first: usize, last: usize, n: usize, k: usize) -> usize {
    if n == 1 {
        first
    } else {
        first + (last - first) * k / (n - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::tokenizer;
    use super::super::{Compiled, SpecialWords};
    use super::*;

    fn spoken(pairs: &[(&str, &str)]) -> Compiled {
        let mut list = SpecialWords::new();
        for (written, form) in pairs {
            list.add_spoken(written, form).unwrap();
        }
        let (c, unusable) = Compiled::new(&list, &tokenizer());
        assert!(unusable.is_empty(), "{unusable:?}");
        c
    }

    /// Forms found, each as the tokens it takes and its word's tokens.
    type Found = Vec<(Range<usize>, Vec<u32>)>;

    /// The forms found in tokens `ids`.
    fn found(c: &Compiled, ids: &[u32]) -> Found {
        c.spoken
            .find(&tokenizer(), ids)
            .iter()
            .map(|r| (r.heard.clone(), c.spoken.tokens(r.entry).to_vec()))
            .collect()
    }

    // Tokens of the test vocabulary.
    const THE: u32 = 6;
    const CUBE: u32 = 4;
    const CONTROL: u32 = 5;
    const COMMA: u32 = 7;
    const S: u32 = 47;
    const NEW: u32 = 42;
    const YORK: u32 = 43;
    const CITY: u32 = 44;
    const BE_DOT: u32 = 59;
    const DOT: u32 = 60;
    const DASH: u32 = 61;
    const APOSTROPHE_S: u32 = 62;
    const HYPHEN: u32 = 63;

    #[test]
    fn whole_words_of_a_form_are_found_in_any_capitals() {
        let c = spoken(&[("kubectl", "cube control")]);
        let kubectl = || vec![(1..3, vec![1, 2, 3])];
        assert_eq!(found(&c, &[THE, CUBE, CONTROL, COMMA, THE]), kubectl());
        assert_eq!(found(&c, &[THE, CUBE, CONTROL]), kubectl());
        // A word that goes on isn't a whole-word match.
        assert_eq!(found(&c, &[CUBE, CONTROL, S, THE]), vec![]);
        assert_eq!(found(&c, &[THE, CUBE, THE]), vec![]);
        let tok = tokenizer();
        // A form spelled with other tokens than its own is found too.
        let split = [THE, 36, 2, CONTROL, COMMA];
        assert_eq!(tok.decode(&split), "the cube control,");
        assert_eq!(found(&c, &split), vec![(1..4, vec![1, 2, 3])]);
        for (text, matches) in [
            ("the Cube control,", true),
            ("the Cube Control,", true),
            ("the cube Control,", true),
            ("the cube contro,", false),
        ] {
            let ids = tok.encode(text);
            assert_eq!(tok.decode(&ids), text);
            assert_eq!(found(&c, &ids).len(), matches as usize, "{text}");
        }
    }

    #[test]
    fn any_character_but_a_letter_or_digit_ends_a_form() {
        let c = spoken(&[("kubectl", "cube control")]);
        let tok = tokenizer();
        let kubectl = || vec![(1..3, vec![1, 2, 3])];
        for ids in [
            [THE, CUBE, CONTROL, APOSTROPHE_S],
            [THE, CUBE, CONTROL, HYPHEN],
            [THE, CUBE, CONTROL, DASH],
        ] {
            assert_eq!(found(&c, &ids), kubectl(), "{}", tok.decode(&ids));
        }
        let text = |ids: &[u32]| {
            let tokens = ids
                .iter()
                .map(|&id| Token {
                    id,
                    frame: 0,
                    duration: 1,
                })
                .collect();
            let out: Vec<u32> = c
                .spoken
                .rewrite(&tok, tokens)
                .iter()
                .map(|t| t.id)
                .collect();
            tok.decode(&out)
        };
        assert_eq!(
            text(&[THE, CUBE, CONTROL, APOSTROPHE_S, THE]),
            "the kubectl's the"
        );
        assert_eq!(text(&[THE, CUBE, CONTROL, HYPHEN, S]), "the kubectl-s");
        assert_eq!(text(&[THE, CUBE, CONTROL, DASH, THE]), "the kubectl— the");
        // A letter or digit goes on with the word.
        assert_eq!(text(&[THE, CUBE, CONTROL, S, THE]), "the cube controls the");
    }

    #[test]
    fn a_written_word_has_its_own_entrys_spellings_and_keeps_what_follows_it() {
        let mut spoken = Spoken::default();
        let cpp = spoken.add_word("C++", vec![58, 66, 66]);
        spoken.insert("c plus plus", cpp);
        let csharp = spoken.add_word("C#", vec![58, 67]);
        spoken.insert("c sharp", csharp);
        assert_eq!(spoken.add_word("C#", vec![58, 67]), csharp);
        // Words that differ only in case.
        let lower = spoken.add_word("c#", vec![30, 67]);
        let kubectl = spoken.add_word("kubectl", vec![1, 2, 3]);
        let capital = spoken.add_word("Kubectl", vec![8, 2, 3]);
        let spellings = |text: &str| {
            let (entries, len) = spoken.written_at(text.as_bytes(), usize::MAX)?;
            Some(spoken.spellings(entries, &text[len..]).collect::<Vec<_>>())
        };
        let words = |texts: &[&str]| Some(texts.iter().map(|t| normalized_words(t)).collect());
        assert_eq!(spellings("C#"), words(&["C#", "c sharp"]));
        assert_eq!(spellings("C++"), words(&["C++", "c plus plus"]));
        // In other capitals, before any character that isn't a letter or digit, with what
        // follows and without it.
        let cpps = words(&["C++'s", "C++", "c plus plus's", "c plus plus"]);
        assert_eq!(spellings("c++'s"), cpps);
        let based = words(&["C#-based", "C#", "c sharp-based", "c sharp"]);
        assert_eq!(spellings("C#-based"), based);
        assert_eq!(
            spellings("C#’s"),
            words(&["C#’s", "C#", "c sharp’s", "c sharp"])
        );
        assert_eq!(spellings("C#."), words(&["C#", "c sharp"]));
        // Not where the word goes on, or doesn't start the text.
        assert_eq!(spellings("C#s"), None);
        assert_eq!(spellings("C"), None);
        assert_eq!(spellings("(C#"), None);
        // Of words that differ only in case, the one spelled as in the text, or else each.
        let at = |text: &str| spoken.written_at(text.as_bytes(), usize::MAX);
        assert_eq!(at("C#!"), Some((Entries::One(csharp), 2)));
        assert_eq!(at("c#'s"), Some((Entries::One(lower), 2)));
        let Some((Entries::Alike(group), 7)) = at("KUBECTL") else {
            panic!("{:?}", at("KUBECTL"));
        };
        assert_eq!(spoken.ids(Entries::Alike(group)), vec![kubectl, capital]);
    }

    #[test]
    fn an_entrys_word_heard_as_written_is_found_between_spoken_forms() {
        let c = spoken(&[("New York", "cube control"), ("kubectl", "york city")]);
        let tok = tokenizer();
        let units = |ids: &[u32]| -> Vec<(Range<usize>, bool)> {
            let units = c.spoken.units(&tok, ids);
            units
                .iter()
                .map(|u| (u.heard(), matches!(u, Unit::Form(_))))
                .collect()
        };
        assert_eq!(units(&[THE, NEW, YORK, COMMA]), vec![(1..3, false)]);
        let both = vec![(0..2, true), (2..4, false)];
        assert_eq!(units(&[CUBE, CONTROL, NEW, YORK]), both);
        // A form comes first, and a word doesn't reach into it.
        assert_eq!(units(&[NEW, YORK, CITY]), vec![(1..3, true)]);
        // Not where the word goes on.
        assert_eq!(units(&[THE, NEW, YORK, S]), vec![]);
        assert_eq!(units(&[THE, NEW, THE]), vec![]);
    }

    #[test]
    fn a_word_given_as_its_own_form_is_written_as_spelled() {
        let c = spoken(&[("Nginx", "Nginx")]);
        let tok = tokenizer();
        let nginx = tok.encode("Nginx");
        for text in ["the nginx,", "the Nginx,"] {
            let ids = tok.encode(text);
            assert_eq!(
                found(&c, &ids),
                vec![(1..ids.len() - 1, nginx.clone())],
                "{text}"
            );
        }
    }

    #[test]
    fn the_leftmost_form_wins_then_the_longest() {
        let c = spoken(&[
            ("kubectl", "cube control"),
            ("nginx", "control the"),
            ("New York", "the cube"),
            ("York", "the cube control"),
        ]);
        let f = found(&c, &[THE, CUBE, CONTROL, THE, COMMA]);
        assert_eq!(f, vec![(0..3, vec![YORK])]);
        let f = found(&c, &[CUBE, CONTROL, THE, COMMA]);
        assert_eq!(f, vec![(0..2, vec![1, 2, 3])]);
        let f = found(&c, &[THE, CUBE, THE, CUBE, CONTROL, THE, COMMA]);
        assert_eq!(f, vec![(0..2, vec![NEW, YORK]), (2..5, vec![YORK])]);
    }

    #[test]
    fn a_form_may_end_inside_a_token_before_punctuation() {
        let c = spoken(&[("New York", "the cube")]);
        let tok = tokenizer();
        let token = |id, frame| Token {
            id,
            frame,
            duration: 1,
        };
        // "▁the ▁cu be." spells "the cube." with the period in the form's last token, which
        // stays after the word on that token's frame.
        let heard = vec![token(THE, 0), token(36, 2), token(BE_DOT, 4)];
        assert_eq!(tok.decode(&[THE, 36, BE_DOT]), "the cube.");
        let out = c.spoken.rewrite(&tok, heard);
        assert_eq!(
            out,
            vec![
                Token {
                    id: NEW,
                    frame: 0,
                    duration: 4
                },
                Token {
                    id: YORK,
                    frame: 4,
                    duration: 0
                },
                token(DOT, 4)
            ]
        );
        let ids: Vec<u32> = out.iter().map(|t| t.id).collect();
        assert_eq!(tok.decode(&ids), "New York.");
    }

    #[test]
    fn a_form_is_open_while_the_words_that_may_change_could_go_on_with_it() {
        let c = spoken(&[("kubectl", "cube control"), ("York", "cube control the")]);
        let tok = tokenizer();
        let open = |ids: &[u32], frontier| c.spoken.open_before(&tok, ids, frontier);
        // "cube" before words that may still change could still become "cube control".
        assert_eq!(open(&[THE, CUBE, NEW], 2), Some(1));
        assert_eq!(open(&[THE, CUBE, NEW], 3), None);
        // "cu" is a word of its own, so it can't become "cube".
        assert_eq!(open(&[THE, 36, NEW], 2), None);
        // A form that stopped matching before them, or ended with nothing longer to come.
        assert_eq!(open(&[THE, CUBE, THE, NEW], 3), None);
        assert_eq!(open(&[THE, CUBE, CONTROL, NEW], 3), Some(1));
        let c = spoken(&[("kubectl", "cube control")]);
        let open = |ids: &[u32], frontier| c.spoken.open_before(&tok, ids, frontier);
        assert_eq!(open(&[THE, CUBE, CONTROL, NEW], 3), None);
        // The form's start in a token with punctuation.
        assert_eq!(open(&[THE, 36, BE_DOT, NEW], 3), None);
        assert_eq!(open(&[THE, 36, 2, NEW], 3), Some(1));
    }

    /// Granite's tokenizer, from the model named by `GRANITE_SPEECH_DIR`.
    fn granite_tokenizer() -> Tokenizer {
        let dir = std::env::var_os("GRANITE_SPEECH_DIR").expect("set GRANITE_SPEECH_DIR");
        Tokenizer::load(&std::path::Path::new(&dir).join("tokenizer.json")).unwrap()
    }

    #[test]
    #[ignore = "needs a model, named by GRANITE_SPEECH_DIR"]
    fn granites_digit_and_period_pieces_end_a_form_inside_them() {
        let tok = granite_tokenizer();
        let mut list = SpecialWords::new();
        list.add_spoken("Haiku 5.5", "Haiku 5.5").unwrap();
        list.add_spoken("Gemini 3.8 Flash", "Gemini 3.8 Flash")
            .unwrap();
        list.add_spoken("kubectl", "cube control").unwrap();
        let (c, unusable) = Compiled::new(&list, &tok);
        assert!(unusable.is_empty());
        // Granite spells "haiku 5.5." with the last period in the piece "5.".
        let ids = tok.encode("haiku 5.5.");
        let tokens = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| Token {
                id,
                frame: i as u32,
                duration: 1,
            })
            .collect();
        let out: Vec<u32> = c
            .spoken
            .rewrite(&tok, tokens)
            .iter()
            .map(|t| t.id)
            .collect();
        assert_eq!(tok.decode(&out), "Haiku 5.5.");
        // While "3.8." may still change, "Gemini 3.8 Flash" could still be under way.
        let ids = tok.encode("Gemini 3.8.");
        let frontier = ids.len() - 2;
        assert_eq!(tok.decode(&ids[frontier..]), "3.8.");
        assert_eq!(c.spoken.open_before(&tok, &ids, frontier), Some(0));
    }

    #[test]
    fn the_word_takes_the_frames_of_the_form() {
        let c = spoken(&[("kubectl", "cube control"), ("nginx", "the")]);
        let tok = tokenizer();
        let token = |id, frame, duration| Token {
            id,
            frame,
            duration,
        };
        // "cube control ," at frames 2 and 5, "control" lasting two frames.
        let heard = vec![token(4, 2, 1), token(5, 5, 2), token(7, 8, 1)];
        let out = c.spoken.rewrite(&tok, heard);
        assert_eq!(
            out,
            vec![
                token(1, 2, 1),
                token(2, 3, 2),
                token(3, 5, 2),
                token(7, 8, 1)
            ]
        );
        // One token on one frame becomes two, the first lasting no frames.
        let out = c.spoken.rewrite(&tok, vec![token(6, 4, 1)]);
        assert_eq!(out, vec![token(9, 4, 0), token(10, 4, 1)]);
        let ids: Vec<u32> = out.iter().map(|t| t.id).collect();
        assert_eq!(tok.decode(&ids), "nginx");
    }

    #[test]
    fn spoken_forms_follow_the_rules_of_the_list() {
        use super::super::Rejection;
        let mut list = SpecialWords::new();
        assert_eq!(list.add_spoken("kubectl", "cube control"), Ok(()));
        assert_eq!(list.add_spoken("rg", "rip grep"), Ok(()));
        assert_eq!(
            list.add_spoken("nginx", "Cube Control"),
            Err(Rejection::Duplicate)
        );
        // A word may be its own form, so it's written as spelled in any capitals.
        assert_eq!(list.add_spoken("Nginx", "nginx"), Ok(()));
        assert_eq!(list.add_spoken("Nginx", "NGINX"), Err(Rejection::Duplicate));
        assert_eq!(list.add_spoken("k8s", "k 8"), Err(Rejection::TooShort));
        assert_eq!(list.add_spoken("k8s", " "), Err(Rejection::Empty));
        assert_eq!(list.add_spoken("", "kates"), Err(Rejection::Empty));
        assert_eq!(
            list.add_spoken("k8s", "a b c d e f"),
            Err(Rejection::TooLong)
        );
        assert_eq!(
            list.add_spoken("東京", "tokyo"),
            Err(Rejection::UnsupportedCharacters)
        );
        assert_eq!(list.add_spoken("k8s", "k eight s"), Ok(()));
        let pairs: Vec<(&str, &str)> = list.spoken().collect();
        assert_eq!(
            pairs,
            vec![
                ("kubectl", "cube control"),
                ("rg", "rip grep"),
                ("Nginx", "nginx"),
                ("k8s", "k eight s")
            ]
        );
        assert_eq!(list.words().count(), 0);
        // Spoken forms fill the list as words do.
        for i in list.len()..SpecialWords::MAX_WORDS {
            list.add(&format!("word{i}")).unwrap();
        }
        assert_eq!(list.add_spoken("k9s", "k nine s"), Err(Rejection::ListFull));
        assert_eq!(list.add("another"), Err(Rejection::ListFull));
    }

    #[test]
    fn forms_and_words_the_vocabulary_cant_spell_are_listed() {
        let tok = tokenizer();
        let mut list = SpecialWords::new();
        list.add_spoken("Zed", "cube control").unwrap();
        list.add_spoken("kubectl", "zip control").unwrap();
        list.add_spoken("nginx", "engine x").unwrap();
        let (c, unusable) = Compiled::new(&list, &tok);
        let names: Vec<&str> = unusable.iter().map(|(w, _)| w.as_str()).collect();
        assert_eq!(names, vec!["cube control", "zip control"]);
        // "engine x" is spelled in letters, which the vocabulary has.
        assert!(!c.spoken.is_empty());
    }
}
