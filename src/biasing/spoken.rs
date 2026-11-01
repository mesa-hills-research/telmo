//! Spoken forms written as their words. After decoding, a whole-word match of a spoken form
//! ("engine x") in the transcript's text, whatever the case of its ASCII letters, is replaced by
//! the tokens of its word ("Nginx") on the same frames, so the transcript's text stays the
//! decoding of its tokens.

use std::ops::Range;

use crate::parakeet::Token;
use crate::tokenizer::Tokenizer;

/// No word.
const NONE: u32 = u32::MAX;

#[derive(Debug, Clone)]
struct Node {
    /// (byte, node), sorted by byte.
    children: Vec<(u8, u32)>,
    /// The word a form ending here stands for, or `NONE`.
    word: u32,
}

/// Spoken forms in a tree of their bytes with ASCII letters in lower case, and the tokens of the
/// words they stand for.
#[derive(Debug, Clone, Default)]
pub(crate) struct Spoken {
    /// The tree, empty without forms. The root is node 0.
    nodes: Vec<Node>,
    words: Vec<Vec<u32>>,
}

/// A spoken form found in a transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Replacement {
    /// The tokens it takes.
    pub heard: Range<usize>,
    /// The word written in its place.
    pub word: u32,
}

impl Spoken {
    /// Adds a word's tokens, and returns its index for [`Spoken::insert`].
    pub fn add_word(&mut self, tokens: Vec<u32>) -> u32 {
        self.words.push(tokens);
        self.words.len() as u32 - 1
    }

    /// Writes word `word` for `form`, unless an earlier word has the form.
    pub fn insert(&mut self, form: &str, word: u32) {
        if self.nodes.is_empty() {
            self.nodes.push(Node {
                children: Vec::new(),
                word: NONE,
            });
        }
        let mut node = 0;
        for b in form.bytes().map(|b| b.to_ascii_lowercase()) {
            node = match self.child(node, b) {
                Some(c) => c,
                None => {
                    let new = self.nodes.len() as u32;
                    self.nodes.push(Node {
                        children: Vec::new(),
                        word: NONE,
                    });
                    let children = &mut self.nodes[node as usize].children;
                    let at = children.partition_point(|&(x, _)| x < b);
                    children.insert(at, (b, new));
                    new
                }
            };
        }
        let end = &mut self.nodes[node as usize].word;
        if *end == NONE {
            *end = word;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The tokens of the word written in place of `r`.
    pub fn word(&self, r: &Replacement) -> &[u32] {
        &self.words[r.word as usize]
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
    /// ends at the end of a token followed by a word boundary or by nothing. Also returns the
    /// first token where a form could still be under way if more tokens followed, or
    /// `ids.len()`. Punctuation at the end doesn't stop a form there, since a model often closes
    /// a sentence where its audio ends and drops the punctuation once more audio comes.
    pub fn find(&self, tokenizer: &Tokenizer, ids: &[u32]) -> (Vec<Replacement>, usize) {
        let mut found = Vec::new();
        let mut open = ids.len();
        if self.is_empty() {
            return (found, open);
        }
        let mut text = Vec::new();
        // The end of each token's bytes in `text`, and of the last token with a letter or digit.
        let mut ends = Vec::with_capacity(ids.len());
        let mut last_word_end = 0;
        for &id in ids {
            let piece = tokenizer.piece_bytes(id);
            text.extend_from_slice(piece);
            ends.push(text.len());
            if piece.iter().any(u8::is_ascii_alphanumeric) || !piece.is_ascii() {
                last_word_end = text.len();
            }
        }
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
            let mut at_end = true;
            for (p, &b) in text.iter().enumerate().skip(start + 1) {
                if p == last_word_end {
                    open = open.min(i);
                }
                match self.child(node, b.to_ascii_lowercase()) {
                    Some(c) => node = c,
                    None => {
                        at_end = false;
                        break;
                    }
                }
                while ends[token] < p + 1 {
                    token += 1;
                }
                let word = self.nodes[node as usize].word;
                if word != NONE
                    && ends[token] == p + 1
                    && ids
                        .get(token + 1)
                        .is_none_or(|&t| tokenizer.is_word_boundary(t))
                {
                    longest = Some(Replacement {
                        heard: i..token + 1,
                        word,
                    });
                }
            }
            if at_end {
                open = open.min(i);
            }
            match longest {
                Some(r) => {
                    i = r.heard.end;
                    found.push(r);
                }
                None => i += 1,
            }
        }
        (found, open)
    }

    /// `tokens` with each spoken form written as its word. The word's tokens take the frames
    /// the form took, as [`place`] gives them.
    pub fn rewrite(&self, tokenizer: &Tokenizer, tokens: Vec<Token>) -> Vec<Token> {
        if self.is_empty() {
            return tokens;
        }
        let ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
        let (found, _) = self.find(tokenizer, &ids);
        if found.is_empty() {
            return tokens;
        }
        let mut out = Vec::with_capacity(tokens.len());
        let mut next = 0;
        for r in &found {
            out.extend_from_slice(&tokens[next..r.heard.start]);
            let heard = &tokens[r.heard.clone()];
            let (first, last) = (heard[0], heard[heard.len() - 1]);
            let word = self.word(r);
            let frame = |k| place(first.frame as usize, last.frame as usize, word.len(), k) as u32;
            out.extend(word.iter().enumerate().map(|(k, &id)| {
                let end = if k + 1 == word.len() {
                    last.frame + last.duration
                } else {
                    frame(k + 1)
                };
                Token {
                    id,
                    frame: frame(k),
                    duration: end - frame(k),
                }
            }));
            next = r.heard.end;
        }
        out.extend_from_slice(&tokens[next..]);
        out
    }
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

    /// The forms found in tokens `ids`, and where a form could still be under way.
    fn found(c: &Compiled, ids: &[u32]) -> (Found, usize) {
        let (found, open) = c.spoken.find(&tokenizer(), ids);
        let found = found
            .iter()
            .map(|r| (r.heard.clone(), c.spoken.word(r).to_vec()))
            .collect();
        (found, open)
    }

    // Tokens of the test vocabulary.
    const THE: u32 = 6;
    const CUBE: u32 = 4;
    const CONTROL: u32 = 5;
    const COMMA: u32 = 7;
    const S: u32 = 47;
    const NEW: u32 = 42;
    const YORK: u32 = 43;

    #[test]
    fn whole_words_of_a_form_are_found_in_any_capitals() {
        let c = spoken(&[("kubectl", "cube control")]);
        let kubectl = || vec![(1..3, vec![1, 2, 3])];
        assert_eq!(found(&c, &[THE, CUBE, CONTROL, COMMA, THE]), (kubectl(), 5));
        // At the end, it's found and could still be under way, also before punctuation there,
        // which more audio may take away.
        assert_eq!(found(&c, &[THE, CUBE, CONTROL]), (kubectl(), 1));
        assert_eq!(found(&c, &[THE, CUBE, CONTROL, COMMA]), (kubectl(), 1));
        assert_eq!(found(&c, &[THE, CUBE, COMMA]), (vec![], 1));
        // A word that goes on isn't a whole-word match.
        assert_eq!(found(&c, &[CUBE, CONTROL, S, THE]), (vec![], 4));
        assert_eq!(found(&c, &[THE, CUBE, THE]), (vec![], 3));
        // A form's start at the end could still become the form.
        assert_eq!(found(&c, &[THE, CUBE]), (vec![], 1));
        let tok = tokenizer();
        let partial = tok.encode("the cube cont");
        assert_eq!(found(&c, &partial), (vec![], 1));
        // A form spelled with other tokens than its own is found too.
        let split = [THE, 36, 2, CONTROL, COMMA];
        assert_eq!(tok.decode(&split), "the cube control,");
        assert_eq!(found(&c, &split), (vec![(1..4, vec![1, 2, 3])], 1));
        for (text, matches) in [
            ("the Cube control,", true),
            ("the Cube Control,", true),
            ("the cube Control,", true),
            ("the cube contro,", false),
        ] {
            let ids = tok.encode(text);
            assert_eq!(tok.decode(&ids), text);
            assert_eq!(found(&c, &ids).0.len(), matches as usize, "{text}");
        }
    }

    #[test]
    fn a_word_given_as_its_own_form_is_written_as_spelled() {
        let c = spoken(&[("Nginx", "Nginx")]);
        let tok = tokenizer();
        let nginx = tok.encode("Nginx");
        for text in ["the nginx,", "the Nginx,"] {
            let ids = tok.encode(text);
            assert_eq!(
                found(&c, &ids).0,
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
        let (f, _) = found(&c, &[THE, CUBE, CONTROL, THE, COMMA]);
        assert_eq!(f, vec![(0..3, vec![YORK])]);
        let (f, _) = found(&c, &[CUBE, CONTROL, THE, COMMA]);
        assert_eq!(f, vec![(0..2, vec![1, 2, 3])]);
        let (f, _) = found(&c, &[THE, CUBE, THE, CUBE, CONTROL, THE, COMMA]);
        assert_eq!(f, vec![(0..2, vec![NEW, YORK]), (2..5, vec![YORK])]);
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
