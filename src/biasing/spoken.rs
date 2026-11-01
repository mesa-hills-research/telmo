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
    /// The tokens of what follows the form in its last token, written after the word, such as
    /// the period of a token "5." where the form ends in "5".
    pub rest: Vec<u32>,
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
    /// ends where the text has no letter or digit before the next space, which may be inside a
    /// token. Also returns the first token where a form could still be under way if more tokens
    /// followed, or `ids.len()`. Punctuation at the end doesn't stop a form there, since a model
    /// often closes a sentence where its audio ends and drops the punctuation once more comes.
    pub fn find(&self, tokenizer: &Tokenizer, ids: &[u32]) -> (Vec<Replacement>, usize) {
        let mut found = Vec::new();
        let mut open = ids.len();
        if self.is_empty() {
            return (found, open);
        }
        let mut text = Vec::new();
        // The end of each token's bytes in `text`.
        let mut ends = Vec::with_capacity(ids.len());
        for &id in ids {
            text.extend_from_slice(tokenizer.piece_bytes(id));
            ends.push(text.len());
        }
        let last_word_end = last_alphanumeric_end(&text);
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
                    && ends_word(&text[p + 1..])
                    && let Some(rest) = continuation(tokenizer, &text[p + 1..ends[token]])
                {
                    longest = Some(Replacement {
                        heard: i..token + 1,
                        word,
                        rest,
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

    /// The tokens written for `r` with their frames, given the frames of the first and last token
    /// heard for it. The word's tokens spread over those frames as [`place`] puts them, and the
    /// rest of the last heard token keeps that token's frame.
    pub fn written(&self, r: &Replacement, first: usize, last: usize) -> Vec<(u32, usize)> {
        let word = self.word(r);
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
        let (found, _) = self.find(tokenizer, &ids);
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

/// Whether a word ends at the start of `text`, meaning no letter or digit comes before the next
/// space.
fn ends_word(text: &[u8]) -> bool {
    let until_space = &text[..text.iter().position(|&b| b == b' ').unwrap_or(text.len())];
    !String::from_utf8_lossy(until_space)
        .chars()
        .any(char::is_alphanumeric)
}

/// Where the last letter or digit of `text` ends, or 0. Text cut inside a character's bytes
/// counts as letters to its end.
fn last_alphanumeric_end(text: &[u8]) -> usize {
    match std::str::from_utf8(text) {
        Ok(text) => text
            .char_indices()
            .rev()
            .find(|(_, c)| c.is_alphanumeric())
            .map_or(0, |(i, c)| i + c.len_utf8()),
        Err(_) => text.len(),
    }
}

/// The tokens of `rest`, the end of a token after a form, written as part of the word before
/// it, if the vocabulary spells it.
fn continuation(tokenizer: &Tokenizer, rest: &[u8]) -> Option<Vec<u32>> {
    let rest = std::str::from_utf8(rest).ok()?;
    let tokens = tokenizer.encode_continuation(rest);
    (tokenizer.decode_raw(&tokens) == rest).then_some(tokens)
}

/// The frame of token `k` of `n` written in place of a form whose tokens start at frames
/// `first` to `last`. The first takes the form's first frame, the last the frame of its last
/// token, and those between spread evenly over the frames between.
fn place(first: usize, last: usize, n: usize, k: usize) -> usize {
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
    const BE_DOT: u32 = 59;
    const DOT: u32 = 60;
    const DASH: u32 = 61;

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
    fn punctuation_in_the_last_token_at_the_end_leaves_a_form_open() {
        let c = spoken(&[("kubectl", "cube control")]);
        assert_eq!(found(&c, &[THE, 36, BE_DOT]), (vec![], 1));
    }

    #[test]
    fn punctuation_outside_ascii_at_the_end_leaves_a_form_open() {
        let c = spoken(&[("kubectl", "cube control")]);
        assert_eq!(found(&c, &[THE, CUBE, DASH]), (vec![], 1));
        assert_eq!(found(&c, &[THE, CUBE, DASH, THE]), (vec![], 4));
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
        // "Gemini 3.8." ends in the piece "8.", and a period at the end may go once more audio
        // comes, so "Gemini 3.8 Flash" could still be under way.
        assert_eq!(c.spoken.find(&tok, &tok.encode("Gemini 3.8.")).1, 0);
        assert_eq!(c.spoken.find(&tok, &tok.encode("cube—")).1, 0);
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
