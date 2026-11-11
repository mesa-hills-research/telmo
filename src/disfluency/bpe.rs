//! The byte-level BPE tokenizer of GPT-2 and ModernBERT, read from a Hugging Face
//! `tokenizer.json` with a `ByteLevel` pre-tokenizer.
//!
//! Text is split as GPT-2's pattern splits it (contractions, letters, digits, other marks, each
//! with an optional space before), each piece's UTF-8 bytes start as one token each, and the BPE
//! merges join them, lowest rank first.
//!
//! shortcut: the file's NFC normalization isn't applied, and added tokens such as `[CLS]` aren't
//! matched in the text. Speech models write composed letters and never write those tokens. Apply
//! NFC first if callers pass decomposed text.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::json;

/// A byte-level BPE vocabulary.
#[derive(Debug, Clone)]
pub struct ByteLevelTokenizer {
    /// The id of each byte as a token of its own. Bytes that UTF-8 never holds may have none.
    byte_ids: [Option<u32>; 256],
    /// The BPE merges, from the ids of a pair to the merge's rank and the merged token's id.
    merges: HashMap<(u32, u32), (u32, u32)>,
    size: usize,
    /// The bytes of the longest piece.
    longest: usize,
}

impl ByteLevelTokenizer {
    /// The tokenizer a Hugging Face `tokenizer.json` describes, which must be a byte-level BPE.
    pub fn from_json(bytes: &[u8]) -> Result<ByteLevelTokenizer> {
        let bad = |m: &str| Error::Model(format!("The tokenizer file isn't usable. {m}"));
        let root = json::parse(bytes)?;
        let model = root.get("model").ok_or_else(|| bad("It has no model."))?;
        if model.get("type").and_then(|t| t.as_str()) != Some("BPE") {
            return Err(bad("Its model isn't BPE."));
        }
        let pre = root.get("pre_tokenizer");
        if pre.and_then(|p| p.get("type")).and_then(|t| t.as_str()) != Some("ByteLevel")
            || pre
                .and_then(|p| p.get("add_prefix_space"))
                .and_then(|a| a.as_bool())
                == Some(true)
        {
            return Err(bad(
                "Its pre-tokenizer isn't byte-level without a prefix space.",
            ));
        }
        let to_byte = byte_of_char();
        let piece_bytes = |piece: &str| -> Option<Vec<u8>> {
            piece.chars().map(|c| to_byte.get(&c).copied()).collect()
        };
        let vocab = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| bad("It has no vocabulary."))?;
        let mut ids: HashMap<Vec<u8>, u32> = HashMap::with_capacity(vocab.len());
        let mut size = 0;
        for (piece, id) in vocab {
            let id = id
                .as_u64()
                .filter(|&id| id < 1 << 20)
                .ok_or_else(|| bad("A vocabulary id isn't a plausible number."))?
                as u32;
            size = size.max(id as usize + 1);
            // A piece outside the byte alphabet, such as a run of spaces, only ever comes from
            // an added token.
            if let Some(bytes) = piece_bytes(piece)
                && ids.insert(bytes, id).is_some()
            {
                return Err(bad("Two pieces have the same bytes."));
            }
        }
        for token in root
            .get("added_tokens")
            .and_then(|a| a.as_array())
            .unwrap_or(&[])
        {
            let id = token
                .get("id")
                .and_then(|i| i.as_u64())
                .filter(|&id| id < 1 << 20)
                .ok_or_else(|| bad("An added token has no id."))? as u32;
            size = size.max(id as usize + 1);
        }
        let mut byte_ids = [None; 256];
        for (b, slot) in byte_ids.iter_mut().enumerate() {
            *slot = ids.get(&vec![b as u8]).copied();
        }
        if !(0..=0x7F).chain(0xC2..=0xF4).all(|b| byte_ids[b].is_some()) {
            return Err(bad("A byte of UTF-8 text has no token of its own."));
        }
        let mut merges = HashMap::new();
        let list = model
            .get("merges")
            .and_then(|m| m.as_array())
            .unwrap_or(&[]);
        for (rank, merge) in list.iter().enumerate() {
            // Newer files write a merge as a pair, older ones as "left right".
            let pair = match merge {
                json::Value::String(s) => s.split_once(' '),
                json::Value::Array(p) if p.len() == 2 => p[0].as_str().zip(p[1].as_str()),
                _ => None,
            };
            let (left, right) = pair.ok_or_else(|| bad("A merge isn't a pair of pieces."))?;
            let (Some(left), Some(right)) = (piece_bytes(left), piece_bytes(right)) else {
                return Err(bad("A merge holds a character outside the byte alphabet."));
            };
            let merged = [left.as_slice(), right.as_slice()].concat();
            if let (Some(&l), Some(&r), Some(&m)) =
                (ids.get(&left), ids.get(&right), ids.get(&merged))
            {
                merges.entry((l, r)).or_insert((rank as u32, m));
            }
        }
        let longest = ids.keys().map(Vec::len).max().unwrap_or(1);
        Ok(ByteLevelTokenizer {
            byte_ids,
            merges,
            size,
            longest,
        })
    }

    /// The bytes of the longest piece, so a text of `n` bytes takes at least `n` divided by it
    /// tokens.
    pub(crate) fn longest_piece(&self) -> usize {
        self.longest
    }

    /// The vocabulary size, added tokens included.
    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// The token ids of `text`.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for piece in split(text) {
            self.encode_piece(piece.as_bytes(), &mut out);
        }
        out
    }

    fn encode_piece(&self, bytes: &[u8], out: &mut Vec<u32>) {
        // Text is UTF-8, so every byte has a token.
        let mut symbols: Vec<u32> = (bytes.iter())
            .filter_map(|&b| self.byte_ids[b as usize])
            .collect();
        // Merge the pair of lowest rank, the leftmost of equals, until no pair merges.
        while let Some((_, i, merged)) = symbols
            .windows(2)
            .enumerate()
            .filter_map(|(i, p)| {
                let &(rank, merged) = self.merges.get(&(p[0], p[1]))?;
                Some((rank, i, merged))
            })
            .min()
        {
            symbols[i] = merged;
            symbols.remove(i + 1);
        }
        out.extend(symbols);
    }
}

/// GPT-2's map from bytes to the characters its vocabulary writes them as, inverted. Printable
/// Latin-1 bytes stand for themselves, and the others for characters from U+0100 on.
pub(crate) fn byte_of_char() -> HashMap<char, u8> {
    let printable = |b: u32| (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || b >= 0xAE;
    let mut next = 256;
    (0..256u32)
        .map(|b| {
            let c = if printable(b) {
                b
            } else {
                next += 1;
                next - 1
            };
            (char::from_u32(c).expect("a character"), b as u8)
        })
        .collect()
}

/// `text` split as GPT-2's pattern splits it,
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`.
fn split(text: &str) -> Vec<&str> {
    let is_letter = |c: char| c.is_alphabetic() && !c.is_numeric();
    let is_digit = |c: char| c.is_numeric();
    let is_other = |c: char| !c.is_whitespace() && !is_letter(c) && !is_digit(c);
    let mut out = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let len = match_at(rest, is_letter, is_digit, is_other);
        out.push(&rest[..len]);
        rest = &rest[len..];
    }
    out
}

/// The length of the pattern's match at the start of `s`, which isn't empty.
fn match_at(
    s: &str,
    is_letter: impl Fn(char) -> bool,
    is_digit: impl Fn(char) -> bool,
    is_other: impl Fn(char) -> bool,
) -> usize {
    for contraction in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
        if s.starts_with(contraction) {
            return contraction.len();
        }
    }
    let run = |s: &str, class: &dyn Fn(char) -> bool| -> usize {
        s.char_indices()
            .find(|&(_, c)| !class(c))
            .map_or(s.len(), |(i, _)| i)
    };
    let classes: [&dyn Fn(char) -> bool; 3] = [&is_letter, &is_digit, &is_other];
    for class in classes {
        let after = s.strip_prefix(' ').unwrap_or(s);
        for start in [s.len() - after.len(), 0] {
            let n = run(&s[start..], class);
            if n > 0 {
                return start + n;
            }
        }
    }
    // Whitespace. A run followed by something else leaves its last character to the next
    // match, as `\s+(?!\S)` does, unless the run is that one character.
    let n = run(s, &|c: char| c.is_whitespace());
    let last = s[..n].chars().next_back().map_or(0, char::len_utf8);
    if n < s.len() && n > last { n - last } else { n }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_splits_as_gpt2_splits_it() {
        let cases: [(&str, &[&str]); 9] = [
            (" don't", &[" don", "'t"]),
            ("I'm", &["I", "'m"]),
            (" 's", &[" '", "s"]),
            (" 'cause", &[" '", "cause"]),
            (" SessionStore,", &[" SessionStore", ","]),
            (" 2.5kg", &[" 2", ".", "5", "kg"]),
            (" dogs'.", &[" dogs", "'."]),
            ("a  b", &["a", " ", " b"]),
            (" café?!", &[" café", "?!"]),
        ];
        for (text, want) in cases {
            assert_eq!(split(text), want, "{text:?}");
        }
        assert_eq!(split("x "), ["x", " "]);
        assert!(split("").is_empty());
    }

    #[test]
    fn every_byte_has_a_character_and_merges_join_by_rank() {
        let map = byte_of_char();
        assert_eq!(map.len(), 256);
        assert_eq!(map[&'Ġ'], b' ');
        assert_eq!(map[&'a'], b'a');
        // A vocabulary of the bytes a, b and space, and three merges.
        let mut vocab = String::new();
        for (c, b) in &map {
            if !vocab.is_empty() {
                vocab.push(',');
            }
            vocab.push_str(&format!("\"{}\":{}", json_char(*c), b));
        }
        let json = format!(
            r#"{{"model":{{"type":"BPE","vocab":{{{vocab},"Ġa":300,"ab":301,"Ġab":302}},
                "merges":[["a","b"],["Ġ","a"],["Ġ","ab"]]}},
              "pre_tokenizer":{{"type":"ByteLevel","add_prefix_space":false}},
              "added_tokens":[{{"id":303,"content":"[CLS]","special":true}}]}}"#
        );
        let tok = ByteLevelTokenizer::from_json(json.as_bytes()).unwrap();
        assert_eq!(tok.len(), 304);
        // "ab" merges first (rank 0), then " " with "ab" (rank 2).
        assert_eq!(tok.encode(" ab"), [302]);
        assert_eq!(tok.encode(" ac"), [300, b'c' as u32]);
        assert_eq!(tok.encode("é"), [0xC3, 0xA9]);
    }

    fn json_char(c: char) -> String {
        match c {
            '"' => "\\\"".into(),
            '\\' => "\\\\".into(),
            _ => c.to_string(),
        }
    }
}
