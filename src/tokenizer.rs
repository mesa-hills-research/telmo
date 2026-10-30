//! Turns token ids back into text, for the BPE vocabularies of Hugging Face `tokenizer.json`
//! files that use a Metaspace (`▁`) pre-tokenizer.

use std::path::Path;

use crate::error::{Error, Result};
use crate::json;

#[derive(Debug, Clone, PartialEq)]
enum Piece {
    /// Text with `▁` already turned into spaces.
    Text(String),
    /// A byte-fallback piece such as `<0x0A>`.
    Byte(u8),
    /// An added token such as `<unk>` or a language tag, left out of the text.
    Control,
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pieces: Vec<Piece>,
}

impl Tokenizer {
    pub fn load(path: &Path) -> Result<Tokenizer> {
        let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
        Tokenizer::from_json(&bytes)
    }

    pub fn from_json(bytes: &[u8]) -> Result<Tokenizer> {
        let bad = |m: &str| Error::Model(format!("The tokenizer file isn't usable. {m}"));
        let root = json::parse(bytes)?;
        let model = root.get("model").ok_or_else(|| bad("It has no model."))?;
        if model.get("type").and_then(|t| t.as_str()) != Some("BPE") {
            return Err(bad("Its model isn't BPE."));
        }
        let vocab = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| bad("It has no vocabulary."))?;
        let mut entries: Vec<(usize, &str)> = Vec::with_capacity(vocab.len());
        for (piece, id) in vocab {
            let id = id
                .as_u64()
                .ok_or_else(|| bad("A vocabulary id isn't a number."))?
                as usize;
            entries.push((id, piece));
        }
        let mut added = Vec::new();
        if let Some(tokens) = root.get("added_tokens").and_then(|a| a.as_array()) {
            for token in tokens {
                let id = token
                    .get("id")
                    .and_then(|i| i.as_u64())
                    .ok_or_else(|| bad("An added token has no id."))?
                    as usize;
                added.push(id);
            }
        }
        let size = entries
            .iter()
            .map(|e| e.0)
            .chain(added.iter().copied())
            .max()
            .map_or(0, |m| m + 1);
        if size > 1 << 20 {
            return Err(bad("Its ids are implausibly large."));
        }
        let mut pieces = vec![Piece::Control; size];
        let mut seen = vec![false; size];
        for (id, piece) in entries {
            if std::mem::replace(&mut seen[id], true) {
                return Err(bad("Two pieces share an id."));
            }
            pieces[id] = parse_piece(piece);
        }
        for id in added {
            pieces[id] = Piece::Control;
        }
        Ok(Tokenizer { pieces })
    }

    /// The vocabulary size, added tokens included.
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Joins the pieces of `ids` into text. Added tokens are left out, byte pieces are joined
    /// into UTF-8, and the space the first word's `▁` stands for is dropped.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            match self.pieces.get(id as usize) {
                Some(Piece::Text(text)) => bytes.extend_from_slice(text.as_bytes()),
                Some(Piece::Byte(b)) => bytes.push(*b),
                Some(Piece::Control) | None => {}
            }
        }
        let text = String::from_utf8_lossy(&bytes);
        text.strip_prefix(' ').unwrap_or(&text).to_string()
    }
}

fn parse_piece(piece: &str) -> Piece {
    if let Some(hex) = piece.strip_prefix("<0x").and_then(|p| p.strip_suffix('>'))
        && hex.len() == 2
        && let Ok(b) = u8::from_str_radix(hex, 16)
    {
        return Piece::Byte(b);
    }
    Piece::Text(piece.replace('\u{2581}', " "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pieces_join_into_words() {
        let json = r#"{"added_tokens":[{"id":0,"content":"<unk>"},{"id":5,"content":"<blank>"}],
            "model":{"type":"BPE","vocab":{"<unk>":0,"▁hel":1,"lo":2,"▁wor":3,"ld,":4,"<0xC3>":6,"<0xA9>":7}}}"#;
        let tok = Tokenizer::from_json(json.as_bytes()).unwrap();
        assert_eq!(tok.len(), 8);
        assert_eq!(tok.decode(&[1, 2, 5, 3, 4, 0]), "hello world,");
        assert_eq!(tok.decode(&[1, 6, 7]), "hel\u{e9}");
        assert_eq!(tok.decode(&[]), "");
        assert_eq!(tok.decode(&[99]), "");
    }

    #[test]
    fn other_models_and_duplicate_ids_are_refused() {
        assert!(Tokenizer::from_json(br#"{"model":{"type":"Unigram","vocab":[]}}"#).is_err());
        assert!(
            Tokenizer::from_json(br#"{"model":{"type":"BPE","vocab":{"a":1,"b":1}}}"#).is_err()
        );
    }
}
