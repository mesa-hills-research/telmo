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
    /// An added token marked special, such as `<unk>` or `<blank>`, left out of the text.
    Control,
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pieces: Vec<Piece>,
    /// Whether the file's decoder includes `ByteFallback`, which decodes each run of byte pieces
    /// on its own and turns a run that isn't valid UTF-8 into one U+FFFD per byte.
    byte_runs: bool,
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
                let content = token.get("content").and_then(|c| c.as_str()).unwrap_or("");
                let special = token
                    .get("special")
                    .and_then(|s| s.as_bool())
                    .unwrap_or(true);
                added.push((id, special, content));
            }
        }
        let size = entries
            .iter()
            .map(|e| e.0)
            .chain(added.iter().map(|a| a.0))
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
        // Added tokens that aren't special, such as the digits, are text like any piece.
        for (id, special, content) in added {
            pieces[id] = if special {
                Piece::Control
            } else {
                parse_piece(content)
            };
        }
        let decoder = root.get("decoder");
        let byte_runs = decoder.is_some_and(|d| {
            d.get("type").and_then(|t| t.as_str()) == Some("ByteFallback")
                || d.get("decoders").and_then(|l| l.as_array()).is_some_and(|l| {
                    l.iter()
                        .any(|d| d.get("type").and_then(|t| t.as_str()) == Some("ByteFallback"))
                })
        });
        Ok(Tokenizer { pieces, byte_runs })
    }

    /// The vocabulary size, added tokens included.
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Whether the vocabulary starts with these pieces (as written in the vocabulary, with
    /// `▁` for spaces). Special tokens match anything.
    pub fn has_pieces(&self, pieces: &[&str]) -> bool {
        pieces.len() <= self.pieces.len()
            && pieces
                .iter()
                .zip(&self.pieces)
                .all(|(p, mine)| *mine == Piece::Control || parse_piece(p) == *mine)
    }

    /// Joins the pieces of `ids` into text. Special tokens are left out, byte pieces are joined
    /// into UTF-8, and the space the first word's `▁` stands for is dropped.
    ///
    /// With a `ByteFallback` decoder, as in Hugging Face's tokenizers, each run of byte pieces
    /// decodes on its own, and a run that isn't valid UTF-8 becomes one U+FFFD per byte. Without
    /// it, the bytes of all pieces decode together and each invalid sequence becomes one U+FFFD.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        let mut text = String::new();
        for &id in ids {
            match self.pieces.get(id as usize) {
                Some(Piece::Text(piece)) if self.byte_runs => {
                    flush_byte_run(&mut bytes, &mut text);
                    text.push_str(piece);
                }
                Some(Piece::Text(piece)) => bytes.extend_from_slice(piece.as_bytes()),
                Some(Piece::Byte(b)) => bytes.push(*b),
                Some(Piece::Control) | None => {}
            }
        }
        if self.byte_runs {
            flush_byte_run(&mut bytes, &mut text);
        } else {
            text = String::from_utf8_lossy(&bytes).into_owned();
        }
        match text.strip_prefix(' ') {
            Some(rest) => rest.to_string(),
            None => text,
        }
    }
}

/// Appends a run of byte pieces as `ByteFallback` decodes it.
fn flush_byte_run(bytes: &mut Vec<u8>, text: &mut String) {
    match std::str::from_utf8(bytes) {
        Ok(s) => text.push_str(s),
        Err(_) => text.extend(std::iter::repeat_n('\u{FFFD}', bytes.len())),
    }
    bytes.clear();
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
        let json = r#"{"added_tokens":[{"id":0,"content":"<unk>","special":true},{"id":5,"content":"<blank>","special":true},{"id":8,"content":"7","special":false}],
            "model":{"type":"BPE","vocab":{"<unk>":0,"▁hel":1,"lo":2,"▁wor":3,"ld,":4,"<0xC3>":6,"<0xA9>":7,"7":8}}}"#;
        let tok = Tokenizer::from_json(json.as_bytes()).unwrap();
        assert_eq!(tok.len(), 9);
        assert_eq!(tok.decode(&[1, 2, 5, 3, 4, 0]), "hello world,");
        assert_eq!(tok.decode(&[3, 8, 8]), "wor77");
        assert_eq!(tok.decode(&[1, 6, 7]), "hel\u{e9}");
        assert_eq!(tok.decode(&[]), "");
        assert_eq!(tok.decode(&[99]), "");
        assert!(tok.has_pieces(&["<unk>", "▁hel", "lo"]));
        assert!(!tok.has_pieces(&["<unk>", "▁hel", "la"]));
    }

    #[test]
    fn a_byte_fallback_decoder_decodes_each_run_of_bytes_alone() {
        let vocab = r#""model":{"type":"BPE","vocab":{"<unk>":0,"▁a":1,"b":2,"<0xC3>":3,"<0xA9>":4,"<0x41>":5,"<0xFF>":6}}"#;
        let added = r#""added_tokens":[{"id":0,"content":"<unk>","special":true}]"#;
        let plain = Tokenizer::from_json(format!("{{{added},{vocab}}}").as_bytes()).unwrap();
        let runs = Tokenizer::from_json(
            format!(
                r#"{{{added},"decoder":{{"type":"Sequence","decoders":[{{"type":"Replace"}},{{"type":"ByteFallback"}},{{"type":"Fuse"}}]}},{vocab}}}"#
            )
            .as_bytes(),
        )
        .unwrap();
        for tok in [&plain, &runs] {
            assert_eq!(tok.decode(&[1, 3, 4, 2]), "a\u{e9}b");
            // A special token between byte pieces doesn't end their run.
            assert_eq!(tok.decode(&[3, 0, 4]), "\u{e9}");
        }
        // A run that isn't valid UTF-8 gives one replacement character per byte, the valid ones too.
        assert_eq!(runs.decode(&[5, 6, 2]), "\u{FFFD}\u{FFFD}b");
        assert_eq!(plain.decode(&[5, 6, 2]), "A\u{FFFD}b");
        // Bytes on both sides of a text piece don't join.
        assert_eq!(runs.decode(&[3, 2, 4]), "\u{FFFD}b\u{FFFD}");
        assert_eq!(plain.decode(&[3, 2, 4]), "\u{FFFD}b\u{FFFD}");
        assert_eq!(runs.decode(&[1]), "a");
    }

    #[test]
    fn other_models_and_duplicate_ids_are_refused() {
        assert!(Tokenizer::from_json(br#"{"model":{"type":"Unigram","vocab":[]}}"#).is_err());
        assert!(
            Tokenizer::from_json(br#"{"model":{"type":"BPE","vocab":{"a":1,"b":1}}}"#).is_err()
        );
    }
}
