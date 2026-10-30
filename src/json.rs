//! A small JSON reader for model configs, tokenizers and safetensors headers.
//!
//! It follows RFC 8259, keeps object members in their order, refuses duplicate keys and
//! limits nesting depth, so a hostile file can't exhaust the stack.

use crate::error::{Error, Result};

const MAX_DEPTH: usize = 128;

/// A parsed JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    /// A number, kept as its source text so integers past 2^53 stay exact.
    Number(String),
    String(String),
    Array(Vec<Value>),
    /// Members in document order.
    Object(Vec<(String, Value)>),
}

impl Value {
    /// The member named `key`, if this is an object that has it.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(members) => members.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Value)]> {
        match self {
            Value::Object(members) => Some(members),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Number(text) => text.parse().ok(),
            _ => None,
        }
    }

    /// The value as an unsigned integer, if it is one exactly (no fraction or exponent).
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Number(text) => text.parse().ok(),
            _ => None,
        }
    }

    /// The value as a signed integer, if it is one exactly (no fraction or exponent).
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Number(text) => text.parse().ok(),
            _ => None,
        }
    }
}

impl Value {
    /// The value as compact JSON text. Numbers are written as they were read.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Number(n) => out.push_str(n),
            Value::String(s) => quote(s, out),
            Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Value::Object(members) => {
                out.push('{');
                for (i, (key, value)) in members.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    quote(key, out);
                    out.push(':');
                    value.write(out);
                }
                out.push('}');
            }
        }
    }

    /// A number value.
    pub fn number(n: impl std::fmt::Display) -> Value {
        Value::Number(n.to_string())
    }
}

/// Writes `s` as a JSON string.
fn quote(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Parses a whole document. Whitespace may surround the value, nothing else.
pub fn parse(text: &[u8]) -> Result<Value> {
    let mut parser = Parser { text, pos: 0 };
    parser.skip_whitespace();
    let value = parser.value(0)?;
    parser.skip_whitespace();
    if parser.pos != text.len() {
        return Err(parser.error("text after the value"));
    }
    Ok(value)
}

struct Parser<'a> {
    text: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn error(&self, message: &'static str) -> Error {
        Error::Json {
            offset: self.pos,
            message,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.text.get(self.pos).copied()
    }

    fn skip_whitespace(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.peek() {
            self.pos += 1;
        }
    }

    fn expect_word(&mut self, word: &[u8]) -> Result<()> {
        if self.text[self.pos..].starts_with(word) {
            self.pos += word.len();
            Ok(())
        } else {
            Err(self.error("unknown word"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth >= MAX_DEPTH {
            return Err(self.error("nested too deeply"));
        }
        match self.peek() {
            None => Err(self.error("unexpected end")),
            Some(b'n') => self.expect_word(b"null").map(|_| Value::Null),
            Some(b't') => self.expect_word(b"true").map(|_| Value::Bool(true)),
            Some(b'f') => self.expect_word(b"false").map(|_| Value::Bool(false)),
            Some(b'"') => self.string().map(Value::String),
            Some(b'[') => self.array(depth),
            Some(b'{') => self.object(depth),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.error("unexpected character")),
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value> {
        self.pos += 1;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.skip_whitespace();
            items.push(self.value(depth + 1)?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(self.error("expected , or ]")),
            }
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value> {
        self.pos += 1;
        let mut members: Vec<(String, Value)> = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Object(members));
        }
        // Large objects (a tokenizer's vocabulary) are checked for duplicates once at the end.
        loop {
            self.skip_whitespace();
            if self.peek() != Some(b'"') {
                return Err(self.error("expected a string key"));
            }
            let key = self.string()?;
            self.skip_whitespace();
            if self.peek() != Some(b':') {
                return Err(self.error("expected :"));
            }
            self.pos += 1;
            self.skip_whitespace();
            let value = self.value(depth + 1)?;
            members.push((key, value));
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(self.error("expected , or }")),
            }
        }
        let mut keys: Vec<&str> = members.iter().map(|(k, _)| k.as_str()).collect();
        keys.sort_unstable();
        if keys.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(self.error("duplicate key"));
        }
        Ok(Value::Object(members))
    }

    fn number(&mut self) -> Result<Value> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => self.digits(),
            _ => return Err(self.error("expected a digit")),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("expected a digit after ."));
            }
            self.digits();
        }
        if let Some(b'e' | b'E') = self.peek() {
            self.pos += 1;
            if let Some(b'+' | b'-') = self.peek() {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("expected a digit in the exponent"));
            }
            self.digits();
        }
        // The slice holds only ASCII digits, signs, '.', 'e' and 'E'.
        let text = std::str::from_utf8(&self.text[start..self.pos]).expect("ASCII");
        Ok(Value::Number(text.to_string()))
    }

    fn digits(&mut self) {
        while let Some(b'0'..=b'9') = self.peek() {
            self.pos += 1;
        }
    }

    fn string(&mut self) -> Result<String> {
        self.pos += 1; // the opening quote
        let mut out = Vec::new();
        loop {
            let run_start = self.pos;
            while let Some(b) = self.peek() {
                if b == b'"' || b == b'\\' || b < 0x20 {
                    break;
                }
                self.pos += 1;
            }
            out.extend_from_slice(&self.text[run_start..self.pos]);
            match self.peek() {
                None => return Err(self.error("unterminated string")),
                Some(b'"') => {
                    self.pos += 1;
                    break;
                }
                Some(b'\\') => {
                    self.pos += 1;
                    self.escape(&mut out)?;
                }
                Some(_) => return Err(self.error("control character in a string")),
            }
        }
        String::from_utf8(out).map_err(|_| self.error("invalid UTF-8 in a string"))
    }

    fn escape(&mut self, out: &mut Vec<u8>) -> Result<()> {
        let Some(c) = self.peek() else {
            return Err(self.error("unterminated escape"));
        };
        self.pos += 1;
        let byte = match c {
            b'"' => b'"',
            b'\\' => b'\\',
            b'/' => b'/',
            b'b' => 0x08,
            b'f' => 0x0c,
            b'n' => b'\n',
            b'r' => b'\r',
            b't' => b'\t',
            b'u' => {
                let first = self.hex4()?;
                let code = if (0xD800..0xDC00).contains(&first) {
                    if !self.text[self.pos..].starts_with(b"\\u") {
                        return Err(self.error("unpaired surrogate"));
                    }
                    self.pos += 2;
                    let second = self.hex4()?;
                    if !(0xDC00..0xE000).contains(&second) {
                        return Err(self.error("unpaired surrogate"));
                    }
                    0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                } else if (0xDC00..0xE000).contains(&first) {
                    return Err(self.error("unpaired surrogate"));
                } else {
                    first
                };
                let ch = char::from_u32(code).ok_or_else(|| self.error("invalid code point"))?;
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                return Ok(());
            }
            _ => return Err(self.error("unknown escape")),
        };
        out.push(byte);
        Ok(())
    }

    fn hex4(&mut self) -> Result<u32> {
        let digits = self
            .text
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| self.error("short \\u escape"))?;
        let mut code = 0u32;
        for &d in digits {
            let v = (d as char)
                .to_digit(16)
                .ok_or_else(|| self.error("bad hex digit"))?;
            code = code * 16 + v;
        }
        self.pos += 4;
        Ok(code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(text: &str) -> Value {
        parse(text.as_bytes()).unwrap()
    }

    fn bad(text: &str) {
        assert!(
            parse(text.as_bytes()).is_err(),
            "{text:?} should be refused"
        );
    }

    #[test]
    fn values_of_every_kind_parse() {
        let v = ok(r#" {"a": [1, -2.5e3, true, false, null], "b": {"c": "d"}, "e": ""} "#);
        let a = v.get("a").unwrap().as_array().unwrap();
        assert_eq!(a[0].as_u64(), Some(1));
        assert_eq!(a[1].as_f64(), Some(-2500.0));
        assert_eq!(a[2], Value::Bool(true));
        assert_eq!(a[4], Value::Null);
        assert_eq!(v.get("b").unwrap().get("c").unwrap().as_str(), Some("d"));
        assert_eq!(v.get("e").unwrap().as_str(), Some(""));
        assert_eq!(v.as_object().unwrap()[0].0, "a");
    }

    #[test]
    fn large_integers_stay_exact() {
        assert_eq!(ok("18446744073709551615").as_u64(), Some(u64::MAX));
        assert_eq!(ok("9007199254740993").as_u64(), Some(9007199254740993));
        assert_eq!(ok("1.5").as_u64(), None);
        assert_eq!(ok("-3").as_i64(), Some(-3));
    }

    #[test]
    fn values_write_back_as_the_same_json() {
        let text = br#"{"a":[1,-2.5e3,true,null],"b":"q\"uote\\ \n\u0001 \u00e9","c":{}}"#;
        let value = parse(text).unwrap();
        assert_eq!(parse(value.to_json().as_bytes()).unwrap(), value);
        assert_eq!(Value::String("\u{1}".into()).to_json(), "\"\\u0001\"");
    }

    #[test]
    fn escapes_and_unicode_decode() {
        assert_eq!(
            ok(r#""a\"b\\c\/d\n\t\u00e9\u2581""#).as_str(),
            Some("a\"b\\c/d\n\té▁")
        );
        assert_eq!(ok(r#""\ud83d\ude00""#).as_str(), Some("😀"));
        assert_eq!(ok("\"▁the\"").as_str(), Some("▁the"));
    }

    #[test]
    fn malformed_documents_are_refused() {
        for text in [
            "",
            "{",
            "[1,]",
            "{\"a\":1,}",
            "01",
            "1.",
            "-",
            "1e",
            "tru",
            "\"abc",
            "\"\\x\"",
            "\"\\ud800\"",
            "\"\\udc00\"",
            "\"a\u{1}b\"",
            "{\"a\":1 \"b\":2}",
            "[1] 2",
            "{1:2}",
            "{\"a\":1,\"a\":2}",
            "\"\\u12\"",
        ] {
            bad(text);
        }
        assert!(parse(b"\"\xff\"").is_err());
    }

    #[test]
    fn deep_nesting_is_refused_without_overflowing_the_stack() {
        let deep = "[".repeat(100_000);
        bad(&deep);
        let fine = format!("{}{}", "[".repeat(100), "]".repeat(100));
        ok(&fine);
    }
}
