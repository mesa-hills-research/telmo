use super::*;
use crate::parakeet::Token;

fn prose() -> Symbols {
    Symbols::for_language("en").unwrap()
}

#[test]
fn the_cases_come_out_as_written_in_prose_and_in_a_code_field() {
    let cases = include_str!("../../tests/symbols/cases.tsv");
    let mut checked = 0;
    for line in cases
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let fields: Vec<&str> = line.split('\t').collect();
        let (said, in_prose) = (fields[0], fields[1]);
        assert_eq!(prose().write(said, None), in_prose, "{said}");
        if let Some(&in_code) = fields.get(2) {
            let want = if in_code == "=" { in_prose } else { in_code };
            assert_eq!(
                prose().for_code().write(said, None),
                want,
                "code field: {said}"
            );
        }
        checked += 1;
    }
    assert!(checked >= 90);
}

#[test]
fn only_english_has_symbols() {
    assert_eq!(Symbols::for_language("en-GB"), Symbols::for_language("en"));
    assert!(Symbols::for_language("de").is_none());
    assert!(!prose().is_for_code());
    assert!(prose().for_code().is_for_code());
}

#[test]
fn a_hesitation_inside_a_run_goes_with_it_when_fillers_go() {
    let words: Vec<&str> = "open source slash um engine dot uh rs now"
        .split(' ')
        .collect();
    let joins = prose().plan(&words, true, None);
    assert_eq!(
        joins,
        [Join {
            words: 1..8,
            text: "source/engine.rs".into()
        }]
    );
    // When fillers stay, so do the hesitations, and no run reads as code.
    assert_eq!(prose().plan(&words, false, None), []);
}

#[test]
fn a_listed_path_is_written_as_spelled_and_a_new_file_joins_its_folder() {
    let paths = Paths::from_list([
        "src/engine.rs",
        "src/biasing/spoken.rs",
        "README.md",
        "docs/how-to.md",
    ]);
    for (said, written) in [
        (
            "So in source slash engine dot rs in the loop function,",
            "So in src/engine.rs in the loop function,",
        ),
        ("Update the readme dot md.", "Update the README.md."),
        (
            "Add source slash biasing slash paths dot rs.",
            "Add src/biasing/paths.rs.",
        ),
        ("Open source slash new dot rs.", "Open src/new.rs."),
        ("Open source slash notes.", "Open src/notes."),
        ("Read docs slash how to dot md.", "Read docs/how-to.md."),
        ("Open source slash engine dot r s.", "Open src/engine.rs."),
        ("Open source slash engine dot rb.", "Open src/engine.rb."),
        ("Open sauce slash engine dot rb.", "Open sauce/engine.rb."),
        // Prose keeps its words.
        ("Use and slash or here.", "Use and slash or here."),
        ("It's open source.", "It's open source."),
    ] {
        assert_eq!(prose().write(said, Some(&paths)), written, "{said}");
    }
}

#[test]
fn words_written_as_code_are_told_apart() {
    for word in [
        "src/engine.rs",
        "engine.rs,",
        "lead_system",
        "std::fs",
        "--force",
        "-v",
        ".gitignore",
        "~/projects",
        "(src/lib.rs)",
    ] {
        assert!(is_code_word(word), "{word}");
    }
    for word in [
        "U.S.",
        "e.g.",
        "well-known",
        "-",
        "...",
        "3.5",
        "a.",
        "so",
        "x-ray",
    ] {
        assert!(!is_code_word(word), "{word}");
    }
}

#[test]
fn a_run_still_open_at_the_end_of_a_stream_holds_its_words() {
    let open = |text: &str| prose().open_from(&text.split(' ').collect::<Vec<_>>());
    assert_eq!(open("so in source slash engine dot"), Some(2));
    assert_eq!(open("so in source slash engine"), Some(2));
    assert_eq!(open("so in source slash"), Some(2));
    // The word before a flag waits too, though a flag never joins it.
    assert_eq!(open("run it with dash dash force"), Some(2));
    assert_eq!(open("so in src/engine.rs in"), None);
    assert_eq!(open("use and slash or the"), None);
    assert_eq!(open("hello there"), None);
    assert_eq!(open("engine."), None);
    assert_eq!(open("engine dot, rs"), None);
}

/// A tokenizer with a piece for each character, with and without the word-start marker, so any
/// text of them spells.
fn letters() -> Tokenizer {
    let chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._-,";
    let mut vocab = vec!["\"<blank>\":0".to_string(), "\"▁\":1".to_string()];
    let mut merges = Vec::new();
    for (i, c) in chars.chars().enumerate() {
        vocab.push(format!("\"{c}\":{}", 2 * i + 2));
        vocab.push(format!("\"▁{c}\":{}", 2 * i + 3));
        merges.push(format!("[\"▁\",\"{c}\"]"));
    }
    let json = format!(
        r#"{{"added_tokens":[{{"id":0,"content":"<blank>","special":true}}],"model":{{"type":"BPE","vocab":{{{}}},"merges":[{}]}}}}"#,
        vocab.join(","),
        merges.join(",")
    );
    Tokenizer::from_json(json.as_bytes()).unwrap()
}

/// `text`'s tokens, each word's on frames of its own, ten apart.
fn heard(tok: &Tokenizer, text: &str) -> Vec<Token> {
    let mut out = Vec::new();
    for (w, word) in text.split(' ').enumerate() {
        for (k, id) in tok.encode(word).into_iter().enumerate() {
            out.push(Token {
                id,
                frame: (10 * w + k) as u32,
                duration: 1,
            });
        }
    }
    out
}

#[test]
fn a_run_is_written_as_tokens_on_the_frames_of_its_words() {
    let tok = letters();
    let tokens = heard(&tok, "in source slash engine dot rs, now");
    let ids = |t: &[Token]| t.iter().map(|t| t.id).collect::<Vec<_>>();
    let spread = |word: &[Token], ids: Vec<u32>| {
        let (first, last) = (word[0], word[word.len() - 1]);
        let frame = |k| {
            crate::biasing::place(first.frame as usize, last.frame as usize, ids.len(), k) as u32
        };
        (0..ids.len())
            .map(|k| Token {
                id: ids[k],
                frame: frame(k),
                duration: 1,
            })
            .collect()
    };
    let plan = |texts: &[&str]| prose().plan(texts, false, None);
    let out = rewrite(&tok, tokens.clone(), |t| t.id, plan, spread);
    assert_eq!(tok.decode(&ids(&out)), "in source/engine.rs, now");
    // The run's tokens take the frames from its first word's to its last word's.
    let run: Vec<&Token> = out.iter().skip(2).take(17).collect();
    assert_eq!(run.first().unwrap().frame, 10);
    assert_eq!(run.last().unwrap().frame, 52);
    assert_eq!(out.last().unwrap().frame, 62);
    // Without a run, the tokens stay as they are.
    let plain = heard(&tok, "nothing to see here");
    assert_eq!(rewrite(&tok, plain.clone(), |t| t.id, plan, spread), plain);
    // ASCII the vocabulary lacks ("~") takes written-only pieces.
    let tokens = heard(&tok, "lead underscore x and tilde slash y");
    let out = rewrite(&tok, tokens, |t| t.id, plan, spread);
    assert_eq!(tok.decode(&ids(&out)), "lead_x and ~/y");
    // A run whose text the vocabulary can't spell stays as heard.
    let tokens = heard(&tok, "open x dot rs");
    let plan = |_: &[&str]| {
        vec![Join {
            words: 1..4,
            text: "naïve.rs".into(),
        }]
    };
    let out = rewrite(&tok, tokens.clone(), |t| t.id, plan, spread);
    assert_eq!(out, tokens);
}
