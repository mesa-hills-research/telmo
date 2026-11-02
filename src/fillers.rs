//! Fillers a transcript can leave out, for dictation. English is the only language so far.
//!
//! Two kinds of words go.
//!
//! - A hesitation sound such as um, umm, uh, uhh, uhm, er, erm, ah, hm, hmm, mm or mmm, or a longer
//!   spelling of one, the whole word in lower case or with one capital. "mm" after a number
//!   stays, as millimetres, and so does a word in capitals (UM), as an acronym.
//! - The first of two adjacent copies of a word that can't end a clause, in any case, with a comma
//!   or a hesitation between them or nothing ("the the", "I, I", "it's it's"). A period, question
//!   mark or exclamation mark between them keeps both. The words are a, an, the, and, or, but, I,
//!   my, our, your, their, we, they, he, she, it's, that's, there's, what's, let's, I'm, we're,
//!   they're, you're, he's, she's, I've, we've, they've, you've, I'll, we'll, they'll, you'll and
//!   it'll.
//!
//! A word with a quote mark of its own stays, as a mention ("I often go “Um, so”"), and so do
//! both copies when a quote mark or bracket comes between them. Every other word stays as
//! written, so "uh-huh", "oh", "very very", "had had", "it it", repeats of several words, false
//! starts and words such as "like" or "you know" are left alone.
//!
//! Punctuation on a removed word goes with it, except a sentence end, which moves to the word
//! before ("We go there um. And then" becomes "We go there. And then"). When the removed word
//! had a capital at the start of a sentence, the next word takes it if it's in lower case and
//! isn't a special word ("Um, so I think" becomes "So I think").

use std::ops::Range;

use crate::tokenizer::Tokenizer;

/// The fillers of one language that a transcript can leave out, as the [module](self)
/// describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fillers {
    language: Language,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Language {
    English,
}

/// Words that can't end a clause, so two copies side by side are always a stutter.
const REPEATED: [&str; 35] = [
    "a", "an", "the", "and", "or", "but", "i", "my", "our", "your", "their", "we", "they", "he",
    "she", "it's", "that's", "there's", "what's", "let's", "i'm", "we're", "they're", "you're",
    "he's", "she's", "i've", "we've", "they've", "you've", "i'll", "we'll", "they'll", "you'll",
    "it'll",
];

const SENTENCE_END: [char; 3] = ['.', '?', '!'];

/// Quote marks. A word with one of its own is quoted, a mention rather than a filler, and one
/// between two copies keeps both.
const QUOTES: [char; 14] = [
    '"', '\'', '“', '”', '„', '‟', '‘', '’', '‚', '‛', '«', '»', '‹', '›',
];

/// Brackets, which keep both copies when they come between them.
const BRACKETS: [char; 6] = ['(', ')', '[', ']', '{', '}'];

/// What happens to a word.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Edit {
    Keep,
    Delete,
    /// Kept, written as this.
    Change(String),
    /// Deleted, leaving this sentence end after the text before it.
    Leave(char),
}

impl Fillers {
    /// The fillers of a language given as an ISO 639-1 code, with or without a region (`en`,
    /// `en-GB`), or none if Telmo has no rules for it.
    pub fn for_language(code: &str) -> Option<Fillers> {
        let language = code.split(['-', '_']).next()?.to_ascii_lowercase();
        (language == "en").then_some(Fillers {
            language: Language::English,
        })
    }

    /// `text` without these fillers, its words separated by single spaces.
    pub fn remove(&self, text: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let edits = self.plan(None, &words, &|_, _| false);
        let mut out = String::new();
        for (word, edit) in words.iter().zip(&edits) {
            let word = match edit {
                Edit::Keep => *word,
                Edit::Change(changed) => changed,
                Edit::Delete => continue,
                Edit::Leave(end) => {
                    out.push(*end);
                    continue;
                }
            };
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(word);
        }
        out
    }

    /// What happens to each of `words`, which follow the word `before` of a text that can't
    /// change, if any. `special(i, word)` says whether word `i`, given in lower case, keeps its
    /// spelling, which keeps it from being taken for a hesitation or given a capital.
    pub(crate) fn plan(
        &self,
        before: Option<&str>,
        words: &[&str],
        special: &dyn Fn(usize, &str) -> bool,
    ) -> Vec<Edit> {
        let Language::English = self.language;
        let parts: Vec<(&str, &str, &str)> = words.iter().map(|w| split(w)).collect();
        let before = before.map(split);
        let mut deleted = vec![false; words.len()];
        for (i, &(lead, core, trail)) in parts.iter().enumerate() {
            let previous = match i {
                0 => before.map(|b| b.1),
                _ => Some(parts[i - 1].1),
            };
            deleted[i] = !core.is_empty()
                && !quoted(lead, trail)
                && is_hesitation(core, previous)
                && !special(i, &core.to_lowercase());
        }
        // The first of two copies goes, or the second when the first can't change. Each copy
        // is held with its punctuation after it and whether it's quoted.
        let mut previous = before.map(|(lead, core, trail)| {
            (
                None,
                core.to_lowercase(),
                trail.to_string(),
                quoted(lead, trail),
            )
        });
        for (i, &(lead, core, trail)) in parts.iter().enumerate() {
            if deleted[i] {
                continue;
            }
            let core = core.to_lowercase();
            if let Some((copy, earlier, between, earlier_quoted)) = &mut previous
                && !earlier.is_empty()
                && *earlier == core
                && REPEATED.contains(&core.as_str())
                && !format!("{between}{lead}").contains(|c| {
                    SENTENCE_END.contains(&c) || QUOTES.contains(&c) || BRACKETS.contains(&c)
                })
            {
                match copy {
                    Some(copy) if !*earlier_quoted => deleted[*copy] = true,
                    None if !quoted(lead, trail) => {
                        deleted[i] = true;
                        between.push_str(trail);
                        continue;
                    }
                    _ => {}
                }
            }
            previous = Some((Some(i), core, trail.to_string(), quoted(lead, trail)));
        }
        self.tidy(before, &parts, &deleted, special)
    }

    /// The edits that delete `deleted`, with punctuation and capitals as the module describes.
    fn tidy(
        &self,
        before: Option<(&str, &str, &str)>,
        parts: &[(&str, &str, &str)],
        deleted: &[bool],
        special: &dyn Fn(usize, &str) -> bool,
    ) -> Vec<Edit> {
        let mut edits = vec![Edit::Keep; parts.len()];
        // The last word kept, as written now, and whether it's the word before, which can't
        // change.
        let mut last: Option<(Option<usize>, String, String, String)> =
            before.map(|(lead, core, trail)| (None, lead.into(), core.into(), trail.into()));
        let mut sentence_start = last.as_ref().is_none_or(|l| l.3.contains(SENTENCE_END));
        let mut capital = false;
        for (i, &(lead, core, trail)) in parts.iter().enumerate() {
            if deleted[i] {
                edits[i] = Edit::Delete;
                capital |= sentence_start && core.starts_with(char::is_uppercase);
                let end = trail.trim_end_matches([',', ';', ':']);
                if let Some(mark) = end.chars().last().filter(|_| end.contains(SENTENCE_END)) {
                    if let Some((index, lead, core, trail)) = &mut last
                        && !trail.contains(SENTENCE_END)
                    {
                        match index {
                            Some(j) => {
                                let kept = trail.trim_end_matches([',', ';', ':']);
                                *trail = format!("{kept}{mark}");
                                edits[*j] = Edit::Change(format!("{lead}{core}{trail}"));
                            }
                            // Settled text can't lose its comma, so the sentence end goes too.
                            None if trail.ends_with([',', ';', ':']) => {}
                            None => {
                                trail.push(mark);
                                edits[i] = Edit::Leave(mark);
                            }
                        }
                    }
                    sentence_start = true;
                }
                continue;
            }
            let mut core = core.to_string();
            if capital && is_plain_lower(&core) && !special(i, &core) {
                core = capitalized(&core);
                edits[i] = Edit::Change(format!("{lead}{core}{trail}"));
            }
            capital = false;
            sentence_start = trail.contains(SENTENCE_END);
            last = Some((Some(i), lead.into(), core, trail.into()));
        }
        edits
    }

    /// The edits of `tokens`, whose ids `id` gives, with the words that go left out and the
    /// words that change spelled anew by `respell` from the word's tokens and its new ids.
    /// `before` is the word of a text that can't change before them, and `special` says
    /// whether a word's tokens keep their spelling.
    pub(crate) fn rewrite<T>(
        &self,
        tokenizer: &Tokenizer,
        tokens: Vec<T>,
        id: impl Fn(&T) -> u32,
        before: Option<&str>,
        special: impl Fn(&[T], &str) -> bool,
        respell: impl Fn(&[T], Vec<u32>) -> Vec<T>,
    ) -> Vec<T>
    where
        T: Clone,
    {
        let ids: Vec<u32> = tokens.iter().map(&id).collect();
        let spans = words(tokenizer, &ids);
        let texts: Vec<String> = spans
            .iter()
            .map(|s| {
                tokenizer
                    .decode_raw(&ids[s.clone()])
                    .trim_start()
                    .to_string()
            })
            .collect();
        let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
        let edits = self.plan(before, &texts, &|i, word| {
            special(&tokens[spans[i].clone()], word)
        });
        if edits.iter().all(|e| *e == Edit::Keep) {
            return tokens;
        }
        let mut out = Vec::with_capacity(tokens.len());
        for (span, edit) in spans.into_iter().zip(edits) {
            let word = &tokens[span];
            match edit {
                Edit::Keep => out.extend_from_slice(word),
                Edit::Delete => {}
                Edit::Change(text) => match spelled(tokenizer, &text, true) {
                    Some(ids) => out.extend(respell(word, ids)),
                    None => out.extend_from_slice(word),
                },
                Edit::Leave(end) => {
                    if let Some(ids) = spelled(tokenizer, &end.to_string(), false) {
                        out.extend(respell(word, ids));
                    }
                }
            }
        }
        out
    }
}

/// The token ranges of the words of `ids`, each starting at a token with the word-start
/// marker, except a first word without it.
fn words(tokenizer: &Tokenizer, ids: &[u32]) -> Vec<Range<usize>> {
    let mut spans: Vec<Range<usize>> = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        match spans.last_mut() {
            Some(span) if !tokenizer.piece_bytes(id).starts_with(b" ") => span.end = i + 1,
            _ => spans.push(i..i + 1),
        }
    }
    spans
}

/// The ids of `text` as a word, with the word-start marker if `start`, or none if the
/// vocabulary can't spell it.
fn spelled(tokenizer: &Tokenizer, text: &str, start: bool) -> Option<Vec<u32>> {
    let ids = if start {
        tokenizer.encode(text)
    } else {
        tokenizer.encode_continuation(text)
    };
    let want = if start {
        format!(" {text}")
    } else {
        text.to_string()
    };
    (tokenizer.decode_raw(&ids) == want).then_some(ids)
}

/// A word's punctuation before, its letters and digits with what's between them, and its
/// punctuation after.
fn split(word: &str) -> (&str, &str, &str) {
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    let Some(start) = word.find(is_word) else {
        return (word, "", "");
    };
    let end = word
        .char_indices()
        .rfind(|&(_, c)| is_word(c))
        .map_or(word.len(), |(i, c)| i + c.len_utf8());
    (&word[..start], &word[start..end], &word[end..])
}

/// Whether `word`, a word's letters, is a hesitation sound in lower case or with one capital,
/// after the word `previous`.
fn is_hesitation(word: &str, previous: Option<&str>) -> bool {
    let lower = word.to_lowercase();
    if word != lower && word != capitalized(&lower) {
        return false;
    }
    // Runs of one letter, such as "u", "mm" for "umm".
    let mut runs: Vec<(char, usize)> = Vec::new();
    for c in lower.chars() {
        match runs.last_mut() {
            Some((last, n)) if *last == c => *n += 1,
            _ => runs.push((c, 1)),
        }
    }
    let letters: String = runs.iter().map(|r| r.0).collect();
    let sound = match letters.as_str() {
        "um" | "uh" | "uhm" | "erm" | "ah" | "hm" => true,
        "er" => runs == [('e', 1), ('r', 1)],
        "m" => runs[0].1 >= 2,
        _ => false,
    };
    sound && !(lower.starts_with("mm") && previous.is_some_and(is_number))
}

/// Whether a word with punctuation `lead` before it and `trail` after it is quoted.
fn quoted(lead: &str, trail: &str) -> bool {
    lead.contains(QUOTES) || trail.contains(QUOTES)
}

/// Whether `word` is a number such as 5, 2.5 or 1,000.
fn is_number(word: &str) -> bool {
    word.ends_with(|c: char| c.is_ascii_digit())
        && word
            .chars()
            .all(|c| c.is_ascii_digit() || c == '.' || c == ',')
}

/// Whether `word` is plain lower case, letters with at most one apostrophe inside ("don't").
fn is_plain_lower(word: &str) -> bool {
    let mut parts = word.split('\'');
    let lower = |p: &str| !p.is_empty() && p.chars().all(|c| c.is_ascii_lowercase());
    parts.next().is_some_and(lower) && parts.next().is_none_or(lower) && parts.next().is_none()
}

fn capitalized(word: &str) -> String {
    let mut chars = word.chars();
    chars
        .next()
        .map_or(String::new(), |c| c.to_uppercase().chain(chars).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn english() -> Fillers {
        Fillers::for_language("en").unwrap()
    }

    #[test]
    fn english_is_the_only_language() {
        assert!(Fillers::for_language("en-GB").is_some());
        assert!(Fillers::for_language("EN_us").is_some());
        assert!(Fillers::for_language("de").is_none());
        assert!(Fillers::for_language("").is_none());
    }

    #[test]
    fn hesitations_and_stutters_go_with_their_punctuation() {
        for (text, want) in [
            ("Um, so I think", "So I think"),
            ("I I think", "I think"),
            ("the the idea", "the idea"),
            ("I, I think the, uh, the idea", "I think the idea"),
            ("It's it's fine", "It's fine"),
            ("And and when", "And when"),
            ("a a a sophisticated chip", "a sophisticated chip"),
            ("but but uh but", "but"),
            ("Mm, that's fine.", "That's fine."),
            ("Ah, that's why", "That's why"),
            ("Hmm.", ""),
            ("Is it um?", "Is it?"),
            ("That's the plan. Um.", "That's the plan."),
            ("We go there um. And then", "We go there. And then"),
            ("We go there, um. And then", "We go there. And then"),
            (
                "Then uh we see you in about uh.",
                "Then we see you in about.",
            ),
            (
                "Here is what this thing should be this thing we are gonna um uh design uh is a new remote control",
                "Here is what this thing should be this thing we are gonna design is a new remote control",
            ),
            (
                "Yeah that's that's a good that's a good one",
                "Yeah that's a good that's a good one",
            ),
            (
                "No no no it won uh i don't think",
                "No no no it won i don't think",
            ),
            ("", ""),
        ] {
            assert_eq!(english().remove(text), want, "{text}");
        }
    }

    #[test]
    fn meant_words_stay() {
        for text in [
            "Thank you, you were great.",
            "It was 5 mm wide.",
            "It was 2.5 mm wide.",
            "I had had enough",
            "very very good",
            "Call UM tomorrow",
            "ER is busy",
            "I want to, to be honest",
            "go to to the shop",
            "if they chew it it breaks",
            "Mm'kay.",
            "I. I think so.",
            "uh-huh, oh, eh, huh, err",
            "Five minutes that was i guess that that's the old message.",
            "when when is",
            "Also the “the workers’ news goes to the agent” status message",
            "Say the (the old one) twice",
            "I often go “Um, so” stuff like that",
            "He said 'uh' twice",
        ] {
            assert_eq!(english().remove(text), text, "{text}");
        }
    }

    #[test]
    fn removing_words_leaves_no_stray_punctuation() {
        assert_eq!(
            english().remove("Thank you, um, you saved"),
            "Thank you, you saved"
        );
        // Every sequence of up to five of these words, whose punctuation sits on the words.
        let words = [
            "the", "the,", "The", "um", "um,", "um.", "Um,", "uh?", "so", "so,", "so.", "I,",
        ];
        let stray = |text: &str| {
            let joined: String = text.chars().filter(|c| *c != ' ').collect();
            [",,", ",.", ",?", "..", ".,", "?.", "?,"]
                .iter()
                .any(|s| joined.contains(s))
                || text.starts_with([',', '.', '?'])
        };
        let mut sequence = vec![0usize];
        while sequence.len() <= 5 {
            let text: Vec<&str> = sequence.iter().map(|&k| words[k]).collect();
            let text = text.join(" ");
            let out = english().remove(&text);
            assert!(!stray(&out), "{text} became {out}");
            // The next sequence, counting in base words.len().
            let mut k = 0;
            loop {
                if k == sequence.len() {
                    sequence.push(0);
                    break;
                }
                sequence[k] += 1;
                if sequence[k] < words.len() {
                    break;
                }
                sequence[k] = 0;
                k += 1;
            }
        }
    }

    #[test]
    fn a_special_word_keeps_its_spelling() {
        let words = ["Uh,", "kubectl", "get", "pods"];
        let special = |_: usize, word: &str| word == "kubectl";
        let edits = english().plan(None, &words, &special);
        assert_eq!(edits, [Edit::Delete, Edit::Keep, Edit::Keep, Edit::Keep]);
        let words = ["Hmm"];
        let special = |_: usize, word: &str| word == "hmm";
        assert_eq!(english().plan(None, &words, &special), [Edit::Keep]);
    }

    #[test]
    fn words_after_text_that_cant_change() {
        let plan = |before, words: &[&str]| english().plan(Some(before), words, &|_, _| false);
        // The earlier copy has settled, so the later one goes.
        assert_eq!(plan("the", &["the", "idea"]), [Edit::Delete, Edit::Keep]);
        assert_eq!(
            plan("the", &["the.", "The"]),
            [Edit::Leave('.'), Edit::Keep]
        );
        assert_eq!(plan("the.", &["The", "idea"]), [Edit::Keep, Edit::Keep]);
        // A sentence end moves after the settled word, unless its comma would have to go.
        assert_eq!(
            plan("there", &["um.", "And"]),
            [Edit::Leave('.'), Edit::Keep]
        );
        assert_eq!(plan("there,", &["um.", "And"]), [Edit::Delete, Edit::Keep]);
        assert_eq!(
            plan("there.", &["Um,", "so"]),
            [Edit::Delete, Edit::Change("So".into())]
        );
        // "mm" after a settled number.
        assert_eq!(plan("5", &["mm"]), [Edit::Keep]);
    }
}
