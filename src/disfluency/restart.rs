//! Whether the words after a pause start again what the words before it ended with, by a
//! network of hand-set units over the junction's words, as the cleanup's own are built. A
//! restart says the tail's last word, or its last two to four, again at the head's start, after
//! a hesitation, a marker or an editing word if any ("we should ship it, I guess, | well, I guess
//! we should"). A copy after a real sentence end is an echo instead, a backchannel or an answer
//! said twice ("Mm-hmm. | Mm-hmm."), a number or a question said back ("Why not? | Why not?"),
//! or a content word that opens the next sentence ("the budget. | The budget is"), and none of
//! those is a restart.

use std::sync::OnceLock;

use super::rules;
use super::units::{self, FAR, Input, Network};

/// Words that edit what was just said. One may stand between the two copies or close the tail.
const EDITING: &str = "well sorry actually rather like no or basically";
/// Editing words that also join a clause, so a copy may end with one ("this or | or that", "I
/// feel like | I feel like we").
const EDITING_OR_JOINING: [&str; 2] = ["or", "like"];
/// Two-word editing phrases, as (first, second).
const EDITING_PAIRS: [(&str, &str); 2] = [("i", "mean"), ("you", "know")];
/// Words after which a clause can't end, so one said again after a pause is a restart on its
/// own. Any other word said again alone opens a new sentence or echoes an answer.
const ONE_WORD: &str = "a an the my our your their to of for with into from than as and or because
if although whereas whether in i we they he she it's that's there's what's let's i'm we're they're
you're he's she's i've we've they've you've i'll we'll they'll you'll it'll i'd we'd you'd they'd
he'd she'd it'd";
/// The tail's last words the network reads, the longest copy.
const TAIL: usize = 4;
/// The most words that may stand between the copies.
const BETWEEN: usize = 3;
/// The head's words the network reads, enough for the words between and the longest copy.
pub(crate) const HEAD: usize = TAIL + BETWEEN;

/// A restart found at a pause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Restart {
    /// How many of the head's words say the tail's last words again, the words between the
    /// copies included.
    pub(crate) words: usize,
    /// Whether the copy in the head stays, with a comma before it, even where a tagger would cut
    /// it. An editing word that closed the tail ("I guess, well, | I guess") is settled, and
    /// would be left hanging. And after a "no" between the copies the speaker takes back more
    /// than the copy.
    pub(crate) keeps_copy: bool,
    /// Whether the head's first word is a hesitation, a marker, an editing word or a function
    /// word, which takes lower case once the sentence goes on. A name keeps its capital.
    pub(crate) lower_first: bool,
}

/// The detectors, each as its feature and the copy's length and the words between the copies.
type Detectors = Vec<(usize, usize, usize)>;

fn network() -> &'static (Network, Detectors) {
    static NETWORK: OnceLock<(Network, Detectors)> = OnceLock::new();
    NETWORK.get_or_init(|| {
        let mut s = Network::new(&[("edit", 0), ("one_word", 0), ("tail_last", 0), ("quest", 0)]);
        s.look_limit = FAR;
        let mut a = Vec::new();
        // A word that may stand between the copies.
        s.unit(
            &mut a,
            "skip",
            "",
            &[],
            &[(0, "hes"), (0, "marker"), (0, "edit")],
            &[],
            Some(0.0),
        );
        // Before a copy of one word only a hesitation or an editing word may stand, since a
        // marker and a lone function word more often open a new sentence ("program the |
        // Yeah, so the first").
        s.unit(
            &mut a,
            "skip_edit",
            "",
            &[],
            &[(0, "hes"), (0, "edit")],
            &[],
            Some(0.0),
        );
        // A word that is echoed rather than restarted, such as a backchannel, an answer, a
        // marker or a number.
        s.unit(
            &mut a,
            "echo",
            "",
            &[],
            &[(0, "not_clause"), (0, "numeric")],
            &[],
            Some(0.0),
        );
        // A word the clause goes on from, so a copy ending in it was cut short ("we can",
        // "I guess"), where a copy ending in a content word opens a new sentence ("the budget.
        // | The budget is").
        s.unit(
            &mut a,
            "open",
            "",
            &[],
            &[(0, "func"), (0, "clause_verb")],
            &[],
            Some(0.0),
        );
        s.close(a);

        // A copy of the tail's last `l` words after `g` words between, at the copy's first word.
        let mut b = Vec::new();
        let mut names = Vec::new();
        for l in 1..=TAIL {
            for g in 0..=BETWEEN {
                let same = format!("same{}", l + g);
                let mut need: Vec<Input> = (0..l).map(|j| (j as i32, same.as_str())).collect();
                need.push((l as i32 - 1, "tail_last"));
                let skip = if l == 1 { "skip_edit" } else { "skip" };
                need.extend((0..g).map(|k| ((l + k) as i32, skip)));
                need.push((l as i32 - 1, if l == 1 { "one_word" } else { "open" }));
                // No copy of a question or an echoed word, none that ends a sentence itself
                // ("a two. | A two.") or has a sentence end inside ("it's done. We | Done. We
                // can"), and no sentence end among the words between, where a new sentence
                // starts ("how do I, uh | No, no, no. I pressed").
                let mut block: Vec<Input> =
                    vec![(l as i32 - 1, "quest"), ((2 * l + g) as i32 - 1, "end")];
                block.extend((0..l).map(|j| (j as i32, "echo")));
                block.extend((0..g).map(|k| ((l + k) as i32, "end")));
                block.extend((0..l - 1).map(|j| (j as i32, "end")));
                block.extend((0..l - 1).map(|j| ((l + g + j) as i32, "end")));
                let name = format!("copy{l}_{g}");
                s.unit(&mut b, &name, "restart", &need, &[], &block, None);
                names.push((name, l, g));
            }
        }
        s.close(b);
        let detectors = (names.iter())
            .map(|(name, l, g)| (s.feature(name), *l, *g))
            .collect();
        (s, detectors)
    })
}

/// Whether `word`, in lower case without punctuation, is an editing word on its own.
fn is_editing(word: &str) -> bool {
    EDITING.split_whitespace().any(|w| w == word)
}

/// Whether `words[i]` is an editing word, or a word of a two-word editing phrase.
fn is_editing_at(words: &[&str], i: usize) -> bool {
    let norm = |k: usize| rules::norm(words[k]);
    let w = norm(i);
    is_editing(&w)
        || EDITING_PAIRS.iter().any(|&(first, second)| {
            (w == first && i + 1 < words.len() && norm(i + 1) == second)
                || (w == second && i > 0 && norm(i - 1) == first)
        })
}

/// Whether `words[i]` is a hesitation, a marker or an editing word.
fn is_skippable(words: &[&str], i: usize) -> bool {
    rules::is_hesitation(words[i])
        || units::has_class("marker", &rules::norm(words[i]))
        || is_editing_at(words, i)
}

/// The restart at a pause, if the first words after it, `head`, say the last words of the text
/// before it, `previous`, again. Hesitations and editing words at the text's end aren't its last
/// words, unless the copy ends with one that also joins a clause ("this or | or that").
pub(crate) fn restart(previous: &str, head: &[&str]) -> Option<Restart> {
    let mut tail: Vec<&str> = previous.split_whitespace().collect();
    let mut edited_tail = false;
    while let Some(&last) = tail.last() {
        let word = rules::norm(last);
        let before = tail.len().checked_sub(2).map(|i| rules::norm(tail[i]));
        let pair = EDITING_PAIRS.iter().find(|&&(_, second)| word == second);
        if word.is_empty() || rules::is_hesitation(last) {
            tail.pop();
            continue;
        }
        if EDITING_OR_JOINING.contains(&word.as_str())
            && let Some(found) = copy(&tail, head, edited_tail)
        {
            return Some(found);
        }
        if is_editing(&word) {
            edited_tail = true;
            tail.pop();
        } else if let Some(&(first, _)) = pair
            && before.as_deref() == Some(first)
        {
            edited_tail = true;
            tail.pop();
            tail.pop();
        } else {
            break;
        }
    }
    copy(&tail, head, edited_tail)
}

/// The restart whose copy ends with `tail`'s last word, if `head` says it again. `edited_tail`
/// is whether an editing word closed the text after `tail`.
fn copy(tail: &[&str], head: &[&str], edited_tail: bool) -> Option<Restart> {
    if tail.is_empty() || head.is_empty() {
        return None;
    }
    let tail = &tail[tail.len().saturating_sub(TAIL)..];
    let n = tail.len();
    let words: Vec<&str> = (tail.iter().chain(head.iter().take(HEAD)))
        .copied()
        .collect();
    let (network, detectors) = network();
    let mut z = units::base(network, &words);
    let (edit, one_word, tail_last, quest) = (
        network.feature("edit"),
        network.feature("one_word"),
        network.feature("tail_last"),
        network.feature("quest"),
    );
    for i in 0..words.len() {
        if i >= n && is_editing_at(&words, i) {
            z[i][edit] = 1.0;
        }
        if ONE_WORD
            .split_whitespace()
            .any(|w| w == rules::norm(words[i]))
        {
            z[i][one_word] = 1.0;
        }
    }
    z[n - 1][tail_last] = 1.0;
    // The text keeps the mark that closed it.
    if words[n - 1]
        .trim_end_matches(['"', '\'', ')'])
        .ends_with('?')
    {
        z[n - 1][quest] = 1.0;
    }
    let lower_first = is_skippable(&words, n)
        || units::has_class("func", &rules::norm(words[n]))
        || z[n][one_word] > 0.5;
    let z = network.run(z);
    // The longest copy, and of those the one with the fewest words between.
    let (l, g) = (detectors.iter())
        .filter(|&&(f, l, _)| l <= n && z[n - l][f] > 0.5)
        .max_by_key(|&&(_, l, g)| (l, BETWEEN - g))
        .map(|&(_, l, g)| (l, g))?;
    // After "no" the speaker takes back more than the copy ("how do I, uh | no, no, I pressed"),
    // so cutting the copy would join the words before it to the new ones.
    let rejected = head[..g].iter().any(|w| rules::norm(w) == "no");
    Some(Restart {
        words: l + g,
        keeps_copy: edited_tail || rejected,
        lower_first,
    })
}

/// `text`, an utterance's cleaned words, without the words at its start that are `restart`'s,
/// the restart's words as decoded, with any the cleanup already left out passed over. The
/// decode hears from a little before the utterance's audio, so its first words may run ahead of
/// the text's, and those are passed over too. None when the text doesn't start with the rest.
pub(crate) fn without_restart(text: &str, restart: &[&str]) -> Option<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let matched = (0..restart.len()).find_map(|start| {
        let mut i = 0;
        for j in start..restart.len() {
            if i < words.len() && rules::norm(words[i]) == rules::norm(restart[j]) {
                i += 1;
            } else if !is_skippable(restart, j) {
                return None;
            }
        }
        (i > 0).then_some(i)
    })?;
    let rest = words[matched..].join(" ");
    Some(if rest.is_empty() {
        rest
    } else {
        format!(" {rest}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(previous: &str, head: &str) -> Option<usize> {
        let head: Vec<&str> = head.split_whitespace().collect();
        restart(previous, &head).map(|r| r.words)
    }

    fn found(previous: &str, head: &str) -> Option<Restart> {
        let head: Vec<&str> = head.split_whitespace().collect();
        restart(previous, &head)
    }

    #[test]
    fn a_copy_of_the_last_words_after_editing_words_is_a_restart() {
        // The copy and the words between it and the pause are the restart's.
        assert_eq!(
            at("we should ship it, I guess", "well, I guess we should"),
            Some(3)
        );
        assert_eq!(
            at("we should ship it, I guess", "I guess we should"),
            Some(2)
        );
        assert_eq!(
            at("so the plan is, we can", "um, so we can mix them"),
            Some(4)
        );
        assert_eq!(at("and we want", "we want large buttons"), Some(2));
        assert_eq!(at("put my ideas in, uh", "in here and then"), Some(1));
        assert_eq!(at("up to the", "the, um, the retailer"), Some(1));
        assert_eq!(
            at("satisfied we are with, um", "I mean, with these four"),
            Some(3)
        );
        assert_eq!(at("they're, uh", "they're not as prominent"), Some(1));
        assert_eq!(at("how do I, uh", "No, no, no. I pressed the"), None);
        assert_eq!(at("how do I, uh", "No, no, no, I pressed the"), Some(4));
        // A marker before a copy of one word opens a new sentence, before a longer copy not.
        assert_eq!(at("you can program the", "Yeah, so the first thing"), None);
        assert_eq!(at("because we can", "Yeah, so we can mix them"), Some(4));
        // A copy that ends in a content word opens a new sentence, alone or as a phrase.
        assert_eq!(
            at("remote control, like the budget", "The budget is simple"),
            None
        );
        assert_eq!(at("we have red", "Red and yellow"), None);
        assert_eq!(at("fix the login", "login page"), None);
        assert_eq!(at("I'll do it", "It should work"), None);
        // Echoes of backchannels, answers, markers and numbers.
        assert_eq!(at("okay", "Okay, so the next slide"), None);
        assert_eq!(at("yeah, true, true", "True. Yeah. We have"), None);
        assert_eq!(
            at("sixteen plus six, twenty two", "Twenty two, yeah?"),
            None
        );
        assert_eq!(at("I think so", "So we keep it"), None);
        // A copy that is a sentence of its own, or answers a question.
        assert_eq!(at("difficult to lose it, a two", "A two. Okay."), None);
        assert_eq!(
            at(
                "repetitive strain injury",
                "Repetitive strain injury. So if"
            ),
            None
        );
        assert_eq!(
            at("options should be, why not?", "Why not? Let's hope"),
            None
        );
        // A sentence end inside either copy.
        assert_eq!(at("it's done. We", "done, we can go"), None);
        assert_eq!(at("it's done, we", "Done. We can go"), None);
        // Nothing to copy.
        assert_eq!(at("", "we can"), None);
        assert_eq!(at("we can", ""), None);
        assert_eq!(at("uh, um", "um, uh"), None);
    }

    #[test]
    fn an_editing_word_that_closed_the_tail_is_passed_over_and_reported() {
        let head = "I guess we should";
        let restart = |words, keeps_copy| {
            Some(Restart {
                words,
                keeps_copy,
                lower_first: true,
            })
        };
        assert_eq!(
            found("we should ship it, I guess, well", head),
            restart(2, true)
        );
        assert_eq!(
            found("we should ship it, I guess, um", head),
            restart(2, false)
        );
        assert_eq!(
            found("we should ship it, I guess, or", head),
            restart(2, true)
        );
        // A marker closing the tail is one of its words.
        assert_eq!(found("I guess, so", head), None);
        // "or" and "like" also join a clause, so a copy may end with one.
        assert_eq!(
            found("it works with or", "with or without it"),
            restart(2, false)
        );
        assert_eq!(
            found("I feel like", "I feel like we should"),
            restart(3, false)
        );
    }

    #[test]
    fn a_no_between_the_copies_keeps_the_copy() {
        // "how do I" was taken back, so a cut would leave "how do I pressed the".
        let r = found("how do I, uh", "No, no, no, I pressed the");
        assert_eq!(r.map(|r| (r.words, r.keeps_copy)), Some((4, true)));
        let r = found("leave it on the, um", "no, the other shelf");
        assert_eq!(r.map(|r| (r.words, r.keeps_copy)), Some((2, true)));
        let r = found("leave it on the, um", "well, the other shelf");
        assert_eq!(r.map(|r| (r.words, r.keeps_copy)), Some((2, false)));
    }

    #[test]
    fn only_a_function_word_or_an_editing_word_first_takes_lower_case() {
        let first = |previous, head| found(previous, head).map(|r| r.lower_first);
        assert_eq!(
            first("we should ship it, I guess", "Well, I guess we"),
            Some(true)
        );
        assert_eq!(first("up to the", "The, um, the retailer"), Some(true));
        assert_eq!(first("so we can", "Um, we can go"), Some(true));
        // A name or a day that starts the copy keeps its capital.
        assert_eq!(
            first("let's meet on Monday at", "Monday at noon."),
            Some(false)
        );
        assert_eq!(
            first("we asked Priya and", "Priya and Tom said"),
            Some(false)
        );
    }

    #[test]
    fn the_restart_goes_from_the_cleaned_text_when_its_words_are_there() {
        fn heard(s: &str) -> Vec<&str> {
            s.split_whitespace().collect()
        }
        // "well," already went with the fillers, and the copy follows.
        assert_eq!(
            without_restart(" I guess we should ship it.", &heard("well, I guess")),
            Some(" we should ship it.".into())
        );
        assert_eq!(
            without_restart(" Well, I guess we should.", &heard("well, I guess")),
            Some(" we should.".into())
        );
        assert_eq!(
            without_restart(" I guess.", &heard("I guess")),
            Some(String::new())
        );
        // The decode heard a word before the text's first.
        assert_eq!(
            without_restart(" He recalled the words", &heard("and he recalled the")),
            Some(" words".into())
        );
        // Other words first, or none of the copy's, leave the text as it is.
        assert_eq!(
            without_restart(" we should ship it.", &heard("well, I guess")),
            None
        );
        assert_eq!(
            without_restart(" Well, we should.", &heard("well, I guess")),
            None
        );
        assert_eq!(without_restart(" I should.", &heard("well, I guess")), None);
    }
}
