//! Capitals and punctuation after the cleanup, by a network of hand-set units over each word's
//! flags, as the cleanup's own are built. Each word stays one word, and only a letter's case
//! and the marks at its end change.
//!
//! - The pronoun "i" and its contractions take a capital.
//! - A period goes where a speaker only paused, after a word that leaves its sentence unfinished
//!   ("I'll take the. Red one."), before a one-word sentence of a content word ("a patch.
//!   Release."), before "than", and before a short fragment with no subject or verb ("We should
//!   check. The report."). "so" keeps its period where it ends a sentence of its own ("I think
//!   so."). The word after a period that goes takes lower case where the sentence goes on.
//! - The wider setting also joins a one-word sentence's own period and a fragment's, a period
//!   after an auxiliary with no subject before it ("My brother is. Learning"), and one before a
//!   preposition after four words or more ("the library stays open. Until nine.").
//! - A comma goes from a word that can't end a clause, where a hesitation that went left it ("the,
//!   uh, little").

use std::sync::OnceLock;

use super::rules;
use super::units::{self, Input, Network};

/// Words after which a sentence isn't finished.
const DANGLING: &str = "a an the and or but because if although whereas whether my our your their
it's that's there's what's let's i'm we're they're you're he's she's i've we've they've you've
i'll we'll they'll you'll it'll i'd we'd you'd they'd he'd she'd it'd i we they he she can could
would will should might may must shall be just like very so to of for with into from than as";
/// Words after which "so" is a sentence of its own ("I think so", "or so", "if so").
const SO_ENDERS: &str = "think thought hope hoped guess guessed suppose supposed believe believed
say said assume imagine expect reckon afraid or not if even do did done doing more less";
/// Words after which a clause can't end, so a comma there is a pause.
const NEVER_END: &str = "a an the my our your their to of for with into from than as and or because
if although whereas whether it's that's there's what's let's i'm we're they're you're he's she's
i've we've they've you've i'll we'll they'll you'll it'll i'd we'd you'd they'd he'd she'd it'd
very";
/// Words after which a sentence that went on starts in lower case.
const LOWER_AFTER: &str = "the to of for with into from than as very my our your their because if
whether although whereas";
/// Words that make a sentence on their own, besides answers, markers and numbers.
const MORE_ANSWERS: &str =
    "'kay kay nope thanks sorry please exactly indeed bye hello hi wow ooh huh nothing";
const PREPOSITIONS: &str = "at from to of on in for with by until than within beside instead along
near about into through during after before over under between across behind without";
/// Auxiliaries, copulas and modals, after which a sentence without a subject isn't whole.
const AUX: &str = "am is are was were be been being has have had do does did can could will would
should might may must shall";
/// Verbs that need an object.
const VERB_OBJ: &str = "put send bring review check open find take carry protect attach place keep
keeps include includes require requires want wants need needs see saw leave left replace replaced
plant planted complete completed delete deleted compare choose discuss show thank make made give gave
get got tell told ask asked use used buy bought order sent brought reviewed checked opened found took
carried attached placed kept included wanted needed add added read write wrote build built move moved
reduce submit submitted finish finished postpone hire organize save revealed recommended drew caught
prefer prefers expect expects contain contains wrote donated offers offer sends press restart turn
turned cram cover covers hold holds change changed design designed press pressed reach reached";
/// Subjects and words that open a clause.
const CLAUSE_SUBJECT: &str = "i you we they he she it this that there what who which these those";
/// More words that show a clause, besides the auxiliaries, verbs and subjects above.
const MORE_CLAUSE_WORDS: &str = "think know go goes going come comes came say says said mean means
meant like want wants need needs look looks looked seem seems work works worked try tried happen
happens happened start starts started stop stops stopped talk talked speak spoke feel felt believe
guess suppose hope agree remember forget forgot wonder wondered decide decided decided plan plans
planned learn learned run runs ran play played live lives lived stay stays stayed wait waits waited
arrive arrives arrived begin begins began end ends ended leave leaves win won lose lost i'm it's
that's we're you're they're i'll we'll i've we've i'd there's he's she's isn't aren't don't doesn't
didn't can't won't wouldn't couldn't shouldn't haven't hasn't";
/// Words ending in "ing" that aren't progressives.
const NOT_PROGRESSIVE: &str = "thing something anything nothing everything morning evening during";

const FLAGS: [&str; 24] = [
    "end",
    "comma",
    "cap",
    "dangling",
    "so_ender",
    "never_end",
    "lower_after",
    "answer",
    "numeric",
    "first_person",
    "content",
    "start",
    "last",
    "allcaps",
    "lone_period",
    "w_so",
    "prep",
    "w_than",
    "aux",
    "verb_obj",
    "ing",
    "clause_subject",
    "clause_word",
    "plain",
];

/// The units that drop a period, each at the word that carries it.
const DROPS: [&str; 10] = [
    "dangle",
    "lone_before",
    "lone_own",
    "before_prep",
    "than_next",
    "aux_end",
    "verb_end",
    "ing_end",
    "frag_before",
    "frag_own",
];
/// The drops on by default, which join at least three pauses for each real sentence end they
/// join.
const DEFAULT: [&str; 4] = ["dangle", "lone_before", "than_next", "frag_before"];
/// The drops on in the wider setting.
const WIDE: [&str; 8] = [
    "dangle",
    "lone_before",
    "than_next",
    "frag_before",
    "before_prep",
    "aux_end",
    "frag_own",
    "lone_own",
];
/// The drops after which the next word goes on in lower case.
const JOINS: [&str; 9] = [
    "lone_before",
    "lone_own",
    "before_prep",
    "than_next",
    "aux_end",
    "verb_end",
    "ing_end",
    "frag_before",
    "frag_own",
];

fn has(list: &str, word: &str) -> bool {
    list.split_whitespace().any(|w| w == word)
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `word` with the pronoun "i" capitalized, if it's the pronoun or one of its contractions with
/// only punctuation around it ("i", "(i),", "i'm", "i’ll").
fn first_person(word: &str) -> Option<String> {
    let start = word.find(is_word_char).unwrap_or(word.len());
    let rest = word[start..].strip_prefix('i')?;
    let ending = ["m", "d", "ll", "ve"].into_iter().find_map(|e| {
        let after = rest.strip_prefix(['\'', '’'])?.strip_prefix(e)?;
        (!after.contains(is_word_char)).then_some(after)
    });
    if ending.is_none() && rest.contains(is_word_char) {
        return None;
    }
    Some(format!("{}I{rest}", &word[..start]))
}

/// The flags of each word.
fn flags(network: &Network, words: &[&str]) -> Vec<Vec<f32>> {
    let n = words.len();
    let width = network.layers.iter().map(Vec::len).sum::<usize>();
    let mut z = vec![vec![0f32; network.look.len() - width]; n];
    for (i, &w) in words.iter().enumerate() {
        let a = rules::norm(w);
        let length = a.chars().count();
        let core: String = (w.replace('’', "'").chars())
            .filter(|&c| is_word_char(c) || c == '\'')
            .collect();
        let tail = w.trim_end_matches(['"', '\'', ')']);
        let allcaps = core.chars().count() > 1 && rules::is_upper(&core);
        let numeric = a.chars().any(|c| c.is_ascii_digit()) || has(units::NUMBER_WORDS, &a);
        let answer = units::has_class("not_clause", &a)
            || has(units::NUMBER_WORDS, &a)
            || has(MORE_ANSWERS, &a);
        let dangling = has(DANGLING, &a);
        let progressive = a.ends_with("ing") && length > 5;
        let clause_word = has(AUX, &a)
            || has(VERB_OBJ, &a)
            || has(CLAUSE_SUBJECT, &a)
            || has(MORE_CLAUSE_WORDS, &a)
            || units::has_class("clause_verb", &a)
            || progressive;
        let on = [
            tail.ends_with(['.', '?', '!']),
            tail.ends_with([',', ';', ':']),
            core.chars().next().is_some_and(char::is_uppercase) && !allcaps,
            dangling,
            has(SO_ENDERS, &a),
            has(NEVER_END, &a),
            has(LOWER_AFTER, &a),
            answer,
            numeric,
            first_person(w).is_some(),
            !a.is_empty()
                && !answer
                && !units::has_class("func", &a)
                && !dangling
                && !numeric
                && a != "i"
                && length > 1,
            i == 0,
            i + 1 == n,
            allcaps,
            w.ends_with('.') && !w.ends_with("..") && w != "A.",
            a == "so",
            has(PREPOSITIONS, &a),
            a == "than",
            has(AUX, &a),
            has(VERB_OBJ, &a),
            progressive && !has(NOT_PROGRESSIVE, &a),
            has(CLAUSE_SUBJECT, &a),
            clause_word,
            length > 1 && !clause_word && !answer,
        ];
        for (name, on) in FLAGS.iter().zip(on) {
            if on {
                z[i][network.feature(name)] = 1.0;
            }
        }
    }
    z
}

/// The network with the drops `enabled` on, and the features of its four decisions.
fn build(enabled: &[&str]) -> (Network, [usize; 4]) {
    let base: Vec<(&str, i32)> = FLAGS.iter().map(|&f| (f, 0)).collect();
    let mut s = Network::with_base(&base);
    let on: Vec<&str> = DROPS.into_iter().filter(|u| enabled.contains(u)).collect();
    let unit = |u: &str| format!("u_{u}");

    let mut a = Vec::new();
    // "so" that ends a sentence of its own.
    let so: [Input; 3] = [(0, "lone_period"), (0, "w_so"), (-1, "so_ender")];
    s.unit(&mut a, "dangle_so", "", &so, &[], &[], None);
    // A content word that makes a sentence on its own.
    let block: [Input; 2] = [(0, "last"), (0, "allcaps")];
    let lone: [Input; 3] = [(0, "end"), (-1, "end"), (0, "content")];
    s.unit(&mut a, "lone_word", "", &lone, &[], &block, None);
    // Fragments of one to three plain words up to a period, after one.
    let need: [Input; 3] = [(-1, "end"), (0, "end"), (0, "plain")];
    s.unit(&mut a, "frag1", "", &need, &[], &block, None);
    let need: [Input; 4] = [(-2, "end"), (-1, "plain"), (0, "end"), (0, "plain")];
    let block2: [Input; 4] = [(-1, "end"), (0, "last"), (-1, "allcaps"), (0, "allcaps")];
    s.unit(&mut a, "frag2", "", &need, &[], &block2, None);
    let need: [Input; 5] = [
        (-3, "end"),
        (-2, "plain"),
        (-1, "plain"),
        (0, "end"),
        (0, "plain"),
    ];
    let block3: [Input; 6] = [
        (-2, "end"),
        (-1, "end"),
        (0, "last"),
        (-2, "allcaps"),
        (-1, "allcaps"),
        (0, "allcaps"),
    ];
    s.unit(&mut a, "frag3", "", &need, &[], &block3, None);
    s.close(a);

    // The drops, each at the word that carries the period.
    let mut b = Vec::new();
    let period: Input = (0, "lone_period");
    s.unit(
        &mut b,
        "u_dangle",
        "",
        &[period, (0, "dangling")],
        &[],
        &[(0, "dangle_so")],
        None,
    );
    s.unit(
        &mut b,
        "u_lone_before",
        "",
        &[period, (1, "lone_word")],
        &[],
        &[],
        None,
    );
    s.unit(
        &mut b,
        "u_lone_own",
        "",
        &[(0, "lone_word"), period],
        &[],
        &[],
        None,
    );
    s.unit(
        &mut b,
        "u_than_next",
        "",
        &[period, (1, "w_than")],
        &[],
        &[(0, "answer"), (0, "last"), (0, "dangling")],
        None,
    );
    s.unit(
        &mut b,
        "u_aux_end",
        "",
        &[period, (0, "aux")],
        &[],
        &[
            (-1, "clause_subject"),
            (0, "last"),
            (0, "start"),
            (0, "dangling"),
        ],
        None,
    );
    s.unit(
        &mut b,
        "u_verb_end",
        "",
        &[period, (0, "verb_obj")],
        &[],
        &[(0, "last"), (0, "start")],
        None,
    );
    s.unit(
        &mut b,
        "u_ing_end",
        "",
        &[period, (0, "ing"), (-1, "aux")],
        &[],
        &[(0, "last")],
        None,
    );
    s.unit(
        &mut b,
        "u_frag_before",
        "",
        &[period],
        &[(1, "frag1"), (2, "frag2"), (3, "frag3")],
        &[(0, "answer"), (0, "last"), (0, "dangling")],
        Some(-1.0),
    );
    s.unit(
        &mut b,
        "u_frag_own",
        "",
        &[period],
        &[(0, "frag1"), (0, "frag2"), (0, "frag3")],
        &[(0, "last")],
        Some(-1.0),
    );
    s.close(b);

    // A sentence end that no drop takes, which a sentence before a preposition counts from.
    let mut b2 = Vec::new();
    let others: Vec<String> = (on.iter())
        .filter(|&&u| u != "before_prep")
        .map(|u| unit(u))
        .collect();
    let block: Vec<Input> = others.iter().map(|u| (0, u.as_str())).collect();
    s.unit(&mut b2, "kept_end", "", &[(0, "end")], &[], &block, None);
    s.close(b2);
    let mut b3 = Vec::new();
    let block: [Input; 10] = [
        (-1, "kept_end"),
        (-2, "kept_end"),
        (-3, "kept_end"),
        (0, "start"),
        (-1, "start"),
        (-2, "start"),
        (0, "answer"),
        (0, "dangling"),
        (0, "last"),
        (1, "last"),
    ];
    s.unit(
        &mut b3,
        "u_before_prep",
        "",
        &[period, (1, "prep")],
        &[],
        &block,
        None,
    );
    s.close(b3);

    let mut c = Vec::new();
    let drops: Vec<String> = on.iter().map(|u| unit(u)).collect();
    let any: Vec<Input> = drops.iter().map(|u| (0, u.as_str())).collect();
    s.unit(&mut c, "drop_end", "", &[], &any, &[], Some(0.0));
    let joins: Vec<Input> = (drops.iter().zip(&on))
        .filter(|(_, u)| JOINS.contains(u))
        .map(|(d, _)| (0, d.as_str()))
        .collect();
    s.unit(&mut c, "joined", "", &[], &joins, &[], Some(0.0));
    let after: [Input; 2] = [(0, "u_dangle"), (0, "lower_after")];
    s.unit(&mut c, "lower_after_drop", "", &after, &[], &[], None);
    let comma: [Input; 2] = [(0, "comma"), (0, "never_end")];
    s.unit(&mut c, "drop_comma", "", &comma, &[], &[], None);
    s.unit(
        &mut c,
        "upper_i",
        "",
        &[(0, "first_person")],
        &[],
        &[],
        None,
    );
    s.close(c);

    // The word after a join, and a one-word sentence's own word, take lower case.
    let mut d = Vec::new();
    let any: [Input; 3] = [(-1, "joined"), (-1, "lower_after_drop"), (0, "u_lone_own")];
    let block: [Input; 2] = [(0, "first_person"), (0, "allcaps")];
    s.unit(&mut d, "lower", "", &[(0, "cap")], &any, &block, Some(-1.0));
    s.close(d);
    let outputs = ["drop_end", "lower", "drop_comma", "upper_i"].map(|n| s.feature(n));
    (s, outputs)
}

/// The network of the default setting, or of the wider one.
fn network(wide: bool) -> &'static (Network, [usize; 4]) {
    static DEFAULT_NETWORK: OnceLock<(Network, [usize; 4])> = OnceLock::new();
    static WIDE_NETWORK: OnceLock<(Network, [usize; 4])> = OnceLock::new();
    if wide {
        WIDE_NETWORK.get_or_init(|| build(&WIDE))
    } else {
        DEFAULT_NETWORK.get_or_init(|| build(&DEFAULT))
    }
}

/// `word` with what each of `drop_end`, `lower`, `drop_comma` and `upper_i` asks done.
fn apply(word: &str, [drop_end, lower, drop_comma, upper_i]: [bool; 4]) -> String {
    let mut word = word.to_string();
    if upper_i && let Some(capital) = first_person(&word) {
        word = capital;
    }
    if lower && let Some((at, c)) = word.char_indices().find(|&(_, c)| is_word_char(c)) {
        word = format!(
            "{}{}{}",
            &word[..at],
            c.to_lowercase(),
            &word[at + c.len_utf8()..]
        );
    }
    // The run of marks at the end goes, keeping any quote marks or bracket after it.
    let drop = |word: &str, marks: [char; 3]| -> Option<String> {
        let inner = word.trim_end_matches(['"', '\'', ')']);
        let kept = inner.trim_end_matches(marks);
        (kept.len() < inner.len()).then(|| format!("{kept}{}", &word[inner.len()..]))
    };
    if drop_end && let Some(dropped) = drop(&word, ['.', '?', '!']) {
        word = dropped;
    }
    if drop_comma && let Some(dropped) = drop(&word, [',', ';', ':']) {
        word = dropped;
    }
    word
}

/// `words` with their capitals and punctuation fixed, one word for each, in the wider setting
/// if `wide`.
pub(crate) fn punctuate(words: &[&str], wide: bool) -> Vec<String> {
    let (network, outputs) = network(wide);
    let z = network.run(flags(network, words));
    (words.iter().zip(&z))
        .map(|(word, row)| apply(word, outputs.map(|f| row[f] > 0.5)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capitals_and_periods_follow_the_sentence() {
        let default = [
            (
                "we're shipping a patch. Release. That is due on Friday.",
                "we're shipping a patch release. That is due on Friday.",
            ),
            (
                "We could. test. The new branch",
                "We could test. The new branch",
            ),
            (
                "We should check. The report. Next week.",
                "We should check the report. Next week.",
            ),
            (
                "The tests run. Much faster. Than we hoped.",
                "The tests run much faster than we hoped.",
            ),
            ("the, little", "the little"),
            ("Can you, is it ready?", "Can you, is it ready?"),
            ("i think i'm ready.", "I think I'm ready."),
            ("an. the. And. OR. but.", "an the and OR but"),
            ("I think so. Hope so.", "I think so. Hope so."),
            ("you can go, so.", "you can go, so"),
            ("Yes. No. Okay. Three.", "Yes. No. Okay. Three."),
            ("It works. Great. Thanks.", "It works. Great. Thanks."),
            ("the plan to. Finish it.", "the plan to finish it."),
            ("we need, a way", "we need, a way"),
            ("and, I think", "and I think"),
            ("I'll take the. Red one.", "I'll take the red one."),
            (
                "It works. On Friday we meet.",
                "It works. On Friday we meet.",
            ),
            ("(i), 'i' \"i'm\" i’ll i’ve.", "(I), 'I' \"I'm\" I’ll I’ve."),
            ("", ""),
        ];
        let wide = [
            (
                "we're shipping a patch. Release. That is due on Friday.",
                "we're shipping a patch release that is due on Friday.",
            ),
            (
                "We could. test. The new branch",
                "We could test the new branch",
            ),
            (
                "the library stays open. Until nine on Mondays.",
                "the library stays open until nine on Mondays.",
            ),
            (
                "We should check. The report. Next week.",
                "We should check the report next week.",
            ),
            (
                "My brother is. Learning how. To fix bikes.",
                "My brother is learning how to fix bikes.",
            ),
            ("Yes. The red one. I think.", "Yes. The red one. I think."),
        ];
        for (cases, is_wide) in [(&default[..], false), (&wide[..], true)] {
            for (text, want) in cases {
                let words: Vec<&str> = text.split_whitespace().collect();
                assert_eq!(punctuate(&words, is_wide).join(" "), *want, "{text:?}");
            }
        }
    }

    #[test]
    fn the_networks_look_three_words_ahead_at_most() {
        for wide in [false, true] {
            let (network, _) = network(wide);
            assert!(network.layers.iter().flatten().all(|u| u.look <= 3));
        }
    }
}
