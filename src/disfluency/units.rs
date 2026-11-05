//! Networks of hand-set units over each word's spelling, its word classes and whether it
//! matches the next words. One, the hand-written cleanup, deletes disfluencies. The other, the
//! guards, vetoes deletions that change what dictation says.
//!
//! Every input is 0 or 1. A unit adds 1 for each input it needs or takes any of, subtracts 3 for
//! each that blocks it, adds its bias (1 less the count of needs by default) and clips the sum to
//! between 0 and 1, so it computes an AND, an OR or either with blockers. Units read the words up
//! to three places either side, and each layer's outputs join the features the next layer
//! reads. Nothing looks more than three words ahead, so a stream that waits for three words
//! decides as the whole utterance does.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use super::rules;

/// The longest word whose proper prefixes are matched.
const MAXP: usize = 10;
/// How far ahead any unit may look.
const LOOK: i32 = 3;

const DETERMINERS: &str = "a an the my your our their his her its";
const DEMONSTRATIVES: &str = "this that these those";
const PREPOSITIONS: &str =
    "to of in on at for from with into onto over under about through by without between";
const SUBJECTS: &str = "i you we they he she it";
const AUX: &str =
    "am is are was were has have had do does did can could may might must should would will";
const CONTRACTIONS: &str = "it's that's there's what's let's i'm we're they're you're he's she's
i've we've they've you've i'll we'll they'll you'll it'll i'd we'd you'd they'd he'll she'd she'll";
const MARKERS: &str = "so well yeah okay ok oh right anyway alright";
const BACKCHANNEL: &str = "yeah yes yep yup okay ok right alright sure mm-hmm uh-huh";
const ANSWERS: &str = "yes no yep yup nope sure exactly thanks thank sorry please fine cool great
good true maybe perhaps absolutely definitely probably mm-hmm uh-huh huh";
/// Words whose adjacent double is said on purpose ("no no", "very very", "sixteen sixteen").
const MEANT: &str = "yeah yes no okay ok oh very really bye ha blah etcetera yep now far long never
many had that right sure hey knock mm-hmm uh-huh huh ooh true bling go zero one two three four five
six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen
twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion please more
much again round bit little";
const MORE_FUNCTION: &str =
    "and or but if what which how when where why who because 'cause not just
some any than then there as so too also";
const SAID_TWICE: &str = "and or but please";
const SO_ENDERS: &str =
    "think thought hope hoped guess guessed suppose supposed believe believed say
said assume imagine expect reckon afraid or not if even do did done doing more less";
const YOU_KNOW_LEAD: &str = "do did does if as let to don't didn't you";
const LEXICAL_SO: &str = "that far much many";
const LEXICAL_WELL: &str = "within done over below above worth known i it you";
const MORE_CLAUSE_VERBS: &str =
    "think know want need mean agree feel guess suggest like see say believe prefer understand";
const WORD_FLAGS: &str = "so well yeah and but or you know i mean like it";
const I_CUT: &str = "it it's its is isn't if in into it'll it'd i'm i'll i've";
const CLAUSE_BEFORE: &str = "and but so because cause 'cause if when then that or which where what
how why who can could would will should do did does are were is was have has had may might must
shall yeah okay ok well oh now i we they he she you it think guess mean know hope suppose believe
dunno reckon say said maybe no right sure just also even actually basically really probably perhaps
yes";
/// Short words the cut-off rules took for fragments in meeting transcripts.
const MORE_KNOWN: &str = "wee mac mic sec veg alt ooh ada ami ass eh ta hi";
/// Spoken names of symbols, as in a dictated flag or path ("dash dash force dash push").
const SYMBOLS: &str =
    "dash hyphen slash backslash underscore tilde caret ampersand asterisk pipe hashtag";
pub(super) const NUMBER_WORDS: &str =
    "zero one two three four five six seven eight nine ten eleven twelve
thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty
seventy eighty ninety hundred thousand million billion";
const NEGATION: &str = "not no never none nothing don't doesn't didn't isn't aren't wasn't weren't
won't can't cannot couldn't shouldn't wouldn't haven't hasn't hadn't ain't nor neither without";
/// Units a spelled-out number may take ("five mm").
const UNIT_NAMES: &str = "mm cm km kg mg ml ms sec secs min mins hr hrs gb mb kb tb hz khz mhz ghz
px pt db dpi fps ft lb lbs oz mph kph";
/// The second words of filler phrases ("I mean,", "you know,", "let's see,").
const PHRASE_TAILS: &str = "mean know think guess see dunno";

/// The spelling and punctuation features of a word.
const SPELLING: [&str; 18] = [
    "hes",
    "comma",
    "end",
    "punct",
    "bare",
    "cap",
    "allcaps",
    "letters",
    "ordinary",
    "quote",
    "start",
    "last",
    "numeric",
    "digits",
    "short3",
    "hes_um",
    "plural_hes",
    "quoted",
];
/// Whether a word equals the word k places on (`same_k`), or is a proper prefix of it
/// (`pre_k`), with how far ahead each looks.
const MATCHES: [(&str, i32); 8] = [
    ("same1", 1),
    ("same2", 2),
    ("same3", 3),
    ("pre1", 1),
    ("pre2", 2),
    ("pre3", 3),
    ("same4", 4),
    ("same5", 5),
];

/// The word classes, by feature name.
fn classes() -> &'static Vec<(String, HashSet<String>)> {
    static CLASSES: OnceLock<Vec<(String, HashSet<String>)>> = OnceLock::new();
    CLASSES.get_or_init(|| {
        let set = |lists: &[&str]| -> HashSet<String> {
            lists
                .iter()
                .flat_map(|l| l.split_whitespace())
                .map(str::to_string)
                .collect()
        };
        let mut out: Vec<(String, HashSet<String>)> = vec![
            ("known".into(), set(&[rules::WORDS, MORE_KNOWN])),
            ("ai".into(), set(&["a i"])),
            ("det".into(), set(&[DETERMINERS])),
            ("dem".into(), set(&[DEMONSTRATIVES])),
            ("prep".into(), set(&[PREPOSITIONS])),
            ("subj".into(), set(&[SUBJECTS])),
            ("aux".into(), set(&[AUX])),
            ("contr".into(), set(&[CONTRACTIONS])),
            ("marker".into(), set(&[MARKERS])),
            ("backch".into(), set(&[BACKCHANNEL])),
            ("meant".into(), set(&[MEANT])),
            ("said_twice".into(), set(&[SAID_TWICE])),
            ("so_ender".into(), set(&[SO_ENDERS])),
            ("yk_lead".into(), set(&[YOU_KNOW_LEAD])),
            ("lex_so".into(), set(&[AUX, LEXICAL_SO])),
            ("lex_well".into(), set(&[LEXICAL_WELL])),
            ("not_clause".into(), set(&[MARKERS, BACKCHANNEL, ANSWERS])),
            ("clause_verb".into(), set(&[AUX, MORE_CLAUSE_VERBS])),
            (
                "other_marker".into(),
                set(&["yeah okay ok oh right anyway alright"]),
            ),
            (
                "func".into(),
                set(&[
                    DETERMINERS,
                    DEMONSTRATIVES,
                    PREPOSITIONS,
                    SUBJECTS,
                    AUX,
                    CONTRACTIONS,
                    MORE_FUNCTION,
                ]),
            ),
            ("pron_double".into(), set(&["it you"])),
            ("clause_before".into(), set(&[CLAUSE_BEFORE])),
            ("i_cut".into(), set(&[I_CUT])),
            ("w_a".into(), set(&["a"])),
            ("symbol".into(), set(&[SYMBOLS])),
            ("numword".into(), set(&[NUMBER_WORDS])),
            ("neg".into(), set(&[NEGATION])),
            ("unitname".into(), set(&[UNIT_NAMES])),
            ("phrase_tail".into(), set(&[PHRASE_TAILS])),
        ];
        for w in WORD_FLAGS.split_whitespace() {
            out.push((format!("w_{w}"), set(&[w])));
        }
        out
    })
}

/// One unit: its name, the kind of tag or guard it gives (empty for one other units read), its
/// inputs as (offset, feature, weight), and its bias.
pub(super) struct Unit {
    name: String,
    kind: &'static str,
    inputs: Vec<(i32, usize, f32)>,
    bias: f32,
    /// How far ahead it looks, blockers included.
    pub(super) look: i32,
}

/// The features the units read, by name, with how far ahead each looks, and the layers of units
/// that add to them.
pub(super) struct Network {
    index: HashMap<String, usize>,
    pub(super) look: Vec<i32>,
    pub(super) layers: Vec<Vec<Unit>>,
}

/// An input of a unit, the word `offset` places on and the feature's name.
pub(super) type Input<'a> = (i32, &'a str);

impl Network {
    /// A network whose base features are `base`, each with how far ahead it looks.
    pub(super) fn with_base(base: &[(&str, i32)]) -> Network {
        let mut network = Network {
            index: HashMap::new(),
            look: Vec::new(),
            layers: Vec::new(),
        };
        for &(name, look) in base {
            network.push(name, look);
        }
        network
    }

    /// A network over each word's spelling, classes and matches, and `extra`.
    fn new(extra: &[(&str, i32)]) -> Network {
        let mut network = Network::with_base(&[]);
        let lengths: Vec<String> = (1..=MAXP + 1).map(|l| format!("len{l}")).collect();
        let classes = classes().iter().map(|(name, _)| name.as_str());
        for name in SPELLING
            .into_iter()
            .chain(lengths.iter().map(String::as_str))
            .chain(classes)
        {
            network.push(name, 0);
        }
        for &(name, look) in MATCHES.iter().chain(extra) {
            network.push(name, look);
        }
        network
    }

    fn push(&mut self, name: &str, look: i32) {
        let fresh = self.index.insert(name.to_string(), self.look.len());
        assert!(fresh.is_none(), "two features are named {name}");
        self.look.push(look);
    }

    pub(super) fn feature(&self, name: &str) -> usize {
        *self
            .index
            .get(name)
            .unwrap_or_else(|| panic!("no feature {name}"))
    }

    /// Adds to `layer` a unit that is 1 when every `need` input is and no `block` input is, or,
    /// with `any`, when any of those is too.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn unit(
        &self,
        layer: &mut Vec<Unit>,
        name: &str,
        kind: &'static str,
        need: &[Input],
        any: &[Input],
        block: &[Input],
        bias: Option<f32>,
    ) {
        let look = (need.iter().chain(any).chain(block))
            .map(|&(o, f)| o + self.look[self.feature(f)])
            .fold(0, i32::max);
        assert!(look <= LOOK, "{name} looks {look} words ahead");
        let mut inputs = Vec::new();
        for (list, weight) in [(need, 1.0), (any, 1.0), (block, -3.0)] {
            inputs.extend(list.iter().map(|&(o, f)| (o, self.feature(f), weight)));
        }
        layer.push(Unit {
            name: name.to_string(),
            kind,
            inputs,
            bias: bias.unwrap_or(1.0 - need.len() as f32),
            look,
        });
    }

    /// Makes `layer`'s units features the next layers read.
    pub(super) fn close(&mut self, layer: Vec<Unit>) {
        for unit in &layer {
            self.push(&unit.name, unit.look);
        }
        self.layers.push(layer);
    }

    /// Every feature of every word, the base features `z` first.
    pub(super) fn run(&self, mut z: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let n = z.len();
        for layer in &self.layers {
            let out: Vec<Vec<f32>> = (0..n)
                .map(|i| {
                    (layer.iter())
                        .map(|u| {
                            let mut sum = u.bias;
                            for &(o, f, w) in &u.inputs {
                                let j = i as i32 + o;
                                if (0..n as i32).contains(&j) {
                                    sum += w * z[j as usize][f];
                                }
                            }
                            sum.clamp(0.0, 1.0)
                        })
                        .collect()
                })
                .collect();
            for (row, units) in z.iter_mut().zip(out) {
                row.extend(units);
            }
        }
        z
    }

    /// The units of every layer that give a tag or a guard, with their features.
    fn outputs(&self) -> Vec<(usize, &'static str)> {
        (self.layers.iter().flatten())
            .filter(|u| !u.kind.is_empty())
            .map(|u| (self.feature(&u.name), u.kind))
            .collect()
    }
}

/// The letters of `word` with each run of one letter written once ("umm" gives "um").
fn runs(word: &str) -> String {
    let mut out = String::new();
    for c in word.chars() {
        if !out.ends_with(c) {
            out.push(c);
        }
    }
    out
}

/// The features of each word that no unit computes, as `network` indexes them.
fn base(network: &Network, words: &[&str]) -> Vec<Vec<f32>> {
    let n = words.len();
    let width = network.look.len() - network.layers.iter().map(Vec::len).sum::<usize>();
    let norms: Vec<String> = words.iter().map(|w| rules::norm(w)).collect();
    let mut z = vec![vec![0f32; width]; n];
    for (i, (&w, a)) in words.iter().zip(&norms).enumerate() {
        let (_, core, _) = crate::fillers::split(w);
        let stripped = w.trim_end_matches(['"', '\'', ')']);
        let hes = rules::is_hesitation(w);
        let comma = stripped.ends_with([',', ';', ':']);
        let end = stripped.ends_with(['.', '?', '!']);
        let allcaps = core.chars().count() >= 2 && rules::is_upper(core);
        let only = |s: &str, f: fn(char) -> bool| !s.is_empty() && s.chars().all(f);
        let digit = a.chars().any(|c| c.is_ascii_digit());
        let length = a.chars().count();
        let rest = a.strip_suffix('s').unwrap_or("");
        let squeezed = runs(rest);
        let mut set = |name: &str| z[i][network.feature(name)] = 1.0;
        let pairs = [
            ("hes", hes),
            ("comma", comma),
            ("end", end),
            ("punct", comma || end),
            ("bare", !a.is_empty() && w.to_lowercase() == *a),
            ("allcaps", allcaps),
            (
                "cap",
                core.chars().next().is_some_and(char::is_uppercase) && !allcaps,
            ),
            ("letters", only(a, |c| c.is_ascii_lowercase())),
            (
                "ordinary",
                only(a, |c| c.is_ascii_lowercase() || c == '\'' || c == '-')
                    && !has_class("not_clause", a)
                    && !hes,
            ),
            ("quote", w.starts_with(['\'', '"'])),
            // An opening quote mark, or a closing double one. A word that is all punctuation
            // counts too, as the reference's empty last character matches any mark.
            (
                "quoted",
                w.starts_with(['\'', '"', '“', '‘'])
                    || (w
                        .trim_end_matches(['.', ',', '?', '!', ';', ':'])
                        .chars()
                        .next_back())
                    .is_none_or(|c| c == '"' || c == '”'),
            ),
            ("start", i == 0),
            ("last", i + 1 == n),
            ("numeric", digit || has_class("numword", a)),
            ("digits", digit),
            ("short3", (1..=3).contains(&length)),
            // A hesitation that is never a unit or a word ("um", "uhm", "err").
            (
                "hes_um",
                ["um", "uh", "uhm", "er", "erm"].contains(&runs(a).as_str()),
            ),
            // A hesitation talked about as a word ("ums", "ahs").
            (
                "plural_hes",
                ["um", "uh", "er", "erm", "ah", "hm"].contains(&squeezed.as_str())
                    || (squeezed == "m" && rest.len() >= 2),
            ),
        ];
        for (name, on) in pairs {
            if on {
                set(name);
            }
        }
        if length > 0 {
            set(&format!("len{}", length.min(MAXP + 1)));
        }
        for (name, list) in classes() {
            if list.contains(a.as_str()) {
                set(name);
            }
        }
        // Matches with the words 1 to 5 places on.
        let chars: Vec<char> = a.chars().collect();
        for k in 1..=5 {
            let Some(b) = norms.get(i + k) else { break };
            if !a.is_empty() && a == b {
                set(&format!("same{k}"));
            }
            if k <= 3
                && (1..=MAXP).contains(&chars.len())
                && b.chars().count() > chars.len()
                && b.chars().zip(&chars).all(|(x, &y)| x == y)
            {
                set(&format!("pre{k}"));
            }
        }
    }
    z
}

pub(super) fn has_class(name: &str, word: &str) -> bool {
    (classes().iter())
        .find(|(n, _)| n == name)
        .is_some_and(|(_, list)| list.contains(word))
}

/// The hand-written cleanup's network and its detectors' features and tags.
fn hand_written_network() -> &'static (Network, Vec<(usize, &'static str)>) {
    static NETWORK: OnceLock<(Network, Vec<(usize, &'static str)>)> = OnceLock::new();
    NETWORK.get_or_init(|| {
        let mut s = Network::new(&[]);
        let longer: Vec<String> = (4..=MAXP + 1).map(|l| format!("len{l}")).collect();
        let longer: Vec<Input> = longer.iter().map(|l| (0, l.as_str())).collect();

        // Composite flags of a word and its neighbours.
        let mut a = Vec::new();
        // A short token that isn't a word, a cut-off fragment ("th", "s", "cha").
        let block: Vec<Input> = [
            (0, "known"),
            (0, "punct"),
            (0, "hes"),
            (0, "allcaps"),
            (0, "cap"),
        ]
        .into_iter()
        .chain(longer.iter().copied())
        .collect();
        s.unit(&mut a, "frag", "", &[(0, "letters")], &[], &block, None);
        // A capitalized token of one or two letters that isn't a word ("S so like").
        let block: Vec<Input> = [(0, "known"), (0, "punct"), (0, "hes"), (0, "len3")]
            .into_iter()
            .chain(longer.iter().copied())
            .collect();
        s.unit(
            &mut a,
            "frag_cap",
            "",
            &[(0, "cap")],
            &[(0, "len1"), (0, "len2")],
            &block,
            Some(-1.0),
        );
        // The word starts a sentence, after its start, a sentence end, a hesitation, or a
        // marker that does ("Okay so I think").
        s.unit(
            &mut a,
            "sent0",
            "",
            &[],
            &[(0, "start"), (-1, "end")],
            &[],
            Some(0.0),
        );
        s.unit(&mut a, "sent_hes", "", &[(-1, "hes")], &[], &[], None);
        s.unit(
            &mut a,
            "sent_mark",
            "",
            &[(-1, "marker"), (-1, "bare")],
            &[(-1, "start"), (-2, "end")],
            &[],
            Some(-2.0),
        );
        s.unit(
            &mut a,
            "sent_mark2",
            "",
            &[(-2, "marker"), (-2, "bare"), (-1, "marker"), (-1, "bare")],
            &[(-2, "start"), (-3, "end")],
            &[],
            Some(-4.0),
        );
        // A clause follows, with or without a hesitation before it.
        s.unit(
            &mut a,
            "cl_contr",
            "",
            &[(1, "contr"), (1, "bare")],
            &[],
            &[(2, "marker")],
            None,
        );
        let after_subject = |o| [(o, "not_clause"), (o, "subj"), (o, "hes"), (o, "punct")];
        s.unit(
            &mut a,
            "cl_subj",
            "",
            &[(1, "subj"), (1, "bare")],
            &[],
            &after_subject(2),
            None,
        );
        s.unit(
            &mut a,
            "cl_aux",
            "",
            &[(1, "aux"), (1, "bare"), (2, "subj")],
            &[],
            &[],
            None,
        );
        s.unit(
            &mut a,
            "cl_dem",
            "",
            &[(1, "dem"), (1, "bare"), (2, "aux")],
            &[],
            &[],
            None,
        );
        s.unit(
            &mut a,
            "clh_contr",
            "",
            &[(1, "hes"), (2, "contr"), (2, "bare")],
            &[],
            &[(3, "marker")],
            None,
        );
        s.unit(
            &mut a,
            "clh_subj",
            "",
            &[(1, "hes"), (2, "subj"), (2, "bare")],
            &[],
            &after_subject(3),
            None,
        );
        s.unit(
            &mut a,
            "clh_aux",
            "",
            &[(1, "hes"), (2, "aux"), (2, "bare"), (3, "subj")],
            &[],
            &[],
            None,
        );
        // Two ordinary words follow.
        s.unit(
            &mut a,
            "two_ord",
            "",
            &[(1, "ordinary"), (2, "ordinary")],
            &[],
            &[(1, "end"), (1, "hes"), (2, "hes")],
            None,
        );
        s.unit(
            &mut a,
            "two_ord_hes",
            "",
            &[(1, "hes"), (2, "ordinary"), (3, "ordinary")],
            &[],
            &[(2, "end"), (1, "end")],
            None,
        );
        s.unit(
            &mut a,
            "and_but_so",
            "",
            &[],
            &[(0, "w_and"), (0, "w_but"), (0, "w_so")],
            &[],
            Some(0.0),
        );
        s.unit(
            &mut a,
            "and_but",
            "",
            &[],
            &[(0, "w_and"), (0, "w_but")],
            &[],
            Some(0.0),
        );
        // "it" or "you" where its double may be two clauses.
        s.unit(
            &mut a,
            "pron_mid",
            "",
            &[(0, "pron_double")],
            &[],
            &[(-1, "clause_before"), (-1, "punct"), (0, "start")],
            None,
        );
        s.close(a);

        // What a word is in its sentence.
        let mut b = Vec::new();
        s.unit(
            &mut b,
            "junk",
            "",
            &[],
            &[(0, "hes"), (0, "frag"), (0, "frag_cap")],
            &[],
            Some(0.0),
        );
        s.unit(
            &mut b,
            "sent",
            "",
            &[],
            &[
                (0, "sent0"),
                (0, "sent_hes"),
                (0, "sent_mark"),
                (0, "sent_mark2"),
            ],
            &[],
            Some(0.0),
        );
        let clauses = [
            "cl_contr",
            "cl_subj",
            "cl_aux",
            "cl_dem",
            "clh_contr",
            "clh_subj",
            "clh_aux",
        ];
        let any: Vec<Input> = clauses.iter().map(|&c| (0, c)).collect();
        s.unit(&mut b, "clause", "", &[], &any, &[], Some(0.0));
        s.unit(
            &mut b,
            "two",
            "",
            &[],
            &[(0, "two_ord"), (0, "two_ord_hes")],
            &[],
            Some(0.0),
        );
        // The word may be the first copy of a repeat.
        s.unit(
            &mut b,
            "rep_ok",
            "",
            &[(0, "bare")],
            &[],
            &[(0, "meant"), (0, "quote"), (0, "hes"), (0, "pron_mid")],
            None,
        );
        s.close(b);

        // The detectors, one per deletion rule.
        let mut c = Vec::new();
        s.unit(
            &mut c,
            "H",
            "H",
            &[(0, "hes")],
            &[],
            &[(0, "allcaps")],
            None,
        );
        // A word said again at once, or with one or two hesitations or fragments between.
        s.unit(
            &mut c,
            "rep1",
            "RM",
            &[(0, "same1"), (0, "rep_ok")],
            &[],
            &[(1, "quote")],
            None,
        );
        s.unit(
            &mut c,
            "rep1_gap",
            "RM",
            &[(0, "same2"), (0, "rep_ok"), (1, "junk")],
            &[],
            &[(1, "end")],
            None,
        );
        s.unit(
            &mut c,
            "rep1_gap2",
            "RM",
            &[(0, "same3"), (0, "rep_ok"), (1, "junk"), (2, "junk")],
            &[],
            &[(1, "end"), (2, "end")],
            None,
        );
        // A two-word phrase said twice at once, both words of its first copy.
        s.unit(
            &mut c,
            "rep2_a",
            "RM",
            &[
                (0, "same2"),
                (1, "same2"),
                (0, "bare"),
                (1, "bare"),
                (2, "bare"),
            ],
            &[],
            &[
                (0, "same1"),
                (0, "backch"),
                (0, "meant"),
                (0, "said_twice"),
                (1, "said_twice"),
                (0, "hes"),
                (1, "hes"),
                (3, "end"),
            ],
            None,
        );
        s.unit(
            &mut c,
            "rep2_b",
            "RM",
            &[
                (0, "same2"),
                (-1, "same2"),
                (-1, "bare"),
                (0, "bare"),
                (1, "bare"),
            ],
            &[],
            &[
                (-1, "same1"),
                (-1, "backch"),
                (-1, "meant"),
                (-1, "said_twice"),
                (0, "said_twice"),
                (-1, "hes"),
                (0, "hes"),
                (2, "end"),
            ],
            None,
        );
        // A cut-off word that the next word, or the one after a hesitation or fragment,
        // completes ("des design"), and a cut-off "a" or "i".
        let not_short = [(0, "known"), (0, "allcaps"), (0, "meant")];
        s.unit(
            &mut c,
            "cut1",
            "RM",
            &[(0, "pre1"), (0, "letters"), (0, "bare")],
            &[],
            &not_short,
            None,
        );
        s.unit(
            &mut c,
            "cut1_a",
            "RM",
            &[(0, "pre1"), (0, "w_a"), (0, "bare")],
            &[],
            &[],
            None,
        );
        s.unit(
            &mut c,
            "cut1_i",
            "RM",
            &[(0, "w_i"), (0, "bare"), (1, "i_cut")],
            &[],
            &[],
            None,
        );
        let block: Vec<Input> = not_short.into_iter().chain([(1, "end")]).collect();
        s.unit(
            &mut c,
            "cut2",
            "RM",
            &[(0, "pre2"), (0, "letters"), (0, "bare"), (1, "junk")],
            &[],
            &block,
            None,
        );
        s.unit(
            &mut c,
            "cut2_a",
            "RM",
            &[(0, "pre2"), (0, "w_a"), (0, "bare"), (1, "junk")],
            &[],
            &[(1, "end")],
            None,
        );
        s.unit(
            &mut c,
            "cut2_i",
            "RM",
            &[(0, "w_i"), (0, "bare"), (1, "junk"), (2, "i_cut")],
            &[],
            &[(1, "end")],
            None,
        );
        let block: Vec<Input> = not_short
            .into_iter()
            .chain([(1, "end"), (2, "end")])
            .collect();
        s.unit(
            &mut c,
            "cut3",
            "RM",
            &[
                (0, "pre3"),
                (0, "letters"),
                (0, "bare"),
                (1, "junk"),
                (2, "junk"),
            ],
            &[],
            &block,
            None,
        );
        // A fragment on its own ("the s the").
        s.unit(&mut c, "frag_alone", "RM", &[(0, "frag")], &[], &[], None);
        s.unit(
            &mut c,
            "frag_cap_alone",
            "RM",
            &[(0, "frag_cap")],
            &[],
            &[],
            None,
        );
        // A sentence-opening "so", "well" or "I mean" before a clause, and "so" or "well" before
        // two ordinary words.
        for (name, word, lexical, after) in [
            ("open_so", "w_so", "lex_so", "clause"),
            ("open_well", "w_well", "lex_well", "clause"),
        ] {
            s.unit(
                &mut c,
                name,
                "F",
                &[(0, word), (0, "sent"), (0, after)],
                &[],
                &[(0, "end"), (1, lexical), (0, "quote")],
                None,
            );
        }
        let clause_next: Vec<Input> = ["cl_contr", "cl_subj", "cl_aux", "cl_dem"]
            .iter()
            .map(|&c| (1, c))
            .collect();
        s.unit(
            &mut c,
            "open_imean",
            "F",
            &[(0, "w_i"), (0, "bare"), (1, "w_mean"), (0, "sent")],
            &clause_next,
            &[(1, "end"), (1, "quote")],
            Some(-4.0),
        );
        let clause_here: Vec<Input> = clause_next.iter().map(|&(_, c)| (0, c)).collect();
        s.unit(
            &mut c,
            "open_imean_b",
            "F",
            &[(0, "w_mean"), (-1, "w_i"), (-1, "bare"), (-1, "sent")],
            &clause_here,
            &[(0, "end"), (0, "quote")],
            Some(-4.0),
        );
        for (name, word, lexical) in [
            ("plain_so", "w_so", "lex_so"),
            ("plain_well", "w_well", "lex_well"),
        ] {
            s.unit(
                &mut c,
                name,
                "F",
                &[(0, word), (0, "sent"), (0, "two")],
                &[],
                &[(0, "end"), (1, lexical), (0, "quote")],
                None,
            );
        }
        // Trailing "so", "and so", "and yeah", "but yeah", "so yeah" and "you know".
        s.unit(
            &mut c,
            "trail_so",
            "F",
            &[(0, "w_so"), (0, "last")],
            &[],
            &[(-1, "so_ender"), (-1, "end"), (0, "quote")],
            None,
        );
        s.unit(
            &mut c,
            "trail_and_so",
            "F",
            &[(0, "and_but"), (1, "w_so"), (1, "last")],
            &[],
            &[(0, "comma"), (-1, "end")],
            None,
        );
        s.unit(
            &mut c,
            "trail_yeah",
            "F",
            &[(0, "w_yeah"), (0, "last"), (-1, "and_but_so")],
            &[],
            &[(-1, "comma"), (-1, "end"), (-2, "so_ender")],
            None,
        );
        s.unit(
            &mut c,
            "trail_c_yeah",
            "F",
            &[(0, "and_but_so"), (1, "w_yeah"), (1, "last")],
            &[],
            &[(0, "comma"), (0, "end"), (-1, "so_ender")],
            None,
        );
        s.unit(
            &mut c,
            "trail_you",
            "F",
            &[(0, "w_you"), (1, "w_know"), (1, "last")],
            &[],
            &[(-1, "yk_lead"), (-1, "end"), (0, "punct")],
            None,
        );
        s.unit(
            &mut c,
            "trail_know",
            "F",
            &[(0, "w_know"), (-1, "w_you"), (0, "last")],
            &[],
            &[(-2, "yk_lead"), (-2, "end"), (-1, "punct")],
            None,
        );
        // "you know" and "I mean" between commas.
        for (first, second) in [("you", "know"), ("i", "mean")] {
            let (w1, w2) = (format!("w_{first}"), format!("w_{second}"));
            s.unit(
                &mut c,
                &format!("comma_{first}"),
                "F",
                &[(0, &w1), (1, &w2), (-1, "comma"), (1, "comma")],
                &[],
                &[(-1, "end"), (0, "punct")],
                None,
            );
            s.unit(
                &mut c,
                &format!("comma_{second}"),
                "F",
                &[(0, &w2), (-1, &w1), (-2, "comma"), (0, "comma")],
                &[],
                &[(-2, "end"), (-1, "punct")],
                None,
            );
        }
        s.close(c);
        let outputs = s.outputs();
        (s, outputs)
    })
}

/// The guards' network and their features and kinds.
fn guard_network() -> &'static (Network, Vec<(usize, &'static str)>) {
    static NETWORK: OnceLock<(Network, Vec<(usize, &'static str)>)> = OnceLock::new();
    NETWORK.get_or_init(|| {
        let mut s = Network::new(&[("del", 0)]);
        let mut g = Vec::new();
        // A spoken symbol, or a word next to one ("dash dash force dash push").
        s.unit(
            &mut g,
            "g_symbol",
            "symbol",
            &[],
            &[(0, "symbol"), (-1, "symbol"), (1, "symbol")],
            &[],
            Some(0.0),
        );
        // A short token after digits ("5 mm", "3 d"), or a unit name after a spelled-out number.
        s.unit(
            &mut g,
            "g_unit",
            "unit",
            &[(-1, "digits"), (0, "short3")],
            &[],
            &[(0, "hes_um"), (0, "known"), (0, "same1"), (-1, "del")],
            None,
        );
        s.unit(
            &mut g,
            "g_unit_name",
            "unit",
            &[(-1, "numeric"), (0, "unitname")],
            &[],
            &[(0, "same1"), (-1, "del")],
            None,
        );
        // A filler talked about as a word ("delete ums"), or a quoted word.
        s.unit(
            &mut g,
            "g_mention",
            "mention",
            &[],
            &[(0, "plural_hes"), (0, "quoted")],
            &[],
            Some(0.0),
        );
        // List items of two or three words whose first word starts the next item too ("cleaner
        // build, cleaner layout"), with no marker, "like" or hesitation in them.
        let lead = |o| [(o, "marker"), (o, "backch"), (o, "w_like"), (o, "hes")];
        let block: Vec<Input> = lead(0)
            .into_iter()
            .chain([(1, "hes"), (1, "same2"), (1, "phrase_tail")])
            .collect();
        s.unit(
            &mut g,
            "g_list_2a",
            "list",
            &[(0, "bare"), (0, "same2"), (1, "comma")],
            &[],
            &block,
            None,
        );
        let block: Vec<Input> = lead(-1)
            .into_iter()
            .chain([(0, "hes"), (0, "same2"), (0, "phrase_tail")])
            .collect();
        s.unit(
            &mut g,
            "g_list_2b",
            "list",
            &[(0, "comma"), (-1, "same2")],
            &[],
            &block,
            None,
        );
        let block: Vec<Input> = lead(0)
            .into_iter()
            .chain([(1, "punct"), (1, "hes"), (2, "hes")])
            .collect();
        s.unit(
            &mut g,
            "g_list_3a",
            "list",
            &[(0, "bare"), (0, "same3"), (2, "comma")],
            &[],
            &block,
            None,
        );
        let block: Vec<Input> = lead(-1)
            .into_iter()
            .chain([(0, "punct"), (0, "hes"), (1, "hes")])
            .collect();
        s.unit(
            &mut g,
            "g_list_3b",
            "list",
            &[(-1, "same3"), (1, "comma")],
            &[],
            &block,
            None,
        );
        let block: Vec<Input> = lead(-2)
            .into_iter()
            .chain([(-1, "punct"), (-1, "hes"), (0, "hes")])
            .collect();
        s.unit(
            &mut g,
            "g_list_3c",
            "list",
            &[(0, "comma"), (-2, "same3")],
            &[],
            &block,
            None,
        );
        // A cut of one to three words with a subject pronoun two before it, one kept word
        // between, and the same pronoun right after it ("we just [need to] we need to").
        for l in 1..=3i32 {
            let around = format!("same{}", l + 2);
            let first_copy = format!("same{l}");
            let mut need: Vec<Input> = (0..l).map(|o| (o, "del")).collect();
            need.extend([(-2, "subj"), (-2, around.as_str())]);
            let mut block: Vec<Input> = vec![
                (-1, "del"),
                (l, "del"),
                (-2, "del"),
                (0, "hes"),
                (0, "short3"),
                (0, first_copy.as_str()),
            ];
            if l <= 2 {
                block.push((-1, around.as_str()));
            }
            s.unit(
                &mut g,
                &format!("g_str_subj_{l}"),
                "straddle",
                &need,
                &[],
                &block,
                None,
            );
        }
        // Parallel and contrasting phrases: a one-word content cut whose frame comes back right
        // after it, with or without a negation, or a lone cut word that returns three on.
        let content = [
            (0, "hes"),
            (0, "short3"),
            (0, "func"),
            (0, "w_like"),
            (0, "marker"),
            (0, "backch"),
            (0, "neg"),
        ];
        let with = |more: &[Input<'static>]| -> Vec<Input<'static>> {
            content
                .iter()
                .copied()
                .chain(more.iter().copied())
                .collect()
        };
        s.unit(
            &mut g,
            "g_par_frame2",
            "parallel",
            &[(0, "del"), (-2, "same3"), (-1, "same3")],
            &[],
            &with(&[
                (-2, "del"),
                (-1, "del"),
                (1, "del"),
                (-1, "hes"),
                (-2, "hes"),
            ]),
            None,
        );
        s.unit(
            &mut g,
            "g_par_lead",
            "parallel",
            &[(0, "del"), (0, "same3")],
            &[],
            &with(&[(1, "del"), (2, "del"), (3, "del"), (1, "hes"), (2, "hes")]),
            None,
        );
        s.unit(
            &mut g,
            "g_con_frame2",
            "contrast",
            &[(0, "del"), (-2, "same3"), (-1, "same3"), (3, "neg")],
            &[],
            &with(&[(-2, "del"), (-1, "del"), (1, "del")]),
            None,
        );
        s.unit(
            &mut g,
            "g_con_frame1",
            "contrast",
            &[(0, "del"), (-1, "same2"), (2, "neg")],
            &[],
            &with(&[(-1, "del"), (1, "del"), (-1, "hes")]),
            None,
        );
        s.close(g);
        // The straddle veto reaches the later words of the cut.
        let mut h = Vec::new();
        let any: Vec<Input> = [("g_str_subj_1"), ("g_str_subj_2"), ("g_str_subj_3")]
            .iter()
            .map(|&u| (0, u))
            .collect();
        s.unit(&mut h, "g_str_any", "", &[], &any, &[], Some(0.0));
        s.close(h);
        let mut r = Vec::new();
        s.unit(
            &mut r,
            "g_str_run1",
            "straddle",
            &[(0, "del"), (-1, "del"), (-1, "g_str_any")],
            &[],
            &[(0, "hes")],
            None,
        );
        s.unit(
            &mut r,
            "g_str_run2",
            "straddle",
            &[(0, "del"), (-1, "del"), (-2, "del"), (-2, "g_str_any")],
            &[],
            &[(0, "hes")],
            None,
        );
        s.close(r);
        let outputs = s.outputs();
        (s, outputs)
    })
}

/// For each of `words`, the kind of the hand-written cleanup's first detector that deletes it ("H",
/// "RM" or "F"), or none.
pub(crate) fn hand_written(words: &[&str]) -> Vec<Option<&'static str>> {
    let (network, detectors) = hand_written_network();
    let z = network.run(base(network, words));
    (z.iter())
        .map(|row| {
            (detectors.iter())
                .find(|&&(f, _)| row[f] > 0.0)
                .map(|&(_, kind)| kind)
        })
        .collect()
}

/// For each of `words`, the kinds of guard that veto its deletion, `deleted` being any stage's
/// deletions. A word that isn't deleted has none.
pub(crate) fn vetoes(words: &[&str], deleted: &[bool]) -> Vec<Vec<&'static str>> {
    let (network, guards) = guard_network();
    let mut z = base(network, words);
    let del = network.feature("del");
    for (row, &d) in z.iter_mut().zip(deleted) {
        row[del] = if d { 1.0 } else { 0.0 };
    }
    let z = network.run(z);
    (z.iter().zip(deleted))
        .map(|(row, &d)| {
            let mut kinds: Vec<&'static str> = (guards.iter())
                .filter(|&&(f, _)| d && row[f] > 0.5)
                .map(|&(_, kind)| kind)
                .collect();
            kinds.sort_unstable();
            kinds.dedup();
            kinds
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn marked(text: &str) -> String {
        let words: Vec<&str> = text.split(' ').collect();
        (words.iter().zip(hand_written(&words)))
            .map(|(w, k)| match k {
                Some(k) => format!("[{w}/{k}]"),
                None => w.to_string(),
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn the_hand_written_network_deletes_fragments_repeats_and_openers() {
        assert_eq!(
            marked("Um so I I think the the uh the des design is is good so."),
            "[Um/H] [so/F] [I/RM] I think [the/RM] [the/RM] [uh/H] the [des/RM] design [is/RM] is good [so./F]"
        );
        assert_eq!(
            marked("we will move to f conceptual design"),
            "we will move to [f/RM] conceptual design"
        );
        assert!(hand_written(&[]).is_empty());
    }

    #[test]
    fn the_networks_look_three_words_ahead_at_most() {
        for (network, _) in [hand_written_network(), guard_network()] {
            assert!(network.layers.iter().flatten().all(|u| u.look <= LOOK));
        }
    }

    #[test]
    fn guards_veto_deletions_that_change_the_meaning() {
        let veto = |text: &str, deleted: &[usize]| {
            let words: Vec<&str> = text.split(' ').collect();
            let mask: Vec<bool> = (0..words.len()).map(|i| deleted.contains(&i)).collect();
            vetoes(&words, &mask)
        };
        for (text, deleted, kept, kind) in [
            ("Dash dash force dash push", &[2][..], &[2][..], "symbol"),
            ("Dash dash force dash push", &[1], &[1], "symbol"),
            ("the dash v build option", &[2], &[2], "symbol"),
            ("The panel is 3 cm too tall", &[4], &[4], "unit"),
            (
                "Editors should usually delete ums and then fix spacing",
                &[4],
                &[4],
                "mention",
            ),
            (
                "set up like a cleaner build, cleaner layout for that one screen, cleaner tests around it",
                &[4, 5],
                &[4, 5],
                "list",
            ),
            (
                "we just need to uh we need to check it",
                &[2, 3, 4],
                &[2, 3],
                "straddle",
            ),
            (
                "what they prefer what they not prefer",
                &[2],
                &[2],
                "parallel",
            ),
            (
                "what they prefer what they not prefer",
                &[2],
                &[2],
                "contrast",
            ),
            ("they prefer they not prefer", &[1], &[1], "contrast"),
            (
                "enhance our profit enhance our sales",
                &[0],
                &[0],
                "parallel",
            ),
        ] {
            let got = veto(text, deleted);
            for &i in kept {
                assert!(got[i].contains(&kind), "{text:?} word {i}: {:?}", got[i]);
            }
        }
        for (text, deleted) in [
            ("I I think the the idea is good", &[0, 3][..]),
            ("we need some uh some help", &[2, 3]),
            ("it was 5 uh 5 mm wide", &[3]),
            ("the red, sorry, the blue one", &[1, 2, 3]),
            ("yeah so des design a new remote", &[2]),
            ("I know i i like to have", &[2]),
            ("I added slides, um, I added a", &[2, 3]),
            ("I mean, I suppose it would", &[0, 1]),
            ("do you wan do you wanna finish", &[2]),
            ("twenty s functions of a remote", &[1]),
        ] {
            assert!(veto(text, deleted).iter().all(Vec::is_empty), "{text:?}");
        }
    }
}
