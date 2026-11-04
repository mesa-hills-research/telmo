//! Rules that delete disfluencies without the model, run beside the filler rules.
//!
//! Each rule looks at most three words past the word it deletes, and each gives the positions it
//! deletes.
//!
//! 1. A cut-off word goes when the word after it, hesitations skipped and at most three words on,
//!    starts with it and goes on past it ("des design", "mic microphone", "it' it's"). A short
//!    word on the [`WORDS`] list stays ("we were", "use user"), and so does a word the next one
//!    extends only by a contraction ("that that's", "do don't"). A lone "i" before "it", "is",
//!    "if" or "in" is cut off too ("i it's").
//! 2. A cut-off of up to four letters also goes when it resumes after function words or
//!    hesitations ("th I think the"), unless it's in capitals or punctuation comes between.
//! 3. A preposition and determiner go when the same preposition and another determiner follow at
//!    once ("in our in your folder").
//! 4. A reparandum of one to three words goes with "sorry" set off by commas or "or rather" after
//!    it, when the repair starts with its first word or is one word ("the red, sorry, the blue
//!    one").
//! 5. A two-word phrase said twice in a row loses its first copy ("has to has to fulfil"), unless
//!    punctuation is inside it or after its first copy, both words are meant doubles ("no no"),
//!    it holds "and", "or", "but" or "please", or it starts with a backchannel.
//! 6. A doubled function word loses its first copy ("have to to make it"). "it" and "you" double
//!    only where the first copy starts a clause, and a particle after an object pronoun stays
//!    ("hand them in in a week").
//! 7. A sentence-opening "so", "well", "yeah", "okay", "oh", "right", "anyway", "alright" or "I
//!    mean" goes before a clause ("So I think", "Well, we need").
//! 8. A trailing "so", "and yeah", "but yeah", "so yeah" or "you know" goes, and so do "you know"
//!    and "I mean" between commas ("we need, you know, a way").
//! 9. A sentence-opening "so" or "well" also goes before two ordinary words ("So the method is"),
//!    but not as "so that", "so much" or "well done", nor before a word that answers on its own
//!    ("So no.").

use std::collections::HashSet;

/// How far past a word a rule may look.
const LOOK: usize = 3;
const PUNCT: [char; 6] = [',', '.', '?', '!', ';', ':'];
const SENTENCE_END: [char; 3] = ['.', '?', '!'];

/// English words of one to four letters, never taken for a cut-off word.
const WORDS: &str = "
a i
ad am an as at be by do go he hi if in is it me my no of oh ok on or so to up us we
ace act add ado age ago aid aim air all and ant any ape apt arc are ark arm art ash ask ate awe
axe bad bag ban bar bat bay bed bee beg bet bid big bin bit bob bog boo bow box boy bud bug bun
bus but buy bye cab can cap car cat cod cop cot cow coy cry cub cue cup cut dab dad day den dew
did die dig dim din dip doc doe dog don dot dry dub dud due dug dye ear eat ebb egg ego elf elk
elm end era eve eye fab fad fan far fat fax fed fee few fig fin fir fit fix fly foe fog for fox
fry fun fur gag gap gas gel gem get gig gin god got gum gun gut guy gym had ham has hat hay hem
hen her hey hid him hip his hit hog hop hot how hub hue hug huh hum hut ice icy ill ink inn ion
its ivy jab jam jar jaw jay jet jig job jog jot joy jug keg key kid kin kit lab lad lag lap law
lay led leg let lid lie lip lit log lot low mad man map mat may men met mid mix mob mod mom mop
mow mud mug mum nag nap nay net new nil nip nod nor not now nut oak oar oat odd off oil old one
opt orb ore our out owe owl own pad pal pan par pat paw pay pea peg pen pep per pet pew pie pig
pin pit ply pod pop pot pro pry pub pun pup put rag ram ran rap rat raw ray red ref rib rid rig
rim rip rob rod rot row rub rug rum run rut sad sag sat saw say sea see set sew she shy sin sip
sir sit six ski sky sly sob son soy spa spy sub sue sum sun sup tab tad tag tan tap tar tax tea
tee ten the tie tin tip toe ton too top tow toy try tub tug two use van vat vet via vow wag war
was wax way web wed wet who why wig win wit woe wok won wow yea yen yep yes yet you zap zip zoo
able ache acid acre aged ages also amid area arms army arts asks atom aunt auto avid away baby
back bake bald ball band bank bare bark barn base bath bead beak beam bean bear beat beef been
beer bell belt bend bent best bias bike bill bind bird bite blew blow blue blur boat body boil
bold bolt bomb bond bone book boom boot bore born boss both bowl bulk bull bump burn bury bush
busy cafe cage cake call calm came camp card care cart case cash cast cell chat chef chip city
clad clap claw clay clip club clue coal coat code coin cold come cook cool cope copy cord core
cork corn cost cosy cozy crew crop crow cube cure curl cute damp dare dark darn dash data date
dawn days dead deaf deal dean dear debt deck deed deem deep deer demo deny desk dial dice died
diet dire dirt dish disk dive dock does doll dome done doom door dose dots down drag draw drew
drip drop drug drum dual duck dude duke dull dumb dump dune dusk dust duty each earn ease east
easy edge edit else even ever evil exam exit expo eyes face fact fade fail fair fake fall fame
fare farm fast fate fear feat feed feel fees feet fell felt fern file fill film find fine fire
firm fish fist five flag flat flaw fled flee flew flip flow foam foil fold folk fond font food
fool foot fork form fort foul four free frog from fuel full fund fuse fuss gain gala game gang
gate gave gaze gear gene gift girl give glad glow glue goal goat goes gold golf gone good gown
grab gray grew grey grid grim grin grip grow gulf guts hack hail hair half hall halt hand hang
hard harm hate haul have head heal heap hear heat heel held hell help herb here hero hers hide
high hike hill hint hire hold hole holy home hood hook hope horn hose host hour huge hung hunt
hurt idea idle inch info into iron isle item jack jail jazz jean jeep jerk jobs join joke jump
junk jury just keen keep kept kick kids kill kind king kiss kite knee knew knit knob knot know
lack lady laid lake lamb lamp land lane last late lawn lazy lead leaf leak lean leap left lend
lens less lest liar lick life lift like lime limp line link lion lips list live load loan lock
loft logo long look loop lord lose loss lost lots loud love luck lump lung lure lush made mail
main make male mall many mark mask mass mast mate math maze meal mean meat meet melt memo mend
menu mere mesh mess mice mild mile milk mill mind mine mint miss mist mode mold mole mood moon
more moss most moth move much mule must myth nail name navy near neat neck need neon nest news
next nice nine node none noon norm nose note noun null nuts oath obey odds okay once ones only
onto open oral oven over owed owes owns pace pack page paid pain pair pale palm pane park part
pass past path peak pear peel peer pick pier pile pill pine pink pint pipe plan play plea plot
plug plus poem poet pole poll pond pony pool poor pope pork port pose post pour pray prey prop
pull pump punk pure push puts quit quiz race rack rage raid rail rain rank rare rash rate read
real rear rely rent rest rice rich ride ring riot rise risk road roam roar robe rock rode role
roll roof room root rope rose ruby rude ruin rule rush rust safe saga said sail sake sale salt
same sand sang save says scan scar seal seat seed seek seem seen self sell send sent shed ship
shoe shop shot show shut sick side sigh sign silk sing sink site size skin skip slam slap slid
slim slip slot slow snap snow soap soar sock sofa soft soil sold sole some song soon sore sort
soul soup sour spin spot star stay stem step stir stop such suit sung sure swap swim tail take
tale talk tall tank tape task taxi team tear tech tell tend tens term test text than that them
then they thin this thus tick tide tidy tied tier ties tile till time tiny tire toad told toll
tomb tone took tool tops tore torn toss tour town toys trap tray tree trim trio trip true tube
tune turn twin type ugly undo unit upon urge used user uses vain vary vast verb very vest veto
vice view vine visa vote wade wage wait wake walk wall want ward warm warn wary wash wave ways
weak wear week well went were west what when whip whom wide wife wild will wind wine wing wipe
wire wise wish with woke wolf wood wool word wore work worm worn wrap yard yeah year your zero
zone
";

/// Words a lone "i" can't stand before, so the "i" is a cut-off ("i it's").
const I_CUT: [&str; 10] = [
    "it", "it's", "its", "is", "isn't", "if", "in", "into", "it'll", "it'd",
];

/// Words said twice on purpose, which a doubled phrase made only of them keeps.
const MEANT: &str =
    "yeah yes no okay ok oh so very really bye ha blah etcetera yep well now far long
never many had that is right sure hey knock mm-hmm uh-huh huh ooh true bling go zero one two three
four five six seven eight nine ten eleven twelve twenty thirty forty fifty hundred thousand please
more much again all round over on off up down bit little";

/// The function words the filler rules already take two copies of.
const STUTTERED: &str = "a an the and or but i my our your their we they he she it's that's there's
what's let's i'm we're they're you're he's she's i've we've they've you've i'll we'll they'll
you'll it'll";

/// Function words whose double is a stutter, except those in [`MEANT`] or [`STUTTERED`].
const DOUBLED_CANDIDATES: &str =
    "to in at with for of from about into by this these those what which
how when where why who some any just not than then should can could would will are was were";

/// Doubles that one copy says as well as two do ("no no no no", "dwell on on the look").
const DOUBLED_ANYWAY: [&str; 7] = ["no", "on", "up", "off", "over", "down", "all"];

/// Words after which "it" or "you" starts a clause.
const CLAUSE_BEFORE: &str =
    "and but so because cause 'cause if when then that or which where what how
why who can could would will should do did does are were is was have has had may might must shall
yeah okay ok well oh now i we they he she you it think guess mean know hope suppose believe dunno
reckon say said maybe no right sure just also even actually basically really probably perhaps yes";

const PARTICLES: [&str; 9] = ["in", "at", "about", "by", "on", "up", "off", "over", "down"];
const OBJECTS: [&str; 9] = [
    "it", "them", "him", "her", "me", "us", "you", "this", "that",
];
const PREPOSITIONS: &str =
    "to of in on at for from with into onto over under about through by without between";
const DETERMINERS: &str = "a an the my your our their his her its";
/// Words a cut-off may resume after, besides prepositions, determiners and the stuttered words.
const RESTART: &str =
    "you it is are was were be been has have had do does did could should would can
will must might not just only very really also maybe actually like well yeah no so how what when
where if because all then than now think mean know";

const BACKCHANNEL: [&str; 11] = [
    "yeah", "yes", "yep", "yup", "okay", "ok", "right", "alright", "sure", "mm-hmm", "uh-huh",
];
/// A two-word phrase holding one of these is said twice on purpose ("such and such and").
const SAID_TWICE: [&str; 4] = ["and", "or", "but", "please"];

const INITIAL_MARKERS: [&str; 9] = [
    "so", "well", "yeah", "okay", "ok", "oh", "right", "anyway", "alright",
];
const CLAUSE_SUBJECTS: [&str; 7] = ["i", "you", "we", "they", "he", "she", "it"];
const CLAUSE_AUX: &str =
    "am is are was were has have had do does did can could may might must should
would will";
const MORE_CONTRACTIONS: [&str; 7] = ["i'd", "we'd", "you'd", "they'd", "he'll", "she'd", "she'll"];
const MORE_VERBS: &str =
    "think know want need mean agree feel guess suggest like see say believe prefer understand";

/// After these, "so" is an intensifier or a conjunction ("so that", "so much").
const LEXICAL_SO: [&str; 4] = ["that", "far", "much", "many"];
/// After these, "well" is an adverb ("well done"), or a pronoun follows that the
/// sentence-opening rule decides.
const LEXICAL_WELL: [&str; 10] = [
    "within", "done", "over", "below", "above", "worth", "known", "i", "it", "you",
];
/// Words that answer or react on their own, so they don't start a clause ("So no.").
const ANSWERS: &str =
    "yes no yep yup nope sure exactly thanks thank sorry please fine cool great good
true maybe perhaps absolutely definitely probably mm-hmm uh-huh huh";

/// Words after which a trailing "so" answers and stays ("I think so").
const SO_ENDERS: &str =
    "think thought hope hoped guess guessed suppose supposed believe believed say
said assume imagine expect reckon afraid or not if even do did done doing more less";
/// Words before "you know" that make it a question or an object ("do you know").
const YOU_KNOW_LEAD: [&str; 10] = [
    "do", "did", "does", "if", "as", "let", "to", "don't", "didn't", "you",
];

fn has(list: &str, word: &str) -> bool {
    list.split_whitespace().any(|w| w == word)
}

fn is_stuttered(word: &str) -> bool {
    has(STUTTERED, word)
}

fn is_doubled(word: &str) -> bool {
    (has(DOUBLED_CANDIDATES, word) && !has(MEANT, word) && !is_stuttered(word))
        || DOUBLED_ANYWAY.contains(&word)
}

fn is_restart(word: &str) -> bool {
    !["and", "or", "but"].contains(&word)
        && (has(PREPOSITIONS, word)
            || has(DETERMINERS, word)
            || is_stuttered(word)
            || has(RESTART, word))
}

fn is_clause_contraction(word: &str) -> bool {
    (is_stuttered(word) && word.contains('\'')) || MORE_CONTRACTIONS.contains(&word)
}

fn is_clause_verb(word: &str) -> bool {
    has(CLAUSE_AUX, word) || has(MORE_VERBS, word)
}

fn is_initial_marker(word: &str) -> bool {
    INITIAL_MARKERS.contains(&word)
}

/// A word's punctuation before, its letters and digits with what's between them, and its
/// punctuation after.
fn split(word: &str) -> (&str, &str, &str) {
    crate::fillers::split(word)
}

/// Whether the word is a hesitation sound ("um", "Uh", "hmm"), a capitalized "Ah" included.
fn is_hesitation(word: &str) -> bool {
    let core = split(word).1;
    if core.is_empty() {
        return false;
    }
    let lower = core.to_lowercase();
    let mut chars = lower.chars();
    let capital: String = chars
        .next()
        .map(|c| c.to_uppercase().chain(chars).collect())
        .unwrap_or_default();
    if core != lower && core != capital {
        return false;
    }
    let mut runs: Vec<(char, usize)> = Vec::new();
    for c in lower.chars() {
        match runs.last_mut() {
            Some((last, n)) if *last == c => *n += 1,
            _ => runs.push((c, 1)),
        }
    }
    let letters: String = runs.iter().map(|r| r.0).collect();
    match letters.as_str() {
        "um" | "uh" | "uhm" | "erm" | "ah" | "hm" => true,
        "er" => runs == [('e', 1), ('r', 1)],
        "m" => runs[0].1 >= 2,
        _ => false,
    }
}

/// `word` in lower case without the punctuation at its ends, apostrophes kept.
pub(crate) fn norm(word: &str) -> String {
    word.to_lowercase()
        .trim_matches(|c: char| !(c.is_alphanumeric() || c == '_' || c == '\''))
        .to_string()
}

/// Whether `word` has a cased letter and every cased letter is a capital.
fn is_upper(word: &str) -> bool {
    word.chars().any(char::is_uppercase) && !word.chars().any(char::is_lowercase)
}

fn only(word: &str, allowed: impl Fn(char) -> bool) -> bool {
    !word.is_empty() && word.chars().all(allowed)
}

/// Lower-case letters and apostrophes.
fn is_plain(word: &str) -> bool {
    only(word, |c| c.is_ascii_lowercase() || c == '\'')
}

/// The words of a text as the rules read them.
struct Words<'a> {
    raw: &'a [&'a str],
    /// Each word normalized by [`norm`].
    n: Vec<String>,
    /// Each word in lower case, punctuation kept.
    lower: Vec<String>,
    hes: Vec<bool>,
}

impl<'a> Words<'a> {
    fn new(raw: &'a [&'a str]) -> Words<'a> {
        Words {
            raw,
            n: raw.iter().map(|w| norm(w)).collect(),
            lower: raw.iter().map(|w| w.to_lowercase()).collect(),
            hes: raw.iter().map(|w| is_hesitation(w)).collect(),
        }
    }

    fn len(&self) -> usize {
        self.raw.len()
    }

    fn ends_with(&self, i: usize, marks: &[char]) -> bool {
        self.raw[i].ends_with(marks)
    }

    /// Whether the word is plain, its lower case equal to its normal form.
    fn bare(&self, i: usize) -> bool {
        self.lower[i] == self.n[i]
    }
}

/// Which of `words` the rules delete.
pub(crate) fn deletions(words: &[&str]) -> Vec<bool> {
    apply(words, &RULES)
}

type Rule = fn(&Words) -> HashSet<usize>;

const RULES: [Rule; 9] = [
    cut_offs,
    interrupted_cut_offs,
    preposition_restarts,
    sure_edits,
    phrase_repeats,
    doubled,
    initial_markers,
    set_off_fillers,
    plain_openers,
];

fn apply(words: &[&str], rules: &[Rule]) -> Vec<bool> {
    let w = Words::new(words);
    let mut out = vec![false; words.len()];
    for rule in rules {
        for i in rule(&w) {
            out[i] = true;
        }
    }
    out
}

fn cut_offs(w: &Words) -> HashSet<usize> {
    let n = &w.n;
    let mut out = HashSet::new();
    for (i, a) in n.iter().enumerate() {
        if !is_plain(a) || w.hes[i] || w.ends_with(i, &PUNCT) {
            continue;
        }
        let mut j = i + 1;
        while j < n.len() && j <= i + LOOK && (w.hes[j] || n[j].is_empty()) {
            j += 1;
        }
        if j >= n.len() || j > i + LOOK {
            continue;
        }
        let b = &n[j];
        let cut = if a == "i" {
            I_CUT.contains(&b.as_str())
        } else {
            extends(a, b) && !has(WORDS, a)
        };
        if cut {
            out.insert(i);
        }
    }
    out
}

/// Whether `b` starts with `a` and goes on past it by more than a contraction's ending.
fn extends(a: &str, b: &str) -> bool {
    b.len() > a.len()
        && b.starts_with(a)
        && !b[a.len()..].starts_with(['\''])
        && !b[a.len()..].starts_with("n't")
}

fn interrupted_cut_offs(w: &Words) -> HashSet<usize> {
    let n = &w.n;
    let mut out = HashSet::new();
    for (i, a) in n.iter().enumerate() {
        if a.chars().count() > 4 || !is_plain(a) || has(WORDS, a) || w.hes[i] || is_upper(w.raw[i])
        {
            continue;
        }
        for (j, b) in n.iter().enumerate().take(i + LOOK + 1).skip(i + 1) {
            if !w.bare(j - 1) {
                break;
            }
            if j >= i + 2 && extends(a, b) && !w.hes[j] {
                out.insert(i);
                break;
            }
            if !is_restart(b) && b != a && !w.hes[j] {
                break;
            }
        }
    }
    out
}

fn preposition_restarts(w: &Words) -> HashSet<usize> {
    let n = &w.n;
    let mut out = HashSet::new();
    for i in 0..n.len().saturating_sub(3) {
        if has(PREPOSITIONS, &n[i])
            && has(DETERMINERS, &n[i + 1])
            && n[i + 2] == n[i]
            && has(DETERMINERS, &n[i + 3])
            && (i..i + 3).all(|j| w.bare(j))
        {
            out.extend([i, i + 1]);
        }
    }
    out
}

/// The explicit edits whose term is "sorry" set off by commas, or "or rather".
fn sure_edits(w: &Words) -> HashSet<usize> {
    let n = &w.n;
    let terms: [(&[&str], bool); 2] = [(&["sorry"], true), (&["or", "rather"], false)];
    let mut out = HashSet::new();
    for s in 1..n.len() {
        for (term, commas) in terms {
            let m = term.len();
            if s + m >= n.len() || !n[s..s + m].iter().zip(term).all(|(a, b)| a == b) {
                continue;
            }
            if commas && !(w.raw[s - 1].ends_with(',') && w.raw[s + m - 1].ends_with(',')) {
                continue;
            }
            let mut y_end = s + m;
            while y_end < n.len() - 1 && !w.ends_with(y_end, &PUNCT) {
                y_end += 1;
            }
            let y = &n[s + m..=y_end];
            let mut found: Option<std::ops::Range<usize>> = None;
            for size in 1..=3 {
                if size > s {
                    break;
                }
                let x = s - size..s;
                let blocked = (x.start..s - 1).any(|i| w.ends_with(i, &PUNCT))
                    || w.ends_with(s - 1, &SENTENCE_END)
                    || x.clone().any(|i| {
                        w.hes[i]
                            || n[i].is_empty()
                            || n[i] == term[0]
                            || BACKCHANNEL.contains(&n[i].as_str())
                    });
                if blocked {
                    break;
                }
                if n[x.start] == y[0] {
                    found = Some(x);
                    break;
                }
                if size == 1 && y.len() == 1 {
                    found = found.or(Some(x));
                }
            }
            if let Some(x) = found {
                out.extend(x);
                out.extend(s..s + m);
            }
        }
    }
    out
}

fn phrase_repeats(w: &Words) -> HashSet<usize> {
    let n = &w.n;
    let mut out = HashSet::new();
    let mut i = 0;
    while i + 3 < n.len() {
        let (a, b) = (&n[i], &n[i + 1]);
        let j = i + 2;
        let hit = !a.is_empty()
            && !b.is_empty()
            && (&n[j], &n[j + 1]) == (a, b)
            && ![i, i + 1, j]
                .iter()
                .any(|&x| w.hes[x] || w.ends_with(x, &PUNCT))
            && !w.ends_with(j + 1, &SENTENCE_END)
            && !(has(MEANT, a) && has(MEANT, b))
            && !SAID_TWICE.contains(&a.as_str())
            && !SAID_TWICE.contains(&b.as_str())
            && !BACKCHANNEL.contains(&a.as_str());
        if hit {
            out.extend([i, i + 1]);
            // The second copy may start the next repeat of a chain.
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

fn doubled(w: &Words) -> HashSet<usize> {
    let n = &w.n;
    let mut out = HashSet::new();
    for (i, a) in n.iter().enumerate() {
        let pronoun = a == "it" || a == "you";
        if !(is_doubled(a) || pronoun)
            || w.ends_with(i, &PUNCT)
            || (PARTICLES.contains(&a.as_str()) && i > 0 && OBJECTS.contains(&n[i - 1].as_str()))
        {
            continue;
        }
        if pronoun {
            // "it" and "you" double only where the first copy starts a clause.
            let before = (0..i).rev().find(|&k| !w.hes[k]);
            if let Some(k) = before
                && !w.ends_with(k, &PUNCT)
                && !has(CLAUSE_BEFORE, &n[k])
            {
                continue;
            }
        }
        let mut j = i + 1;
        while j < n.len() && j <= i + LOOK && w.hes[j] {
            j += 1;
        }
        if j < n.len() && j <= i + LOOK && n[j] == *a {
            out.insert(i);
        }
    }
    out
}

/// How many words the sentence-opening marker at `i` has, or 0.
fn marker_size(w: &Words, i: usize) -> usize {
    let n = &w.n;
    if is_initial_marker(&n[i]) && (w.lower[i] == n[i] || w.lower[i] == format!("{},", n[i])) {
        return 1;
    }
    if i + 1 < n.len()
        && n[i] == "i"
        && n[i + 1] == "mean"
        && w.lower[i] == "i"
        && (w.lower[i + 1] == "mean" || w.lower[i + 1] == "mean,")
    {
        return 2;
    }
    0
}

/// Whether a clause starts at `i`, seen before `stop`. A strict test, after "I mean", needs a
/// verb after a subject.
fn clause_at(w: &Words, i: usize, stop: usize, strict: bool) -> bool {
    let n = &w.n;
    if i + 1 >= stop || !w.bare(i) || w.hes[i + 1] {
        return false;
    }
    let (a, b) = (n[i].as_str(), n[i + 1].as_str());
    if is_clause_contraction(a) {
        return !b.is_empty() && !is_initial_marker(b);
    }
    if CLAUSE_SUBJECTS.contains(&a) {
        return if strict {
            is_clause_verb(b)
        } else {
            !b.is_empty() && !is_initial_marker(b) && !CLAUSE_SUBJECTS.contains(&b)
        };
    }
    if has(CLAUSE_AUX, a) {
        return CLAUSE_SUBJECTS.contains(&b) && w.lower[i + 1] == b;
    }
    if ["this", "that", "these", "those"].contains(&a) {
        return has(CLAUSE_AUX, b);
    }
    i + 2 < stop && has(DETERMINERS, a) && has(CLAUSE_AUX, &n[i + 2])
}

/// Whether word `i` starts a sentence, being the first or after a sentence end.
fn starts_sentence(w: &Words, i: usize) -> bool {
    i == 0 || w.ends_with(i - 1, &SENTENCE_END)
}

fn initial_markers(w: &Words) -> HashSet<usize> {
    let n = &w.n;
    let mut out = HashSet::new();
    for start in 0..n.len() {
        if !starts_sentence(w, start) {
            continue;
        }
        let mut i = start;
        while i < n.len() {
            if w.hes[i] && !w.ends_with(i, &SENTENCE_END) {
                i += 1;
                continue;
            }
            let size = marker_size(w, i);
            if size == 0 {
                break;
            }
            let stop = n.len().min(i + LOOK + 1);
            let mut j = i + size;
            while j < stop {
                if w.hes[j] && !w.ends_with(j, &SENTENCE_END) {
                    j += 1;
                    continue;
                }
                let next = marker_size(w, j);
                if next == 0 {
                    break;
                }
                j += next;
            }
            let lexical_so = n[i] == "so"
                && i + 1 < n.len()
                && (has(CLAUSE_AUX, &n[i + 1]) || n[i + 1] == "that" || n[i + 1] == "far");
            if !lexical_so && clause_at(w, j, stop, size == 2) {
                out.extend(i..i + size);
            }
            i += size;
        }
    }
    out
}

fn set_off_fillers(w: &Words) -> HashSet<usize> {
    let n = &w.n;
    let mut out = HashSet::new();
    let edge = |i: usize| w.raw[i].trim_end_matches(['"', '\'', ')']);
    let ends = |i: usize| edge(i).ends_with(SENTENCE_END);
    let comma = |i: usize| edge(i).ends_with([',', ';', ':']);
    if n.len() >= 2 {
        let last = n.len() - 1;
        let word = w.raw[last].trim_end_matches(['.', '?', '!', ',']);
        if only(word, |c| c.is_ascii_alphabetic() || c == '\'' || c == '-') && !ends(last - 1) {
            let prev = n[last - 1].as_str();
            if n[last] == "so" && !has(SO_ENDERS, prev) {
                out.insert(last);
                if (prev == "and" || prev == "but") && last >= 2 && !comma(last - 1) {
                    out.insert(last - 1);
                }
            } else if (n[last] == "yeah"
                && ["and", "but", "so"].contains(&prev)
                && last >= 2
                && !comma(last - 1)
                && !(prev == "so" && has(SO_ENDERS, &n[last - 2])))
                || (n[last] == "know"
                    && prev == "you"
                    && last >= 4
                    && !YOU_KNOW_LEAD.contains(&n[last - 2].as_str())
                    && !ends(last - 2))
            {
                out.extend([last - 1, last]);
            }
        }
    }
    for i in 1..n.len().saturating_sub(2) {
        let pair = (n[i].as_str(), n[i + 1].as_str());
        if (pair == ("you", "know") || pair == ("i", "mean"))
            && comma(i - 1)
            && comma(i + 1)
            && !ends(i - 1)
        {
            out.extend([i, i + 1]);
        }
    }
    out
}

fn plain_openers(w: &Words) -> HashSet<usize> {
    let n = &w.n;
    let taken = initial_markers(w);
    let is_marker = |j: usize| {
        is_initial_marker(&n[j]) && (w.lower[j] == n[j] || w.lower[j] == format!("{},", n[j]))
    };
    let mut out = HashSet::new();
    for start in 0..n.len() {
        if !starts_sentence(w, start) {
            continue;
        }
        let mut i = start;
        while i < n.len() && w.hes[i] && !w.ends_with(i, &SENTENCE_END) {
            i += 1;
        }
        if i >= n.len() || !is_marker(i) {
            continue;
        }
        let mut run = vec![i];
        let mut j = i + 1;
        while j < n.len() && j <= i + LOOK {
            if w.hes[j] && !w.ends_with(j, &SENTENCE_END) {
                j += 1;
            } else if is_marker(j) {
                run.push(j);
                j += 1;
            } else {
                break;
            }
        }
        let last = run[run.len() - 1];
        if w.ends_with(last, &SENTENCE_END) {
            continue;
        }
        let mut clause = 0;
        let mut k = last + 1;
        while k < n.len() && k <= i + LOOK {
            if w.hes[k] {
                if w.ends_with(k, &SENTENCE_END) {
                    break;
                }
                k += 1;
                continue;
            }
            let ordinary = only(&n[k], |c| c.is_ascii_lowercase() || c == '\'' || c == '-')
                && !is_initial_marker(&n[k])
                && !BACKCHANNEL.contains(&n[k].as_str())
                && !has(ANSWERS, &n[k]);
            if !ordinary {
                break;
            }
            clause += 1;
            if w.ends_with(k, &SENTENCE_END) {
                break;
            }
            k += 1;
        }
        if clause < 2 {
            continue;
        }
        for &m in &run {
            if taken.contains(&m) || !(n[m] == "so" || n[m] == "well") {
                continue;
            }
            let after = n.get(m + 1).map_or("", String::as_str);
            let lexical = (n[m] == "so" && (has(CLAUSE_AUX, after) || LEXICAL_SO.contains(&after)))
                || (n[m] == "well" && LEXICAL_WELL.contains(&after));
            if !lexical {
                out.insert(m);
            }
        }
    }
    out
}

/// The positions of every word in a run of two or more adjacent copies of a span of one to
/// eight words, hesitations between them skipped.
pub(crate) fn copies(words: &[&str]) -> HashSet<usize> {
    let w = Words::new(words);
    let pos: Vec<usize> = (0..w.len())
        .filter(|&i| !w.hes[i] && !w.n[i].is_empty())
        .collect();
    let seq: Vec<&str> = pos.iter().map(|&i| w.n[i].as_str()).collect();
    let mut out = HashSet::new();
    for k in 1..=8 {
        let mut i = 0;
        while i + 2 * k <= seq.len() {
            let mut c = 1;
            while i + (c + 1) * k <= seq.len() && seq[i + c * k..i + (c + 1) * k] == seq[i..i + k] {
                c += 1;
            }
            if c >= 2 {
                out.extend(pos[i..i + c * k].iter().copied());
                i += c * k;
            } else {
                i += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fillers::{Fillers, Protected};

    /// `text` after the filler rules and `rules`, tidied as the filler rules tidy.
    fn cleaned(text: &str, rules: &[Rule]) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let fillers = Fillers::for_language("en").unwrap();
        let protected = Protected::default();
        let mut deleted = fillers.deletions(&words, &protected);
        for (d, r) in deleted.iter_mut().zip(apply(&words, rules)) {
            *d |= r;
        }
        Fillers::text(
            &words,
            &fillers.tidy_deletions(&words, &deleted, &protected),
        )
    }

    fn check(cases: &[(&str, &str)], rules: &[Rule]) {
        for (text, want) in cases {
            assert_eq!(cleaned(text, rules), *want, "{text:?}");
        }
    }

    #[test]
    fn cut_offs_restarts_repeats_and_doubles_go() {
        check(
            &[
                ("I want the red, sorry, the blue one", "I want the blue one"),
                ("the blue or rather the green one", "the green one"),
                ("Oh, yeah, sorry, yeah then.", "Oh, yeah, sorry, yeah then."),
                ("It's three, actually, four.", "It's three, actually, four."),
                (
                    "Can you check the logs the logs please",
                    "Can you check the logs please",
                ),
                ("it has to has to fulfil", "it has to fulfil"),
                ("I think I think we will", "I think we will"),
                (
                    "the case the case the surrounding",
                    "the case the surrounding",
                ),
                ("we can we can we can do it", "we can do it"),
                (
                    "Let's see, let's see, let's give it",
                    "Let's see, let's see, let's give it",
                ),
                ("it is. It is useful", "it is. It is useful"),
                ("we used we used.", "we used we used."),
                ("such and such and look", "such and such and look"),
                (
                    "the mummy please mummy please",
                    "the mummy please mummy please",
                ),
                ("yeah probably yeah probably", "yeah probably yeah probably"),
                ("no no no no", "no"),
                ("very very very very", "very very very very"),
                ("it was uh it was good", "it was it was good"),
                (
                    "Um yeah so des design a new remote control",
                    "Yeah so design a new remote control",
                ),
                ("the mi mic microphone is small", "the microphone is small"),
                ("it' it's fine", "it's fine"),
                ("th uh the button", "the button"),
                ("i it's a good idea", "it's a good idea"),
                ("moving on to compon components", "moving on to components"),
                ("we were done", "we were done"),
                ("it would be better", "it would be better"),
                ("use user settings", "use user settings"),
                ("I think that that's right", "I think that that's right"),
                ("I imagine so", "I imagine so"),
                ("check the mic", "check the mic"),
                ("about 2 20 people", "about 2 20 people"),
                ("pr, products", "pr, products"),
                ("des um uh uh design", "des design"),
                ("on the on the screen", "on the screen"),
                ("you would have to to make it", "you would have to make it"),
                ("we need some uh some help", "we need some help"),
                ("with um uh um with", "with with"),
                ("hand them in in a week", "hand them in in a week"),
                ("chew it it shouldn't", "chew it it shouldn't"),
                ("we dwell on on the look", "we dwell on the look"),
                ("is that you all all finished", "is that you all finished"),
                ("put it on on the table", "put it on on the table"),
                ("and it it was fine", "and it was fine"),
                ("can you you send it", "can you send it"),
                ("It it can become", "It can become"),
                (
                    "you can press it it should be",
                    "you can press it it should be",
                ),
                ("I think you uh you should", "I think you should"),
                (
                    "close to you you probably know",
                    "close to you you probably know",
                ),
                ("it's for, for you", "it's for, for you"),
                ("no no that's not it", "no that's not it"),
                ("No, no, don't send it yet.", "No, no, don't send it yet."),
                ("in our in your folder", "in your folder"),
                ("to the to a shop", "to a shop"),
                ("on the on the on the screen", "on the screen"),
                ("I put it on the on button", "I put it on the on button"),
                ("in the in group", "in the in group"),
                ("aim for the under fifties", "aim for the under fifties"),
                ("on the, on the screen", "on the, on the screen"),
                ("on 'the' on the screen", "on 'the' on the screen"),
                ("on the uh on the screen", "on the on the screen"),
                ("th I think the button", "I think the button"),
                ("loo the look", "the look"),
                ("shou we should go", "we should go"),
                ("tha tha that works", "that works"),
                ("PC in PCI slots", "PC in PCI slots"),
                ("pc and pci", "pc and pci"),
                ("Al met Alex", "Al met Alex"),
                ("ex marketing expert", "ex marketing expert"),
                ("des is not really design", "des is not really design"),
                ("th I, think the button", "th I, think the button"),
                ("whole I wholemeal", "whole I wholemeal"),
                ("we I were", "we I were"),
                ("do I don't", "do I don't"),
            ],
            &RULES[..6],
        );
    }

    #[test]
    fn sentence_opening_markers_go_before_a_clause() {
        check(
            &[
                ("So I think we should stop.", "I think we should stop."),
                (
                    "Well, we need a smaller button.",
                    "We need a smaller button.",
                ),
                (
                    "I mean, we need a smaller button.",
                    "We need a smaller button.",
                ),
                ("Yeah, I think we should stop.", "I think we should stop."),
                ("Okay so it's working.", "It's working."),
                ("Um well we need help.", "We need help."),
                ("Oh, that's useful.", "That's useful."),
                ("Right, the button is broken.", "The button is broken."),
                ("Anyway, it works.", "It works."),
                (
                    "It works. Well, we need more tests.",
                    "It works. We need more tests.",
                ),
                ("So that it works.", "So that it works."),
                ("So do I.", "So do I."),
                ("So far it works.", "So far it works."),
                ("Well done.", "Well done."),
                ("The right one works.", "The right one works."),
                ("Right now we need help.", "Right now we need help."),
                ("I mean it now.", "I mean it now."),
                ("I mean you personally.", "I mean you personally."),
                ("I mean the word maybe.", "I mean the word maybe."),
                ("'So' I think is a marker.", "'So' I think is a marker."),
                (
                    "I mean 'I think' as a phrase.",
                    "I mean 'I think' as a phrase.",
                ),
                ("Okay. I think it works.", "Okay. I think it works."),
                ("Well um. We need help.", "Well. We need help."),
                ("Yeah.", "Yeah."),
                ("Okay", "Okay"),
                ("Yeah okay", "Yeah okay"),
                ("Well I", "Well I"),
                ("Well um uh we need help.", "Well we need help."),
            ],
            &RULES,
        );
    }

    #[test]
    fn set_off_fillers_and_plain_openers_go() {
        check(
            &[
                ("I don't have a red pen so.", "I don't have a red pen."),
                ("we have forty minutes, so.", "we have forty minutes."),
                ("Alright, so.", "Alright."),
                ("So.", "So."),
                ("I think so.", "I think so."),
                ("Hope so.", "Hope so."),
                ("copy the page or so.", "copy the page or so."),
                ("it works, so that's fine.", "it works, so that's fine."),
                ("The shape is okay but yeah.", "The shape is okay."),
                ("we can do that so yeah.", "we can do that."),
                ("it was fine, so yeah.", "it was fine."),
                ("And yeah.", "And yeah."),
                ("Yeah, but yeah.", "Yeah."),
                ("I think so yeah.", "I think so yeah."),
                ("it's hard to say, you know.", "it's hard to say."),
                ("a wheel you know.", "a wheel you know."),
                ("I'll let you know.", "I'll let you know."),
                ("by now, do you know?", "by now, do you know?"),
                ("we need, you know, a way.", "we need, a way."),
                ("It's hot, I mean, it's a mango.", "It's hot, it's a mango."),
                ("I mean it.", "I mean it."),
                ("you know what I mean?", "you know what I mean?"),
                ("done. So.", "done. So."),
                (
                    "So the method is the following.",
                    "The method is the following.",
                ),
                (
                    "So, what does the management say?",
                    "What does the management say?",
                ),
                (
                    "Well, if it's going to be late.",
                    "If it's going to be late.",
                ),
                ("Okay, so each unit has one.", "Okay, each unit has one."),
                ("So um the design step will be.", "The design step will be."),
                ("So that it will record.", "So that it will record."),
                ("So much for that.", "So much for that."),
                ("So do I.", "So do I."),
                (
                    "Well done with the management.",
                    "Well done with the management.",
                ),
                ("Well within budget.", "Well within budget."),
                ("So no.", "So no."),
                ("Well maybe.", "Well maybe."),
                ("So six.", "So six."),
                ("So.", "So."),
                (
                    "Yeah, well, for a remote control it works.",
                    "Yeah, for a remote control it works.",
                ),
                (
                    "fine. So the material is rubber.",
                    "fine. The material is rubber.",
                ),
                (
                    "It's fine so the material is rubber.",
                    "It's fine so the material is rubber.",
                ),
            ],
            &RULES,
        );
        assert!(deletions(&[]).is_empty());
        assert_eq!(deletions(&["so"]), [false]);
    }

    #[test]
    fn copies_are_runs_of_any_span_hesitations_skipped() {
        let words: Vec<&str> = "the the uh the plan we can we can go".split(' ').collect();
        let mut got: Vec<usize> = copies(&words).into_iter().collect();
        got.sort();
        assert_eq!(got, [0, 1, 3, 5, 6, 7, 8]);
        assert!(copies(&[]).is_empty());
    }

    #[test]
    fn a_capitalized_ah_is_a_hesitation_here() {
        assert!(is_hesitation("Ah,") && is_hesitation("umm") && is_hesitation("Er."));
        assert!(!is_hesitation("err") && !is_hesitation("UM") && !is_hesitation("uhh-huh"));
    }
}
