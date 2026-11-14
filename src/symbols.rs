//! Spoken symbols written as characters inside runs of words that read as code, such as "engine
//! dot rs" for engine.rs, while "a dot on the map" stays as said.
//!
//! A run is words and spoken symbols in turn ("source slash engine dot rs"), which may start
//! with symbols ("dash dash force"). In prose, a function word beside a symbol ends a run there,
//! and a run is written as one word only when something shows it's code: a listed path, a file
//! extension or a domain after a dot, the shape of a flag or of a path from the home or current
//! folder, two symbols or more, a symbol name rare in prose (underscore, hyphen), a word the
//! model already wrote as code, or a word beside the run that names a file, a folder or a flag.

use std::ops::Range;

use crate::paths::{Paths, Snap};
use crate::tokenizer::Tokenizer;

/// Writes spoken symbols ("slash", "dot", "underscore") as characters inside runs of words that
/// read as code, such as a path, a file name, an identifier, a flag or a URL, and leaves them
/// as words in prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Symbols {
    code: bool,
}

impl Symbols {
    /// The spoken symbols of a language given as an ISO 639-1 code, with or without a region,
    /// or none if Telmo has none for it. English is the only one so far.
    pub fn for_language(code: &str) -> Option<Symbols> {
        let language = code.split(['-', '_']).next()?.to_ascii_lowercase();
        (language == "en").then_some(Symbols { code: false })
    }

    /// The same symbols for text that goes into a terminal or a code editor, where every run of
    /// words joined by spoken symbols is code, as in "self dot name" for self.name.
    pub fn for_code(self) -> Symbols {
        Symbols { code: true }
    }

    /// Whether every run is taken for code.
    pub fn is_for_code(&self) -> bool {
        self.code
    }

    /// `text` with the spoken symbols of its code-like runs written as characters, and each run
    /// that spells one of `paths` written as that path, its words separated by single spaces.
    pub fn write(&self, text: &str, paths: Option<&Paths>) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let joins = self.plan(&words, false, paths);
        let mut out: Vec<&str> = Vec::with_capacity(words.len());
        let mut next = 0;
        for join in &joins {
            out.extend(&words[next..join.words.start]);
            out.push(&join.text);
            next = join.words.end;
        }
        out.extend(&words[next..]);
        out.join(" ")
    }

    /// The runs of `words` to write as one word each, in order. With `hesitations_go`, a
    /// hesitation ("um") inside a run goes with it.
    pub(crate) fn plan(
        &self,
        words: &[&str],
        hesitations_go: bool,
        paths: Option<&Paths>,
    ) -> Vec<Join> {
        let snaps = paths.map_or_else(Vec::new, |p| p.snap(words, hesitations_go));
        let els = elements(words, hesitations_go, &snaps);
        let mut joins = Vec::new();
        let mut joined = vec![false; els.len()];
        for run in candidates(&els) {
            for part in self.split(&els, &run, words) {
                if !(self.code || evidence(&els, &part, words)) {
                    continue;
                }
                if let Some(join) = write_run(&els, &part, words) {
                    let first = part.prefix.as_ref().map_or(part.atoms[0], |p| p.els.start);
                    let last = *part.atoms.last().expect("a run has an atom");
                    joined[first..=last].fill(true);
                    joins.push(join);
                }
            }
        }
        // A path said whole is a run of its own.
        for (k, e) in els.iter().enumerate() {
            if e.snapped && !joined[k] {
                let (lead, _, _) = split_word(words[e.words.start]);
                let (_, _, trail) = split_word(words[e.words.end - 1]);
                joins.push(Join {
                    words: e.words.clone(),
                    text: format!("{lead}{}{trail}", e.text),
                });
            }
        }
        joins.sort_by_key(|j| j.words.start);
        joins
    }

    /// The first of `words` that may still join the words after the last of them, if a run with
    /// a symbol in it could still go on past the last, or none. With `hesitations_go`, a
    /// hesitation inside the run doesn't end it.
    pub(crate) fn open_from(&self, words: &[&str], hesitations_go: bool) -> Option<usize> {
        let els = elements(words, hesitations_go, &[]);
        let last = els.last()?;
        let open = matches!(last.kind, Kind::Symbol(..) | Kind::Hesitation)
            || last.is_atom() && last.open_after;
        if !open {
            return None;
        }
        let mut k = els.len() - 1;
        while k > 0 {
            let before = &els[k - 1];
            let joinable = before.open_after
                && els[k].open_before
                && matches!(
                    (&before.kind, &els[k].kind),
                    (Kind::Symbol(..), _)
                        | (_, Kind::Symbol(..))
                        | (Kind::Hesitation, _)
                        | (_, Kind::Hesitation)
                );
            if !joinable {
                break;
            }
            k -= 1;
        }
        // Words without a symbol among them wait for nothing, since the next pass shows one.
        let symbol = els[k..].iter().any(|e| matches!(e.kind, Kind::Symbol(..)));
        symbol.then_some(els[k].words.start)
    }

    /// The parts of `run` that may join. A word of code that starts with symbols ("dash dash
    /// force", "dash v", "tilde slash") starts a part of its own, and in prose a function word
    /// beside a symbol ("a dot on", "and slash or") ends one, the symbol staying a word or
    /// starting the next part.
    fn split(&self, els: &[El], run: &Run, words: &[&str]) -> Vec<Run> {
        let function = |a: usize| !self.code && els[a].is_function();
        // A single letter after a dash is a flag before a word naming one, even "a" or "i"
        // ("the dash i command parameter").
        let flag_letter = |group: &Group, a: usize| {
            let e = &els[a];
            group.chars == "-"
                && e.text.chars().count() == 1
                && e.open_after
                && (words.get(e.words.end)).is_some_and(|w| listed(FLAG_CUES, &bare(w)))
        };
        let strong = run.groups.iter().filter(|g| g.strong).count();
        let mut parts = Vec::new();
        let mut prefix = (run.prefix.clone())
            .filter(|p| flag_letter(p, run.atoms[0]) || !function(run.atoms[0]));
        let mut atoms = vec![run.atoms[0]];
        let mut groups: Vec<Group> = Vec::new();
        for (k, group) in run.groups.iter().enumerate() {
            let (left, right) = (run.atoms[k], run.atoms[k + 1]);
            let in_flag = prefix.as_ref().is_some_and(Group::starts_flag) && group.chars == "-";
            // Two names rare in prose show an identifier, function words and all ("get
            // underscore or underscore insert").
            let blocked = (function(left) || function(right)) && !(group.strong && strong >= 2);
            if group.starts_token(&els[right]) || !in_flag && blocked {
                parts.push(Run {
                    prefix: prefix.take(),
                    atoms: std::mem::take(&mut atoms),
                    groups: std::mem::take(&mut groups),
                });
                prefix = (flag_letter(group, right) || !function(right)).then(|| group.clone());
            } else {
                groups.push(group.clone());
            }
            atoms.push(right);
        }
        parts.push(Run {
            prefix,
            atoms,
            groups,
        });
        (parts.into_iter())
            .filter(|p| p.prefix.is_some() || !p.groups.is_empty())
            .collect()
    }
}

/// A run of words and the one word it's written as.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Join {
    pub words: Range<usize>,
    pub text: String,
}

/// `tokens` with each run that `plan` gives, from the texts of their words, written as one word
/// on the frames of its words, as `respell` places the new tokens. A run the vocabulary can't
/// spell stays as it was.
pub(crate) fn rewrite<T: Clone>(
    tokenizer: &Tokenizer,
    tokens: Vec<T>,
    id: impl Fn(&T) -> u32,
    plan: impl FnOnce(&[&str]) -> Vec<Join>,
    respell: impl Fn(&[T], Vec<u32>) -> Vec<T>,
) -> Vec<T> {
    let ids: Vec<u32> = tokens.iter().map(&id).collect();
    let spans = crate::fillers::words(tokenizer, &ids);
    let texts: Vec<String> = (spans.iter())
        .map(|s| {
            tokenizer
                .decode_raw(&ids[s.clone()])
                .trim_start()
                .to_string()
        })
        .collect();
    let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
    let joins = plan(&texts);
    if joins.is_empty() {
        return tokens;
    }
    let mut out = Vec::with_capacity(tokens.len());
    let mut next = 0;
    for join in joins {
        let (first, last) = (spans[join.words.start].start, spans[join.words.end - 1].end);
        out.extend_from_slice(&tokens[next..first]);
        // The first word of a transcript may have no word-start marker.
        let start = tokenizer
            .decode_raw(&ids[first..first + 1])
            .starts_with(' ')
            || first > 0;
        match crate::fillers::spelled(tokenizer, &join.text, start) {
            Some(new) => out.extend(respell(&tokens[first..last], new)),
            None => out.extend_from_slice(&tokens[first..last]),
        }
        next = last;
    }
    out.extend_from_slice(&tokens[next..]);
    out
}

/// Whether `word`, as written in a transcript, reads as code: a path, a file name, an
/// identifier with an underscore, a flag or a module path, which the cleanup keeps.
pub(crate) fn is_code_word(word: &str) -> bool {
    let (_, core, _) = split_word(word);
    let letter = |s: &str| s.chars().next().is_some_and(char::is_alphanumeric);
    reads_as_code(core)
        || core.strip_prefix("--").is_some_and(letter)
        || (core.strip_prefix('-')).is_some_and(|r| r.chars().count() == 1 && letter(r))
        || core
            .strip_prefix('.')
            .is_some_and(|r| r.len() >= 2 && letter(r))
        || core.starts_with("~/")
}

/// Symbol names, the characters they stand for, and whether a name is rare enough in prose that
/// one between two words makes code ("underscore", "hyphen").
const NAMES: [(&str, &str, bool); 8] = [
    ("slash", "/", false),
    ("backslash", "\\", true),
    ("dot", ".", false),
    ("underscore", "_", true),
    ("dash", "-", false),
    ("hyphen", "-", true),
    ("colon", ":", false),
    ("tilde", "~", false),
];

/// Words before a symbol name that change it, as in "forward slash" and "double colon", with
/// how many times the symbol counts and the only name they go with, if one.
const MODIFIERS: [(&str, usize, Option<&str>); 4] = [
    ("forward", 1, Some("slash")),
    ("back", 1, Some("slash")),
    ("double", 2, None),
    ("triple", 3, None),
];

/// Function words, which leave a symbol beside them as prose ("a dot on", "and slash or", "dash
/// of salt").
const FUNCTION: &str = "a an the this that these those my your his her its our their me him us
them i you he she it we they and or but nor so yet if then than because as of in on at to for
from by with without into onto over under about above below between through during before after
against among within upon off up down out is are was were be been being am do does did have has
had will would shall should can could may might must not no it's that's i'm you're we're they're
he's she's i'd you'd he'd she'd we'd they'd i'll you'll we'll they'll i've you've we've they've
let's there here what which who whom whose when where why how all any some each every both either
neither such very just also too only even";

/// File extensions, which make code of a dot before them ("engine dot rs").
const EXTENSIONS: &str = "rs py js ts tsx jsx mjs cjs md mdx toml json jsonl yaml yml txt sh bash
zsh fish ps1 bat c h cc cpp cxx hpp hh go java kt kts swift rb php pl lua zig dart vue svelte astro
html htm css scss sass less sql lock xml csv tsv log ini cfg conf env wav mp3 flac ogg m4a mp4 mov
webm png jpg jpeg gif svg webp ico pdf zip tar gz tgz xz zst bz2 ipynb proto wasm so dll dylib exe
bin gradle cmake mk tf hcl nix vim patch diff pem crt gguf safetensors onnx pack";

/// Top-level domains, which make code of a dot before them between words ("example dot com").
const DOMAINS: &str = "com org net io dev ai app co uk us edu gov me xyz info biz tv fm ly gg";

/// Names written after a leading dot ("dot gitignore").
const DOTFILES: &str = "gitignore gitattributes gitmodules github git env envrc vscode cargo
config ssh bashrc zshrc profile npmrc nvmrc editorconfig dockerignore prettierrc eslintrc cache
local";

/// Words beside a run that say it names a file, a folder or a flag ("the source slash engine
/// file").
const CUES: &str = "file files folder folders directory directories dir path paths module
crate package repo repository url link domain website site address branch flag flags option
options parameter parameters argument arguments switch extension script command";

/// Words after a flag that say it is one ("the dash i command parameter").
const FLAG_CUES: &str = "flag flags option options parameter parameters argument arguments switch
command";

/// Numbers said as words, which a joined run writes as digits.
const NUMBERS: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];

/// Quote marks and brackets, which a run keeps around it.
const QUOTES: [char; 14] = [
    '"', '\'', '“', '”', '„', '‟', '‘', '’', '‚', '‛', '«', '»', '‹', '›',
];
const BRACKETS: [char; 6] = ['(', ')', '[', ']', '{', '}'];

fn listed(list: &str, word: &str) -> bool {
    list.split_whitespace().any(|w| w == word)
}

/// A word's core in lower case.
fn bare(word: &str) -> String {
    split_word(word).1.to_lowercase()
}

/// `word`'s quotes and brackets before, its core, and its quotes, brackets and sentence
/// punctuation after.
pub(crate) fn split_word(word: &str) -> (&str, &str, &str) {
    let lead = |c: char| QUOTES.contains(&c) || BRACKETS.contains(&c);
    let trail = |c: char| lead(c) || matches!(c, '.' | ',' | ';' | ':' | '?' | '!');
    let start = word.len() - word.trim_start_matches(lead).len();
    let rest = &word[start..];
    let core = rest.trim_end_matches(trail);
    (&word[..start], core, &rest[core.len()..])
}

/// The characters of a symbol name, and whether it's rare in prose.
fn symbol(bare: &str) -> Option<(&'static str, bool)> {
    (NAMES.iter())
        .find(|(name, _, _)| *name == bare)
        .map(|&(_, chars, strong)| (chars, strong))
}

/// The spoken symbol that starts at word `i` of `words`, with its characters and the words it
/// takes, one or two ("forward slash", "double colon").
pub(crate) fn spoken_symbol(words: &[&str], i: usize) -> Option<(String, usize)> {
    spoken_symbol_strength(words, i).map(|(chars, _, taken)| (chars, taken))
}

/// The same, with whether the symbol is rare in prose.
fn spoken_symbol_strength(words: &[&str], i: usize) -> Option<(String, bool, usize)> {
    let (_, core, trail) = split_word(words[i]);
    let name = core.to_lowercase();
    if trail.is_empty()
        && let Some(next) = words.get(i + 1)
        && let Some(&(_, times, only)) = MODIFIERS.iter().find(|m| m.0 == name)
    {
        let (lead, core, _) = split_word(next);
        let next = core.to_lowercase();
        if lead.is_empty()
            && only.is_none_or(|o| o == next)
            && let Some((chars, strong)) = symbol(&next)
        {
            let chars = chars.repeat(times);
            let strong = strong || chars == "::";
            return Some((chars, strong, 2));
        }
    }
    symbol(&name).map(|(chars, strong)| (chars.to_string(), strong, 1))
}

/// Whether a word is a hesitation sound, which goes with the run it's in.
pub(crate) fn is_hesitation(core: &str) -> bool {
    crate::fillers::is_hesitation(core, None)
}

/// A piece of a word written as code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Part<'a> {
    Word(&'a str),
    Symbol(String),
}

/// A word's core split at the characters of code ("api.py" into api, ".", py). A word a model
/// wrote with hyphens between symbol names ("dash-dash", "dot-com") splits into the names'
/// characters and the other words.
pub(crate) fn split_code(core: &str) -> Vec<Part<'_>> {
    let hyphened: Vec<&str> = core.split('-').collect();
    if hyphened.len() > 1
        && hyphened.iter().all(|p| !p.is_empty())
        && hyphened.iter().any(|p| symbol(&p.to_lowercase()).is_some())
    {
        return (hyphened.into_iter())
            .flat_map(|p| match symbol(&p.to_lowercase()) {
                Some((chars, _)) => vec![Part::Symbol(chars.to_string())],
                None => split_code(p),
            })
            .collect();
    }
    let code = |c: char| matches!(c, '/' | '.' | '_' | '-' | ':' | '\\' | '~');
    let mut out = Vec::new();
    let mut start = 0;
    let mut symbol = String::new();
    for (i, c) in core.char_indices() {
        if code(c) {
            if start < i {
                out.push(Part::Word(&core[start..i]));
            }
            symbol.push(c);
            start = i + c.len_utf8();
        } else if !symbol.is_empty() {
            out.push(Part::Symbol(std::mem::take(&mut symbol)));
        }
    }
    if start < core.len() {
        out.push(Part::Word(&core[start..]));
    }
    if !symbol.is_empty() {
        out.push(Part::Symbol(symbol));
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Kind {
    /// A word with a letter or digit.
    Atom,
    /// A spoken symbol, with its characters and whether it's rare in prose.
    Symbol(String, bool),
    /// A hesitation that goes with the run it's in.
    Hesitation,
    /// Punctuation alone, which ends a run.
    Other,
}

/// A word, a piece of one ("dash" of "dash-dash"), or words read as one ("forward slash", "S R
/// C", a snapped path).
#[derive(Debug, Clone)]
struct El {
    kind: Kind,
    /// The words it takes, and whether it starts the first and ends the last.
    words: Range<usize>,
    starts_word: bool,
    ends_word: bool,
    /// The core as written, for an atom.
    text: String,
    /// Whether a run may go on before it and after it, which punctuation stops.
    open_before: bool,
    open_after: bool,
    /// Whether it's single letters spelled out ("S R C").
    letters: bool,
    /// Whether it's a listed path, written as `text`.
    snapped: bool,
}

impl El {
    fn bare(&self) -> String {
        self.text.to_lowercase()
    }

    /// Whether it's a function word or a hesitation kept as a word, either of which leaves a
    /// symbol beside it as prose.
    fn is_function(&self) -> bool {
        self.kind == Kind::Atom
            && !self.letters
            && !self.snapped
            && (listed(FUNCTION, &self.bare()) || is_hesitation(&self.text))
    }

    fn is_atom(&self) -> bool {
        self.kind == Kind::Atom
    }
}

/// The elements of `words`, with each of `snaps` as one atom.
fn elements(words: &[&str], hesitations_go: bool, snaps: &[Snap]) -> Vec<El> {
    let mut out: Vec<El> = Vec::new();
    let mut snaps = snaps.iter().peekable();
    let mut i = 0;
    while i < words.len() {
        let (lead, core, trail) = split_word(words[i]);
        let el = |kind, text: &str, words: Range<usize>, open_before, open_after| El {
            kind,
            words,
            starts_word: true,
            ends_word: true,
            text: text.to_string(),
            open_before,
            open_after,
            letters: false,
            snapped: false,
        };
        if let Some(snap) = snaps.next_if(|s| s.words.start == i) {
            let (_, _, trail) = split_word(words[snap.words.end - 1]);
            out.push(El {
                snapped: true,
                ..el(
                    Kind::Atom,
                    &snap.text,
                    snap.words.clone(),
                    lead.is_empty(),
                    trail.is_empty(),
                )
            });
            i = snap.words.end;
            continue;
        }
        if let Some((chars, strong, taken)) = spoken_symbol_strength(words, i) {
            let (_, _, trail) = split_word(words[i + taken - 1]);
            out.push(el(
                Kind::Symbol(chars, strong),
                "",
                i..i + taken,
                lead.is_empty(),
                trail.is_empty(),
            ));
            i += taken;
            continue;
        }
        let parts = split_code(core);
        let named = |c: &str| symbol_name_part(core, c);
        if parts.len() > 1
            && parts
                .iter()
                .any(|p| matches!(p, Part::Symbol(c) if named(c)))
        {
            // A model writes "dash-dash" or "dot-com" as one word.
            let n = parts.len();
            for (k, p) in parts.into_iter().enumerate() {
                let (first, last) = (k == 0, k + 1 == n);
                let (kind, text) = match p {
                    Part::Symbol(chars) => {
                        let strong = chars == "_";
                        (Kind::Symbol(chars, strong), String::new())
                    }
                    Part::Word(w) => (Kind::Atom, w.to_string()),
                };
                let open = (!first || lead.is_empty(), !last || trail.is_empty());
                out.push(El {
                    starts_word: first,
                    ends_word: last,
                    ..el(kind, &text, i..i + 1, open.0, open.1)
                });
            }
        } else if hesitations_go && lead.is_empty() && is_hesitation(core) {
            out.push(el(Kind::Hesitation, core, i..i + 1, true, trail.is_empty()));
        } else if core.chars().any(char::is_alphanumeric) {
            out.push(el(
                Kind::Atom,
                core,
                i..i + 1,
                lead.is_empty(),
                trail.is_empty(),
            ));
        } else {
            out.push(el(Kind::Other, core, i..i + 1, false, false));
        }
        i += 1;
    }
    merge_letters(out)
}

/// Whether `core` split into `chars` because of a symbol name inside it ("dash-dash"), rather
/// than code the model wrote ("api.py"), which stays one atom.
fn symbol_name_part(core: &str, chars: &str) -> bool {
    (core.split('-')).any(|p| symbol(&p.to_lowercase()).is_some_and(|(c, _)| c == chars))
}

/// Single letters said in a row ("S R C") read as one atom.
fn merge_letters(els: Vec<El>) -> Vec<El> {
    let single = |e: &El| {
        e.kind == Kind::Atom
            && !e.snapped
            && e.starts_word
            && e.ends_word
            && e.text.chars().count() == 1
            && e.text.chars().all(|c| c.is_ascii_alphabetic())
    };
    let mut out: Vec<El> = Vec::new();
    for e in els {
        if let Some(last) = out.last_mut()
            && single(&e)
            && (last.letters || single(last))
            && last.open_after
            && e.open_before
        {
            last.text.push_str(&e.text);
            last.words.end = e.words.end;
            last.open_after = e.open_after;
            last.letters = true;
            continue;
        }
        out.push(e);
    }
    out
}

/// Symbols said together, as one joint ("dash dash", "colon slash slash").
#[derive(Debug, Clone, PartialEq, Eq)]
struct Group {
    chars: String,
    /// Whether a name rare in prose is among them.
    strong: bool,
    /// The elements it takes.
    els: Range<usize>,
}

impl Group {
    /// Whether it starts a word of code before `next`, which never joins the word before, as a
    /// flag does ("dash dash force", "dash v") and a path from the home, current or parent folder
    /// ("tilde slash", "dot slash", "dot dot slash").
    fn starts_token(&self, next: &El) -> bool {
        let c = self.chars.as_str();
        c.starts_with("--")
            || c == "-" && next.text.chars().count() == 1 && !next.letters && !next.snapped
            || c.starts_with('~')
            || c.starts_with("./")
            || c.starts_with("../")
    }

    /// Whether it starts a long flag, inside which words join whatever they are ("dash dash
    /// force dash with dash lease").
    fn starts_flag(&self) -> bool {
        self.chars.starts_with("--")
    }
}

/// Words and symbols said in turn, an optional group of symbols first: [group] atom (group
/// atom)...
#[derive(Debug, Clone)]
struct Run {
    prefix: Option<Group>,
    /// The elements of its atoms.
    atoms: Vec<usize>,
    /// The groups between them.
    groups: Vec<Group>,
}

/// The longest runs of atoms and symbols in turn with at least one symbol, where nothing but
/// hesitations comes between and punctuation stops them.
fn candidates(els: &[El]) -> Vec<Run> {
    let mut runs = Vec::new();
    let mut i = 0;
    while i < els.len() {
        let mut j = i;
        let prefix = group_at(els, &mut j);
        if (prefix.as_ref()).is_some_and(|g| !els[g.els.end - 1].open_after) {
            i = j.max(i + 1);
            continue;
        }
        let Some(first) = atom_at(els, &mut j, prefix.is_some()) else {
            i = if prefix.is_some() { j } else { i + 1 };
            continue;
        };
        let mut run = Run {
            prefix,
            atoms: vec![first],
            groups: Vec::new(),
        };
        loop {
            let last = *run.atoms.last().expect("a run has an atom");
            if !els[last].open_after {
                break;
            }
            let mut k = last + 1;
            let Some(group) = group_at(els, &mut k) else {
                break;
            };
            if !els[group.els.end - 1].open_after {
                break;
            }
            let Some(next) = atom_at(els, &mut k, true) else {
                break;
            };
            run.groups.push(group);
            run.atoms.push(next);
        }
        let end = *run.atoms.last().expect("a run has an atom") + 1;
        if run.prefix.is_some() || !run.groups.is_empty() || els[first].snapped {
            runs.push(run);
        }
        i = end;
    }
    runs
}

/// The group of symbols at `*i`, past hesitations between them, moving `*i` after it.
fn group_at(els: &[El], i: &mut usize) -> Option<Group> {
    let mut k = *i;
    while k < els.len() && els[k].kind == Kind::Hesitation && els[k].open_after {
        k += 1;
    }
    let start = k;
    let mut chars = String::new();
    let mut strong = false;
    let mut end = k;
    while k < els.len() {
        match &els[k].kind {
            Kind::Symbol(c, s) => {
                if k > start && !(els[k - 1].open_after && els[k].open_before) {
                    break;
                }
                chars.push_str(c);
                strong |= *s;
                k += 1;
                end = k;
            }
            Kind::Hesitation if els[k].open_after => k += 1,
            _ => break,
        }
    }
    if chars.is_empty() || !els[start].open_before && start != *i {
        return None;
    }
    *i = end;
    // Two colons together are a path separator in code ("std colon colon fs").
    strong |= chars.contains("::");
    Some(Group {
        chars,
        strong,
        els: start..end,
    })
}

/// The atom at `*i`, past hesitations if `after_symbol`, moving `*i` after it.
fn atom_at(els: &[El], i: &mut usize, after_symbol: bool) -> Option<usize> {
    let mut k = *i;
    if after_symbol {
        while k < els.len() && els[k].kind == Kind::Hesitation && els[k].open_after {
            k += 1;
        }
    }
    let e = els.get(k)?;
    if !e.is_atom() || after_symbol && !e.open_before {
        return None;
    }
    *i = k + 1;
    Some(k)
}

/// Whether a part of a run reads as code in prose.
fn evidence(els: &[El], run: &Run, words: &[&str]) -> bool {
    let atom = |a: usize| els[a].bare();
    let last = *run.atoms.last().expect("a run has an atom");
    let first = run.atoms[0];
    // A listed path, or the start of one.
    if run.atoms.iter().any(|&a| els[a].snapped) {
        return true;
    }
    // A file or a domain ("engine dot rs", "example dot com").
    if let Some(g) = run.groups.last()
        && g.chars == "."
        && (listed(EXTENSIONS, &atom(last)) || listed(DOMAINS, &atom(last)))
    {
        return true;
    }
    if let Some(p) = &run.prefix {
        match p.chars.as_str() {
            // A long flag, or a short one ("dash dash force", "dash v").
            "--" => return true,
            "-" if els[first].text.chars().count() == 1 => return true,
            // A path from the home, current or parent folder.
            c if c.starts_with("~/") || c.starts_with("./") || c.starts_with("../") => {
                return true;
            }
            // An absolute path with two slashes or more.
            "/" if run.groups.iter().any(|g| g.chars.contains('/')) => return true,
            // A dotfile or the files of an extension ("dot gitignore", "the dot rs files").
            "." if run.groups.is_empty()
                && (listed(DOTFILES, &atom(first)) || listed(EXTENSIONS, &atom(first))) =>
            {
                return true;
            }
            _ => {}
        }
    }
    // Two joints or more ("source slash biasing slash spoken").
    if run.groups.len() >= 2 {
        return true;
    }
    for (k, g) in run.groups.iter().enumerate() {
        let next = run.atoms[k + 1];
        // A name rare in prose ("lead underscore system"), a URL's scheme or a port.
        if g.strong
            || g.chars == "://"
            || g.chars == ":" && els[next].text.chars().all(|c| c.is_ascii_digit())
        {
            return true;
        }
    }
    // A word that already reads as code ("source slash engine.rs").
    if run.atoms.iter().any(|&a| reads_as_code(&els[a].text)) {
        return true;
    }
    // A word beside it that names a file, a folder or a flag.
    let start =
        (run.prefix.as_ref()).map_or(els[first].words.start, |p| els[p.els.start].words.start);
    let end = els[last].words.end;
    let cue = |w: Option<&&str>| w.is_some_and(|w| listed(CUES, &bare(w)));
    let before =
        start > 0 && split_word(words[start - 1]).2.is_empty() && cue(words.get(start - 1));
    before || els[last].open_after && cue(words.get(end))
}

/// Whether `text` has a character of code between letters or digits, as in "engine.rs",
/// "snake_case" or "std::fs", other than the dots of initials ("U.S").
fn reads_as_code(text: &str) -> bool {
    if text.contains(['/', '\\', '_']) || text.contains("::") {
        return text.chars().any(char::is_alphanumeric);
    }
    (text.split('.').skip(1)).any(|p| p.chars().filter(|c| c.is_alphabetic()).count() >= 2)
        && text.split('.').next().is_some_and(|p| !p.is_empty())
}

/// The run's text, with the punctuation around it, or none if it starts or ends inside a word.
fn write_run(els: &[El], run: &Run, words: &[&str]) -> Option<Join> {
    let first = run.prefix.as_ref().map_or(run.atoms[0], |p| p.els.start);
    let last = *run.atoms.last().expect("a run has an atom");
    if !els[first].starts_word || !els[last].ends_word {
        return None;
    }
    let atom = |a: usize| {
        let e = &els[a];
        let bare = e.bare();
        match NUMBERS.iter().position(|n| *n == bare) {
            Some(n) if !e.snapped => n.to_string(),
            _ if e.letters => bare,
            _ => e.text.clone(),
        }
    };
    let mut text = run
        .prefix
        .as_ref()
        .map_or(String::new(), |p| p.chars.clone());
    text.push_str(&atom(run.atoms[0]));
    for (k, g) in run.groups.iter().enumerate() {
        text.push_str(&g.chars);
        text.push_str(&atom(run.atoms[k + 1]));
    }
    let range = els[first].words.start..els[last].words.end;
    let (lead, _, _) = split_word(words[range.start]);
    let (_, _, trail) = split_word(words[range.end - 1]);
    Some(Join {
        words: range,
        text: format!("{lead}{text}{trail}"),
    })
}

#[cfg(test)]
mod tests;
