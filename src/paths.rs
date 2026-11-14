//! File and folder names under a folder, so that a dictated path comes out as it's spelled
//! there ("source slash engine dot rs" as src/engine.rs).
//!
//! [`Paths::scan`] lists names breadth first and keeps to `.gitignore` files. It reads folder
//! listings and `.gitignore` files, nothing else, and follows no symlinks. The names stay in
//! memory as long as the list does.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::ops::Range;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::symbols;

/// How much of a tree [`Paths::scan`] lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathLimits {
    /// The most names to list, files and folders together. The walk goes breadth first, so a
    /// larger tree keeps its shallow names.
    pub max_entries: usize,
    /// The most folders below the root to go into.
    pub max_depth: usize,
    /// Whether to list hidden folders other than `.git`, which is always left out. Hidden files
    /// such as `.gitignore` are listed either way.
    pub hidden_folders: bool,
    /// Whether to leave out what `.gitignore` files ignore.
    pub gitignore: bool,
}

impl Default for PathLimits {
    fn default() -> Self {
        PathLimits {
            max_entries: 20_000,
            max_depth: 12,
            hidden_folders: false,
            gitignore: true,
        }
    }
}

/// The relative paths of files and folders that dictated paths snap to.
#[derive(Clone, Default)]
pub struct Paths {
    entries: Vec<Entry>,
    /// The entries by their last component's letters and digits in lower case ("enginers" for
    /// src/engine.rs).
    by_name: HashMap<String, Vec<u32>>,
    complete: bool,
}

#[derive(Debug, Clone)]
struct Entry {
    /// The path from the root with `/` between components.
    path: String,
    folder: bool,
}

impl fmt::Debug for Paths {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Paths")
            .field("len", &self.len())
            .field("complete", &self.complete)
            .finish_non_exhaustive()
    }
}

/// A folder the walk has yet to list, with its path from the root, its depth and the ignore
/// files that apply in it.
type Pending = (PathBuf, String, usize, Vec<usize>);

impl Paths {
    /// The names of the files and folders under `root`, breadth first, as far as `limits` go.
    /// Unless `limits` say otherwise, what a `.gitignore` file ignores is left out, and so are
    /// hidden folders. Symlinks are listed and never followed, and names that aren't UTF-8 are
    /// left out.
    pub fn scan(root: &Path, limits: &PathLimits) -> Result<Paths> {
        let top = std::fs::read_dir(root).map_err(|e| Error::io(root, e))?;
        let mut paths = Paths {
            complete: true,
            ..Paths::default()
        };
        let mut rules: Vec<Rules> = Vec::new();
        let mut queue: VecDeque<Pending> = VecDeque::new();
        let mut first = Some(top);
        queue.push_back((root.to_path_buf(), String::new(), 0, Vec::new()));
        'walk: while let Some((dir, rel, depth, mut applying)) = queue.pop_front() {
            let listing = match first.take() {
                Some(listing) => Ok(listing),
                None => std::fs::read_dir(&dir),
            };
            let Ok(listing) = listing else {
                paths.complete = false;
                continue;
            };
            let mut found: Vec<(String, bool)> = Vec::new();
            for item in listing {
                let Ok(item) = item else {
                    paths.complete = false;
                    continue;
                };
                let Ok(name) = item.file_name().into_string() else {
                    continue;
                };
                // The entry's own type, so a symlink to a folder isn't one.
                let Ok(kind) = item.file_type() else {
                    continue;
                };
                found.push((name, kind.is_dir()));
            }
            found.sort();
            if limits.gitignore
                && found.iter().any(|(n, dir)| n == ".gitignore" && !dir)
                && let Ok(text) = std::fs::read_to_string(dir.join(".gitignore"))
            {
                applying.push(rules.len());
                rules.push(Rules::parse(&text, &rel));
            }
            for (name, is_dir) in found {
                if name == ".git" || is_dir && name.starts_with('.') && !limits.hidden_folders {
                    continue;
                }
                let path = if rel.is_empty() {
                    name.clone()
                } else {
                    format!("{rel}/{name}")
                };
                if limits.gitignore && ignored(&rules, &applying, &path, is_dir) {
                    continue;
                }
                if paths.entries.len() == limits.max_entries {
                    paths.complete = false;
                    break 'walk;
                }
                paths.push(path.clone(), is_dir);
                if is_dir {
                    if depth < limits.max_depth {
                        queue.push_back((dir.join(&name), path, depth + 1, applying.clone()));
                    } else {
                        paths.complete = false;
                    }
                }
            }
        }
        Ok(paths)
    }

    /// The paths given, each relative with `/` between components, such as an editor's list of
    /// open files or the output of `git ls-files`. The folders they're in are added.
    pub fn from_list<I, S>(list: I) -> Paths
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut paths = Paths {
            complete: true,
            ..Paths::default()
        };
        let mut seen = HashSet::new();
        for path in list {
            let path = path.as_ref().replace('\\', "/");
            let parts: Vec<&str> = (path.split('/'))
                .filter(|p| !p.is_empty() && *p != ".")
                .collect();
            for k in 1..=parts.len() {
                let sub = parts[..k].join("/");
                if seen.insert(sub.clone()) {
                    paths.push(sub, k < parts.len());
                }
            }
        }
        paths
    }

    fn push(&mut self, path: String, folder: bool) {
        let name = path.rsplit('/').next().unwrap_or(&path);
        let key = letters(name);
        if !key.is_empty() {
            let id = self.entries.len() as u32;
            self.by_name.entry(key).or_default().push(id);
        }
        self.entries.push(Entry { path, folder });
    }

    /// The files and folders listed.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether every name under the root was listed, which a limit or an unreadable folder
    /// stops.
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Each path, folders included, as listed.
    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|e| e.path.as_str())
    }

    /// The parts of names worth favouring while decoding, most used first, each of three letters
    /// or more and spelled as most names spell it, for [`crate::SpecialWords::add_code_word`].
    /// Parts are split at `.`, `_`, `-`, spaces, a change of case and a change between letters
    /// and digits.
    pub fn words(&self) -> Vec<String> {
        let mut counts: HashMap<String, (usize, HashMap<String, usize>)> = HashMap::new();
        for entry in &self.entries {
            let name = entry.path.rsplit('/').next().unwrap_or(&entry.path);
            for piece in pieces(name) {
                if piece.chars().filter(|c| c.is_alphabetic()).count() < 3 {
                    continue;
                }
                let (n, spellings) = counts.entry(piece.to_lowercase()).or_default();
                *n += 1;
                *spellings.entry(piece.to_string()).or_default() += 1;
            }
        }
        let mut words: Vec<(usize, String)> = counts
            .into_iter()
            .map(|(lower, (n, spellings))| {
                let best = (spellings.into_iter())
                    .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
                    .map_or(lower, |(s, _)| s);
                (n, best)
            })
            .collect();
        words.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        words.into_iter().map(|(_, w)| w).collect()
    }

    /// The stretches of `words` that spell a listed path or the end of one, each with the path's
    /// spelling. A stretch has a spoken symbol in it ("engine dot rs") or comes before a spoken
    /// slash ("source" before "slash biasing"). With `hesitations_go`, a hesitation inside one
    /// goes with it.
    pub(crate) fn snap(&self, words: &[&str], hesitations_go: bool) -> Vec<Snap> {
        if self.entries.is_empty() {
            return Vec::new();
        }
        let heard = heard(words, hesitations_go);
        let mut found: Vec<(Snap, Score)> = Vec::new();
        for window in windows(&heard) {
            if let Some(found_here) = self.best(&heard, window) {
                found.push(found_here);
            }
        }
        // The matches covering the most words first, then the most exact.
        found.sort_by(|a, b| {
            let len = |s: &Snap| s.words.len();
            (len(&b.0).cmp(&len(&a.0)))
                .then(a.1.table.cmp(&b.1.table))
                .then(a.0.words.start.cmp(&b.0.words.start))
        });
        let mut taken: Vec<Snap> = Vec::new();
        for (snap, _) in found {
            let apart =
                |t: &Snap| t.words.end <= snap.words.start || snap.words.end <= t.words.start;
            if taken.iter().all(apart) {
                taken.push(snap);
            }
        }
        taken.sort_by_key(|s| s.words.start);
        taken
    }

    /// The best path for the heard elements `window`, unless two write it differently.
    fn best(&self, heard: &[Heard], window: Range<usize>) -> Option<(Snap, Score)> {
        let part = &heard[window.clone()];
        // A folder named by itself snaps only before a slash.
        let has_symbol = part.iter().any(|h| matches!(h.kind, Kind::Symbol(_)));
        let before_slash = !part.last()?.stop_after && starts_slash(heard.get(window.end));
        let options = options(part);
        let last_slash =
            (part.iter()).rposition(|h| matches!(&h.kind, Kind::Symbol(c) if c.contains('/')));
        let name_from = last_slash.map_or(0, |k| k + 1);
        let slashes: usize = (part.iter())
            .map(|h| match &h.kind {
                Kind::Symbol(c) => c.matches('/').count(),
                _ => 0,
            })
            .sum();
        let mut best: Option<(Score, String)> = None;
        let mut tied = false;
        for key in name_keys(part, &options, name_from) {
            for &id in self.by_name.get(&key).into_iter().flatten() {
                let entry = &self.entries[id as usize];
                if !has_symbol && (!before_slash || !entry.folder) {
                    continue;
                }
                // The end of the path with as many slashes as were said.
                let path = entry.path.as_str();
                let Some(start) = suffix_start(path, slashes) else {
                    continue;
                };
                let suffix = &path[start..];
                let Some(mut score) = align(part, &options, suffix) else {
                    continue;
                };
                score.depth = path.matches('/').count();
                let better = match &best {
                    None => true,
                    Some((b, text)) => match score.cmp_key().cmp(&b.cmp_key()) {
                        std::cmp::Ordering::Less => {
                            tied = false;
                            true
                        }
                        std::cmp::Ordering::Equal => {
                            tied |= text != suffix;
                            false
                        }
                        std::cmp::Ordering::Greater => false,
                    },
                };
                if better {
                    best = Some((score, suffix.to_string()));
                }
            }
        }
        let (score, text) = best?;
        if tied {
            return None;
        }
        let first = part.first()?.word;
        let last = part.last()?.word;
        // A stretch takes whole words.
        let starts = window.start == 0 || heard[window.start - 1].word != first;
        let ends = heard.get(window.end).is_none_or(|h| h.word != last);
        if !starts || !ends {
            return None;
        }
        Some((
            Snap {
                words: first..last + 1,
                text,
            },
            score,
        ))
    }
}

/// A stretch of a transcript's words written as a path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Snap {
    pub words: Range<usize>,
    /// The path's spelling, from the component the stretch starts at.
    pub text: String,
}

/// How well a match fits. Fewer words from the table, more letters in the transcript's case,
/// then a shallower path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Score {
    table: usize,
    case: usize,
    depth: usize,
}

impl Score {
    fn cmp_key(&self) -> (usize, std::cmp::Reverse<usize>, usize) {
        (self.table, std::cmp::Reverse(self.case), self.depth)
    }
}

/// Whether `heard` is a spoken symbol that starts with a slash.
fn starts_slash(heard: Option<&Heard>) -> bool {
    matches!(heard, Some(Heard { kind: Kind::Symbol(c), .. }) if c.starts_with('/'))
}

/// The byte where the last `slashes + 1` components of `path` start.
fn suffix_start(path: &str, slashes: usize) -> Option<usize> {
    if slashes == 0 {
        return Some(path.rfind('/').map_or(0, |i| i + 1));
    }
    let mut seen = 0;
    for (i, b) in path.bytes().enumerate().rev() {
        if b == b'/' {
            if seen == slashes {
                return Some(i + 1);
            }
            seen += 1;
        }
    }
    (seen == slashes).then_some(0)
}

/// `text`'s letters and digits, in lower case.
fn letters(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// The parts of a name, split at what isn't a letter or digit, a change from lower to upper
/// case, and a change between letters and digits ("SpecialWords" gives "Special" and "Words").
fn pieces(name: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = None;
    let mut prev: Option<char> = None;
    for (i, c) in name.char_indices() {
        if !c.is_alphanumeric() {
            if let Some(s) = start.take() {
                out.push(&name[s..i]);
            }
            prev = None;
            continue;
        }
        let cut = prev.is_some_and(|p| {
            p.is_lowercase() && c.is_uppercase() || p.is_alphabetic() != c.is_alphabetic()
        });
        if cut && let Some(s) = start {
            out.push(&name[s..i]);
            start = Some(i);
        }
        start.get_or_insert(i);
        prev = Some(c);
    }
    if let Some(s) = start {
        out.push(&name[s..]);
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Kind {
    /// A word's letters and digits, as written.
    Word(String),
    /// Spoken symbols' characters ("/", "--").
    Symbol(String),
}

/// A word or symbol of the transcript, for matching paths.
#[derive(Debug, Clone)]
struct Heard {
    kind: Kind,
    /// The word it's in, or the first of the two a symbol takes ("forward slash").
    word: usize,
    /// Whether punctuation stands between it and the next.
    stop_after: bool,
}

/// The words and symbols of `words`. A word written with code characters ("api.py") splits at
/// them, and hesitations are left out with `hesitations_go`.
fn heard(words: &[&str], hesitations_go: bool) -> Vec<Heard> {
    let mut out: Vec<Heard> = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let (lead, core, trail) = symbols::split_word(words[i]);
        if !lead.is_empty()
            && let Some(last) = out.last_mut()
        {
            last.stop_after = true;
        }
        if let Some((chars, taken)) = symbols::spoken_symbol(words, i) {
            let (_, _, trail) = symbols::split_word(words[i + taken - 1]);
            out.push(Heard {
                kind: Kind::Symbol(chars),
                word: i,
                stop_after: !trail.is_empty(),
            });
            i += taken;
            continue;
        }
        if hesitations_go && lead.is_empty() && trail.is_empty() && symbols::is_hesitation(core) {
            i += 1;
            continue;
        }
        for part in symbols::split_code(core) {
            let kind = match part {
                symbols::Part::Word(w) => Kind::Word(w.to_string()),
                symbols::Part::Symbol(c) => Kind::Symbol(c),
            };
            out.push(Heard {
                kind,
                word: i,
                stop_after: false,
            });
        }
        if !trail.is_empty()
            && let Some(last) = out.last_mut()
        {
            last.stop_after = true;
        }
        i += 1;
    }
    out
}

/// Plain words a path's name may have in a row, as in "zero zero one two underscore init".
const REACH: usize = 5;

/// The stretches of `heard` worth matching. Each ends with a word, starts with a word or a dot,
/// never crosses punctuation, and has a symbol in it or comes before a slash, with at most
/// [`REACH`] words in a row without a symbol.
fn windows(heard: &[Heard]) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let is_word = |k: usize| matches!(heard[k].kind, Kind::Word(_));
    for start in 0..heard.len() {
        let dot = matches!(&heard[start].kind, Kind::Symbol(c) if c == ".");
        if !is_word(start) && !dot {
            continue;
        }
        let mut plain = 0;
        let mut symbol = false;
        for end in start..heard.len() {
            if is_word(end) {
                plain += 1;
                if plain > REACH {
                    break;
                }
            } else {
                plain = 0;
                symbol = true;
            }
            let before_slash = !heard[end].stop_after && starts_slash(heard.get(end + 1));
            if is_word(end) && (symbol || before_slash) {
                out.push(start..end + 1);
            }
            if heard[end].stop_after {
                break;
            }
        }
    }
    out
}

/// A way to spell heard elements in a name: the elements it takes, its letters and digits in
/// lower case, and whether it comes from the table of spoken words.
type Spelling = (usize, String, bool);

/// For each heard element, the ways it and the words after it may be spelled in a name.
fn options(part: &[Heard]) -> Vec<Vec<Spelling>> {
    let word = |k: usize| match part.get(k).map(|h| &h.kind) {
        Some(Kind::Word(w)) => Some(w.to_lowercase()),
        _ => None,
    };
    (0..part.len())
        .map(|k| {
            let Some(w) = word(k) else {
                return Vec::new();
            };
            let mut out = vec![(1, w.clone(), false)];
            if let Some(n) = number(&w) {
                out.push((1, n.to_string(), false));
                // "twenty one"
                if n >= 20
                    && let Some(unit) = word(k + 1).and_then(|u| number(&u))
                    && (1..10).contains(&unit)
                {
                    out.push((2, (n + unit).to_string(), false));
                }
            }
            for &(written, spoken) in SPOKEN {
                let said: Vec<&str> = spoken.split(' ').collect();
                let fits =
                    (said.iter().enumerate()).all(|(j, s)| word(k + j).as_deref() == Some(*s));
                if fits {
                    out.push((said.len(), written.to_string(), true));
                }
            }
            out
        })
        .collect()
}

/// The keys of the names the elements of `part` from `from` on may spell, at most 64.
fn name_keys(part: &[Heard], options: &[Vec<Spelling>], from: usize) -> Vec<String> {
    let mut keys: Vec<(usize, String)> = vec![(from, String::new())];
    let mut done = Vec::new();
    while let Some((k, key)) = keys.pop() {
        if k == part.len() {
            if !done.contains(&key) {
                done.push(key);
            }
            continue;
        }
        if keys.len() + done.len() > 64 {
            break;
        }
        match &part[k].kind {
            Kind::Symbol(_) => keys.push((k + 1, key)),
            Kind::Word(_) => {
                for (taken, spelling, _) in &options[k] {
                    keys.push((k + taken, format!("{key}{spelling}")));
                }
            }
        }
    }
    done
}

/// How `part` spells `path`, from its start to its end, if it does. Each symbol stands for its
/// characters in the path, and each word for letters and digits of it. `/` and `.` must be said,
/// and `_`, `-` and the like may be left out.
fn align(part: &[Heard], options: &[Vec<Spelling>], path: &str) -> Option<Score> {
    let lower: Vec<char> = path.chars().flat_map(char::to_lowercase).collect();
    let written: Vec<char> = path.chars().collect();
    if lower.len() != written.len() {
        return None;
    }
    let n = lower.len();
    // best[h][p] is the best score with `h` elements taken and `p` characters spelled.
    let mut best: Vec<Vec<Option<Score>>> = vec![vec![None; n + 1]; part.len() + 1];
    best[0][0] = Some(Score::default());
    let better = |slot: &mut Option<Score>, s: Score| {
        if slot.is_none_or(|b| s.cmp_key() < b.cmp_key()) {
            *slot = Some(s);
        }
    };
    for h in 0..=part.len() {
        for p in 0..=n {
            let Some(score) = best[h][p] else {
                continue;
            };
            // A separator of a name left unsaid, after a word.
            if p < n
                && p > 0
                && !lower[p].is_alphanumeric()
                && lower[p] != '/'
                && lower[p] != '.'
                && lower[p - 1].is_alphanumeric()
                && !matches!(part.get(h).map(|x| &x.kind), Some(Kind::Symbol(_)))
            {
                better(&mut best[h][p + 1], score);
            }
            let Some(element) = part.get(h) else {
                continue;
            };
            match &element.kind {
                Kind::Symbol(chars) => {
                    let c: Vec<char> = chars.chars().collect();
                    if lower[p..].starts_with(&c) {
                        better(&mut best[h + 1][p + c.len()], score);
                    }
                }
                Kind::Word(as_written) => {
                    for (taken, spelling, table) in &options[h] {
                        let s: Vec<char> = spelling.chars().collect();
                        if !lower[p..].starts_with(&s) {
                            continue;
                        }
                        let mut next = score;
                        if *table {
                            next.table += 1;
                        } else if *taken == 1 {
                            next.case += (as_written.chars())
                                .zip(&written[p..p + s.len()])
                                .filter(|(a, b)| a == *b)
                                .count();
                        }
                        better(&mut best[h + taken][p + s.len()], next);
                    }
                }
            }
        }
    }
    best[part.len()][n]
}

/// The number a word says, from zero to ninety.
fn number(word: &str) -> Option<usize> {
    const UNITS: [&str; 20] = [
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
    const TENS: [&str; 8] = [
        "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];
    (UNITS.iter().position(|u| *u == word))
        .or_else(|| TENS.iter().position(|t| *t == word).map(|k| 20 + 10 * k))
}

/// Parts of names and the words people say for them.
const SPOKEN: &[(&str, &str)] = &[
    ("src", "source"),
    ("lib", "library"),
    ("libs", "libraries"),
    ("bin", "binary"),
    ("bin", "binaries"),
    ("cfg", "config"),
    ("cfg", "configuration"),
    ("conf", "config"),
    ("conf", "configuration"),
    ("config", "configuration"),
    ("pkg", "package"),
    ("pkgs", "packages"),
    ("tmp", "temp"),
    ("tmp", "temporary"),
    ("env", "environment"),
    ("dev", "development"),
    ("prod", "production"),
    ("doc", "document"),
    ("doc", "documentation"),
    ("docs", "documents"),
    ("docs", "documentation"),
    ("img", "image"),
    ("imgs", "images"),
    ("msg", "message"),
    ("msgs", "messages"),
    ("cmd", "command"),
    ("cmds", "commands"),
    ("ctx", "context"),
    ("util", "utility"),
    ("utils", "utilities"),
    ("utils", "utility"),
    ("impl", "implementation"),
    ("init", "initialize"),
    ("init", "initialise"),
    ("str", "string"),
    ("vec", "vector"),
    ("fn", "function"),
    ("func", "function"),
    ("funcs", "functions"),
    ("num", "number"),
    ("arg", "argument"),
    ("args", "arguments"),
    ("param", "parameter"),
    ("params", "parameters"),
    ("attr", "attribute"),
    ("attrs", "attributes"),
    ("ref", "reference"),
    ("refs", "references"),
    ("spec", "specification"),
    ("specs", "specifications"),
    ("info", "information"),
    ("auth", "authentication"),
    ("db", "database"),
    ("app", "application"),
    ("apps", "applications"),
    ("repo", "repository"),
    ("dir", "directory"),
    ("dirs", "directories"),
    ("mgr", "manager"),
    ("svc", "service"),
    ("srv", "server"),
    ("std", "standard"),
    ("rs", "rust"),
    ("py", "python"),
    ("js", "javascript"),
    ("ts", "typescript"),
    ("md", "markdown"),
    ("yml", "yaml"),
    ("yml", "yammel"),
    ("yaml", "yammel"),
    ("json", "jason"),
    ("txt", "text"),
    ("sh", "shell"),
    ("cpp", "c plus plus"),
    ("k8s", "kubernetes"),
    ("i18n", "internationalization"),
    ("e2e", "end to end"),
    ("sql", "sequel"),
    ("wasm", "web assembly"),
];

/// Ignore rules of one `.gitignore` file.
#[derive(Debug, Clone)]
struct Rules {
    /// The folder the file is in, relative to the root, empty for the root.
    base: String,
    rules: Vec<Rule>,
}

#[derive(Debug, Clone)]
struct Rule {
    pattern: Vec<u8>,
    /// Whether it re-includes what it matches ("!").
    negated: bool,
    /// Whether it matches folders only (a trailing "/").
    folders: bool,
    /// Whether it matches a path from the file's folder rather than a name at any depth.
    anchored: bool,
}

impl Rules {
    fn parse(text: &str, base: &str) -> Rules {
        let mut rules = Vec::new();
        for line in text.lines() {
            let line = line.trim_end_matches(['\r', ' ', '\t']);
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (negated, line) = match line.strip_prefix('!') {
                Some(rest) => (true, rest),
                None => (false, line.strip_prefix('\\').unwrap_or(line)),
            };
            let (folders, line) = match line.strip_suffix('/') {
                Some(rest) => (true, rest),
                None => (false, line),
            };
            let anchored = line.contains('/');
            let line = line.strip_prefix('/').unwrap_or(line);
            if line.is_empty() {
                continue;
            }
            rules.push(Rule {
                pattern: line.as_bytes().to_vec(),
                negated,
                folders,
                anchored,
            });
        }
        Rules {
            base: base.to_string(),
            rules,
        }
    }
}

/// Whether `path`, relative to the root, is ignored by the rule files `applying` lists, the
/// last matching rule deciding.
fn ignored(rules: &[Rules], applying: &[usize], path: &str, folder: bool) -> bool {
    let mut out = false;
    for &k in applying {
        let file = &rules[k];
        let rel = if file.base.is_empty() {
            path
        } else {
            match path
                .strip_prefix(&file.base)
                .and_then(|r| r.strip_prefix('/'))
            {
                Some(rel) => rel,
                None => continue,
            }
        };
        let name = rel.rsplit('/').next().unwrap_or(rel);
        for rule in &file.rules {
            if rule.folders && !folder {
                continue;
            }
            let text = if rule.anchored { rel } else { name };
            if glob(&rule.pattern, text.as_bytes()) {
                out = !rule.negated;
            }
        }
    }
    out
}

/// Whether `text` matches the gitignore glob `pattern`. `*` and `?` match within a component,
/// `**` matches across them, and `[...]` matches a class.
fn glob(pattern: &[u8], text: &[u8]) -> bool {
    match pattern.first() {
        None => text.is_empty(),
        Some(b'*') if pattern.get(1) == Some(&b'*') => {
            // "**/" matches any folders, none included, and a trailing "**" everything.
            let rest = &pattern[2..];
            let rest = rest.strip_prefix(b"/").unwrap_or(rest);
            if rest.is_empty() {
                return true;
            }
            (0..=text.len()).any(|i| (i == 0 || text[i - 1] == b'/') && glob(rest, &text[i..]))
        }
        Some(b'*') => {
            let rest = &pattern[1..];
            for i in 0..=text.len() {
                if glob(rest, &text[i..]) {
                    return true;
                }
                if text.get(i) == Some(&b'/') {
                    break;
                }
            }
            false
        }
        Some(b'?') => text.first().is_some_and(|&c| c != b'/') && glob(&pattern[1..], &text[1..]),
        Some(b'[') => {
            let close = pattern.iter().skip(2).position(|&c| c == b']');
            let Some(close) = close.map(|i| i + 2) else {
                return text.first() == Some(&b'[') && glob(&pattern[1..], &text[1..]);
            };
            let Some(&c) = text.first() else {
                return false;
            };
            let class = &pattern[1..close];
            let (negated, class) = match class.first() {
                Some(b'!') | Some(b'^') => (true, &class[1..]),
                _ => (false, class),
            };
            let mut hit = false;
            let mut k = 0;
            while k < class.len() {
                if k + 2 < class.len() && class[k + 1] == b'-' {
                    hit |= (class[k]..=class[k + 2]).contains(&c);
                    k += 3;
                } else {
                    hit |= class[k] == c;
                    k += 1;
                }
            }
            hit != negated && c != b'/' && glob(&pattern[close + 1..], &text[1..])
        }
        Some(b'\\') if pattern.len() > 1 => {
            text.first() == Some(&pattern[1]) && glob(&pattern[2..], &text[1..])
        }
        Some(&p) => text.first() == Some(&p) && glob(&pattern[1..], &text[1..]),
    }
}

#[cfg(test)]
mod tests;
