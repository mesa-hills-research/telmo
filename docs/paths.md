# Symbols and paths

Speech models write spoken symbols as words ("source slash engine dot rs"). With spoken symbols
on, Telmo writes them as characters where the words read as code, and leaves them as words in
prose. Given a folder, it also writes a dictated path as the file or folder it names.

```text
So in source slash engine dot rs, add a dash dash dry dash run flag.
So in src/engine.rs, add a --dry-run flag.
```

```rust
use std::sync::Arc;
use telmo::{PathLimits, Paths, SpecialWords, Symbols};

let mut words = SpecialWords::new();
words.set_symbols(Symbols::for_language("en"));
// Lists names breadth first, keeping to .gitignore.
let paths = Paths::scan("/home/ada/proj".as_ref(), &PathLimits::default())?;
words.set_paths(Some(Arc::new(paths)));
let text = model.transcribe_biased(&samples, &words)?.text;

// On text alone.
let symbols = Symbols::for_language("en").unwrap();
assert_eq!(symbols.write("open engine dot rs", None), "open engine.rs");
```

Both are off by default. Paths snap the runs the spoken symbols find, so with paths set the
English symbols are on whatever `set_symbols` says. `telmo transcribe --symbols` and `--paths
DIR` do the same from the command. A stream takes both with `set_special_words`, from the next
utterance on.

## Spoken symbols

| Said | Written |
|---|---|
| slash, forward slash | `/` |
| backslash, back slash | `\` |
| dot | `.` |
| underscore | `_` |
| dash, hyphen | `-` |
| colon | `:` |
| tilde | `~` |
| double or triple before any of them | the symbol twice or three times |

A model's "dash-dash" or "dot-com" counts as the names in it. Single letters in a row inside a
run ("S R C") are written as one word in lower case, and the numbers zero to nineteen as digits.

## When words read as code

A run is words and spoken symbols in turn, as in "source slash engine dot rs", and it may start
with symbols, as in "dash dash force". Punctuation ends it. With fillers or a tagger set, a
hesitation inside a run goes with it.

Some symbols start a word of code and never join the word before them. Those are a flag ("dash
dash force", and "dash" before one letter) and a path from the home, current or parent folder
("tilde slash", "dot slash", "dot dot slash"). So "git push dash dash force" gives `git push
--force`.

In prose, a function word beside a symbol (an article, a pronoun, a preposition, a conjunction
or an auxiliary) ends the run there. So "a dot on the map", "and slash or", "a dash of salt" and
"to underscore the point" keep their words. Inside a long flag words join whatever they are
("--force-with-lease"), and so do they in a run with two names rare in prose
("get_or_insert").

A run is written as code when one of these shows it is.

| Sign | Said | Written |
|---|---|---|
| A listed path or folder | source slash engine dot rs | src/engine.rs |
| A dot and a file extension or a domain | engine dot rs, example dot com | engine.rs, example.com |
| A flag | dash dash allow dash empty, dash v | --allow-empty, -v |
| A path from home, here or the parent | tilde slash notes, dot slash build dot sh | ~/notes, ./build.sh |
| An absolute path with two slashes | slash var slash log | /var/log |
| A leading dot and a dotfile or an extension | dot gitignore, the dot rs files | .gitignore, the .rs files |
| Two symbols or more | source slash biasing slash spoken | source/biasing/spoken |
| A name rare in prose | lead underscore system, std colon colon fs | lead_system, std::fs |
| A URL's scheme or a port | https colon slash slash x dot org, localhost colon 8080 | https://x.org, localhost:8080 |
| A word already written as code | source slash api.py | source/api.py |
| A word beside it naming a file, a folder or a flag | the source slash engine module | the source/engine module |

A single letter after "dash" is a flag even when it's "a" or "i", if a word that names a flag
follows ("the dash i command parameter").

`tests/symbols/cases.tsv` holds 95 cases in prose and in a code field, and a test checks each.

### In a code field

`Symbols::for_language("en").map(Symbols::for_code)` is for a terminal or a code editor, where
every run is code with no sign needed and no function word ends one. So "self dot name" gives
self.name and "force dash push" force-push. A flag and a path from home still never join the
word before.

### What prose leaves as words

| Said | Written | To write it as code |
|---|---|---|
| self dot name, force dash push, input slash output | as said | a listed path, a code field, or a word such as "file" beside it |
| use underscore separated names | use_separated names | rare |
| ada at example dot com | ada at example.com | a spoken form |
| plan dash B | plan -B | rare |

A joined run keeps the model's capitals. A listed path is written as it's spelled.

## Paths

`Paths::scan` lists the names of the files and folders under a folder, and `Paths::from_list`
takes relative paths the caller has, such as an editor's open files or the output of `git
ls-files`.

| Limit | Default |
|---|---|
| Names, files and folders | 20,000 (`PathLimits::max_entries`). The walk goes breadth first, so a larger tree keeps its shallow names, and `is_complete()` says whether all were listed |
| Depth | 12 folders (`max_depth`) |
| .gitignore | Each folder's file applies below it, with `*`, `?`, `**`, classes, `!`, a leading `/` and a trailing `/`. The global excludes file and `.git/info/exclude` aren't read. `gitignore: false` lists everything |
| Hidden folders | Left out, and `.git` always (`hidden_folders`). Hidden files such as `.gitignore` are listed |
| Symlinks | Listed by name and never followed |

It reads folder listings and `.gitignore` files, nothing else, and keeps the names in memory for
as long as the list lives. It doesn't watch the tree, so scan again when files change.

### How a path is matched

Each name splits into parts at `.`, `_`, `-`, spaces, a change of case and a change between
letters and digits, so `SpecialWords.rs` has the parts Special, Words and rs. A part may be said
as written in any case, letter by letter ("R S"), with its digits as words ("v two"), or as the
word a table gives for it, such as source for src, library for lib, config for cfg, rust for rs,
markdown for md and c plus plus for cpp. A slash and a dot must be said, an underscore or a dash
inside a name may be left out ("tdt beam dot rs" for tdt_beam.rs), and a change of case is never
said.

A stretch of the transcript snaps when it has a symbol said by its name and its words spell the
last components of a listed path whole, every word and symbol of them, from a component's start.
Telmo writes them as spelled there, so "engine dot rs" gives engine.rs and "source slash engine
dot rs" gives src/engine.rs. It never adds folders that weren't said. A word said before a slash
snaps to a folder alone, so a run that reads as code but names no file still gets its folders
("source slash biasing slash new dot rs" gives src/biasing/new.rs when src/biasing exists).
Symbols a model wrote inside a word don't count as said, so "telmo 0.2.0" stays as it is next to
a folder telmo-0.2.0.

Of the paths a stretch spells, the one that covers the most words wins, then the one that needs
the fewest words from the table, then the one whose capitals agree with the transcript, then the
shallowest. If two are still equal and spelled differently, nothing snaps. There's no fuzzy
matching, so a near miss is written as the symbols say.

`Paths::words()` gives the parts of names, most used first, to favour with `add_code_word` where
the model mishears them. Favouring isn't automatic.

## With the cleanup, in a stream, and with each model

Spoken forms are written first, so "cube control dot yaml" gives kubectl.yaml with a spoken form
for kubectl. The cleanup comes after and keeps every word written as code (a path, a file name,
an identifier with an underscore or a flag), as it keeps special words. Symbol names left as
words in prose stay under the guards, as [disfluencies.md](disfluencies.md) says.

In a stream a run's words settle together. A word doesn't settle while the words after it, up to
those that may still change, are spoken symbols or words joined by them. A path name of several
words whose first word settled before the rest was heard doesn't snap.

A transcript's text is the decoding of its tokens, and each written run's tokens take the frames
of the words it replaced. For the printable ASCII characters a model's vocabulary can't spell,
such as Parakeet's `_`, `~`, `\`, `@` and brackets, Telmo's tokenizer adds written-only pieces
after the vocabulary's end (`Tokenizer::is_written_only`). Text written after decoding may use
them, and decoding never emits or favours them. So `Transcript::tokens` may hold ids at or past
the vocabulary's size, which only Telmo's tokenizer (`Model::tokenizer`) decodes.

## Accuracy

On prose with no code the symbols pass should change nothing.

| Text | Words | Symbol names | Runs written as code |
|---|---:|---:|---:|
| LibriSpeech test-clean and test-other, Granite's transcripts | 104,909 | 2 | 0 |
| The same, Parakeet's transcripts | 105,001 | 2 | 0 |
| FLEURS English dev and test, references | 22,503 | 0 | 0 |
| AMI meetings of three sets, references | 225,444 | 34 | 4 |

The four AMI runs are people spelling labels aloud ("T underscore V underscore", "R dash C"),
which is closer to dictating symbols than to prose. The other 30 names stayed words, such as
"segments slash utterances", "on slash off" and "a dot in between".

```text
cargo run --release --example symbols -- --quiet TRANSCRIPTS.tsv...
```

## Speed

On the machine in [measurements.md](measurements.md#machine), the symbols pass took 26 ms for
the 52,564 words of LibriSpeech test-clean, about half a microsecond a word. `Paths::scan` listed
20,000 names of a tree of Rust crates in 42 ms with the folders cached (490 ms the first time),
and the list took about 4 MB. With it, three sentences with five paths took 5 ms, and 1,132
messages of 39,019 words took 34 ms. `--paths DIR` with the same example measures it.

## Gaps

- English only.
- An email keeps "at" as a word.
- In a stream, a path name of several words whose first word settled early doesn't snap.
- A joined run that names no listed path keeps the model's capitals.
