# Disfluencies

The [filler rules](fillers.md) take out hesitations and stutters. A disfluency tagger goes
further, taking out false starts, repairs, repeated phrases and filler words, once over each
whole utterance. It only deletes, so every word left is a word the speech model wrote.

```text
We need to, we need to fix the the login page first.
We need to fix the login page first.
```

```rust
use std::sync::Arc;
use telmo::disfluency::Tagger;
use telmo::{Options, SpecialWords};

let tagger = Arc::new(Tagger::load("disfluency-q8_0.pack".as_ref(), &Options::default())?);
let mut words = SpecialWords::new();
words.set_disfluencies(Some(tagger.clone()));
let text = model.transcribe_biased(&samples, &words)?.text;

// Or on text alone.
let clean = tagger.remove("Um, so I think we should move the the config.");
```

The tagger is trained on English, so set it only for English speech. The filler rules run as
part of it, whatever `set_fillers` says. Special words and spoken forms' words always stay.

`Tagger::hand_written()` is a tagger with no pack, a network of hand-set rules over each word's
spelling, its word class and the next three words. Beside the fillers it takes out cut-off words
and fragments ("the des design"), a word or two-word phrase said again, also across a
hesitation or fragment ("who w who"), and sentence-opening and trailing markers. It runs an
utterance in well under a millisecond.

## Guards

`words.set_guards(Guards::for_language("en"))` keeps words that some deletion would take where
dictation needs them, whichever stage made it, the filler rules included.

| Guard | Keeps |
|---|---|
| Symbols | a spoken symbol and the words beside it, as in "dash dash force dash push" |
| Units | a short unit after a number, as in "5 mm" |
| Mentions | a filler talked about as a word ("delete ums") and a quoted word |
| Lists | items that share their first word, as in "cleaner build, cleaner layout" |
| Straddles | a cut that would leave a broken repeat, as in "we just need to uh we need to" |
| Parallels | a phrase said again in parallel or in contrast, as in "what they prefer what they not prefer" |

On 112,000 words of meeting speech they undo 65 of the tagger's 12,521 deletions.
`Guards::veto` applies them to any deletions.

## What goes

A word goes when the filler rules delete it, when one of these rules does, or when the tagger
tags it as a disfluency with enough confidence.

| Rule | Example |
|---|---|
| A word cut off and started again | "the mi mic microphone" |
| A preposition and determiner started again | "in our in your folder" |
| A repair after "sorry" between commas, or "or rather" | "the red, sorry, the blue one" |
| A two-word phrase said twice in a row | "it has to has to work" |
| A function word said twice | "you would have to to make it" |
| "so", "well", "yeah", "okay", "oh", "right", "anyway" or "I mean" opening a sentence before a clause | "So I think we should stop." |
| A trailing "so", "and yeah", "but yeah" or "you know", and "you know" or "I mean" between commas | "we need, you know, a way" |

The tagger reads the utterance as the speech model wrote it, hesitations included, since they
often mark a repair. It gives each word a probability of being a hesitation, a filler, the
abandoned start of a repair (a reparandum) or an editing term ("I mean", "sorry"). A word in a
run of repeated words goes at one threshold, and any other word at the threshold of its most
likely tag. The pack carries the thresholds the tagger was tuned with.

Punctuation and capitals are tidied as for fillers. A sentence end on a deleted word moves to the
word before, and a capital at a sentence's start moves to the next word.

## Capitals and punctuation

`words.set_punctuation(Punctuation::for_language("en"))` fixes capitals and punctuation after the
cleanup. The pronoun "i" takes a capital, and a period goes where a speaker only paused, after a
word that leaves its sentence unfinished ("I'll take the. Red one."), before a one-word sentence
or a short fragment with no verb, and before "than". The next word then goes on in lower case
where it should. A comma goes from a word that can't end a clause, where a hesitation that went
left it ("the, uh, little" gives "the little"), and with the rules on they decide every such
comma. Only a letter's case and the marks at a word's end change.

`Punctuation::joining_more()` joins more of what a pause split, as dictation has it.

```text
We're shipping a patch. Release. That is due on Friday.
We're shipping a patch release that is due on Friday.
```

It also joins a fragment's own period, one after an auxiliary with no subject ("My brother is.
Learning how") and one before a preposition after four words or more ("the library stays open. Until
nine."). In conversation a few real sentence ends go with them.

## In a stream

A stream can't change settled text, and the tagger reads a whole utterance. So with a tagger
set, an utterance's words stay tentative until it ends. The tentative text shows the utterance so
far with its fillers left out, and the last pass settles the cleaned text. That is one tagger
pass per utterance, after its last pass.

## The pack

The tagger is Ettin's 68M encoder (a ModernBERT) fine-tuned in two members whose probabilities
are averaged. `telmo transcribe --remove-disfluencies` downloads its pack on first use, from a
release of this repository that `disfluency::PACK_DOWNLOAD` pins by size and SHA-256, and
`disfluency::PACK_NOTICE` says what it's made from and under which licences. A checkpoint
(`model.safetensors`, `config.json`, `tokenizer.json` and `tagger.json`) converts into a pack.

```text
cargo run --release --example pack -- disfluency-tagger disfluency-q8_0.pack
```

The pack's weights are q8_0, as for Parakeet, and it takes 158 MB. Its deletions equal those of
the f32 weights on 99.6% of the development utterances.
`Tagger::load` also takes the checkpoint's directory and converts it in memory.

## Speed

On 4 threads of an Intel Core i9-13980HX, a 40-word text (about 49 tokens) takes 55 ms at the
median and 65 ms at the 95th percentile, both members, the rules and the tokenizer included.
`cargo bench --bench disfluency -- PACK ROWS.jsonl --threads 4 --words 40` measures it.

Streamed with Granite on the same 4 threads, 143 utterances of dictation settled a median 0.2 s
later from the end of speech than with the filler rules alone. `examples/stream` gives both
with `--fillers` and `--disfluencies`.
