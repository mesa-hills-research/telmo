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

## In a stream

A stream can't change settled text, and the tagger reads a whole utterance. So with a tagger
set, an utterance's words stay tentative until it ends. The tentative text shows the utterance so
far with its fillers left out, and the last pass settles the cleaned text. That is one tagger
pass per utterance, after its last pass.

## The pack

The tagger is Ettin's 68M encoder (a ModernBERT) fine-tuned in two members whose probabilities
are averaged. Its checkpoint (`model.safetensors`, `config.json`, `tokenizer.json` and
`tagger.json`) converts into a pack once.

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
