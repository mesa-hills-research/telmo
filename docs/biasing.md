# Special words

Speech models misspell names and technical terms they rarely heard in training, such as
"Fauchelevent" or "kubectl". Given a list of such special words, the decoder favours them where
the audio allows, and ordinary words come out as before.

```rust
use telmo::{Model, Options, SpecialWords};

let model = Model::load("granite-int8.pack".as_ref(), &Options::default())?;
let mut words = SpecialWords::new();
words.add("Fauchelevent");
words.add("kubectl");
// A word seen in code keeps its place even with only two letters.
words.add_code_word("rg");
let text = model.transcribe_biased(&samples, &words)?.text;
```

Give each word in its usual spelling. The list is meant for a few hundred words, and several
words separated by spaces ("Visual Studio Code") are favoured together. A stream takes a list
with `Stream::set_special_words` or `Streamer::set_special_words`, and the list applies from the
next utterance on, so it can change between utterances. With an empty list both models decode
exactly as `transcribe` does.

## How it works

Each word is split into the model's tokens as written and capitalized ("kubectl" and
"Kubectl"), with the word-start marker on its first token.

Granite decodes greedily, then spots each special word in the CTC output. A dynamic program
over the frames finds the word's best alignment with a free start and end, and compares it with
the greedy path over the same frames. Where the alignment plus a bonus for each of its tokens
beats the greedy path by a margin, the word replaces the greedy words there, which it must cover
whole, at least one of them. Where spotted words overlap, the best one wins.

Parakeet decodes with a beam search of 4 hypotheses over a prefix tree of the words' tokens with
failure links (Aho-Corasick). A token that continues a special word earns a bonus, which is
taken back unless the word is completed and followed by a new word, punctuation or the end of
the audio. The hypotheses at the earliest frame expand together, so the joint network and the
prediction network read their weights once for all of them. The final hypothesis is the one with
the best score per token, as in NVIDIA's NeMo, because a path that skips speech with long blanks
can otherwise score as well as the right text.

Some rules keep ordinary words safe.

- Words of one or two letters or digits are left out unless added with `add_code_word`.
- A word that is a single token of the vocabulary is one the model knows, so it earns half the
  bonus. For Granite that means it is never spotted at the default margin.
- Partial words earn nothing in the end, and a completed word only when a word boundary
  follows it.
- A spotted word must beat what Granite already heard by the margin.

`SpecialWords::boost` holds the settings.

| Setting | Default | Meaning |
|---|---:|---|
| `bonus` | 1.5 | Parakeet's bonus for each token of a special word, in log probability |
| `beam` | 4 | Hypotheses Parakeet's beam search keeps |
| `spotting_bonus` | 4 | Granite's bonus for each token of a spotted word |
| `spotting_margin` | 2 | How much a spotted word with its bonus must beat the greedy words |

## Accuracy

### Rare words

The test set is the LibriSpeech test-clean and test-other utterances with words that a
170,000-word English list (SCOWL size 70, with British spellings) doesn't know, most of them
names such as Fauchelevent, Leocadia or Bartley. Their chapters were split in two halves, one for
tuning the settings and one for the numbers below, 441 utterances with 544 occurrences of 323
such words. The chapters of each half fall into four groups, and each group's utterances were
transcribed with a list of 300 words, the group's own rare words and rare words of the other half
that don't occur in it.

| | Granite | Granite with the list | Parakeet | Parakeet with the list |
|---|---:|---:|---:|---:|
| Rare words transcribed right | 66.2% | 84.6% | 44.7% | 73.0% |
| Word error rate | 3.81% | 2.83% | 5.61% | 4.03% |
| Listed words put where the reference has another word | 1 | 5 | 2 | 5 |

### Ordinary speech

The same models transcribed ordinary speech with a list of 300 identifiers from a Rust code base
that the word list doesn't know, such as kubectl, serde, tokio, impl, cfg, mutex, uuid and
ratatui. LibriSpeech here means test-clean and test-other without the 1,000 utterances used for
tuning, and FLEURS the first 100 test utterances of each language.

| Set | Granite | Granite with the list | Parakeet | Parakeet with the list |
|---|---:|---:|---:|---:|
| LibriSpeech test-clean (2,120 utterances) | 1.33% | 1.33% | 1.93% | 1.91% |
| LibriSpeech test-other (2,439 utterances) | 2.55% | 2.55% | 3.63% | 3.68% |
| FLEURS English | 7.70% | 7.70% | 7.46% | 7.36% |
| FLEURS German | 6.56% | 6.56% | 6.13% | 6.21% |
| FLEURS French | 9.52% | 9.59% | 6.27% | 6.31% |
| FLEURS Spanish | 5.87% | 5.95% | 4.70% | 4.70% |
| FLEURS Italian | 7.41% | 7.41% | 5.43% | 5.35% |
| FLEURS Portuguese | 11.12% | 11.12% | 7.77% | 7.77% |
| FLEURS Japanese, character error rate | 5.65% | 5.65% | | |

Every difference lies inside the 95% interval of a paired bootstrap over the utterances, which
includes zero. Granite put a listed word in place of another word 4 times in the 4,559
LibriSpeech utterances ("and" became "json" once) and 5 times in the 600 European FLEURS
utterances, where short identifiers resemble foreign words ("MP" became "mcp"). Parakeet did so 5 times, all in one
test-other chapter whose character Wicker it heard as "waker" (a Rust type on the list).

### Streaming

50 utterances of the reporting half went through `UtteranceStream` in steps of half a second,
with a pass after each step and a final pass at the end.

| | Granite | Granite with the list | Parakeet | Parakeet with the list |
|---|---:|---:|---:|---:|
| Rare words transcribed right, streamed | 66.7% | 85.7% | 38.1% | 71.4% |
| Rare words transcribed right, offline | 66.7% | 85.7% | 39.7% | 74.6% |
| Word error rate, streamed | 4.58% | 3.65% | 6.52% | 4.97% |
| Word error rate, offline | 4.19% | 3.18% | 6.21% | 4.50% |

### Choosing the settings

The settings were chosen on the tuning half and the 1,000 ordinary utterances, as the strongest
that left the ordinary word error rate inside the bootstrap interval with few insertions.

| Model | Setting | Rare words right, tuning half | Ordinary WER change, clean and other | Listed words inserted, 1,000 ordinary utterances |
|---|---|---:|---|---:|
| Granite | greedy | 64.7% | | 1 |
| Granite | bonus 3, margin 2 | 81.0% | +0.00, +0.00 | 1 |
| Granite | bonus 4, margin 2 (default) | 86.9% | +0.00, +0.00 | 4 |
| Granite | bonus 5, margin 0 | 91.5% | +0.06 (beyond noise), +0.02 | 20 |
| Parakeet | greedy | 45.3% | | 0 |
| Parakeet | bonus 1.5 (default) | 70.9% | +0.03, −0.02 | 0 |
| Parakeet | bonus 2 | 74.0% | +0.03, +0.02 | 3 |
| Parakeet | bonus 3 | 79.6% | +0.15, +0.26 (both beyond noise) | 5 |

Parakeet's bonus of 2 looked as safe on the tuning set, but on the reporting half of test-other
it raised the word error rate by 0.08 points, just beyond the interval (+0.01 to +0.15), with 9
listed words inserted. Its bonus is therefore 1.5, which stays inside the interval on every set.
Its numbers above come from the same reporting half, so they aren't independent of that choice.

A larger `spotting_margin` trades recall for safety. With a margin of 4 Granite found 82.5% of
the rare words in the reporting half and put no listed word into the ordinary LibriSpeech
utterances.

For Granite, CTC prefix beam search over the same token tree was tried as well. At the same
recall it raised the word error rate on test-other by 0.14 points, while word spotting left it
unchanged, so word spotting is what Granite uses.

### Spoken technical terms

People say "nginx" as "engine x" and "kubectl" as "cube control". To see how far biasing gets
there, espeak-ng read 21 sentences with eight technical terms in three voices, once with each
term as written (which espeak reads as it guesses from the spelling) and once with the term in
the words people say. The synthetic speech is robotic, and the models get about 14% of its words
wrong, so these numbers are only a rough guide.

| Terms transcribed as written | Granite | Granite with the list | Parakeet | Parakeet with the list |
|---|---:|---:|---:|---:|
| espeak reading the spelling | 44% | 68% | 49% | 52% |
| espeak saying the spoken form | 44% | 60% | 38% | 60% |

Words close to their spelling come out right ("Graphfana" becomes Grafana, "Kbectel" Kubectl).
Words heard as other words mostly stay as they are. "serde" read aloud stayed "third" or
"survey", "tokio" mostly stayed "Tokyo", and Granite turned one "cube control" into cubecl,
another identifier on the list. Parakeet turned some "cube control" into kubectl.

## Speed

On 4 threads, Granite's spotting with 300 words adds about 30 ms per minute of audio to the
150 ms its CTC head and greedy decoding take, and Parakeet's beam search takes about 470 ms per
minute of audio where greedy decoding takes 210 ms. The encoders take 4.5 to 6 seconds per
minute of audio, so the list adds under 1% to Granite's time and about 5% to Parakeet's. In the
streaming test above, the passes took 2% longer with the list for Granite and 12% longer for
Parakeet, whose passes decode the whole utterance again. The machine is the one in
[measurements.md](measurements.md#machine), with other work running.

## Limits

- A word the model hears as other words stays unfound. People say "nginx" as "engine x" and
  "kubectl" as "cube control", and favouring the written spelling can't turn those into it.
  The synthetic sentences above show it, where "serde" read aloud became "third" or "survey".
- The tokenizer files' Unicode normalization isn't applied to the words, which matters only for
  compatibility characters such as full-width letters.
