# Special words

Speech models misspell names and technical terms they rarely heard in training, such as
"Fauchelevent" or "kubectl". Given a list of such special words, the decoder favours them where
the audio allows. On the test sets below, the accuracy on ordinary speech changed little.

```rust
use telmo::{Model, Options, SpecialWords};

let model = Model::load("granite-int8.pack".as_ref(), &Options::default())?;
let mut words = SpecialWords::new();
words.add("Fauchelevent")?;
words.add("kubectl")?;
// A word seen in code keeps its place even with only two letters.
words.add_code_word("rg")?;
for (word, why) in model.unusable_special_words(&words) {
    eprintln!("{word} is left out. {why}");
}
let text = model.transcribe_biased(&samples, &words)?.text;
```

Give each word in its usual spelling, composed (NFC), as the user writes it. A list given
"mcp" favours "mcp", and one given "MCP" favours "MCP". Several words separated by spaces
("Visual Studio Code") are favoured together. A stream takes a list with
`Stream::set_special_words` or `Streamer::set_special_words`, and the list applies from the next
utterance on, so it can change between utterances. With an empty list both models decode
exactly as `transcribe` does.

Special words must be written with spaces between words. Japanese, Chinese, Thai and other
scripts written without spaces aren't supported yet, and the list refuses their characters.

## Limits

`SpecialWords::add` and `add_code_word` return a `Rejection` for a word the list refuses, and
`Model::unusable_special_words` lists the words a model's vocabulary can't spell.

| Limit | Value |
|---|---|
| Words in a list | 1,000 (`SpecialWords::MAX_WORDS`) |
| Characters in a word, spaces included | 64 (`MAX_CHARS`) |
| Words separated by spaces in one entry | 5 (`MAX_PARTS`) |
| Tokens of a spelling in the model's vocabulary | 32 (`MAX_TOKENS`), longer spellings are left out |
| Letters or digits, unless added with `add_code_word` | at least 3 |
| Characters | no control characters, combining marks or scripts written without spaces |

The tokenizer files' Unicode normalization isn't applied to the words. Decomposed accents
("e" followed by a combining acute) are refused, and compatibility characters such as
full-width letters aren't folded.

## How it works

Each word is split into the model's tokens as written and capitalized ("kubectl" and
"Kubectl"), with the word-start marker on its first token.

Granite decodes greedily, then spots each special word in the CTC output. A dynamic program
over the frames finds the word's best alignment with a free start and end, keeping apart the
paths that have covered a greedy word and those that haven't, and compares it with the greedy
path over the same frames. Where the alignment plus a bonus for each of its tokens beats the
greedy path by a margin, the word replaces the greedy words there, which it must cover whole, at
least one of them. Where spotted words overlap, the best one wins. In a stream, a spotted word
settles whole or not at all.

Parakeet decodes with a beam search over a prefix tree of the words' tokens with failure links
(Aho-Corasick). A search state is the text with the frame the next step reads, the tokens
already emitted on that frame and where the text stands in the tree. Equal states merge and the
best 4 survive each step. Each hypothesis expands to the blank and to the 4 tokens whose
probability plus change in bonus is best, each with every duration. The hypotheses at the
earliest frame expand together, so the joint network and the prediction network read their
weights once for all of them.

A token of a special word under way earns a provisional bonus. A word becomes final when a new
word, punctuation or the end of the audio follows it, and then each of its tokens keeps the
bonus, once even where completed words overlap ("New" and "New York"). A word that ends inside a
longer one that doesn't complete keeps its own bonus ("York" in a list with "New York city"),
and the rest of the provisional bonus goes. A hypothesis that reaches the end of the audio takes
its final score at once. The final hypothesis is the one with the best score per token, as in
NVIDIA's NeMo, because a path that skips speech with long blanks can otherwise score as well as
the right text.

Some rules keep ordinary words safe.

- Words of one or two letters or digits are refused unless added with `add_code_word`.
- A word that is a single token of the vocabulary is one the model knows, so it earns half the
  bonus. For Granite that means it is never spotted at the default margin.
- A spotted word must beat what Granite already heard by the margin, and a short word, which
  carries little sound, by more. A word with fewer than five letters or digits adds the
  short-word margin once for each letter it lacks, so at the defaults a word of four letters
  needs 2 + 3 = 5 and one of three letters 2 + 3 × 2 = 8.

`Boost::new` sets the strength, `Boost::with_short_word_margin` the short-word margin, and
`SpecialWords::set_boost` gives them to a list.

| Setting | Default | Range | Meaning |
|---|---:|---|---|
| `bonus` | 1.5 | 0 to 10 | Parakeet's bonus for each token of a special word, in log probability |
| `beam` | 4 | 1 to 16 | Hypotheses Parakeet's beam search keeps |
| `spotting_bonus` | 4 | 0 to 10 | Granite's bonus for each token of a spotted word |
| `spotting_margin` | 2 | 0 to 50 | How much a spotted word with its bonus must beat the greedy words |
| `short_word_margin` | 3 | 0 to 50 | What a spotted word adds to that for each letter or digit it has below five |

## Accuracy

The numbers are word error rates (WER) with the Open ASR leaderboard's English normalizer, for
Granite with int8 weights and Parakeet with q8_0, each against greedy decoding of the same audio.
Changes come with the 95% interval of a paired bootstrap over the utterances.

Rare words are words that a 170,000-word English list (SCOWL size 70, with British spellings)
doesn't know, mostly names such as Fauchelevent, Nyiragongo or Tendulkar. Each test transcribes
the utterances that have them in four groups (by chapter, or by sentence for FLEURS), each with a
list of 300 words, the group's own rare words and rare words of other sets that don't occur in
it. Ordinary speech is transcribed with a list of 300 identifiers from a Rust code base that the
word list doesn't know, such as kubectl, serde, tokio, impl, cfg, mutex, uuid and ratatui.

The settings were tuned on LibriSpeech test-clean and test-other, then checked on sets they
weren't tuned on, against limits written down before each run. The WER of ordinary speech may
rise by at most 0.05 points, with the upper end of its interval at most 0.10. Recall of rare
words must rise by at least 10 points, and the WER of their utterances must not rise. At most 2
listed words in 1,000 ordinary utterances may take the place of other words.

### FLEURS English

FLEURS is an evaluation set for both models, so they shouldn't have been trained on it. All 1,041
utterances of its English dev and test splits served as ordinary speech, and the 181 with rare
words (293 occurrences of 133 words) as the rare-word test. Both models met every limit.

| | Granite | Granite with the list | Parakeet | Parakeet with the list |
|---|---:|---:|---:|---:|
| Rare words transcribed right | 34.5% | 68.9% | 32.1% | 46.8% |
| WER of the rare-word utterances | 11.98% | 9.54% | 12.30% | 11.36% |
| WER of all 1,041 utterances | 6.38% | 6.38% | 6.59% | 6.53% |
| Listed words put in place of other words | | 0 | | 0 |

The change in WER of all utterances was +0.00 points for Granite and −0.06 (−0.14 to +0.02) for
Parakeet.

### LibriSpeech dev

The first check, on dev-clean (2,703 utterances) and dev-other (2,864), with 874 rare-word
utterances, came before Granite's short-word margin. Parakeet met every limit, its interval on
dev-other just inside. Granite met the WER and recall limits but put 22 listed words in place of
others in the 5,567 utterances, 4 in 1,000, nearly all identifiers of three or four letters
("mamma" became "mma", "lean" became "lan", "say" became "sha"). The short-word margin came out of
that. With it, Granite put 1 listed word in place of another on dev, which no longer counts as a
check, since dev had been seen by then.

| | Granite | Granite with the list | Parakeet | Parakeet with the list |
|---|---:|---:|---:|---:|
| Rare words transcribed right | 59.8% | 86.8% | 43.7% | 72.7% |
| WER of the rare-word utterances | 3.95% | 2.39% | 5.53% | 3.78% |
| WER change, dev-clean | | +0.01 (+0.00 to +0.02) | | −0.09 (−0.20 to −0.01) |
| WER change, dev-other | | +0.01 (+0.00 to +0.02) | | +0.03 (−0.03 to +0.10) |
| Listed words put in place of other words | | 22 | | 2 |

### LibriSpeech train-clean

The second check took 114 chapters of train-clean-100 at random, with 1,399 rare-word utterances
and the first 2,000 utterances as ordinary speech. Both models met every limit, but they have
most likely been trained on these recordings, which makes the test lenient.

| | Granite | Granite with the list | Parakeet | Parakeet with the list |
|---|---:|---:|---:|---:|
| Rare words transcribed right | 84.9% | 95.0% | 49.2% | 74.3% |
| WER of the rare-word utterances | 1.51% | 1.12% | 3.82% | 2.74% |
| WER of 2,000 utterances | 0.82% | 0.82% | 2.06% | 2.05% |
| Listed words put in place of other words | | 0 | | 0 |

### Choosing the settings

The settings come from LibriSpeech test-clean and test-other. Parakeet's bonus was chosen on half
of the rare-word groups and 1,000 ordinary utterances.

| Parakeet bonus | Rare words right | Ordinary WER change, clean and other | Listed words inserted |
|---|---:|---|---:|
| greedy | 45.3% | | 0 |
| 1 | 67.3% | −0.03, −0.11 | 0 |
| 1.5 (default) | 71.2% | +0.00, −0.03 | 1 |
| 2 | 75.0% | +0.03, +0.03 | 3 |

A bonus of 2 went beyond the interval on part of test-other in an earlier version of the search,
so the default is 1.5. Granite's short-word margin was chosen on both halves of the rare-word
groups and all 5,559 ordinary test utterances, with a spotting bonus of 4 and a margin of 2.

| Granite short-word margin | Rare words right | Of them, 4-letter words | Listed words inserted in 5,559 utterances |
|---|---:|---:|---:|
| greedy | 65.4% | 56.2% | |
| 0 | 85.8% | 76.4% | 7 |
| 1 | 85.6% | 73.0% | 1 |
| 3 (default) | 85.6% | 73.0% | 0 |
| 4 | 85.1% | 67.4% | 0 |

Words of five letters or more aren't affected by it. For Granite, CTC prefix beam search over the
same token tree was tried as well, and raised the WER of ordinary speech where word spotting left
it unchanged, so word spotting is what Granite uses.

### Other languages

On the first 100 FLEURS test utterances of the other languages, the identifier list put no listed
word in place of another.

| Language | Granite | Granite with the list | Parakeet | Parakeet with the list |
|---|---:|---:|---:|---:|
| German | 6.56% | 6.56% | 6.13% | 6.30% |
| French | 9.52% | 9.52% | 6.27% | 6.38% |
| Spanish | 5.87% | 5.87% | 4.70% | 4.70% |
| Italian | 7.41% | 7.41% | 5.43% | 5.39% |
| Portuguese | 11.12% | 11.12% | 7.77% | 7.73% |
| Japanese, character error rate | 5.65% | 5.65% | | |

Parakeet's changes lie inside their intervals, which are wide on 100 utterances (German +0.17,
−0.04 to +0.44).

### Spoken technical terms

People say "nginx" as "engine x" and "kubectl" as "cube control". To see how far biasing gets
there, espeak-ng read 21 sentences with eight technical terms in three voices, once with each
term as written (which espeak reads as it guesses from the spelling) and once with the term in
the words people say. The synthetic speech is robotic, and the models get about 14% of its words
wrong, so these numbers are only a rough guide.

| Terms transcribed as written | Granite | Granite with the list | Parakeet | Parakeet with the list |
|---|---:|---:|---:|---:|
| espeak reading the spelling | 44% | 70% | 49% | 54% |
| espeak saying the spoken form | 44% | 60% | 38% | 51% |

Words close to their spelling come out right ("Graphfana" becomes Grafana, "Kbectel" Kubectl).
Words heard as other words mostly stay as they are. "serde" read aloud stayed "third" or
"survey", "tokio" mostly stayed "Tokyo", and "cube control" sometimes became cubecl, another
identifier on the list.

### Streaming

50 rare-word utterances of LibriSpeech test-clean and test-other went through `UtteranceStream`
in steps of half a second, with a pass after each step and a final pass at the end.

| | Granite | Granite with the list | Parakeet | Parakeet with the list |
|---|---:|---:|---:|---:|
| Rare words transcribed right, streamed | 66.7% | 85.7% | 38.1% | 73.0% |
| Rare words transcribed right, offline | 66.7% | 85.7% | 39.7% | 77.8% |
| WER, streamed | 4.58% | 3.65% | 6.52% | 4.42% |
| WER, offline | 4.19% | 3.18% | 6.21% | 4.19% |

## Speed

On 3 threads of the machine in [measurements.md](measurements.md#machine), with other work
running, Granite's spotting with 300 words added about 75 ms per minute of audio to the 200 ms
its CTC head and greedy decoding took. Parakeet's beam search took about 860 ms per minute of
audio where greedy decoding took 280 ms. The encoders took 6.8 and 7.5 seconds per minute of
audio, so the list added about 1% to Granite's time and 8% to Parakeet's. In the streaming test
above, the passes took 2% longer with the list for Granite and 5% longer for Parakeet.

## Gaps

- A word the model hears as other words stays unfound. People say "nginx" as "engine x" and
  "kubectl" as "cube control", and favouring the written spelling can't turn those into it.
  The synthetic sentences above show it, where "serde" read aloud became "third" or "survey".
- Scripts written without spaces, and a word boundary shared by spotting and streaming for them,
  are left for a later release.
