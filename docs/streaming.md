# Streaming

`Stream` transcribes audio as it arrives, on a thread of its own. Push audio in pieces of any
size at any sample rate, and text comes back as events. Settled words never change and are
appended in order. The tentative words after them are replaced by each pass. Streaming is
experimental, so its API and behaviour may change in any release.

```rust
use std::sync::Arc;
use telmo::stream::{Event, Stream, StreamConfig};
use telmo::{Model, Options};

let model = Arc::new(Model::load("granite-int8.pack".as_ref(), &Options::default())?);
let config = StreamConfig { input_rate: 48000, ..StreamConfig::default() };
let stream = Stream::start(model, config, |event| match event {
    Event::Settled { text, .. } => print!("{text}"),
    Event::Tentative { text, .. } => eprintln!("[{text}]"),
    _ => {}
});
// For each piece of microphone audio:
stream.push(&samples);
// At the end:
stream.finish()?;
```

## How it works

1. The audio is resampled to 16 kHz and split into utterances by voice activity. An utterance
   starts after 150 ms above an adaptive noise floor and keeps the 300 ms before that. The
   floor starts at the quietest of the first 300 ms, or of all the audio when it ends sooner.
   Audio that opens in the middle of a word keeps that word when speech is found in its first
   2 s, since its first utterance then starts at the first sample. An utterance ends after
   600 ms below the floor once the model has heard no word in the last 400 ms, and one longer
   than 30 s is cut at its quietest point. After a pause of up to 2 s the next utterance starts
   where the last one's audio ends, so words said too quietly for voice activity between the two
   are in the next one rather than lost, and no audio is in both.
2. Passes over the current utterance run at least `min_interval` (0.5 s) apart, and at least
   1.5 times the last pass's duration apart, so a slow computer shows tentative text less often
   instead of falling behind.
3. Parakeet re-encodes the utterance so far, decodes it from its start, and keeps the tokens
   after the settled text. With less of the utterance before it, a window that starts in the
   middle of a sentence sometimes decodes to nothing. Granite keeps its encoder's state and
   recomputes only the attention blocks new audio reaches, keeping the previous 10.24 s block
   live until the next one has 1.28 s.
4. A word settles once two passes in a row agree on it and on every word before it, it ends a
   second before the audio does, and the model has heard a word after it. Parakeet's hypothesis
   of a word can still change until it has heard several seconds after it, and often hears a word
   said twice ("yeah, yeah") as one before then. So with Parakeet a word must end 4 s before the
   audio does, and nothing settles before the utterance has 3 s of audio. An UtteranceStream fed
   more than 30 s, which voice activity never makes, keeps the 1 s margin past 30 s, since its
   windows then start inside the utterance, where Parakeet sometimes hears nothing. On 1,695
   streamed AMI meeting clips holding a word said twice, Parakeet wrote both copies of 96.4% of
   those doubles with the 4 s margin against 94.4% with 1 s, and a word settled a median 4.0 s
   after it ended instead of 1.3 s. Tentative text shows as early as before. When the utterance
   ends, a last pass settles everything. With a list that leaves out
   [disfluencies](disfluencies.md), nothing settles before then, and the last pass settles the
   whole utterance cleaned.
5. A sentence end that closes an utterance stays tentative until the next utterance's first
   words settle, or the stream finishes, since a pause can split a sentence. It goes when they
   start with a comma or a lower-case word that joins words, such as "and", "rather", "of" or
   "which", and stays otherwise. A new sentence's first word gets a capital where Granite wrote
   it in lower case. On 261 such boundaries in streamed meeting speech (AMI) and 148 in
   LibriSpeech, the boundaries the rule was chosen on, the text was then right at 78.9% and
   93.2%, against 73.6% and 91.2% left as it was.

`Streamer` is the same machinery without the thread, for callers that decide themselves when
passes run.

## Latency and accuracy

Ten recordings of about two minutes each, made of consecutive LibriSpeech test-clean or
test-other utterances with pauses of 0.8 to 2 s and quiet noise (163 utterances in all), were
replayed at 48 kHz in 20 ms pieces with a clock that advances with the audio and with each pass's
measured time, on 5 threads of the machine described in
[measurements.md](measurements.md#machine), with `TELMO_ISA=avx2`. `examples/stream` builds and
replays such recordings (`stream run MODEL sessions.tsv --threads 5`).

| | Granite, int8 | Parakeet, q8_0 |
|---|---:|---:|
| First tentative word after speech starts, median | 0.78 s | 1.07 s |
| First tentative word, 90th percentile | 1.11 s | 3.57 s |
| Words settled after speech ends, median | 1.20 s | 1.77 s |
| Words settled, 90th percentile | 1.75 s | 3.14 s |
| Tentative words replaced per 100 settled | 68 | 107 |
| WER streamed | 2.44% | 2.62% |
| WER offline, on the same utterances | 2.12% | 2.44% |
| CPU | 1.0 cores | 1.1 cores |

The offline row transcribes each utterance the stream found in one go. The words have settled
when the utterance ends, and most of that wait is the 600 ms of quiet that ends it. A sentence
end that closes the utterance settles later, with the next utterance's first words, or when the
stream finishes. Passes run at least 0.5 s apart by default (`StreamConfig::min_interval`). Passes
further apart show the first word later, replace fewer tentative words and use less CPU. Before
Parakeet's 4 s margin its words settled a median 1.53 s after speech ended (2.77 s at the 90th
percentile), it replaced 100 tentative words per 100 and its streamed WER was 2.76%.

FLEURS recordings of six or seven test utterances per language gave these word error rates, where
a single word moves a figure by about a point. Offline here means transcribing the original
clips.

| Language | Granite, streamed | Granite, offline | Parakeet, streamed | Parakeet, offline |
|---|---:|---:|---:|---:|
| German | 1.5% | 3.9% | 1.5% | 2.3% |
| Spanish | 5.9% | 5.3% | 5.3% | 4.6% |
| French | 14.4% | 8.1% | 13.2% | 7.5% |

The French recording starts with a clip about 35 dB quieter than the others, barely above the
added noise, and voice activity missed its first second. Clicks and breaths before the speech in
some FLEURS clips start an utterance early, so the first word takes longer there, a median of
0.76 s with Granite and 1.27 s with Parakeet, and 2.9 s and 2.3 s at the 90th percentile.
