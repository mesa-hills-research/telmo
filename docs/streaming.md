# Streaming

`Stream` transcribes audio as it arrives, on a thread of its own. Push audio in pieces of any
size at any sample rate, and text comes back as events. Settled words never change and are
appended in order. The tentative words after them are replaced by each pass. Streaming is
experimental, so its API and behaviour may change in any release.

```rust
use std::sync::Arc;
use notate::stream::{Event, Stream, StreamConfig};
use notate::{Model, Options};

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
   starts after 150 ms above an adaptive noise floor and keeps the 300 ms before that. It ends
   after 600 ms below the floor once the model has heard no word in the last 400 ms, and an
   utterance longer than 30 s is cut at its quietest point.
2. Passes over the current utterance run at least `min_interval` (0.5 s) apart, and at least
   1.5 times the last pass's duration apart, so a slow computer shows tentative text less often
   instead of falling behind.
3. Parakeet re-encodes the utterance so far, decodes it from its start, and keeps the tokens
   after the settled text. With less of the utterance before it, a window that starts in the
   middle of a sentence sometimes decodes to nothing. Granite keeps its encoder's state and
   recomputes only the attention blocks new audio reaches, keeping the previous 10.24 s block
   live until the next one has 1.28 s.
4. A word settles once two passes in a row agree on it and on every word before it, it ends a
   second before the audio does, and the model has heard a word after it. When the utterance
   ends, a last pass settles everything.

`Streamer` is the same machinery without the thread, for callers that decide themselves when
passes run.

## Latency and accuracy

Ten recordings of about two minutes each, made of consecutive LibriSpeech test-clean or
test-other utterances with pauses of 0.8 to 2 s and quiet noise (163 utterances in all), were
replayed in real time at 48 kHz in 20 ms pieces, on 5 threads of the machine described in
[measurements.md](measurements.md#machine). `examples/stream` builds and replays such
recordings.

| | Granite, int8 | Parakeet, q8_0 |
|---|---:|---:|
| First tentative word after speech starts, median | 0.77 s | 0.75 s |
| First tentative word, 90th percentile | 0.97 s | 1.81 s |
| Final text after speech ends, median | 1.13 s | 1.09 s |
| Final text, 90th percentile | 1.73 s | 1.69 s |
| Tentative words replaced per 100 settled | 46 | 63 |
| WER streamed | 2.33% | 2.69% |
| WER offline, on the same utterances | 2.12% | 2.44% |
| CPU | 1.2 cores | 1.8 cores |

The offline row transcribes each utterance the stream found in one go. Most of the wait for the
final text is the 600 ms of quiet that ends an utterance. Passes run at least 0.5 s apart by
default (`StreamConfig::min_interval`). At 1 s apart, Granite's first word comes after 1.29 s
instead, it replaces 26 tentative words per 100 and uses 0.85 cores, at the same WER.

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
