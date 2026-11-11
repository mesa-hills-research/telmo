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
   than 30 s is cut at its quietest point. When the next onset comes at most 2 s after the
   last utterance's speech ended, the stream also keeps the 2 s of audio before the next
   utterance, the end of the last one and the pause (step 5).
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
   audio does, and nothing settles before the utterance has 3 s of audio. Past 30 s of audio the
   margin is 1 s again, since a window can then start inside the utterance, where Parakeet
   sometimes hears nothing. That happens with an UtteranceStream fed more than 30 s, and with an
   utterance that goes on after voice activity ended it because the model still heard a word
   there, since voice activity counts its 30 s from the new start. On 1,695 streamed AMI meeting
   clips holding a word said twice, Parakeet wrote both copies of 96.4% of those doubles with the
   4 s margin against 94.4% with 1 s, and a word settled a median 4.0 s after it ended instead of
   1.3 s. Tentative text shows as early as before. When the utterance ends, a last pass settles
   everything. With a list that leaves out [disfluencies](disfluencies.md), nothing settles
   before then, and the last pass settles the whole utterance cleaned.
5. A sentence end that closes an utterance stays tentative until the next utterance's first words
   settle, or the stream finishes, since a pause can split a sentence. After a pause of at most 2 s,
   the pass that first settles the next utterance's words also decodes, once, the 2 s of audio
   before it with up to its first 3 s, so the model hears both sides of the pause, and the mark
   follows what it hears there. With Parakeet the mark stays if Parakeet writes a sentence end in
   the pause. Granite seldom writes one at a pause whose both sides it hears, even between
   sentences, so the mark stays if Granite's probability of a sentence end on the pause's frames
   reaches 0.003. That threshold was chosen on dictation and on the AMI windows below, where it gets
   as many pauses right as 0.0003, the best value on the same meetings' pauses decoded offline.
   Where the sentence goes on, the next utterance's first word takes the case the model gave it
   there when the model wrote the same word, passing over hesitations before it, so "Sometimes"
   after "because" becomes "sometimes". After a longer pause the mark goes when the next words
   start with a comma or a lower-case word that joins words, such as "and", "rather", "of" or
   "which", and stays otherwise. Either way, a new sentence's first word gets a capital where the
   model wrote it in lower case. Unless the speaker starts over, only the mark and the case
   change, and the words are the ones the passes settled. After a pause of any length, when the
   next utterance's first words say the last words before the pause again, after a hesitation or
   an editing word if any ("we should ship it, I guess.", then "Well, I guess we should"), the
   sentence goes on, whatever the model heard. A comma marks the pause, and with a tagger the copy
   and the words before it go instead ("we should ship it, I guess we should"). A copy counts only
   where the clause was cut short, so "the budget.", then "The budget is", and echoes such as
   "Okay.", then "Okay", keep their sentence ends. Streamed, the text at short pauses then had the
   mark and the case right this often.

   | | Granite, before | Granite, after | Parakeet, before | Parakeet, after |
   |---|---:|---:|---:|---:|
   | 47 AMI windows, pauses AMI's punctuation places (140, 128) | 50.0% | 65.0% | 68.8% | 76.6% |
   | IS1009's turns joined with pauses (246, 59) | 78.9% | 80.5% | 83.1% | 84.7% |

`Streamer` is the same machinery without the thread, for callers that decide themselves when
passes run.

## Latency and accuracy

Ten recordings of about two minutes each, made of consecutive LibriSpeech test-clean or
test-other utterances with pauses of 0.8 to 2 s and quiet noise (163 utterances in all), were
replayed in real time at 48 kHz in 20 ms pieces, on 5 threads of an otherwise idle AMD EPYC 7642,
with `TELMO_ISA=avx2`. `examples/stream` builds and replays such recordings
(`stream run MODEL sessions.tsv --threads 5 --realtime`). The last column is Parakeet with the
1 s settle margin it had before the 4 s floor, measured on the same machine at the same time.
Granite's figures were the same then.

| | Granite, int8 | Parakeet, q8_0 | Parakeet, 1 s margin |
|---|---:|---:|---:|
| First tentative word after speech starts, median | 0.73 s | 0.75 s | 0.75 s |
| First tentative word, 90th percentile | 0.74 s | 1.82 s | 1.82 s |
| Words settled after speech ends, median | 0.90 s | 1.05 s | 1.05 s |
| Words settled, 90th percentile | 1.14 s | 1.70 s | 1.63 s |
| Tentative words replaced per 100 settled | 74 | 145 | 116 |
| WER streamed | 2.59% | 2.51% | 2.91% |
| WER offline, on the same utterances | 2.15% | 2.44% | 2.44% |
| CPU | 1.6 cores | 2.0 cores | 2.0 cores |

The table was measured before the stream decided sentence ends at short pauses (step 5), which costs
one extra decode of up to 5 s of audio per utterance after such a pause. Replayed beside the stream
without it, with the simulated clock on [the shared machine](measurements.md#machine) while other
work loaded it, words settled 0.22 s later at the median with Parakeet. Granite's difference was
smaller than the 0.2 s by which two runs of the same stream differed under that load, so these
figures are approximate. The words stayed the same, apart from the few that differ between any two
runs.

The offline row transcribes each utterance the stream found in one go. The words have settled
when the utterance ends, and most of that wait is the 600 ms of quiet that ends it. A sentence
end that closes the utterance settles later, with the next utterance's first words, or when the
stream finishes. Passes run at least 0.5 s apart by default (`StreamConfig::min_interval`). Passes
further apart show the first word later, replace fewer tentative words and use less CPU. The 4 s
margin leaves the first tentative word and the end of an utterance where they were, since the
last pass settles everything, and words stay tentative longer, so more of them are replaced.

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
