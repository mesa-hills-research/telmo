# Changelog

## 0.3.3

### Filler removal

- A list of special words can leave out English hesitations (um, uh, erm, hmm) and stutters of
  words that can't end a clause ("the the", "I, I"), offline and streamed, with
  `words.set_fillers(Fillers::for_language("en"))`. `Fillers::remove` does the same to text, and
  `telmo transcribe --remove-fillers` to a recording. Quoted words, special words, "Ah" with a
  capital and meant doubles such as "had had" stay. [docs/fillers.md](docs/fillers.md) has the
  rules.
- A comma before a removed hesitation set off by commas goes with it after a, an, the, my, our,
  your, their, its, and, or and but ("click the, uh, little arrow" becomes "click the little
  arrow"). After any other word it stays.
- A lone straight single quote mark or ’ opens no quote, since recognizers write one as noise.
- In a stream, removal goes on from the settled text, and each utterance goes on from the words
  the one before it settled, so a stutter split by a pause still loses its second copy.

On 2,444 sentences written to read like dictation to a coding assistant, removal deleted no
meaningful word, and 99.1% of the 862 sentences it changed came out as two labellers expected,
with a comma after the last kept word before a removed hesitation counted as optional. In
streamed meeting speech (AMI), both copies of a deliberately repeated word ("yeah, yeah") were
kept 94.0% of the time with Granite and 93.8% with Parakeet, short of the 95% aimed for. Most of
the lost copies are places where the stream heard different words from offline transcription.

### Streaming

- Parakeet's words settle at least 4 s after they end while the utterance is within the 30 s a
  pass encodes whole, and nothing settles before the utterance has 3 s of audio. Parakeet often
  hears a word said twice as one until it has heard more, so more deliberate repeats keep both
  copies (96.4% against 94.4% on AMI meeting clips). Words now stay tentative for about 4 s
  rather than 1.3 s, and more tentative words are replaced before they settle. The first
  tentative word and the end of an utterance come as before. A `PassConfig::settle_margin` below
  4 s no longer applies to Parakeet within those 30 s.
- A sentence end that closes an utterance waits until the next utterance shows whether the
  sentence goes on, and a new sentence's first word gets a capital where Granite wrote it in
  lower case.
- A word said twice keeps both copies after the first has settled, and each settled word stands
  for one word of a later pass, so a word heard again isn't written twice.
- Words that a special word of several words may still take in wait to settle until it's heard
  whole.
- Voice activity keeps the first words of audio that opens in speech, and works on audio shorter
  than its 300 ms opening window.
- The latency table in [docs/streaming.md](docs/streaming.md) is measured again, with Parakeet's
  old 1 s margin beside the new one.
