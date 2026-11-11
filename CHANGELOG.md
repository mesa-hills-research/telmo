# Changelog

## 0.4.0

### Disfluency cleanup

- `disfluency::Tagger` leaves false starts, repairs, repeated phrases and filler words out of
  English text, once over each whole utterance, with `words.set_disfluencies(Some(tagger))`. It
  only deletes, so every word left is one the speech model wrote. Its pack is Ettin encoder 68M
  fine-tuned on the AMI Meeting Corpus's annotations, in q8_0 (158 MB), from this repository's
  `disfluency-tagger-1` release. `disfluency::PACK_DOWNLOAD` pins it by size and SHA-256, and
  `disfluency::PACK_NOTICE` says what it's made from and under which licences.
- `Tagger::hand_written()` cleans up with hand-set rules and no pack. Beside the fillers it takes
  out cut-off words and fragments, a word or two-word phrase said again, and sentence-opening and
  trailing markers, in well under a millisecond an utterance.
- `words.set_guards(Guards::for_language("en"))` keeps words that a cleanup would take where
  dictation needs them, such as spoken symbols, a unit after a number, a filler talked about as a
  word, a list's items, and "so" and "well" in their own sense. On 112,000 words of seven AMI
  meeting series they undo 58 of the tagger's 12,521 deletions.
- `words.set_punctuation(Punctuation::for_language("en"))` fixes capitals and punctuation after
  the cleanup, and `Punctuation::joining_more()` joins more of what a pause split.
- `telmo transcribe --remove-disfluencies` downloads the tagger on first use, after asking, and
  runs it with the guards. A pack put in the cache by hand is checked against the same pins before
  its first use.
- In a stream with a tagger, an utterance's words stay tentative until it ends, and its last pass
  settles the cleaned text.
- `RemoteFile` is a file to download with its URL, size and SHA-256.

[docs/disfluencies.md](docs/disfluencies.md) has the rules, the guards and the tagger's speed.

### Streaming

- After a pause of at most 2 s, the pass that first settles the next utterance's words also
  decodes the audio on both sides of the pause, once, and the sentence end at the pause follows
  what the model hears there. The next word takes the case the model gave it. On 47 windows of
  AMI meetings, the mark and the case at short pauses were right 65.0% of the time with Granite
  (50.0% before) and 76.6% with Parakeet (68.8% before). The extra decode settles Parakeet's words
  about 0.2 s later at the median.
- When an utterance starts by saying the last words before the pause again, after a hesitation or
  an editing word if any, the sentence goes on with a comma at the pause. With a tagger the copy
  and the words before it go. They stay when they start a special word or follow a "no".

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
