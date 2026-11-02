# Fillers

Both models write what they hear, hesitations and stutters included. For dictation, a list of
special words can leave out English fillers, offline and in a stream. The library keeps every
word unless asked.

```rust
use telmo::{Fillers, SpecialWords};

let mut words = SpecialWords::new();
words.set_fillers(Fillers::for_language("en"));
let text = model.transcribe_biased(&samples, &words)?.text;
```

`Fillers::for_language` takes an ISO 639-1 code with or without a region and gives `None` for
a language Telmo has no rules for, so a caller can pass its dictation language as it is.
English is the only one so far. The rules would delete real words elsewhere ("um acht Uhr" in
German). `Fillers::remove` applies the same rules to text. The command takes
`telmo transcribe --remove-fillers`.

## What goes

| Words | Examples |
|---|---|
| A hesitation sound, the whole word in lower case or with one capital | um, umm, uh, uhh, uhm, er, erm, ah, hm, hmm, mm, mmm |
| The first of two copies of a word that can't end a clause, side by side in any case, with a comma or a hesitation between them or nothing | the the, a a, and and, I, I, we we, it's it's, I'm I'm |

The words of the second row are a, an, the, and, or, but, I, my, our, your, their, we, they,
he, she, it's, that's, there's, what's, let's, I'm, we're, they're, you're, he's, she's, I've,
we've, they've, you've, I'll, we'll, they'll, you'll and it'll. A period, question mark or
exclamation mark between the copies keeps both.

Punctuation on a removed word goes with it, except a sentence end, which moves to the word
before ("We go there um. And then" becomes "We go there. And then"). When the removed word had
a capital at the start of a sentence, the next word takes it if it's in lower case ("Um, so I
think" becomes "So I think"). Other words keep their spelling.

## What stays

- "mm" after a number, as millimetres, and a hesitation in capitals (UM, ER), as an acronym.
- A quoted word, as a mention ("I often go “Um, so”", "'um'"). That's a word from an opening
  quote mark to the closing one of its kind, or to the end of the utterance when none comes. A
  closing mark alone quotes nothing, so "uh'" goes. Neither does the apostrophe of an elision from
  a short list ('bout, 'cause, 'cos, 'cuz, 'em, 'kay, 'n, 'neath, 'nother, 'round, 'til, 'tis and
  'twas) or of a contraction's ending written apart from its word ('s, 'll, 're, 've, 'd, 'm and
  't, as in "we 'll"), so the "um" in "'cause um" goes too.
- Both copies when a quote mark, bracket or sentence end comes between them, also on a removed
  hesitation ("I um. I think" becomes "I. I think").
- Every word of a special word or a spoken form's word, wherever the whole of it occurs ("the
  the" given as a name). The first word of each also keeps its case.
- Backchannels and interjections such as mm-hmm, uh-huh, oh and huh.
- Doubles of words that can end a clause ("chew it, it breaks", "put them in, in a minute") and
  meant doubles ("had had", "very very", "no no").
- Repeats of several words, false starts, repairs and words such as "so", "like" or "you know".

## In a stream

Each pass leaves out fillers from all its words before any settle, going on from the settled text,
so tentative text never shows a hesitation, and a quote that began in the settled text still keeps
its words. A word that a special word of several words may still take in waits to settle until the
special word is heard whole or something else comes, so its fillers stay with it. A word whose fate
depends on the next one waits for it, because the last word of a pass never settles, and a word
followed only by hesitations is the last. When the first copy of a stutter has settled, the second
goes instead. A sentence end left by a hesitation follows the settled text, unless that ends in a
comma. A pass often hears the settled words again, and a stream drops those, but a word said twice
("no, no") keeps both copies. Each settled word stands for one word of the pass, heard on its audio
or starting less than 240 ms after it, and any other copy was said again.

## Accuracy

On meeting speech, 781 utterances (69 minutes) of the AMI corpus's IS1009 meetings recorded with
headsets, transcribed offline.

| | Granite | Parakeet |
|---|---:|---:|
| Words left out | 566 | 563 |
| Hesitations written for a spoken um or uh that went | 430 of 430 | 428 of 428 |
| Stutters of the listed words written twice that went | 57 of 57 | 57 of 57 |
| Meant words left out | 0 | 0 |

Granite also left out two words and Parakeet one that the reference counts as meant, each a copy
of a stutter the model wrote once more than the reference spells it ("a a a sophisticated chip"
for "a a sophisticated chip"). Where a model wrote a hesitation for a word ("uh" for "a"), the
word was lost with it, 15 times with Granite and 17 with Parakeet.

Streamed in steps of half a second, on 50 of those utterances, no settled or tentative text
showed a hesitation and no stutter was left, and the words left out were those offline left out
wherever the stream wrote the same words as offline.

On read speech (LibriSpeech's dev sets, 5,567 utterances) the text of 24 utterances changed with
Granite and 27 with Parakeet. In 20 of each, the change was an "Ah" of the book's text ("Ah,
maiden, said he"), and four more were stutters the books print ("I I don't know").
