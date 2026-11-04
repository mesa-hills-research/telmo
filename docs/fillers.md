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
| A hesitation sound, the whole word in lower case or with one capital, and ah in lower case only | um, umm, uh, uhh, uhm, er, erm, ah, hm, hmm, mm, mmm |
| The first of two copies of a word that can't end a clause, side by side in any case, with a comma or a hesitation between them or nothing | the the, a a, and and, I, I, we we, it's it's, I'm I'm |

The words of the second row are a, an, the, and, or, but, I, my, our, your, their, we, they,
he, she, it's, that's, there's, what's, let's, I'm, we're, they're, you're, he's, she's, I've,
we've, they've, you've, I'll, we'll, they'll, you'll and it'll. A period, question mark or
exclamation mark between the copies keeps both.

Punctuation on a removed word goes with it, except a sentence end, which moves to the word
before ("We go there um. And then" becomes "We go there. And then"). A hesitation set off by
commas takes the comma before it too when the word before can't end a phrase ("click the, uh,
little arrow" becomes "click the little arrow"). Those words are the second row's, the
prepositions, "to", the auxiliaries and modals, and the determiners other than "no". After any
other word the comma marks a boundary and stays ("Okay, um, so" becomes "Okay, so"). When the
removed word had a capital at the start of a sentence, the next word takes it if it's in lower
case ("Um, so I think" becomes "So I think"). Other words keep their spelling.

## What stays

- "mm" after a number, as millimetres, and a hesitation in capitals (UM, ER), as an acronym.
- "Ah" with a capital. It's usually the interjection ("Ah, that's why"), which both models write
  with a capital even in the middle of a sentence ("to you, Ah, madam"). A hesitation a model
  writes with a capital, as at the start of a sentence, stays too. That's on purpose, since a
  hesitation left in costs less than a meant word lost.
- A quoted word, as a mention ("I often go “Um, so”", "'um'"). That's a word from an opening
  quote mark to the closing one of its kind, or to the end of the utterance when none comes. A
  closing mark alone quotes nothing, so "uh'" goes. A straight ' or a ’ standing alone quotes
  nothing either, though it closes a quote that's open, since a recognizer writes one now and then
  as noise ("So so ' okay i i might" loses an "i"). Neither does the apostrophe of an elision from
  a short list ('bout, 'cause, 'cos, 'cuz, 'em, 'kay, 'n, 'neath, 'nother, 'round, 'til, 'tis and
  'twas) or of a contraction's ending written apart from its word ('s, 'll, 're, 've, 'd, 'm and
  't, as in "we 'll"), so the "um" in "'cause um" goes too. A straight ' or a ’ right before one
  of these words is taken for its apostrophe even where it could open a quote, so a quote in
  straight marks that begins with one ('re um…') isn't kept. Other marks, such as ‘ and ‹, always
  open a quote.
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
so tentative text never shows a hesitation that goes, and a quote that began in the settled text
still keeps its words. A word that a special word of several words may still take in waits to settle until the
special word is heard whole or something else comes, so its fillers stay with it. A word whose fate
depends on the next one waits for it, because the last word of a pass never settles, and a word
followed only by hesitations is the last. When the first copy of a stutter has settled, the second
goes instead. A sentence end left by a hesitation follows the settled text, unless that ends in a
comma, and a comma already settled stays even where a hesitation after it would take it. A pass often hears the settled words again, and a stream drops those, but a word said twice
("no, no") keeps both copies. Each settled word stands for one word of the pass, heard on its audio
or starting less than 240 ms after it, and any other copy was said again.

An utterance goes on from the last words the one before it settled, so a stutter split by a pause
loses its second copy ("to design the" then "the product"), unless a sentence end comes between
them. A quote the earlier utterance left open doesn't go on. When the earlier utterance's sentence
end turns out to go on with the next words, a copy of its last word that opens them goes with the
mark ("and." then "and then" gives "and then").

## Accuracy

On meeting speech, 781 utterances (69 minutes) of the AMI corpus's IS1009 meetings recorded with
headsets, transcribed offline.

| | Granite | Parakeet |
|---|---:|---:|
| Words left out | 565 | 562 |
| Hesitations written for a spoken um or uh that went | 430 of 430 | 428 of 428 |
| Stutters of the listed words written twice that went | 57 of 57 | 57 of 57 |
| Meant words left out | 0 | 0 |

Granite also left out two words and Parakeet one that the reference counts as meant, each a copy
of a stutter the model wrote once more than the reference spells it ("a a a sophisticated chip"
for "a a sophisticated chip"). Where a model wrote a hesitation for a word ("uh" for "a"), the
word was lost with it, 15 times with Granite and 17 with Parakeet. Each model also left an "Ah,"
that starts a sentence and that AMI's annotators mark as a hesitation.

Streamed in steps of half a second, on 50 of those utterances, no settled or tentative text
showed a hesitation and no stutter was left, and the words left out were those offline left out
wherever the stream wrote the same words as offline.

On read speech, the rule as it was before "Ah" with a capital stayed changed the text of 24 of
LibriSpeech's 5,567 dev utterances with Granite and 27 with Parakeet. In 20 of each, the change
was an "Ah" of the book's text ("Ah, maiden, said he"), and four more were stutters the books
print ("I I don't know"). With the rule as it is now, 6 of 2,912 LibriSpeech train-clean-100
utterances changed with Granite and 8 of 2,904 with Parakeet. All but one left out a hesitation
or a stutter the book prints ("We—er—I came", "I—I did not know") or a word the model wrote
twice, and the other was an "ah" Parakeet wrote in lower case where the book meant it.
