"""Generate test vectors for the Rust port of the leaderboard's normalizer and WER scoring.

Usage: python -I gen_vectors.py <leaderboard dir> <tests dir> [<full sweep output>]

Writes into <tests dir>:
- normalize-vectors.tsv, inputs and the leaderboard normalizer's output for them (hand-written
  lines from handwritten.txt beside this script, spoken numbers, random mixes of number words,
  digits and symbols),
- unicode-vectors.tsv, every character the Rust tables single out (lowercase mappings,
  decompositions, class boundaries, all decimal digits), normalized in runs,
- score-vectors.txt, jiwer's counts for random word sequences that the Rust test rebuilds from
  the same generator.
With a third argument, also writes every code point in runs to that file, for a one-off check.
"""

import importlib.metadata
import os
import random
import sys
import unicodedata

src_dir, tests_dir = sys.argv[1], sys.argv[2]
full_sweep = sys.argv[3] if len(sys.argv) > 3 else None
sys.path.insert(0, src_dir)

import jiwer  # noqa: E402

from normalizer.normalizer import EnglishTextNormalizer  # noqa: E402

normalizer = EnglishTextNormalizer()
here = os.path.dirname(os.path.abspath(__file__))


def escape(s):
    """Backslash escapes for the TSV files. Unprintable characters, and spaces at either end, become \\u{...}."""
    out = []
    for i, ch in enumerate(s):
        if ch == "\\":
            out.append("\\\\")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == " " and (i == 0 or i == len(s) - 1):
            out.append("\\u{20}")
        elif ch != " " and (not ch.isprintable() or ch.isspace()):
            out.append("\\u{%x}" % ord(ch))
        else:
            out.append(ch)
    return "".join(out)


def write_vectors(path, header, inputs):
    seen = set()
    count = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        for text in inputs:
            if text in seen:
                continue
            seen.add(text)
            f.write(escape(text) + "\t" + escape(normalizer(text)) + "\n")
            count += 1
    return count


# Hand-written lines, and a few that a text file can't hold.
handwritten = open(os.path.join(here, "handwritten.txt"), encoding="utf-8").read().split("\n")
handwritten += [
    "",
    " ",
    "   ",
    "\t",
    "\n",
    "\r\n",
    "a b",
    "non breaking space",
    "narrow no break",
    "soft­hyphen",
    "zero​width",
    "joiner‍here and non‌joiner",
    "left‎to right",
    "word⁠joiner",
    "line separator",
    "para separator",
    "next\u0085line",
    "vertical\u000btab",
    "form\u000cfeed",
    "file\u001cgroup\u001drecord\u001eunit\u001fseparators",
    "null\u0000char",
    "bell\u0007char",
    "delete\u007fchar",
    "ideographic　space",
    "em space and en space",
    "ogham space",
    "é decomposed",
    "café",
    "naïve",
    "à́̂̃ marks",
    "́ leading mark",
    "x⃝ enclosed",
    "devanagari किताब",
    "hangul 가각힣",
    "private  use",
    "noncharacter ￾",
    "replacement � char",
    "bom﻿here",
    "emoji \U0001F600 face",
    "flag \U0001F1FA\U0001F1F8",
    "family \U0001F468‍\U0001F469‍\U0001F467",
    "astral digits \U0001D7CE\U0001D7CF and \U0001D7D8",
    "um’s",
    "it’s",
    "don’t",
    "rock ’n’ roll",
    "‘tis",
    "o’clock",
    "mr smith",
    "5 %",
    "$ 5",
    " leading nbsp",
    "trailing nbsp ",
    "1 000",
    "one hundred",
    "and a half",
    "two and a half",
    "two and\ta\nhalf",
    "you 're",
    "you ’re",
    "you\t're",
    "you  'll",
    "I 've",
    "won' t",
]

ONES = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
SCALES = [(10**9, "billion"), (10**6, "million"), (1000, "thousand")]


def spell(n, and_word=False, hyphen=False):
    """A plain English spelling of n, to make inputs. It needn't be perfect English."""
    if n == 0:
        return "zero"
    words = []
    for value, name in SCALES:
        if n >= value:
            words.append(spell(n // value, and_word, hyphen) + " " + name)
            n %= value
    if n >= 100:
        words.append(ONES[n // 100 - 1] + " hundred")
        n %= 100
    if n:
        if words and and_word:
            words.append("and")
        if n < 20:
            words.append(ONES[n - 1])
        elif n % 10:
            sep = "-" if hyphen else " "
            words.append(TENS[n // 10 - 2] + sep + ONES[n % 10 - 1])
        else:
            words.append(TENS[n // 10 - 2])
    return " ".join(words)


rng = random.Random(20251029)
spoken = []
numbers = list(range(0, 130)) + [rng.randrange(130, 10**10) for _ in range(250)]
numbers += [1000, 1001, 1010, 1100, 2000, 2001, 2020, 2024, 10**6, 10**6 + 1, 10**9, 999999, 123456789]
for n in numbers:
    variant = rng.randrange(4)
    text = spell(n, and_word=variant & 1 == 1, hyphen=variant & 2 == 2)
    template = rng.choice(
        [
            "{}",
            "It was {} years ago.",
            "{} dollars",
            "{} percent",
            "about {} people",
            "{} point five",
            "the {} pounds",
            "minus {}",
            "{} and a half",
        ]
    )
    spoken.append(template.format(text))
for year in [1066, 1492, 1776, 1805, 1900, 1905, 1984, 1999, 2000, 2001, 2010, 2024]:
    hi, lo = divmod(year, 100)
    if lo == 0:
        spoken.append(f"In {spell(hi)} hundred, it rained.")
    elif lo < 10:
        spoken.append(f"In {spell(hi)} oh {spell(lo)}, it rained.")
    else:
        spoken.append(f"In {spell(hi)} {spell(lo)}, it rained.")
for _ in range(60):
    digits = [rng.randrange(10) for _ in range(rng.randrange(2, 11))]
    names = ["oh" if d == 0 and rng.random() < 0.6 else ("zero" if d == 0 else ONES[d - 1]) for d in digits]
    spoken.append("call " + " ".join(names))

# Random mixes of the tokens the number normalizer reacts to, to reach its odd corners.
TOKENS = (
    ["zero", "oh", "o", "first", "second", "third", "fifth", "eighth", "nineth", "ninth", "twelfth", "zeroth"]
    + ONES
    + TENS
    + ["sixes", "ones", "twos", "tens", "twenties", "thirtieth", "hundred", "thousand", "million", "billion", "decillion"]
    + ["hundreds", "thousandth", "millions", "point", "and", "a", "half", "double", "triple", "minus", "negative"]
    + ["plus", "positive", "dollar", "dollars", "cent", "cents", "pound", "pounds", "euro", "euros", "per", "percent"]
    + ["$", "£", "€", "¢", "%", ".", ",", "-", "'", "1", "2", "10", "007", "1.5", "3.14", "0.05", "1,000", "2:30"]
    + ["1st", "2nd", "3rd", "4th", "1990s", "5g", "a1", "the", "is", "it", "st", "s", "th", "nd", "rd", "1.0"]
    + ["100", "0", "00", "$5", "$0", "€3", "$1", "1s", "½", "²", "٣", "x", "2.00", "0.5", ".5", "5.", "1.2.3"]
    + ["Mr.", "Dr", "'s", "n't", "um", "uh", "(x)", "[y]", "won't", "ok", "colour", "and a half", " "]
)
fuzz = []
for _ in range(4000):
    n = rng.randrange(1, 13)
    parts = [rng.choice(TOKENS) for _ in range(n)]
    seps = [rng.choice([" ", " ", " ", " ", "", "  ", "-", ", "]) for _ in range(n - 1)]
    text = parts[0]
    for sep, part in zip(seps, parts[1:]):
        text += sep + part
    if rng.random() < 0.3:
        text = text.capitalize() + rng.choice([".", "?", "!", ""])
    fuzz.append(text)

count = write_vectors(
    os.path.join(tests_dir, "normalize-vectors.tsv"),
    "# Input and the leaderboard normalizer's output, tab-separated, with backslash escapes. "
    "Generated by scripts/gen_vectors.py with the leaderboard at commit "
    "6a3e6d4e3ec28b81aa5d9a15a5467e13e2b98622.\n",
    handwritten + spoken + fuzz,
)
print(f"normalize-vectors.tsv: {count} vectors")

# Every character the tables single out, in runs.
special = set(range(128))
for c in range(0x110000):
    if 0xD800 <= c < 0xE000:
        continue
    ch = chr(c)
    if ch.lower() != ch or ch.isdecimal():
        special.add(c)
    if unicodedata.normalize("NFKD", ch) != ch and not 0xAC00 < c < 0xD7A3:
        special.add(c)
prev = None
for c in range(0x110000):
    key = (unicodedata.category(chr(c)), chr(c).isspace(), chr(c).isalnum())
    if key != prev and not 0xD800 <= c < 0xE000:
        special.add(c)
        if c > 0 and not 0xD800 <= c - 1 < 0xE000:
            special.add(c - 1)
    prev = key
special.update([0xAC00, 0xAC01, 0xD7A3, 0x10FFFF])
special = sorted(special)


def runs(chars, size):
    for i in range(0, len(chars), size):
        chunk = [chr(c) for c in chars[i : i + size]]
        yield " ".join(chunk)
        yield "".join(chunk)
        yield "a".join(chunk) + "1"


count = write_vectors(
    os.path.join(tests_dir, "unicode-vectors.tsv"),
    "# Characters the Unicode tables single out, in runs, and the leaderboard normalizer's output. "
    "Generated by scripts/gen_vectors.py.\n",
    runs(special, 24),
)
print(f"unicode-vectors.tsv: {count} vectors from {len(special)} characters")

if full_sweep:
    every = [c for c in range(0x110000) if not 0xD800 <= c < 0xE000]
    count = write_vectors(full_sweep, "# Every code point in runs.\n", runs(every, 64))
    print(f"{full_sweep}: {count} vectors")

# Word sequences from splitmix64, rebuilt the same way by the Rust test.
MASK = (1 << 64) - 1


class SplitMix:
    def __init__(self, seed):
        self.state = seed & MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def below(self, n):
        return self.next() % n


def pair(r):
    """One reference and hypothesis. Mirrors score_pair in tests/vectors.rs."""
    kind = r.below(10)
    vocabulary = 2 + r.below(8)
    if kind < 5:
        reference = [r.below(vocabulary) for _ in range(r.below(20))]
        hypothesis = [r.below(vocabulary) for _ in range(r.below(20))]
    else:
        length = r.below(100) if kind < 9 else 65 + r.below(400)
        rate = r.below(30 if kind < 9 else 8)
        reference = [r.below(vocabulary * 4) for _ in range(length)]
        hypothesis = []
        for word in reference:
            roll = r.below(100)
            if roll < rate:
                choice = r.below(3)
                if choice == 0:
                    hypothesis.append(r.below(vocabulary * 4))
                elif choice == 1:
                    hypothesis.append(word)
                    hypothesis.append(r.below(vocabulary * 4))
            else:
                hypothesis.append(word)
    return " ".join(f"w{w}" for w in reference), " ".join(f"w{w}" for w in hypothesis)


r = SplitMix(42)
PAIRS = 5000
pairs = [pair(r) for _ in range(PAIRS)]
incorrect = total = 0
lines = []
for reference, hypothesis in pairs:
    m = jiwer.process_words(reference, hypothesis)
    incorrect += m.substitutions + m.deletions + m.insertions
    total += m.substitutions + m.deletions + m.hits
    lines.append(f"{m.substitutions} {m.deletions} {m.insertions} {m.substitutions + m.deletions + m.hits}")
# The leaderboard's corpus WER, as evaluate's "wer" metric computes it.
wer = incorrect / total
assert wer == jiwer.wer([p[0] for p in pairs], [p[1] for p in pairs])
with open(os.path.join(tests_dir, "score-vectors.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write(
        f"# jiwer {importlib.metadata.version('jiwer')} counts (substitutions deletions "
        f"insertions reference words) of {PAIRS} pairs from splitmix64 with seed 42, and the corpus WER. "
        "Generated by scripts/gen_vectors.py.\n"
    )
    f.write(f"wer {wer!r}\n")
    f.write("\n".join(lines) + "\n")
print(f"score-vectors.txt: {PAIRS} pairs, corpus WER {wer!r}")
