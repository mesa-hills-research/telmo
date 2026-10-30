"""Make the golden clips' WAVs from the LibriSpeech TSVs and describe them in clips.json.

Usage: python -I make_clips.py <librispeech dir with test-clean.tsv/test-other.tsv> <goldens dir> <clips.json> [subset ...]

Each clip is 16 kHz mono 16-bit PCM, exactly the LibriSpeech samples. The joined clip is the plain
concatenation of consecutive utterances of one speaker and chapter (no gaps), and its reference text is
their transcripts joined with single spaces. `subset` limits the clips to those whose id starts with it
(for example `clean`).
"""

import json
import os
import sys

import numpy as np
import soundfile as sf

# id -> (LibriSpeech subset, source utterance ids)
CLIPS = [
    ("clean-3575-170457-0009", "test-clean", ["3575-170457-0009"]),
    ("clean-1995-1826-0007", "test-clean", ["1995-1826-0007"]),
    ("clean-260-123286-0018", "test-clean", ["260-123286-0018"]),
    ("other-367-130732-0011", "test-other", ["367-130732-0011"]),
    ("other-4350-9170-0030", "test-other", ["4350-9170-0030"]),
    (
        "clean-joined-40s",
        "test-clean",
        [f"1089-134686-{i:04d}" for i in range(3, 9)],
    ),
]


def read_tsv(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            utt, wav, text = line.rstrip("\n").split("\t")
            rows[utt] = (wav, text)
    return rows


def main() -> None:
    ls_dir, out_dir, clips_json = (os.path.abspath(p) for p in sys.argv[1:4])
    prefixes = tuple(sys.argv[4:])
    os.makedirs(out_dir, exist_ok=True)
    tsvs = {}
    clips = []
    for clip_id, subset, utts in CLIPS:
        if prefixes and not clip_id.startswith(prefixes):
            continue
        if subset not in tsvs:
            tsvs[subset] = read_tsv(os.path.join(ls_dir, subset + ".tsv"))
        rows = tsvs[subset]
        parts = []
        for utt in utts:
            data, sr = sf.read(rows[utt][0], dtype="int16", always_2d=False)
            assert sr == 16000 and data.ndim == 1
            parts.append(data)
        samples = np.concatenate(parts)
        wav = os.path.join(out_dir, clip_id + ".wav")
        sf.write(wav, samples, 16000, subtype="PCM_16", format="WAV")
        back, _ = sf.read(wav, dtype="int16")
        assert np.array_equal(back, samples)
        clips.append(
            {
                "id": clip_id,
                "librispeech_subset": subset,
                "source_utterances": utts,
                "source_utterance_samples": [len(p) for p in parts],
                "wav": wav,
                "samples": int(len(samples)),
                "seconds": len(samples) / 16000,
                "reference_text": " ".join(rows[u][1] for u in utts),
            }
        )
        print(f"{clip_id}\t{len(samples) / 16000:.2f} s\t{clips[-1]['reference_text'][:80]}")
    with open(clips_json, "w", encoding="utf-8") as f:
        json.dump(clips, f, indent=1)


if __name__ == "__main__":
    main()
