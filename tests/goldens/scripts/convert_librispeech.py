"""Convert a LibriSpeech subset's FLAC files to 16 kHz mono 16-bit PCM WAV and write a TSV.

Usage: python -I convert_librispeech.py <subset dir, e.g. .../LibriSpeech/test-clean> <out wav dir> <out tsv>

The samples are copied exactly. Each FLAC is read as int16 and written as PCM_16, then read back and compared.
TSV lines are `utt_id<TAB>absolute wav path<TAB>transcript`, sorted by utt_id, no header.
"""

import os
import sys

import numpy as np
import soundfile as sf


def main() -> None:
    subset_dir, out_dir, out_tsv = (os.path.abspath(p) for p in sys.argv[1:4])
    os.makedirs(out_dir, exist_ok=True)
    transcripts = {}
    flacs = {}
    for root, _dirs, files in os.walk(subset_dir):
        for name in files:
            path = os.path.join(root, name)
            if name.endswith(".trans.txt"):
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        utt, text = line.split(" ", 1)
                        assert utt not in transcripts, utt
                        transcripts[utt] = text
            elif name.endswith(".flac"):
                flacs[name[: -len(".flac")]] = path
    assert set(transcripts) == set(flacs), (len(transcripts), len(flacs))
    rows = []
    total = 0
    for utt in sorted(flacs):
        info = sf.info(flacs[utt])
        assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16", (utt, info)
        data, sr = sf.read(flacs[utt], dtype="int16", always_2d=False)
        assert sr == 16000 and data.ndim == 1 and data.dtype == np.int16
        wav = os.path.join(out_dir, utt + ".wav")
        sf.write(wav, data, 16000, subtype="PCM_16", format="WAV")
        back, sr2 = sf.read(wav, dtype="int16", always_2d=False)
        assert sr2 == 16000 and np.array_equal(back, data), utt
        total += len(data)
        rows.append(f"{utt}\t{wav}\t{transcripts[utt]}\n")
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.writelines(rows)
    print(f"{len(rows)} utterances, {total} samples, {total / 16000 / 3600:.3f} h -> {out_tsv}")


if __name__ == "__main__":
    main()
