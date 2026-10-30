"""Fetch the first utterances of FLEURS test splits by streaming each archive and stopping early.

Usage:
  python -I fetch_fleurs.py <out dir> [--count 100] [--langs en_us,de_de,...]

For each language it reads data/<lang>/test.tsv and streams data/<lang>/audio/test.tar.gz from
google/fleurs at a pinned revision, keeping the first <count> WAV members of the archive (in archive
order) that the TSV lists. The archive holds 16 kHz mono float WAVs. Each is written as 16-bit PCM
(round(x * 32768), clipped) to <out dir>/<lang>/<file stem>.wav. <out dir>/<lang>.tsv gets `id<TAB>abs wav path<TAB>raw transcription`
sorted by id, where id is the file stem. The language's whole test.tsv is kept as <out dir>/<lang>/test.tsv.
"""

import argparse
import io
import os
import tarfile
import urllib.request

import numpy as np
import soundfile as sf

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
BASE = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/data"
LANGS = "en_us,de_de,fr_fr,es_419,it_it,pt_br,ja_jp"


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "fetch_fleurs"}), timeout=60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--langs", default=LANGS)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    for lang in args.langs.split(","):
        lang_dir = os.path.join(out, lang)
        os.makedirs(lang_dir, exist_ok=True)
        tsv = fetch(f"{BASE}/{lang}/test.tsv").read()
        with open(os.path.join(lang_dir, "test.tsv"), "wb") as f:
            f.write(tsv)
        raw = {}
        for line in tsv.decode("utf-8").splitlines():
            cols = line.split("\t")
            if len(cols) < 3:
                continue
            raw[cols[1]] = cols[2]
        rows = []
        streamed = 0
        with fetch(f"{BASE}/{lang}/audio/test.tar.gz") as resp:
            with tarfile.open(fileobj=io.BufferedReader(resp, 1 << 20), mode="r|gz") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    name = os.path.basename(member.name)
                    if not name.endswith(".wav") or name not in raw:
                        continue
                    data = tar.extractfile(member).read()
                    streamed += len(data)
                    samples, sr = sf.read(io.BytesIO(data), dtype="float64", always_2d=True)
                    if sr != 16000 or samples.shape[1] != 1:
                        print(f"{lang}: skipping {name}, {sr} Hz, {samples.shape[1]} channels")
                        continue
                    scaled = np.round(samples[:, 0] * 32768.0)
                    clipped = int(np.count_nonzero((scaled < -32768) | (scaled > 32767)))
                    pcm = np.clip(scaled, -32768, 32767).astype(np.int16)
                    if clipped:
                        print(f"{lang}: {name} has {clipped} clipped samples")
                    stem = name[: -len(".wav")]
                    path = os.path.join(lang_dir, name)
                    sf.write(path, pcm, 16000, subtype="PCM_16")
                    check, _ = sf.read(path, dtype="int16")
                    assert np.array_equal(check, pcm)
                    text = raw[name]
                    assert "\t" not in text and "\n" not in text
                    rows.append((stem, path, text))
                    if len(rows) >= args.count:
                        break
        rows.sort()
        with open(os.path.join(out, lang + ".tsv"), "w", encoding="utf-8") as f:
            for row in rows:
                f.write("\t".join(row) + "\n")
        print(f"{lang}: {len(rows)} utterances, {streamed / 1e6:.1f} MB of WAV", flush=True)


if __name__ == "__main__":
    main()
