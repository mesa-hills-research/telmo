"""List the clips of the Granite goldens.

Usage:
  python -I granite_clips.py <parakeet goldens dir> <granite goldens dir> <out clips.json>
                             --ids <id,id,...> [--fleurs <lang>:<lang tsv>:<utterance id>]

The LibriSpeech clips are the WAVs of the Parakeet goldens, named by --ids and described by that
folder's manifest.json. With --fleurs, one FLEURS utterance from a list written by fetch_fleurs.py
is copied into the Granite goldens folder as fleurs-<lang>-<utterance id>.wav.
"""

import argparse
import json
import os
import shutil

import soundfile as sf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parakeet_dir")
    ap.add_argument("granite_dir")
    ap.add_argument("out")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--fleurs", default=None)
    args = ap.parse_args()
    manifest = json.load(open(os.path.join(args.parakeet_dir, "manifest.json"), encoding="utf-8"))
    by_id = {c["id"]: c for c in manifest["clips"]}
    clips = []
    for cid in args.ids.split(","):
        c = by_id[cid]
        wav = os.path.abspath(os.path.join(args.parakeet_dir, c["wav"]))
        clips.append(
            {
                "id": cid,
                "wav": wav,
                "wav_in_repo": os.path.relpath(wav, os.path.abspath(args.granite_dir)),
                "source": "LibriSpeech " + c["librispeech_subset"],
                "source_utterances": c["source_utterances"],
                "seconds": c["seconds"],
                "samples": c["samples"],
                "reference_text": c["reference_text"],
            }
        )
    if args.fleurs:
        lang, tsv, utt = args.fleurs.split(":")
        rows = {}
        for line in open(tsv, encoding="utf-8"):
            uid, path, text = line.rstrip("\n").split("\t")
            rows[uid] = (path, text)
        path, text = rows[utt]
        cid = f"fleurs-{lang}-{utt}"
        dst = os.path.join(args.granite_dir, cid + ".wav")
        os.makedirs(args.granite_dir, exist_ok=True)
        shutil.copyfile(path, dst)
        info = sf.info(dst)
        assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16"
        clips.append(
            {
                "id": cid,
                "wav": os.path.abspath(dst),
                "wav_in_repo": cid + ".wav",
                "source": f"FLEURS {lang} test",
                "source_utterances": [utt],
                "seconds": info.frames / 16000,
                "samples": info.frames,
                "reference_text": text,
            }
        )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(clips, f, indent=1, ensure_ascii=False)
    for c in clips:
        print(c["id"], c["samples"], c["wav_in_repo"])


if __name__ == "__main__":
    main()
