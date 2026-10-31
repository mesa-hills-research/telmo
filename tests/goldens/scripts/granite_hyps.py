"""Reference hypotheses from transformers' GraniteSpeech5ForCTC with greedy CTC decoding over a TSV of utterances.

Usage:
  python -I granite_hyps.py <model dir> <in tsv> <out tsv> [--subset 500 --seed 0] [--limit N] [--threads 6]

The input TSV has `utt_id<TAB>wav<TAB>reference` (16 kHz mono 16-bit WAVs). The output TSV has
`utt_id<TAB>hypothesis`, sorted by utt_id, with the raw text of processor.batch_decode(generate(...),
skip_special_tokens=True). Each utterance runs alone (batch size 1) in float32 with eager attention,
and its audio goes in as int16/32768 float32, as in the goldens. --subset draws a fixed random subset,
random.Random(seed).sample(sorted ids, n), then sorted. A JSON summary with the timings goes to <out tsv>.json.
"""

import argparse
import json
import random
import time

import numpy as np
import soundfile as sf
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("tsv")
    ap.add_argument("out")
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    rows = {}
    for line in open(args.tsv, encoding="utf-8"):
        utt, wav, _ref = line.rstrip("\n").split("\t")
        rows[utt] = wav
    ids = sorted(rows)
    if args.subset:
        ids = sorted(random.Random(args.seed).sample(ids, args.subset))
    if args.limit:
        ids = ids[: args.limit]

    import transformers
    from transformers import AutoProcessor, GraniteSpeech5ForCTC

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = GraniteSpeech5ForCTC.from_pretrained(
        args.model_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    load_secs = time.time() - t0

    hyps = {}
    samples_total = 0
    t1 = time.time()
    for n, utt in enumerate(ids, 1):
        data, sr = sf.read(rows[utt], dtype="int16", always_2d=False)
        assert sr == 16000 and data.ndim == 1, utt
        samples_total += len(data)
        audio = (data.astype(np.float32) / 32768.0).astype(np.float32)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            seq = model.generate(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"])
        text = processor.batch_decode(seq, skip_special_tokens=True)[0]
        assert "\t" not in text and "\n" not in text, (utt, text)
        hyps[utt] = text
        if n % 50 == 0 or n == len(ids):
            elapsed = time.time() - t1
            print(f"{n}/{len(ids)} utterances, {elapsed:.0f} s, RTF {elapsed / (samples_total / 16000):.3f}", flush=True)
    decode_secs = time.time() - t1

    with open(args.out, "w", encoding="utf-8") as f:
        for utt in ids:
            f.write(f"{utt}\t{hyps[utt]}\n")
    summary = {
        "tsv": args.tsv,
        "utterances": len(ids),
        "subset": args.subset or None,
        "seed": args.seed if args.subset else None,
        "limit": args.limit or None,
        "audio_seconds": samples_total / 16000,
        "model_load_seconds": load_secs,
        "transcribe_wall_seconds": decode_secs,
        "rtf": decode_secs / (samples_total / 16000),
        "torch_threads": torch.get_num_threads(),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
