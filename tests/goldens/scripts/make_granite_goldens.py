"""Write the Granite golden files, the tokenizer vectors and manifest.json from the transformers run.

Usage:
  DATA=<data folder> python -I make_granite_goldens.py <clips.json> <ref dir> <goldens dir> <commands.txt>

DATA holds models/granite-speech-5.0-470m-turboctc/, whose hashes go in the manifest and whose
tokenizer makes the decoding vectors. <ref dir> holds ref_granite.py's outputs for every clip.
"""

import hashlib
import importlib.metadata as md
import json
import os
import platform
import random
import sys

import soundfile as sf
import torch
from safetensors.torch import load_file, save_file

DATA = os.environ.get("DATA") or sys.exit("set DATA to the data folder")
MODEL_DIR = os.path.join(DATA, "models/granite-speech-5.0-470m-turboctc")
REVISION = "1dd23f46f6d4e6eae2531982ba2e33220086dd97"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
SAVED = ["features", "layer.0", "layer.1", "layer.7", "selfcond_injection", "layer.15"]
# The 40 s clip keeps the first and last block at 12.5 frames a second, where it spans four
# attention blocks, and leaves out the rest to keep the folder small.
LONG_CLIP_SAVED = ["layer.1", "layer.15"]
# The FLEURS clip is there for its language and its tokens, so it keeps only the ends.
FLEURS_SAVED = ["features", "layer.15"]
PACKAGES = ["torch", "torchaudio", "transformers", "numpy", "soundfile", "safetensors", "tokenizers",
            "huggingface_hub", "pip"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tokenizer_vectors(tokenizer, clips):
    """Decoded text for every single id, for random id sequences and for encoded transcripts."""
    vocab = len(tokenizer)
    assert vocab == 16384, vocab

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=True, group_tokens=False)

    singles = [decode([i]) for i in range(vocab)]
    rng = random.Random(0)
    byte_ids = [i for i in range(vocab) if tokenizer.convert_ids_to_tokens(i).startswith("<0x")]
    special = sorted(tokenizer.all_special_ids)
    sequences = []
    for _ in range(400):
        ids = []
        for _ in range(rng.randint(1, 30)):
            r = rng.random()
            if r < 0.3:
                ids.append(rng.choice(byte_ids))
            elif r < 0.4:
                ids.append(rng.choice(special))
            else:
                ids.append(rng.randrange(byte_ids[-1] + 1, vocab))
        sequences.append({"ids": ids, "text": decode(ids)})
    # Valid UTF-8 split into byte pieces, the way the model spells rare characters.
    for text in ["日本語のテキスト", "Ça coûte 5 €, déjà vu", "Größe — ½ “quoted” 𝄞"]:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        sequences.append({"ids": ids, "text": decode(ids)})
    for clip in clips:
        if clip["source"].startswith("FLEURS"):
            ids = tokenizer(clip["reference_text"], add_special_tokens=False)["input_ids"]
            sequences.append({"ids": ids, "text": decode(ids)})
    return {"single": singles, "sequences": sequences}


def main() -> None:
    clips_json, ref_dir, out_dir, commands_txt = sys.argv[1:5]
    clips = json.load(open(clips_json, encoding="utf-8"))
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    manifest_clips = []
    for clip in clips:
        cid = clip["id"]
        ref = load_file(os.path.join(ref_dir, cid + ".safetensors"))
        info = json.load(open(os.path.join(ref_dir, cid + ".json"), encoding="utf-8"))
        assert info["runs_bit_identical"] and info["runs"] >= 2, cid
        wav_path = os.path.normpath(os.path.join(out_dir, clip["wav_in_repo"]))
        samples, sr = sf.read(wav_path, dtype="int16")
        assert sr == 16000 and len(samples) == clip["samples"] == info["samples"], cid
        assert sha256(wav_path) == sha256(clip["wav"]), cid

        ids = ref["ctc_ids"]
        frames = ids.shape[0]
        nonblank = torch.nonzero(ids != 0).flatten()
        first = int(nonblank[0]) if len(nonblank) else 0
        logit_frames = sorted({first, frames - 1} | ({0} if first == frames - 1 else set()))
        if clip["source"].startswith("FLEURS"):
            names = FLEURS_SAVED
        elif clip["seconds"] > 20:
            names = LONG_CLIP_SAVED
        else:
            names = SAVED
        tensors = {name: ref[name] for name in names}
        tensors.update(
            {
                "ctc_ids": ids,
                "logit_max": ref["logit_max"],
                "logit_margin": ref["logit_margin"],
                "logit_frames": torch.tensor(logit_frames, dtype=torch.int64),
                "logits": ref["logits"][logit_frames],
                "tokens": ref["tokens"],
            }
        )
        tensors = {k: v.contiguous() for k, v in tensors.items()}
        for k, v in tensors.items():
            assert v.dtype in (torch.float32, torch.int64), (k, v.dtype)
        st_name = cid + ".safetensors"
        save_file(
            tensors,
            os.path.join(out_dir, st_name),
            # one key only, because safetensors writes a metadata map in a random order and the bytes would change
            metadata={"about": f"transformers outputs of ibm-granite/granite-speech-5.0-470m-turboctc at {REVISION} (Apache-2.0) for {cid}"},
        )
        manifest_clips.append(
            {
                "id": cid,
                "wav": clip["wav_in_repo"],
                "tensors": st_name,
                "source": clip["source"],
                "source_utterances": clip["source_utterances"],
                "seconds": clip["seconds"],
                "samples": clip["samples"],
                "reference_text": clip["reference_text"],
                "text": info["text"],
                "features_frames": info["features_shape"][0],
                "frames": {k: v for k, v in info["frames"].items() if k in ("layer.0", "layer.1", "layer.15")},
                "ctc_frames": frames,
                "tokens": len(info["tokens"]),
                "byte_tokens": len(info["byte_tokens"]),
                "min_logit_margin": info["min_logit_margin"],
                "saved": names,
                "logit_frames": logit_frames,
                "wav_sha256": sha256(wav_path),
                "tensors_sha256": sha256(os.path.join(out_dir, st_name)),
            }
        )
        print(f"{cid}: {frames} frames, {len(info['tokens'])} tokens, {info['text']!r}")

    vectors = tokenizer_vectors(processor.tokenizer, clips)
    with open(os.path.join(out_dir, "tokenizer-vectors.json"), "w", encoding="utf-8") as f:
        json.dump(vectors, f, ensure_ascii=False, indent=0)
        f.write("\n")

    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    versions["libsndfile"] = sf.__libsndfile_version__
    freeze = sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions())
    model_files = {}
    for name in ["model.safetensors", "tokenizer.json", "config.json", "generation_config.json",
                 "processor_config.json", "tokenizer_config.json"]:
        p = os.path.join(MODEL_DIR, name)
        model_files[name] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
    commands = [line.rstrip("\n") for line in open(commands_txt, encoding="utf-8") if line.strip()]

    manifest = {
        "about": "Golden tensors for Granite Speech 5.0 470M TurboCTC, saved from transformers' GraniteSpeech5ForCTC "
        "(the reference). See README.md.",
        "model": {
            "repo": "ibm-granite/granite-speech-5.0-470m-turboctc",
            "url": "https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc",
            "revision": REVISION,
            "license": "Apache-2.0",
            "files": model_files,
            "loaded_from": "config.json, model.safetensors, tokenizer and processor files "
            "(GraniteSpeech5ForCTC.from_pretrained, AutoProcessor.from_pretrained)",
        },
        "audio": {
            "librispeech": {
                "source": "LibriSpeech ASR corpus (test-clean, test-other), Vassil Panayotov et al.",
                "url": "https://www.openslr.org/12",
                "license": "CC-BY-4.0",
                "files": "The WAVs of the Parakeet TDT goldens in ../parakeet-tdt-0.6b-v3/, which list the archives' md5.",
            },
            "fleurs": {
                "source": "FLEURS (google/fleurs), test split",
                "url": "https://huggingface.co/datasets/google/fleurs",
                "revision": FLEURS_REVISION,
                "license": "CC-BY-4.0",
                "conversion": "The dataset's 16 kHz float WAV written as 16-bit PCM, round(x * 32768) clipped to the int16 range.",
            },
            "format": "16 kHz mono 16-bit PCM WAV. The model input is int16/32768 as float32.",
        },
        "environment": {
            "python": sys.version.split()[0],
            "python_build": "python-build-standalone cpython-3.12.16+20261026-x86_64-unknown-linux-gnu-install_only",
            "platform": platform.platform(),
            "cpu": "6 vCPUs of an Intel Core i9-13980HX (AVX2, FMA, no AVX-512)",
            "packages": versions,
            "torch_index": "https://download.pytorch.org/whl/cpu",
            "pip_freeze": freeze,
        },
        "settings": {
            "device": "cpu",
            "dtype": "float32 (the bf16 weights widened, BatchNorm statistics stored as float32)",
            "attention": "eager",
            "mode": "eval",
            "torch_threads": 6,
            "batch_size": 1,
            "runs": "Each clip ran twice in one process, and every tensor was bit-identical.",
        },
        "tensors": {
            "features": "F32 [T0, 320]. The processor's input_features: 80 log-mel bins and their deltas per 10 ms frame, "
            "two frames stacked. The 40 s clip leaves them out.",
            "saved": "Each clip lists the tensors it keeps in `saved`. The 40 s clip keeps layer.1 and layer.15, and the "
            "FLEURS clip features and layer.15.",
            "layer.N": "F32 [T, 1024]. Output of encoder block N (forward hook). Blocks 0 and 1 halve the frame rate, so "
            "layer.0 has T0 // 2 frames and the rest T0 // 4. layer.7 is block 7's output before self-conditioning. "
            "layer.15 is the encoder output.",
            "selfcond_injection": "F32 [T, 1024]. encoder.out_mid(softmax(encoder.out(layer.7))), added to layer.7 before block 8.",
            "ctc_ids": "I64 [T]. Argmax of the CTC logits per frame (generate's sequences).",
            "logit_max": "F32 [T]. The largest CTC logit per frame.",
            "logit_margin": "F32 [T]. The largest CTC logit minus the second largest, per frame.",
            "logit_frames": "I64 [K]. Frames whose full logits are saved: the first frame whose argmax isn't blank, the last frame.",
            "logits": "F32 [K, 16384]. Raw CTC logits (ctc_head, tied to encoder.out) at logit_frames.",
            "tokens": "I64 [N]. ctc_ids with repeats collapsed and blanks (id 0) dropped, before special tokens are left out.",
        },
        "tokenizer_vectors": "tokenizer-vectors.json holds tokenizer.decode(ids, skip_special_tokens=True, "
        "group_tokens=False) for every single id (`single`) and for 400 random id sequences mixing byte pieces, "
        "special tokens and other pieces plus a few encoded texts (`sequences`).",
        "decoding": "Greedy CTC. Argmax per frame, repeats collapsed, blank (id 0, pad_token_id) dropped, then "
        "processor.batch_decode(sequences, skip_special_tokens=True), which leaves out the 255 special tokens.",
        "commands": commands,
        "clips": manifest_clips,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    total = sum(os.path.getsize(os.path.join(out_dir, n)) for n in os.listdir(out_dir))
    print("wrote", os.path.join(out_dir, "manifest.json"), f"folder {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
