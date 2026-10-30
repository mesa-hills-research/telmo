"""Write the golden files and manifest.json from the NeMo and transformers runs.

Usage:
  DATA=<data folder> python -I make_goldens.py <clips.json> <nemo dir> <hf dir> <ablation dir> <goldens dir> <commands.txt>

DATA holds models/parakeet-tdt-0.6b-v3/ and models/parakeet-tdt-0.6b-v3-nemo/, whose hashes go in the manifest.

The goldens come from NeMo (the primary reference). transformers is the second check, and the
ablation (NeMo with ConvSubsampling's time masking turned off) only feeds the comparison numbers.
"""

import hashlib
import importlib.metadata as md
import json
import os
import platform
import sys

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file, save_file

DATA = os.environ.get("DATA") or sys.exit("set DATA to the data folder")
MODEL_DIR = os.path.join(DATA, "models/parakeet-tdt-0.6b-v3")
NEMO_FILE = os.path.join(DATA, "models/parakeet-tdt-0.6b-v3-nemo/parakeet-tdt-0.6b-v3.nemo")
REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
SAVED_LAYERS = [0, 1, 12, 23]
LONG_CLIP_LAYERS = [0, 23]  # the 40 s clip leaves out layer.1 and layer.12 to save space
PACKAGES = [
    "torch", "nemo_toolkit", "transformers", "numpy", "soundfile", "safetensors", "tokenizers", "sentencepiece",
    "huggingface_hub", "lightning", "pytorch-lightning", "omegaconf", "hydra-core", "librosa", "lhotse", "scipy",
    "numba", "llvmlite", "torchmetrics", "pip",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel_l2(a, b):
    a = a.double()
    b = b.double()
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))


def cos(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    return float(a @ b / (a.norm() * b.norm()))


def compare_hf(nemo, ninfo, hf, hinfo, T, F):
    cmp = {
        "features_max_abs_diff_valid_frames": float((nemo["features"][:F] - hf["features"][:F]).abs().max()),
        "features_max_abs_diff_all_frames": float((nemo["features"] - hf["features"]).abs().max()),
        "features_shape_equal": list(nemo["features"].shape) == list(hf["features"].shape),
        "features_len_equal": int(hf["features_len"][0]) == F,
        "encoded_len_equal": int(hf["encoded_len"][0]) == T,
        "subsampling_frames": [int(nemo["subsampling"].shape[0]), int(hf["subsampling"].shape[0])],
        "relative_l2_valid_frames": {},
        "relative_l2_valid_frames_on_nemo_features": {},
        "tokens_equal": ninfo["tokens"] == hinfo["tokens"],
        "token_frames_equal": ninfo["token_frame"] == hinfo["token_frame"],
        "token_durations_equal": ninfo["token_duration"] == hinfo["token_duration"],
        "transformers_generate_equals_loop": hinfo["loop_equals_generate"],
        "texts_equal": ninfo["transcribe_text"] == hinfo["generate_text"],
    }
    for name in ["subsampling"] + [f"layer.{n}" for n in range(24)] + ["enc_proj"]:
        cmp["relative_l2_valid_frames"][name] = rel_l2(hf[name][:T], nemo[name][:T])
        cmp["relative_l2_valid_frames_on_nemo_features"][name] = rel_l2(hf["on_nemo_features." + name][:T], nemo[name][:T])
    n_pred = min(nemo["pred"].shape[0], hf["pred"].shape[0])
    cmp["pred_relative_l2"] = rel_l2(hf["pred"][:n_pred], nemo["pred"][:n_pred])
    n_joint = min(nemo["joint"].shape[0], hf["joint"].shape[0])
    cmp["joint_relative_l2"] = rel_l2(hf["joint"][:n_joint], nemo["joint"][:n_joint])
    cmp["joint_frames_equal"] = torch.equal(hf["joint_frame"][:n_joint], nemo["joint_frame"][:n_joint])
    return cmp


def compare_ablation(nemo, ninfo, abl, ainfo, T):
    names = ["subsampling", "layer.0", "layer.12", "layer.23"]
    return {
        "relative_l2_valid_frames": {name: rel_l2(abl[name][:T], nemo[name][:T]) for name in names},
        "max_abs_diff_valid_frames": {name: float((abl[name][:T] - nemo[name][:T]).abs().max()) for name in names},
        "subsampling_valid_frames_that_differ": [
            int(i) for i in torch.nonzero((abl["subsampling"][:T] - nemo["subsampling"][:T]).abs().amax(-1) > 0).flatten()
        ],
        "subsampling_last_valid_frame_relative_l2": rel_l2(abl["subsampling"][T - 1], nemo["subsampling"][T - 1]),
        "layer.23_last_valid_frame_relative_l2": rel_l2(abl["layer.23"][T - 1], nemo["layer.23"][T - 1]),
        "tokens_equal": ainfo["tokens"] == ninfo["tokens"],
        "token_frames_equal": ainfo["token_frame"] == ninfo["token_frame"],
    }


def main() -> None:
    clips_json, nemo_dir, hf_dir, abl_dir, out_dir, commands_txt = sys.argv[1:7]
    clips = json.load(open(clips_json, encoding="utf-8"))
    manifest_clips = []
    for clip in clips:
        cid = clip["id"]
        nemo = load_file(os.path.join(nemo_dir, cid + ".safetensors"))
        ninfo = json.load(open(os.path.join(nemo_dir, cid + ".json"), encoding="utf-8"))
        have_hf = os.path.exists(os.path.join(hf_dir, cid + ".json"))
        have_abl = os.path.exists(os.path.join(abl_dir, cid + ".json"))
        if have_hf:
            hf = load_file(os.path.join(hf_dir, cid + ".safetensors"))
            hinfo = json.load(open(os.path.join(hf_dir, cid + ".json"), encoding="utf-8"))
        if have_abl:
            abl = load_file(os.path.join(abl_dir, cid + ".safetensors"))
            ainfo = json.load(open(os.path.join(abl_dir, cid + ".json"), encoding="utf-8"))

        # the WAV holds exactly the samples the model got
        wav_path = os.path.join(out_dir, cid + ".wav")
        samples, sr = sf.read(wav_path, dtype="int16")
        assert sr == 16000 and len(samples) == clip["samples"]

        # checks before writing
        assert ninfo["encoder_out_equals_last_layer"], cid
        assert torch.equal(nemo["encoder_out"], nemo["layer.23"]), cid
        assert ninfo["loop_equals_transcribe"] and ninfo["loop_equals_decoding"], cid
        assert ninfo["loop_text"] == ninfo["transcribe_text"] == ninfo["decoding_text"], cid
        T = int(nemo["encoded_len"][0])
        F = int(nemo["features_len"][0])
        assert nemo["enc_proj"].shape[0] == nemo["subsampling"].shape[0]

        layers = LONG_CLIP_LAYERS if clip["seconds"] > 20 else SAVED_LAYERS
        tensors = {
            "features": nemo["features"],
            "features_len": nemo["features_len"],
            "subsampling": nemo["subsampling"],
            "encoded_len": nemo["encoded_len"],
        }
        for n in layers:
            tensors[f"layer.{n}"] = nemo[f"layer.{n}"]
        for k in ["enc_proj", "pred", "joint", "joint_frame", "tokens", "token_frame", "token_duration"]:
            tensors[k] = nemo[k]
        tensors = {k: v.contiguous() for k, v in tensors.items()}
        for k, v in tensors.items():
            assert v.dtype in (torch.float32, torch.int64), (k, v.dtype)
        st_name = cid + ".safetensors"
        save_file(
            tensors,
            os.path.join(out_dir, st_name),
            # one key only, because safetensors writes a metadata map in a random order and the bytes would change
            metadata={"about": f"NVIDIA NeMo outputs of nvidia/parakeet-tdt-0.6b-v3 at {REVISION} (CC BY 4.0) for {cid}"},
        )

        # NeMo vs transformers
        cmp = compare_hf(nemo, ninfo, hf, hinfo, T, F) if have_hf else None
        # NeMo with and without ConvSubsampling's time masking
        ablation = compare_ablation(nemo, ninfo, abl, ainfo, T) if have_abl else None

        manifest_clips.append(
            {
                "id": cid,
                "wav": cid + ".wav",
                "tensors": st_name,
                "librispeech_subset": clip["librispeech_subset"],
                "source_utterances": clip["source_utterances"],
                "seconds": clip["seconds"],
                "samples": clip["samples"],
                "reference_text": clip["reference_text"],
                "nemo_text": ninfo["transcribe_text"],
                "transformers_text": hinfo["generate_text"] if have_hf else None,
                "features_frames": int(nemo["features"].shape[0]),
                "features_len": F,
                "subsampling_frames": int(nemo["subsampling"].shape[0]),
                "encoded_len": T,
                "tokens": len(ninfo["tokens"]),
                "joint_steps_saved": int(nemo["joint"].shape[0]),
                "saved_layers": layers,
                "wav_sha256": sha256(wav_path),
                "tensors_sha256": sha256(os.path.join(out_dir, st_name)),
                "nemo_vs_transformers": cmp,
                "nemo_without_subsampling_mask": ablation,
            }
        )
        line = f"{cid}: {T} frames, {len(ninfo['tokens'])} tokens, {ninfo['transcribe_text']!r}"
        if cmp:
            rl = cmp["relative_l2_valid_frames"]
            line += (
                f"\n  vs transformers: features {cmp['features_max_abs_diff_valid_frames']:.2e}, "
                f"sub {rl['subsampling']:.2e}, L0 {rl['layer.0']:.2e}, L12 {rl['layer.12']:.2e}, "
                f"L23 {rl['layer.23']:.2e}, L23 on NeMo features "
                f"{cmp['relative_l2_valid_frames_on_nemo_features']['layer.23']:.2e}, "
                f"tokens equal {cmp['tokens_equal']}, texts equal {cmp['texts_equal']}"
            )
        if ablation:
            line += f"\n  without the subsampling mask: L23 {ablation['relative_l2_valid_frames']['layer.23']:.2e}"
        print(line)

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
    model_files["parakeet-tdt-0.6b-v3.nemo"] = {"sha256": sha256(NEMO_FILE), "bytes": os.path.getsize(NEMO_FILE)}
    commands = [line.rstrip("\n") for line in open(commands_txt, encoding="utf-8") if line.strip()]

    manifest = {
        "about": "Golden tensors for Parakeet TDT 0.6B v3, saved from NVIDIA NeMo (the reference) with "
        "transformers' port as a second check. See README.md.",
        "model": {
            "repo": "nvidia/parakeet-tdt-0.6b-v3",
            "url": "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
            "revision": REVISION,
            "license": "CC-BY-4.0",
            "files": model_files,
            "nemo_loaded_from": "parakeet-tdt-0.6b-v3.nemo (ASRModel.restore_from)",
            "transformers_loaded_from": "config.json, model.safetensors, tokenizer and processor files (ParakeetForTDT.from_pretrained)",
        },
        "audio": {
            "source": "LibriSpeech ASR corpus (test-clean, test-other), Vassil Panayotov et al.",
            "url": "https://www.openslr.org/12",
            "license": "CC-BY-4.0",
            "archives_md5": {"test-clean.tar.gz": "32fa31d27d2e1cad72775fee3f4849a9",
                             "test-other.tar.gz": "fb5a50374b501bb3bac4815ee91d3135"},
            "note": "Phase 0 has no recordings of the owner's dictation, so all six clips are LibriSpeech.",
            "format": "16 kHz mono 16-bit PCM WAV, the FLAC's samples unchanged. The model input is int16/32768 as float32. "
            "The joined clip is a plain concatenation of the source utterances with no gaps.",
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
            "dtype": "float32",
            "mode": "eval",
            "torch_threads": 6,
            "nemo_preprocessor": "featurizer.dither = 0.0, featurizer.pad_to = 0",
            "nemo_decoding": "greedy_batch (GreedyBatchedTDTInfer, label looping), max_symbols 10, durations [0, 1, 2, 3, 4]",
            "batch_size": 1,
        },
        "tensors": {
            "features": "F32 [F, 128]. NeMo preprocessor output (processed_signal transposed to frames x mels), all F frames it returned. F = samples // 160 + 1. Only the first features_len frames are valid, and the rest are zero.",
            "features_len": "I64 [1]. processed_signal_length = samples // 160.",
            "subsampling": "F32 [T, 1024]. Output of encoder.pre_encode (ConvSubsampling dw_striding and its linear), all T frames it returned. T can be encoded_len + 1, and that extra frame comes only from padding.",
            "encoded_len": "I64 [1]. The encoder's output length.",
            "layer.N": "F32 [T, 1024]. Output of Conformer layer N (forward hook). layer.23 is the encoder output (checked equal). The 40 s clip has only layer.0 and layer.23.",
            "enc_proj": "F32 [T, 640]. joint.enc applied to the encoder output (transformers encoder_projector).",
            "pred": "F32 [N+1, 640]. joint.pred applied to the prediction network output, first for the start state (blank as SOS, a zero embedding, zero LSTM state), then after each emitted token.",
            "joint": "F32 [S, 8198]. Raw joint logits (joint.joint_net output, no softmax), 8192 tokens, blank 8192, then 5 duration logits, for the first S = min(20, all) steps of greedy decoding, blank steps included.",
            "joint_frame": "I64 [S]. The encoder frame of each saved joint step.",
            "tokens": "I64 [N]. Emitted token ids, no blanks.",
            "token_frame": "I64 [N]. The encoder frame each token was emitted at.",
            "token_duration": "I64 [N]. The duration in frames predicted with each token.",
        },
        "decoding": "At frame t run the joint, take argmax over the 8193 token logits and argmax over the 5 duration logits. "
        "A non-blank token is emitted and the prediction network advances with it. A blank with duration 0 counts as 1. "
        "t += duration. After 10 tokens at one frame, t += 1. This is NeMo's GreedyBatchedTDTLabelLoopingComputer.torch_impl, "
        "and the loop's tokens, frames and durations equal NeMo's decoding and model.transcribe on every clip.",
        "comparison_notes": "Relative L2 is ||transformers - NeMo|| / ||NeMo|| over the first encoded_len frames. "
        "relative_l2_valid_frames_on_nemo_features feeds NeMo's features to transformers' encoder. "
        "nemo_without_subsampling_mask reruns NeMo with MaskedConvSequential's masking turned off.",
        "commands": commands,
        "clips": manifest_clips,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote", os.path.join(out_dir, "manifest.json"))


if __name__ == "__main__":
    main()
