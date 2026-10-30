"""Run transformers' GraniteSpeech5ForCTC on the golden clips and save every intermediate tensor.

Usage:
  python -I ref_granite.py <model dir> <clips.json> <out dir> [--threads 6] [--runs 2]

<model dir> holds config.json, model.safetensors, the tokenizer and processor files. clips.json is a
list of objects with at least "id" and "wav" (16 kHz mono 16-bit). For each clip it writes
<out dir>/<id>.safetensors and <out dir>/<id>.json.

The model runs in eval mode in float32 (the bf16 weights widen exactly, BatchNorm stays float32),
with eager attention, batch size 1, the processor's features and attention mask. Forward hooks
record the output of every block, the self-conditioning logits (encoder.out) and injection
(encoder.out_mid), and the CTC logits. The whole forward pass runs --runs times, and every tensor
must be bit-identical across runs.
"""

import argparse
import itertools
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import save_file


def load_audio(path):
    data, sr = sf.read(path, dtype="int16", always_2d=False)
    assert sr == 16000 and data.ndim == 1, path
    return (data.astype(np.float32) / 32768.0).astype(np.float32)


def ctc_collapse(ids, blank):
    return [k for k, _ in itertools.groupby(ids) if k != blank]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("clips")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    import transformers
    from transformers import AutoProcessor, GraniteSpeech5ForCTC

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = GraniteSpeech5ForCTC.from_pretrained(
        args.model_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    print(f"loaded transformers {transformers.__version__} GraniteSpeech5ForCTC in {time.time() - t0:.1f} s", flush=True)
    assert not any(m.training for m in model.modules())
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(b.dtype == torch.float32 for n, b in model.named_buffers() if "running" in n)
    enc = model.encoder
    assert enc.config._attn_implementation == "eager", enc.config._attn_implementation
    # the CTC head is tied to the encoder's self-conditioning head
    assert model.ctc_head.weight.data_ptr() == enc.out.weight.data_ptr()
    assert model.ctc_head.bias.data_ptr() == enc.out.bias.data_ptr()
    blank = model.config.pad_token_id
    n_layers = len(enc.layers)
    mid = n_layers // 2 - 1  # self-conditioning follows this block

    captured = {}

    def keep(name):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            captured.setdefault(name, []).append(out.detach().clone())

        return hook

    hooks = [enc.input_linear.register_forward_hook(keep("input_linear"))]
    for n, layer in enumerate(enc.layers):
        hooks.append(layer.register_forward_hook(keep(f"layer.{n}")))
    hooks.append(enc.out.register_forward_hook(keep("selfcond_logits")))
    hooks.append(enc.out_mid.register_forward_hook(keep("selfcond_injection")))
    hooks.append(model.ctc_head.register_forward_hook(keep("logits")))

    for clip in json.load(open(args.clips, encoding="utf-8")):
        audio = load_audio(clip["wav"])
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        feats = inputs["input_features"]
        mask = inputs["attention_mask"]
        assert feats.dtype == torch.float32 and feats.shape[0] == 1 and feats.shape[2] == 320
        assert bool(mask.all()), "a single clip has no padding"
        runs = []
        secs = []
        for _ in range(args.runs):
            captured.clear()
            t1 = time.time()
            with torch.inference_mode():
                out = model(input_features=feats, attention_mask=mask)
            secs.append(time.time() - t1)
            assert len(captured["selfcond_logits"]) == 1 and len(captured["logits"]) == 1
            run = {k: v[0][0].contiguous() for k, v in captured.items()}
            assert torch.equal(run["logits"], out.logits[0])
            runs.append(run)
        identical = all(
            torch.equal(runs[0][k], r[k]) and runs[0][k].dtype == r[k].dtype for r in runs[1:] for k in runs[0]
        )
        assert identical, "runs differ"
        run = runs[0]
        with torch.inference_mode():
            sequences = model.generate(input_features=feats, attention_mask=mask)
        text = processor.batch_decode(sequences, skip_special_tokens=True)[0]

        logits = run["logits"]
        ids = logits.argmax(-1)
        assert torch.equal(ids, sequences[0]), "generate's ids differ from the logits' argmax"
        top2 = logits.topk(2, dim=-1).values
        conditioned = run[f"layer.{mid}"] + run["selfcond_injection"]
        tokens = ctc_collapse(ids.tolist(), blank)
        special = set(processor.tokenizer.all_special_ids)
        text_tokens = [k for k in tokens if k not in special]
        own_text = processor.tokenizer.decode(tokens, skip_special_tokens=True, group_tokens=False)
        assert own_text == text, (own_text, text)
        tensors = {
            "features": feats[0].contiguous(),
            "input_linear": run["input_linear"],
        }
        for n in range(n_layers):
            tensors[f"layer.{n}"] = run[f"layer.{n}"]
        tensors.update(
            {
                "selfcond_injection": run["selfcond_injection"],
                f"layer.{mid}.conditioned": conditioned.contiguous(),
                "selfcond_logits_max": run["selfcond_logits"].max(-1).values.contiguous(),
                "logits": logits,
                "ctc_ids": ids.to(torch.int64).contiguous(),
                "logit_max": top2[:, 0].contiguous(),
                "logit_margin": (top2[:, 0] - top2[:, 1]).contiguous(),
                "tokens": torch.tensor(tokens, dtype=torch.int64),
            }
        )
        save_file(tensors, os.path.join(args.out, clip["id"] + ".safetensors"))
        info = {
            "id": clip["id"],
            "wav": clip["wav"],
            "samples": len(audio),
            "forward_seconds": secs,
            "runs": args.runs,
            "runs_bit_identical": identical,
            "features_shape": list(feats.shape[1:]),
            "attention_mask_sum": int(mask.sum()),
            "frames": {name: int(t.shape[0]) for name, t in tensors.items() if name.startswith("layer.")},
            "ctc_frames": int(ids.shape[0]),
            "tokens": tokens,
            "text_tokens": text_tokens,
            "byte_tokens": [k for k in tokens if processor.tokenizer.convert_ids_to_tokens(k).startswith("<0x")],
            "text": text,
            "min_logit_margin": float(top2[:, 0].sub(top2[:, 1]).min()),
        }
        with open(os.path.join(args.out, clip["id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1, ensure_ascii=False)
        print(
            f"{clip['id']}: {[round(s, 2) for s in secs]} s, features {list(feats.shape[1:])}, "
            f"{ids.shape[0]} CTC frames, {len(tokens)} tokens, identical {identical}, "
            f"min margin {info['min_logit_margin']:.3g}\n  {text}",
            flush=True,
        )
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
