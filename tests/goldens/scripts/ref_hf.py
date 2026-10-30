"""Run transformers' port of Parakeet TDT 0.6B v3 (ParakeetForTDT) on the golden clips, as a second check.

Usage:
  python -I ref_hf.py <model dir with config.json, model.safetensors, tokenizer files> <clips.json> <out dir>
                      [--nemo-dir <dir of ref_nemo.py outputs>] [--threads 6]

For each clip it writes <out dir>/<id>.safetensors (the processor's features, subsampling output, all 24
layer outputs, the encoder output, encoder_projector output, the decoding trace) and <out dir>/<id>.json
(texts and tokens from generate and from the same greedy loop as ref_nemo.py). With --nemo-dir it also
runs the encoder on NeMo's features (tensors prefixed `on_nemo_features.`) to separate encoder differences
from feature differences. The model runs in eval mode, float32, on the CPU.
"""

import argparse
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file, save_file

MAX_SYMBOLS = 10
JOINT_STEPS = 20


def load_clip_audio(path):
    data, sr = sf.read(path, dtype="int16", always_2d=False)
    assert sr == 16000 and data.ndim == 1
    return (data.astype(np.float32) / 32768.0).astype(np.float32)


def greedy_tdt(model, enc_proj, enc_len):
    """The same greedy TDT loop as ref_nemo.py, over transformers' modules. enc_proj: [T, 640]."""
    cfg = model.config
    blank = cfg.blank_token_id
    durations = list(cfg.durations)
    num_durations = len(durations)
    dec = model.decoder

    def pred_step(label, state):
        emb = dec.embedding(torch.tensor([[label]], dtype=torch.long))
        out, state = dec.lstm(emb, state)
        return dec.decoder_projector(out)[0, 0], state

    zeros = torch.zeros(cfg.num_decoder_layers, 1, cfg.decoder_hidden_size)
    pred, state = pred_step(blank, (zeros, zeros.clone()))
    preds = [pred]
    joints, joint_frames = [], []
    tokens, token_frames, token_durations = [], [], []
    t = 0
    run_frame, run_count = -1, 0
    while t < enc_len:
        logits = model.joint(decoder_hidden_states=pred, encoder_hidden_states=enc_proj[t])
        if len(joints) < JOINT_STEPS:
            joints.append(logits.clone())
            joint_frames.append(t)
        k = int(logits[:-num_durations].argmax())
        d = durations[int(logits[-num_durations:].argmax())]
        if k == blank:
            t += max(d, 1)
            continue
        tokens.append(k)
        token_frames.append(t)
        token_durations.append(d)
        pred, state = pred_step(k, state)
        preds.append(pred)
        if t == run_frame:
            run_count += 1
        else:
            run_frame, run_count = t, 1
        t += d
        if d == 0 and run_count >= MAX_SYMBOLS:
            t += 1
    return {
        "pred": torch.stack(preds),
        "joint": torch.stack(joints),
        "joint_frame": torch.tensor(joint_frames, dtype=torch.int64),
        "tokens": torch.tensor(tokens, dtype=torch.int64),
        "token_frame": torch.tensor(token_frames, dtype=torch.int64),
        "token_duration": torch.tensor(token_durations, dtype=torch.int64),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("clips")
    ap.add_argument("out")
    ap.add_argument("--nemo-dir", default=None)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    import transformers
    from transformers import AutoProcessor, ParakeetForTDT

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = ParakeetForTDT.from_pretrained(args.model_dir, dtype=torch.float32).eval()
    print(f"loaded transformers {transformers.__version__} ParakeetForTDT in {time.time() - t0:.1f} s", flush=True)
    blank = model.config.blank_token_id
    assert float(model.decoder.embedding.weight[blank].abs().max()) == 0.0, "blank embedding row is not zero"

    captured = {}
    hooks = [
        model.encoder.subsampling.register_forward_hook(
            lambda m, i, o: captured.__setitem__("subsampling", o.detach().clone())
        )
    ]
    for n, layer in enumerate(model.encoder.layers):
        hooks.append(
            layer.register_forward_hook(
                lambda m, i, o, n=n: captured.__setitem__(
                    f"layer.{n}", (o[0] if isinstance(o, tuple) else o).detach().clone()
                )
            )
        )
    n_layers = len(model.encoder.layers)

    for clip in json.load(open(args.clips, encoding="utf-8")):
        audio = load_clip_audio(clip["wav"])
        captured.clear()
        t1 = time.time()
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        feats = inputs["input_features"].to(torch.float32)  # [1, F, 128]
        mask = inputs["attention_mask"]
        with torch.inference_mode():
            enc_out = model.get_audio_features(input_features=feats, attention_mask=mask, output_attention_mask=True)
            enc = enc_out.last_hidden_state[0]  # [T, 1024]
            enc_proj = enc_out.pooler_output[0]  # encoder_projector, [T, 640]
            enc_len = int(enc_out.attention_mask.sum()) if enc_out.attention_mask is not None else enc.shape[0]
            layers = {k: v[0].contiguous() for k, v in captured.items()}
            trace = greedy_tdt(model, enc_proj, enc_len)
            gen = model.generate(input_features=feats, attention_mask=mask)
        secs = time.time() - t1
        seq = gen.sequences[0].tolist()
        durs = gen.durations[0].tolist()
        # sequences start with decoder_start_token_id (blank) with a zero duration, then one entry per step
        gen_tokens, gen_frames, gen_durations = [], [], []
        frame = 0
        for tok, dur in zip(seq[1:], durs[1:]):
            if tok != blank:
                gen_tokens.append(tok)
                gen_frames.append(frame)
                gen_durations.append(dur)
            frame += dur
        gen_text = processor.batch_decode(gen.sequences, skip_special_tokens=True)[0]
        loop_tokens = trace["tokens"].tolist()
        tensors = {
            "features": feats[0].contiguous(),
            "features_len": mask.sum(-1).to(torch.int64).reshape(1),
            "subsampling": layers["subsampling"],
            "encoded_len": torch.tensor([enc_len], dtype=torch.int64),
            "encoder_out": enc.contiguous(),
            "enc_proj": enc_proj.contiguous(),
        }
        for n in range(n_layers):
            tensors[f"layer.{n}"] = layers[f"layer.{n}"]
        tensors.update({k: v.contiguous() for k, v in trace.items()})
        info = {
            "id": clip["id"],
            "seconds": secs,
            "features_shape": list(feats.shape[1:]),
            "features_len": int(mask.sum()),
            "subsampling_shape": list(layers["subsampling"].shape),
            "encoded_len": enc_len,
            "encoder_out_equals_last_layer": bool(torch.equal(enc, layers[f"layer.{n_layers - 1}"])),
            "tokens": loop_tokens,
            "token_frame": trace["token_frame"].tolist(),
            "token_duration": trace["token_duration"].tolist(),
            "loop_text": processor.decode(loop_tokens, skip_special_tokens=True),
            "generate_tokens": gen_tokens,
            "generate_token_frame": gen_frames,
            "generate_token_duration": gen_durations,
            "generate_text": gen_text,
            "loop_equals_generate": loop_tokens == gen_tokens
            and trace["token_frame"].tolist() == gen_frames
            and trace["token_duration"].tolist() == gen_durations,
        }
        if args.nemo_dir:
            nemo = load_file(os.path.join(args.nemo_dir, clip["id"] + ".safetensors"))
            nf = nemo["features"][None]
            nmask = (torch.arange(nf.shape[1]) < int(nemo["features_len"][0]))[None]
            captured.clear()
            with torch.inference_mode():
                out2 = model.get_audio_features(input_features=nf, attention_mask=nmask, output_attention_mask=True)
            tensors["on_nemo_features.subsampling"] = captured["subsampling"][0].contiguous()
            for n in range(n_layers):
                tensors[f"on_nemo_features.layer.{n}"] = captured[f"layer.{n}"][0].contiguous()
            tensors["on_nemo_features.enc_proj"] = out2.pooler_output[0].contiguous()
        save_file(tensors, os.path.join(args.out, clip["id"] + ".safetensors"))
        with open(os.path.join(args.out, clip["id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1)
        print(
            f"{clip['id']}: {secs:.1f} s, F={feats.shape[1]} len={int(mask.sum())}, "
            f"T={layers['subsampling'].shape[0]} len={enc_len}, {len(loop_tokens)} tokens, "
            f"loop==generate {info['loop_equals_generate']}\n  {gen_text}",
            flush=True,
        )
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
