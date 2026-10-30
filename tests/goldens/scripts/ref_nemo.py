"""Run NVIDIA NeMo's Parakeet TDT 0.6B v3 on the golden clips and save every tensor the goldens need.

Usage:
  python -I ref_nemo.py <model.nemo> <clips.json> <out dir> [--threads 6] [--no-subsampling-mask]

For each clip it writes <out dir>/<id>.safetensors (features, subsampling output, all 24 layer outputs,
the encoder output, the joint's projections, the decoding trace) and <out dir>/<id>.json (texts, tokens,
checks). The model runs in eval mode, float32, on the CPU, with the preprocessor's dither and pad_to at 0.

--no-subsampling-mask is an ablation: it turns off the time masking inside NeMo's ConvSubsampling
(MaskedConvSequential) to measure what that masking changes. Its outputs are not goldens.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import save_file

MAX_SYMBOLS = 10
JOINT_STEPS = 20


def load_clip_audio(path):
    data, sr = sf.read(path, dtype="int16", always_2d=False)
    assert sr == 16000 and data.ndim == 1
    return (data.astype(np.float32) / 32768.0).astype(np.float32)


def greedy_tdt(model, enc_proj, enc_len):
    """Greedy TDT over NeMo's modules, following GreedyBatchedTDTLabelLoopingComputer.torch_impl.

    enc_proj: [T, 640] joint.enc(encoder output). Returns the decoding trace.
    """
    joint = model.joint
    decoder = model.decoder
    blank = decoder.blank_idx
    durations = list(model.cfg.decoding.durations)
    num_durations = len(durations)

    def pred_step(label, state):
        g, state, *_ = decoder.predict(torch.tensor([[label]], dtype=torch.long), state, add_sos=False, batch_size=1)
        return joint.project_prednet(g)[0, 0], state  # joint.pred(decoder output), [640]

    # start state: the SOS label is blank (a zero embedding row) with a zero LSTM state, as in torch_impl
    state = decoder.initialize_state(enc_proj[None])
    pred, state = pred_step(blank, state)
    preds = [pred]
    joints, joint_frames = [], []
    tokens, token_frames, token_durations = [], [], []
    t = 0
    run_frame, run_count = -1, 0  # non-blank labels emitted at the same frame
    while t < enc_len:
        logits = joint.joint_net(enc_proj[t] + pred)  # raw logits, [8198]
        # NeMo's joint_after_projection would log_softmax these on CPU (log_softmax: null); argmaxes agree
        if len(joints) < JOINT_STEPS:
            joints.append(logits.clone())
            joint_frames.append(t)
        k = int(logits[:-num_durations].argmax())
        d = durations[int(logits[-num_durations:].argmax())]
        if k == blank:
            t += max(d, 1)  # a blank with duration 0 advances by 1
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
            t += 1  # at most MAX_SYMBOLS labels per frame
    return {
        "pred": torch.stack(preds),
        "joint": torch.stack(joints),
        "joint_frame": torch.tensor(joint_frames, dtype=torch.int64),
        "tokens": torch.tensor(tokens, dtype=torch.int64),
        "token_frame": torch.tensor(token_frames, dtype=torch.int64),
        "token_duration": torch.tensor(token_durations, dtype=torch.int64),
    }


def as_list(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return [int(v) for v in x.tolist()]
    return [int(v) for v in x]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("nemo")
    ap.add_argument("clips")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--no-subsampling-mask", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated clip ids")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.parts.submodules import subsampling as nemo_subsampling

    if args.no_subsampling_mask:
        nemo_subsampling.apply_channel_mask = lambda tensor, mask: tensor

    t0 = time.time()
    model = ASRModel.restore_from(args.nemo, map_location="cpu")
    model = model.float().eval()
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
    # same greedy_batch decoding, but have NeMo report each token's duration too
    from omegaconf import open_dict

    dcfg = model.cfg.decoding
    with open_dict(dcfg):
        dcfg.tdt_include_token_duration = True
    model.change_decoding_strategy(dcfg, verbose=False)
    model.eval()
    load_secs = time.time() - t0
    print(f"loaded {type(model).__name__} in {load_secs:.1f} s", flush=True)
    emb = model.decoder.prediction["embed"].weight
    assert float(emb[model.decoder.blank_idx].abs().max()) == 0.0, "blank embedding row is not zero"

    captured = {}
    hooks = [
        model.encoder.pre_encode.register_forward_hook(
            lambda m, i, o: captured.__setitem__("subsampling", o[0].detach().clone())
        )
    ]
    for n, layer in enumerate(model.encoder.layers):
        hooks.append(
            layer.register_forward_hook(
                lambda m, i, o, n=n: captured.__setitem__(f"layer.{n}", (o[0] if isinstance(o, tuple) else o).detach().clone())
            )
        )

    clips = json.load(open(args.clips, encoding="utf-8"))
    if args.only:
        keep = set(args.only.split(","))
        clips = [c for c in clips if c["id"] in keep]
    for clip in clips:
        audio = load_clip_audio(clip["wav"])
        assert len(audio) == clip["samples"]
        # model.transcribe (NeMo 3.0.0) ends with encoder/decoder/joint.unfreeze(), which calls .train() and
        # turns dropout on for any later direct call, so eval mode is restored after it and checked here
        training = [name for name, m in model.named_modules() if m.training]
        assert not training, f"modules in training mode: {training[:5]}"
        captured.clear()
        t1 = time.time()
        with torch.inference_mode():
            signal = torch.from_numpy(audio)[None]
            length = torch.tensor([len(audio)], dtype=torch.int64)
            feats, feats_len = model.preprocessor(input_signal=signal, length=length)
            enc, enc_len = model.encoder(audio_signal=feats, length=feats_len)
            enc_t = enc.transpose(1, 2)[0]  # [T, 1024]
            enc_proj = model.joint.project_encoder(enc_t)  # [T, 640]
            trace = greedy_tdt(model, enc_proj, int(enc_len[0]))
        secs = time.time() - t1
        tensors = {
            "features": feats[0].transpose(0, 1).contiguous(),  # [F, 128]
            "features_len": feats_len.to(torch.int64).reshape(1),
            "subsampling": captured["subsampling"][0].contiguous(),
            "encoded_len": enc_len.to(torch.int64).reshape(1),
            "encoder_out": enc_t.contiguous(),
            "enc_proj": enc_proj.contiguous(),
        }
        for n in range(len(model.encoder.layers)):
            tensors[f"layer.{n}"] = captured[f"layer.{n}"][0].contiguous()
        tensors.update({k: v.contiguous() for k, v in trace.items()})
        same_last_layer = torch.equal(tensors["encoder_out"], tensors[f"layer.{len(model.encoder.layers) - 1}"])
        info = {
            "id": clip["id"],
            "seconds": secs,
            "features_shape": list(tensors["features"].shape),
            "features_len": int(feats_len[0]),
            "subsampling_shape": list(tensors["subsampling"].shape),
            "encoded_len": int(enc_len[0]),
            "encoder_out_equals_last_layer": bool(same_last_layer),
            "tokens": as_list(trace["tokens"]),
            "token_frame": as_list(trace["token_frame"]),
            "token_duration": as_list(trace["token_duration"]),
            "loop_text": model.tokenizer.ids_to_text(as_list(trace["tokens"])),
            "no_subsampling_mask": args.no_subsampling_mask,
        }
        if not args.no_subsampling_mask:
            # NeMo's own decoding over the same encoder output, and model.transcribe end to end
            with torch.inference_mode():
                dec_hyps = model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=enc, encoded_lengths=enc_len, return_hypotheses=True
                )
            dec_hyp = dec_hyps[0][0] if isinstance(dec_hyps, tuple) else dec_hyps[0]
            info["decoding_tokens"] = as_list(dec_hyp.y_sequence)
            info["decoding_token_frame"] = as_list(getattr(dec_hyp, "timestamp", None))
            info["decoding_token_duration"] = as_list(getattr(dec_hyp, "token_duration", None))
            info["decoding_text"] = dec_hyp.text
            # NeMo's joint as the decoder calls it: log_softmax over all 8198 outputs on CPU (log_softmax: null)
            with torch.inference_mode():
                j0 = model.joint.joint_after_projection(enc_proj[None, :1], trace["pred"][None, :1])[0, 0, 0]
            info["joint_log_softmax_setting"] = model.joint.log_softmax
            info["joint_after_projection_minus_raw_max_abs"] = float((j0 - trace["joint"][0]).abs().max())
            info["joint_after_projection_minus_log_softmax_raw_max_abs"] = float(
                (j0 - trace["joint"][0].log_softmax(-1)).abs().max()
            )
            out = model.transcribe([audio], batch_size=1, return_hypotheses=True, verbose=False)
            model.eval()
            hyp = out[0][0] if isinstance(out, tuple) else out[0]
            info["transcribe_tokens"] = as_list(hyp.y_sequence)
            info["transcribe_text"] = hyp.text
            ts = getattr(hyp, "timestamp", None)
            info["transcribe_token_frame"] = as_list(ts) if isinstance(ts, (list, torch.Tensor)) else None
            info["transcribe_token_duration"] = as_list(getattr(hyp, "token_duration", None))
            info["loop_equals_transcribe"] = (
                info["tokens"] == info["transcribe_tokens"]
                and info["token_frame"] == info["transcribe_token_frame"]
                and info["token_duration"] == info["transcribe_token_duration"]
            )
            info["loop_equals_decoding"] = (
                info["tokens"] == info["decoding_tokens"]
                and info["token_frame"] == info["decoding_token_frame"]
                and info["token_duration"] == info["decoding_token_duration"]
            )
        save_file(tensors, os.path.join(args.out, clip["id"] + ".safetensors"))
        with open(os.path.join(args.out, clip["id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1)
        print(
            f"{clip['id']}: {secs:.1f} s, F={tensors['features'].shape[0]} len={int(feats_len[0])}, "
            f"T={tensors['subsampling'].shape[0]} len={int(enc_len[0])}, {len(info['tokens'])} tokens, "
            f"loop==transcribe {info.get('loop_equals_transcribe')}, loop==decoding {info.get('loop_equals_decoding')}\n"
            f"  {info.get('transcribe_text', info['loop_text'])}",
            flush=True,
        )
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
