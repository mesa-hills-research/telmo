# Measurements

Parakeet TDT 0.6B v3 (`nvidia/parakeet-tdt-0.6b-v3` at revision `541d1f9`) with f32 weights and
greedy decoding.

## Machine

6 vCPUs of an Intel Core i9-13980HX under VMware, one thread per core, with AVX2 and FMA (no
AVX-512), 12 MiB of L2 and 36 MiB of L3, 22 GiB of memory, Ubuntu 26.04 and Rust 1.97.1. The VM
shares its host with other work, so timings are the best of several runs.

## Word error rate

| Set | Utterances | Words | Notate | Model card |
|---|---:|---:|---:|---:|
| LibriSpeech test-clean | 2,620 | 53,027 | 1.93% | 1.93% |
| LibriSpeech test-other | 2,939 | 52,882 | 3.60% | 3.59% |

Both texts are normalized as the Open ASR leaderboard normalized them in 2025, when the model
card's numbers were published (`examples/wer/normalize.rs`). The word errors of all utterances
are then summed and divided by the reference words.

NVIDIA NeMo 3.0.0 transcribed a fixed random subset of 500 utterances of each set on the same
machine, also in f32 with greedy decoding.

| Subset | NeMo | Notate | Identical transcripts |
|---|---:|---:|---:|
| test-clean, 500 utterances | 1.89% | 1.89% | 500 |
| test-other, 500 utterances | 3.45% | 3.45% | 499 |

The one transcript that differs reads "authority - that is" where NeMo's reads "authority- that
is", which is the same after normalization.

```text
cargo run --release --example wer -- MODEL_DIR test-clean.tsv test-other.tsv --out hyp
```

## Agreement with NeMo, layer by layer

`tests/goldens.rs` compares each stage with tensors that NeMo computed on the CPU in f32 for six
LibriSpeech clips (five of 4 to 7 s and one of 39 s). Errors are relative L2
(‖Notate − NeMo‖ / ‖NeMo‖) except for the features, which are absolute.

| Clip | Features, max | Features, mean | Subsampling | Layer 0 | Layer 12 | Layer 23 | Joint logits | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| clean-3575-170457-0009 | 4.5e-5 | 3.3e-7 | 1.1e-6 | 1.2e-6 | 1.4e-6 | 2.5e-5 | 2.3e-6 | identical |
| clean-1995-1826-0007 | 4.6e-5 | 3.3e-7 | 1.3e-6 | 1.3e-6 | 1.2e-6 | 1.6e-6 | 5.2e-7 | identical |
| clean-260-123286-0018 | 5.7e-5 | 2.8e-7 | 1.1e-6 | 1.2e-6 | 1.3e-6 | 2.7e-6 | 9.0e-7 | identical |
| other-367-130732-0011 | 1.6e-4 | 8.4e-7 | 3.5e-6 | 3.1e-6 | 1.5e-6 | 2.7e-6 | 6.9e-7 | identical |
| other-4350-9170-0030 | 4.4e-5 | 4.3e-7 | 1.2e-6 | 1.3e-6 | 1.5e-6 | 2.1e-6 | 7.4e-7 | identical |
| clean-joined-40s | 6.2e-5 | 2.8e-7 | 1.0e-6 | 1.0e-6 | | 3.5e-6 | 6.9e-7 | identical |

"Tokens" covers the token ids, the frame of each token and its duration. The largest feature
differences sit in the lowest mel filters, which sum only one or two FFT bins, where NeMo's f32
STFT rounds most. The scalar kernels give the same tokens as the AVX2 kernels on every clip, and
the encoder output is bit-identical at 1, 2, 3 and 6 threads.

## Speed

| Audio | 1 thread, ms | Times real time | 6 threads, ms | Times real time |
|---:|---:|---:|---:|---:|
| 2 s | 495 | 4.0 | 121 | 16.5 |
| 5 s | 1,009 | 5.0 | 245 | 20.4 |
| 10 s | 1,867 | 5.4 | 447 | 22.4 |
| 20 s | 3,657 | 5.5 | 883 | 22.7 |
| 30 s | 5,691 | 5.3 | 1,382 | 21.7 |
| 60 s | 12,646 | 4.7 | 3,024 | 19.8 |

`cargo bench --bench rtf -- MODEL_DIR test-clean.tsv` joins consecutive test-clean utterances
into clips of each length and reports the best of 5 passes. Short passes pay fixed costs per
call and long ones pay for attention over every pair of frames, so 10 to 30 s run fastest per
second of audio.

This is where the time of a 60 s pass goes.

| Step | 1 thread, ms | 6 threads, ms |
|---|---:|---:|
| Features | 33 | 44 |
| Subsampling | 329 | 108 |
| Feed-forward in (2 per layer, with SiLU) | 3,204 | 700 |
| Feed-forward out (2 per layer) | 3,207 | 765 |
| q, k and v projections | 1,189 | 269 |
| Position projections | 825 | 203 |
| Attention scores, softmax and values | 1,447 | 319 |
| Attention output projection | 396 | 98 |
| Convolution module, first pointwise | 801 | 184 |
| Convolution module, second pointwise | 403 | 93 |
| LayerNorm, GLU and depthwise convolution | 96 | 44 |
| Prediction network (LSTM) | 405 | 109 |
| Joint | 282 | 61 |
| Other | 23 | 21 |

Matrix products, attention's included, take nearly 90% of a pass. The decoder (prediction
network and joint) reads about 49 MB of weights for every emitted token, so it's bound by memory
bandwidth and gains least from more threads.

`cargo bench --bench matmul` measures the matrix product on the encoder's shapes, in GFLOPS.

| m | k | n | 1 thread | 6 threads |
|---:|---:|---:|---:|---:|
| 8 | 1024 | 4096 | 72 | 227 |
| 32 | 1024 | 4096 | 84 | 328 |
| 128 | 1024 | 4096 | 90 | 391 |
| 512 | 1024 | 4096 | 94 | 446 |
| 8 | 4096 | 1024 | 66 | 286 |
| 32 | 4096 | 1024 | 84 | 332 |
| 128 | 4096 | 1024 | 99 | 470 |
| 512 | 4096 | 1024 | 92 | 407 |
| 8 | 1024 | 1024 | 77 | 241 |
| 32 | 1024 | 1024 | 89 | 244 |
| 128 | 1024 | 1024 | 96 | 279 |
| 512 | 1024 | 1024 | 93 | 416 |

m is the number of encoder frames (12.5 a second). The kernel keeps a 6 x 16 tile of sums in
registers and walks the weights in blocks of 256 inputs.

## Memory and loading

| | |
|---|---:|
| Weights in memory (f32, packed) | 2,506 MB |
| Peak resident memory, 5 s pass | 2,680 MB |
| Peak resident memory, 60 s pass | 2,909 MB |
| Load, file in the page cache | 2.3 s |
| Load, file evicted from the page cache | 7.0 s |

Loading reads `model.safetensors` tensor by tensor and packs each weight matrix into the
kernel's layout, so the file isn't kept in memory.
