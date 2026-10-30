# Measurements

Parakeet TDT 0.6B v3 (`nvidia/parakeet-tdt-0.6b-v3` at revision `541d1f9`) and Granite Speech
5.0 470M TurboCTC (`ibm-granite/granite-speech-5.0-470m-turboctc` at revision `1dd23f4`, the
multilingual release), both with greedy decoding. Streaming's latency and accuracy are in
[streaming.md](streaming.md#latency-and-accuracy).

## Machine

6 vCPUs of an Intel Core i9-13980HX under VMware, one thread per core, with AVX2, FMA and F16C
(no AVX-512), 12 MiB of L2 and 36 MiB of L3, 22 GiB of memory, Ubuntu 26.04 and Rust 1.97.1. The
VM shares its host with other work. The f32 speed tables of Parakeet were taken in a quiet
moment. The tables that compare precisions and models were taken while other jobs kept a load
average of 7 to 9 on the 6 cores, so their times are up to 1.7 times a quiet machine's and only
compare with each other.

## Word error rate

| Model | Weights | test-clean | test-other | Model card |
|---|---|---:|---:|---:|
| Parakeet TDT 0.6B v3 | f32 | 1.93% | 3.60% | 1.93%, 3.59% |
| Parakeet TDT 0.6B v3 | int8 | 1.94% | 3.60% | |
| Parakeet TDT 0.6B v3 | NVIDIA's q8_0 GGUF | 1.92% | 3.59% | |
| Granite Speech 5.0 TurboCTC | bf16 | 1.32% | 2.56% | 1.03%, 2.26% |
| Granite Speech 5.0 TurboCTC | int8 | 1.33% | 2.57% | |
| Granite Speech 5.0 TurboCTC | q8_0 | 1.34% | 2.55% | |

LibriSpeech test-clean has 2,620 utterances and 53,027 words, test-other 2,939 and 52,882. Both
texts are normalized as the Open ASR leaderboard normalized them in 2025, when Parakeet's card
was published (`examples/wer/normalize.rs`), then the word errors of all utterances are summed
and divided by the reference words. `--normalizer 2026` applies the leaderboard's current
normalizer instead, which lowers these numbers by 0.04 points at most (Granite's to 1.28% and
2.55%).

```text
cargo run --release --example wer -- MODEL test-clean.tsv test-other.tsv --precision int8 --out hyp
```

Each narrower precision is compared with f32 (bf16 for Granite) by a paired bootstrap over
utterances (10,000 resamples, `--compare`), which gives the change in WER and its 95% interval.

| Model | Weights | Set | Change | 95% interval | Transcripts equal to f32's or bf16's |
|---|---|---|---:|---:|---:|
| Parakeet | int8 | test-clean | +0.02 | −0.02 to +0.05 | 92.1% (97.3% after normalization) |
| Parakeet | int8 | test-other | −0.00 | −0.04 to +0.04 | 90.4% (96.1%) |
| Parakeet | q8_0 GGUF | test-clean | −0.00 | −0.03 to +0.02 | 96.5% (98.5%) |
| Parakeet | q8_0 GGUF | test-other | −0.01 | −0.04 to +0.02 | 95.0% (97.7%) |
| Granite | int8 | test-clean | +0.02 | −0.00 to +0.03 | 96.7% (99.4%) |
| Granite | int8 | test-other | +0.01 | −0.01 to +0.03 | 94.6% (97.7%) |
| Granite | q8_0 | test-clean | +0.02 | +0.00 to +0.03 | 97.6% (99.5%) |
| Granite | q8_0 | test-other | −0.01 | −0.03 to +0.02 | 96.3% (98.5%) |

Granite's f32 and bf16 weights give bit-identical results, since its checkpoint is bf16.

### Choosing a precision

8-bit weights take about a quarter of f32's memory and cost Parakeet nothing measurable. Its q8_0
weights, with a scale for each block of 32 inputs, agree with f32 more often than int8 with one
scale per output channel. NVIDIA's Parakeet GGUF file (714 MB) converts straight into a q8_0
pack, so it can be downloaded instead of the 2.5 GB f32 checkpoint.

Granite's 8-bit weights cost about 0.02 points on test-clean and nothing on test-other. For int8
that stays inside the bootstrap interval, and for q8_0 it lands just outside (+0.002 to +0.034).
The two don't differ measurably from each other.

`ModelType::recommended_precision` is q8_0 for Parakeet (723 MB) and int8 for Granite (477 MB,
534 MB in q8_0), and the pack example converts to it unless told otherwise.

### Choosing a model

Granite with int8 weights is Telmo's default (`ModelType::DEFAULT`). It has the smallest pack and
the lowest WER on both LibriSpeech sets, within the bootstrap interval of bf16. `ModelType::for_language`
falls back to Parakeet with q8_0 weights for the 19 European languages Granite doesn't know. In the
FLEURS table below Parakeet makes fewer errors in every language the two share, most of all in
French (6.31% against 9.30%) and Portuguese (7.65% against 11.08%), so it's worth choosing for those
two despite its larger pack.

### Agreement with the reference implementations

NVIDIA NeMo 3.0.0 and transformers 5.22.0 transcribed a fixed random subset of 500 utterances of
each set on the same machine, in f32.

| Model | Subset | Reference | Telmo | Identical transcripts |
|---|---|---:|---:|---:|
| Parakeet (NeMo) | test-clean, 500 | 1.89% | 1.89% | 500 |
| Parakeet (NeMo) | test-other, 500 | 3.45% | 3.45% | 499 |
| Granite (transformers) | test-clean, 500 | 1.33% | 1.33% | 500 |
| Granite (transformers) | test-other, 500 | 2.65% | 2.65% | 500 |

The one Parakeet transcript that differs reads "authority - that is" where NeMo's reads
"authority- that is", the same after normalization. Granite's card reports 1.03% and 2.26% from
IBM's own GPU run in bf16. Telmo's transcripts equal transformers' on every utterance here, and
the 2026 normalizer only brings the full sets to 1.28% and 2.55%, so the gap lies outside both.

On the full sets Granite's WER is lower than Parakeet's by 0.61 points on test-clean (95%
interval 0.49 to 0.72) and 1.04 points on test-other (0.88 to 1.20).

### FLEURS

The first 100 utterances of the FLEURS test sets, scored against the raw transcriptions (with
digits and punctuation) after Whisper's multilingual normalizer, or the leaderboard's English one
for English. Japanese is character error rate.

| Language | Parakeet, f32 | Granite, f32 |
|---|---:|---:|
| English | 7.41% | 7.65% |
| German | 6.17% | 6.60% |
| French | 6.31% | 9.30% |
| Spanish | 4.66% | 5.95% |
| Italian | 5.47% | 7.45% |
| Portuguese | 7.65% | 11.08% |
| Japanese | | 5.61% |

Both models write numbers and some compounds differently from FLEURS's transcriptions, which
raises both columns. On these 100 utterances per language Parakeet does better everywhere, where
IBM's chart, over other test sets, puts Granite ahead in German, Spanish, French and Italian.
Granite's transcripts equal transformers' on all 700.

## Agreement layer by layer

`tests/goldens.rs` and `tests/granite_goldens.rs` compare each stage with tensors that NeMo and
transformers computed on the CPU in f32. Errors are relative L2 (‖Telmo − reference‖ /
‖reference‖) except for the features, which are absolute.

| Parakeet clip | Features, max | Features, mean | Subsampling | Layer 0 | Layer 12 | Layer 23 | Joint logits | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| clean-3575-170457-0009 | 4.5e-5 | 3.3e-7 | 1.1e-6 | 1.1e-6 | 1.4e-6 | 2.3e-5 | 8.3e-7 | identical |
| clean-1995-1826-0007 | 4.6e-5 | 3.3e-7 | 1.2e-6 | 1.3e-6 | 1.2e-6 | 1.6e-6 | 6.6e-7 | identical |
| clean-260-123286-0018 | 5.7e-5 | 2.8e-7 | 1.1e-6 | 1.2e-6 | 1.3e-6 | 3.2e-6 | 5.5e-7 | identical |
| other-367-130732-0011 | 1.6e-4 | 8.4e-7 | 3.5e-6 | 3.1e-6 | 1.4e-6 | 2.4e-6 | 5.4e-7 | identical |
| other-4350-9170-0030 | 4.4e-5 | 4.3e-7 | 1.2e-6 | 1.2e-6 | 1.6e-6 | 2.1e-6 | 6.7e-7 | identical |
| clean-joined-40s | 6.2e-5 | 2.8e-7 | 1.0e-6 | 9.8e-7 | | 7.6e-6 | 7.0e-7 | identical |

| Granite clip | Features, max | Block 0 | Block 1 | Block 7 | Self-conditioning | Block 15 | Max logit | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| clean-1995-1826-0007 | 1.1e-5 | 1.1e-6 | 9.0e-7 | 1.1e-6 | 9.1e-7 | 1.4e-6 | 5.1e-7 | identical |
| clean-260-123286-0018 | 8.3e-6 | 1.0e-6 | 8.6e-7 | 1.3e-6 | 8.6e-7 | 4.6e-6 | 4.0e-6 | identical |
| other-367-130732-0011 | 3.2e-6 | 8.6e-7 | 7.8e-7 | 1.2e-6 | 8.8e-7 | 1.4e-6 | 4.4e-7 | identical |
| other-4350-9170-0030 | 6.6e-6 | 1.1e-6 | 9.4e-7 | 1.2e-6 | 9.9e-7 | 1.7e-6 | 5.3e-7 | identical |
| clean-joined-40s | | | 1.1e-6 | | | 3.0e-6 | 2.1e-6 | identical |
| fleurs-ja_jp-11258446014622421491 | 2.7e-6 | | | | | 1.9e-6 | 5.9e-7 | identical |

Parakeet's tokens include each token's frame and duration, and Granite's the CTC argmax of every
frame. The largest feature differences sit in the lowest mel filters, which sum only one or two
FFT bins, where the references' f32 STFT rounds most. The scalar kernels give the same tokens as
the AVX2 kernels on every clip, and the encoder output is bit-identical at 1, 2, 3 and 6 threads.

With narrower weights the same tests compare every saved layer by cosine similarity.

| Weights | Parakeet, worst layer | Parakeet, clips with identical tokens | Granite, worst layer | Granite, clips with identical tokens |
|---|---:|---:|---:|---:|
| f16 | 0.99988 | 6 of 6 | 1.00000 | 6 of 6 |
| bf16 | 0.99976 | 6 of 6 | exact | 6 of 6 |
| int8 | 0.99629 | 6 of 6 | 0.99829 | 5 of 6 |
| q8_0 | 0.99928 | 6 of 6 | 0.99870 | 5 of 6 |
| NVIDIA's q8_0 GGUF | 0.99841 | 6 of 6 | | |

Parakeet's worst layer is always the last, whose output has a few channels far larger than the
rest. Granite's two changed clips gain two commas (Japanese, int8) and split one name (q8_0).

## Speed

Parakeet with f32 weights, on a quiet machine.

| Audio | 1 thread, ms | Times real time | 6 threads, ms | Times real time |
|---:|---:|---:|---:|---:|
| 2 s | 495 | 4.0 | 121 | 16.5 |
| 5 s | 1,009 | 5.0 | 245 | 20.4 |
| 10 s | 1,867 | 5.4 | 447 | 22.4 |
| 20 s | 3,657 | 5.5 | 883 | 22.7 |
| 30 s | 5,691 | 5.3 | 1,382 | 21.7 |
| 60 s | 12,646 | 4.7 | 3,024 | 19.8 |

`cargo bench --bench rtf -- MODEL test-clean.tsv` joins consecutive test-clean utterances
into clips of each length and reports the best of several passes. Short passes pay fixed costs per
call and long ones pay for attention over every pair of frames, so 10 to 30 s run fastest per
second of audio.

Both models and their precisions, with other jobs keeping a load average of 7 to 9 (best of 3,
times real time).

| Model | Weights | 5 s, 1 thread | 60 s, 1 thread | 5 s, 6 threads | 60 s, 6 threads |
|---|---|---:|---:|---:|---:|
| Parakeet | f32 | 3.9 | 3.8 | 10.2 | 11.6 |
| Parakeet | int8 | 5.0 | 3.9 | 13.1 | 11.2 |
| Parakeet | q8_0 GGUF | 4.3 | 4.0 | 13.1 | 13.2 |
| Granite | bf16 | 4.3 | 5.0 | 17.8 | 17.1 |
| Granite | int8 | 5.1 | 4.1 | 19.1 | 21.8 |

Narrower weights barely change the encoder's time over a whole utterance, because the matrix
product widens each block of weights once and reuses it for every frame, so it stays bound by
arithmetic. Within one model the differences are mostly the load. Where weights are read for few
rows, as in Parakeet's decoder, they help. Granite runs a 60 s pass about 1.5 times faster than
Parakeet on 6 threads, since its attention stays inside blocks and it has no transducer decoder.

This is where the time of a 60 s pass of Parakeet in f32 goes, on a quiet machine.

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

Granite spends about half of a pass in its feed-forward layers, 14% in the convolution module's
first pointwise layer, 11% in q, k and v, 5% in self-conditioning, 3% in the CTC head and 3% in
attention.

### Parakeet's decoder

Each emitted token runs the prediction network (two LSTM layers and a projection), and each step
of the joint reads the joint's weights. These are matrix products of one row, which read every
weight once, so their weight precision sets their cost.

| Weights | MB read per token (LSTM, projection, joint) | LSTM step, 1 thread, us | Joint step, 1 thread, us | Decoder in a 60 s pass, 1 thread, ms |
|---|---:|---:|---:|---:|
| f32 | 48.9 | 1,821 | 1,111 | 934 |
| f16 or bf16 | 24.4 | | | |
| int8 | 12.3 | 665 | 465 | 368 |
| q8_0 | 13.8 | 690 | 488 | 384 |

These times come from the 60 s passes of the table above. One thread doesn't reach the memory's
bandwidth with 8-bit weights, so widening them costs more than reading them and the gain is 2.6
times rather than 4. `cargo bench --bench weights` times the same shapes on their own.

### Matrix product

`cargo bench --bench matmul` measures the matrix product on the encoder's shapes, in GFLOPS, with
f32 weights on a quiet machine.

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
registers and walks the weights in blocks of 256 inputs. f16, bf16 and 8-bit weights are widened
in registers when all rows fit in one tile, and otherwise into a buffer that stays in L1 while
every row reuses it.

## Memory and loading

A pack is mapped, so its weights count as shared file memory, and loading takes milliseconds
once the file is in the page cache. Converting a downloaded checkpoint in memory takes seconds.

| Model | Weights | Pack | Private memory, 5 s pass | Mapped weights | Load from pack | Convert from the download |
|---|---|---:|---:|---:|---:|---:|
| Parakeet | f32 | 2,509 MB | 7 MB | 2,493 MB | 11 ms | 4.1 s |
| Parakeet | f16 | 1,267 MB | 7 MB | 1,251 MB | 10 ms | 3.7 s |
| Parakeet | int8 | 647 MB | 8 MB | 638 MB | 13 ms | 6.3 s |
| Parakeet | q8_0 from the GGUF | 723 MB | 8 MB | 707 MB | 12 ms | 1.7 s |
| Granite | bf16 | 946 MB | 12 MB | 949 MB | 15 to 37 ms | 2.5 s |
| Granite | int8 | 477 MB | 12 MB | 480 MB | 15 to 31 ms | 3.0 s |

A 60 s pass adds about 25 MB of private memory. A model converted in memory keeps every weight in
private memory instead (2,662 MB for Parakeet in f32). The first pass after the pack was evicted
from the page cache took 1.1 to 2.1 s for 5 s of audio instead of about 0.6 s, as the weights were
read from disk.

```text
cargo run --release --example pack -- MODEL model.pack --precision q8_0
```
