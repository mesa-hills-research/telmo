# Notate

Notate is a speech recognition engine in pure Rust that runs on the CPU. It transcribes with
two models, both with punctuation and capitals.

- IBM's [Granite Speech 5.0 470M TurboCTC](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc),
  for English, French, German, Spanish, Portuguese, Italian and Japanese. It's the default, with
  8-bit weights in a 477 MB pack.
- NVIDIA's [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), for
  English and 24 other European languages, with a timestamp for every token.
  `ModelType::for_language` picks it for the 18 languages Granite doesn't know.

On short FLEURS samples Parakeet makes fewer errors than Granite in French (6.3% of words against
9.3%) and Portuguese (7.7% against 11.1%), so it can be the better choice there too.

It has no dependencies. The core uses only the standard library and brings its own
safetensors, GGUF and JSON readers, FFT, matrix kernels and thread pool.

- Weights can be f32, f16, bf16 or 8-bit integers. A model is converted once into a pack, the
  engine's own weight file, which loads in milliseconds by mapping it into memory.
- It picks AVX-512, AVX2 or NEON kernels at run time, with a scalar fallback for other CPUs.
  The AVX-512 and NEON kernels compile but haven't been run yet.
- Its output is bit-identical at every thread count, because threads split the matrix products
  by output columns.

## Accuracy and speed

| Model | Weights | Pack | LibriSpeech clean | LibriSpeech other |
|---|---|---:|---:|---:|
| Granite Speech 5.0 TurboCTC | int8 (default) | 477 MB | 1.33% | 2.57% |
| Granite Speech 5.0 TurboCTC | bf16 | 946 MB | 1.32% | 2.56% |
| Parakeet TDT 0.6B v3 | NVIDIA's q8_0 | 723 MB | 1.92% | 3.59% |
| Parakeet TDT 0.6B v3 | f32 | 2,509 MB | 1.93% | 3.60% |

The numbers are word error rates. Parakeet's equal its model card's. Granite's are about 0.3
points above the card's, which come from IBM's own GPU run, and Notate's transcripts equal those of
transformers' Granite on every utterance compared (and NVIDIA NeMo's Parakeet on 999 of 1,000). On
6 cores of an Intel Core i9-13980HX, one pass over 60 s of audio runs about 20 times faster than
real time with Parakeet, and 4.7 times on one core. The method, other languages, each
precision's speed and memory, and load times are in [docs/measurements.md](docs/measurements.md).

## Usage

Convert a model once into a pack, in its recommended 8-bit precision unless `--precision` says
otherwise. `MODEL` is a directory with the model's `model.safetensors`, `config.json` and
`tokenizer.json`, or NVIDIA's Parakeet GGUF file with `tokenizer.json` beside it.

```text
cargo run --release --example pack -- MODEL model.pack
```

Then load the pack and transcribe 16 kHz audio.

```rust
use notate::{Model, Options};

let model = Model::load("model.pack".as_ref(), &Options::default())?;
let audio = notate::wav::read("speech.wav".as_ref())?;
let transcript = model.transcribe(&audio.samples)?;
println!("{}", transcript.text);
```

`Model::load` also takes the model's directory or GGUF file, which it converts in memory to
`Options::precision`. The `transcribe` example does the same from the command line, and
`examples/wer` scores a test set the way the Open ASR leaderboard does.

```text
cargo run --release --example transcribe -- model.pack speech.wav
```

## Tests

`cargo test` checks the readers, the kernels, the front ends and the WER scorer. The tests that
compare every layer with NVIDIA NeMo's and transformers' outputs need the model files.

```text
PARAKEET_TDT_DIR=parakeet-tdt-0.6b-v3 cargo test --release --test goldens -- --include-ignored
GRANITE_SPEECH_DIR=granite-speech-5.0-470m-turboctc cargo test --release --test granite_goldens -- --include-ignored
```

## License

The code is under MIT or Apache-2.0, at your option. The models aren't part of this repository.
Parakeet is NVIDIA's, under CC BY 4.0, and Granite Speech is IBM's, under Apache 2.0. The test
clips come from LibriSpeech and FLEURS (both CC BY 4.0), as the READMEs in
[tests/goldens](tests/goldens) describe.
