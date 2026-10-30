# Notate

Notate is a speech recognition engine in pure Rust that runs on the CPU. It transcribes English
and 24 other European languages with NVIDIA's
[Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), with punctuation,
capitals and a timestamp for every token.

- It has no dependencies. The core uses only the standard library and brings its own
  safetensors and JSON readers, FFT, matrix kernels and thread pool.
- Its AVX2 and FMA kernels are chosen at run time, with a scalar fallback for other CPUs.
- Its output is bit-identical at every thread count, because threads split the matrix products
  by output columns.

## Accuracy and speed

| | Notate | Model card |
|---|---:|---:|
| LibriSpeech test-clean WER | 1.93% | 1.93% |
| LibriSpeech test-other WER | 3.60% | 3.59% |

| Audio | 1 thread | 6 threads |
|---:|---:|---:|
| 5 s | 5.0x | 20.4x |
| 20 s | 5.5x | 22.7x |
| 60 s | 4.7x | 19.8x |

The second table gives how many times faster than real time one pass runs, on 6 cores of an
Intel Core i9-13980HX with f32 weights. The transcripts equal NVIDIA NeMo's on 999 of 1,000
LibriSpeech utterances. The method, each step's time, memory and load time are in
[docs/measurements.md](docs/measurements.md).

## Usage

Put the model's `model.safetensors`, `config.json` and `tokenizer.json` in one directory.

```rust
use notate::{Options, Parakeet};

let model = Parakeet::load("parakeet-tdt-0.6b-v3".as_ref(), &Options::default())?;
let audio = notate::wav::read("speech.wav".as_ref())?; // 16 kHz
let transcript = model.transcribe(&audio.samples)?;
println!("{}", transcript.text);
```

The `transcribe` example does the same from the command line.

```text
cargo run --release --example transcribe -- parakeet-tdt-0.6b-v3 speech.wav
```

`examples/wer` scores a test set the way the Open ASR leaderboard does.

## Tests

`cargo test` checks the readers, the kernels, the front end and the WER scorer. The tests that
compare every layer with NVIDIA NeMo's outputs need the model files:

```text
PARAKEET_TDT_DIR=parakeet-tdt-0.6b-v3 cargo test --release --test goldens -- --include-ignored
```

## License

The code is under MIT or Apache-2.0, at your option. The model is NVIDIA's, under CC BY 4.0,
and isn't part of this repository. The test clips come from LibriSpeech (CC BY 4.0), as
[tests/goldens/parakeet-tdt-0.6b-v3/README.md](tests/goldens/parakeet-tdt-0.6b-v3/README.md)
describes.
