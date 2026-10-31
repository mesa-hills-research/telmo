# Telmo

Telmo is a speech recognition engine in pure Rust that runs on the CPU, in a library with no
dependencies. It transcribes with two open models, both with punctuation and capitals.

- IBM's [Granite Speech 5.0 470M TurboCTC](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc),
  for English, French, German, Spanish, Portuguese, Italian and Japanese. It's the default, with
  8-bit weights in a 477 MB pack.
- NVIDIA's [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), for
  English and 24 other European languages, with a timestamp for every token. Telmo uses it for
  the 19 languages Granite doesn't know.

The core uses only the standard library and brings its own safetensors, GGUF and JSON readers,
FFT, resampler, matrix kernels and thread pool. A model is converted once into a pack, Telmo's
own weight file, which loads in milliseconds by mapping it into memory.

## Install

```text
cargo install telmo
```

The `telmo` command downloads models over HTTPS with [ureq](https://crates.io/crates/ureq) and
[rustls](https://crates.io/crates/rustls).

## Transcribe

```text
telmo transcribe speech.wav
```

The first time, Telmo asks before it downloads Granite Speech from Hugging Face (about 950 MB,
packed to 477 MB) into your cache directory. `--model parakeet` uses Parakeet instead, from
NVIDIA's 8-bit file (714 MB, packed to 723 MB). `telmo models` lists what's installed, and
`telmo help` lists the options.

To download and pack a model yourself, fetch its files from Hugging Face and convert them.

```text
curl -L --create-dirs -o granite/config.json https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc/resolve/1dd23f46f6d4e6eae2531982ba2e33220086dd97/config.json
curl -L -o granite/tokenizer.json https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc/resolve/1dd23f46f6d4e6eae2531982ba2e33220086dd97/tokenizer.json
curl -L -o granite/model.safetensors https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc/resolve/1dd23f46f6d4e6eae2531982ba2e33220086dd97/model.safetensors
telmo pack granite granite-int8.pack
telmo transcribe --model granite-int8.pack speech.wav
```

## Use it as a library

The default `download` feature is for the command. Turn it off and the library builds with no
dependencies.

```toml
[dependencies]
telmo = { version = "0.1", default-features = false }
```

```rust
use telmo::{Model, Options};

let model = Model::load("granite-int8.pack".as_ref(), &Options::default())?;
let audio = telmo::wav::read("speech.wav".as_ref())?;
let samples = telmo::resample::resample(&audio.samples, audio.sample_rate, 16000);
println!("{}", model.transcribe(&samples)?.text);
```

`Model::load` also takes a model's directory or NVIDIA's GGUF file and converts it in memory.
`ModelType::for_language` picks the model for a language. `telmo::stream` transcribes audio
as it arrives, with settled and tentative text, and is described in
[docs/streaming.md](docs/streaming.md).

## Accuracy and speed

| Model | Weights | Pack | LibriSpeech clean | LibriSpeech other |
|---|---|---:|---:|---:|
| Granite Speech 5.0 TurboCTC | int8 (default) | 477 MB | 1.33% | 2.57% |
| Granite Speech 5.0 TurboCTC | bf16 | 946 MB | 1.32% | 2.56% |
| Parakeet TDT 0.6B v3 | NVIDIA's q8_0 | 723 MB | 1.92% | 3.59% |
| Parakeet TDT 0.6B v3 | f32 | 2,509 MB | 1.93% | 3.60% |

The numbers are word error rates on the full test sets. Parakeet's equal its model card's.
Granite's are about 0.3 points above its card, which comes from IBM's own GPU run, and Telmo's
transcripts equal those of the reference implementation (Hugging Face transformers) on every
utterance compared. On short FLEURS samples Parakeet makes fewer errors than Granite in French
(6.3% of words against 9.3%) and Portuguese (7.7% against 11.1%), so it can be the better choice
there.

On 6 cores of an Intel Core i9-13980HX, one pass over 60 s of audio runs about 20 times faster
than real time with Parakeet, and Granite runs about 1.5 times faster than that. The method,
the other measurements and the hardware are in [docs/measurements.md](docs/measurements.md).

## Status

- Offline transcription of both models matches their reference implementations and is ready to
  use.
- Streaming is experimental. Its API may change, and its accuracy is a little below offline
  transcription of the same audio.
- The AVX2 and plain Rust kernels are the defaults. AVX-512 and NEON kernels exist and pass the
  tests against the plain ones, but they haven't run on real hardware yet, so they're used only
  when you ask for them (`--isa avx512` or `neon`, `Options::isa`, or the `TELMO_ISA`
  environment variable). On Apple silicon that means the plain kernels by default, which are
  slower.
- Output is bit-identical at every thread count, because threads split the matrix products by
  output columns.

## Tests

`cargo test` checks the readers, the kernels, the front ends, voice activity, the resampler and
the WER scorer. A clone of the repository also holds golden tensors from the reference
implementations, and the tests that compare every layer with them need the model files.

```text
PARAKEET_TDT_DIR=parakeet-tdt-0.6b-v3 cargo test --release --test goldens -- --include-ignored
GRANITE_SPEECH_DIR=granite-speech-5.0-470m-turboctc cargo test --release --test granite_goldens -- --include-ignored
```

## License

Telmo is under the MIT License or the Apache License 2.0, at your option. The models aren't
part of it. Granite Speech is IBM's, under the Apache License 2.0. Parakeet is NVIDIA's, under
CC BY 4.0, which asks anyone who shares it, or a pack made from it, to credit NVIDIA. The test
clips in the repository come from LibriSpeech and FLEURS (both CC BY 4.0). [NOTICE](NOTICE)
lists the details.
