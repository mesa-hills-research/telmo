//! Speech recognition on the CPU in pure Rust, with no dependencies.
//!
//! Telmo runs IBM's Granite Speech 5.0 470M TurboCTC (English, French, German, Spanish,
//! Portuguese, Italian and Japanese), the default, and NVIDIA's Parakeet TDT 0.6B v3 (25
//! European languages). Their weights aren't part of the crate. Convert a downloaded model once
//! into a pack with [`pack`] (or the `telmo pack` command), then load it with [`Model::load`].
//!
//! ```no_run
//! use telmo::{Model, Options};
//!
//! let model = Model::load("granite-int8.pack".as_ref(), &Options::default())?;
//! let audio = telmo::wav::read("speech.wav".as_ref())?;
//! let samples = telmo::resample::resample(&audio.samples, audio.sample_rate, 16000);
//! println!("{}", model.transcribe(&samples)?.text);
//! # Ok::<(), telmo::Error>(())
//! ```
//!
//! The default `download` feature is for the `telmo` command, which downloads models over
//! HTTPS. Turn it off and the library builds with no dependencies.
//!
//! ```toml
//! [dependencies]
//! telmo = { version = "0.2", default-features = false }
//! ```
//!
//! Offline transcription is what Telmo does today. Streaming ([`stream`]) is experimental. The
//! kernels use AVX-512, AVX2 or NEON where the CPU has them and plain Rust elsewhere, unless
//! [`Options::isa`] or the `TELMO_ISA` environment variable names others.

pub mod error;
pub mod features;
pub mod fft;
pub mod gguf;
pub mod granite;
pub mod granite_features;
pub mod granite_stream;
pub mod json;
pub mod kernels;
pub mod model;
pub mod pack;
pub mod parakeet;
pub mod quant;
pub mod region;
pub mod resample;
pub mod safetensors;
pub mod source;
pub mod stream;
pub mod threads;
pub mod tokenizer;
pub mod vad;
pub mod wav;

pub use error::{Error, Result};
pub use granite::Granite;
pub use granite_stream::GraniteStream;
pub use model::Model;
pub use parakeet::{Options, Parakeet, Transcript};
pub use quant::Precision;
