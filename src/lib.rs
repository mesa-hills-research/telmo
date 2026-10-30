//! Speech recognition on the CPU in pure Rust.

pub mod error;
pub mod features;
pub mod fft;
pub mod json;
pub mod kernels;
pub mod parakeet;
pub mod quant;
pub mod region;
pub mod safetensors;
pub mod threads;
pub mod tokenizer;
pub mod wav;

pub use error::{Error, Result};
pub use parakeet::{Options, Parakeet, Transcript};
pub use quant::Precision;
