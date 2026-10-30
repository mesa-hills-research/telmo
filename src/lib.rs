//! Speech recognition on the CPU in pure Rust.

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
pub mod safetensors;
pub mod source;
pub mod threads;
pub mod tokenizer;
pub mod wav;

pub use error::{Error, Result};
pub use granite::Granite;
pub use granite_stream::GraniteStream;
pub use model::Model;
pub use parakeet::{Options, Parakeet, Transcript};
pub use quant::Precision;
