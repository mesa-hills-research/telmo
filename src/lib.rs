//! Speech recognition on the CPU in pure Rust.

pub mod error;
pub mod features;
pub mod fft;
pub mod json;
pub mod safetensors;
pub mod wav;

pub use error::{Error, Result};
