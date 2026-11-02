//! Stands in for the microphone in a build without the `record` feature.

use std::sync::Arc;

use crate::capture::Capture;

const MESSAGE: &str =
    "This build can't record. Install it with cargo install telmo --features record.";

pub struct Mic {
    pub name: String,
    pub sample_rate: u32,
    pub capture: Arc<Capture>,
}

pub fn devices() -> Result<Vec<(String, bool)>, String> {
    Err(MESSAGE.into())
}

pub fn open(_device: Option<&str>) -> Result<Mic, String> {
    Err(MESSAGE.into())
}
