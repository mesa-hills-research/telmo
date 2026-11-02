//! Stands in for the microphone in a build without the `record` feature.

const MESSAGE: &str =
    "This build can't record. Install it with cargo install telmo --features record.";

pub struct Mic {
    pub name: String,
    pub sample_rate: u32,
}

pub fn devices() -> Result<Vec<(String, bool)>, String> {
    Err(MESSAGE.into())
}

pub fn open(_device: Option<&str>) -> Result<Mic, String> {
    Err(MESSAGE.into())
}

impl Mic {
    pub fn start(&self) {}

    pub fn stop(&self) -> Vec<f32> {
        Vec::new()
    }

    pub fn level(&self) -> f32 {
        0.0
    }

    pub fn failure(&self) -> Option<String> {
        None
    }
}
