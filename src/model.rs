//! Either supported model, chosen by the `model_type` in its `config.json`.

use std::path::Path;

use crate::error::{Error, Result};
use crate::granite::Granite;
use crate::json;
use crate::kernels::Isa;
use crate::parakeet::{Options, Parakeet, Transcript};
use crate::quant::Precision;

/// A loaded model of either kind.
pub enum Model {
    Parakeet(Box<Parakeet>),
    Granite(Box<Granite>),
}

impl Model {
    /// Loads the model in `dir`, Parakeet TDT for a `model_type` of `parakeet_tdt` and Granite
    /// Speech CTC for `granite_speech5_ctc`.
    pub fn load(dir: &Path, options: &Options) -> Result<Model> {
        let path = dir.join("config.json");
        let bytes = std::fs::read(&path).map_err(|e| Error::io(&path, e))?;
        let root = json::parse(&bytes)?;
        match root.get("model_type").and_then(|t| t.as_str()) {
            Some("parakeet_tdt") => Ok(Model::Parakeet(Box::new(Parakeet::load(dir, options)?))),
            Some("granite_speech5_ctc") => Ok(Model::Granite(Box::new(Granite::load(dir, options)?))),
            other => Err(Error::Model(format!(
                "config.json names a model type Notate doesn't run ({}).",
                other.unwrap_or("none")
            ))),
        }
    }

    /// The model's name for messages.
    pub fn name(&self) -> &'static str {
        match self {
            Model::Parakeet(_) => "Parakeet TDT",
            Model::Granite(_) => "Granite Speech",
        }
    }

    /// Transcribes 16 kHz mono samples.
    pub fn transcribe(&self, samples: &[f32]) -> Result<Transcript> {
        match self {
            Model::Parakeet(m) => m.transcribe(samples),
            Model::Granite(m) => m.transcribe(samples),
        }
    }

    pub fn isa(&self) -> Isa {
        match self {
            Model::Parakeet(m) => m.isa(),
            Model::Granite(m) => m.isa(),
        }
    }

    pub fn threads(&self) -> usize {
        match self {
            Model::Parakeet(m) => m.threads(),
            Model::Granite(m) => m.threads(),
        }
    }

    /// The precision of the weight matrices, if they share one.
    pub fn precision(&self) -> Option<Precision> {
        match self {
            Model::Parakeet(m) => m.precision(),
            Model::Granite(_) => Some(Precision::F32),
        }
    }

    /// Bytes of weights held in memory.
    pub fn weight_bytes(&self) -> usize {
        match self {
            Model::Parakeet(m) => m.weight_bytes(),
            Model::Granite(m) => m.weight_bytes(),
        }
    }
}
