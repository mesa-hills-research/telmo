//! Either supported model, chosen by its pack's model name or its `config.json`.

use std::path::Path;

use crate::error::{Error, Result};
use crate::granite::{self, Granite};
use crate::json;
use crate::kernels::Isa;
use crate::pack::{Pack, Sink};
use crate::parakeet::{self, Options, Parakeet, Transcript};
use crate::quant::Precision;

/// The models Notate runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// NVIDIA's Parakeet TDT, `model_type` `parakeet_tdt`.
    Parakeet,
    /// IBM's Granite Speech CTC, `model_type` `granite_speech5_ctc`.
    Granite,
}

impl ModelType {
    /// The model at `path`, a pack, a checkpoint directory or a GGUF file (which only Parakeet
    /// has).
    pub fn detect(path: &Path) -> Result<ModelType> {
        if path.is_dir() {
            let config = path.join("config.json");
            let bytes = std::fs::read(&config).map_err(|e| Error::io(&config, e))?;
            let root = json::parse(&bytes)?;
            return match root.get("model_type").and_then(|t| t.as_str()) {
                Some("parakeet_tdt") => Ok(ModelType::Parakeet),
                Some("granite_speech5_ctc") => Ok(ModelType::Granite),
                other => Err(Error::Model(format!(
                    "config.json names a model type Notate doesn't run ({}).",
                    other.unwrap_or("none")
                ))),
            };
        }
        if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            return Ok(ModelType::Parakeet);
        }
        ModelType::of_pack(&Pack::open(path)?)
    }

    fn of_pack(pack: &Pack) -> Result<ModelType> {
        match pack.model() {
            parakeet::PACK_MODEL => Ok(ModelType::Parakeet),
            granite::PACK_MODEL => Ok(ModelType::Granite),
            other => Err(Error::Model(format!(
                "The pack holds a model Notate doesn't run ({other})."
            ))),
        }
    }

    /// The model name its packs carry.
    pub fn pack_model(self) -> &'static str {
        match self {
            ModelType::Parakeet => parakeet::PACK_MODEL,
            ModelType::Granite => granite::PACK_MODEL,
        }
    }

    /// Converts the checkpoint at `path` into pack entries in `precision`.
    pub fn convert_path(self, path: &Path, precision: Precision, sink: &mut dyn Sink) -> Result<()> {
        match self {
            ModelType::Parakeet => parakeet::convert_path(path, precision, sink),
            ModelType::Granite => granite::convert_path(path, precision, sink),
        }
    }
}

/// A loaded model of either kind.
pub enum Model {
    Parakeet(Box<Parakeet>),
    Granite(Box<Granite>),
}

impl Model {
    /// Loads the model at `path`, a pack or what [`Parakeet::load`] or [`Granite::load`] takes.
    pub fn load(path: &Path, options: &Options) -> Result<Model> {
        let is_pack = path.is_file() && path.extension().and_then(|e| e.to_str()) != Some("gguf");
        if is_pack {
            return Model::from_pack(&Pack::open(path)?, options);
        }
        Ok(match ModelType::detect(path)? {
            ModelType::Parakeet => Model::Parakeet(Box::new(Parakeet::load(path, options)?)),
            ModelType::Granite => Model::Granite(Box::new(Granite::load(path, options)?)),
        })
    }

    /// Builds the model a pack holds.
    pub fn from_pack(pack: &Pack, options: &Options) -> Result<Model> {
        Ok(match ModelType::of_pack(pack)? {
            ModelType::Parakeet => Model::Parakeet(Box::new(Parakeet::from_pack(pack, options)?)),
            ModelType::Granite => Model::Granite(Box::new(Granite::from_pack(pack, options)?)),
        })
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
            Model::Granite(m) => m.precision(),
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
