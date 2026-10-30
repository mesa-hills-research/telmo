//! The engine's own weight file, a pack: a model converted once into the layout and precision
//! the kernels read, so that loading maps the file and builds views without copying.
//!
//! A pack holds named entries of three kinds: weight matrices in panels (see [`crate::quant`]),
//! f32 tensors, and raw bytes (the model's `config.json` and `tokenizer.json`). The file is
//!
//! ```text
//! 0      "TELMOPAK", then the format version as a little-endian u32, then 4 zero bytes
//! 16     the entries' data, each starting at a multiple of 64 bytes
//! I      the index, a JSON object listing the model, its metadata and every entry
//! end-24 the index's offset and length as little-endian u64s, then "TELMOIDX"
//! ```
//!
//! The index is checked as a whole before any view is made. Every entry needs a known kind, a
//! size that matches its shape and precision, a range inside the data section aligned to 64
//! bytes, and no two ranges may overlap. A pack is only read on little-endian CPUs.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::json::{self, Value};
use crate::kernels::PackedWeights;
use crate::quant::{Precision, Rows};
use crate::region::{ALIGN, Bytes, Region, Vector};

pub const MAGIC: [u8; 8] = *b"TELMOPAK";
const INDEX_MAGIC: [u8; 8] = *b"TELMOIDX";
/// The format version this crate writes and reads.
pub const VERSION: u32 = 1;
const PREFIX: usize = 16;
const TRAILER: usize = 24;
const MAX_INDEX: u64 = 64 << 20;

/// What an entry holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// A weight matrix of `n` outputs and `k` inputs in panels.
    Matrix {
        precision: Precision,
        n: usize,
        k: usize,
    },
    /// An f32 tensor.
    F32 { shape: Vec<usize> },
    /// Raw bytes.
    Bytes,
}

/// One entry of a pack.
#[derive(Clone)]
pub struct Entry {
    pub name: String,
    pub kind: Kind,
    values: Bytes,
    scales: Option<Bytes>,
}

impl Entry {
    /// Bytes of data, scales included.
    pub fn size(&self) -> usize {
        self.values.len() + self.scales.as_ref().map_or(0, |s| s.len())
    }
}

/// A pack, read from a file or built in memory.
pub struct Pack {
    model: String,
    metadata: Vec<(String, String)>,
    entries: Vec<Entry>,
    by_name: HashMap<String, usize>,
    mapped: bool,
}

impl Pack {
    /// Opens a pack file, mapping it read-only where the system allows.
    pub fn open(path: &Path) -> Result<Pack> {
        let mut file = File::open(path).map_err(|e| Error::io(path, e))?;
        let len = file.metadata().map_err(|e| Error::io(path, e))?.len();
        let len = usize::try_from(len).map_err(|_| bad("The file is too large."))?;
        if len < PREFIX + TRAILER {
            return Err(bad("The file is too short."));
        }
        let (region, mapped) = Region::of_file(&mut file, len).map_err(|e| Error::io(path, e))?;
        let mut pack = Pack::parse(region.share())?;
        pack.mapped = mapped;
        Ok(pack)
    }

    /// Reads a pack held in memory.
    pub fn from_bytes(bytes: &[u8]) -> Result<Pack> {
        let mut region = Region::zeroed(bytes.len());
        region.as_mut_slice().copy_from_slice(bytes);
        Pack::parse(region.share())
    }

    fn parse(file: Bytes) -> Result<Pack> {
        if cfg!(target_endian = "big") {
            return Err(Error::Model("Packs need a little-endian CPU.".into()));
        }
        let data = file.as_slice();
        let len = data.len();
        if len < PREFIX + TRAILER || data[..8] != MAGIC {
            return Err(bad("It isn't a pack."));
        }
        let version = u32::from_le_bytes(data[8..12].try_into().unwrap());
        if version != VERSION {
            return Err(Error::Model(format!(
                "The pack has format version {version}, and this version of the engine reads {VERSION}. Convert the model again."
            )));
        }
        let trailer = &data[len - TRAILER..];
        if trailer[16..] != INDEX_MAGIC {
            return Err(bad("Its index is missing, so the file may be cut short."));
        }
        let index_at = u64::from_le_bytes(trailer[..8].try_into().unwrap());
        let index_len = u64::from_le_bytes(trailer[8..16].try_into().unwrap());
        if index_len > MAX_INDEX
            || index_at < PREFIX as u64
            || index_at.checked_add(index_len) != Some((len - TRAILER) as u64)
        {
            return Err(bad("Its index is out of place."));
        }
        let index_at = index_at as usize;
        let root = json::parse(&data[index_at..len - TRAILER])?;
        let model = root
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("The index names no model."))?
            .to_string();
        let mut metadata = Vec::new();
        if let Some(members) = root.get("metadata").and_then(Value::as_object) {
            for (key, value) in members {
                let value = value
                    .as_str()
                    .ok_or_else(|| bad("A metadata value isn't a string."))?;
                metadata.push((key.clone(), value.to_string()));
            }
        }
        let list = root
            .get("entries")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("The index lists no entries."))?;

        let mut entries = Vec::with_capacity(list.len());
        let mut by_name = HashMap::with_capacity(list.len());
        let mut ranges = Vec::with_capacity(list.len());
        for item in list {
            let name = item
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| bad("An entry has no name."))?
                .to_string();
            let num = |key: &str| -> Result<usize> {
                item.get(key)
                    .and_then(Value::as_u64)
                    .and_then(|v| usize::try_from(v).ok())
                    .filter(|&v| v < 1 << 40)
                    .ok_or_else(|| bad(&format!("Entry {name} has no valid {key}.")))
            };
            let kind = match item.get("type").and_then(Value::as_str) {
                Some("matrix") => {
                    let precision = item
                        .get("precision")
                        .and_then(Value::as_str)
                        .and_then(Precision::from_name)
                        .ok_or_else(|| bad(&format!("Entry {name} has an unknown precision.")))?;
                    let (n, k) = (num("n")?, num("k")?);
                    if n == 0 || k == 0 || n > 1 << 24 || k > 1 << 24 || !precision.fits(k) {
                        return Err(bad(&format!("Entry {name} has an impossible shape.")));
                    }
                    Kind::Matrix { precision, n, k }
                }
                Some("f32") => {
                    let shape = item
                        .get("shape")
                        .and_then(Value::as_array)
                        .and_then(|dims| {
                            dims.iter()
                                .map(|d| d.as_u64().and_then(|d| usize::try_from(d).ok()))
                                .collect::<Option<Vec<usize>>>()
                        })
                        .filter(|s| s.len() <= 8)
                        .ok_or_else(|| bad(&format!("Entry {name} has no valid shape.")))?;
                    Kind::F32 { shape }
                }
                Some("bytes") => Kind::Bytes,
                _ => return Err(bad(&format!("Entry {name} has an unknown type."))),
            };
            let (offset, bytes) = (num("offset")?, num("bytes")?);
            let expected = match &kind {
                Kind::Matrix { precision, n, k } => Some(precision.panel_bytes(*n, *k)),
                Kind::F32 { shape } => shape.iter().try_fold(4usize, |acc, &d| acc.checked_mul(d)),
                Kind::Bytes => Some(bytes),
            };
            if expected != Some(bytes) {
                return Err(bad(&format!("Entry {name} has the wrong size.")));
            }
            let mut check_range = |offset: usize, bytes: usize| -> Result<Bytes> {
                if offset < PREFIX
                    || !offset.is_multiple_of(ALIGN)
                    || offset.checked_add(bytes).is_none_or(|end| end > index_at)
                {
                    return Err(bad(&format!("Entry {name} is out of place.")));
                }
                ranges.push((offset, offset + bytes));
                Ok(file.slice(offset, bytes))
            };
            let values = check_range(offset, bytes)?;
            let scales = match &kind {
                Kind::Matrix { precision, n, k } if precision.scale_count(*n, *k) > 0 => {
                    let scale_bytes = num("scales_bytes")?;
                    if scale_bytes != precision.scale_count(*n, *k) * 4 {
                        return Err(bad(&format!(
                            "Entry {name} has the wrong number of scales."
                        )));
                    }
                    Some(check_range(num("scales_offset")?, scale_bytes)?)
                }
                _ => None,
            };
            if by_name.insert(name.clone(), entries.len()).is_some() {
                return Err(bad(&format!("Two entries are named {name}.")));
            }
            entries.push(Entry {
                name,
                kind,
                values,
                scales,
            });
        }
        ranges.sort();
        if ranges.windows(2).any(|w| w[1].0 < w[0].1) {
            return Err(bad("Two entries overlap."));
        }
        Ok(Pack {
            model,
            metadata,
            entries,
            by_name,
            mapped: false,
        })
    }

    /// The model the pack holds, such as `parakeet-tdt` or `granite-speech-ctc`.
    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }

    pub fn meta(&self, key: &str) -> Option<&str> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Whether the weights are a mapping of the file (or, if not, a copy in memory).
    pub fn is_mapped(&self) -> bool {
        self.mapped
    }

    fn entry(&self, name: &str) -> Result<&Entry> {
        self.by_name
            .get(name)
            .map(|&i| &self.entries[i])
            .ok_or_else(|| bad(&format!("Entry {name} is missing.")))
    }

    /// A weight matrix of `n` outputs and `k` inputs.
    pub fn matrix(&self, name: &str, n: usize, k: usize) -> Result<PackedWeights> {
        let entry = self.entry(name)?;
        let Kind::Matrix {
            precision,
            n: en,
            k: ek,
        } = entry.kind
        else {
            return Err(bad(&format!("Entry {name} isn't a matrix.")));
        };
        if (en, ek) != (n, k) {
            return Err(bad(&format!(
                "Entry {name} is {en} x {ek}, and the model needs {n} x {k}."
            )));
        }
        PackedWeights::from_parts(
            n,
            k,
            precision,
            entry.values.clone(),
            entry.scales.clone().map(Vector::new),
        )
        .ok_or_else(|| bad(&format!("Entry {name} doesn't fit its shape.")))
    }

    /// An f32 tensor of the given shape.
    pub fn vector(&self, name: &str, shape: &[usize]) -> Result<Vector> {
        let entry = self.entry(name)?;
        match &entry.kind {
            Kind::F32 { shape: s } if s == shape => Ok(Vector::new(entry.values.clone())),
            Kind::F32 { shape: s } => Err(bad(&format!(
                "Entry {name} has shape {s:?}, and the model needs {shape:?}."
            ))),
            _ => Err(bad(&format!("Entry {name} isn't an f32 tensor."))),
        }
    }

    /// A bytes entry.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let entry = self.entry(name)?;
        match entry.kind {
            Kind::Bytes => Ok(entry.values.as_slice()),
            _ => Err(bad(&format!("Entry {name} isn't a bytes entry."))),
        }
    }

    /// The precision of the weight matrices, if they share one.
    pub fn precision(&self) -> Option<Precision> {
        let mut found = None;
        for entry in &self.entries {
            if let Kind::Matrix { precision, .. } = entry.kind {
                match found {
                    None => found = Some(precision),
                    Some(p) if p != precision => return None,
                    _ => {}
                }
            }
        }
        found
    }

    /// Bytes of all entries' data.
    pub fn data_bytes(&self) -> usize {
        self.entries.iter().map(Entry::size).sum()
    }
}

/// Metadata describing the conversion of the model at `path` (a checkpoint directory or a
/// GGUF file) into a pack.
pub fn metadata_for(path: &Path, precision: Precision) -> Vec<(String, String)> {
    let source = path
        .file_name()
        .map_or(String::new(), |n| n.to_string_lossy().into_owned());
    let bytes = if path.is_dir() {
        std::fs::metadata(path.join("model.safetensors")).map(|m| m.len())
    } else {
        std::fs::metadata(path).map(|m| m.len())
    };
    vec![
        ("precision".into(), precision.name().into()),
        ("source".into(), source),
        (
            "source_bytes".into(),
            bytes.map_or(String::new(), |b| b.to_string()),
        ),
        (
            "engine".into(),
            format!("telmo {}", env!("CARGO_PKG_VERSION")),
        ),
    ]
}

fn bad(message: &str) -> Error {
    Error::Model(format!("The pack file isn't usable. {message}"))
}

/// Where a model's converter puts its entries: a pack in memory, or a file.
pub trait Sink {
    /// A weight matrix, laid out in panels in `precision`.
    fn matrix(
        &mut self,
        name: &str,
        rows: &Rows,
        n: usize,
        k: usize,
        precision: Precision,
    ) -> Result<()>;
    /// An f32 tensor.
    fn f32(&mut self, name: &str, shape: &[usize], data: &[f32]) -> Result<()>;
    /// Raw bytes.
    fn bytes(&mut self, name: &str, data: &[u8]) -> Result<()>;
}

fn encode_matrix(rows: &Rows, n: usize, k: usize, precision: Precision) -> (Bytes, Option<Bytes>) {
    let w = PackedWeights::encode(rows, n, k, precision);
    (w.values().clone(), w.scales().map(|s| s.bytes().clone()))
}

fn f32_bytes(shape: &[usize], data: &[f32]) -> Result<Bytes> {
    if shape.iter().product::<usize>() != data.len() {
        return Err(Error::Model("A tensor doesn't match its shape.".into()));
    }
    Ok(Vector::from_slice(data).bytes().clone())
}

/// Builds a pack in memory, each entry in its own allocation.
pub struct MemorySink {
    model: String,
    metadata: Vec<(String, String)>,
    entries: Vec<Entry>,
}

impl MemorySink {
    pub fn new(model: &str, metadata: Vec<(String, String)>) -> MemorySink {
        MemorySink {
            model: model.to_string(),
            metadata,
            entries: Vec::new(),
        }
    }

    pub fn finish(self) -> Result<Pack> {
        let mut by_name = HashMap::new();
        for (i, e) in self.entries.iter().enumerate() {
            if by_name.insert(e.name.clone(), i).is_some() {
                return Err(bad(&format!("Two entries are named {}.", e.name)));
            }
        }
        Ok(Pack {
            model: self.model,
            metadata: self.metadata,
            entries: self.entries,
            by_name,
            mapped: false,
        })
    }
}

impl Sink for MemorySink {
    fn matrix(
        &mut self,
        name: &str,
        rows: &Rows,
        n: usize,
        k: usize,
        precision: Precision,
    ) -> Result<()> {
        let (values, scales) = encode_matrix(rows, n, k, precision);
        self.entries.push(Entry {
            name: name.to_string(),
            kind: Kind::Matrix { precision, n, k },
            values,
            scales,
        });
        Ok(())
    }

    fn f32(&mut self, name: &str, shape: &[usize], data: &[f32]) -> Result<()> {
        self.entries.push(Entry {
            name: name.to_string(),
            kind: Kind::F32 {
                shape: shape.to_vec(),
            },
            values: f32_bytes(shape, data)?,
            scales: None,
        });
        Ok(())
    }

    fn bytes(&mut self, name: &str, data: &[u8]) -> Result<()> {
        let mut region = Region::zeroed(data.len());
        region.as_mut_slice().copy_from_slice(data);
        self.entries.push(Entry {
            name: name.to_string(),
            kind: Kind::Bytes,
            values: region.share(),
            scales: None,
        });
        Ok(())
    }
}

/// Writes a pack file entry by entry, holding one entry in memory at a time. The file is
/// written under a temporary name and renamed when complete, so a pack is never half-written.
pub struct FileSink {
    out: BufWriter<File>,
    path: PathBuf,
    temp: PathBuf,
    pos: u64,
    model: String,
    metadata: Vec<(String, String)>,
    index: Vec<Value>,
}

impl FileSink {
    pub fn create(path: &Path, model: &str, metadata: Vec<(String, String)>) -> Result<FileSink> {
        let mut name = path.file_name().unwrap_or_default().to_os_string();
        name.push(".partial");
        let temp = path.with_file_name(name);
        let file = File::create(&temp).map_err(|e| Error::io(&temp, e))?;
        let mut sink = FileSink {
            out: BufWriter::with_capacity(4 << 20, file),
            path: path.to_path_buf(),
            temp,
            pos: 0,
            model: model.to_string(),
            metadata,
            index: Vec::new(),
        };
        let mut prefix = MAGIC.to_vec();
        prefix.extend(VERSION.to_le_bytes());
        prefix.extend([0u8; 4]);
        sink.write(&prefix)?;
        Ok(sink)
    }

    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        self.out
            .write_all(bytes)
            .map_err(|e| Error::io(&self.temp, e))?;
        self.pos += bytes.len() as u64;
        Ok(())
    }

    /// Writes padding up to the next multiple of 64, then `bytes`, and returns their offset.
    fn put(&mut self, bytes: &[u8]) -> Result<u64> {
        let pad = self.pos.next_multiple_of(ALIGN as u64) - self.pos;
        self.write(&vec![0u8; pad as usize])?;
        let at = self.pos;
        self.write(bytes)?;
        Ok(at)
    }

    fn record(&mut self, name: &str, kind: &[(&str, Value)], offset: u64, bytes: usize) {
        let mut members = vec![("name".to_string(), Value::String(name.to_string()))];
        members.extend(kind.iter().map(|(k, v)| (k.to_string(), v.clone())));
        members.push(("offset".into(), Value::number(offset)));
        members.push(("bytes".into(), Value::number(bytes)));
        self.index.push(Value::Object(members));
    }

    /// Writes the index and renames the file into place.
    pub fn finish(mut self) -> Result<()> {
        let metadata = self
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect();
        let root = Value::Object(vec![
            ("model".into(), Value::String(self.model.clone())),
            ("metadata".into(), Value::Object(metadata)),
            (
                "entries".into(),
                Value::Array(std::mem::take(&mut self.index)),
            ),
        ]);
        let index = root.to_json();
        let at = self.pos;
        self.write(index.as_bytes())?;
        let mut trailer = at.to_le_bytes().to_vec();
        trailer.extend((index.len() as u64).to_le_bytes());
        trailer.extend(INDEX_MAGIC);
        self.write(&trailer)?;
        self.out.flush().map_err(|e| Error::io(&self.temp, e))?;
        self.out
            .get_ref()
            .sync_all()
            .map_err(|e| Error::io(&self.temp, e))?;
        std::fs::rename(&self.temp, &self.path).map_err(|e| Error::io(&self.path, e))
    }
}

impl Sink for FileSink {
    fn matrix(
        &mut self,
        name: &str,
        rows: &Rows,
        n: usize,
        k: usize,
        precision: Precision,
    ) -> Result<()> {
        let (values, scales) = encode_matrix(rows, n, k, precision);
        let at = self.put(values.as_slice())?;
        let mut kind = vec![
            ("type", Value::String("matrix".into())),
            ("precision", Value::String(precision.name().into())),
            ("n", Value::number(n)),
            ("k", Value::number(k)),
        ];
        if let Some(scales) = scales {
            let scales_at = self.put(scales.as_slice())?;
            kind.push(("scales_offset", Value::number(scales_at)));
            kind.push(("scales_bytes", Value::number(scales.len())));
        }
        self.record(name, &kind, at, values.len());
        Ok(())
    }

    fn f32(&mut self, name: &str, shape: &[usize], data: &[f32]) -> Result<()> {
        let bytes = f32_bytes(shape, data)?;
        let at = self.put(bytes.as_slice())?;
        let shape = Value::Array(shape.iter().map(|&d| Value::number(d)).collect());
        self.record(
            name,
            &[("type", Value::String("f32".into())), ("shape", shape)],
            at,
            bytes.len(),
        );
        Ok(())
    }

    fn bytes(&mut self, name: &str, data: &[u8]) -> Result<()> {
        let at = self.put(data)?;
        self.record(
            name,
            &[("type", Value::String("bytes".into()))],
            at,
            data.len(),
        );
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A small pack with one entry of each kind.
    pub(crate) fn sample(sink: &mut dyn Sink) {
        let w: Vec<f32> = (0..40 * 64).map(|i| (i % 17) as f32 * 0.1 - 0.8).collect();
        sink.matrix("w8", &Rows::F32(&w), 40, 64, Precision::Int8)
            .unwrap();
        sink.matrix("wq", &Rows::F32(&w), 40, 64, Precision::Q8_0)
            .unwrap();
        sink.matrix("wh", &Rows::F32(&w), 40, 64, Precision::F16)
            .unwrap();
        sink.f32("bias", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        sink.bytes("config.json", br#"{"a":1}"#).unwrap();
    }

    pub(crate) fn sample_file() -> Vec<u8> {
        let dir = std::env::temp_dir().join(format!("telmo-pack-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sample.pack");
        let mut sink =
            FileSink::create(&path, "test", vec![("source".into(), "x".into())]).unwrap();
        sample(&mut sink);
        sink.finish().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        bytes
    }

    #[test]
    fn a_file_pack_reads_back_like_a_memory_pack() {
        let mut memory = MemorySink::new("test", vec![]);
        sample(&mut memory);
        let memory = memory.finish().unwrap();
        let dir = std::env::temp_dir().join(format!("telmo-pack-rt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.pack");
        let mut sink =
            FileSink::create(&path, "test", vec![("source".into(), "x".into())]).unwrap();
        sample(&mut sink);
        sink.finish().unwrap();
        let file = Pack::open(&path).unwrap();
        assert_eq!(file.model(), "test");
        assert_eq!(file.meta("source"), Some("x"));
        assert_eq!(
            file.is_mapped(),
            cfg!(any(all(unix, target_pointer_width = "64"), windows))
        );
        for name in ["w8", "wq", "wh"] {
            let (a, b) = (
                memory.matrix(name, 40, 64).unwrap(),
                file.matrix(name, 40, 64).unwrap(),
            );
            assert_eq!(a.precision(), b.precision());
            assert_eq!(a.to_rows(), b.to_rows());
            assert_eq!(b.values().as_ptr() as usize % ALIGN, 0);
        }
        assert!(file.matrix("w8", 64, 40).is_err());
        assert_eq!(
            &file.vector("bias", &[2, 3]).unwrap()[..],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert!(file.vector("bias", &[6]).is_err());
        assert_eq!(file.bytes("config.json").unwrap(), br#"{"a":1}"#);
        assert!(file.bytes("missing").is_err());
        drop(file);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn damaged_packs_are_refused() {
        let good = sample_file();
        assert!(Pack::from_bytes(&good).is_ok());
        assert!(Pack::from_bytes(&good[..good.len() - 1]).is_err());
        let mut wrong_version = good.clone();
        wrong_version[8] = 2;
        assert!(Pack::from_bytes(&wrong_version).is_err());
        let mut no_magic = good.clone();
        no_magic[0] = 0;
        assert!(Pack::from_bytes(&no_magic).is_err());
    }
}
