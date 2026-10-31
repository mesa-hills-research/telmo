//! A reader for GGUF files (versions 2 and 3), the format of NVIDIA's 8-bit Parakeet file.
//!
//! The header and tensor table are checked as a whole before any tensor is read. Every tensor
//! must have a supported type, at most four dimensions, a byte range that matches its type and
//! shape, inside the file and aligned as the file declares, and no two ranges may overlap.
//! Dimensions are listed innermost first, as GGUF stores them.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::quant::{Q8_BLOCK, bf16_to_f32, f16_to_f32};

const MAGIC: [u8; 4] = *b"GGUF";
/// Limits that keep a damaged header from asking for absurd allocations.
const MAX_STRING: u64 = 1 << 24;
const MAX_ARRAY: u64 = 1 << 24;
const MAX_ENTRIES: u64 = 1 << 20;

/// Tensor types the reader takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    F32,
    F16,
    Bf16,
    /// Blocks of 32 int8 values with one f16 scale, 34 bytes each.
    Q8_0,
}

impl TensorType {
    fn from_id(id: u32) -> Option<TensorType> {
        Some(match id {
            0 => TensorType::F32,
            1 => TensorType::F16,
            8 => TensorType::Q8_0,
            30 => TensorType::Bf16,
            _ => return None,
        })
    }

    /// Bytes of `elements` values, or `None` if they don't fill whole blocks.
    fn bytes(self, elements: u64, row: u64) -> Option<u64> {
        match self {
            TensorType::F32 => elements.checked_mul(4),
            TensorType::F16 | TensorType::Bf16 => elements.checked_mul(2),
            TensorType::Q8_0 => row
                .is_multiple_of(Q8_BLOCK as u64)
                .then(|| elements / Q8_BLOCK as u64 * 34),
        }
    }
}

/// A metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    UInt(u64),
    Float(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::UInt(v) => Some(v),
            Value::Int(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::Float(v) => Some(v),
            Value::UInt(v) => Some(v as f64),
            Value::Int(v) => Some(v as f64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

/// One tensor's entry in the table. `offset` counts bytes from the start of the data section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    /// Innermost first.
    pub dims: Vec<u64>,
    pub ty: TensorType,
    pub offset: u64,
    pub bytes: u64,
}

impl TensorInfo {
    /// The shape outermost first, as PyTorch writes it.
    pub fn shape(&self) -> Vec<usize> {
        self.dims.iter().rev().map(|&d| d as usize).collect()
    }

    pub fn elements(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }
}

/// A tensor's values, either floats or Q8_0 blocks with their scales widened to f32.
pub enum Data {
    F32(Vec<f32>),
    Q8_0 { d: Vec<f32>, q: Vec<i8> },
}

enum Source {
    File {
        file: BufReader<File>,
        path: PathBuf,
    },
    Bytes(Vec<u8>),
}

/// An open GGUF file whose header has been checked.
pub struct Gguf {
    pub version: u32,
    pub metadata: Vec<(String, Value)>,
    pub tensors: Vec<TensorInfo>,
    data_offset: u64,
    source: Source,
}

/// Reads the header from a stream, checking every length against the bytes left in the file.
struct Reader<'a> {
    input: &'a mut dyn Read,
    pos: u64,
    len: u64,
}

impl Reader<'_> {
    fn bad(&self, message: &str) -> Error {
        Error::Model(format!(
            "The GGUF file is damaged near byte {}. {message}",
            self.pos
        ))
    }

    fn take(&mut self, n: u64) -> Result<Vec<u8>> {
        if n > self.len - self.pos {
            return Err(self.bad("It ends early."));
        }
        let mut buf = vec![0u8; n as usize];
        self.input
            .read_exact(&mut buf)
            .map_err(|_| self.bad("It ends early."))?;
        self.pos += n;
        Ok(buf)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.take(N as u64)?.try_into().unwrap())
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()?;
        if len > MAX_STRING {
            return Err(self.bad("A string is implausibly long."));
        }
        String::from_utf8(self.take(len)?).map_err(|_| self.bad("A string isn't UTF-8."))
    }

    fn value(&mut self, ty: u32, depth: usize) -> Result<Value> {
        Ok(match ty {
            0 => Value::UInt(self.array::<1>()?[0] as u64),
            1 => Value::Int(self.array::<1>()?[0] as i8 as i64),
            2 => Value::UInt(u16::from_le_bytes(self.array()?) as u64),
            3 => Value::Int(i16::from_le_bytes(self.array()?) as i64),
            4 => Value::UInt(self.u32()? as u64),
            5 => Value::Int(i32::from_le_bytes(self.array()?) as i64),
            6 => Value::Float(f32::from_le_bytes(self.array()?) as f64),
            7 => match self.array::<1>()?[0] {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                _ => return Err(self.bad("A boolean isn't 0 or 1.")),
            },
            8 => Value::String(self.string()?),
            9 => {
                if depth > 2 {
                    return Err(self.bad("Arrays are nested too deeply."));
                }
                let inner = self.u32()?;
                let count = self.u64()?;
                // Every element takes at least one byte, which bounds the count.
                if count > MAX_ARRAY || count > self.len - self.pos {
                    return Err(self.bad("An array is implausibly long."));
                }
                let mut items = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    items.push(self.value(inner, depth + 1)?);
                }
                Value::Array(items)
            }
            10 => Value::UInt(self.u64()?),
            11 => Value::Int(i64::from_le_bytes(self.array()?)),
            12 => Value::Float(f64::from_le_bytes(self.array()?)),
            _ => return Err(self.bad("A metadata value has an unknown type.")),
        })
    }
}

impl Gguf {
    /// Opens a file and checks its header and tensor table. Tensor data stays on disk until it
    /// is read.
    pub fn open(path: &Path) -> Result<Gguf> {
        let file = File::open(path).map_err(|e| Error::io(path, e))?;
        let len = file.metadata().map_err(|e| Error::io(path, e))?.len();
        let mut file = BufReader::with_capacity(1 << 20, file);
        let (version, metadata, tensors, data_offset) = Gguf::parse(&mut file, len)?;
        Ok(Gguf {
            version,
            metadata,
            tensors,
            data_offset,
            source: Source::File {
                file,
                path: path.to_path_buf(),
            },
        })
    }

    /// Checks a whole file held in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Gguf> {
        let (version, metadata, tensors, data_offset) =
            Gguf::parse(&mut &bytes[..], bytes.len() as u64)?;
        Ok(Gguf {
            version,
            metadata,
            tensors,
            data_offset,
            source: Source::Bytes(bytes),
        })
    }

    #[allow(clippy::type_complexity)]
    fn parse(
        input: &mut dyn Read,
        len: u64,
    ) -> Result<(u32, Vec<(String, Value)>, Vec<TensorInfo>, u64)> {
        let mut r = Reader { input, pos: 0, len };
        if r.array::<4>()? != MAGIC {
            return Err(Error::Model("The file isn't a GGUF file.".into()));
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(Error::Model(format!(
                "GGUF version {version} isn't supported (versions 2 and 3 are)."
            )));
        }
        let tensor_count = r.u64()?;
        let kv_count = r.u64()?;
        if tensor_count > MAX_ENTRIES || kv_count > MAX_ENTRIES {
            return Err(r.bad("The header counts are implausibly large."));
        }
        let mut metadata = Vec::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            let key = r.string()?;
            let ty = r.u32()?;
            let value = r.value(ty, 0)?;
            if metadata.iter().any(|(k, _)| *k == key) {
                return Err(r.bad("A metadata key appears twice."));
            }
            metadata.push((key, value));
        }
        let alignment = match metadata.iter().find(|(k, _)| k == "general.alignment") {
            None => 32,
            Some((_, v)) => v
                .as_u64()
                .filter(|a| a.is_power_of_two() && *a <= 1 << 16)
                .ok_or_else(|| r.bad("general.alignment isn't a power of two."))?,
        };
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = r.string()?;
            let n_dims = r.u32()?;
            if !(1..=4).contains(&n_dims) {
                return Err(r.bad("A tensor has more than four dimensions or none."));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let ty_id = r.u32()?;
            let ty = TensorType::from_id(ty_id).ok_or_else(|| {
                Error::Model(format!(
                    "Tensor {name} has GGUF type {ty_id}, which isn't supported (F32, F16, BF16 and Q8_0 are)."
                ))
            })?;
            let offset = r.u64()?;
            let elements = dims
                .iter()
                .try_fold(1u64, |acc, &d| acc.checked_mul(d))
                .filter(|&e| e > 0 && e < 1 << 40)
                .ok_or_else(|| r.bad("A tensor's shape is empty or too large."))?;
            let bytes = ty
                .bytes(elements, dims[0])
                .ok_or_else(|| r.bad("A Q8_0 tensor's rows aren't whole blocks."))?;
            if !offset.is_multiple_of(alignment) {
                return Err(r.bad("A tensor isn't aligned."));
            }
            if tensors.iter().any(|t: &TensorInfo| t.name == name) {
                return Err(r.bad("Two tensors share a name."));
            }
            tensors.push(TensorInfo {
                name,
                dims,
                ty,
                offset,
                bytes,
            });
        }
        let data_offset = r.pos.next_multiple_of(alignment);
        if data_offset > len {
            return Err(Error::Model("The GGUF file ends before its data.".into()));
        }
        let data_len = len - data_offset;
        let mut ranges: Vec<(u64, u64, &str)> = tensors
            .iter()
            .map(|t| (t.offset, t.offset.saturating_add(t.bytes), t.name.as_str()))
            .collect();
        ranges.sort();
        for (i, &(start, end, name)) in ranges.iter().enumerate() {
            if end > data_len {
                return Err(Error::Model(format!(
                    "Tensor {name} runs past the end of the GGUF file."
                )));
            }
            if i > 0 && start < ranges[i - 1].1 {
                return Err(Error::Model(format!(
                    "Tensor {name} overlaps another in the GGUF file."
                )));
            }
        }
        Ok((version, metadata, tensors, data_offset))
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    fn read_raw(&mut self, info: &TensorInfo) -> Result<Vec<u8>> {
        let start = self.data_offset + info.offset;
        let mut buf = vec![0u8; info.bytes as usize];
        match &mut self.source {
            Source::Bytes(bytes) => {
                let len = buf.len();
                buf.copy_from_slice(&bytes[start as usize..start as usize + len])
            }
            Source::File { file, path } => {
                file.seek(SeekFrom::Start(start))
                    .map_err(|e| Error::io(path, e))?;
                file.read_exact(&mut buf).map_err(|e| Error::io(path, e))?;
            }
        }
        Ok(buf)
    }

    /// Reads a tensor, checking its shape (outermost first) when `shape` is given. Q8_0 tensors
    /// keep their blocks.
    pub fn read(&mut self, name: &str, shape: Option<&[usize]>) -> Result<(TensorInfo, Data)> {
        let info = self
            .info(name)
            .ok_or_else(|| Error::Model(format!("Tensor {name} is missing from the GGUF file.")))?
            .clone();
        if let Some(shape) = shape {
            // Dimensions of 1 at either end leave the same tensor, as a pointwise
            // convolution's `[n, k, 1]` and a linear layer's `[n, k]` are.
            let got: Vec<usize> = info.shape();
            let strip = |s: &[usize]| -> Vec<usize> {
                let first = s.iter().position(|&d| d != 1).unwrap_or(s.len());
                let last = s.iter().rposition(|&d| d != 1).map_or(first, |l| l + 1);
                s[first..last].to_vec()
            };
            if strip(&got) != strip(shape) {
                return Err(Error::Model(format!(
                    "Tensor {name} has shape {got:?} in the GGUF file, expected {shape:?}."
                )));
            }
        }
        let raw = self.read_raw(&info)?;
        let data = match info.ty {
            TensorType::F32 => Data::F32(
                raw.chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
            ),
            TensorType::F16 => Data::F32(
                raw.chunks_exact(2)
                    .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                    .collect(),
            ),
            TensorType::Bf16 => Data::F32(
                raw.chunks_exact(2)
                    .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                    .collect(),
            ),
            TensorType::Q8_0 => {
                let blocks = raw.len() / 34;
                let mut d = Vec::with_capacity(blocks);
                let mut q = Vec::with_capacity(blocks * Q8_BLOCK);
                for block in raw.chunks_exact(34) {
                    d.push(f16_to_f32(u16::from_le_bytes([block[0], block[1]])));
                    q.extend(block[2..].iter().map(|&b| b as i8));
                }
                Data::Q8_0 { d, q }
            }
        };
        Ok((info, data))
    }

    /// Reads a tensor as f32, dequantizing Q8_0 blocks.
    pub fn read_f32(&mut self, name: &str, shape: Option<&[usize]>) -> Result<Vec<f32>> {
        Ok(match self.read(name, shape)?.1 {
            Data::F32(v) => v,
            Data::Q8_0 { d, q } => q
                .iter()
                .enumerate()
                .map(|(i, &v)| v as f32 * d[i / Q8_BLOCK])
                .collect(),
        })
    }
}

/// Writes a GGUF v3 file, for tests.
#[doc(hidden)]
pub fn write_for_tests(
    metadata: &[(&str, Value)],
    tensors: &[(&str, &[u64], TensorType, &[u8])],
) -> Vec<u8> {
    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }
    fn value(out: &mut Vec<u8>, v: &Value) {
        match v {
            Value::UInt(x) => out.extend(x.to_le_bytes()),
            Value::Int(x) => out.extend(x.to_le_bytes()),
            Value::Float(x) => out.extend((*x as f32).to_le_bytes()),
            Value::Bool(b) => out.push(*b as u8),
            Value::String(s) => string(out, s),
            Value::Array(items) => {
                out.extend(type_id(&items[0]).to_le_bytes());
                out.extend((items.len() as u64).to_le_bytes());
                for item in items {
                    value(out, item);
                }
            }
        }
    }
    fn type_id(v: &Value) -> u32 {
        match v {
            Value::UInt(_) => 10,
            Value::Int(_) => 11,
            Value::Float(_) => 6,
            Value::Bool(_) => 7,
            Value::String(_) => 8,
            Value::Array(_) => 9,
        }
    }
    let mut out = MAGIC.to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend((metadata.len() as u64).to_le_bytes());
    for (k, v) in metadata {
        string(&mut out, k);
        out.extend(type_id(v).to_le_bytes());
        value(&mut out, v);
    }
    let mut offset = 0u64;
    for (name, dims, ty, data) in tensors {
        string(&mut out, name);
        out.extend((dims.len() as u32).to_le_bytes());
        for d in *dims {
            out.extend(d.to_le_bytes());
        }
        let id: u32 = match ty {
            TensorType::F32 => 0,
            TensorType::F16 => 1,
            TensorType::Q8_0 => 8,
            TensorType::Bf16 => 30,
        };
        out.extend(id.to_le_bytes());
        out.extend(offset.to_le_bytes());
        offset = (offset + data.len() as u64).next_multiple_of(32);
    }
    out.resize(out.len().next_multiple_of(32), 0);
    for (_, _, _, data) in tensors {
        out.extend(*data);
        out.resize(out.len().next_multiple_of(32), 0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::f32_to_f16;

    pub(crate) fn sample() -> Vec<u8> {
        let f32s: Vec<u8> = [1.0f32, -2.0, 0.5, 4.0, 8.0, 16.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut q8 = Vec::new();
        for row in 0..2 {
            q8.extend(f32_to_f16(0.25 * (row + 1) as f32).to_le_bytes());
            q8.extend((0..32).map(|i| (i as i8 - 16) as u8));
        }
        let f16s: Vec<u8> = [1.5f32, -0.25]
            .iter()
            .flat_map(|&v| f32_to_f16(v).to_le_bytes())
            .collect();
        write_for_tests(
            &[
                ("general.architecture", Value::String("asr".into())),
                ("asr.encoder.d_model", Value::UInt(1024)),
                (
                    "asr.tdt.durations",
                    Value::Array(vec![Value::Int(0), Value::Int(1)]),
                ),
            ],
            &[
                ("a", &[3, 2], TensorType::F32, &f32s),
                ("q", &[32, 2], TensorType::Q8_0, &q8),
                ("h", &[2], TensorType::F16, &f16s),
            ],
        )
    }

    #[test]
    fn a_small_file_reads_back() {
        let mut g = Gguf::from_bytes(sample()).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(
            g.get("asr.encoder.d_model").and_then(Value::as_u64),
            Some(1024)
        );
        assert_eq!(
            g.get("asr.tdt.durations")
                .and_then(Value::as_array)
                .map(|a| a.len()),
            Some(2)
        );
        assert_eq!(g.info("a").unwrap().shape(), vec![2, 3]);
        assert_eq!(
            g.read_f32("a", Some(&[2, 3])).unwrap(),
            vec![1.0, -2.0, 0.5, 4.0, 8.0, 16.0]
        );
        assert!(g.read_f32("a", Some(&[3, 2])).is_err());
        match g.read("q", Some(&[2, 32])).unwrap().1 {
            Data::Q8_0 { d, q } => {
                assert_eq!(d, vec![0.25, 0.5]);
                assert_eq!(q[0], -16);
                assert_eq!(q[63], 15);
            }
            Data::F32(_) => panic!("Q8_0 should keep its blocks"),
        }
        assert_eq!(g.read_f32("q", None).unwrap()[33], -15.0 * 0.5);
        assert_eq!(g.read_f32("h", Some(&[1, 2])).unwrap(), vec![1.5, -0.25]);
    }

    #[test]
    fn bad_headers_are_refused() {
        let good = sample();
        let mut wrong_magic = good.clone();
        wrong_magic[0] = b'X';
        assert!(Gguf::from_bytes(wrong_magic).is_err());
        let mut wrong_version = good.clone();
        wrong_version[4] = 9;
        assert!(Gguf::from_bytes(wrong_version).is_err());
        // The last tensor holds 4 bytes and 28 of padding follow it.
        assert!(Gguf::from_bytes(good[..good.len() - 28].to_vec()).is_ok());
        assert!(Gguf::from_bytes(good[..good.len() - 29].to_vec()).is_err());
        assert!(Gguf::from_bytes(good[..40].to_vec()).is_err());
    }
}
