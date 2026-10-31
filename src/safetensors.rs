//! A reader for safetensors files.
//!
//! The header is checked as a whole before any tensor is read. Every tensor must have a known
//! dtype, a shape whose size matches its byte range, a range inside the data section and aligned
//! to its element size, and no two ranges may overlap.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::json;

/// The longest header the reader accepts. Real headers are well under 1 MB.
const MAX_HEADER: u64 = 100 << 20;

/// Element types of the safetensors format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bool,
    U8,
    I8,
    U16,
    I16,
    F16,
    BF16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
}

impl Dtype {
    fn from_name(name: &str) -> Option<Dtype> {
        Some(match name {
            "BOOL" => Dtype::Bool,
            "U8" => Dtype::U8,
            "I8" => Dtype::I8,
            "U16" => Dtype::U16,
            "I16" => Dtype::I16,
            "F16" => Dtype::F16,
            "BF16" => Dtype::BF16,
            "U32" => Dtype::U32,
            "I32" => Dtype::I32,
            "F32" => Dtype::F32,
            "U64" => Dtype::U64,
            "I64" => Dtype::I64,
            "F64" => Dtype::F64,
            _ => return None,
        })
    }

    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            Dtype::Bool | Dtype::U8 | Dtype::I8 => 1,
            Dtype::U16 | Dtype::I16 | Dtype::F16 | Dtype::BF16 => 2,
            Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
            Dtype::U64 | Dtype::I64 | Dtype::F64 => 8,
        }
    }
}

/// One tensor's entry in the header. `start` and `end` count bytes from the start of the data section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub start: u64,
    pub end: u64,
}

impl TensorInfo {
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A checked header, holding the tensors in file order and the free-form metadata.
#[derive(Debug, Clone)]
pub struct Header {
    pub tensors: Vec<TensorInfo>,
    pub metadata: Vec<(String, String)>,
    /// Where the data section starts in the file (8 + the header's length).
    pub data_offset: u64,
}

/// Parses and checks a header.
///
/// `header` is the JSON text after the 8-byte length, and `file_len` is the whole file's size.
pub fn parse_header(header: &[u8], file_len: u64) -> Result<Header> {
    let bad = |message: String| Error::SafeTensors(message);
    let data_offset = 8 + header.len() as u64;
    if data_offset > file_len {
        return Err(bad("The header runs past the end of the file.".into()));
    }
    let data_len = file_len - data_offset;
    let root = json::parse(header)?;
    let members = root
        .as_object()
        .ok_or_else(|| bad("The header isn't a JSON object.".into()))?;

    let mut tensors = Vec::with_capacity(members.len());
    let mut metadata = Vec::new();
    for (name, entry) in members {
        if name == "__metadata__" {
            let fields = entry
                .as_object()
                .ok_or_else(|| bad("__metadata__ isn't an object.".into()))?;
            for (key, value) in fields {
                let value = value
                    .as_str()
                    .ok_or_else(|| bad(format!("Metadata {key} isn't a string.")))?;
                metadata.push((key.clone(), value.to_string()));
            }
            continue;
        }
        let field = |key: &str| {
            entry
                .get(key)
                .ok_or_else(|| bad(format!("Tensor {name} has no {key}.")))
        };
        let dtype_name = field("dtype")?
            .as_str()
            .ok_or_else(|| bad(format!("Tensor {name}'s dtype isn't a string.")))?;
        let dtype = Dtype::from_name(dtype_name)
            .ok_or_else(|| bad(format!("Tensor {name} has unknown dtype {dtype_name}.")))?;
        let shape = field("shape")?
            .as_array()
            .ok_or_else(|| bad(format!("Tensor {name}'s shape isn't an array.")))?
            .iter()
            .map(|d| d.as_u64().and_then(|d| usize::try_from(d).ok()))
            .collect::<Option<Vec<usize>>>()
            .ok_or_else(|| {
                bad(format!(
                    "Tensor {name}'s shape holds something other than sizes."
                ))
            })?;
        let offsets = field("data_offsets")?
            .as_array()
            .filter(|o| o.len() == 2)
            .ok_or_else(|| bad(format!("Tensor {name}'s data_offsets isn't a pair.")))?;
        let (Some(start), Some(end)) = (offsets[0].as_u64(), offsets[1].as_u64()) else {
            return Err(bad(format!(
                "Tensor {name}'s data_offsets aren't byte offsets."
            )));
        };
        if start > end || end > data_len {
            return Err(bad(format!(
                "Tensor {name}'s bytes {start}..{end} lie outside the data section of {data_len} bytes."
            )));
        }
        let bytes = shape
            .iter()
            .try_fold(dtype.size() as u64, |acc, &d| acc.checked_mul(d as u64))
            .ok_or_else(|| bad(format!("Tensor {name}'s shape is too large.")))?;
        if bytes != end - start {
            return Err(bad(format!(
                "Tensor {name} has shape {shape:?} of {dtype_name} ({bytes} bytes) but {} bytes of data.",
                end - start
            )));
        }
        if !(data_offset + start).is_multiple_of(dtype.size() as u64) {
            return Err(bad(format!(
                "Tensor {name} isn't aligned to its {}-byte elements in the file.",
                dtype.size()
            )));
        }
        tensors.push(TensorInfo {
            name: name.clone(),
            dtype,
            shape,
            start,
            end,
        });
    }

    let mut by_start: Vec<&TensorInfo> = tensors.iter().filter(|t| t.start < t.end).collect();
    by_start.sort_by_key(|t| t.start);
    for pair in by_start.windows(2) {
        if pair[1].start < pair[0].end {
            return Err(bad(format!(
                "Tensors {} and {} share bytes.",
                pair[0].name, pair[1].name
            )));
        }
    }
    Ok(Header {
        tensors,
        metadata,
        data_offset,
    })
}

/// Splits a file prefix into the header length, checking it against the file's size.
fn header_len(prefix: [u8; 8], file_len: u64) -> Result<u64> {
    let len = u64::from_le_bytes(prefix);
    if len > MAX_HEADER || len > file_len.saturating_sub(8) {
        return Err(Error::SafeTensors(format!(
            "The header length {len} doesn't fit a {file_len}-byte file."
        )));
    }
    Ok(len)
}

enum Source {
    File { file: File, path: PathBuf },
    Bytes(Vec<u8>),
}

/// An open safetensors file, or one held in memory. Tensors are read on request.
pub struct SafeTensors {
    header: Header,
    source: Source,
}

impl SafeTensors {
    /// Opens a file and checks its header. Tensor data stays on disk until it is read.
    pub fn open(path: &Path) -> Result<SafeTensors> {
        let mut file = File::open(path).map_err(|e| Error::io(path, e))?;
        let file_len = file.metadata().map_err(|e| Error::io(path, e))?.len();
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)
            .map_err(|e| Error::io(path, e))?;
        let len = header_len(prefix, file_len)?;
        let mut header = vec![0u8; len as usize];
        file.read_exact(&mut header)
            .map_err(|e| Error::io(path, e))?;
        let header = parse_header(&header, file_len)?;
        Ok(SafeTensors {
            header,
            source: Source::File {
                file,
                path: path.to_path_buf(),
            },
        })
    }

    /// Checks a whole file held in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<SafeTensors> {
        let prefix: [u8; 8] = bytes
            .get(..8)
            .and_then(|p| p.try_into().ok())
            .ok_or_else(|| Error::SafeTensors("The file is shorter than 8 bytes.".into()))?;
        let len = header_len(prefix, bytes.len() as u64)? as usize;
        let header = parse_header(&bytes[8..8 + len], bytes.len() as u64)?;
        Ok(SafeTensors {
            header,
            source: Source::Bytes(bytes),
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.header.tensors
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.header.tensors.iter().find(|t| t.name == name)
    }

    fn require(&self, name: &str) -> Result<&TensorInfo> {
        self.info(name)
            .ok_or_else(|| Error::SafeTensors(format!("Tensor {name} is missing.")))
    }

    /// Reads a tensor's raw little-endian bytes.
    pub fn read_bytes(&mut self, name: &str) -> Result<(TensorInfo, Vec<u8>)> {
        let info = self.require(name)?.clone();
        let mut bytes = vec![0u8; (info.end - info.start) as usize];
        self.read_into(&info, &mut bytes)?;
        Ok((info, bytes))
    }

    fn read_into(&mut self, info: &TensorInfo, dst: &mut [u8]) -> Result<()> {
        let start = self.header.data_offset + info.start;
        debug_assert_eq!(dst.len() as u64, info.end - info.start);
        match &mut self.source {
            Source::Bytes(bytes) => {
                dst.copy_from_slice(&bytes[start as usize..start as usize + dst.len()])
            }
            Source::File { file, path } => {
                file.seek(SeekFrom::Start(start))
                    .map_err(|e| Error::io(path, e))?;
                file.read_exact(dst).map_err(|e| Error::io(path, e))?;
            }
        }
        Ok(())
    }

    /// Reads an F32 tensor, checking its shape when `shape` is given.
    pub fn read_f32(&mut self, name: &str, shape: Option<&[usize]>) -> Result<Vec<f32>> {
        let info = self.require(name)?.clone();
        check(&info, Dtype::F32, shape)?;
        let mut out = vec![0f32; info.elements()];
        // SAFETY: the bytes of an f32 slice may hold any value, and u8 needs no alignment.
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4) };
        self.read_into(&info, bytes)?;
        if cfg!(target_endian = "big") {
            for v in &mut out {
                *v = f32::from_bits(v.to_bits().swap_bytes());
            }
        }
        Ok(out)
    }

    /// Reads an I64 tensor, checking its shape when `shape` is given.
    pub fn read_i64(&mut self, name: &str, shape: Option<&[usize]>) -> Result<Vec<i64>> {
        let (info, bytes) = self.read_bytes(name)?;
        check(&info, Dtype::I64, shape)?;
        Ok(bytes
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
}

fn check(info: &TensorInfo, dtype: Dtype, shape: Option<&[usize]>) -> Result<()> {
    if info.dtype != dtype {
        return Err(Error::SafeTensors(format!(
            "Tensor {} is {:?}, expected {dtype:?}.",
            info.name, info.dtype
        )));
    }
    if let Some(shape) = shape
        && info.shape != shape
    {
        return Err(Error::SafeTensors(format!(
            "Tensor {} has shape {:?}, expected {shape:?}.",
            info.name, info.shape
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a file from a header text and a data section.
    fn file(header: &str, data: &[u8]) -> Vec<u8> {
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    fn refuse(header: &str, data_len: usize, why: &str) {
        let err = SafeTensors::from_bytes(file(header, &vec![0u8; data_len])).err();
        let message = err.map(|e| e.to_string()).unwrap_or_default();
        assert!(
            message.contains(why),
            "{header}: got {message:?}, expected it to mention {why:?}"
        );
    }

    #[test]
    fn a_valid_file_reads_back() {
        // A 6-byte-long header plus padding keeps the data 8-byte aligned.
        let header = r#"{"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"n":{"dtype":"I64","shape":[1],"data_offsets":[16,24]}}"#;
        let pad = (8 - (8 + header.len()) % 8) % 8;
        let header = format!("{header}{}", " ".repeat(pad));
        let mut data = Vec::new();
        for v in [1.0f32, -2.0, 0.5, 3.25] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        data.extend_from_slice(&42i64.to_le_bytes());
        let mut st = SafeTensors::from_bytes(file(&header, &data)).unwrap();
        assert_eq!(
            st.header().metadata,
            vec![("format".to_string(), "pt".to_string())]
        );
        assert_eq!(
            st.read_f32("w", Some(&[2, 2])).unwrap(),
            vec![1.0, -2.0, 0.5, 3.25]
        );
        assert_eq!(st.read_i64("n", None).unwrap(), vec![42]);
        assert!(st.read_f32("w", Some(&[4])).is_err());
        assert!(st.read_f32("n", None).is_err());
        assert!(st.read_f32("missing", None).is_err());
    }

    #[test]
    fn malformed_headers_are_refused() {
        // Header lengths here are 8-aligned or the tensors are U8, so only the named fault applies.
        refuse(
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
            4,
            "outside the data section",
        );
        refuse(
            r#"{"a":{"dtype":"U8","shape":[3],"data_offsets":[0,4]}}"#,
            4,
            "4 bytes of data",
        );
        refuse(
            r#"{"a":{"dtype":"Q4","shape":[3],"data_offsets":[0,3]}}"#,
            4,
            "unknown dtype",
        );
        refuse(
            r#"{"a":{"dtype":"U8","shape":[-1],"data_offsets":[0,3]}}"#,
            4,
            "other than sizes",
        );
        refuse(
            r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[2,0]}}"#,
            4,
            "outside",
        );
        refuse(r#"{"a":{"dtype":"U8","shape":[2]}}"#, 4, "no data_offsets");
        refuse(
            r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[2],"data_offsets":[2,4]}}"#,
            4,
            "share bytes",
        );
        refuse(
            r#"{"a":{"dtype":"U8","shape":[4294967296,4294967296,4294967296],"data_offsets":[0,4]}}"#,
            4,
            "too large",
        );
        refuse(r#"[1,2]"#, 0, "isn't a JSON object");
        refuse(r#"{"__metadata__":{"a":1}}"#, 0, "isn't a string");
        // This header puts the data at an offset that isn't 8-aligned for an I64.
        let header = r#"{"a":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}}"#;
        assert_ne!((8 + header.len()) % 8, 0);
        refuse(header, 8, "aligned");
    }

    #[test]
    fn a_header_length_past_the_end_is_refused() {
        let mut bytes = (1000u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        assert!(SafeTensors::from_bytes(bytes).is_err());
        assert!(SafeTensors::from_bytes(vec![1, 2, 3]).is_err());
        let mut huge = u64::MAX.to_le_bytes().to_vec();
        huge.extend_from_slice(b"{}");
        assert!(SafeTensors::from_bytes(huge).is_err());
    }

    #[test]
    fn empty_tensors_may_share_an_offset() {
        let header = r#"{"a":{"dtype":"U8","shape":[0],"data_offsets":[0,0]},"b":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;
        let st = SafeTensors::from_bytes(file(header, &[7, 8])).unwrap();
        assert_eq!(st.tensors().len(), 2);
    }
}
