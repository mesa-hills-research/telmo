//! Weights as downloaded, read by their PyTorch names for conversion into a pack, from a
//! safetensors file in any float type or a GGUF file whose 8-bit blocks are kept as they are.

use crate::error::{Error, Result};
use crate::gguf::{self, Gguf};
use crate::quant::{Q8_BLOCK, Rows, bf16_to_f32, f16_to_f32};
use crate::safetensors::{Dtype, SafeTensors};

/// A row-major `[n, k]` weight matrix as a source stores it.
pub enum Matrix {
    F32(Vec<f32>),
    /// Q8_0 blocks. `d` holds `n * k / 32` scales, `q` the `n * k` values.
    Q8_0 {
        d: Vec<f32>,
        q: Vec<i8>,
    },
}

impl Matrix {
    pub fn rows(&self) -> Rows<'_> {
        match self {
            Matrix::F32(w) => Rows::F32(w),
            Matrix::Q8_0 { d, q } => Rows::Q8_0 { d, q },
        }
    }

    fn into_f32(self) -> Vec<f32> {
        match self {
            Matrix::F32(w) => w,
            Matrix::Q8_0 { d, q } => q
                .iter()
                .enumerate()
                .map(|(i, &v)| v as f32 * d[i / Q8_BLOCK])
                .collect(),
        }
    }

    /// Matrices with the same inputs stacked, the outputs of the first coming first. Q8_0
    /// blocks stay blocks when every part has them.
    pub fn stack(parts: Vec<Matrix>) -> Matrix {
        if parts.iter().all(|p| matches!(p, Matrix::Q8_0 { .. })) {
            let (mut d, mut q) = (Vec::new(), Vec::new());
            for part in parts {
                if let Matrix::Q8_0 { d: pd, q: pq } = part {
                    d.extend(pd);
                    q.extend(pq);
                }
            }
            return Matrix::Q8_0 { d, q };
        }
        Matrix::F32(parts.into_iter().flat_map(Matrix::into_f32).collect())
    }

    /// `[a | b]` for each of `n` rows, with the inputs of `b` after those of `a`. Q8_0 blocks stay
    /// blocks when both have them and `a`'s inputs fill whole blocks.
    pub fn concat_inputs(a: Matrix, ka: usize, b: Matrix, kb: usize, n: usize) -> Matrix {
        match (a, b) {
            (Matrix::Q8_0 { d: da, q: qa }, Matrix::Q8_0 { d: db, q: qb })
                if ka.is_multiple_of(Q8_BLOCK) && kb.is_multiple_of(Q8_BLOCK) =>
            {
                let (ba, bb) = (ka / Q8_BLOCK, kb / Q8_BLOCK);
                let (mut d, mut q) = (Vec::new(), Vec::new());
                for o in 0..n {
                    d.extend_from_slice(&da[o * ba..(o + 1) * ba]);
                    d.extend_from_slice(&db[o * bb..(o + 1) * bb]);
                    q.extend_from_slice(&qa[o * ka..(o + 1) * ka]);
                    q.extend_from_slice(&qb[o * kb..(o + 1) * kb]);
                }
                Matrix::Q8_0 { d, q }
            }
            (a, b) => {
                let (a, b) = (a.into_f32(), b.into_f32());
                let mut w = Vec::with_capacity(n * (ka + kb));
                for o in 0..n {
                    w.extend_from_slice(&a[o * ka..(o + 1) * ka]);
                    w.extend_from_slice(&b[o * kb..(o + 1) * kb]);
                }
                Matrix::F32(w)
            }
        }
    }
}

/// Where a converter reads weights from, by their names in the Hugging Face checkpoint.
pub trait Source {
    /// A tensor in f32, checked against `shape`.
    fn tensor(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>>;
    /// A matrix whose shape is `[n, k]` followed by any 1s.
    fn matrix(&mut self, name: &str, shape: &[usize]) -> Result<Matrix> {
        Ok(Matrix::F32(self.tensor(name, shape)?))
    }
}

impl Source for SafeTensors {
    fn tensor(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (info, bytes) = self.read_bytes(name)?;
        if info.shape != shape {
            return Err(Error::SafeTensors(format!(
                "Tensor {name} has shape {:?}, expected {shape:?}.",
                info.shape
            )));
        }
        let u16s = || {
            bytes
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
        };
        Ok(match info.dtype {
            Dtype::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            Dtype::BF16 => u16s().map(bf16_to_f32).collect(),
            Dtype::F16 => u16s().map(f16_to_f32).collect(),
            other => {
                return Err(Error::SafeTensors(format!(
                    "Tensor {name} is {other:?}, expected a float type."
                )));
            }
        })
    }
}

/// A GGUF file read through a table from checkpoint names to its own names.
pub struct GgufSource<'a> {
    pub gguf: &'a mut Gguf,
    pub rename: fn(&str) -> String,
}

impl Source for GgufSource<'_> {
    fn tensor(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        self.gguf.read_f32(&(self.rename)(name), Some(shape))
    }

    fn matrix(&mut self, name: &str, shape: &[usize]) -> Result<Matrix> {
        Ok(match self.gguf.read(&(self.rename)(name), Some(shape))?.1 {
            gguf::Data::F32(w) => Matrix::F32(w),
            gguf::Data::Q8_0 { d, q } => Matrix::Q8_0 { d, q },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_matrices_stack_and_join_without_requantizing() {
        let block = |base: i8| -> Matrix {
            Matrix::Q8_0 {
                d: vec![0.5, 0.25],
                q: (0..64).map(|i| base + (i % 32) as i8).collect(),
            }
        };
        // Two rows of 32 inputs each.
        let joined = Matrix::concat_inputs(block(0), 32, block(50), 32, 2);
        let Matrix::Q8_0 { d, q } = &joined else {
            panic!("blocks should stay blocks")
        };
        assert_eq!(d, &[0.5, 0.5, 0.25, 0.25]);
        assert_eq!((q[0], q[32], q[64], q[96]), (0, 50, 0, 50));
        let stacked = Matrix::stack(vec![block(0), Matrix::F32(vec![1.0; 64])]);
        let Matrix::F32(w) = stacked else {
            panic!("a mixed stack is f32")
        };
        assert_eq!(w.len(), 128);
        assert_eq!((w[1], w[127]), (0.5, 1.0));
    }
}
