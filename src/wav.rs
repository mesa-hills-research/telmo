//! A reader for WAV files: integer PCM of 8, 16, 24 or 32 bits and 32 or 64-bit float,
//! plain or in WAVE_FORMAT_EXTENSIBLE, with any number of channels.

use std::path::Path;

use crate::error::{Error, Result};

/// Decoded audio, mixed down to one channel, with samples in [-1, 1).
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    pub sample_rate: u32,
    /// The channel count of the file before mixing.
    pub channels: u16,
    pub samples: Vec<f32>,
}

impl Audio {
    pub fn seconds(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }
}

/// Reads and decodes a WAV file.
pub fn read(path: &Path) -> Result<Audio> {
    let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
    parse(&bytes)
}

#[derive(Clone, Copy, PartialEq)]
enum Encoding {
    Int,
    Float,
}

struct Format {
    encoding: Encoding,
    channels: u16,
    sample_rate: u32,
    bits: u16,
}

const PCM: u16 = 1;
const IEEE_FLOAT: u16 = 3;
const EXTENSIBLE: u16 = 0xFFFE;

/// Decodes a whole WAV file held in memory.
pub fn parse(bytes: &[u8]) -> Result<Audio> {
    let bad = |message: &str| Error::Wav(message.to_string());
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad("It doesn't start with a RIFF WAVE header."));
    }
    let mut pos = 12;
    let mut format = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = pos + 8;
        if id == b"fmt " {
            let end = body
                .checked_add(size)
                .filter(|&e| e <= bytes.len())
                .ok_or_else(|| bad("The fmt chunk is cut short."))?;
            format = Some(parse_format(&bytes[body..end])?);
        } else if id == b"data" {
            let format = format.ok_or_else(|| bad("The data chunk comes before the fmt chunk."))?;
            // Streaming writers leave the size at 0 or 0xFFFFFFFF, meaning "up to the end".
            let available = bytes.len() - body;
            let size = if size == 0 || size == 0xFFFF_FFFF {
                available
            } else {
                size
            };
            if size > available {
                return Err(bad("The data chunk is cut short."));
            }
            return decode(&format, &bytes[body..body + size]);
        }
        // Chunks are padded to an even length.
        pos = body
            .checked_add(size)
            .and_then(|p| p.checked_add(size & 1))
            .ok_or_else(|| bad("A chunk size is too large."))?;
    }
    Err(bad("It has no data chunk."))
}

fn parse_format(chunk: &[u8]) -> Result<Format> {
    let bad = |message: &str| Error::Wav(message.to_string());
    if chunk.len() < 16 {
        return Err(bad("The fmt chunk is shorter than 16 bytes."));
    }
    let u16_at = |i: usize| u16::from_le_bytes([chunk[i], chunk[i + 1]]);
    let mut tag = u16_at(0);
    let channels = u16_at(2);
    let sample_rate = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
    let block_align = u16_at(12);
    let bits = u16_at(14);
    if tag == EXTENSIBLE {
        if chunk.len() < 40 {
            return Err(bad("The extensible fmt chunk is shorter than 40 bytes."));
        }
        // The sub-format GUID starts with the plain format tag.
        tag = u16_at(24);
    }
    let encoding = match (tag, bits) {
        (PCM, 8 | 16 | 24 | 32) => Encoding::Int,
        (IEEE_FLOAT, 32 | 64) => Encoding::Float,
        _ => {
            return Err(Error::Wav(format!(
                "Format {tag} with {bits}-bit samples isn't supported. Use integer PCM or float."
            )));
        }
    };
    if channels == 0 || sample_rate == 0 {
        return Err(bad("It has no channels or a sample rate of 0."));
    }
    if block_align as usize != channels as usize * (bits as usize / 8) {
        return Err(bad(
            "Its block alignment doesn't match its channels and sample size.",
        ));
    }
    Ok(Format {
        encoding,
        channels,
        sample_rate,
        bits,
    })
}

fn decode(format: &Format, data: &[u8]) -> Result<Audio> {
    let width = format.bits as usize / 8;
    let channels = format.channels as usize;
    let frame = width * channels;
    // A trailing partial frame is dropped.
    let frames = data.len() / frame;
    let sample = |bytes: &[u8]| -> f32 {
        match (format.encoding, width) {
            (Encoding::Int, 1) => (bytes[0] as f32 - 128.0) / 128.0,
            (Encoding::Int, 2) => i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0,
            (Encoding::Int, 3) => {
                let v = i32::from_le_bytes([0, bytes[0], bytes[1], bytes[2]]) >> 8;
                v as f32 / 8_388_608.0
            }
            (Encoding::Int, _) => {
                (i32::from_le_bytes(bytes.try_into().unwrap()) as f64 / 2_147_483_648.0) as f32
            }
            (Encoding::Float, 4) => f32::from_le_bytes(bytes.try_into().unwrap()),
            (Encoding::Float, _) => f64::from_le_bytes(bytes.try_into().unwrap()) as f32,
        }
    };
    let mut samples = Vec::with_capacity(frames);
    for f in data[..frames * frame].chunks_exact(frame) {
        if channels == 1 {
            samples.push(sample(f));
        } else {
            let sum: f64 = f.chunks_exact(width).map(|s| sample(s) as f64).sum();
            samples.push((sum / channels as f64) as f32);
        }
    }
    Ok(Audio {
        sample_rate: format.sample_rate,
        channels: format.channels,
        samples,
    })
}

/// Encodes 16-bit mono PCM. Samples are clamped to [-1, 1] and rounded.
pub fn encode_pcm16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = samples.len() * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&PCM.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32768.0)
            .round()
            .clamp(-32768.0, 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(
        tag: u16,
        channels: u16,
        rate: u32,
        bits: u16,
        data: &[u8],
        extra_chunk: bool,
    ) -> Vec<u8> {
        let mut fmt = Vec::new();
        let align = channels * bits / 8;
        let outer_tag = if tag == EXTENSIBLE { EXTENSIBLE } else { tag };
        fmt.extend_from_slice(&outer_tag.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&(rate * align as u32).to_le_bytes());
        fmt.extend_from_slice(&align.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());
        if tag == EXTENSIBLE {
            fmt.extend_from_slice(&22u16.to_le_bytes());
            fmt.extend_from_slice(&bits.to_le_bytes());
            fmt.extend_from_slice(&0u32.to_le_bytes());
            fmt.extend_from_slice(&IEEE_FLOAT.to_le_bytes());
            fmt.extend_from_slice(&[0u8; 14]);
        }
        let mut out = b"RIFF\0\0\0\0WAVE".to_vec();
        if extra_chunk {
            out.extend_from_slice(b"LIST");
            out.extend_from_slice(&3u32.to_le_bytes());
            out.extend_from_slice(b"abc\0");
        }
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        out.extend_from_slice(&fmt);
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn pcm16_round_trips() {
        let samples = [0.0, 0.5, -0.5, -1.0, 32767.0 / 32768.0];
        let audio = parse(&encode_pcm16(&samples, 16000)).unwrap();
        assert_eq!(audio.sample_rate, 16000);
        assert_eq!(audio.samples, samples);
    }

    #[test]
    fn every_sample_format_decodes() {
        let i16s: Vec<u8> = [-32768i16, 16384]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            parse(&wav(PCM, 1, 8000, 16, &i16s, true)).unwrap().samples,
            [-1.0, 0.5]
        );
        let u8s = [0u8, 192];
        assert_eq!(
            parse(&wav(PCM, 1, 8000, 8, &u8s, false)).unwrap().samples,
            [-1.0, 0.5]
        );
        let i24s = [0x00, 0x00, 0x80, 0x00, 0x00, 0x40];
        assert_eq!(
            parse(&wav(PCM, 1, 8000, 24, &i24s, false)).unwrap().samples,
            [-1.0, 0.5]
        );
        let i32s: Vec<u8> = [i32::MIN, 1 << 30]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            parse(&wav(PCM, 1, 8000, 32, &i32s, false)).unwrap().samples,
            [-1.0, 0.5]
        );
        let f32s: Vec<u8> = [0.25f32, -0.75]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            parse(&wav(IEEE_FLOAT, 1, 8000, 32, &f32s, false))
                .unwrap()
                .samples,
            [0.25, -0.75]
        );
        assert_eq!(
            parse(&wav(EXTENSIBLE, 1, 8000, 32, &f32s, false))
                .unwrap()
                .samples,
            [0.25, -0.75]
        );
        let f64s: Vec<u8> = [0.25f64, -0.75]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            parse(&wav(IEEE_FLOAT, 1, 8000, 64, &f64s, false))
                .unwrap()
                .samples,
            [0.25, -0.75]
        );
    }

    #[test]
    fn stereo_is_mixed_to_mono() {
        let data: Vec<u8> = [16384i16, -16384, 8192, 8192]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let audio = parse(&wav(PCM, 2, 48000, 16, &data, false)).unwrap();
        assert_eq!(audio.channels, 2);
        assert_eq!(audio.samples, [0.0, 0.25]);
    }

    #[test]
    fn malformed_files_are_refused() {
        assert!(parse(b"RIFX\0\0\0\0WAVE").is_err());
        assert!(parse(&wav(PCM, 1, 8000, 12, &[0, 0], false)).is_err());
        assert!(parse(&wav(PCM, 0, 8000, 16, &[], false)).is_err());
        assert!(parse(&wav(7, 1, 8000, 8, &[0], false)).is_err());
        let mut short = wav(PCM, 1, 8000, 16, &[0, 0, 0, 0], false);
        short.truncate(short.len() - 2);
        assert!(parse(&short).is_err());
        let no_data = &wav(PCM, 1, 8000, 16, &[], false)[..36];
        assert!(parse(no_data).is_err());
    }
}
