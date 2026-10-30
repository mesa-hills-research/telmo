//! The readers must return an error, never panic, on damaged input. Each test mutates a valid
//! file in many random ways (flipped bits, overwritten bytes, truncation, inserted junk).

use telmo::gguf::{self, Gguf, TensorType};
use telmo::pack::{FileSink, Kind, Pack, Sink};
use telmo::quant::{Precision, Rows};
use telmo::safetensors::SafeTensors;
use telmo::tokenizer::Tokenizer;
use telmo::{json, wav};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn mutate(input: &[u8], rng: &mut Rng) -> Vec<u8> {
    let mut out = input.to_vec();
    for _ in 0..1 + rng.below(4) {
        match rng.below(5) {
            0 if !out.is_empty() => {
                let i = rng.below(out.len());
                out[i] ^= 1 << rng.below(8);
            }
            1 if !out.is_empty() => {
                let i = rng.below(out.len());
                out[i] = rng.next() as u8;
            }
            2 => out.truncate(rng.below(out.len() + 1)),
            3 => {
                let i = rng.below(out.len() + 1);
                let junk = [b'"', b'{', b'[', b'\\', b'-', b'9', 0xff, 0x00, b'e', b':'];
                out.insert(i, junk[rng.below(junk.len())]);
            }
            _ if out.len() > 2 => {
                let i = rng.below(out.len() - 1);
                out.remove(i);
            }
            _ => {}
        }
    }
    out
}

fn safetensors_file() -> Vec<u8> {
    let header = r#"{"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"b":{"dtype":"I64","shape":[2],"data_offsets":[24,40]},"c":{"dtype":"BF16","shape":[4],"data_offsets":[40,48]}}"#;
    let header = format!("{header}{}", " ".repeat((8 - (8 + header.len()) % 8) % 8));
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend((0..48u8).map(|b| b.wrapping_mul(37)));
    out
}

#[test]
fn damaged_json_is_refused_without_panicking() {
    let valid = br#"{"model":{"type":"BPE","vocab":{"a":0,"_b":1,"<0x0A>":2}},"added_tokens":[{"id":3,"content":"<unk>","special":true}],"n":[1.5e-3,-0,true,null,"x\ny"]}"#;
    assert!(json::parse(valid).is_ok());
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for _ in 0..20_000 {
        let input = mutate(valid, &mut rng);
        let _ = json::parse(&input);
        let _ = Tokenizer::from_json(&input);
    }
}

#[test]
fn damaged_safetensors_are_refused_without_panicking() {
    let valid = safetensors_file();
    let mut st = SafeTensors::from_bytes(valid.clone()).unwrap();
    assert_eq!(st.read_f32("a", Some(&[2, 3])).unwrap().len(), 6);
    let mut rng = Rng(0x0fed_cba9_8765_4321);
    let mut accepted = 0;
    for _ in 0..20_000 {
        let input = mutate(&valid, &mut rng);
        if let Ok(mut st) = SafeTensors::from_bytes(input) {
            accepted += 1;
            for name in ["a", "b", "c", "missing"] {
                let _ = st.read_f32(name, None);
                let _ = st.read_i64(name, None);
                let _ = st.read_bytes(name);
            }
        }
    }
    // Mutations that only touch tensor data leave a valid file.
    assert!(accepted > 0);
}

#[test]
fn damaged_wav_files_are_refused_without_panicking() {
    let samples: Vec<f32> = (0..64).map(|i| (i as f32 * 0.3).sin() * 0.5).collect();
    let valid = wav::encode_pcm16(&samples, 16000);
    assert_eq!(wav::parse(&valid).unwrap().samples.len(), 64);
    let mut rng = Rng(0xdead_beef_cafe_f00d);
    for _ in 0..20_000 {
        let input = mutate(&valid[..60], &mut rng);
        let _ = wav::parse(&input);
        let mut whole = mutate(&valid, &mut rng);
        whole.truncate(rng.below(whole.len() + 1));
        let _ = wav::parse(&whole);
    }
}

fn gguf_file() -> Vec<u8> {
    let f32s: Vec<u8> = (0..6)
        .flat_map(|i| (i as f32 * 0.5).to_le_bytes())
        .collect();
    let mut q8 = Vec::new();
    for row in 0..2u16 {
        q8.extend((0x3400 + row).to_le_bytes());
        q8.extend((0..32).map(|i| i as u8));
    }
    gguf::write_for_tests(
        &[
            ("general.architecture", gguf::Value::String("asr".into())),
            (
                "asr.tokenizer.vocab",
                gguf::Value::Array(vec![gguf::Value::String("a".into()); 3]),
            ),
        ],
        &[
            ("a", &[3, 2], TensorType::F32, &f32s),
            ("q", &[32, 2], TensorType::Q8_0, &q8),
        ],
    )
}

#[test]
fn damaged_gguf_files_are_refused_without_panicking() {
    let valid = gguf_file();
    assert!(Gguf::from_bytes(valid.clone()).is_ok());
    let mut rng = Rng(0x5eed_0f09_9907_1234);
    let mut accepted = 0;
    for _ in 0..20_000 {
        let input = mutate(&valid, &mut rng);
        if let Ok(mut g) = Gguf::from_bytes(input) {
            accepted += 1;
            let names: Vec<String> = g.tensors.iter().map(|t| t.name.clone()).collect();
            for name in names {
                let _ = g.read_f32(&name, None);
                let _ = g.read(&name, Some(&[2, 32]));
            }
        }
    }
    assert!(accepted > 0);
}

fn pack_file() -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!("telmo-corrupt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("small.pack");
    let mut sink = FileSink::create(&path, "test", vec![("source".into(), "x".into())]).unwrap();
    let w: Vec<f32> = (0..20 * 32).map(|i| (i % 13) as f32 - 6.0).collect();
    sink.matrix("q", &Rows::F32(&w), 20, 32, Precision::Q8_0)
        .unwrap();
    sink.matrix("h", &Rows::F32(&w), 20, 32, Precision::Bf16)
        .unwrap();
    sink.f32("v", &[3], &[1.0, 2.0, 3.0]).unwrap();
    sink.bytes("b", b"{}").unwrap();
    sink.finish().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    bytes
}

#[test]
fn damaged_packs_are_refused_without_panicking() {
    let valid = pack_file();
    assert!(Pack::from_bytes(&valid).is_ok());
    let mut rng = Rng(0x0bad_cafe_1357_9bdf);
    let mut accepted = 0;
    for _ in 0..20_000 {
        let input = mutate(&valid, &mut rng);
        if let Ok(pack) = Pack::from_bytes(&input) {
            accepted += 1;
            for entry in pack.entries() {
                match &entry.kind {
                    Kind::Matrix { n, k, .. } => {
                        let w = pack.matrix(&entry.name, *n, *k).unwrap();
                        let _ = w.to_rows();
                    }
                    Kind::F32 { shape } => {
                        let v = pack.vector(&entry.name, shape).unwrap();
                        let _ = v.iter().sum::<f32>();
                    }
                    Kind::Bytes => {
                        let _ = pack.bytes(&entry.name).unwrap();
                    }
                }
            }
        }
    }
    assert!(accepted > 0);
}
