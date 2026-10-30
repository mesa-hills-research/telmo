//! Converts a model into a pack, the engine's own weight file, which later loads by mapping it.
//!
//! ```text
//! cargo run --release --example pack -- MODEL OUT.pack [--precision P]
//! cargo run --release --example pack -- --list FILE.pack
//! ```
//!
//! MODEL is a directory holding `model.safetensors`, `config.json` and `tokenizer.json` of
//! Parakeet TDT or Granite Speech, or NVIDIA's Parakeet GGUF file with `tokenizer.json` beside
//! it. `--precision` is f32, f16, bf16, int8 or q8_0. By default it is the model's recommended
//! 8-bit precision, q8_0 for Parakeet (which keeps a GGUF file's blocks as they are) and int8 for
//! Granite. `--list` prints a pack's metadata and entries.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use notate::Precision;
use notate::model::ModelType;
use notate::pack::{self, FileSink, Kind, Pack};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut precision = None;
    let mut list = false;
    let mut paths: Vec<PathBuf> = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--precision" => {
                precision = Some(
                    args.next()
                        .as_deref()
                        .and_then(Precision::from_name)
                        .ok_or("--precision takes f32, f16, bf16, int8 or q8_0")?,
                )
            }
            "--list" => list = true,
            _ => paths.push(arg.into()),
        }
    }
    if list {
        let [path] = paths.as_slice() else {
            return Err("--list takes one pack".into());
        };
        let pack = Pack::open(path).map_err(|e| e.to_string())?;
        println!("model {}", pack.model());
        for (k, v) in pack.metadata() {
            println!("{k} {v}");
        }
        for e in pack.entries() {
            let kind = match &e.kind {
                Kind::Matrix { precision, n, k } => format!("{precision} {n}x{k}"),
                Kind::F32 { shape } => format!("f32 {shape:?}"),
                Kind::Bytes => "bytes".into(),
            };
            println!("{}\t{kind}\t{}", e.name, e.size());
        }
        return Ok(());
    }
    let [input, output] = paths.as_slice() else {
        return Err("Usage: pack MODEL OUT.pack [--precision P], or pack --list FILE.pack".into());
    };
    let start = Instant::now();
    let model = ModelType::detect(input).map_err(|e| e.to_string())?;
    let precision = precision.unwrap_or(model.recommended_precision());
    let mut sink = FileSink::create(
        output,
        model.pack_model(),
        pack::metadata_for(input, precision),
    )
    .map_err(|e| e.to_string())?;
    model
        .convert_path(input, precision, &mut sink)
        .map_err(|e| e.to_string())?;
    sink.finish().map_err(|e| e.to_string())?;
    let bytes = std::fs::metadata(output).map_err(|e| e.to_string())?.len();
    eprintln!(
        "Wrote {} ({:.0} MB, {precision}) in {:.1} s.",
        output.display(),
        bytes as f64 / 1e6,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
