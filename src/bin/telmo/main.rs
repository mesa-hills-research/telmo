//! The `telmo` command, which transcribes WAV files, packs models and downloads Granite Speech
//! or Parakeet TDT on first use.

mod cache;
mod install;
mod sha256;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use telmo::kernels::Isa;
use telmo::model::ModelType;
use telmo::pack::{self, FileSink};
use telmo::{Model, Options, Precision};

use install::KnownModel;

const HELP: &str = "\
Telmo transcribes speech on the CPU.

Usage
  telmo transcribe [OPTIONS] FILE.wav...
  telmo pack MODEL OUT.pack [--precision P]
  telmo models
  telmo help

Commands
  transcribe   Prints each file's text, after its name when there are several,
               and the timings on stderr
  pack         Converts a model into a pack, which loads in milliseconds
  models       Lists the installed models and where they are

Options
  --model M        granite (the default), parakeet, or the path of a pack, a
                   model directory or a GGUF file
  --threads N      Threads to use (default all cores)
  --isa I          scalar, avx2, avx512 or neon (default TELMO_ISA if set,
                   else avx2 where the CPU has it, else scalar). AVX-512 and
                   NEON are untested on real hardware, so they run only when
                   asked for.
  --precision P    f32, f16, bf16, int8 or q8_0, for pack and for a model that
                   isn't a pack yet (default int8 for Granite and q8_0 for
                   Parakeet)
  --yes            Downloads a missing model without asking

Granite and Parakeet download on first use, after asking, into the cache
directory that telmo models shows. A model directory holds model.safetensors,
config.json and tokenizer.json, and Parakeet's GGUF file needs tokenizer.json
beside it. WAV files may have any sample rate.
";

#[derive(Debug, PartialEq)]
enum Command {
    Transcribe(Transcribe),
    Pack {
        model: PathBuf,
        out: PathBuf,
        precision: Option<Precision>,
    },
    Models,
    Help,
}

#[derive(Debug, PartialEq)]
struct Transcribe {
    model: ModelArg,
    threads: Option<usize>,
    isa: Option<Isa>,
    precision: Option<Precision>,
    yes: bool,
    files: Vec<PathBuf>,
}

#[derive(Debug, PartialEq)]
enum ModelArg {
    Known(&'static KnownModel),
    Path(PathBuf),
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = match parse(&args) {
        Ok(command) => command,
        Err(message) => {
            eprintln!("{message} Run telmo help to see the commands and options.");
            return ExitCode::from(2);
        }
    };
    let result = match command {
        Command::Transcribe(args) => transcribe(&args),
        Command::Pack {
            model,
            out,
            precision,
        } => pack(&model, &out, precision),
        Command::Models => cache::dir().map(|dir| print!("{}", install::list(&dir))),
        Command::Help => {
            print!("{HELP}");
            Ok(())
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn parse(args: &[String]) -> Result<Command, String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        return Ok(Command::Help);
    }
    let Some((command, rest)) = args.split_first() else {
        return Ok(Command::Help);
    };
    if !["transcribe", "pack", "models", "help"].contains(&command.as_str()) {
        return Err(format!("Unknown command {command}."));
    }
    let default_model = ModelArg::Known(install::default_model());
    let mut transcribe = Transcribe {
        model: ModelArg::Known(install::default_model()),
        threads: None,
        isa: None,
        precision: None,
        yes: false,
        files: Vec::new(),
    };
    let mut rest = rest.iter();
    while let Some(arg) = rest.next() {
        let mut value = |error: &str| rest.next().ok_or_else(|| error.to_string());
        match arg.as_str() {
            "--model" => {
                let name = value("--model takes granite, parakeet or a path.")?;
                transcribe.model = KnownModel::named(name)
                    .map_or_else(|| ModelArg::Path(name.into()), ModelArg::Known);
            }
            "--threads" => {
                const ERROR: &str = "--threads takes a number above 0.";
                let n = value(ERROR)?.parse().map_err(|_| ERROR)?;
                transcribe.threads = Some(n).filter(|&n| n > 0);
                if transcribe.threads.is_none() {
                    return Err(ERROR.into());
                }
            }
            "--isa" => {
                const ERROR: &str = "--isa takes scalar, avx2, avx512 or neon.";
                transcribe.isa = Some(Isa::from_name(value(ERROR)?).ok_or(ERROR)?);
            }
            "--precision" => {
                const ERROR: &str = "--precision takes f32, f16, bf16, int8 or q8_0.";
                transcribe.precision = Some(Precision::from_name(value(ERROR)?).ok_or(ERROR)?);
            }
            "--yes" => transcribe.yes = true,
            option if option.starts_with('-') => {
                return Err(format!("Unknown option {option}."));
            }
            file => transcribe.files.push(file.into()),
        }
    }
    let only = |allowed: &[&str]| {
        let given = [
            ("--model", transcribe.model != default_model),
            ("--threads", transcribe.threads.is_some()),
            ("--isa", transcribe.isa.is_some()),
            ("--precision", transcribe.precision.is_some()),
            ("--yes", transcribe.yes),
        ];
        match given
            .iter()
            .find(|(name, set)| *set && !allowed.contains(name))
        {
            Some((name, _)) => Err(format!("{command} doesn't take {name}.")),
            None => Ok(()),
        }
    };
    match command.as_str() {
        "transcribe" if transcribe.files.is_empty() => {
            Err("transcribe takes one or more WAV files.".into())
        }
        "transcribe" => Ok(Command::Transcribe(transcribe)),
        "pack" => {
            only(&["--precision"])?;
            match <[PathBuf; 2]>::try_from(transcribe.files) {
                Ok([model, out]) => Ok(Command::Pack {
                    model,
                    out,
                    precision: transcribe.precision,
                }),
                Err(_) => Err("pack takes a model and the pack file to write.".into()),
            }
        }
        "models" => {
            only(&[])?;
            match transcribe.files.first() {
                Some(arg) => Err(format!("models doesn't take {}.", arg.display())),
                None => Ok(Command::Models),
            }
        }
        _ => Ok(Command::Help),
    }
}

fn transcribe(args: &Transcribe) -> Result<(), String> {
    let path = match &args.model {
        ModelArg::Path(path) => path.clone(),
        ModelArg::Known(model) => {
            let cache = cache::dir()?;
            let pack = cache.join(model.pack_file);
            if pack.is_file() {
                pack
            } else {
                install::install(model, &cache, args.yes)?
            }
        }
    };
    let model_type = ModelType::detect(&path).map_err(|e| e.to_string())?;
    let mut options = Options {
        isa: args.isa,
        precision: args.precision.unwrap_or(model_type.recommended_precision()),
        ..Options::default()
    };
    if let Some(threads) = args.threads {
        options.threads = threads;
    }
    let start = Instant::now();
    let model = Model::load(&path, &options).map_err(|e| e.to_string())?;
    eprintln!(
        "Loaded {} in {:.2} s, {} threads, {} kernels, {} weights.",
        model.name(),
        start.elapsed().as_secs_f64(),
        model.threads(),
        model.isa().name(),
        model.precision().map_or("mixed", |p| p.name())
    );

    let several = args.files.len() > 1;
    let mut failures = 0;
    for file in &args.files {
        if let Err(e) = transcribe_file(&model, file, several) {
            if !several {
                return Err(e);
            }
            eprintln!("{}: {e}", file.display());
            failures += 1;
        }
    }
    match failures {
        0 => Ok(()),
        n => Err(format!(
            "Couldn't transcribe {n} of {} files.",
            args.files.len()
        )),
    }
}

/// Prints a file's text, after its name if `named`, and the time it took on stderr.
fn transcribe_file(model: &Model, file: &Path, named: bool) -> Result<(), String> {
    let audio = telmo::wav::read(file).map_err(|e| e.to_string())?;
    let samples = telmo::resample::resample(&audio.samples, audio.sample_rate, 16000);
    let start = Instant::now();
    let transcript = model.transcribe(&samples).map_err(|e| e.to_string())?;
    let elapsed = start.elapsed().as_secs_f64();
    if named {
        println!("{}: {}", file.display(), transcript.text);
    } else {
        println!("{}", transcript.text);
    }
    eprintln!(
        "{}: {:.2} s of audio in {elapsed:.2} s ({:.1}x real time)",
        file.display(),
        audio.seconds(),
        audio.seconds() / elapsed
    );
    Ok(())
}

/// Converts the model at `input` into a pack at `output`, by default in the model's
/// recommended precision.
fn pack(input: &Path, output: &Path, precision: Option<Precision>) -> Result<(), String> {
    let start = Instant::now();
    let model = ModelType::detect(input).map_err(|e| e.to_string())?;
    let precision = precision.unwrap_or(model.recommended_precision());
    let written = (|| {
        let metadata = pack::metadata_for(input, precision);
        let mut sink = FileSink::create(output, model.pack_model(), metadata)?;
        model.convert_path(input, precision, &mut sink)?;
        sink.finish()
    })();
    if let Err(e) = written {
        // FileSink writes OUT.partial and renames it once complete, so a failure leaves it.
        let mut partial = output.as_os_str().to_owned();
        partial.push(".partial");
        let _ = std::fs::remove_file(partial);
        return Err(e.to_string());
    }
    let bytes = std::fs::metadata(output).map_or(0, |m| m.len());
    eprintln!(
        "Wrote {} ({:.0} MB, {precision}) in {:.1} s.",
        output.display(),
        bytes as f64 / 1e6,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A new, empty directory for a test's files.
    pub fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("telmo-cli-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn parse(args: &str) -> Result<Command, String> {
        let args: Vec<String> = args.split_whitespace().map(String::from).collect();
        super::parse(&args)
    }

    #[test]
    fn no_arguments_or_help_anywhere_shows_the_help() {
        for args in [
            "",
            "help",
            "--help",
            "-h",
            "transcribe a.wav --help",
            "pack -h",
        ] {
            assert_eq!(parse(args), Ok(Command::Help), "{args:?}");
        }
    }

    #[test]
    fn transcribe_takes_files_and_options() {
        assert_eq!(
            parse("transcribe a.wav b.wav"),
            Ok(Command::Transcribe(Transcribe {
                model: ModelArg::Known(&install::GRANITE),
                threads: None,
                isa: None,
                precision: None,
                yes: false,
                files: vec!["a.wav".into(), "b.wav".into()],
            }))
        );
        assert_eq!(
            parse(
                "transcribe --model parakeet a.wav --threads 4 --isa scalar --precision f16 --yes"
            ),
            Ok(Command::Transcribe(Transcribe {
                model: ModelArg::Known(&install::PARAKEET),
                threads: Some(4),
                isa: Some(Isa::Scalar),
                precision: Some(Precision::F16),
                yes: true,
                files: vec!["a.wav".into()],
            }))
        );
        let Ok(Command::Transcribe(args)) = parse("transcribe --model models/x.pack a.wav") else {
            panic!("a model path");
        };
        assert_eq!(args.model, ModelArg::Path("models/x.pack".into()));
    }

    #[test]
    fn bad_transcribe_arguments_say_what_is_wrong() {
        for (args, error) in [
            ("transcribe", "transcribe takes one or more WAV files."),
            (
                "transcribe a.wav --model",
                "--model takes granite, parakeet or a path.",
            ),
            (
                "transcribe a.wav --threads",
                "--threads takes a number above 0.",
            ),
            (
                "transcribe a.wav --threads 0",
                "--threads takes a number above 0.",
            ),
            (
                "transcribe a.wav --threads x",
                "--threads takes a number above 0.",
            ),
            (
                "transcribe a.wav --isa avx",
                "--isa takes scalar, avx2, avx512 or neon.",
            ),
            (
                "transcribe a.wav --precision q4",
                "--precision takes f32, f16, bf16, int8 or q8_0.",
            ),
            ("transcribe a.wav --fast", "Unknown option --fast."),
            ("transcribe-all a.wav", "Unknown command transcribe-all."),
        ] {
            assert_eq!(parse(args), Err(error.to_string()), "{args:?}");
        }
    }

    #[test]
    fn pack_takes_a_model_an_output_and_a_precision() {
        assert_eq!(
            parse("pack model-dir out.pack"),
            Ok(Command::Pack {
                model: "model-dir".into(),
                out: "out.pack".into(),
                precision: None,
            })
        );
        assert_eq!(
            parse("pack --precision bf16 model-dir out.pack"),
            Ok(Command::Pack {
                model: "model-dir".into(),
                out: "out.pack".into(),
                precision: Some(Precision::Bf16),
            })
        );
        let error = "pack takes a model and the pack file to write.";
        assert_eq!(parse("pack model-dir"), Err(error.into()));
        assert_eq!(parse("pack a b c"), Err(error.into()));
        assert_eq!(
            parse("pack a b --threads 2"),
            Err("pack doesn't take --threads.".into())
        );
    }

    #[test]
    fn models_takes_no_arguments() {
        assert_eq!(parse("models"), Ok(Command::Models));
        assert_eq!(
            parse("models extra"),
            Err("models doesn't take extra.".into())
        );
        assert_eq!(
            parse("models --yes"),
            Err("models doesn't take --yes.".into())
        );
    }

    #[test]
    fn help_fits_80_columns_and_names_every_option() {
        assert!(HELP.lines().all(|l| l.len() <= 80));
        for option in ["--model", "--threads", "--isa", "--precision", "--yes"] {
            assert!(HELP.contains(&format!("\n  {option} ")), "{option}");
        }
    }
}
