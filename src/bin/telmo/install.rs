//! The models `telmo` downloads by name, and how they get into the cache.

use std::fmt::Write as _;
use std::fs;
use std::io::{self, IsTerminal};
use std::path::{Path, PathBuf};
use std::process::Command;

use telmo::model::ModelType;

use crate::sha256;

/// A file of a model on Hugging Face.
#[derive(Debug, PartialEq)]
pub struct RemoteFile {
    pub name: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
}

/// A model `--model` names, downloaded from a pinned revision and packed into the cache.
#[derive(Debug, PartialEq)]
pub struct KnownModel {
    /// The name `--model` takes.
    pub name: &'static str,
    /// The name in messages.
    pub title: &'static str,
    pub model_type: ModelType,
    /// The Hugging Face repository.
    pub repo: &'static str,
    pub revision: &'static str,
    pub files: &'static [RemoteFile],
    /// The downloaded file that gets packed, or `None` for the directory of files.
    pub source_file: Option<&'static str>,
    /// The pack's file name in the cache, in the model's recommended precision.
    pub pack_file: &'static str,
    pub pack_megabytes: u64,
    pub attribution: &'static str,
}

pub const GRANITE: KnownModel = KnownModel {
    name: "granite",
    title: "Granite Speech",
    model_type: ModelType::Granite,
    repo: "ibm-granite/granite-speech-5.0-470m-turboctc",
    revision: "1dd23f46f6d4e6eae2531982ba2e33220086dd97",
    files: &[
        RemoteFile {
            name: "model.safetensors",
            bytes: 946_311_576,
            sha256: "5b569c3744c42a9172ac976881539926b66176d03fbaaa3c51b5d2960067f6fc",
        },
        RemoteFile {
            name: "tokenizer.json",
            bytes: 1_699_784,
            sha256: "a1c3ca763ce0690faf0df3456365358547acfdf766122c9e4a69be8e8708042e",
        },
        RemoteFile {
            name: "config.json",
            bytes: 921,
            sha256: "aabd64a2da736a1a1c21237c896b71628f6dd7364f1f5ec8a77f5e04b3ef1011",
        },
    ],
    source_file: None,
    pack_file: "granite-speech-5.0-int8.pack",
    pack_megabytes: 477,
    attribution: "Granite Speech 5.0 470M TurboCTC by IBM, licensed under Apache 2.0 \
                  (https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc).",
};

pub const PARAKEET: KnownModel = KnownModel {
    name: "parakeet",
    title: "Parakeet TDT",
    model_type: ModelType::Parakeet,
    repo: "nvidia/parakeet-tdt-0.6b-v3",
    revision: "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
    files: &[
        RemoteFile {
            name: "parakeet-tdt-0.6b-v3.q8_0.gguf",
            bytes: 713_975_456,
            sha256: "e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e",
        },
        RemoteFile {
            name: "tokenizer.json",
            bytes: 1_159_960,
            sha256: "bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91",
        },
    ],
    source_file: Some("parakeet-tdt-0.6b-v3.q8_0.gguf"),
    pack_file: "parakeet-tdt-0.6b-v3-q8_0.pack",
    pack_megabytes: 723,
    attribution: "Parakeet TDT 0.6B v3 by NVIDIA, licensed under CC BY 4.0 \
                  (https://creativecommons.org/licenses/by/4.0/).",
};

pub const KNOWN_MODELS: [&KnownModel; 2] = [&GRANITE, &PARAKEET];

/// The command that runs curl in the steps printed for a manual install. PowerShell has `curl`
/// as an alias of another command.
const CURL: &str = if cfg!(windows) { "curl.exe" } else { "curl" };

fn megabytes(bytes: u64) -> u64 {
    (bytes + 500_000) / 1_000_000
}

impl KnownModel {
    pub fn named(name: &str) -> Option<&'static KnownModel> {
        KNOWN_MODELS.into_iter().find(|m| m.name == name)
    }

    fn url(&self, file: &str) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{file}",
            self.repo, self.revision
        )
    }

    /// The directory the files download into, named like the repository.
    fn download_dir_name(&self) -> &'static str {
        self.repo.rsplit('/').next().unwrap_or(self.repo)
    }

    fn download_megabytes(&self) -> u64 {
        megabytes(self.files.iter().map(|f| f.bytes).sum())
    }
}

/// The model `transcribe` uses unless `--model` names another.
pub fn default_model() -> &'static KnownModel {
    KNOWN_MODELS
        .into_iter()
        .find(|m| m.model_type == ModelType::DEFAULT)
        .expect("the default model is known")
}

/// Whether to download a missing model. `--yes` downloads it at once. Without a terminal to ask
/// on it isn't downloaded, and otherwise `ask` decides, where an empty answer means yes and
/// `None` (the end of input) means no.
pub fn should_download(yes: bool, interactive: bool, ask: impl FnOnce() -> Option<String>) -> bool {
    if yes {
        return true;
    }
    interactive
        && ask().is_some_and(|answer| {
            matches!(
                answer.trim().to_ascii_lowercase().as_str(),
                "" | "y" | "yes"
            )
        })
}

/// How to download and pack a model by hand, into the pack `transcribe` looks for.
pub fn manual_steps(model: &KnownModel, cache: &Path) -> String {
    let dir = model.download_dir_name();
    let mut steps = format!(
        "To install {} by hand, download its files at revision {} and pack them into the cache.\n\n",
        model.title, model.revision
    );
    for file in model.files {
        let url = model.url(file.name);
        let _ = writeln!(
            steps,
            "  {CURL} -fL --create-dirs -o {dir}/{} {url}",
            file.name
        );
    }
    let source = match model.source_file {
        Some(file) => format!("{dir}/{file}"),
        None => dir.to_string(),
    };
    let _ = write!(
        steps,
        "  telmo pack {source} \"{}\"\n\nThe downloaded files can go once the pack is written.",
        cache.join(model.pack_file).display()
    );
    steps
}

/// Downloads a model into the cache and packs it, once the user agrees, and returns the pack.
pub fn install(model: &KnownModel, cache: &Path, yes: bool) -> Result<PathBuf, String> {
    eprintln!("{}", model.attribution);
    let size = format!(
        "{} MB, {} MB once packed",
        model.download_megabytes(),
        model.pack_megabytes
    );
    let interactive = io::stdin().is_terminal();
    let ask = || {
        eprint!(
            "Download {} ({size}) to {}? [Y/n] ",
            model.title,
            cache.display()
        );
        let mut answer = String::new();
        match io::stdin().read_line(&mut answer) {
            Ok(n) if n > 0 => Some(answer),
            _ => None,
        }
    };
    if !should_download(yes, interactive, ask) {
        let steps = manual_steps(model, cache);
        return Err(if interactive {
            steps
        } else {
            format!(
                "{} isn't installed. Run telmo again with --yes to download it, or install it by hand.\n\n{steps}",
                model.title
            )
        });
    }
    if yes {
        eprintln!(
            "Downloading {} ({size}) to {}.",
            model.title,
            cache.display()
        );
    }
    download_and_pack(model, cache)
}

fn download_and_pack(model: &KnownModel, cache: &Path) -> Result<PathBuf, String> {
    let downloads = cache.join("downloads");
    let dir = downloads.join(model.download_dir_name());
    fs::create_dir_all(&dir).map_err(|e| format!("Couldn't create {} ({e}).", dir.display()))?;
    for file in model.files {
        let path = dir.join(file.name);
        let have = fs::metadata(&path).map_or(0, |m| m.len());
        if have != file.bytes {
            // A file larger than it should be can't be resumed.
            if have > file.bytes {
                fs::remove_file(&path)
                    .map_err(|e| format!("Couldn't delete {} ({e}).", path.display()))?;
            }
            match megabytes(file.bytes) {
                0 => eprintln!("{}", file.name),
                mb => eprintln!("{} ({mb} MB)", file.name),
            }
            download(&model.url(file.name), &path).map_err(|e| match e {
                Download::NoCurl => format!(
                    "Telmo downloads models with curl, which isn't installed. Install curl, or install the model by hand.\n\n{}",
                    manual_steps(model, cache)
                ),
                Download::Failed => format!(
                    "Couldn't download {}. Run the command again to resume.",
                    file.name
                ),
            })?;
        }
        let digest = sha256::file_hex(&path)
            .map_err(|e| format!("Couldn't read {} ({e}).", path.display()))?;
        if digest != file.sha256 {
            let _ = fs::remove_file(&path);
            return Err(format!(
                "{} didn't match its SHA-256 checksum, so Telmo deleted it. Run the command again to download it.",
                file.name
            ));
        }
    }
    let pack = cache.join(model.pack_file);
    eprintln!("Packing {} into {}.", model.title, pack.display());
    let source = model.source_file.map_or(dir.clone(), |file| dir.join(file));
    crate::pack(&source, &pack, None)?;
    if let Err(e) = fs::remove_dir_all(&dir) {
        eprintln!(
            "Couldn't delete the downloaded files in {} ({e}).",
            dir.display()
        );
    }
    // Only removes the directory once no other model's files are in it.
    let _ = fs::remove_dir(&downloads);
    Ok(pack)
}

enum Download {
    NoCurl,
    Failed,
}

/// Downloads `url` to `path` with curl, resuming what's there. curl shows the progress, and
/// its errors.
fn download(url: &str, path: &Path) -> Result<(), Download> {
    let status = Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--continue-at",
            "-",
            "--progress-bar",
            "--output",
        ])
        .arg(path)
        .arg(url)
        .status();
    match status {
        Ok(status) if status.success() => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Err(Download::NoCurl),
        _ => Err(Download::Failed),
    }
}

/// The installed models and where they are.
pub fn list(cache: &Path) -> String {
    let mut out = format!("Cache  {}\n", cache.display());
    for model in KNOWN_MODELS {
        let status = match fs::metadata(cache.join(model.pack_file)) {
            Ok(meta) if meta.is_file() => {
                format!("{:>4} MB  {}", megabytes(meta.len()), model.pack_file)
            }
            _ => "Not installed".to_string(),
        };
        let _ = writeln!(out, "{:<10}{status}", model.name);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_granite_packed_in_its_recommended_precision() {
        assert_eq!(default_model(), &GRANITE);
        assert_eq!(KnownModel::named("parakeet"), Some(&PARAKEET));
        assert_eq!(KnownModel::named("whisper"), None);
        for model in KNOWN_MODELS {
            let precision = model.model_type.recommended_precision().name();
            assert!(model.pack_file.ends_with(&format!("-{precision}.pack")));
            assert!(model.files.iter().all(|f| f.sha256.len() == 64));
        }
        assert_eq!(GRANITE.download_megabytes(), 948);
        assert_eq!(PARAKEET.download_megabytes(), 715);
    }

    #[test]
    fn yes_downloads_without_asking() {
        for interactive in [false, true] {
            assert!(should_download(true, interactive, || panic!("asked")));
        }
    }

    #[test]
    fn without_a_terminal_or_yes_nothing_is_downloaded() {
        assert!(!should_download(false, false, || panic!("asked")));
    }

    #[test]
    fn on_a_terminal_the_answer_decides_and_enter_means_yes() {
        for answer in ["\n", "y\n", "Y\n", "yes\n", " YES \r\n", ""] {
            assert!(
                should_download(false, true, || Some(answer.into())),
                "{answer:?}"
            );
        }
        for answer in ["n\n", "no\n", "N\n", "maybe\n", "yy\n"] {
            assert!(
                !should_download(false, true, || Some(answer.into())),
                "{answer:?}"
            );
        }
        assert!(!should_download(false, true, || None), "the end of input");
    }

    #[test]
    fn manual_steps_name_the_urls_at_the_revision_and_the_pack_command() {
        let cache = Path::new("cache");
        let steps = manual_steps(&GRANITE, cache);
        assert!(steps.contains(GRANITE.revision));
        for file in ["model.safetensors", "tokenizer.json", "config.json"] {
            assert!(steps.contains(&format!(
                " -o granite-speech-5.0-470m-turboctc/{file} https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc/resolve/1dd23f46f6d4e6eae2531982ba2e33220086dd97/{file}\n"
            )));
        }
        let pack = cache.join("granite-speech-5.0-int8.pack");
        assert!(steps.contains(&format!(
            "\n  telmo pack granite-speech-5.0-470m-turboctc \"{}\"\n",
            pack.display()
        )));

        let steps = manual_steps(&PARAKEET, cache);
        assert!(steps.contains("https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/parakeet-tdt-0.6b-v3.q8_0.gguf\n"));
        assert!(steps.contains("https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/tokenizer.json\n"));
        assert!(
            steps.contains("telmo pack parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.q8_0.gguf \"")
        );
    }

    #[test]
    fn models_lists_what_the_cache_holds() {
        let cache = crate::tests::temp_dir("models");
        let empty = list(&cache);
        assert_eq!(
            empty,
            format!(
                "Cache  {}\ngranite   Not installed\nparakeet  Not installed\n",
                cache.display()
            )
        );

        fs::write(
            cache.join("granite-speech-5.0-int8.pack"),
            vec![0; 2_400_000],
        )
        .unwrap();
        assert_eq!(
            list(&cache),
            format!(
                "Cache  {}\ngranite      2 MB  granite-speech-5.0-int8.pack\nparakeet  Not installed\n",
                cache.display()
            )
        );
        fs::remove_dir_all(&cache).unwrap();
        assert_eq!(list(&cache), empty, "a cache that doesn't exist yet");
    }
}
