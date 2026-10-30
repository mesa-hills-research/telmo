use std::fmt;
use std::path::{Path, PathBuf};

/// Everything that can go wrong while loading a model or reading audio.
#[derive(Debug)]
pub enum Error {
    /// A file couldn't be opened or read.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// A JSON document is malformed. The offset counts bytes from its start.
    Json {
        offset: usize,
        message: &'static str,
    },
    /// A safetensors file is malformed or doesn't hold what the model needs.
    SafeTensors(String),
    /// A WAV file is malformed or uses an encoding the reader doesn't take.
    Wav(String),
    /// The model files don't match the model's architecture, or the input doesn't fit the model.
    Model(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Error {
    pub(crate) fn io(path: &Path, source: std::io::Error) -> Self {
        Error::Io {
            path: path.to_path_buf(),
            source,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, source } => write!(f, "Couldn't read {} ({source})", path.display()),
            Error::Json { offset, message } => {
                write!(f, "Malformed JSON at byte {offset} ({message})")
            }
            Error::SafeTensors(message) => write!(f, "Bad safetensors file. {message}"),
            Error::Wav(message) => write!(f, "Bad WAV file. {message}"),
            Error::Model(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
