/// A file to download, pinned to its size and SHA-256, such as the trained disfluency tagger's
/// pack, [`disfluency::PACK_DOWNLOAD`](crate::disfluency::PACK_DOWNLOAD).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct RemoteFile {
    /// Where it downloads from.
    pub url: &'static str,
    /// The file's name, which a downloaded copy keeps.
    pub name: &'static str,
    /// Its size in bytes.
    pub bytes: u64,
    /// Its SHA-256 in lower-case hex.
    pub sha256: &'static str,
}

impl RemoteFile {
    /// The file `name` at `url`, of `bytes` bytes, whose SHA-256 is `sha256` in lower-case hex.
    pub const fn new(
        url: &'static str,
        name: &'static str,
        bytes: u64,
        sha256: &'static str,
    ) -> RemoteFile {
        RemoteFile {
            url,
            name,
            bytes,
            sha256,
        }
    }
}
