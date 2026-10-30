//! Where downloaded models are kept.

use std::ffi::OsString;
use std::path::PathBuf;

/// The cache directory for an operating system, as `std::env::consts::OS` names it, given its
/// environment variables. `%LOCALAPPDATA%\notate` on Windows, `~/Library/Caches/notate` on macOS
/// and `$XDG_CACHE_HOME/notate` or `~/.cache/notate` elsewhere. Empty variables count as unset.
pub fn dir_for(os: &str, var: impl Fn(&str) -> Option<OsString>) -> Option<PathBuf> {
    let var = |name: &str| var(name).filter(|v| !v.is_empty()).map(PathBuf::from);
    let base = match os {
        "windows" => var("LOCALAPPDATA")?,
        "macos" => var("HOME")?.join("Library").join("Caches"),
        // The XDG spec has relative paths ignored. `has_root` keeps the rule testable on Windows.
        _ => match var("XDG_CACHE_HOME").filter(|p| p.has_root()) {
            Some(dir) => dir,
            None => var("HOME")?.join(".cache"),
        },
    };
    Some(base.join("notate"))
}

/// This system's cache directory.
pub fn dir() -> Result<PathBuf, String> {
    dir_for(std::env::consts::OS, |name| std::env::var_os(name)).ok_or_else(|| {
        let variable = if cfg!(windows) {
            "LOCALAPPDATA"
        } else {
            "HOME"
        };
        format!("Couldn't find a cache directory for models. Set {variable} and run it again.")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(os: &str, vars: &[(&str, &str)]) -> Option<PathBuf> {
        dir_for(os, |name| {
            vars.iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| OsString::from(v))
        })
    }

    #[test]
    fn windows_uses_local_app_data() {
        let local = r"C:\Users\Ada\AppData\Local";
        assert_eq!(
            dir("windows", &[("LOCALAPPDATA", local), ("HOME", "/home/ada")]),
            Some(PathBuf::from(local).join("notate"))
        );
        assert_eq!(dir("windows", &[("HOME", "/home/ada")]), None);
        assert_eq!(dir("windows", &[("LOCALAPPDATA", "")]), None);
    }

    #[test]
    fn macos_uses_library_caches() {
        assert_eq!(
            dir(
                "macos",
                &[("HOME", "/Users/ada"), ("XDG_CACHE_HOME", "/tmp/xdg")]
            ),
            Some(PathBuf::from("/Users/ada/Library/Caches/notate"))
        );
        assert_eq!(dir("macos", &[]), None);
    }

    #[test]
    fn linux_and_others_follow_the_xdg_rules() {
        for os in ["linux", "freebsd"] {
            assert_eq!(
                dir(
                    os,
                    &[("HOME", "/home/ada"), ("XDG_CACHE_HOME", "/data/cache")]
                ),
                Some(PathBuf::from("/data/cache/notate"))
            );
            assert_eq!(
                dir(os, &[("HOME", "/home/ada")]),
                Some(PathBuf::from("/home/ada/.cache/notate"))
            );
            for ignored in ["", "relative/cache"] {
                assert_eq!(
                    dir(os, &[("HOME", "/home/ada"), ("XDG_CACHE_HOME", ignored)]),
                    Some(PathBuf::from("/home/ada/.cache/notate"))
                );
            }
            assert_eq!(dir(os, &[]), None);
            assert_eq!(dir(os, &[("HOME", "")]), None);
        }
    }
}
