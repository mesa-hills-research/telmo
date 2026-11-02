# Notes for coding agents

Telmo is a speech recognition engine in pure Rust. README.md says what it does and `docs/` how it works.

- The library has no dependencies, and stays that way. A crate may serve an optional feature, such as the download feature's HTTP client, behind a small interface of our own, never the core.
- A change in behaviour comes with a test. A claim about accuracy or speed comes with the numbers and the command that produced them, from the goldens in `tests/` and the benches in `benches/`.
- Commit subjects are plain sentences saying the behaviour, or `telmo: <sentence>` for a release or a merge, as `git log` shows. Messages are public, so they follow the `prose` skill.
- Everything a user reads (the command's output, docs, comments) follows the `prose` skill.
