# Notes for coding agents

Telmo is a speech recognition engine in pure Rust. README.md says what it does and `docs/` how it works.

- The library has no dependencies, and stays that way. A crate may serve an optional feature, such as the download feature's HTTP client, behind a small interface of our own, never the core.
- A change in behaviour comes with a test. A claim about accuracy or speed comes with the numbers and the command that produced them, from the goldens in `tests/` and the benches in `benches/`.
- Tests, fixtures, examples and docs use made-up paths and names (`/home/ada`, `C:\Users\Ada`), never a real machine's or person's.
- Commit subjects are plain sentences saying the behaviour, or `telmo: <sentence>` for a release or a merge, as `git log` shows. Messages are public, so they say what changed and why for any user of the project.
- Everything a user reads (the command's output, docs, comments) uses the fewest plain words that keep the meaning, as a person would explain it to a colleague. Join clauses with a period or a comma, not a colon or a semicolon.
