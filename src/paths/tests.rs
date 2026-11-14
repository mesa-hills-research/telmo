use super::*;

fn tree() -> Paths {
    Paths::from_list([
        "src/engine.rs",
        "src/lib.rs",
        "src/biasing/spoken.rs",
        "src/biasing/tdt_beam.rs",
        "src/SpecialWords.rs",
        "tests/engine.rs",
        "docs/README.md",
        "README.md",
        "Cargo.toml",
        ".gitignore",
        ".github/workflows/ci.yml",
        "migrations/0001_init.sql",
        "assets/v2/logo.png",
    ])
}

/// The stretches of `text` that snap to `paths`, each as said and as written.
fn snapped(paths: &Paths, text: &str) -> Vec<(String, String)> {
    let words: Vec<&str> = text.split_whitespace().collect();
    (paths.snap(&words, true).into_iter())
        .map(|s| (words[s.words.clone()].join(" "), s.text))
        .collect()
}

fn pair(said: &str, written: &str) -> Vec<(String, String)> {
    vec![(said.to_string(), written.to_string())]
}

#[test]
fn a_dictated_path_snaps_to_its_spelling() {
    let paths = tree();
    for (said, stretch, written) in [
        (
            "open source slash engine dot rs now",
            "source slash engine dot rs",
            "src/engine.rs",
        ),
        ("open engine dot rs", "engine dot rs", "engine.rs"),
        (
            "in source slash biasing slash tdt beam dot rs",
            "source slash biasing slash tdt beam dot rs",
            "src/biasing/tdt_beam.rs",
        ),
        (
            "the tdt underscore beam dot rs file",
            "tdt underscore beam dot rs",
            "tdt_beam.rs",
        ),
        ("check readme dot md", "readme dot md", "README.md"),
        ("edit cargo dot toml", "cargo dot toml", "Cargo.toml"),
        (
            "see special words dot rs",
            "special words dot rs",
            "SpecialWords.rs",
        ),
        ("add it to dot gitignore", "dot gitignore", ".gitignore"),
        (
            "run zero zero zero one underscore init dot sql",
            "zero zero zero one underscore init dot sql",
            "0001_init.sql",
        ),
        (
            "the S R C slash lib dot R S file",
            "S R C slash lib dot R S",
            "src/lib.rs",
        ),
        (
            "open assets slash v two slash logo dot png",
            "assets slash v two slash logo dot png",
            "assets/v2/logo.png",
        ),
        (
            "look at source slash lib dot rust",
            "source slash lib dot rust",
            "src/lib.rs",
        ),
        (
            "in source slash um engine dot rs",
            "source slash um engine dot rs",
            "src/engine.rs",
        ),
        (
            "dot github slash workflows slash ci dot yml",
            "dot github slash workflows slash ci dot yml",
            ".github/workflows/ci.yml",
        ),
        (
            "so source slash engine.rs",
            "source slash engine.rs",
            "src/engine.rs",
        ),
        (
            "in source forward slash engine dot rs",
            "source forward slash engine dot rs",
            "src/engine.rs",
        ),
    ] {
        assert_eq!(snapped(&paths, said), pair(stretch, written), "{said}");
    }
}

#[test]
fn a_folder_snaps_before_a_slash_and_a_near_miss_stays() {
    let paths = tree();
    // The folder alone, before what follows it.
    assert_eq!(
        snapped(&paths, "make source slash new dot rs"),
        pair("source", "src")
    );
    assert_eq!(
        snapped(&paths, "make source slash biasing slash new dot rs"),
        pair("source slash biasing", "src/biasing")
    );
    // Punctuation ends a stretch, and the folder before it still snaps.
    assert_eq!(
        snapped(&paths, "open source slash engine, dot rs"),
        pair("source", "src")
    );
    for said in [
        "open source code",
        "a dot on the map",
        "and slash or",
        "open engine dot r",
        "open spokin dot rs",
        "the engine",
        "readme",
    ] {
        assert_eq!(snapped(&paths, said), vec![], "{said}");
    }
}

#[test]
fn two_paths_written_differently_leave_a_stretch_as_said() {
    let paths = Paths::from_list(["a/Notes.md", "b/notes.md", "c/d/NOTES.md"]);
    // The spelling whose case agrees most with the transcript's wins.
    assert_eq!(snapped(&paths, "open notes dot md")[0].1, "notes.md");
    assert_eq!(snapped(&paths, "open Notes dot md")[0].1, "Notes.md");
    assert_eq!(snapped(&paths, "open NOTES dot md")[0].1, "NOTES.md");
    // These two agree equally with "notes", at the same depth.
    let paths = Paths::from_list(["a/nOtes.md", "b/noTes.md"]);
    assert_eq!(snapped(&paths, "open notes dot md"), vec![]);
    // A shallower path breaks a tie.
    let paths = Paths::from_list(["a/nOtes.md", "b/c/noTes.md"]);
    assert_eq!(snapped(&paths, "open notes dot md")[0].1, "nOtes.md");
    // The same spelling in two places is no tie.
    let paths = Paths::from_list(["src/a/mod.rs", "src/b/mod.rs"]);
    assert_eq!(snapped(&paths, "open mod dot rs")[0].1, "mod.rs");
}

#[test]
fn names_split_into_parts_worth_favouring() {
    assert_eq!(pieces("SpecialWords.rs"), ["Special", "Words", "rs"]);
    assert_eq!(pieces("tdt_beam"), ["tdt", "beam"]);
    assert_eq!(pieces("v2"), ["v", "2"]);
    assert_eq!(pieces("README.md"), ["README", "md"]);
    let paths = Paths::from_list(["src/spoken.rs", "tests/spoken.rs", "src/tdt_beam.rs"]);
    let words = paths.words();
    assert_eq!(words[0], "spoken");
    assert!(words.contains(&"beam".to_string()));
    assert!(words.contains(&"src".to_string()));
    assert!(!words.contains(&"rs".to_string()));
}

#[test]
fn gitignore_globs_match_as_git_does() {
    for (pattern, text, want) in [
        ("*.log", "build.log", true),
        ("*.log", "logs/build.log", false),
        ("target", "target", true),
        ("**/node_modules", "a/b/node_modules", true),
        ("**/node_modules", "node_modules", true),
        ("docs/**", "docs/a/b.md", true),
        ("a/**/b", "a/b", true),
        ("a/**/b", "a/x/y/b", true),
        ("[abc].txt", "b.txt", true),
        ("[!abc].txt", "b.txt", false),
        ("[a-c]x", "bx", true),
        ("?.rs", "a.rs", true),
        ("?.rs", "ab.rs", false),
        ("\\#notes", "#notes", true),
    ] {
        assert_eq!(
            glob(pattern.as_bytes(), text.as_bytes()),
            want,
            "{pattern} {text}"
        );
    }
    let rules = vec![
        Rules::parse("target/\n*.log\n!keep.log\n/local\n# a comment\n\n", ""),
        Rules::parse("*.tmp\n", "sub"),
    ];
    let all = [0, 1];
    for (path, folder, want) in [
        ("target", true, true),
        ("target", false, false),
        ("a/build.log", false, true),
        ("keep.log", false, false),
        ("local", true, true),
        ("a/local", true, false),
        ("sub/x.tmp", false, true),
        ("x.tmp", false, false),
    ] {
        assert_eq!(ignored(&rules, &all, path, folder), want, "{path}");
    }
}

#[test]
fn a_scan_lists_names_breadth_first_and_keeps_to_its_limits() {
    let root = std::env::temp_dir().join(format!("telmo-paths-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    for dir in [
        "src/biasing",
        "target/debug",
        ".git/objects",
        ".cache/x",
        "docs",
    ] {
        std::fs::create_dir_all(root.join(dir)).unwrap();
    }
    for file in [
        "src/engine.rs",
        "src/biasing/spoken.rs",
        "target/debug/app",
        ".git/HEAD",
        ".cache/x/y",
        "docs/notes.md",
        "docs/draft.log",
        "Cargo.toml",
    ] {
        std::fs::write(root.join(file), "").unwrap();
    }
    std::fs::write(root.join(".gitignore"), "target/\n*.log\n").unwrap();
    let paths = Paths::scan(&root, &PathLimits::default()).unwrap();
    let listed: Vec<&str> = paths.iter().collect();
    assert_eq!(
        listed,
        [
            ".gitignore",
            "Cargo.toml",
            "docs",
            "src",
            "docs/notes.md",
            "src/biasing",
            "src/engine.rs",
            "src/biasing/spoken.rs"
        ]
    );
    assert!(paths.is_complete());
    let small = PathLimits {
        max_entries: 3,
        ..PathLimits::default()
    };
    let paths = Paths::scan(&root, &small).unwrap();
    assert_eq!(paths.len(), 3);
    assert!(!paths.is_complete());
    let shallow = PathLimits {
        max_depth: 0,
        hidden_folders: true,
        gitignore: false,
        ..PathLimits::default()
    };
    let listed: Vec<String> = (Paths::scan(&root, &shallow).unwrap().iter())
        .map(String::from)
        .collect();
    assert!(listed.contains(&".cache".to_string()));
    assert!(listed.contains(&"target".to_string()));
    assert!(!listed.iter().any(|p| p.contains('/') || p == ".git"));
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(root.join("src"), root.join("link")).unwrap();
        let paths = Paths::scan(&root, &PathLimits::default()).unwrap();
        assert!(paths.iter().any(|p| p == "link"));
        assert!(!paths.iter().any(|p| p.starts_with("link/")));
    }
    assert!(Paths::scan(&root.join("missing"), &PathLimits::default()).is_err());
    std::fs::remove_dir_all(&root).unwrap();
}
