//! `telmo record` walks through the numbered prompts of a text file and saves a WAV file for
//! each one. The session works on lines of input, so it needs no terminal library.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use crate::capture::{Capture, Refused};
use crate::mic;

/// One numbered prompt.
#[derive(Debug, PartialEq)]
pub struct Item {
    /// The heading of the part the prompt is under, empty if none.
    pub part: String,
    pub number: u32,
    pub prompt: String,
    /// The recording's file name without `.wav`.
    pub stem: String,
}

/// Reads the numbered items (`1. text` or `1) text`) of a text or Markdown file. A heading
/// that starts with `Part N` starts a part, and an item belongs to the part above it.
///
/// A recording is named by its item's number, padded with zeros to at least two digits. When
/// several parts share numbers, a later part's files start with `p` and the part's number, as
/// in `p3-01`. A part may choose its own start by giving a file name in backticks before its
/// first item, such as `t01.m4a` for `t`.
pub fn parse_prompts(text: &str) -> Result<Vec<Item>, String> {
    struct Found {
        part: u32,
        title: String,
        prefix: Option<String>,
        number: u32,
        prompt: String,
    }
    let (mut part, mut title, mut prefix) = (0, String::new(), None);
    let mut found: Vec<Found> = Vec::new();
    let mut in_part = false;
    for line in text.lines() {
        if let Some(heading) = line.strip_prefix('#') {
            let heading = heading.trim_start_matches('#').trim();
            if let Some(n) = heading.strip_prefix("Part ").and_then(leading_number) {
                (part, title, prefix, in_part) = (n, heading.to_string(), None, false);
            }
        } else if let Some((number, prompt)) = numbered(line) {
            in_part = true;
            found.push(Found {
                part,
                title: title.clone(),
                prefix: prefix.clone(),
                number,
                prompt: prompt.to_string(),
            });
        } else if line.starts_with(char::is_whitespace) && !line.trim().is_empty() && in_part {
            if let Some(last) = found.last_mut() {
                last.prompt = format!("{} {}", last.prompt, line.trim());
            }
        } else if !in_part && prefix.is_none() {
            prefix = file_prefix(line);
        }
    }
    let Some(largest) = found.iter().map(|f| f.number).max() else {
        return Err("The prompts file has no numbered items.".into());
    };
    let width = largest.to_string().len().max(2);
    let mut items: Vec<Item> = Vec::new();
    for (i, f) in found.iter().enumerate() {
        let shared = found[..i]
            .iter()
            .any(|o| o.part != f.part && o.number == f.number);
        let start = match (&f.prefix, shared) {
            (Some(prefix), _) => prefix.clone(),
            (None, true) => format!("p{}-", f.part),
            (None, false) => String::new(),
        };
        let stem = format!("{start}{:0width$}", f.number);
        if let Some(twin) = items.iter().find(|o| o.stem == stem) {
            return Err(format!(
                "Two items would be saved as {stem}.wav, \"{}\" and \"{}\".",
                twin.prompt, f.prompt
            ));
        }
        items.push(Item {
            part: f.title.clone(),
            number: f.number,
            prompt: f.prompt.clone(),
            stem,
        });
    }
    Ok(items)
}

/// The digits a text starts with.
fn leading_number(text: &str) -> Option<u32> {
    let end = text
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(text.len());
    text[..end].parse().ok()
}

/// A line such as `12. Say this`, as its number and its text.
fn numbered(line: &str) -> Option<(u32, &str)> {
    let line = line.trim_start();
    let number = leading_number(line)?;
    let rest = line.trim_start_matches(|c: char| c.is_ascii_digit());
    let rest = rest.strip_prefix(['.', ')'])?;
    let prompt = rest.strip_prefix(char::is_whitespace)?.trim();
    (!prompt.is_empty()).then_some((number, prompt))
}

/// The letters of a file name such as `t01.m4a` written in backticks in a line.
fn file_prefix(line: &str) -> Option<String> {
    line.split('`').skip(1).step_by(2).find_map(|code| {
        let (name, extension) = code.rsplit_once('.')?;
        let digits = name.trim_start_matches(|c: char| c.is_ascii_alphabetic());
        let letters = &name[..name.len() - digits.len()];
        let valid = !letters.is_empty()
            && !digits.is_empty()
            && digits.bytes().all(|b| b.is_ascii_digit())
            && !extension.is_empty()
            && extension.bytes().all(|b| b.is_ascii_alphanumeric());
        valid.then(|| letters.to_lowercase())
    })
}

/// The items that have no WAV file in `out_dir` yet.
pub fn pending<'a>(items: &'a [Item], out_dir: &Path) -> Vec<&'a Item> {
    items
        .iter()
        .filter(|item| !wav_path(out_dir, item).exists())
        .collect()
}

fn wav_path(out_dir: &Path, item: &Item) -> PathBuf {
    out_dir.join(format!("{}.wav", item.stem))
}

/// The index of the device that `pick` names, by its 1-based number, its whole name or a part
/// of it, ignoring case.
#[cfg_attr(not(feature = "record"), allow(dead_code))] // Only the microphone calls it.
pub fn pick_device(names: &[String], pick: &str) -> Result<usize, String> {
    if let Ok(number) = pick.parse::<usize>() {
        return if (1..=names.len()).contains(&number) {
            Ok(number - 1)
        } else {
            Err(format!("There is no input device number {number}."))
        };
    }
    let pick = pick.to_lowercase();
    let matching = |whole: bool| -> Vec<usize> {
        (0..names.len())
            .filter(|&i| {
                let name = names[i].to_lowercase();
                if whole {
                    name == pick
                } else {
                    name.contains(&pick)
                }
            })
            .collect()
    };
    let exact = matching(true);
    let found = if exact.is_empty() {
        matching(false)
    } else {
        exact
    };
    match found[..] {
        [one] => Ok(one),
        [] => Err(format!(
            "No input device matches {pick}. Run telmo record --device list to see them."
        )),
        _ => Err(format!(
            "More than one input device matches {pick}. Use its number from telmo record --device list."
        )),
    }
}

/// Prints the input devices, numbered.
pub fn list_devices() -> Result<(), String> {
    for (i, (name, is_default)) in mic::devices()?.iter().enumerate() {
        let default = if *is_default { "  (default)" } else { "" };
        println!("{:>3}  {name}{default}", i + 1);
    }
    Ok(())
}

/// A peak below this, about -50 dB, is a microphone that heard nothing.
const SILENT: f32 = 0.003;
const METER_WIDTH: usize = 24;

/// A bar for a peak level, from -60 dB (empty) to 0 dB (full).
fn meter(level: f32) -> String {
    let db = 20.0 * level.max(1e-6).log10();
    let filled = ((db + 60.0) / 60.0).clamp(0.0, 1.0) * METER_WIDTH as f32;
    let filled = filled.round() as usize;
    format!(
        "[{}{}]",
        "#".repeat(filled),
        "-".repeat(METER_WIDTH - filled)
    )
}

fn clock(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    format!("{}:{:02}", seconds / 60, seconds % 60)
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum Key {
    Enter,
    Redo,
    Skip,
    Quit,
    Other,
}

impl Key {
    fn from_line(line: &str) -> Key {
        match line.trim().to_lowercase().as_str() {
            "" => Key::Enter,
            "r" => Key::Redo,
            "s" => Key::Skip,
            "q" => Key::Quit,
            _ => Key::Other,
        }
    }
}

/// Shows `prompt` and waits for a line that is one of `allowed`. Closed input quits.
fn ask(lines: &Receiver<String>, prompt: &str, allowed: &[Key]) -> Key {
    loop {
        print!("{prompt} ");
        let _ = std::io::stdout().flush();
        match lines.recv().map(|line| Key::from_line(&line)) {
            Ok(key) if allowed.contains(&key) => return key,
            Ok(_) => {}
            Err(_) => return Key::Quit,
        }
    }
}

/// Records until a line arrives, showing the time and level. `None` if the input closed.
/// A take that lost samples or went through a device failure is refused, even when the line
/// arrived first.
fn record_until_enter(
    capture: &Capture,
    lines: &Receiver<String>,
) -> Result<Option<Vec<f32>>, Refused> {
    capture.start()?;
    let (start, mut peak) = (Instant::now(), 0.0_f32);
    loop {
        match lines.recv_timeout(Duration::from_millis(100)) {
            Ok(_) => return capture.stop().map(Some),
            Err(RecvTimeoutError::Disconnected) => return Ok(None),
            Err(RecvTimeoutError::Timeout) => {}
        }
        capture.poll()?;
        let level = capture.level();
        peak = peak.max(level);
        let elapsed = start.elapsed();
        let hint = if elapsed > Duration::from_secs(2) && peak < SILENT {
            "Hearing nothing. Enter stops"
        } else {
            "Enter stops"
        };
        print!(
            "\rRecording {}  {}  {hint:<30}",
            clock(elapsed),
            meter(level)
        );
        let _ = std::io::stdout().flush();
    }
}

/// Writes the item's WAV file, whole or not at all.
fn save(out_dir: &Path, item: &Item, samples: &[f32], sample_rate: u32) -> Result<PathBuf, String> {
    let path = wav_path(out_dir, item);
    let partial = path.with_extension("wav.partial");
    let bytes = telmo::wav::encode_pcm16(samples, sample_rate);
    std::fs::write(&partial, bytes)
        .and_then(|()| std::fs::rename(&partial, &path))
        .map_err(|e| format!("Couldn't save {} ({e}).", path.display()))?;
    Ok(path)
}

/// Runs the session. It asks about each item that has no file yet, in order.
pub fn run(prompts: &Path, out_dir: &Path, device: Option<&str>) -> Result<(), String> {
    let text = std::fs::read_to_string(prompts)
        .map_err(|e| format!("Couldn't read {} ({e}).", prompts.display()))?;
    let items = parse_prompts(&text)?;
    std::fs::create_dir_all(out_dir)
        .map_err(|e| format!("Couldn't create {} ({e}).", out_dir.display()))?;
    let todo = pending(&items, out_dir);
    if todo.is_empty() {
        println!(
            "All {} items are recorded in {}.",
            items.len(),
            out_dir.display()
        );
        return Ok(());
    }
    let mic = mic::open(device)?;
    println!(
        "Recording from {} at {} Hz. {} of {} items are left, saved as WAV files in {}.",
        mic.name,
        mic.sample_rate,
        todo.len(),
        items.len(),
        out_dir.display()
    );
    println!(
        "Enter starts and stops a recording. After it, Enter keeps it, r redoes it, s skips it and q quits."
    );

    let (send, lines) = mpsc::channel();
    std::thread::spawn(move || {
        for line in std::io::stdin().lines().map_while(Result::ok) {
            if send.send(line).is_err() {
                break;
            }
        }
    });

    let (mut saved, mut skipped, mut part) = (0, 0, None);
    'items: for (n, item) in todo.iter().enumerate() {
        if part != Some(&item.part) {
            part = Some(&item.part);
            if !item.part.is_empty() {
                println!("\n{}", item.part);
            }
        }
        println!("\n{}. {}\n", item.number, item.prompt);
        let start = format!(
            "[{}/{}] Enter records, s skips, q quits.",
            n + 1,
            todo.len()
        );
        loop {
            match ask(&lines, &start, &[Key::Enter, Key::Skip, Key::Quit]) {
                Key::Skip => {
                    skipped += 1;
                    continue 'items;
                }
                Key::Quit => break 'items,
                _ => {}
            }
            let samples = match record_until_enter(&mic.capture, &lines) {
                Ok(Some(samples)) => samples,
                Ok(None) => break 'items,
                Err(Refused::Failed(reason)) => {
                    println!();
                    return Err(format!(
                        "The microphone stopped ({reason}). The items saved so far are kept."
                    ));
                }
                Err(Refused::Dropped) => {
                    println!("\nThe computer fell behind and part of the recording is missing.");
                    continue;
                }
            };
            let seconds = samples.len() as f64 / f64::from(mic.sample_rate);
            let peak = samples.iter().fold(0.0_f32, |m, s| m.max(s.abs()));
            let silent = peak < SILENT;
            let (message, allowed): (&str, &[Key]) = if silent {
                println!(
                    "Recorded {seconds:.1} s but heard nothing. Check the microphone and its volume."
                );
                (
                    "Enter redoes it, s skips it, q quits.",
                    &[Key::Enter, Key::Skip, Key::Quit],
                )
            } else {
                println!("Recorded {seconds:.1} s.");
                (
                    "Enter keeps it, r redoes it, s skips it, q quits.",
                    &[Key::Enter, Key::Redo, Key::Skip, Key::Quit],
                )
            };
            match (ask(&lines, message, allowed), silent) {
                (Key::Enter, false) => {
                    let path = save(out_dir, item, &samples, mic.sample_rate)?;
                    println!("Saved {}.", path.display());
                    saved += 1;
                    continue 'items;
                }
                (Key::Enter | Key::Redo, _) => {}
                (Key::Skip, _) => {
                    skipped += 1;
                    continue 'items;
                }
                _ => break 'items,
            }
        }
    }
    println!("\nSaved {saved}, skipped {skipped}.");
    if saved < todo.len() {
        println!("Run the same command again to continue with the rest.");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    const SAMPLE: &str = "\
# Recording script

How to record

- Record in a quiet room. Name files such as 01.m4a.

## Part 1, dictate freely

1. Ask an agent about the build.
2. Describe a bug,
   and how to reproduce it.

## Part 2, read as written

3. I had had enough.
4) No, no, don't merge it yet.

## Part 3, think out loud

Name each file `t01.m4a` to `t02.m4a`.

1. Plan the next piece of work.
2. Walk through a bug.

### How they are checked

Text with `sets/dictation-seed.txt`.
";

    fn stems(items: &[Item]) -> Vec<&str> {
        items.iter().map(|i| i.stem.as_str()).collect()
    }

    #[test]
    fn prompts_keep_their_number_part_and_text() {
        let items = parse_prompts(SAMPLE).unwrap();
        assert_eq!(items.len(), 6);
        assert_eq!(items[0].number, 1);
        assert_eq!(items[0].part, "Part 1, dictate freely");
        assert_eq!(items[0].prompt, "Ask an agent about the build.");
        assert_eq!(items[1].prompt, "Describe a bug, and how to reproduce it.");
        assert_eq!(items[3].part, "Part 2, read as written");
        assert_eq!(items[3].prompt, "No, no, don't merge it yet.");
        assert_eq!(items[5].part, "Part 3, think out loud");
    }

    #[test]
    fn files_are_named_by_number_and_a_part_that_repeats_numbers_names_its_own() {
        let items = parse_prompts(SAMPLE).unwrap();
        assert_eq!(stems(&items), ["01", "02", "03", "04", "t01", "t02"]);
    }

    #[test]
    fn a_repeated_number_without_a_chosen_name_gets_the_part_number() {
        let text = "## Part 1\n1. a\n2. b\n## Part 2\n1. c\n";
        assert_eq!(stems(&parse_prompts(text).unwrap()), ["01", "02", "p2-01"]);
    }

    #[test]
    fn numbers_are_padded_to_the_largest() {
        let items = parse_prompts("1. a\n120. b\n").unwrap();
        assert_eq!(stems(&items), ["001", "120"]);
        assert_eq!(items[0].part, "");
    }

    #[test]
    fn bad_prompt_files_say_what_is_wrong() {
        assert_eq!(
            parse_prompts("Nothing numbered.\n1.5 is a number\n"),
            Err("The prompts file has no numbered items.".into())
        );
        let error = parse_prompts("1. a\n1. b\n").unwrap_err();
        assert!(error.contains("01.wav"), "{error}");
    }

    #[test]
    fn items_with_a_file_are_done() {
        let dir = crate::tests::temp_dir("record-resume");
        let items = parse_prompts("1. a\n2. b\n3. c\n").unwrap();
        assert_eq!(pending(&items, &dir).len(), 3);
        std::fs::write(dir.join("01.wav"), b"x").unwrap();
        std::fs::write(dir.join("03.wav"), b"x").unwrap();
        let left = pending(&items, &dir);
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].stem, "02");
    }

    #[test]
    fn a_saved_recording_is_16_bit_mono_and_reads_back() {
        let dir = crate::tests::temp_dir("record-save");
        let item = &parse_prompts("7. a\n").unwrap()[0];
        let samples = [0.0, 0.5, -0.5, 1.0, -1.0, 2.0];
        let path = save(&dir, item, &samples, 44100).unwrap();
        assert_eq!(path, dir.join("07.wav"));
        assert!(!dir.join("07.wav.partial").exists());
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 44 + samples.len() * 2);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..16], b"WAVEfmt ");
        assert_eq!(u16::from_le_bytes([bytes[20], bytes[21]]), 1, "PCM");
        assert_eq!(u16::from_le_bytes([bytes[22], bytes[23]]), 1, "mono");
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 44100);
        assert_eq!(u16::from_le_bytes([bytes[34], bytes[35]]), 16, "bits");
        assert_eq!(&bytes[36..40], b"data");
        let audio = telmo::wav::parse(&bytes).unwrap();
        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(audio.samples.len(), samples.len());
        assert_eq!(audio.samples[1], 0.5);
        assert_eq!(audio.samples[4], -1.0);
        assert_eq!(audio.samples[5], 32767.0 / 32768.0, "clipped, not wrapped");
    }

    #[test]
    fn a_device_is_picked_by_number_or_name() {
        let names: Vec<String> = [
            "default",
            "USB Microphone",
            "Built-in Mic",
            "Built-in Mic 2",
        ]
        .map(String::from)
        .to_vec();
        assert_eq!(pick_device(&names, "2"), Ok(1));
        assert_eq!(pick_device(&names, "usb"), Ok(1));
        assert_eq!(
            pick_device(&names, "built-in mic"),
            Ok(2),
            "the whole name wins"
        );
        assert!(pick_device(&names, "0").is_err());
        assert!(pick_device(&names, "5").is_err());
        assert!(
            pick_device(&names, "headset")
                .unwrap_err()
                .starts_with("No input device matches")
        );
        assert!(
            pick_device(&names, "built")
                .unwrap_err()
                .starts_with("More than one")
        );
    }

    #[test]
    fn the_meter_fills_with_level_and_silence_is_below_the_floor() {
        assert_eq!(meter(0.0), format!("[{}]", "-".repeat(METER_WIDTH)));
        assert_eq!(meter(1.0), format!("[{}]", "#".repeat(METER_WIDTH)));
        assert!(meter(0.1).matches('#').count() > meter(0.01).matches('#').count());
        assert_eq!(clock(Duration::from_secs(75)), "1:15");
    }

    #[test]
    fn a_stop_that_arrives_with_a_device_failure_keeps_nothing() {
        let capture = Arc::new(Capture::new());
        let (send, lines) = mpsc::channel();
        // The stop line is waiting when the device fails, so the first thing the take
        // sees is the line.
        let failing = Arc::clone(&capture);
        let device = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            failing.push([0.5; 4800].into_iter(), 1);
            failing.fail("device disconnected".into());
            send.send(String::new()).unwrap();
        });
        let taken = record_until_enter(&capture, &lines);
        device.join().unwrap();
        assert_eq!(taken, Err(Refused::Failed("device disconnected".into())));
        let (send, lines) = mpsc::channel();
        send.send(String::new()).unwrap();
        assert_eq!(
            record_until_enter(&capture, &lines),
            Err(Refused::Failed("device disconnected".into())),
            "a device that failed earlier starts no take"
        );
    }

    #[test]
    fn a_stop_ends_a_healthy_take_with_its_audio() {
        let capture = Arc::new(Capture::new());
        let (send, lines) = mpsc::channel();
        let device = {
            let capture = Arc::clone(&capture);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(20));
                capture.push([0.5; 480].into_iter(), 1);
                send.send(String::new()).unwrap();
            })
        };
        assert_eq!(
            record_until_enter(&capture, &lines),
            Ok(Some(vec![0.5; 480]))
        );
        device.join().unwrap();
        assert_eq!(
            record_until_enter(&capture, &lines),
            Ok(None),
            "input closed"
        );
    }

    #[test]
    fn keys_are_few_and_plain() {
        for (line, key) in [
            ("", Key::Enter),
            ("  ", Key::Enter),
            ("r", Key::Redo),
            ("S", Key::Skip),
            ("q\n", Key::Quit),
            ("redo", Key::Other),
        ] {
            assert_eq!(Key::from_line(line), key, "{line:?}");
        }
    }
}
