//! Measures streaming recognition on long recordings replayed as they would arrive from a
//! microphone.
//!
//! ```text
//! cargo run --release --example stream -- sessions OUT_DIR LIST.tsv [--seconds 120] [--count 10] [--seed 1]
//! cargo run --release --example stream -- run MODEL SESSIONS.tsv [--realtime] [--threads N] [--interval S] [--factor F] [--margin S] [--context S] [--input-rate HZ] [--piece-ms MS] [--basic] [--out DIR] [--limit N]
//! ```
//!
//! `sessions` joins consecutive utterances of a list (`id<TAB>wav<TAB>text`, 16 kHz WAVs, the
//! same chapter for LibriSpeech ids) into recordings of about `--seconds`, with pauses of 0.8 to
//! 2 s between them and quiet noise throughout. It writes the WAVs and `sessions.tsv`
//! (`id<TAB>wav<TAB>reference<TAB>source wavs`).
//!
//! `run` streams each recording through the model in pieces of `--piece-ms` (20 ms by default)
//! at `--input-rate` (48 kHz by default, resampled from the file). With `--realtime` the pieces
//! arrive at the pace of real audio and the stream runs on its own thread. Without it, a
//! simulated clock advances with the audio and by the measured duration of each pass, which
//! gives the same timing without waiting. It reports, over all utterances, the time from the
//! onset of speech to the first tentative word, the time from the end of speech to the final
//! text, how many tentative words were replaced before settling, the word error rate of the
//! streamed text and of offline transcription of the same audio, and the CPU used.

#[path = "../wer/basic.rs"]
#[allow(dead_code)]
mod basic;
#[path = "../wer/normalize.rs"]
#[allow(dead_code)]
mod normalize;
#[path = "../wer/score.rs"]
#[allow(dead_code)]
mod score;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use telmo::stream::{Event, SAMPLE_RATE, Stream, StreamConfig, Streamer};
use telmo::{Model, Options, Precision};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("sessions") => sessions(&args[1..]),
        Some("run") => run(&args[1..]),
        _ => {
            Err("Usage: stream sessions OUT_DIR LIST.tsv... | stream run MODEL SESSIONS.tsv".into())
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

/// A small deterministic random number generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Roughly normal noise.
    fn noise(&mut self) -> f32 {
        ((0..4).map(|_| self.next()).sum::<f64>() - 2.0) as f32
    }
}

fn sessions(args: &[String]) -> Result<(), String> {
    let mut seconds = 120.0;
    let mut count = 10;
    let mut seed = 1u64;
    let mut positional = Vec::new();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = || it.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--seconds" => seconds = value()?.parse().map_err(|_| "--seconds takes a number")?,
            "--count" => count = value()?.parse().map_err(|_| "--count takes a number")?,
            "--seed" => seed = value()?.parse().map_err(|_| "--seed takes a number")?,
            _ => positional.push(PathBuf::from(arg)),
        }
    }
    let Some((out_dir, lists)) = positional.split_first() else {
        return Err("sessions takes OUT_DIR and lists".into());
    };
    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let mut rows: Vec<(String, PathBuf, String)> = Vec::new();
    for list in lists {
        let text = std::fs::read_to_string(list).map_err(|e| format!("{}: {e}", list.display()))?;
        for line in text.lines().filter(|l| !l.is_empty()) {
            let mut f = line.splitn(3, '\t');
            if let (Some(id), Some(wav), Some(reference)) = (f.next(), f.next(), f.next()) {
                rows.push((id.to_string(), wav.into(), reference.to_string()));
            }
        }
    }
    // Groups of consecutive rows from one chapter (the id up to its last dash), or of any rows
    // when ids have no chapters.
    let chapter = |id: &str| {
        id.rsplit_once('-')
            .map_or(String::new(), |(c, _)| c.to_string())
    };
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        match groups.last_mut() {
            Some(g) if chapter(&rows[g[0]].0) == chapter(&row.0) => g.push(i),
            _ => groups.push(vec![i]),
        }
    }
    let step = (groups.len() / count.max(1)).max(1);
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut index = String::new();
    for (made, group) in groups.iter().step_by(step).take(count).enumerate() {
        let mut audio: Vec<f32> = Vec::new();
        let mut references = Vec::new();
        let mut sources = Vec::new();
        let pause = |audio: &mut Vec<f32>, s: f64| {
            audio.extend(std::iter::repeat_n(0.0, (s * SAMPLE_RATE as f64) as usize))
        };
        pause(&mut audio, 1.0);
        for &i in group {
            let (_, wav, reference) = &rows[i];
            let clip = telmo::wav::read(wav).map_err(|e| e.to_string())?;
            if clip.sample_rate != SAMPLE_RATE as u32 {
                return Err(format!("{} isn't 16 kHz", wav.display()));
            }
            audio.extend_from_slice(&clip.samples);
            references.push(reference.clone());
            sources.push(wav.display().to_string());
            if audio.len() as f64 >= seconds * SAMPLE_RATE as f64 {
                break;
            }
            let gap = 0.8 + 1.2 * rng.next();
            pause(&mut audio, gap);
        }
        pause(&mut audio, 1.5);
        // Quiet noise throughout, about 60 dB below full scale.
        for s in &mut audio {
            *s += 0.001 * rng.noise();
        }
        let id = format!("session-{made:02}");
        let path = out_dir.join(format!("{id}.wav"));
        std::fs::write(&path, telmo::wav::encode_pcm16(&audio, SAMPLE_RATE as u32))
            .map_err(|e| e.to_string())?;
        index.push_str(&format!(
            "{id}\t{}\t{}\t{}\n",
            std::fs::canonicalize(&path)
                .map_err(|e| e.to_string())?
                .display(),
            references.join(" "),
            sources.join(",")
        ));
        eprintln!(
            "{id}: {:.0} s from {} utterances",
            audio.len() as f64 / SAMPLE_RATE as f64,
            sources.len()
        );
    }
    std::fs::write(out_dir.join("sessions.tsv"), index).map_err(|e| e.to_string())
}

struct Session {
    id: String,
    wav: PathBuf,
    reference: String,
    sources: Vec<PathBuf>,
}

fn read_sessions(path: &Path) -> Result<Vec<Session>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 4 {
                return Err(format!(
                    "{}: a line has fewer than four fields",
                    path.display()
                ));
            }
            Ok(Session {
                id: f[0].into(),
                wav: f[1].into(),
                reference: f[2].into(),
                sources: f[3].split(',').map(PathBuf::from).collect(),
            })
        })
        .collect()
}

/// An event and when it came, in seconds of audio from the start.
type Timed = (f64, Event);

/// A replay's events, and its passes' count, total seconds and longest seconds.
type Replay = (Vec<Timed>, (usize, f64, f64));

/// One utterance's timing.
#[derive(Debug, Default, Clone)]
struct Utterance {
    onset: f64,
    first_text: Option<f64>,
    final_text: Option<f64>,
}

#[derive(Default)]
struct Totals {
    utterances: Vec<Utterance>,
    replaced: usize,
    settled_words: usize,
    streamed: score::Errors,
    offline: score::Errors,
    against_offline: score::Errors,
    /// Offline transcription of the utterances the stream found.
    segmented: score::Errors,
    against_segmented: score::Errors,
    audio: f64,
    cpu: f64,
    passes: usize,
    busy: f64,
    longest: f64,
}

fn run(args: &[String]) -> Result<(), String> {
    let mut options = Options::default();
    let mut config = StreamConfig::default();
    let mut realtime = false;
    let mut input_rate = 48000u32;
    let mut piece_ms = 20u64;
    let mut basic = false;
    let mut out: Option<PathBuf> = None;
    let mut limit = usize::MAX;
    let mut positional = Vec::new();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = || it.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--realtime" => realtime = true,
            "--basic" => basic = true,
            "--threads" => {
                options.threads = value()?.parse().map_err(|_| "--threads takes a number")?
            }
            "--precision" => {
                options.precision = Precision::from_name(value()?)
                    .ok_or("--precision takes f32, f16, bf16, int8 or q8_0")?
            }
            "--interval" => {
                config.min_interval = Duration::from_secs_f64(
                    value()?.parse().map_err(|_| "--interval takes seconds")?,
                )
            }
            "--margin" => {
                config.passes.settle_margin =
                    Duration::from_secs_f64(value()?.parse().map_err(|_| "--margin takes seconds")?)
            }
            "--context" => {
                config.passes.context = Duration::from_secs_f64(
                    value()?.parse().map_err(|_| "--context takes seconds")?,
                )
            }
            "--factor" => {
                config.interval_factor = value()?.parse().map_err(|_| "--factor takes a number")?
            }
            "--input-rate" => {
                input_rate = value()?
                    .parse()
                    .map_err(|_| "--input-rate takes a number")?
            }
            "--piece-ms" => piece_ms = value()?.parse().map_err(|_| "--piece-ms takes a number")?,
            "--out" => out = Some(value()?.into()),
            "--limit" => limit = value()?.parse().map_err(|_| "--limit takes a number")?,
            _ => positional.push(PathBuf::from(arg)),
        }
    }
    let [model_path, list] = positional.as_slice() else {
        return Err("run takes MODEL and SESSIONS.tsv".into());
    };
    config.input_rate = input_rate;
    let model = Arc::new(Model::load(model_path, &options).map_err(|e| e.to_string())?);
    eprintln!(
        "{} with {} weights, {} threads, passes at least {:.2} s and {:.1} times their length apart, {} input at {} Hz in {} ms pieces",
        model.name(),
        model.precision().map_or("mixed", |p| p.name()),
        model.threads(),
        config.min_interval.as_secs_f64(),
        config.interval_factor,
        if realtime {
            "real-time"
        } else {
            "simulated real-time"
        },
        input_rate,
        piece_ms
    );
    let normalize = |t: &str| {
        if basic {
            basic::normalize(t)
        } else {
            normalize::normalize_version(t, normalize::Version::V2025)
        }
    };
    let mut totals = Totals::default();
    for session in read_sessions(list)?.into_iter().take(limit) {
        let clip = telmo::wav::read(&session.wav).map_err(|e| e.to_string())?;
        let audio = telmo::resample::resample(&clip.samples, clip.sample_rate, input_rate);
        let seconds = audio.len() as f64 / input_rate as f64;
        let piece = (input_rate as u64 * piece_ms / 1000) as usize;
        let cpu0 = cpu_seconds();
        let (events, stats) = if realtime {
            replay_realtime(&model, &config, &audio, input_rate, piece)?
        } else {
            replay_simulated(&model, &config, &audio, input_rate, piece)?
        };
        let cpu = cpu_seconds() - cpu0;
        let (text, utterances, replaced, settled_words) = summarize(&events);

        // Offline transcription of the same audio, utterance by utterance as recorded.
        let mut offline = Vec::new();
        for source in &session.sources {
            let a = telmo::wav::read(source).map_err(|e| e.to_string())?;
            offline.push(
                model
                    .transcribe(&a.samples)
                    .map_err(|e| e.to_string())?
                    .text,
            );
        }
        let offline = offline.join(" ");
        // Offline transcription of the utterances the stream found, to tell what splitting at
        // pauses costs from what passes over partial audio cost.
        let mut segmented = Vec::new();
        for (from, to) in segments(&events) {
            let (a, b) = (
                (from * SAMPLE_RATE as f64) as usize,
                ((to * SAMPLE_RATE as f64) as usize).min(clip.samples.len()),
            );
            if b > a + SAMPLE_RATE / 10 {
                segmented.push(
                    model
                        .transcribe(&clip.samples[a..b])
                        .map_err(|e| e.to_string())?
                        .text,
                );
            }
        }
        let segmented = segmented.join(" ");
        let (r, s, o) = (
            normalize(&session.reference),
            normalize(&text),
            normalize(&offline),
        );
        let g = normalize(&segmented);
        let streamed = score::count_errors(&r, &s);
        let off = score::count_errors(&r, &o);
        let seg = score::count_errors(&r, &g);
        let vs_seg = score::count_errors(&g, &s);
        let vs = score::count_errors(&o, &s);
        let delays: Vec<f64> = utterances
            .iter()
            .filter_map(|u| u.first_text.map(|t| t - u.onset))
            .collect();
        eprintln!(
            "{}: {:.0} s, {} utterances, WER streamed {:.2}% offline {:.2}%, streamed against offline {:.2}%, first word after {:.2} s (median), {} passes, CPU {:.2} cores",
            session.id,
            seconds,
            utterances.len(),
            100.0 * streamed.wer(),
            100.0 * off.wer(),
            100.0 * vs.wer(),
            median(&delays),
            stats.0,
            cpu / seconds
        );
        if let Some(dir) = &out {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            let mut log = String::new();
            for (t, e) in &events {
                log.push_str(&format!("{t:.3}\t{e:?}\n"));
            }
            std::fs::write(dir.join(format!("{}.events", session.id)), log)
                .map_err(|e| e.to_string())?;
            std::fs::write(
                dir.join(format!("{}.offline.txt", session.id)),
                format!("{offline}\n"),
            )
            .map_err(|e| e.to_string())?;
            std::fs::write(dir.join(format!("{}.txt", session.id)), format!("{text}\n"))
                .map_err(|e| e.to_string())?;
        }
        totals.utterances.extend(utterances);
        totals.replaced += replaced;
        totals.settled_words += settled_words;
        totals.streamed += streamed;
        totals.offline += off;
        totals.against_offline += vs;
        totals.segmented += seg;
        totals.against_segmented += vs_seg;
        totals.audio += seconds;
        totals.cpu += cpu;
        totals.passes += stats.0;
        totals.busy += stats.1;
        totals.longest = totals.longest.max(stats.2);
    }
    report(&totals);
    Ok(())
}

/// Feeds the audio at its real pace to a [`Stream`] on its own thread.
fn replay_realtime(
    model: &Arc<Model>,
    config: &StreamConfig,
    audio: &[f32],
    rate: u32,
    piece: usize,
) -> Result<Replay, String> {
    let events: Arc<Mutex<Vec<Timed>>> = Arc::default();
    let start = Instant::now();
    let sink = events.clone();
    let stream = Stream::start(model.clone(), config.clone(), move |e| {
        sink.lock()
            .unwrap()
            .push((start.elapsed().as_secs_f64(), e));
    });
    let mut fed = 0;
    for chunk in audio.chunks(piece) {
        fed += chunk.len();
        let due = start + Duration::from_secs_f64(fed as f64 / rate as f64);
        if let Some(wait) = due.checked_duration_since(Instant::now()) {
            std::thread::sleep(wait);
        }
        stream.push(chunk);
    }
    let stats = stream.finish().map_err(|e| e.to_string())?;
    let events = std::mem::take(&mut *events.lock().unwrap());
    Ok((
        events,
        (
            stats.passes,
            stats.busy.as_secs_f64(),
            stats.longest.as_secs_f64(),
        ),
    ))
}

/// Runs a [`Streamer`] on a simulated clock that advances with the audio and with the measured
/// duration of each pass, scheduling passes the way [`Stream`] does.
fn replay_simulated(
    model: &Arc<Model>,
    config: &StreamConfig,
    audio: &[f32],
    rate: u32,
    piece: usize,
) -> Result<Replay, String> {
    let mut streamer = Streamer::new(model.clone(), config.clone());
    let mut events: Vec<Timed> = Vec::new();
    let mut clock = 0.0f64;
    let mut fed = 0usize;
    let mut next_pass: Option<f64> = None;
    let (mut passes, mut busy, mut longest) = (0, 0.0f64, 0.0f64);
    let min = config.min_interval.as_secs_f64();
    loop {
        // Everything that arrived by now.
        while fed < audio.len()
            && (fed + piece.min(audio.len() - fed)) as f64 / rate as f64 <= clock + 1e-9
        {
            let end = (fed + piece).min(audio.len());
            let at = end as f64 / rate as f64;
            for e in streamer.push(&audio[fed..end]) {
                events.push((at, e));
            }
            fed = end;
        }
        if !streamer.in_utterance() {
            next_pass = None;
        }
        let due = if streamer.in_utterance() {
            Some(*next_pass.get_or_insert(clock + min))
        } else {
            None
        };
        if streamer.wants_final()
            || due.is_some_and(|d| clock >= d - 1e-9 && streamer.has_new_audio())
        {
            let start = Instant::now();
            let out = streamer.pass().map_err(|e| e.to_string())?;
            let took = start.elapsed().as_secs_f64();
            let begun = clock;
            clock += took;
            events.extend(out.into_iter().map(|e| (clock, e)));
            passes += 1;
            busy += took;
            longest = longest.max(took);
            next_pass = Some(begun + min.max(took * config.interval_factor));
            continue;
        }
        if fed >= audio.len() {
            break;
        }
        // Wait for the next piece or the next pass, whichever comes first.
        let next_piece = (fed + piece).min(audio.len()) as f64 / rate as f64;
        clock = match due {
            Some(d) if d > clock && d < next_piece => d,
            _ => next_piece.max(clock),
        };
    }
    let start = Instant::now();
    let out = streamer.finish().map_err(|e| e.to_string())?;
    clock += start.elapsed().as_secs_f64();
    events.extend(out.into_iter().map(|e| (clock, e)));
    Ok((events, (passes, busy, longest)))
}

/// The settled text, each utterance's timing, the tentative words replaced before settling,
/// and the settled words.
fn summarize(events: &[Timed]) -> (String, Vec<Utterance>, usize, usize) {
    let mut text = String::new();
    let mut utterances: Vec<Utterance> = Vec::new();
    // The tentative words shown, and the words settled since they were shown.
    let mut shown: Vec<String> = Vec::new();
    let mut settled_since: Vec<String> = Vec::new();
    let mut replaced = 0;
    let mut settled_words = 0;
    let words = |t: &str| t.split_whitespace().map(String::from).collect::<Vec<_>>();
    let mut compare =
        |shown: &mut Vec<String>, settled_since: &mut Vec<String>, now: Vec<String>| {
            let after: Vec<&String> = settled_since.iter().chain(now.iter()).collect();
            replaced += shown
                .iter()
                .enumerate()
                .filter(|(i, w)| after.get(*i) != Some(w))
                .count();
            *shown = now;
            settled_since.clear();
        };
    for (t, e) in events {
        match e {
            Event::SpeechStart { utterance, at } => {
                let i = *utterance as usize;
                utterances.resize(utterances.len().max(i + 1), Utterance::default());
                utterances[i].onset = *at;
            }
            Event::Settled { utterance, text: s } => {
                text.push_str(s);
                settled_words += s.split_whitespace().count();
                settled_since.extend(words(s));
                if let Some(u) = utterances.get_mut(*utterance as usize) {
                    u.first_text.get_or_insert(*t);
                }
            }
            Event::Tentative { utterance, text: s } => {
                if let Some(u) = utterances.get_mut(*utterance as usize)
                    && !s.trim().is_empty()
                {
                    u.first_text.get_or_insert(*t);
                }
                compare(&mut shown, &mut settled_since, words(s));
            }
            Event::UtteranceEnd {
                utterance,
                speech_end,
                ..
            } => {
                compare(&mut shown, &mut settled_since, Vec::new());
                if let Some(u) = utterances.get_mut(*utterance as usize) {
                    u.final_text = Some(t - speech_end);
                }
            }
        }
    }
    (text.trim().to_string(), utterances, replaced, settled_words)
}

/// The audio of each utterance a stream found, in seconds: from 300 ms before its onset (but
/// not before the previous one's end) to where its end was decided.
fn segments(events: &[Timed]) -> Vec<(f64, f64)> {
    let mut onsets: Vec<Option<f64>> = Vec::new();
    let mut out = Vec::new();
    let mut previous_end = 0.0f64;
    for (_, e) in events {
        match e {
            Event::SpeechStart { utterance, at } => {
                let i = *utterance as usize;
                onsets.resize(onsets.len().max(i + 1), None);
                onsets[i] = Some(*at);
            }
            Event::UtteranceEnd {
                utterance,
                at,
                speech_end,
            } => {
                if let Some(Some(onset)) = onsets.get(*utterance as usize) {
                    out.push(((onset - 0.3f64).max(previous_end), *at));
                }
                previous_end = *speech_end;
            }
            _ => {}
        }
    }
    out
}

fn median(v: &[f64]) -> f64 {
    quantile(v, 0.5)
}

fn quantile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn report(t: &Totals) {
    let first: Vec<f64> = t
        .utterances
        .iter()
        .filter_map(|u| u.first_text.map(|x| x - u.onset))
        .collect();
    let last: Vec<f64> = t.utterances.iter().filter_map(|u| u.final_text).collect();
    println!(
        "{} utterances over {:.1} min of audio",
        t.utterances.len(),
        t.audio / 60.0
    );
    println!(
        "first tentative word after the onset of speech: median {:.2} s, 90th percentile {:.2} s",
        median(&first),
        quantile(&first, 0.9)
    );
    println!(
        "final text after the end of speech: median {:.2} s, 90th percentile {:.2} s",
        median(&last),
        quantile(&last, 0.9)
    );
    println!(
        "tentative words replaced before settling: {} per 100 settled words",
        format_args!(
            "{:.1}",
            100.0 * t.replaced as f64 / t.settled_words.max(1) as f64
        )
    );
    println!(
        "WER: streamed {:.2}%, offline {:.2}%, streamed against offline {:.2}% ({} reference words)",
        100.0 * t.streamed.wer(),
        100.0 * t.offline.wer(),
        100.0 * t.against_offline.wer(),
        t.streamed.reference_words
    );
    println!(
        "WER of offline transcription of the streamed utterances {:.2}%, streamed against it {:.2}%",
        100.0 * t.segmented.wer(),
        100.0 * t.against_segmented.wer()
    );
    println!(
        "passes: {}, {:.2} s each on average, longest {:.2} s, busy {:.0}% of the time; CPU {:.2} cores on average",
        t.passes,
        t.busy / t.passes.max(1) as f64,
        t.longest,
        100.0 * t.busy / t.audio,
        t.cpu / t.audio
    );
}

/// The process's CPU time in seconds, on Linux.
fn cpu_seconds() -> f64 {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return 0.0;
    };
    // Fields after the command name, which is in parentheses: utime and stime are the 12th and
    // 13th, in clock ticks of 1/100 s.
    let rest = stat.rsplit_once(')').map_or("", |(_, r)| r);
    let f: Vec<&str> = rest.split_whitespace().collect();
    let ticks = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    (ticks(11) + ticks(12)) / 100.0
}
