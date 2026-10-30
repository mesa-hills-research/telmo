//! Voice activity detection by the energy of 10 ms frames of 16 kHz audio.
//!
//! Each frame's energy is compared with a noise floor that falls quickly to quieter frames and
//! rises slowly toward louder ones, so it follows the quiet moments between words. Speech
//! starts once the energy stays above a start threshold for 150 ms, and the utterance's audio
//! begins 300 ms before that onset. It ends once the energy stays below a lower end threshold
//! for 600 ms, at the end of the last frame above it. An utterance that reaches 30 s is cut at
//! the quietest frame of its last 2 s and the next one starts there. [`VadConfig`] holds these
//! settings.
//!
//! The events count samples from the first one pushed, and they are the same whatever the
//! sizes of the chunks the audio comes in.

use std::collections::VecDeque;

/// The sample rate the detector takes.
pub const SAMPLE_RATE: u32 = 16000;
/// Samples in a frame, 10 ms.
pub const FRAME: usize = 160;

/// Settings of [`Vad`]. Energies are in dB relative to a full-scale square wave. Durations are
/// in milliseconds, and all but the pre-roll are rounded to whole frames.
#[derive(Debug, Clone, PartialEq)]
pub struct VadConfig {
    /// How far above the noise floor a frame starts speech.
    pub start_db: f64,
    /// How far above the noise floor a frame keeps speech going. Lower than `start_db`.
    pub end_db: f64,
    /// The lowest the noise floor goes. The thresholds sit above it, so near-silence never
    /// counts as speech.
    pub min_floor_db: f64,
    /// How fast the noise floor rises toward louder frames, in dB per second.
    pub floor_rise_db_per_s: f64,
    /// The time constant of the noise floor's fall toward quieter frames.
    pub floor_fall_ms: u32,
    /// How long frames stay above the start threshold before speech starts.
    pub onset_ms: u32,
    /// Audio kept from before the onset.
    pub pre_roll_ms: u32,
    /// How long frames stay below the end threshold before speech ends.
    pub hangover_ms: u32,
    /// The longest utterance, counted from its first sample.
    pub max_utterance_ms: u32,
    /// How far back from the end of an utterance that's too long to look for a quiet frame to
    /// cut it at.
    pub cut_search_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        VadConfig {
            start_db: 9.0,
            end_db: 5.0,
            min_floor_db: -70.0,
            floor_rise_db_per_s: 3.0,
            floor_fall_ms: 50,
            onset_ms: 150,
            pre_roll_ms: 300,
            hangover_ms: 600,
            max_utterance_ms: 30_000,
            cut_search_ms: 2_000,
        }
    }
}

/// What the detector found, at sample indices counted from the first sample pushed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    /// An utterance started. Its audio begins at `audio_from`, the onset less the pre-roll
    /// (never before the previous utterance's end), and its speech at `onset`, the first
    /// sample of the first loud frame.
    Start { audio_from: u64, onset: u64 },
    /// The utterance ended at `speech_end`, and the detector decided at `decided_at`, the end
    /// of the frame that ended it. Normally that's `hangover_ms` after the end of the last
    /// loud frame, which is `speech_end`. `cut` is true when the utterance reached
    /// `max_utterance_ms` instead. It then ends at the middle of its quietest frame of the last
    /// `cut_search_ms`, where the next utterance starts, or at the end of its last loud frame
    /// if that quiet frame lies in the pause going on.
    End {
        speech_end: u64,
        decided_at: u64,
        cut: bool,
    },
}

#[derive(Debug, Clone)]
enum State {
    /// No speech, with `run` frames in a row above the start threshold.
    Quiet { run: usize },
    /// An utterance whose audio began at `from`, with `quiet` frames in a row below the end
    /// threshold since `last_loud_end`.
    Speech {
        from: u64,
        last_loud_end: u64,
        quiet: usize,
    },
}

/// An energy-based voice activity detector for 16 kHz mono audio.
#[derive(Debug, Clone)]
pub struct Vad {
    config: VadConfig,
    onset_frames: usize,
    pre_roll: u64,
    hangover_frames: usize,
    max_len: u64,
    search_frames: usize,
    rise_per_frame: f64,
    /// The share of the gap the floor falls by in one frame.
    fall: f64,
    /// The sum of squares of the frame being filled, and its sample count.
    frame_sum: f64,
    frame_fill: usize,
    /// Complete frames so far.
    frames: u64,
    floor: Option<f64>,
    state: State,
    /// Energies of the latest frames, the newest last, for choosing where to cut.
    recent: VecDeque<f64>,
    previous_end: u64,
}

impl Vad {
    pub fn new(config: VadConfig) -> Vad {
        let frames = |ms: u32| ((ms as usize + 5) / 10).max(1);
        let fall = if config.floor_fall_ms == 0 {
            1.0
        } else {
            1.0 - (-10.0 / config.floor_fall_ms as f64).exp()
        };
        Vad {
            onset_frames: frames(config.onset_ms),
            pre_roll: config.pre_roll_ms as u64 * (SAMPLE_RATE as u64 / 1000),
            hangover_frames: frames(config.hangover_ms),
            max_len: (frames(config.max_utterance_ms) * FRAME) as u64,
            search_frames: frames(config.cut_search_ms),
            rise_per_frame: config.floor_rise_db_per_s / 100.0,
            fall,
            frame_sum: 0.0,
            frame_fill: 0,
            frames: 0,
            floor: None,
            state: State::Quiet { run: 0 },
            recent: VecDeque::new(),
            previous_end: 0,
            config,
        }
    }

    pub fn config(&self) -> &VadConfig {
        &self.config
    }

    /// Takes the next samples of the stream and returns the events they complete.
    pub fn push(&mut self, samples: &[f32]) -> Vec<VadEvent> {
        let mut events = Vec::new();
        for &s in samples {
            self.frame_sum += s as f64 * s as f64;
            self.frame_fill += 1;
            if self.frame_fill == FRAME {
                self.end_frame(&mut events);
            }
        }
        events
    }

    /// Whether an utterance is going on.
    pub fn in_speech(&self) -> bool {
        matches!(self.state, State::Speech { .. })
    }

    /// The current noise floor in dB.
    pub fn noise_floor_db(&self) -> f64 {
        self.floor.unwrap_or(self.config.min_floor_db)
    }

    /// The samples pushed so far.
    pub fn samples(&self) -> u64 {
        self.frames * FRAME as u64 + self.frame_fill as u64
    }

    /// The first sample a later event can refer to. Audio before it belongs to no utterance
    /// to come, so it can be dropped.
    pub fn keep_from(&self) -> u64 {
        match self.state {
            State::Speech { from, .. } => from,
            State::Quiet { run } => {
                let onset = (self.frames - run as u64) * FRAME as u64;
                onset.saturating_sub(self.pre_roll).max(self.previous_end)
            }
        }
    }

    fn end_frame(&mut self, events: &mut Vec<VadEvent>) {
        let c = &self.config;
        let energy = 10.0 * (self.frame_sum / FRAME as f64 + 1e-20).log10();
        self.frame_sum = 0.0;
        self.frame_fill = 0;
        self.frames += 1;
        let end = self.frames * FRAME as u64;
        let level = energy.max(c.min_floor_db);
        let floor = *self.floor.get_or_insert(level);
        let above_start = level > floor + c.start_db;
        let above_end = level > floor + c.end_db;
        self.recent.push_back(energy);
        if self.recent.len() > self.search_frames {
            self.recent.pop_front();
        }

        match &mut self.state {
            State::Quiet { run } => {
                *run = if above_start { *run + 1 } else { 0 };
                if *run >= self.onset_frames {
                    let onset = end - (*run * FRAME) as u64;
                    let audio_from = onset.saturating_sub(self.pre_roll).max(self.previous_end);
                    events.push(VadEvent::Start { audio_from, onset });
                    self.state = State::Speech {
                        from: audio_from,
                        last_loud_end: end,
                        quiet: 0,
                    };
                }
            }
            State::Speech {
                from,
                last_loud_end,
                quiet,
            } => {
                if above_end {
                    *last_loud_end = end;
                    *quiet = 0;
                } else {
                    *quiet += 1;
                }
                if *quiet >= self.hangover_frames {
                    events.push(VadEvent::End {
                        speech_end: *last_loud_end,
                        decided_at: end,
                        cut: false,
                    });
                    self.previous_end = *last_loud_end;
                    self.state = State::Quiet { run: 0 };
                } else if end - *from >= self.max_len {
                    let quietest = quietest_frame(&self.recent, self.frames, *from);
                    if quietest * FRAME as u64 >= *last_loud_end {
                        // The quietest frame is in the pause going on now, so the utterance
                        // ends where its speech did.
                        events.push(VadEvent::End {
                            speech_end: *last_loud_end,
                            decided_at: end,
                            cut: true,
                        });
                        self.previous_end = *last_loud_end;
                        self.state = State::Quiet { run: 0 };
                    } else {
                        let cut = quietest * FRAME as u64 + FRAME as u64 / 2;
                        events.push(VadEvent::End {
                            speech_end: cut,
                            decided_at: end,
                            cut: true,
                        });
                        events.push(VadEvent::Start {
                            audio_from: cut,
                            onset: cut,
                        });
                        self.previous_end = cut;
                        *from = cut;
                    }
                }
            }
        }

        let next = if level < floor {
            floor + (level - floor) * self.fall
        } else {
            (floor + self.rise_per_frame).min(level)
        };
        self.floor = Some(next.max(c.min_floor_db));
    }
}

/// The index of the quietest of the `recent` frames (the earliest if several are) that lies
/// wholly after sample `from`. `frames` is the count of frames so far.
fn quietest_frame(recent: &VecDeque<f64>, frames: u64, from: u64) -> u64 {
    let first = frames - recent.len() as u64;
    let earliest = from.div_ceil(FRAME as u64);
    let mut best = (frames - 1, f64::MAX);
    for (i, &e) in recent.iter().enumerate() {
        let index = first + i as u64;
        if index >= earliest && e < best.1 {
            best = (index, e);
        }
    }
    best.0
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: usize = SAMPLE_RATE as usize;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        /// Roughly Gaussian noise with unit variance.
        fn gaussian(&mut self) -> f64 {
            ((0..4).map(|_| self.next()).sum::<f64>() - 2.0) * 3f64.sqrt()
        }
    }

    fn amplitude(db: f64) -> f64 {
        10f64.powf(db / 20.0)
    }

    /// Gaussian noise whose power is `db` dB.
    fn noise(len: usize, db: f64, seed: u64) -> Vec<f32> {
        let mut rng = Rng(seed);
        let a = amplitude(db);
        (0..len).map(|_| (rng.gaussian() * a) as f32).collect()
    }

    #[derive(Clone, Copy)]
    enum Kind {
        Tone,
        Noise,
    }

    /// Adds a burst of `kind` with power `db` dB to `signal` over `range`.
    fn add_burst(signal: &mut [f32], range: std::ops::Range<usize>, kind: Kind, db: f64) {
        let mut rng = Rng(range.start as u64 * 7919 + 1);
        let a = amplitude(db);
        for i in range {
            let t = i as f64 / RATE as f64;
            let x = match kind {
                // Three harmonics of 180 Hz with a total power of 1.
                Kind::Tone => {
                    (0..3)
                        .map(|h| (2.0 * std::f64::consts::PI * 180.0 * (h + 1) as f64 * t).sin())
                        .sum::<f64>()
                        * (2.0f64 / 3.0).sqrt()
                }
                Kind::Noise => rng.gaussian(),
            };
            signal[i] += (x * a) as f32;
        }
    }

    fn run(config: VadConfig, signal: &[f32]) -> Vec<VadEvent> {
        Vad::new(config).push(signal)
    }

    /// The utterances as (audio_from, onset, speech_end, decided_at).
    fn utterances(events: &[VadEvent]) -> Vec<(u64, u64, u64, u64)> {
        let mut out = Vec::new();
        let mut open = None;
        for event in events {
            match *event {
                VadEvent::Start { audio_from, onset } => {
                    assert!(open.is_none(), "two starts in a row: {events:?}");
                    open = Some((audio_from, onset));
                }
                VadEvent::End {
                    speech_end,
                    decided_at,
                    ..
                } => {
                    let (from, onset) = open.take().expect("an end without a start");
                    assert!(from <= onset && onset <= speech_end && speech_end <= decided_at);
                    out.push((from, onset, speech_end, decided_at));
                }
            }
        }
        assert!(open.is_none(), "an utterance never ended: {events:?}");
        out
    }

    const NOISE_DB: f64 = -50.0;

    #[test]
    fn bursts_are_found_within_a_frame_or_two() {
        let mut worst = (0i64, 0i64);
        for kind in [Kind::Tone, Kind::Noise] {
            for snr in [30.0, 20.0, 12.0] {
                let mut signal = noise(14 * RATE, NOISE_DB, 5);
                // Bursts of 0.3 to 2 s that start and end inside frames, 2 s apart.
                let bursts = [(1.0031, 0.3), (3.5077, 1.0), (6.7113, 2.0), (10.9001, 0.7)];
                let bursts: Vec<(usize, usize)> = bursts
                    .iter()
                    .map(|&(at, len)| ((at * RATE as f64) as usize, (len * RATE as f64) as usize))
                    .collect();
                for &(at, len) in &bursts {
                    add_burst(&mut signal, at..at + len, kind, NOISE_DB + snr);
                }
                let events = run(VadConfig::default(), &signal);
                assert!(
                    !events
                        .iter()
                        .any(|e| matches!(e, VadEvent::End { cut: true, .. }))
                );
                let found = utterances(&events);
                assert_eq!(found.len(), bursts.len(), "SNR {snr}: {found:?}");
                for (&(from, onset, end, decided), &(at, len)) in found.iter().zip(&bursts) {
                    let onset_error = onset as i64 - at as i64;
                    let end_error = end as i64 - (at + len) as i64;
                    worst.0 = worst.0.max(onset_error.abs());
                    worst.1 = worst.1.max(end_error.abs());
                    assert!(
                        onset_error.abs() <= 2 * FRAME as i64,
                        "SNR {snr}: onset {onset_error}"
                    );
                    assert!(
                        end_error.abs() <= 2 * FRAME as i64,
                        "SNR {snr}: end {end_error}"
                    );
                    assert_eq!(from, onset - 300 * 16);
                    assert_eq!(decided, end + 600 * 16);
                }
            }
        }
        eprintln!(
            "largest errors: onset {} samples, end {} samples",
            worst.0, worst.1
        );
    }

    #[test]
    fn noise_alone_never_starts_speech() {
        // Steady noise, noise whose level drifts by 6 dB over seconds, and near-silence with a
        // faint tone.
        let steady = noise(60 * RATE, NOISE_DB, 11);
        assert_eq!(run(VadConfig::default(), &steady), []);
        let mut drifting = noise(60 * RATE, NOISE_DB, 12);
        for (i, x) in drifting.iter_mut().enumerate() {
            let t = i as f64 / RATE as f64;
            *x *= amplitude(3.0 * (2.0 * std::f64::consts::PI * t / 7.0).sin()) as f32;
        }
        assert_eq!(run(VadConfig::default(), &drifting), []);
        let mut quiet = noise(20 * RATE, -96.0, 13);
        add_burst(&mut quiet, 5 * RATE..8 * RATE, Kind::Tone, -75.0);
        let mut vad = Vad::new(VadConfig::default());
        assert_eq!(vad.push(&quiet), []);
        assert_eq!(vad.noise_floor_db(), -70.0);
        assert_eq!(run(VadConfig::default(), &vec![0.0; 5 * RATE]), []);
    }

    #[test]
    fn the_hangover_bridges_short_gaps() {
        let mut signal = noise(12 * RATE, NOISE_DB, 21);
        // Gaps of 0.3 s and 0.55 s inside the first utterance, then 0.8 s before the second.
        for (from, to) in [(1000, 2000), (2300, 3000), (3550, 4000), (4800, 5500)] {
            add_burst(&mut signal, from * 16..to * 16, Kind::Tone, NOISE_DB + 20.0);
        }
        let found = utterances(&run(VadConfig::default(), &signal));
        let ms: Vec<(u64, u64)> = found.iter().map(|u| (u.1 / 16, u.2 / 16)).collect();
        assert_eq!(ms, [(1000, 4000), (4800, 5500)]);
    }

    /// Words of 400 ms with gaps of 150 ms, 20 dB above the noise, from `from` to `to` ms.
    fn words(signal: &mut [f32], from: usize, to: usize) {
        for start in (from..to).step_by(550) {
            let end = (start + 400).min(to);
            add_burst(signal, start * 16..end * 16, Kind::Noise, NOISE_DB + 20.0);
        }
    }

    #[test]
    fn a_long_utterance_is_cut_at_its_quietest_frame() {
        let mut signal = noise(48 * RATE, NOISE_DB, 31);
        words(&mut signal, 1000, 45_000);
        // The utterance's audio starts at 0.7 s, so it reaches 30 s at 30.7 s. The quietest
        // frame of the 2 s before is this silent one at 29.5 s.
        let quiet = 2950 * FRAME;
        signal[quiet..quiet + FRAME].fill(0.0);
        let events = run(VadConfig::default(), &signal);
        let cut = (quiet + FRAME / 2) as u64;
        assert_eq!(events.len(), 4, "{events:?}");
        assert_eq!(
            events[1],
            VadEvent::End {
                speech_end: cut,
                decided_at: 30_700 * 16,
                cut: true,
            }
        );
        assert_eq!(
            events[2],
            VadEvent::Start {
                audio_from: cut,
                onset: cut
            }
        );
        assert!(matches!(events[3], VadEvent::End { cut: false, .. }));
        let found = utterances(&events);
        assert_eq!(found[0].0, 700 * 16);
        assert_eq!(found[1].2, 44_850 * 16);
    }

    #[test]
    fn a_long_utterance_in_a_pause_ends_where_its_speech_did() {
        let mut signal = noise(48 * RATE, NOISE_DB, 41);
        words(&mut signal, 1000, 30_300);
        words(&mut signal, 30_800, 40_000);
        // The pause from 30.3 s holds the quietest frames, 6 dB below the noise.
        for x in &mut signal[30_300 * 16..30_800 * 16] {
            *x *= 0.5;
        }
        let events = run(VadConfig::default(), &signal);
        assert!(matches!(events[1], VadEvent::End { cut: true, .. }));
        let found = utterances(&events);
        assert_eq!(found.len(), 2, "{found:?}");
        assert_eq!(found[0], (700 * 16, 1000 * 16, 30_300 * 16, 30_700 * 16));
        // The next utterance's pre-roll stops at the end of the first.
        assert_eq!((found[1].0, found[1].1), (30_500 * 16, 30_800 * 16));
    }

    #[test]
    fn events_are_the_same_whatever_the_chunks() {
        let mut signal = noise(50 * RATE, NOISE_DB, 51);
        words(&mut signal, 500, 38_000);
        add_burst(
            &mut signal,
            40 * RATE + 77..43 * RATE,
            Kind::Tone,
            NOISE_DB + 15.0,
        );
        let expected = run(VadConfig::default(), &signal);
        assert!(expected.len() >= 6, "{expected:?}");
        let mut rng = Rng(77);
        let mut vad = Vad::new(VadConfig::default());
        let mut got = Vec::new();
        let mut pos = 0;
        while pos < signal.len() {
            let size = ((rng.next() * 3000.0) as usize).min(signal.len() - pos);
            let keep = vad.keep_from();
            let events = vad.push(&signal[pos..pos + size]);
            pos += size;
            for event in &events {
                match *event {
                    VadEvent::Start { audio_from, .. } => assert!(audio_from >= keep),
                    VadEvent::End { decided_at, .. } => assert!(decided_at <= pos as u64),
                }
            }
            got.extend(events);
        }
        assert_eq!(got, expected);
        assert_eq!(vad.samples(), signal.len() as u64);
    }

    /// The span of the frames of `samples` louder than `db`, as sample indices.
    fn loud_span(samples: &[f32], db: f64) -> (usize, usize) {
        let loud: Vec<usize> = samples
            .chunks_exact(FRAME)
            .enumerate()
            .filter(|(_, f)| {
                let power = f.iter().map(|&x| x as f64 * x as f64).sum::<f64>() / FRAME as f64;
                10.0 * power.log10() > db
            })
            .map(|(i, _)| i)
            .collect();
        (loud[0] * FRAME, (loud[loud.len() - 1] + 1) * FRAME)
    }

    /// The golden clips, joined with 1.5 s between them and around them, over continuous
    /// noise at -55 dB, which is louder than the quietest clips' own background.
    #[test]
    fn real_utterances_are_found_one_by_one() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/goldens/parakeet-tdt-0.6b-v3");
        if !dir.exists() {
            eprintln!(
                "The golden files aren't in this copy of the source, so this test is skipped."
            );
            return;
        }
        let clips = [
            "clean-1995-1826-0007",
            "clean-260-123286-0018",
            "clean-3575-170457-0009",
            "other-367-130732-0011",
            "other-4350-9170-0030",
        ];
        let gap = 3 * RATE / 2;
        let mut signal = Vec::new();
        // Each clip's span in the joined audio, and the span of its frames above -35 dB and
        // above -45 dB, 20 dB and 10 dB over the added noise.
        let mut spans = Vec::new();
        for clip in clips {
            let audio = crate::wav::read(&dir.join(format!("{clip}.wav"))).unwrap();
            signal.resize(signal.len() + gap, 0.0);
            let at = signal.len();
            let shift = |(a, b): (usize, usize)| (at + a, at + b);
            spans.push((
                (at, at + audio.samples.len()),
                shift(loud_span(&audio.samples, -35.0)),
                shift(loud_span(&audio.samples, -45.0)),
            ));
            signal.extend_from_slice(&audio.samples);
        }
        signal.resize(signal.len() + gap, 0.0);
        let background = noise(signal.len(), -55.0, 61);
        for (x, n) in signal.iter_mut().zip(background) {
            *x += n;
        }
        let found = utterances(&run(VadConfig::default(), &signal));
        assert_eq!(found.len(), clips.len(), "{found:?}");
        let ms = |a: usize, b: u64| (b as i64 - a as i64) / 16;
        for ((clip, &(from, onset, end, _)), &(clip_span, loud, speech)) in
            clips.iter().zip(&found).zip(&spans)
        {
            eprintln!(
                "{clip}: onset {} ms after the clip's start and {} ms after its first frame \
                 above -45 dB, end {} ms after its last frame above -45 dB",
                ms(clip_span.0, onset),
                ms(speech.0, onset),
                ms(speech.1, end),
            );
            // The utterance sits in its clip and holds all of its loud speech.
            assert!(onset >= clip_span.0 as u64 && end <= (clip_span.1 + FRAME) as u64);
            assert!(from <= loud.0 as u64 && end >= loud.1 as u64, "{clip}");
            assert!(ms(clip_span.0, onset) <= 700, "{clip}");
            assert!(ms(clip_span.1, end) >= -500, "{clip}");
            assert_eq!(from + 300 * 16, onset);
        }
    }

    /// Joins the utterances of LibriSpeech lists (`id<TAB>wav<TAB>text`) with 1.5 s between
    /// them over noise at -55 dB, and counts the source utterances found as one, as several
    /// (split at a pause longer than the hangover or cut for length) and as none (merged into
    /// the one before or missed).
    ///
    /// ```text
    /// LIBRISPEECH_LISTS=test-clean.tsv,test-other.tsv cargo test --lib vad -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs LibriSpeech, named by LIBRISPEECH_LISTS"]
    fn librispeech_utterances_are_found_one_by_one() {
        let lists = std::env::var("LIBRISPEECH_LISTS").expect("LIBRISPEECH_LISTS");
        for list in lists.split(',') {
            let text = std::fs::read_to_string(list).unwrap();
            let gap = 3 * RATE / 2;
            let mut signal = Vec::new();
            let mut spans = Vec::new();
            for line in text.lines() {
                let wav = line.split('\t').nth(1).expect("a WAV path");
                let audio = crate::wav::read(wav.as_ref()).unwrap();
                signal.resize(signal.len() + gap, 0.0);
                spans.push((
                    signal.len() as u64,
                    (signal.len() + audio.samples.len()) as u64,
                ));
                signal.extend_from_slice(&audio.samples);
            }
            signal.resize(signal.len() + gap, 0.0);
            let background = noise(signal.len(), -55.0, 71);
            for (x, n) in signal.iter_mut().zip(background) {
                *x += n;
            }
            let found = utterances(&run(VadConfig::default(), &signal));
            let (mut one, mut several, mut none, mut long) = (0, 0, 0, 0);
            let mut lead = Vec::new();
            let mut tail = Vec::new();
            for &(start, end) in &spans {
                let inside: Vec<_> = found
                    .iter()
                    .filter(|u| u.1 + FRAME as u64 > start && u.1 < end + FRAME as u64)
                    .collect();
                match inside.len() {
                    0 => none += 1,
                    1 => {
                        one += 1;
                        lead.push((inside[0].1 as i64 - start as i64) / 16);
                        tail.push(end as i64 / 16 - inside[0].2 as i64 / 16);
                    }
                    _ if end - start > 30 * RATE as u64 => long += 1,
                    _ => several += 1,
                }
            }
            lead.sort();
            tail.sort();
            eprintln!(
                "{list}: {} utterances, {one} found as one, {several} as several, {long} cut for \
                 length, {none} not found on their own. Onsets {} ms after the clip's start \
                 (median), ends {} ms before the clip's end (median).",
                spans.len(),
                lead[lead.len() / 2],
                tail[tail.len() / 2],
            );
            // A split is a pause longer than the hangover, which LibriSpeech's utterances often
            // hold. A source utterance with no onset of its own was missed or merged.
            assert!(none * 200 <= spans.len(), "{none} not found");
        }
    }
}
