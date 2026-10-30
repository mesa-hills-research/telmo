//! Streaming recognition of one utterance at a time, as audio arrives.
//!
//! Neither model is built for streaming, so each pass recomputes part of the utterance and
//! decodes from where the settled text ends. Parakeet re-encodes a window made of the unsettled
//! audio and up to a few seconds of settled audio before it, and continues greedy decoding from
//! the decoder state saved after the last settled token. Granite recomputes only its current
//! attention block and decodes CTC frames from the last settled one.
//!
//! A word settles once two passes in a row agree on it, together with every word before it, and
//! it ends at least half a second before the audio does. Settled words never change. When the
//! utterance ends, one last pass settles everything.
//!
//! [`Stream`] runs it all on a thread of its own. Audio of any sample rate goes in as it comes,
//! voice activity splits it into utterances, passes run as often as the computer keeps up with,
//! and the text comes back as [`Event`]s.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::error::Result;
use crate::features::LogMelStream;
use crate::granite::Granite;
use crate::granite_stream::GraniteStream;
use crate::model::Model;
use crate::parakeet::{DecoderState, Parakeet, Timings};
use crate::resample::Resampler;
use crate::tokenizer::Tokenizer;
use crate::vad::{Vad, VadConfig, VadEvent};

/// Samples per second the models take.
pub const SAMPLE_RATE: usize = 16000;
/// Samples per encoder frame (80 ms) of both models.
const FRAME: usize = 1280;
/// Samples per feature frame (10 ms).
const HOP: usize = 160;

/// How passes over an utterance are made and when words settle.
#[derive(Debug, Clone)]
pub struct PassConfig {
    /// Settled audio Parakeet re-encodes before the unsettled part, as context.
    pub context: Duration,
    /// How long before the end of the audio a word must end to settle.
    pub settle_margin: Duration,
    /// A pass's first words are dropped when they repeat the last settled words and end within
    /// this long after them, since they're the settled words heard again.
    pub repeat_window: Duration,
}

impl Default for PassConfig {
    fn default() -> Self {
        PassConfig {
            context: Duration::from_secs(6),
            settle_margin: Duration::from_millis(500),
            repeat_window: Duration::from_millis(500),
        }
    }
}

/// Text that changed in a pass.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PassText {
    /// Words that settled in this pass, to append after all earlier settled text.
    pub settled: String,
    /// The words after the settled ones that may still change, replacing the previous ones.
    pub tentative: String,
    /// Where the model last heard a word (the frame after its last token with a letter or a
    /// digit), in seconds from the utterance's start.
    pub last_word_end: Option<f64>,
    /// Whether the pass was final and settled everything.
    pub last: bool,
}

/// What a pass may settle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassMode {
    /// Words two passes agree on.
    Tentative,
    /// Every word. No audio may follow.
    Final,
    /// Every word if the model heard none in the audio's last stretch of this length, as words
    /// two passes agree on otherwise.
    FinalIfQuiet(Duration),
}

/// Where decoding continues after a settled token.
#[derive(Debug, Clone)]
enum Resume {
    Tdt(DecoderState),
    Ctc { frame: usize, prev: u32 },
}

impl Resume {
    fn frame(&self) -> usize {
        match self {
            Resume::Tdt(s) => s.frame,
            Resume::Ctc { frame, .. } => *frame,
        }
    }
}

/// A token of a pass's hypothesis.
#[derive(Debug, Clone)]
struct Hyp {
    id: u32,
    /// The encoder frame the token was emitted at (Parakeet) or its run starts at (Granite).
    frame: usize,
    /// The encoder frame after the token, from the utterance's start.
    end_frame: usize,
    resume: Resume,
}

/// One utterance's audio and the text settled from it so far.
pub struct UtteranceStream {
    audio: Vec<f32>,
    /// Where decoding continues after the settled tokens.
    resume: Resume,
    /// The last settled token, for deciding where words split.
    last_settled: Option<u32>,
    /// The last few settled words, normalized, to recognize them when a pass repeats them.
    recent: Vec<String>,
    /// The previous pass's unsettled words, as token ids.
    previous: Vec<Vec<u32>>,
    granite: Option<GraniteStream>,
    /// Parakeet's log-mel frames, computed once as the audio arrives.
    log_mel: Option<LogMelStream>,
    /// Whether any text settled in this utterance.
    settled_any: bool,
    passes: usize,
    pub timings: Timings,
}

impl UtteranceStream {
    pub fn new(model: &Model) -> UtteranceStream {
        let mut timings = Timings::default();
        let (resume, granite, log_mel) = match model {
            Model::Parakeet(m) => (
                Resume::Tdt(m.initial_decoder_state(&mut timings)),
                None,
                Some(m.front_end().stream()),
            ),
            Model::Granite(m) => (
                Resume::Ctc {
                    frame: 0,
                    prev: u32::MAX,
                },
                Some(GraniteStream::new(m)),
                None,
            ),
        };
        UtteranceStream {
            audio: Vec::new(),
            resume,
            last_settled: None,
            recent: Vec::new(),
            previous: Vec::new(),
            granite,
            log_mel,
            settled_any: false,
            passes: 0,
            timings,
        }
    }

    /// Appends 16 kHz audio.
    pub fn push(&mut self, samples: &[f32]) {
        self.audio.extend_from_slice(samples);
    }

    /// Drops audio past `samples`, as when the end of an utterance is found after it arrived.
    /// Removes and returns the audio past `samples`, as when the utterance's end is found after
    /// that audio arrived. Audio a pass already gave Granite's blocks stays.
    pub fn split_off(&mut self, samples: usize) -> Vec<f32> {
        let kept = (self.granite.as_ref().map(|g| g.samples()))
            .max(self.log_mel.as_ref().map(|l| l.samples()))
            .unwrap_or(0);
        self.audio
            .split_off(samples.max(kept).min(self.audio.len()))
    }

    pub fn audio(&self) -> &[f32] {
        &self.audio
    }

    /// Seconds of audio so far.
    pub fn seconds(&self) -> f64 {
        self.audio.len() as f64 / SAMPLE_RATE as f64
    }

    pub fn passes(&self) -> usize {
        self.passes
    }

    /// Runs a pass over the audio so far. A final pass settles every word, and no audio may
    /// follow it.
    pub fn pass(&mut self, model: &Model, config: &PassConfig, mode: PassMode) -> Result<PassText> {
        self.passes += 1;
        let hyp = match model {
            Model::Parakeet(m) => self.parakeet_hypothesis(m, config, mode == PassMode::Final)?,
            Model::Granite(m) => self.granite_hypothesis(m, mode == PassMode::Final)?,
        };
        let tokenizer = match model {
            Model::Parakeet(m) => m.tokenizer(),
            Model::Granite(m) => m.tokenizer(),
        };
        let hyp = self.drop_repeats(tokenizer, config, hyp);
        let words = split_words(tokenizer, self.last_settled, &hyp);
        // Where the model last heard speech: the last token with a letter or digit, since
        // punctuation often comes in the silence after the words.
        let last_word_end = hyp
            .iter()
            .rev()
            .find(|h| tokenizer.decode(&[h.id]).chars().any(char::is_alphanumeric))
            .map(|h| ((h.frame + 1) * FRAME) as f64 / SAMPLE_RATE as f64);
        let last = match mode {
            PassMode::Tentative => false,
            PassMode::Final => true,
            PassMode::FinalIfQuiet(quiet) => {
                last_word_end.is_none_or(|end| end + quiet.as_secs_f64() <= self.seconds())
            }
        };

        // Words this pass and the previous one agree on, from the start, may settle once they
        // end far enough before the audio's end.
        let limit = self
            .audio
            .len()
            .saturating_sub((config.settle_margin.as_secs_f64() * SAMPLE_RATE as f64) as usize);
        let mut settle = 0;
        if last {
            settle = words.len();
        } else {
            while settle < words.len()
                && settle < self.previous.len()
                && word_ids(&hyp, &words[settle]) == self.previous[settle]
                && hyp[words[settle].end - 1].end_frame * FRAME <= limit
            {
                settle += 1;
            }
        }
        let settled_tokens = words.get(settle.wrapping_sub(1)).map_or(0, |w| w.end);
        let settled_ids: Vec<u32> = hyp[..settled_tokens].iter().map(|h| h.id).collect();
        let tentative_ids: Vec<u32> = hyp[settled_tokens..].iter().map(|h| h.id).collect();
        if let Some(h) = hyp[..settled_tokens].last() {
            self.resume = h.resume.clone();
            self.last_settled = Some(h.id);
        }
        for w in &words[..settle] {
            self.recent.push(normalized(tokenizer, &word_ids(&hyp, w)));
        }
        let keep = self.recent.len().saturating_sub(MAX_REPEAT);
        self.recent.drain(..keep);
        self.previous = words[settle..].iter().map(|w| word_ids(&hyp, w)).collect();
        let settled = tokenizer.decode_raw(&settled_ids);
        let tentative = tokenizer.decode_raw(&tentative_ids);
        self.settled_any |= !settled.is_empty();
        Ok(PassText {
            settled,
            tentative,
            last_word_end,
            last,
        })
    }

    /// Drops what a pass hears again of the settled text. A model can place a word slightly
    /// earlier near the end of its audio than it does once more audio follows, so a settled
    /// word, or its end, may show up again just after the settled text. Leading tokens that
    /// can't start a word and spell the end of the last settled word are such ends, and leading
    /// words equal to the last settled ones that end within `config.repeat_window` of them are
    /// such words.
    fn drop_repeats(
        &self,
        tokenizer: &Tokenizer,
        config: &PassConfig,
        mut hyp: Vec<Hyp>,
    ) -> Vec<Hyp> {
        if self.last_settled.is_none() {
            return hyp;
        }
        // Tokens that continue the last settled word either repeat its end or complete it.
        let mut skip = 0;
        let mut prev = self.last_settled;
        while skip < hyp.len() && !tokenizer.may_split_before(prev, hyp[skip].id) {
            prev = Some(hyp[skip].id);
            skip += 1;
        }
        let ids: Vec<u32> = hyp[..skip].iter().map(|h| h.id).collect();
        let repeats_end = self
            .recent
            .last()
            .is_some_and(|w| w.ends_with(&normalized(tokenizer, &ids)));
        if !repeats_end {
            skip = 0;
            prev = self.last_settled;
        }
        let words = split_words(tokenizer, prev, &hyp[skip..]);
        let window = (config.repeat_window.as_secs_f64() * SAMPLE_RATE as f64 / FRAME as f64).ceil()
            as usize;
        let limit = self.resume.frame() + window;
        for n in (1..=MAX_REPEAT.min(words.len()).min(self.recent.len())).rev() {
            let tail = &self.recent[self.recent.len() - n..];
            let same = words[..n]
                .iter()
                .zip(tail)
                .all(|(w, t)| normalized(tokenizer, &word_ids(&hyp[skip..], w)) == *t);
            if same && hyp[skip + words[n - 1].end - 1].end_frame <= limit {
                skip += words[n - 1].end;
                break;
            }
        }
        hyp.drain(..skip);
        hyp
    }

    /// Re-encodes the unsettled audio with up to `config.context` of settled audio before it,
    /// and decodes from the state saved after the last settled token. The window's features
    /// are normalized over the window, as NVIDIA's buffered streaming does.
    fn parakeet_hypothesis(
        &mut self,
        model: &Parakeet,
        config: &PassConfig,
        finished: bool,
    ) -> Result<Vec<Hyp>> {
        let Resume::Tdt(state) = &self.resume else {
            unreachable!("Parakeet resumes from a decoder state")
        };
        let log_mel = self
            .log_mel
            .as_mut()
            .expect("Parakeet keeps its log-mel frames");
        // The frames take audio only when a pass runs, so the end of an utterance can still be
        // trimmed until then.
        let timings = &mut self.timings;
        timings.time("features", || {
            log_mel.push(&self.audio[log_mel.samples().min(self.audio.len())..]);
            if finished {
                log_mel.finish();
            }
        });
        let context = (config.context.as_secs_f64() * SAMPLE_RATE as f64) as usize;
        let start = (state.frame * FRAME).saturating_sub(context) / FRAME * FRAME;
        let (first, ready) = (start / HOP, log_mel.frames());
        // The encoder needs a frame of 80 ms.
        if ready < first + FRAME / HOP {
            return Ok(Vec::new());
        }
        let features = timings.time("features", || log_mel.normalize(first..ready));
        let encoded = model.encode(&features.data, features.frames, timings, &mut |_, _, _| {});
        let tokens = model.decode_from(&encoded, start / FRAME, state, timings);
        Ok(tokens
            .into_iter()
            .map(|(t, after)| Hyp {
                id: t.id,
                frame: t.frame as usize,
                end_frame: t.frame as usize + (t.duration as usize).max(1),
                resume: Resume::Tdt(after),
            })
            .collect())
    }

    /// Encodes the new audio, recomputing only the attention blocks it reaches, and decodes
    /// CTC frames from the last settled one.
    fn granite_hypothesis(&mut self, model: &Granite, finished: bool) -> Result<Vec<Hyp>> {
        let blocks = self.granite.as_mut().expect("Granite keeps its blocks");
        // The blocks take audio only when a pass runs, so the end of an utterance can still be
        // trimmed until then.
        blocks.push(&self.audio[blocks.samples().min(self.audio.len())..]);
        blocks.update(model, finished, &mut self.timings);
        let Resume::Ctc { frame, prev } = self.resume else {
            unreachable!("Granite resumes from a CTC frame")
        };
        Ok(ctc_hypothesis(
            blocks.frame_ids(),
            frame,
            prev,
            model.config().blank as u32,
        ))
    }

    /// The decoder's position after the settled text, in encoder frames from the start.
    pub fn settled_frame(&self) -> usize {
        self.resume.frame()
    }
}

/// Settings of a [`Stream`].
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// The sample rate of the audio pushed.
    pub input_rate: u32,
    pub passes: PassConfig,
    pub vad: VadConfig,
    /// The shortest time from the start of one pass to the start of the next.
    pub min_interval: Duration,
    /// Passes start at least this many times the last pass's duration apart, so a slow
    /// computer shows tentative text less often instead of falling behind.
    pub interval_factor: f64,
    /// An utterance ends only once the model has heard no word in this much audio at its end,
    /// which keeps a quiet word from being cut off.
    pub endpoint_quiet: Duration,
}

impl Default for StreamConfig {
    fn default() -> Self {
        StreamConfig {
            input_rate: SAMPLE_RATE as u32,
            passes: PassConfig::default(),
            vad: VadConfig::default(),
            min_interval: Duration::from_millis(500),
            interval_factor: 1.5,
            endpoint_quiet: Duration::from_millis(400),
        }
    }
}

/// What a stream reports. The settled texts join in order, and the current tentative text
/// follows them. Times count seconds of audio from the stream's start.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Speech started at `at`.
    SpeechStart { at: f64 },
    /// The words after the settled text that may still change, replacing the earlier ones.
    Tentative { text: String },
    /// Words that won't change, to append after the earlier settled text.
    Settled { text: String },
    /// The utterance ended. Its speech ended at `speech_end`, and all its text has settled.
    UtteranceEnd { at: f64, speech_end: f64 },
}

/// An utterance being transcribed.
struct Active {
    stream: UtteranceStream,
    /// The stream's sample (16 kHz) the utterance's audio starts at.
    start: u64,
    /// Where voice activity ended it, if it has: the end of speech and where it was decided.
    end: Option<(u64, u64)>,
    /// Whether the end is a cut of an utterance too long, which needs no confirmation.
    cut: bool,
    /// Where voice activity ended speech before the model heard a word at the end, which made
    /// the utterance go on. It ends once the model hears no more words, or with voice
    /// activity's next end.
    reopened: Option<u64>,
    /// Audio after the end voice activity found, kept in case the utterance goes on.
    after_end: Vec<f32>,
    /// The utterance's samples the last pass saw.
    seen: usize,
    /// The tentative text last reported.
    tentative: String,
}

/// The synchronous core of a [`Stream`]: it takes audio, splits it into utterances and runs
/// passes when asked. [`Stream`] decides when.
pub struct Streamer {
    model: Arc<Model>,
    config: StreamConfig,
    resampler: Resampler,
    /// Resampled samples still to drop at the start, the resampler's delay.
    to_drop: usize,
    vad: Vad,
    /// The latest 16 kHz audio, for the audio before an onset.
    history: VecDeque<f32>,
    /// The stream's sample `history[0]` is.
    history_start: u64,
    /// 16 kHz samples taken so far.
    total: u64,
    active: Option<Active>,
    /// Ended utterances waiting for their final pass.
    closing: VecDeque<Active>,
    /// Whether any text has settled, so the next settled text keeps its leading space.
    wrote: bool,
}

/// Audio kept for the time before an onset, comfortably more than voice activity reaches back.
const HISTORY: usize = 3 * SAMPLE_RATE;

impl Streamer {
    pub fn new(model: Arc<Model>, config: StreamConfig) -> Streamer {
        let resampler = Resampler::new(config.input_rate, SAMPLE_RATE as u32);
        Streamer {
            to_drop: resampler.delay(),
            resampler,
            vad: Vad::new(config.vad.clone()),
            model,
            config,
            history: VecDeque::new(),
            history_start: 0,
            total: 0,
            active: None,
            closing: VecDeque::new(),
            wrote: false,
        }
    }

    pub fn config(&self) -> &StreamConfig {
        &self.config
    }

    /// Takes audio at the input rate. Returns the events of voice activity. Passes wait for
    /// [`Streamer::pass`].
    pub fn push(&mut self, input: &[f32]) -> Vec<Event> {
        let mut audio =
            Vec::with_capacity(input.len() * SAMPLE_RATE / self.config.input_rate as usize + 1);
        self.resampler.process(input, &mut audio);
        self.take(audio)
    }

    fn take(&mut self, mut audio: Vec<f32>) -> Vec<Event> {
        let drop = self.to_drop.min(audio.len());
        audio.drain(..drop);
        self.to_drop -= drop;
        let mut events = Vec::new();
        if audio.is_empty() {
            return events;
        }
        self.history.extend(audio.iter().copied());
        self.total += audio.len() as u64;
        let excess = self.history.len().saturating_sub(HISTORY);
        self.history.drain(..excess);
        self.history_start += excess as u64;
        if let Some(a) = &mut self.active {
            if a.end.is_none() {
                a.stream.push(&audio);
            } else {
                a.after_end.extend_from_slice(&audio);
            }
        }
        for event in self.vad.push(&audio) {
            match event {
                VadEvent::Start { audio_from, onset } => {
                    if let Some(a) = &self.active {
                        if a.end.is_none() && a.reopened.is_some() {
                            // The model heard more after the end, and speech went on.
                            self.active
                                .as_mut()
                                .expect("an utterance is active")
                                .reopened = None;
                            continue;
                        }
                        // The new utterance's audio may start before the old one's end, in the
                        // quiet the old one kept. The old one gives it up.
                        let mut ended = self.active.take().expect("an utterance is active");
                        ended
                            .stream
                            .split_off(audio_from.saturating_sub(ended.start) as usize);
                        self.closing.push_back(ended);
                    }
                    let mut stream = UtteranceStream::new(&self.model);
                    stream.push(&self.history_since(audio_from));
                    self.active = Some(Active {
                        stream,
                        start: audio_from.max(self.history_start),
                        end: None,
                        cut: false,
                        reopened: None,
                        after_end: Vec::new(),
                        seen: 0,
                        tentative: String::new(),
                    });
                    events.push(Event::SpeechStart {
                        at: onset as f64 / SAMPLE_RATE as f64,
                    });
                }
                VadEvent::End {
                    speech_end,
                    decided_at,
                    cut,
                } => {
                    if let Some(a) = &mut self.active {
                        a.end = Some((speech_end, decided_at));
                        a.cut = cut;
                        let mut tail = a
                            .stream
                            .split_off(decided_at.saturating_sub(a.start) as usize);
                        tail.append(&mut a.after_end);
                        a.after_end = tail;
                    }
                }
            }
        }
        events
    }

    /// The audio from the stream's sample `from` to now, as far as it's kept.
    fn history_since(&self, from: u64) -> Vec<f32> {
        let skip = from.saturating_sub(self.history_start) as usize;
        self.history.iter().skip(skip).copied().collect()
    }

    /// Whether an utterance is being transcribed or waits for its final pass.
    pub fn in_utterance(&self) -> bool {
        self.active.is_some() || !self.closing.is_empty()
    }

    /// Whether an utterance ended and waits for its final pass, which should run now.
    pub fn wants_final(&self) -> bool {
        !self.closing.is_empty() || self.active.as_ref().is_some_and(|a| a.end.is_some())
    }

    /// Whether the current utterance has audio no pass has seen.
    pub fn has_new_audio(&self) -> bool {
        self.active
            .as_ref()
            .is_some_and(|a| a.stream.audio().len() >= a.seen + FRAME)
    }

    /// Runs the next pass: the final pass of an ended utterance, or a tentative pass over the
    /// current one.
    pub fn pass(&mut self) -> Result<Vec<Event>> {
        let mut events = Vec::new();
        if let Some(mut a) = self.closing.pop_front() {
            self.finish_utterance(&mut a, PassMode::Final, &mut events)?;
            return Ok(events);
        }
        let Some(mut a) = self.active.take() else {
            return Ok(events);
        };
        let quiet = PassMode::FinalIfQuiet(self.config.endpoint_quiet);
        let mode = match a.end {
            Some(_) if a.cut => PassMode::Final,
            Some(_) => quiet,
            None if a.reopened.is_some() && !self.vad.in_speech() => quiet,
            None => PassMode::Tentative,
        };
        if mode == PassMode::Tentative {
            self.run(&mut a, mode, &mut events)?;
            self.active = Some(a);
        } else if !self.finish_utterance(&mut a, mode, &mut events)? {
            // The model still heard a word at the end, so the utterance goes on with the
            // audio that came after the end voice activity found.
            if let Some((speech_end, _)) = a.end.take() {
                a.reopened = Some(speech_end);
            }
            let tail = std::mem::take(&mut a.after_end);
            a.stream.push(&tail);
            self.active = Some(a);
        }
        Ok(events)
    }

    /// Runs a pass and reports what changed. Returns whether it was final.
    fn run(&mut self, a: &mut Active, mode: PassMode, events: &mut Vec<Event>) -> Result<bool> {
        a.seen = a.stream.audio().len();
        let out = a.stream.pass(&self.model, &self.config.passes, mode)?;
        let mut settled = out.settled;
        if !self.wrote {
            settled = settled.trim_start().to_string();
        }
        if !settled.is_empty() {
            self.wrote = true;
            events.push(Event::Settled { text: settled });
        }
        let mut tentative = out.tentative;
        if !self.wrote {
            tentative = tentative.trim_start().to_string();
        }
        if tentative != a.tentative {
            a.tentative = tentative.clone();
            events.push(Event::Tentative { text: tentative });
        }
        Ok(out.last)
    }

    /// Runs the final pass of an ended utterance. Returns false, settling only what two passes
    /// agree on, if `mode` asked for quiet at the end and the model heard a word there.
    fn finish_utterance(
        &mut self,
        a: &mut Active,
        mode: PassMode,
        events: &mut Vec<Event>,
    ) -> Result<bool> {
        if !self.run(a, mode, events)? {
            return Ok(false);
        }
        let now = a.start + a.stream.audio().len() as u64;
        let (speech_end, decided_at) = a.end.unwrap_or((a.reopened.unwrap_or(now), now));
        events.push(Event::UtteranceEnd {
            at: decided_at as f64 / SAMPLE_RATE as f64,
            speech_end: speech_end as f64 / SAMPLE_RATE as f64,
        });
        Ok(true)
    }

    /// Ends the input. Any utterance ends where the audio does, and gets its final pass.
    pub fn finish(&mut self) -> Result<Vec<Event>> {
        let mut audio = Vec::new();
        self.resampler.flush(&mut audio);
        let mut events = self.take(audio);
        if let Some(mut a) = self.active.take() {
            if a.end.is_none() {
                a.end = Some((self.total, self.total));
            }
            self.closing.push_back(a);
        }
        while let Some(mut a) = self.closing.pop_front() {
            self.finish_utterance(&mut a, PassMode::Final, &mut events)?;
        }
        Ok(events)
    }
}

enum Message {
    Audio(Vec<f32>),
    Finish,
}

/// How a stream's passes went.
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    pub passes: usize,
    /// Time spent in passes.
    pub busy: Duration,
    /// The longest pass.
    pub longest: Duration,
}

/// Streaming recognition on a thread of its own. Push audio as it arrives, and events come
/// back through the callback given to [`Stream::start`].
pub struct Stream {
    sender: mpsc::Sender<Message>,
    worker: JoinHandle<Result<StreamStats>>,
}

impl Stream {
    pub fn start(
        model: Arc<Model>,
        config: StreamConfig,
        mut on_event: impl FnMut(Event) + Send + 'static,
    ) -> Stream {
        let (sender, receiver) = mpsc::channel();
        let worker = std::thread::spawn(move || -> Result<StreamStats> {
            let mut streamer = Streamer::new(model, config);
            let mut stats = StreamStats::default();
            let mut next_pass: Option<Instant> = None;
            let mut finishing = false;
            loop {
                let wait = next_pass.map_or(Duration::from_millis(100), |t| {
                    t.saturating_duration_since(Instant::now())
                });
                let mut message = receiver.recv_timeout(wait);
                // Take everything waiting before running a pass, so passes never fall behind.
                loop {
                    match message {
                        Ok(Message::Audio(audio)) => {
                            streamer.push(&audio).into_iter().for_each(&mut on_event)
                        }
                        Ok(Message::Finish) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                            finishing = true
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                    }
                    match receiver.try_recv() {
                        Ok(m) => message = Ok(m),
                        Err(_) => break,
                    }
                }
                if finishing {
                    let start = Instant::now();
                    streamer.finish()?.into_iter().for_each(&mut on_event);
                    stats.busy += start.elapsed();
                    return Ok(stats);
                }
                if !streamer.in_utterance() {
                    next_pass = None;
                    continue;
                }
                let now = Instant::now();
                let due = *next_pass.get_or_insert(now + streamer.config().min_interval);
                if streamer.wants_final() || (now >= due && streamer.has_new_audio()) {
                    let start = Instant::now();
                    streamer.pass()?.into_iter().for_each(&mut on_event);
                    let took = start.elapsed();
                    stats.passes += 1;
                    stats.busy += took;
                    stats.longest = stats.longest.max(took);
                    let config = streamer.config();
                    next_pass = Some(
                        start
                            + config
                                .min_interval
                                .max(took.mul_f64(config.interval_factor)),
                    );
                }
            }
        });
        Stream { sender, worker }
    }

    /// Hands over audio at the configured input rate. It returns at once.
    pub fn push(&self, samples: &[f32]) {
        let _ = self.sender.send(Message::Audio(samples.to_vec()));
    }

    /// Ends the input, waits for the last events and returns how the passes went.
    pub fn finish(self) -> Result<StreamStats> {
        let _ = self.sender.send(Message::Finish);
        self.worker
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    }
}

/// CTC greedy decoding of `ids` from `frame` on, where the frame before had `prev`. Repeats
/// collapse and blanks drop out. Each token ends after its run of frames.
fn ctc_hypothesis(ids: &[u32], frame: usize, prev: u32, blank: u32) -> Vec<Hyp> {
    let mut out: Vec<Hyp> = Vec::new();
    let mut prev = prev;
    let mut extending = false;
    for (f, &id) in ids.iter().enumerate().skip(frame) {
        if id != blank && id != prev {
            out.push(Hyp {
                id,
                frame: f,
                end_frame: f + 1,
                resume: Resume::Ctc {
                    frame: f + 1,
                    prev: id,
                },
            });
            extending = true;
        } else if id != blank && extending {
            let last = out.last_mut().expect("a token is open");
            last.end_frame = f + 1;
            last.resume = Resume::Ctc {
                frame: f + 1,
                prev: id,
            };
        } else {
            extending = false;
        }
        prev = id;
    }
    out
}

/// A word of a hypothesis, as a range of its tokens.
struct Word {
    start: usize,
    end: usize,
}

/// Splits tokens into words where the tokenizer allows a split.
fn split_words(tokenizer: &Tokenizer, before: Option<u32>, hyp: &[Hyp]) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut prev = before;
    for (i, h) in hyp.iter().enumerate() {
        match words.last_mut() {
            Some(w) if !tokenizer.may_split_before(prev, h.id) => w.end = i + 1,
            _ => words.push(Word {
                start: i,
                end: i + 1,
            }),
        }
        prev = Some(h.id);
    }
    words
}

/// The most settled words a pass may repeat.
const MAX_REPEAT: usize = 3;

/// A word's text in lower case without punctuation, for comparing words.
fn normalized(tokenizer: &Tokenizer, ids: &[u32]) -> String {
    tokenizer
        .decode(ids)
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn word_ids(hyp: &[Hyp], w: &Word) -> Vec<u32> {
    hyp[w.start..w.end].iter().map(|h| h.id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctc_runs_collapse_and_resume_where_they_end() {
        // blank 0: frames "0 5 5 0 5 7 7 0"
        let ids = [0, 5, 5, 0, 5, 7, 7, 0];
        let hyp = ctc_hypothesis(&ids, 0, u32::MAX, 0);
        let got: Vec<(u32, usize)> = hyp.iter().map(|h| (h.id, h.end_frame)).collect();
        assert_eq!(got, vec![(5, 3), (5, 5), (7, 7)]);
        // Resuming inside the first run doesn't repeat its token.
        let from = ctc_hypothesis(&ids, 2, 5, 0);
        let got: Vec<u32> = from.iter().map(|h| h.id).collect();
        assert_eq!(got, vec![5, 7]);
        match &hyp[1].resume {
            Resume::Ctc { frame, prev } => assert_eq!((*frame, *prev), (5, 5)),
            Resume::Tdt(_) => unreachable!(),
        }
    }
}
