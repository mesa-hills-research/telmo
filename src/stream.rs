//! Streaming recognition of one utterance at a time, as audio arrives.
//!
//! Streaming is experimental. Its API and behaviour may change in any release, and its
//! accuracy is a little below offline transcription of the same audio.
//!
//! Neither model is built for streaming, so each pass recomputes part of the utterance and keeps
//! the text after the settled words. Parakeet re-encodes the utterance so far and decodes it
//! afresh. Granite recomputes only its current attention block and decodes CTC frames from the
//! last settled one.
//!
//! A word settles once two passes in a row agree on it, together with every word before it, it
//! ends at least a second before the audio does, and the model has heard a word after it. A
//! spoken form written as its word settles whole, and words where a spoken form could still be
//! under way wait. Settled words never change. When the utterance ends, one last pass settles
//! everything.
//!
//! [`Stream`] runs it all on a thread of its own. Audio of any sample rate goes in as it comes,
//! voice activity splits it into utterances, passes run as often as the computer keeps up with,
//! and the text comes back as [`Event`]s.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::biasing::{self, Compiled, Entries, SpecialWords, Spoken, Unit, normalized_words};
use crate::error::Result;
use crate::features::LogMelStream;
use crate::granite::{self, Granite};
use crate::granite_stream::GraniteStream;
use crate::model::Model;
use crate::parakeet::{Parakeet, Timings, Token};
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
    /// Settled audio Parakeet re-encodes before the unsettled part, as context. The default of
    /// 30 s is the longest utterance, so each pass encodes the whole utterance. On a window that
    /// starts in the middle of a sentence, Parakeet sometimes emits nothing at all.
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
            context: Duration::from_secs(30),
            settle_margin: Duration::from_secs(1),
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

/// Where the text after a settled token starts.
#[derive(Debug, Clone)]
enum Resume {
    /// The encoder frame after the token.
    Frame(usize),
    /// The CTC frame after the token, and the argmax of the frame before it.
    Ctc { frame: usize, prev: u32 },
}

impl Resume {
    fn frame(&self) -> usize {
        match self {
            Resume::Frame(frame) => *frame,
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
    /// Whether the token belongs to the word before it whatever the tokenizer says, as the
    /// tokens of a spotted special word or of an entry's spoken form or word do, so they settle
    /// together.
    joined: bool,
    /// How many of the pass's heard tokens this token and those before it stand for, which
    /// differs from their count once spoken forms are written as their words.
    heard_end: usize,
    /// The entries of the spoken form or word that starts with this token.
    entry: Option<Entries>,
}

/// A settled word, kept to recognize its audio when a later pass hears it again.
#[derive(Debug, Clone)]
struct SettledWord {
    /// The normalized words of each way a pass may hear it, as written, as heard, and for an
    /// entry's spoken form or word, as the entry's word and each of its forms, each followed by
    /// what came after it and without that.
    spellings: Vec<Vec<String>>,
    /// The frame of its first heard token, and the frame after its last.
    heard: Range<usize>,
    /// Its tokens as written, and its words as heard, normalized, to join it with the words
    /// settled after it.
    ids: Vec<u32>,
    as_heard: Vec<String>,
}

impl SettledWord {
    /// A settled word written as `ids`, which decode to `text`, and heard on frames `heard` as
    /// `as_heard`. `entries` are those it stands for, with the text after their word.
    fn new(
        spoken: &Spoken,
        text: &str,
        as_heard: Vec<String>,
        entries: Option<(Entries, &str)>,
        ids: Vec<u32>,
        heard: Range<usize>,
    ) -> SettledWord {
        let mut spellings = vec![normalized_words(text), as_heard.clone()];
        if let Some((entries, rest)) = entries {
            for s in spoken.spellings(entries, rest) {
                if !spellings.contains(&s) {
                    spellings.push(s);
                }
            }
        }
        SettledWord {
            spellings,
            heard,
            ids,
            as_heard,
        }
    }
}

/// One utterance's audio and the text settled from it so far. Experimental, like the rest of
/// this module.
pub struct UtteranceStream {
    audio: Vec<f32>,
    /// Where decoding continues after the settled tokens.
    resume: Resume,
    /// The last settled token as heard, for deciding where words split.
    last_settled: Option<u32>,
    /// The last settled words, to recognize the last few when a pass hears them again, and to
    /// join a word's start that settled alone with the rest.
    recent: Vec<SettledWord>,
    /// The previous pass's unsettled words, as token ids.
    previous: Vec<Vec<u32>>,
    granite: Option<GraniteStream>,
    /// Parakeet's log-mel frames, computed once as the audio arrives.
    log_mel: Option<LogMelStream>,
    /// Whether any text settled in this utterance.
    settled_any: bool,
    passes: usize,
    /// The special words passes favour.
    words: Option<Arc<Compiled>>,
    pub timings: Timings,
}

impl UtteranceStream {
    pub fn new(model: &Model) -> UtteranceStream {
        UtteranceStream::with_compiled(model, None)
    }

    /// A stream whose passes favour special words, as [`Model::transcribe_biased`] does.
    pub fn with_special_words(model: &Model, words: &SpecialWords) -> UtteranceStream {
        UtteranceStream::with_compiled(model, compile(model, words))
    }

    fn with_compiled(model: &Model, words: Option<Arc<Compiled>>) -> UtteranceStream {
        let timings = Timings::default();
        let (resume, granite, log_mel) = match model {
            Model::Parakeet(m) => (Resume::Frame(0), None, Some(m.front_end().stream())),
            Model::Granite(m) => {
                let mut blocks = GraniteStream::new(m);
                if let Some(words) = words.as_ref().filter(|w| !w.forms.is_empty()) {
                    blocks.keep_spotting_scores(&words.tokens);
                }
                (
                    Resume::Ctc {
                        frame: 0,
                        prev: u32::MAX,
                    },
                    Some(blocks),
                    None,
                )
            }
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
            words,
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
        let heard = match model {
            Model::Parakeet(m) => self.parakeet_hypothesis(m, config, mode == PassMode::Final)?,
            Model::Granite(m) => self.granite_hypothesis(m, mode == PassMode::Final)?,
        };
        Ok(self.settle(model.tokenizer(), heard, config, mode))
    }

    /// Settles what a pass heard. Repeats of the settled text go and the stream's timing is
    /// read on the tokens as heard, and spoken forms are written as their words last, so a
    /// form's word never splits and the settled text's audio stays known.
    fn settle(
        &mut self,
        tokenizer: &Tokenizer,
        heard: Heard,
        config: &PassConfig,
        mode: PassMode,
    ) -> PassText {
        let mut context = heard.tokens;
        let heard = context.split_off(heard.new);
        let heard = self.drop_repeats(tokenizer, config, heard, &context);
        let none = Spoken::default();
        let spoken = self.words.as_ref().map_or(&none, |w| &w.spoken);
        let hyp = write_spoken(spoken, tokenizer, &heard);
        let words = split_words(tokenizer, self.last_settled, &hyp);
        // Where the model last heard speech, taken as the last token heard with a letter or
        // digit, since punctuation often comes in the silence after the words.
        let last_word_end = heard
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
            // The last word of a pass never settles. It may still grow, and near the end of the
            // audio a model may close it with punctuation that the next audio contradicts.
            while settle + 1 < words.len()
                && settle < self.previous.len()
                && word_ids(&hyp, &words[settle]) == self.previous[settle]
                && hyp[words[settle].end - 1].end_frame * FRAME <= limit
            {
                settle += 1;
            }
            // Nothing settles from where a spoken form could still be under way, since the
            // words from the first that can't settle on may still change. A form holds back at
            // most its own words.
            let ids: Vec<u32> = heard.iter().map(|h| h.id).collect();
            while settle > 0 {
                let frontier = hyp[words[settle].start - 1].heard_end;
                let Some(open) = spoken.open_before(tokenizer, &ids, frontier) else {
                    break;
                };
                settle = words[..settle]
                    .iter()
                    .take_while(|w| hyp[w.end - 1].heard_end <= open)
                    .count();
            }
        }
        let settled_tokens = words.get(settle.wrapping_sub(1)).map_or(0, |w| w.end);
        let settled_ids: Vec<u32> = hyp[..settled_tokens].iter().map(|h| h.id).collect();
        let tentative_ids: Vec<u32> = hyp[settled_tokens..].iter().map(|h| h.id).collect();
        let mut from: usize = 0;
        for w in &words[..settle] {
            let to = hyp[w.end - 1].heard_end;
            let prev = from
                .checked_sub(1)
                .map_or(self.last_settled, |i| Some(heard[i].id));
            let word = &heard[from..to];
            let as_heard = split_words(tokenizer, prev, word)
                .iter()
                .map(|w| normalized(tokenizer, &word_ids(word, w)))
                .collect();
            let ids = word_ids(&hyp, w);
            let text = tokenizer.decode(&ids);
            // The entries of the spoken form or word that starts it, with the text after their
            // word, which is the same length in any capitals.
            let entries = hyp[w.start]
                .entry
                .map(|e| (e, text.get(spoken.word_len(e)..).unwrap_or_default()));
            let frames = word[0].frame..word[word.len() - 1].end_frame;
            let record = SettledWord::new(spoken, &text, as_heard, entries, ids, frames);
            self.recent.push(record);
            from = to;
        }
        if let Some(h) = heard[..from].last() {
            self.resume = h.resume.clone();
            self.last_settled = Some(h.id);
        }
        // Words join before the oldest are dropped, so the words kept are whole.
        if settle > 0 {
            join_entries(&mut self.recent, spoken, tokenizer);
        }
        let keep = self.recent.len().saturating_sub(KEPT);
        self.recent.drain(..keep);
        self.previous = words[settle..].iter().map(|w| word_ids(&hyp, w)).collect();
        let settled = tokenizer.decode_raw(&settled_ids);
        let tentative = tokenizer.decode_raw(&tentative_ids);
        self.settled_any |= !settled.is_empty();
        PassText {
            settled,
            tentative,
            last_word_end,
            last,
        }
    }

    /// Drops what a pass hears again of the settled text, from the tokens it heard. A model can
    /// place a word slightly earlier near the end of its audio than it does once more audio
    /// follows, so a settled word, or its end, may show up again just after the settled text,
    /// and spelled another way, as written or as one of its spoken forms. Leading tokens that
    /// can't start a word are such ends, and so are leading words that repeat the last settled
    /// ones, as [`repeats`] finds them, and end within `config.repeat_window` of them.
    /// `context` holds what the pass heard of the settled audio before `hyp`.
    fn drop_repeats(
        &self,
        tokenizer: &Tokenizer,
        config: &PassConfig,
        mut hyp: Vec<Hyp>,
        context: &[Hyp],
    ) -> Vec<Hyp> {
        if self.last_settled.is_none() {
            return hyp;
        }
        // A settled word was followed by another before it settled, so it's complete, and
        // leading tokens that can't start a word are left over from it or from words before it.
        let mut skip = 0;
        let mut prev = self.last_settled;
        while skip < hyp.len() && !tokenizer.may_split_before(prev, hyp[skip].id) {
            prev = Some(hyp[skip].id);
            skip += 1;
        }
        let words = split_words(tokenizer, prev, &hyp[skip..]);
        let texts: Vec<String> = words
            .iter()
            .map(|w| normalized(tokenizer, &word_ids(&hyp[skip..], w)))
            .collect();
        let window = (config.repeat_window.as_secs_f64() * SAMPLE_RATE as f64 / FRAME as f64).ceil()
            as usize;
        let limit = self.resume.frame() + window;
        for n in (1..=MAX_REPEAT.min(self.recent.len())).rev() {
            let tail = &self.recent[self.recent.len() - n..];
            let start = |i: usize| hyp[skip + words[i].start].frame;
            let in_place = |s: &SettledWord, before| heard_in_place(tokenizer, context, s, before);
            if let Some(k) = repeats(tail, &texts, start, in_place)
                && hyp[skip + words[k - 1].end - 1].end_frame <= limit
            {
                skip += words[k - 1].end;
                break;
            }
        }
        hyp.drain(..skip);
        hyp
    }

    /// Re-encodes the unsettled audio with up to `config.context` of settled audio before it,
    /// decodes the window from its start, and keeps the tokens from the end of the settled text
    /// on. The window's features are normalized over the window, as NVIDIA's buffered streaming
    /// does.
    ///
    /// Decoding the window afresh costs a little more than continuing from the decoder state
    /// saved after the settled text, but the state can disagree with a window that hears more.
    /// A word placed a frame or two early near the end of one window is heard again by the next,
    /// and a transducer that already closed the sentence there may then emit nothing at all.
    fn parakeet_hypothesis(
        &mut self,
        model: &Parakeet,
        config: &PassConfig,
        finished: bool,
    ) -> Result<Heard> {
        let boundary = self.resume.frame();
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
        let start = (boundary * FRAME).saturating_sub(context) / FRAME * FRAME;
        let (first, ready) = (start / HOP, log_mel.frames());
        // The encoder needs a frame of 80 ms.
        if ready < first + FRAME / HOP {
            return Ok(Heard {
                tokens: Vec::new(),
                new: 0,
            });
        }
        let features = timings.time("features", || log_mel.normalize(first..ready));
        let encoded = model.encode(&features.data, features.frames, timings, &mut |_, _, _| {});
        let mut state = model.initial_decoder_state(timings);
        state.frame = start / FRAME;
        let tokens: Vec<Token> = match &self.words {
            Some(words) if !words.forms.is_empty() => {
                model.beam_search(&encoded, start / FRAME, &state, words, timings)
            }
            _ => model
                .decode_from(&encoded, start / FRAME, &state, timings)
                .into_iter()
                .map(|(t, _)| t)
                .collect(),
        };
        let heard = tokens.into_iter().map(|t| {
            let end_frame = t.frame as usize + (t.duration as usize).max(1);
            Hyp {
                id: t.id,
                frame: t.frame as usize,
                end_frame,
                resume: Resume::Frame(end_frame),
                joined: false,
                heard_end: 0,
                entry: None,
            }
        });
        Ok(after_settled(heard, boundary))
    }

    /// Encodes the new audio, recomputing only the attention blocks it reaches, and decodes
    /// CTC frames from the last settled one.
    fn granite_hypothesis(&mut self, model: &Granite, finished: bool) -> Result<Heard> {
        let blocks = self.granite.as_mut().expect("Granite keeps its blocks");
        // The blocks take audio only when a pass runs, so the end of an utterance can still be
        // trimmed until then.
        blocks.push(&self.audio[blocks.samples().min(self.audio.len())..]);
        blocks.update(model, finished, &mut self.timings);
        let Resume::Ctc { frame, prev } = self.resume else {
            unreachable!("Granite resumes from a CTC frame")
        };
        let ids = blocks.frame_ids();
        let blank = model.config().blank as u32;
        // The settled words' frames as this pass aligns them, to tell a word heard again from
        // one said again.
        let repeatable = &self.recent[self.recent.len().saturating_sub(MAX_REPEAT)..];
        let from = repeatable
            .first()
            .map_or(frame, |w| w.heard.start)
            .min(frame);
        let before = from.checked_sub(1).map_or(u32::MAX, |f| ids[f]);
        let mut heard = ctc_hypothesis(&ids[..frame.min(ids.len())], from, before, blank);
        let new = heard.len();
        let hyp = ctc_hypothesis(ids, frame, prev, blank);
        let Some(words) = self.words.as_ref().filter(|w| !w.forms.is_empty()) else {
            heard.extend(hyp);
            return Ok(Heard { tokens: heard, new });
        };
        let greedy: Vec<Token> = hyp
            .iter()
            .map(|h| Token {
                id: h.id,
                frame: h.frame as u32,
                duration: (h.end_frame - h.frame) as u32,
            })
            .collect();
        let (tokens, spotted) = granite::timed(&mut self.timings, "spotting", || {
            biasing::spot(
                words,
                model.tokenizer(),
                &greedy,
                blocks.spotting_scores(),
                frame,
            )
        });
        heard.extend(spotted_hypothesis(&tokens, &spotted, ids));
        Ok(Heard { tokens: heard, new })
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
/// follows them. Times count seconds of audio from the stream's start, and `utterance` numbers
/// the utterances from 0. An utterance's last text can come after the next one has started.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Speech started at `at`.
    SpeechStart { utterance: u64, at: f64 },
    /// The words after the settled text that may still change, replacing the earlier ones.
    Tentative { utterance: u64, text: String },
    /// Words that won't change, to append after the earlier settled text.
    Settled { utterance: u64, text: String },
    /// The utterance ended. Its speech ended at `speech_end`, and all its text has settled.
    UtteranceEnd {
        utterance: u64,
        at: f64,
        speech_end: f64,
    },
}

/// An utterance being transcribed.
struct Active {
    /// The utterance's number in the stream.
    index: u64,
    stream: UtteranceStream,
    /// The stream's sample (16 kHz) the utterance's audio starts at.
    start: u64,
    /// Where voice activity ended it, if it has, as the end of speech and where it was decided.
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

/// The synchronous core of a [`Stream`], experimental like it. It takes audio, splits it into
/// utterances and runs passes when asked. [`Stream`] decides when.
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
    /// Utterances started so far.
    started: u64,
    /// The special words new utterances favour.
    words: Option<Arc<Compiled>>,
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
            started: 0,
            words: None,
        }
    }

    pub fn config(&self) -> &StreamConfig {
        &self.config
    }

    /// Favours these special words from the next utterance on. An utterance under way keeps the
    /// words it started with.
    pub fn set_special_words(&mut self, words: &SpecialWords) {
        self.words = compile(&self.model, words);
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
                    let mut stream =
                        UtteranceStream::with_compiled(&self.model, self.words.clone());
                    stream.push(&self.history_since(audio_from));
                    let index = self.started;
                    self.started += 1;
                    self.active = Some(Active {
                        index,
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
                        utterance: index,
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

    /// Runs the next pass, which is the final pass of an ended utterance or a tentative pass over
    /// the current one.
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
            events.push(Event::Settled {
                utterance: a.index,
                text: settled,
            });
        }
        let mut tentative = out.tentative;
        if !self.wrote {
            tentative = tentative.trim_start().to_string();
        }
        if tentative != a.tentative {
            a.tentative = tentative.clone();
            events.push(Event::Tentative {
                utterance: a.index,
                text: tentative,
            });
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
            utterance: a.index,
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
    Words(SpecialWords),
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

/// Streaming recognition on a thread of its own, which is experimental. Push audio as it
/// arrives, and events come back through the callback given to [`Stream::start`].
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
                        Ok(Message::Words(words)) => streamer.set_special_words(&words),
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

    /// Favours these special words from the next utterance on, as
    /// [`Streamer::set_special_words`] does. It returns at once.
    pub fn set_special_words(&self, words: SpecialWords) {
        let _ = self.sender.send(Message::Words(words));
    }

    /// Ends the input, waits for the last events and returns how the passes went.
    pub fn finish(self) -> Result<StreamStats> {
        let _ = self.sender.send(Message::Finish);
        self.worker
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    }
}

/// The special words in the model's tokens, or none if they leave nothing to favour.
fn compile(model: &Model, words: &SpecialWords) -> Option<Arc<Compiled>> {
    Some(Compiled::new(words, model.tokenizer()).0)
        .filter(|w| !w.is_empty())
        .map(Arc::new)
}

/// A pass's hypothesis from Granite's tokens after word spotting, each resuming after its run
/// of frames, where `ids` holds every frame's most likely token. The tokens of a spotted word,
/// whose ranges `spotted` gives, are joined, so the word settles whole or not at all, and a
/// later pass never resumes inside the frames it took over.
fn spotted_hypothesis(tokens: &[Token], spotted: &[Range<usize>], ids: &[u32]) -> Vec<Hyp> {
    let mut joined = vec![false; tokens.len()];
    for range in spotted {
        for j in &mut joined[range.start + 1..range.end] {
            *j = true;
        }
    }
    tokens
        .iter()
        .zip(joined)
        .map(|(t, joined)| {
            let end_frame = (t.frame + t.duration) as usize;
            Hyp {
                id: t.id,
                frame: t.frame as usize,
                end_frame,
                resume: Resume::Ctc {
                    frame: end_frame,
                    prev: ids[end_frame - 1],
                },
                joined,
                heard_end: 0,
                entry: None,
            }
        })
        .collect()
}

/// A pass's tokens as heard.
struct Heard {
    tokens: Vec<Hyp>,
    /// Where the tokens that may be new start, after those the pass heard of the settled audio.
    new: usize,
}

/// A window decoded from before the settled text's end, at frame `boundary`. Its tokens from a
/// frame before that end may be new, since a token placed a frame earlier than the settled
/// text's end in an earlier window still counts as new. What they repeat of the settled text is
/// dropped later.
fn after_settled(heard: impl Iterator<Item = Hyp>, boundary: usize) -> Heard {
    let tokens: Vec<Hyp> = heard.collect();
    let from = boundary.saturating_sub(1);
    let new = if boundary == 0 {
        0
    } else {
        tokens.partition_point(|h| h.frame < from)
    };
    Heard { tokens, new }
}

/// Whether `context`, what a pass heard of the settled audio, holds the settled word `s` on its
/// own frames, in one of its spellings, ending before frame `before`.
fn heard_in_place(tokenizer: &Tokenizer, context: &[Hyp], s: &SettledWord, before: usize) -> bool {
    let words = split_words(tokenizer, None, context);
    let texts: Vec<String> = words
        .iter()
        .map(|w| normalized(tokenizer, &word_ids(context, w)))
        .collect();
    words.iter().enumerate().any(|(k, w)| {
        let end = context[w.end - 1].end_frame;
        let overlaps = context[w.start].frame < s.heard.end && end > s.heard.start;
        overlaps
            && end < before
            && s.spellings
                .iter()
                .any(|sp| !sp.is_empty() && texts[k..].starts_with(sp))
    })
}

/// The tokens heard, `heard`, with spoken forms written as their words, and entries' words
/// heard as written grouped into one word. A form's word takes the frames [`Spoken::written`]
/// gives it and resumes where the form ends. The tokens of each form's or entry's word are
/// joined so they settle together, and the first records its entries. Each token records how
/// many heard tokens it and those before it stand for.
fn write_spoken(spoken: &Spoken, tokenizer: &Tokenizer, heard: &[Hyp]) -> Vec<Hyp> {
    let ids: Vec<u32> = heard.iter().map(|h| h.id).collect();
    let mut units = spoken.units(tokenizer, &ids).into_iter().peekable();
    let mut out = Vec::with_capacity(heard.len());
    let mut i = 0;
    while i < heard.len() {
        let Some(unit) = units.next_if(|u| u.heard().start == i) else {
            out.push(Hyp {
                heard_end: i + 1,
                ..heard[i].clone()
            });
            i += 1;
            continue;
        };
        let range = unit.heard();
        let entries = unit.entries();
        let Unit::Form(r) = unit else {
            for (k, h) in heard[range.clone()].iter().enumerate() {
                out.push(Hyp {
                    joined: k > 0 || h.joined,
                    heard_end: range.start + k + 1,
                    entry: (k == 0).then_some(entries),
                    ..h.clone()
                });
            }
            i = range.end;
            continue;
        };
        let (first, last) = (&heard[r.heard.start], &heard[r.heard.end - 1]);
        for (k, (id, frame)) in spoken
            .written(&r, first.frame, last.frame)
            .into_iter()
            .enumerate()
        {
            out.push(Hyp {
                id,
                frame,
                end_frame: last.end_frame,
                resume: last.resume.clone(),
                joined: k > 0 || first.joined,
                heard_end: r.heard.end,
                entry: (k == 0).then_some(entries),
            });
        }
        i = r.heard.end;
    }
    out
}

/// Joins settled words that spell an entry's spoken form or word whole into one with its
/// entries, as when a word's start settled before the rest was heard. `recent` holds the words
/// kept from before and all those a pass settled, and is matched whole before any are dropped,
/// so a word's start is found however many words settle after it.
fn join_entries(recent: &mut Vec<SettledWord>, spoken: &Spoken, tokenizer: &Tokenizer) {
    let mut starts = Vec::new();
    let mut ids = Vec::new();
    for w in recent.iter() {
        starts.push(ids.len());
        ids.extend_from_slice(&w.ids);
    }
    // From the last, so the words before stay where they are.
    for unit in spoken.units(tokenizer, &ids).into_iter().rev() {
        let range = unit.heard();
        let Ok(first) = starts.binary_search(&range.start) else {
            continue;
        };
        let last = starts.partition_point(|&s| s < range.end) - 1;
        if first == last {
            continue;
        }
        let parts: Vec<SettledWord> = recent.drain(first..=last).collect();
        let ids: Vec<u32> = parts.iter().flat_map(|w| w.ids.iter().copied()).collect();
        let as_heard = parts
            .iter()
            .flat_map(|w| w.as_heard.iter().cloned())
            .collect();
        let text = tokenizer.decode(&ids);
        let rest = text.get(unit.len()..).unwrap_or_default();
        let frames = parts[0].heard.start..parts[parts.len() - 1].heard.end;
        let entries = Some((unit.entries(), rest));
        let joined = SettledWord::new(spoken, &text, as_heard, entries, ids, frames);
        recent.insert(first, joined);
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
                joined: false,
                heard_end: 0,
                entry: None,
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

/// Splits tokens into words where the tokenizer allows a split and the tokens aren't joined.
fn split_words(tokenizer: &Tokenizer, before: Option<u32>, hyp: &[Hyp]) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut prev = before;
    for (i, h) in hyp.iter().enumerate() {
        match words.last_mut() {
            Some(w) if h.joined || !tokenizer.may_split_before(prev, h.id) => w.end = i + 1,
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

/// The settled words kept once they're joined. The repeat check and Granite's re-alignment use
/// the last [`MAX_REPEAT`], and a later pass can complete an entry's word or form whose first
/// words, at most all but one of [`SpecialWords::MAX_PARTS`], are the last ones kept.
const KEPT: usize = if SpecialWords::MAX_PARTS - 1 > MAX_REPEAT {
    SpecialWords::MAX_PARTS - 1
} else {
    MAX_REPEAT
};

/// How many leading words of a pass, normalized as `texts`, hear the settled words `settled`
/// again, each by one of its spellings, or none. A first word starting, at frame `start(i)` for
/// word `i`, inside the first settled word's heard frames is that audio heard again. One starting
/// after them was spoken again ("no, no") if `in_place(word, frame)` says the pass also heard the
/// settled word on its own frames, ending before that frame. Otherwise it's the settled word
/// placed a little later than the pass before placed it. The first may also be just the end of a
/// spelling ("control" of "cube control"), as the pass's window began inside it, and one word may
/// come back split differently, with only its end the same ("themselves" heard again as
/// "inselves").
fn repeats(
    settled: &[SettledWord],
    texts: &[String],
    start: impl Fn(usize) -> usize,
    in_place: impl Fn(&SettledWord, usize) -> bool,
) -> Option<usize> {
    let first = settled.first()?;
    if texts.is_empty() || start(0) >= first.heard.end && in_place(first, start(0)) {
        return None;
    }
    let mut at = 0;
    for (j, s) in settled.iter().enumerate() {
        let rest = &texts[at..];
        let whole = s
            .spellings
            .iter()
            .filter(|sp| !sp.is_empty() && rest.starts_with(sp))
            .map(Vec::len)
            .max();
        // The end of a spelling comes from a window that began inside the word.
        let tail = || {
            if j > 0 || start(at) >= s.heard.end {
                return None;
            }
            let ends = s
                .spellings
                .iter()
                .flat_map(|sp| (1..sp.len()).map(|t| &sp[t..]));
            ends.filter(|end| rest.starts_with(end))
                .map(<[String]>::len)
                .max()
        };
        let same_end = || {
            let single = settled.len() == 1 && !rest.is_empty();
            let ends_like = |sp: &Vec<String>| sp.len() == 1 && shares_end(&rest[0], &sp[0]);
            (single && s.spellings.iter().any(ends_like)).then_some(1)
        };
        at += whole.or_else(tail).or_else(same_end)?;
    }
    Some(at)
}

/// Whether `heard` looks like the end of `word` heard again, meaning it isn't longer than `word`,
/// and they end with the same five letters or more, at least half of `heard`.
fn shares_end(heard: &str, word: &str) -> bool {
    let common = heard
        .chars()
        .rev()
        .zip(word.chars().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let len = heard.chars().count();
    common >= 5 && 2 * common >= len && len <= word.chars().count()
}

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
    fn a_spotted_phrase_settles_as_one_word() {
        let tok = crate::biasing::tests::tokenizer();
        // "▁the ▁New ▁York ▁city ▁the", the middle three spotted together, two frames each.
        let ids_in = [6, 42, 43, 44, 6];
        let tokens: Vec<Token> = ids_in
            .iter()
            .enumerate()
            .map(|(i, &id)| Token {
                id,
                frame: 2 * i as u32,
                duration: 2,
            })
            .collect();
        let frame_ids = vec![0u32; 10];
        let hyp = spotted_hypothesis(&tokens, std::slice::from_ref(&(1..4)), &frame_ids);
        let words: Vec<(usize, usize)> = split_words(&tok, None, &hyp)
            .iter()
            .map(|w| (w.start, w.end))
            .collect();
        assert_eq!(words, vec![(0, 1), (1, 4), (4, 5)]);
        // Without the spot, each word stands alone.
        let plain = spotted_hypothesis(&tokens, &[], &frame_ids);
        assert_eq!(split_words(&tok, None, &plain).len(), 5);
    }

    #[test]
    fn a_spoken_forms_word_is_joined_and_resumes_where_the_form_ends() {
        let tok = crate::biasing::tests::tokenizer();
        let mut words = SpecialWords::new();
        words.add_spoken("kubectl", "cube control").unwrap();
        let spoken = Compiled::new(&words, &tok).0.spoken;
        // "▁the ▁cube ▁control ▁the", two frames each.
        let heard = |ids: &[u32]| -> Vec<Hyp> {
            ids.iter()
                .enumerate()
                .map(|(i, &id)| Hyp {
                    id,
                    frame: 2 * i,
                    end_frame: 2 * i + 2,
                    resume: Resume::Frame(2 * i + 2),
                    joined: false,
                    heard_end: 0,
                    entry: None,
                })
                .collect()
        };
        let hyp = write_spoken(&spoken, &tok, &heard(&[6, 4, 5, 6]));
        let got: Vec<(u32, usize, bool)> = hyp.iter().map(|h| (h.id, h.frame, h.joined)).collect();
        assert_eq!(
            got,
            vec![
                (6, 0, false),
                (1, 2, false),
                (2, 3, true),
                (3, 4, true),
                (6, 6, false)
            ]
        );
        // Each token of the word resumes where "control" ended.
        assert!(
            hyp[1..4]
                .iter()
                .all(|h| h.resume.frame() == 6 && h.end_frame == 6)
        );
        let words: Vec<(usize, usize)> = split_words(&tok, None, &hyp)
            .iter()
            .map(|w| (w.start, w.end))
            .collect();
        assert_eq!(words, vec![(0, 1), (1, 4), (4, 5)]);
    }

    /// A Parakeet stream over `frames` encoder frames of silence that writes `words`' spoken
    /// forms, with the test tokenizer.
    fn stream_with(words: &SpecialWords, frames: usize) -> (UtteranceStream, Tokenizer) {
        let tok = crate::biasing::tests::tokenizer();
        let compiled = Compiled::new(words, &tok).0;
        let stream = UtteranceStream {
            audio: vec![0.0; frames * FRAME],
            resume: Resume::Frame(0),
            last_settled: None,
            recent: Vec::new(),
            previous: Vec::new(),
            granite: None,
            log_mel: None,
            settled_any: false,
            passes: 0,
            words: Some(Arc::new(compiled)),
            timings: Timings::default(),
        };
        (stream, tok)
    }

    /// Tokens as (id, frame), each lasting a frame.
    fn hyps(tokens: &[(u32, usize)]) -> impl Iterator<Item = Hyp> + '_ {
        tokens.iter().map(|&(id, frame)| Hyp {
            id,
            frame,
            end_frame: frame + 1,
            resume: Resume::Frame(frame + 1),
            joined: false,
            heard_end: 0,
            entry: None,
        })
    }

    /// A Parakeet window of tokens as (id, frame), each lasting a frame.
    fn window(s: &UtteranceStream, tokens: &[(u32, usize)]) -> Heard {
        after_settled(hyps(tokens), s.resume.frame())
    }

    #[test]
    fn a_settled_forms_tail_heard_again_is_dropped() {
        let mut words = SpecialWords::new();
        words.add_spoken("kubectl", "cube control").unwrap();
        let (mut s, tok) = stream_with(&words, 50);
        let config = PassConfig::default();
        // "▁the ▁cube ▁control" then, much later, "▁the".
        let heard = [(6, 0), (4, 2), (5, 4), (6, 40)];
        let first = s.settle(&tok, window(&s, &heard), &config, PassMode::Tentative);
        assert_eq!(first.settled, "");
        let second = s.settle(&tok, window(&s, &heard), &config, PassMode::Tentative);
        assert_eq!(second.settled, " the kubectl");
        // The next window starts a frame before the settled text's end, so it holds "control"
        // again, which is settled audio.
        let third = s.settle(&tok, window(&s, &heard), &config, PassMode::Tentative);
        assert_eq!(
            (third.settled.as_str(), third.tentative.as_str()),
            ("", " the")
        );
    }

    /// A stream that settled "the kubectl", heard as `heard`, after two passes.
    fn settled_kubectl(heard: &[(u32, usize)]) -> (UtteranceStream, Tokenizer) {
        let mut words = SpecialWords::new();
        words.add_spoken("kubectl", "cube control").unwrap();
        let (mut s, tok) = stream_with(&words, 50);
        let config = PassConfig::default();
        s.settle(&tok, window(&s, heard), &config, PassMode::Tentative);
        let second = s.settle(&tok, window(&s, heard), &config, PassMode::Tentative);
        assert_eq!(second.settled, " the kubectl");
        assert_eq!(s.resume.frame(), 5);
        (s, tok)
    }

    #[test]
    fn a_settled_word_is_recognized_in_its_other_spellings() {
        let config = PassConfig::default();
        let written = [(6, 0), (1, 2), (2, 3), (3, 4), (6, 40)];
        let spoken = [(6, 0), (4, 2), (5, 4), (6, 40)];
        // Written as heard, then heard as its spoken form, whose last word the window keeps.
        let (mut s, tok) = settled_kubectl(&written);
        let next = s.settle(&tok, window(&s, &spoken), &config, PassMode::Tentative);
        assert_eq!(next.tentative, " the");
        // Heard as its spoken form, then heard as written, on the window's first frame.
        let (mut s, tok) = settled_kubectl(&spoken);
        let again = [(6, 0), (1, 4), (2, 4), (3, 4), (6, 40)];
        let next = s.settle(&tok, window(&s, &again), &config, PassMode::Tentative);
        assert_eq!(next.tentative, " the");
    }

    #[test]
    fn a_settled_word_is_recognized_in_its_other_spellings_with_what_follows_it() {
        let mut words = SpecialWords::new();
        words.add_spoken("kubectl", "cube control").unwrap();
        let config = PassConfig::default();
        // Settles `first` over two passes, then gives a pass's tentative text and a final pass's
        // settled text over `later`.
        let run = |first: &[(u32, usize)], settled: &str, later: &[(u32, usize)]| {
            let (mut s, tok) = stream_with(&words, 50);
            s.settle(&tok, window(&s, first), &config, PassMode::Tentative);
            let second = s.settle(&tok, window(&s, first), &config, PassMode::Tentative);
            assert_eq!(second.settled, settled);
            let next = s.settle(&tok, window(&s, later), &config, PassMode::Tentative);
            let last = s.settle(&tok, window(&s, later), &config, PassMode::Final);
            (next.tentative, last.settled)
        };
        let the = || (" the".to_string(), " the".to_string());
        for (end, text) in [
            (&[62][..], "'s"),
            (&[77], "’s"),
            (&[63, 47], "-s"),
            (&[60], "."),
            (&[61], "—"),
        ] {
            // The word's tokens, then `end` on the frame of its last token, then "the".
            let heard = |tokens: &[(u32, usize)]| -> Vec<(u32, usize)> {
                let end = end.iter().map(|&id| (id, 4));
                tokens.iter().copied().chain(end).chain([(6, 40)]).collect()
            };
            let written = heard(&[(6, 0), (1, 2), (2, 3), (3, 4)]);
            let spoken = heard(&[(6, 0), (4, 2), (5, 4)]);
            let again = heard(&[(6, 0), (1, 4), (2, 4), (3, 4)]);
            let settled = format!(" the kubectl{text}");
            // Written as heard, then heard as its spoken form, whose last word the window keeps.
            assert_eq!(run(&written, &settled, &spoken), the(), "{text}");
            // Heard as its spoken form, then heard as written, on the window's first frame.
            assert_eq!(run(&spoken, &settled, &again), the(), "{text}");
            // The end of the form said again after the settled word's audio is new speech.
            let later: Vec<(u32, usize)> = [(5, 8)]
                .into_iter()
                .chain(end.iter().map(|&id| (id, 8)))
                .chain([(6, 40)])
                .collect();
            let new = format!(" control{text} the");
            for first in [&written, &spoken] {
                assert_eq!(run(first, &settled, &later), (new.clone(), new.clone()));
            }
        }
    }

    #[test]
    fn a_settled_word_keeps_its_own_spoken_forms_among_words_spelled_alike() {
        let mut words = SpecialWords::new();
        words.add_spoken("C++", "c plus plus").unwrap();
        words.add_spoken("C#", "c sharp").unwrap();
        let (mut s, tok) = stream_with(&words, 50);
        let config = PassConfig::default();
        // "the C#" settles as heard, then a pass hears it as "c sharp" and keeps "sharp".
        let written = [(6, 0), (58, 2), (67, 3), (6, 40)];
        s.settle(&tok, window(&s, &written), &config, PassMode::Tentative);
        let second = s.settle(&tok, window(&s, &written), &config, PassMode::Tentative);
        assert_eq!(second.settled, " the C#");
        let spoken = [(6, 0), (30, 2), (76, 3), (6, 40)];
        let next = s.settle(&tok, window(&s, &spoken), &config, PassMode::Tentative);
        assert_eq!(next.tentative, " the");
    }

    /// A settled word written as `ids` and heard as written on frames `heard`.
    fn settled(tok: &Tokenizer, ids: &[u32], heard: Range<usize>) -> SettledWord {
        let text = tok.decode(ids);
        let words = normalized_words(&text);
        SettledWord::new(&Spoken::default(), &text, words, None, ids.to_vec(), heard)
    }

    /// Settles `first` over two passes, which settle `settled`, then gives a pass's tentative
    /// text and a final pass's settled text over `later`.
    fn settle_then(
        words: &SpecialWords,
        first: &[(u32, usize)],
        settled: &str,
        later: &[(u32, usize)],
    ) -> (String, String) {
        let (mut s, tok) = stream_with(words, 50);
        let config = PassConfig::default();
        s.settle(&tok, window(&s, first), &config, PassMode::Tentative);
        let second = s.settle(&tok, window(&s, first), &config, PassMode::Tentative);
        assert_eq!(second.settled, settled);
        let next = s.settle(&tok, window(&s, later), &config, PassMode::Tentative);
        let last = s.settle(&tok, window(&s, later), &config, PassMode::Final);
        (next.tentative, last.settled)
    }

    fn the() -> (String, String) {
        (" the".to_string(), " the".to_string())
    }

    #[test]
    fn a_word_of_several_words_heard_as_written_settles_as_one() {
        let mut words = SpecialWords::new();
        words.add_spoken("New York", "cube control").unwrap();
        // "New York" as written, then heard as its form, whose last word the window keeps.
        let written = [(6, 0), (42, 2), (43, 4), (6, 40)];
        let spoken = [(6, 0), (4, 2), (5, 4), (6, 40)];
        assert_eq!(
            settle_then(&words, &written, " the New York", &spoken),
            the()
        );
        // Three words, both ways.
        let mut words = SpecialWords::new();
        words.add_spoken("New York city", "cube control").unwrap();
        let written = [(6, 0), (42, 2), (43, 3), (44, 4), (6, 40)];
        let settled = " the New York city";
        assert_eq!(settle_then(&words, &written, settled, &spoken), the());
        let again = [(6, 0), (42, 2), (43, 4), (44, 4), (6, 40)];
        assert_eq!(settle_then(&words, &spoken, settled, &again), the());
    }

    #[test]
    fn the_start_of_a_word_of_several_words_waits_for_the_rest() {
        let mut words = SpecialWords::new();
        words.add_spoken("New York", "cube control").unwrap();
        let (mut s, tok) = stream_with(&words, 50);
        let config = PassConfig::default();
        let mut pass = |heard: &[(u32, usize)], mode| {
            let w = window(&s, heard);
            let out = s.settle(&tok, w, &config, mode);
            (out.settled, out.tentative)
        };
        let tentative = PassMode::Tentative;
        // Two passes agree on "the New", but the word after it may still change.
        pass(&[(6, 0), (42, 2), (6, 4)], tentative);
        assert_eq!(pass(&[(6, 0), (42, 2), (44, 4)], tentative).0, " the");
        let written = [(6, 0), (42, 2), (43, 4), (6, 40)];
        pass(&written, tentative);
        assert_eq!(pass(&written, tentative).0, " New York");
        let spoken = [(6, 0), (4, 2), (5, 4), (6, 40)];
        assert_eq!(pass(&spoken, tentative).1, " the");
        assert_eq!(pass(&spoken, PassMode::Final).0, " the");
    }

    #[test]
    fn a_settled_start_of_a_word_joins_the_rest_once_it_settles() {
        let tok = crate::biasing::tests::tokenizer();
        let config = PassConfig::default();
        for (written, form, start, rest, again) in [
            // "New" settled before "York" was heard, then "cube control" on those frames.
            ("New York", "cube control", 42, 43, &[(4, 2), (5, 4)][..]),
            // "cube" settled as heard before "control", then "kubectl" on those frames.
            ("kubectl", "cube control", 4, 5, &[(1, 4), (2, 4), (3, 4)]),
        ] {
            let mut words = SpecialWords::new();
            words.add_spoken(written, form).unwrap();
            let (mut s, _) = stream_with(&words, 50);
            s.recent = vec![settled(&tok, &[6], 0..1), settled(&tok, &[start], 2..3)];
            s.resume = Resume::Frame(3);
            s.last_settled = Some(start);
            let heard = [(6, 0), (start, 2), (rest, 4), (6, 40)];
            s.settle(&tok, window(&s, &heard), &config, PassMode::Tentative);
            let next = s.settle(&tok, window(&s, &heard), &config, PassMode::Tentative);
            assert_eq!(next.settled, tok.decode_raw(&[rest]), "{written}");
            let later: Vec<(u32, usize)> = [(6, 0)]
                .into_iter()
                .chain(again.iter().copied())
                .chain([(6, 40)])
                .collect();
            let next = s.settle(&tok, window(&s, &later), &config, PassMode::Tentative);
            assert_eq!(next.tentative, " the", "{written}");
        }
    }

    #[test]
    fn a_settled_start_joins_the_longest_word_whatever_settles_after_it() {
        let tok = crate::biasing::tests::tokenizer();
        let config = PassConfig::default();
        let longest = "New York city plus sharp";
        assert_eq!(longest.split(' ').count(), SpecialWords::MAX_PARTS);
        let mut words = SpecialWords::new();
        words.add_spoken(longest, "cube control").unwrap();
        // "New" settled alone, then its other four words settle on frame 4 with one or two
        // words after them.
        for (after, words_after) in [(&[(9, 4), (10, 4)][..], 1), (&[(6, 4), (9, 4), (10, 4)], 2)] {
            let (mut s, _) = stream_with(&words, 50);
            s.recent = vec![settled(&tok, &[6], 0..1), settled(&tok, &[42], 2..3)];
            s.resume = Resume::Frame(3);
            s.last_settled = Some(42);
            let rest = [(6, 0), (42, 2), (43, 4), (44, 4), (71, 4), (76, 4)];
            let heard: Vec<(u32, usize)> = (rest.iter().chain(after).copied())
                .chain([(6, 40)])
                .collect();
            s.settle(&tok, window(&s, &heard), &config, PassMode::Tentative);
            let next = s.settle(&tok, window(&s, &heard), &config, PassMode::Tentative);
            let ids: Vec<u32> = ([43, 44, 71, 76].into_iter())
                .chain(after.iter().map(|&(id, _)| id))
                .collect();
            assert_eq!(next.settled, tok.decode_raw(&ids));
            // The word is one settled word, heard on frames 2 to 4, and so is each after it.
            let frames: Vec<Range<usize>> = s.recent.iter().map(|w| w.heard.clone()).collect();
            let mut expected = vec![0..1, 2..5];
            expected.extend(std::iter::repeat_n(4..5, words_after));
            assert_eq!(frames, expected);
            // Its form on its frames, with the words after it, is the settled audio again.
            let again: Vec<(u32, usize)> = ([(6, 0), (4, 2), (5, 4)].into_iter())
                .chain(after.iter().copied())
                .chain([(6, 40)])
                .collect();
            let next = s.settle(&tok, window(&s, &again), &config, PassMode::Tentative);
            assert_eq!(next.tentative, " the", "{words_after} after");
        }
    }

    #[test]
    fn a_copy_without_what_followed_the_settled_word_is_recognized() {
        let mut words = SpecialWords::new();
        words.add_spoken("kubectl", "cube control").unwrap();
        let written = [(6, 0), (1, 2), (2, 3), (3, 4), (62, 4), (6, 40)];
        let spoken = [(6, 0), (4, 2), (5, 4), (62, 4), (6, 40)];
        // "kubectl's" then "cube control the", and "cube control's" then "kubectl the".
        let bare = [(6, 0), (4, 2), (5, 4), (6, 40)];
        assert_eq!(
            settle_then(&words, &written, " the kubectl's", &bare),
            the()
        );
        let bare = [(6, 0), (1, 4), (2, 4), (3, 4), (6, 40)];
        assert_eq!(settle_then(&words, &spoken, " the kubectl's", &bare), the());
    }

    #[test]
    fn the_end_of_a_settled_form_said_again_later_is_new() {
        let (mut s, tok) = settled_kubectl(&[(6, 0), (4, 2), (5, 4), (6, 40)]);
        // "control" at frame 8 comes after the settled form's audio, so it's new speech.
        let next = s.settle(
            &tok,
            window(&s, &[(5, 8), (6, 40)]),
            &PassConfig::default(),
            PassMode::Tentative,
        );
        assert_eq!(next.tentative, " control the");
    }

    #[test]
    fn a_word_spoken_again_after_it_settled_stays() {
        let config = PassConfig::default();
        // Each case settles the first words over two passes, then a later pass hears more.
        let stream = |first: &[(u32, usize)], later: Heard| {
            let (mut s, tok) = stream_with(&SpecialWords::new(), 50);
            s.settle(&tok, window(&s, first), &config, PassMode::Tentative);
            s.settle(&tok, window(&s, first), &config, PassMode::Tentative);
            s.settle(&tok, later, &config, PassMode::Tentative)
                .tentative
        };
        let at = |tokens: &[(u32, usize)], boundary| after_settled(hyps(tokens), boundary);
        // A Parakeet window hears "the", settled on frame 0, there again, and on frame 3 too
        // ("No | no").
        let the = [(6, 0), (42, 40)];
        assert_eq!(stream(&the, at(&[(6, 0), (6, 3), (42, 40)], 1)), " the New");
        // With no "the" on frame 0, the window placed the settled one later.
        assert_eq!(stream(&the, at(&[(6, 3), (42, 40)], 1)), " New");
        // Two words said again after them ("that's a good | that's a good").
        let both = [(6, 0), (4, 2), (42, 40)];
        let again = [(6, 0), (4, 2), (6, 4), (4, 5), (42, 40)];
        assert_eq!(stream(&both, at(&again, 3)), " the cube New");
    }

    #[test]
    fn a_copy_right_after_the_settled_word_heard_on_its_frames_is_the_same_word() {
        // Granite settled "the" on frames 0 and 1, and a later pass hears more after it.
        let tentative = |tokens: &[(u32, usize, usize)]| {
            let (mut s, tok) = stream_with(&SpecialWords::new(), 50);
            s.recent = vec![settled(&tok, &[6], 0..2)];
            s.resume = Resume::Ctc { frame: 2, prev: 6 };
            s.last_settled = Some(6);
            let tokens = tokens
                .iter()
                .map(|&(id, frame, end_frame)| Hyp {
                    id,
                    frame,
                    end_frame,
                    resume: Resume::Frame(end_frame),
                    joined: false,
                    heard_end: 0,
                    entry: None,
                })
                .collect();
            let heard = Heard { tokens, new: 1 };
            s.settle(&tok, heard, &PassConfig::default(), PassMode::Tentative)
                .tentative
        };
        // The pass aligns "the" on frames 0 to 2, so the frames after the settled text start with
        // the rest of that run.
        assert_eq!(tentative(&[(6, 0, 2), (6, 2, 3), (42, 40, 41)]), " New");
        // After a frame of something else, "the" was said again.
        assert_eq!(tentative(&[(6, 0, 2), (6, 3, 4), (42, 40, 41)]), " the New");
    }

    #[test]
    fn a_form_whose_start_settled_as_heard_stays_as_heard() {
        let mut words = SpecialWords::new();
        words.add_spoken("kubectl", "cube control").unwrap();
        let (mut s, tok) = stream_with(&words, 50);
        // "the cube" settled as heard, "cube" on frame 2.
        s.recent = vec![settled(&tok, &[6], 0..1), settled(&tok, &[4], 2..3)];
        s.resume = Resume::Frame(3);
        s.last_settled = Some(4);
        // A later window hears "cube control". Its "cube" is settled audio, so only "control"
        // is new, and the settled "cube" can't become kubectl.
        let later = s.settle(
            &tok,
            window(&s, &[(6, 0), (4, 2), (5, 4), (6, 40)]),
            &PassConfig::default(),
            PassMode::Tentative,
        );
        assert_eq!(later.tentative, " control the");
    }

    #[test]
    fn a_forms_start_waits_while_its_later_words_may_still_change() {
        let mut words = SpecialWords::new();
        words.add_spoken("kubectl", "cube control").unwrap();
        let (mut s, tok) = stream_with(&words, 50);
        let config = PassConfig::default();
        let mut pass = |heard: &[(u32, usize)]| {
            let w = window(&s, heard);
            s.settle(&tok, w, &config, PassMode::Tentative).settled
        };
        // Two passes agree on "the cube", but the word after it, the last one, differs, so
        // "cube" could still be the start of "cube control". The word before it settles.
        assert_eq!(pass(&[(6, 0), (4, 2), (6, 4)]), "");
        assert_eq!(pass(&[(6, 0), (4, 2), (42, 4)]), " the");
        // Then the pass hears the form, which settles whole once two passes agree.
        let heard = [(6, 0), (4, 2), (5, 4), (6, 40)];
        assert_eq!(pass(&heard), "");
        assert_eq!(pass(&heard), " kubectl");
    }

    #[test]
    fn the_end_of_speech_is_where_a_form_was_heard_to_end() {
        let mut words = SpecialWords::new();
        words.add_spoken("York", "cube control").unwrap();
        // 25 frames are 2 s, and "cube control" is heard until frame 21, with "York" written at
        // its first frame.
        let (mut s, tok) = stream_with(&words, 25);
        let quiet = PassMode::FinalIfQuiet(Duration::from_millis(400));
        let out = s.settle(
            &tok,
            window(&s, &[(4, 0), (5, 20)]),
            &PassConfig::default(),
            quiet,
        );
        assert_eq!(out.tentative, " York");
        assert_eq!(
            out.last_word_end,
            Some((21 * FRAME) as f64 / SAMPLE_RATE as f64)
        );
        assert!(
            !out.last,
            "speech ended 0.32 s before the audio, less than 0.4 s"
        );
    }

    /// The special words of the active utterance and of those waiting for their final pass.
    fn lists(s: &Streamer) -> (Option<Arc<Compiled>>, Vec<Option<Arc<Compiled>>>) {
        (
            s.active.as_ref().and_then(|a| a.stream.words.clone()),
            s.closing.iter().map(|a| a.stream.words.clone()).collect(),
        )
    }

    fn same(a: &Option<Arc<Compiled>>, b: &Option<Arc<Compiled>>) -> bool {
        match (a, b) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        }
    }

    #[test]
    #[ignore = "needs a model, named by GRANITE_SPEECH_DIR"]
    fn each_utterance_keeps_the_list_it_started_with() {
        let dir = std::env::var_os("GRANITE_SPEECH_DIR").expect("set GRANITE_SPEECH_DIR");
        let model = Arc::new(Model::load(dir.as_ref(), &crate::Options::default()).unwrap());
        let mut s = Streamer::new(model, StreamConfig::default());
        let mut seed = 0x9e37_79b9u64;
        let mut noise = |seconds: f64, level: f32| -> Vec<f32> {
            (0..(seconds * SAMPLE_RATE as f64) as usize)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * level
                })
                .collect()
        };
        let list = |word: &str| {
            let mut w = SpecialWords::new();
            w.add(word).unwrap();
            w
        };
        s.set_special_words(&list("kubectl"));
        let a = s.words.clone();
        let quiet = 0.0005;
        s.push(&noise(1.0, quiet));
        s.push(&noise(1.5, 0.2));
        assert!(same(&lists(&s).0, &a), "the first utterance has list A");
        s.set_special_words(&list("serde"));
        let b = s.words.clone();
        assert!(!same(&a, &b));
        assert!(same(&lists(&s).0, &a), "it keeps A after the list changes");
        // Speech ends, and the next utterance starts before the first had its final pass.
        s.push(&noise(1.0, quiet));
        s.push(&noise(1.5, 0.2));
        let (active, closing) = lists(&s);
        assert!(same(&active, &b), "the second utterance has list B");
        assert_eq!(closing.len(), 1);
        assert!(same(&closing[0], &a), "the closing one keeps A");
        s.set_special_words(&SpecialWords::new());
        assert!(s.words.is_none());
        s.push(&noise(1.0, quiet));
        s.push(&noise(1.5, 0.2));
        let (active, closing) = lists(&s);
        assert!(active.is_none(), "the third utterance has no list");
        assert_eq!(closing.len(), 2);
        assert!(same(&closing[0], &a) && same(&closing[1], &b));
    }

    #[test]
    fn a_word_heard_again_shares_the_end_of_the_settled_one() {
        assert!(shares_end("inselves", "themselves"));
        assert!(shares_end("selves", "themselves"));
        assert!(!shares_end("at", "that"));
        assert!(!shares_end("useful", "careful"));
        assert!(!shares_end("wonderful", "careful"));
    }

    #[test]
    fn ctc_runs_collapse_and_resume_where_they_end() {
        // With blank 0, the frames are "0 5 5 0 5 7 7 0"
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
            Resume::Frame(_) => unreachable!(),
        }
    }
}
