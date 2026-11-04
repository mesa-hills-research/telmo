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
//! ends at least a second before the audio does, and the model has heard a word after it. With
//! Parakeet, nothing settles before the utterance has 3 s of audio. A spoken form written as its
//! word settles whole, and words where a spoken form or a special word could still be under way
//! wait. Fillers a list leaves out go from each pass's words before any settle, so tentative
//! text never shows them. Settled words never change. When the utterance ends, one
//! last pass settles everything, except a sentence end at its end, which waits for the next
//! utterance's first words, since a pause can split a sentence. With a list that leaves out
//! disfluencies, nothing settles before that last pass, which settles the utterance cleaned.
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

use crate::biasing::{
    self, Compiled, Entries, SpecialWords, Spoken, Unit, normalized_words, place,
};
use crate::error::Result;
use crate::features::LogMelStream;
use crate::fillers::{self, Context, Protected};
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
    /// How long before the end of the audio a word must end to settle. Parakeet's words wait at
    /// least 4 s, since its hypothesis of a word can change until it has heard that much after
    /// it. A word said twice ("yeah, yeah") is heard once on less, and would settle so.
    pub settle_margin: Duration,
    /// A pass's first words are dropped when they repeat the last settled words and end within
    /// this long after them, since they're the settled words heard again.
    pub repeat_window: Duration,
}

impl PassConfig {
    /// This config as a pass uses it, with the settle margin at least
    /// [`PARAKEET_SETTLE_MARGIN`] for Parakeet.
    fn for_parakeet(&self, parakeet: bool) -> PassConfig {
        let mut config = self.clone();
        if parakeet {
            config.settle_margin = config.settle_margin.max(PARAKEET_SETTLE_MARGIN);
        }
        config
    }
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
    /// The text settled so far, which filler removal goes on from.
    settled: String,
    /// The last words the stream's previous utterance settled, which filler removal goes on from
    /// too, so a stutter split by a pause loses its second copy.
    carried: String,
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
            settled: String::new(),
            carried: String::new(),
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
        let config = &config.for_parakeet(matches!(model, Model::Parakeet(_)));
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
        // Fillers go before anything settles, going on from the settled text, so a word whose
        // fate depends on the next one is the last of the pass until that one comes, and what
        // protects a word from going may have settled before it.
        let settled_context = Context::after_carried(&self.carried, &self.settled);
        let hyp = match self.words.as_ref() {
            Some(words) => remove_fillers(words, tokenizer, &settled_context, hyp),
            None => hyp,
        };
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
        let early = matches!(self.resume, Resume::Frame(_))
            && self.seconds() < PARAKEET_FIRST_SETTLE.as_secs_f64();
        if last {
            settle = words.len();
        } else if !early {
            // The last word of a pass never settles. It may still grow, and near the end of the
            // audio a model may close it with punctuation that the next audio contradicts.
            while settle + 1 < words.len()
                && settle < self.previous.len()
                && word_ids(&hyp, &words[settle]) == self.previous[settle]
                && hyp[words[settle].end - 1].end_frame * FRAME <= limit
            {
                settle += 1;
            }
            // Nothing settles from where a spoken form or a special word could still be under
            // way, since the words from the first that can't settle on may still change. So a
            // word that one may still protect from filler removal settles only once it's whole
            // or can no longer be. A form or word holds back at most its own words.
            let ids: Vec<u32> = heard.iter().map(|h| h.id).collect();
            while settle > 0
                && let Some(words_list) = self.words.as_deref()
            {
                let frontier = hyp[words[settle].start - 1].heard_end;
                let Some(open) =
                    words_list.open_before(tokenizer, &ids, frontier, &settled_context)
                else {
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
        self.settled.push_str(&settled);
        let (settled, tentative) = match self.words.as_deref() {
            Some(words) if words.disfluencies.is_some() => {
                let (settled, tentative) = hold_until_last(words, &self.settled, &tentative, last);
                if last {
                    // The next utterance goes on from the text this one settled, cleaned.
                    self.settled = settled.clone();
                }
                (settled, tentative)
            }
            _ => (settled, tentative),
        };
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
    /// can't start a word are such ends. The pass's words are formed over `context`, what it
    /// heard of the settled audio before `hyp`, and the new tokens together, so a word that
    /// starts on the settled audio stays whole. [`heard_again`] matches the last settled words
    /// to them, and the leading new words it matches go if they end within
    /// `config.repeat_window` of the settled text.
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
        let mut skip = 0;
        let mut prev = self.last_settled;
        // A CTC token whose run goes on from the settled audio is the token heard there.
        if matches!(self.resume, Resume::Ctc { .. })
            && let (Some(last), Some(first)) = (context.last(), hyp.first())
            && first.id == last.id
            && first.frame == last.end_frame
        {
            prev = Some(first.id);
            skip = 1;
        }
        // A settled word was followed by another before it settled, so it's complete, and
        // leading tokens that can't start a word are left over from it or from words before it.
        while skip < hyp.len() && !tokenizer.may_split_before(prev, hyp[skip].id) {
            prev = Some(hyp[skip].id);
            skip += 1;
        }
        let all: Vec<Hyp> = context.iter().chain(&hyp).cloned().collect();
        let old = context.len() + skip;
        let mut words = split_words(tokenizer, None, &all[..old]);
        words.extend(
            split_words(tokenizer, prev, &all[old..])
                .into_iter()
                .map(|w| Word {
                    start: old + w.start,
                    end: old + w.end,
                }),
        );
        let texts: Vec<String> = words
            .iter()
            .map(|w| normalized(tokenizer, &word_ids(&all, w)))
            .collect();
        let frames: Vec<Range<usize>> = words
            .iter()
            .map(|w| all[w.start].frame..all[w.end - 1].end_frame)
            .collect();
        let recent = &self.recent[self.recent.len().saturating_sub(MAX_REPEAT)..];
        let again = heard_again(recent, &texts, &frames);
        let window = (config.repeat_window.as_secs_f64() * SAMPLE_RATE as f64 / FRAME as f64).ceil()
            as usize;
        let limit = self.resume.frame() + window;
        let mut end = old;
        for (k, w) in words.iter().enumerate().filter(|(_, w)| w.start >= old) {
            if !again[k] || frames[k].end > limit {
                break;
            }
            end = w.end;
        }
        hyp.drain(..end - context.len());
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
        // A word whose run now starts before them still counts, from their first frame.
        let mut heard = ctc_hypothesis(&ids[..frame.min(ids.len())], from, u32::MAX, blank);
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
    /// The utterance ended. Its speech ended at `speech_end`, and all its text has settled, except
    /// a sentence end at its end, which stays tentative text of this utterance until the next
    /// one's first words settle or the stream finishes.
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
    /// The latest 16 kHz audio, as far back as voice activity can begin an utterance.
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
    /// The sentence end that closed an utterance, held as tentative text until the next
    /// utterance's first words settle and show whether the sentence goes on, as the utterance's
    /// number and the mark.
    held: Option<(u64, char)>,
    /// Utterances started so far.
    started: u64,
    /// The settled text of the last utterance whose final pass ran, which the next one's filler
    /// removal goes on from.
    last_text: String,
    /// The special words new utterances favour.
    words: Option<Arc<Compiled>>,
}

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
            held: None,
            started: 0,
            words: None,
            last_text: String::new(),
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
        if audio.is_empty() {
            return Vec::new();
        }
        self.history.extend(audio.iter().copied());
        self.total += audio.len() as u64;
        if let Some(a) = &mut self.active {
            if a.end.is_none() {
                a.stream.push(&audio);
            } else {
                a.after_end.extend_from_slice(&audio);
            }
        }
        let found = self.vad.push(&audio);
        let events = self.on_voice_activity(found);
        let excess = (self.history.len() as u64).saturating_sub(self.vad.look_back()) as usize;
        self.history.drain(..excess);
        self.history_start += excess as u64;
        events
    }

    /// Starts and ends utterances where voice activity `found` speech start and end.
    fn on_voice_activity(&mut self, found: Vec<VadEvent>) -> Vec<Event> {
        let mut events = Vec::new();
        for event in found {
            match event {
                VadEvent::Start { audio_from, onset } => {
                    // After a pause the stream still holds, the utterance starts where the last
                    // one's audio ends, so no audio is in both and quiet words between them, which
                    // voice activity heard as neither, are in this one.
                    let audio_from = match (self.active.as_ref()).or(self.closing.back()) {
                        Some(last) if last.end.is_some() => {
                            let end = last.start + last.stream.audio().len() as u64;
                            if end <= onset && onset - end <= self.vad.look_back() {
                                end.max(self.history_start)
                            } else {
                                audio_from
                            }
                        }
                        _ => audio_from,
                    };
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
        if a.stream.passes() == 0 {
            // The previous utterance's final pass runs before this one's first, so its text is
            // whole by now.
            a.stream.carried = last_words(&self.last_text);
        }
        a.seen = a.stream.audio().len();
        let out = a.stream.pass(&self.model, &self.config.passes, mode)?;
        let mut settled = out.settled;
        if !self.wrote {
            settled = settled.trim_start().to_string();
        }
        if !settled.is_empty()
            && let Some(held) = self.held.take()
        {
            let (resolved, text) =
                resolve_held(held, &self.last_text, &settled, a.stream.words.as_deref());
            events.extend(resolved);
            settled = text;
        }
        if out.last
            && let Some(mark) = settled.chars().last().filter(|c| SENTENCE_END.contains(c))
        {
            settled.pop();
            self.held = Some((a.index, mark));
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
        self.last_text = std::mem::take(&mut a.stream.settled);
        let now = a.start + a.stream.audio().len() as u64;
        let (speech_end, decided_at) = a.end.unwrap_or((a.reopened.unwrap_or(now), now));
        events.push(Event::UtteranceEnd {
            utterance: a.index,
            at: decided_at as f64 / SAMPLE_RATE as f64,
            speech_end: speech_end as f64 / SAMPLE_RATE as f64,
        });
        if let Some((utterance, mark)) = self.held
            && utterance == a.index
        {
            events.push(Event::Tentative {
                utterance,
                text: mark.to_string(),
            });
        }
        Ok(true)
    }

    /// Ends the input. Any utterance ends where the audio does, and gets its final pass.
    pub fn finish(&mut self) -> Result<Vec<Event>> {
        let mut audio = Vec::new();
        self.resampler.flush(&mut audio);
        let mut events = self.take(audio);
        // Audio shorter than voice activity waits for before it decides is decided now.
        let found = self.vad.flush();
        events.extend(self.on_voice_activity(found));
        if let Some(mut a) = self.active.take() {
            if a.end.is_none() {
                a.end = Some((self.total, self.total));
            }
            self.closing.push_back(a);
        }
        while let Some(mut a) = self.closing.pop_front() {
            self.finish_utterance(&mut a, PassMode::Final, &mut events)?;
        }
        if let Some((utterance, mark)) = self.held.take() {
            events.push(Event::Settled {
                utterance,
                text: mark.to_string(),
            });
            events.push(Event::Tentative {
                utterance,
                text: String::new(),
            });
        }
        Ok(events)
    }
}

/// Marks that end a sentence.
const SENTENCE_END: [char; 3] = ['.', '?', '!'];

/// The events that settle or drop the sentence end `held` (an utterance's number and its mark)
/// once the next utterance's first words, `settled`, settle, and those words to settle. The mark
/// goes when the sentence goes on, and the next words take a capital when they start a new one.
/// Either way it leaves the held utterance's tentative text. When the sentence goes on with a
/// copy of the held utterance's last word, `previous` ending in it, and that's a word that can't
/// end a clause, the copy is a stutter and goes with the mark ("...and." then "and then").
fn resolve_held(
    held: (u64, char),
    previous: &str,
    settled: &str,
    words: Option<&Compiled>,
) -> (Vec<Event>, String) {
    let (utterance, mark) = held;
    let goes_on = sentence_goes_on(settled, words);
    let settled = match goes_on {
        Some(true) => without_a_copy_of(previous, settled),
        _ => settled,
    };
    let mut events = Vec::new();
    if goes_on != Some(true) {
        events.push(Event::Settled {
            utterance,
            text: mark.to_string(),
        });
    }
    events.push(Event::Tentative {
        utterance,
        text: String::new(),
    });
    let text = match goes_on {
        Some(false) => capitalized(settled, words),
        _ => settled.to_string(),
    };
    (events, text)
}

/// `settled` without its first word when that's a copy of the last word of `previous` that can't
/// end a clause.
fn without_a_copy_of<'a>(previous: &str, settled: &'a str) -> &'a str {
    let core = |w: &str| fillers::split(w).1.to_lowercase();
    let (Some(last), Some(first)) = (
        previous.split_whitespace().next_back(),
        settled.split_whitespace().next(),
    ) else {
        return settled;
    };
    let word = core(first);
    if word != core(last) || !fillers::REPEATED.contains(&word.as_str()) {
        return settled;
    }
    let rest = settled.trim_start();
    // Keep the space before the copy, which separates what follows from the earlier text.
    let space = &settled[..settled.len() - rest.len()];
    match rest.find(char::is_whitespace) {
        Some(end) => &settled[space.len() + end..],
        None => "",
    }
}

/// The last words of an utterance's settled `text`, as many as filler removal goes on from.
fn last_words(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    words[words.len().saturating_sub(SpecialWords::MAX_PARTS)..].join(" ")
}

/// Whether `text`, the first settled words of an utterance after one that ended with a sentence
/// end, goes on with that sentence (`Some(true)`), starts a new one (`Some(false)`), or is left
/// as it is (`None`). Granite writes an utterance's first word with a capital even where the
/// sentence goes on, and sometimes a new sentence's first word without one, so a capital leaves
/// the text as it is, a lower-case word that joins words (and, of, which) goes on, and another
/// lower-case word starts a new sentence. A comma first goes on too.
fn sentence_goes_on(text: &str, words: Option<&Compiled>) -> Option<bool> {
    let text = text.trim_start();
    if text.starts_with([',', ';', ':']) {
        return Some(true);
    }
    let (_, core, _) = fillers::split(text.split_whitespace().next()?);
    if core.is_empty() || core.starts_with(char::is_uppercase) {
        return None;
    }
    let lower = core.to_lowercase();
    if words.is_some_and(|w| w.protected.keeps_case(&lower)) {
        return None;
    }
    Some(JOINING.contains(&lower.as_str()))
}

/// Words that join a sentence's words rather than start one.
const JOINING: [&str; 33] = [
    "and", "or", "but", "nor", "yet", "rather", "because", "which", "who", "whom", "whose", "that",
    "than", "as", "to", "of", "in", "on", "at", "for", "with", "by", "from", "into", "about", "if",
    "when", "while", "where", "though", "although", "unless", "until",
];

/// `text` with its first word capitalized, if that's plain lower case and not a special word.
fn capitalized(text: &str, words: Option<&Compiled>) -> String {
    let start = text.len() - text.trim_start().len();
    let rest = &text[start..];
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    let none = Protected::default();
    let protected = words.map_or(&none, |w| &w.protected);
    match fillers::with_capital(&rest[..end], protected) {
        Some(word) => format!("{}{word}{}", &text[..start], &rest[end..]),
        None => text.to_string(),
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

/// `hyp` with `words`' fillers left out, if any are to go, after the settled text, whose
/// `context` this is. Special words, spoken forms' words and spotted words keep their spelling.
/// A word's new tokens share its frames and resume where it did.
fn remove_fillers(
    words: &Compiled,
    tokenizer: &Tokenizer,
    context: &Context,
    hyp: Vec<Hyp>,
) -> Vec<Hyp> {
    // With a disfluency tagger, the fillers stay until the utterance ends, as the tagger reads
    // them.
    let Some(fillers) = words
        .fillers
        .as_ref()
        .filter(|_| words.disfluencies.is_none())
    else {
        return hyp;
    };
    fillers.rewrite(
        tokenizer,
        hyp,
        |h| h.id,
        context,
        &words.protected,
        |word, ids| {
            let (first, last) = (&word[0], &word[word.len() - 1]);
            (ids.iter().enumerate())
                .map(|(k, &id)| Hyp {
                    id,
                    frame: place(first.frame, last.frame, ids.len(), k),
                    end_frame: last.end_frame,
                    resume: last.resume.clone(),
                    joined: k > 0 || first.joined,
                    heard_end: last.heard_end,
                    entry: None,
                })
                .collect()
        },
    )
}

/// The settled and tentative text of a pass when `words` leave out disfluencies, given the
/// utterance's text settled so far, fillers in, and the pass's tentative text. Nothing settles
/// before the `last` pass, which settles the whole utterance cleaned. Until then the tentative
/// text is the utterance so far without its fillers.
fn hold_until_last(
    words: &Compiled,
    utterance: &str,
    tentative: &str,
    last: bool,
) -> (String, String) {
    let text = format!("{utterance}{tentative}");
    let list: Vec<&str> = text.split_whitespace().collect();
    let edits = match &words.disfluencies {
        Some(tagger) if last => tagger.plan(&list, &words.protected),
        _ => fillers::Fillers::for_language("en")
            .expect("English has filler rules")
            .plan(&Context::default(), &list, &words.protected),
    };
    let mut text = fillers::Fillers::text(&list, &edits);
    if !text.is_empty() {
        text.insert(0, ' ');
    }
    if last {
        (text, String::new())
    } else {
        (String::new(), text)
    }
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

/// Audio a Parakeet utterance needs before any of its words settle. Parakeet decodes the window
/// from the utterance's start with its features normalized over the window, and over the first
/// seconds its hypothesis of the first words still changes as more audio comes, often hearing a
/// word said twice as one ("No, it" where more audio hears "No no no it").
const PARAKEET_FIRST_SETTLE: Duration = Duration::from_millis(3000);

/// The least time before the end of the audio a Parakeet word must end to settle. Parakeet's
/// hypothesis of a word can still change until it has heard about that much after it, and a word
/// said twice ("yeah, yeah") is often heard once on less. The cost is that a word settles about
/// 4 s after it's said, and its tentative text changes more often until then.
const PARAKEET_SETTLE_MARGIN: Duration = Duration::from_secs(4);

/// The most settled words a pass may repeat.
const MAX_REPEAT: usize = 3;

/// Frames (240 ms) after a settled word's frames within which a later pass may place the same
/// word. On the development meetings, a pass placed a settled word at most two frames later than
/// the pass that settled it, and a word said again came three frames or more after it.
const FIT: usize = 3;

/// The settled words kept once they're joined. The repeat check and Granite's re-alignment use
/// the last [`MAX_REPEAT`], and a later pass can complete an entry's word or form whose first
/// words, at most all but one of [`SpecialWords::MAX_PARTS`], are the last ones kept.
const KEPT: usize = if SpecialWords::MAX_PARTS - 1 > MAX_REPEAT {
    SpecialWords::MAX_PARTS - 1
} else {
    MAX_REPEAT
};

/// Which words of a pass, normalized as `texts` and heard on `frames`, hear the settled words
/// `settled` again. Each settled word, oldest first, takes the first word after those taken
/// that starts one of its spellings and fits its frames, so each stands for one word of the
/// pass, and a copy of it said again stays. A word fits when its frames overlap the settled
/// word's or start less than [`FIT`] frames after them. A word on the settled word's frames may
/// also be just the end of a spelling ("control" of "cube control"), as the pass's window began
/// inside it, or the word split another way with only its end the same ("themselves" heard
/// again as "inselves").
fn heard_again(settled: &[SettledWord], texts: &[String], frames: &[Range<usize>]) -> Vec<bool> {
    let mut taken = vec![false; texts.len()];
    let mut from = 0;
    for s in settled {
        let fits = |f: Range<usize>| {
            f.start < s.heard.end && f.end > s.heard.start
                || (s.heard.end..s.heard.end + FIT).contains(&f.start)
        };
        let found = (from..texts.len()).find_map(|j| {
            let rest = &texts[j..];
            let whole = s
                .spellings
                .iter()
                .filter(|sp| !sp.is_empty() && rest.starts_with(sp))
                .map(Vec::len)
                .max();
            let inside = frames[j].start < s.heard.end && frames[j].end > s.heard.start;
            let end = || {
                let ends = s
                    .spellings
                    .iter()
                    .flat_map(|sp| (1..sp.len()).map(|t| &sp[t..]));
                ends.filter(|end| rest.starts_with(end))
                    .map(<[String]>::len)
                    .max()
            };
            let ends_like = |sp: &Vec<String>| sp.len() == 1 && shares_end(&rest[0], &sp[0]);
            let same_end = || s.spellings.iter().any(ends_like).then_some(1);
            let k = whole.or_else(|| inside.then(|| end().or_else(same_end)).flatten())?;
            fits(frames[j].start..frames[j + k - 1].end).then_some((j, k))
        });
        if let Some((j, k)) = found {
            taken[j..j + k].fill(true);
            from = j + k;
        }
    }
    taken
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
        (stream_for(words, &tok, frames), tok)
    }

    /// The same with tokenizer `tok`.
    fn stream_for(words: &SpecialWords, tok: &Tokenizer, frames: usize) -> UtteranceStream {
        let (compiled, rejected) = Compiled::new(words, tok);
        assert!(rejected.is_empty(), "{rejected:?}");
        UtteranceStream {
            audio: vec![0.0; frames * FRAME],
            resume: Resume::Frame(0),
            last_settled: None,
            recent: Vec::new(),
            previous: Vec::new(),
            granite: None,
            log_mel: None,
            settled: String::new(),
            carried: String::new(),
            passes: 0,
            words: Some(Arc::new(compiled)),
            timings: Timings::default(),
        }
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

    /// A list that leaves out English fillers.
    fn fillers() -> SpecialWords {
        let mut words = SpecialWords::new();
        words.set_fillers(crate::Fillers::for_language("en"));
        words
    }

    /// "uh" in the test tokenizer's pieces, on `frame`.
    fn uh(frame: usize) -> [(u32, usize); 3] {
        [(11, frame), (13, frame), (26, frame)]
    }

    #[test]
    fn hesitations_never_show_and_a_stutter_settles_once() {
        let (mut s, tok) = stream_with(&fillers(), 50);
        let config = PassConfig::default();
        let mut settled = String::new();
        let mut shown = Vec::new();
        // "the uh", then "the uh the city the", "uh" on frame 2 and the copy on frame 3.
        let first: Vec<(u32, usize)> = [(6, 0)].into_iter().chain(uh(2)).collect();
        let later: Vec<(u32, usize)> = (first.iter().copied())
            .chain([(6, 3), (44, 4), (6, 40)])
            .collect();
        for (heard, mode) in [
            (&first, PassMode::Tentative),
            (&first, PassMode::Tentative),
            (&later, PassMode::Tentative),
            (&later, PassMode::Tentative),
            (&later, PassMode::Final),
        ] {
            let out = s.settle(&tok, window(&s, heard), &config, mode);
            settled.push_str(&out.settled);
            shown.push(format!("{settled}|{}", out.tentative));
        }
        // "the" waits while only a hesitation follows it, and then one copy settles.
        assert_eq!(shown[1], "| the");
        assert_eq!(settled, " the city the");
        assert!(shown.iter().all(|s| !s.contains("uh")), "{shown:?}");
    }

    #[test]
    fn with_disfluencies_an_utterance_settles_once_cleaned_when_it_ends() {
        let mut words = SpecialWords::new();
        let tagger = crate::disfluency::tests::tiny(2.0, 128, crate::Precision::F32);
        words.set_disfluencies(Some(tagger));
        let (mut s, tok) = stream_with(&words, 50);
        let config = PassConfig::default();
        let first: Vec<(u32, usize)> = [(6, 0)].into_iter().chain(uh(2)).collect();
        let later: Vec<(u32, usize)> = (first.iter().copied())
            .chain([(6, 3), (44, 4), (6, 40)])
            .collect();
        let mut shown = Vec::new();
        for heard in [&first, &first, &later, &later] {
            let out = s.settle(&tok, window(&s, heard), &config, PassMode::Tentative);
            assert_eq!(out.settled, "");
            shown.push(out.tentative);
        }
        // The tentative text is the utterance so far, fillers left out.
        assert_eq!(shown, [" the", " the", " the city the", " the city the"]);
        let out = s.settle(&tok, window(&s, &later), &config, PassMode::Final);
        assert_eq!(
            (out.settled.as_str(), out.tentative.as_str()),
            (" the city the", "")
        );
    }

    #[test]
    fn a_spoken_forms_word_of_list_words_stays_whole() {
        let mut words = fillers();
        words.add_spoken("the the", "cube control").unwrap();
        let (mut s, tok) = stream_with(&words, 50);
        let heard = [(4, 1), (5, 2), (44, 30)];
        let out = s.settle(
            &tok,
            window(&s, &heard),
            &PassConfig::default(),
            PassMode::Tentative,
        );
        assert_eq!(out.tentative, " the the city");
    }

    /// A tokenizer whose words are whole pieces, `words[i]` with id `i + 1`.
    fn word_tokenizer(words: &[&str]) -> Tokenizer {
        let vocab: Vec<String> = (words.iter().enumerate())
            .map(|(i, w)| format!("\"▁{w}\":{}", i + 1))
            .collect();
        let json = format!(
            r#"{{"added_tokens":[{{"id":0,"content":"<blank>","special":true}}],"model":{{"type":"BPE","vocab":{{"<blank>":0,{}}},"merges":[]}}}}"#,
            vocab.join(",")
        );
        Tokenizer::from_json(json.as_bytes()).unwrap()
    }

    #[test]
    fn a_quote_protects_its_words_after_its_start_settles() {
        let (mut s, _) = stream_with(&fillers(), 50);
        let tok = word_tokenizer(&[
            "“the", "city", "of", "New", "York", "is", "very", "um", "big”",
        ]);
        // A word every 480 ms. The second pass settles up to "very", inside the quote.
        let heard: Vec<(u32, usize)> = (1..=9).map(|id| (id, 6 * (id as usize - 1))).collect();
        let config = PassConfig::default();
        s.settle(&tok, window(&s, &heard), &config, PassMode::Tentative);
        let second = s.settle(&tok, window(&s, &heard), &config, PassMode::Tentative);
        assert_eq!(second.settled, " “the city of New York is very");
        let last = s.settle(&tok, window(&s, &heard), &config, PassMode::Final);
        assert_eq!(last.settled, " um big”");
    }

    #[test]
    fn a_quote_protects_its_words_before_its_closing_mark_is_heard() {
        let (mut s, _) = stream_with(&fillers(), 50);
        let tok = word_tokenizer(&[
            "“the", "city", "um", "of", "New", "York", "is", "very", "big", "big”",
        ]);
        // Two passes without the closing mark settle past "um", then the last pass hears it.
        let open: Vec<(u32, usize)> = (1..=9).map(|id| (id, 6 * (id as usize - 1))).collect();
        let mut closed = open.clone();
        closed[8].0 = 10;
        let config = PassConfig::default();
        s.settle(&tok, window(&s, &open), &config, PassMode::Tentative);
        let second = s.settle(&tok, window(&s, &open), &config, PassMode::Tentative);
        let last = s.settle(&tok, window(&s, &closed), &config, PassMode::Final);
        assert_eq!(
            format!("{}{}", second.settled, last.settled),
            " “the city um of New York is very big”"
        );
    }

    #[test]
    fn an_utterance_goes_on_from_the_last_ones_settled_words() {
        let tok = word_tokenizer(&["the", "That's", "and", "then", "um", "product", "big"]);
        for (carried, heard, want) in [
            // A stutter split by a pause loses its second copy, whatever its case.
            ("to design the", &[1, 6][..], " product"),
            ("for something that's", &[2, 7], " big"),
            // A sentence end between the copies keeps both, as in text.
            ("and.", &[3, 4], " and then"),
            // A quote the last utterance left open doesn't go on.
            ("He said “the city", &[5, 7], " big"),
        ] {
            let mut s = stream_for(&fillers(), &tok, 50);
            s.carried = carried.to_string();
            let heard: Vec<(u32, usize)> = (heard.iter().enumerate())
                .map(|(i, &id)| (id, 6 * i))
                .collect();
            let out = s.settle(
                &tok,
                window(&s, &heard),
                &PassConfig::default(),
                PassMode::Final,
            );
            assert_eq!(out.settled, want, "{carried}");
        }
    }

    /// The settled text of a stream over `frames` frames with `words`, after two passes that hear
    /// `first` and a final pass that hears `last`, and the text offline removal gives `last`.
    fn streamed_and_offline(
        words: &SpecialWords,
        tok: &Tokenizer,
        frames: usize,
        first: &[(u32, usize)],
        last: &[(u32, usize)],
    ) -> (String, String) {
        let mut s = stream_for(words, tok, frames);
        let config = PassConfig {
            settle_margin: Duration::from_millis(320),
            ..PassConfig::default()
        };
        let mut text = String::new();
        for (heard, mode) in [
            (first, PassMode::Tentative),
            (first, PassMode::Tentative),
            (last, PassMode::Final),
        ] {
            text.push_str(&s.settle(tok, window(&s, heard), &config, mode).settled);
        }
        let compiled = s.words.clone().unwrap();
        let offline = remove_fillers(&compiled, tok, &Context::default(), hyps(last).collect());
        let offline = tok.decode_raw(&offline.iter().map(|h| h.id).collect::<Vec<_>>());
        (text, offline)
    }

    #[test]
    fn a_special_spellings_unfinished_start_waits_for_the_rest() {
        let mut words = fillers();
        words.add_code_word("New um York city control").unwrap();
        let tok = crate::biasing::tests::tokenizer();
        // Two passes hear "New um York city", and the last hears "control" after it.
        let start = [(42, 0), (11, 3), (13, 3), (79, 3), (43, 6), (44, 10)];
        let mut whole = start.to_vec();
        whole.push((5, 18));
        let (streamed, offline) = streamed_and_offline(&words, &tok, 20, &start, &whole);
        assert_eq!(offline, " New um York city control");
        assert_eq!(streamed, offline);
    }

    #[test]
    fn a_special_spelling_is_whole_or_not_at_every_settle_frontier() {
        let tok = crate::biasing::tests::tokenizer();
        let ids = |word: &str| -> Vec<u32> {
            match word {
                "um" => vec![11, 13, 79],
                "cube" => vec![4],
                "control" => vec![5],
                "the" => vec![6],
                "New" => vec![42],
                "York" => vec![43],
                "city" => vec![44],
                "plus" => vec![71],
                "sharp" => vec![76],
                _ => unreachable!("{word}"),
            }
        };
        let base = ["New", "York", "city", "plus", "sharp"];
        // Five-word spellings with a hesitation at each place, or a listed double at each pair.
        let mut spellings: Vec<Vec<&str>> = Vec::new();
        for at in 0..5 {
            let mut words = base[..4].to_vec();
            words.insert(at, "um");
            spellings.push(words);
        }
        for at in 0..4 {
            let mut words = base[..3].to_vec();
            words.splice(at..at, ["the", "the"]);
            spellings.push(words);
        }
        // A word every 480 ms after "cube", so a copy said again is new speech.
        let heard = |words: &[&str]| -> Vec<(u32, usize)> {
            (std::iter::once("cube")
                .chain(words.iter().copied())
                .enumerate())
            .flat_map(|(k, w)| ids(w).into_iter().map(move |id| (id, 6 * k)))
            .collect()
        };
        for spelling in &spellings {
            let mut words = fillers();
            words.add_code_word(&spelling.join(" ")).unwrap();
            // The first passes hear the spelling's first words, so a frontier falls after each of
            // them, then the last pass hears the rest of it, or another word in its place.
            for heard_first in 1..=5 {
                let first = heard(&spelling[..heard_first]);
                let whole: Vec<&str> = spelling.iter().copied().chain(["cube"]).collect();
                let mut other = spelling[..heard_first.min(4)].to_vec();
                other.extend(["control", "cube"]);
                for (last, complete) in [(heard(&whole), true), (heard(&other), false)] {
                    let (streamed, offline) = streamed_and_offline(&words, &tok, 50, &first, &last);
                    let case = format!("{spelling:?} after {heard_first} words, {complete}");
                    assert_eq!(streamed, offline, "{case}");
                    let kept = offline.contains(&spelling.join(" "));
                    assert_eq!(kept, complete, "{case}");
                }
            }
        }
    }

    #[test]
    fn a_special_spelling_settles_whole_with_its_fillers() {
        let mut words = fillers();
        words.add_code_word("New um").unwrap();
        let (mut s, tok) = stream_with(&words, 50);
        // "New" on frame 40, "um" on frame 46 and "city" on frame 49. Only "New" ends early
        // enough to settle, and it waits, since "um" may still change.
        let heard = [(42, 40), (11, 46), (13, 46), (79, 46), (44, 49)];
        let config = PassConfig {
            settle_margin: Duration::from_millis(320),
            ..PassConfig::default()
        };
        s.settle(&tok, window(&s, &heard), &config, PassMode::Tentative);
        let second = s.settle(&tok, window(&s, &heard), &config, PassMode::Tentative);
        assert_eq!(second.settled, "");
        let last = s.settle(&tok, window(&s, &heard), &config, PassMode::Final);
        assert_eq!(last.settled, " New um city");
        // A settled "New" with the rest of the word after it keeps "um" too.
        let (mut s, tok) = stream_with(&words, 10);
        let heard = [(42, 0), (11, 6), (13, 6), (79, 6), (44, 9)];
        s.recent = vec![settled(&tok, &[42], 0..1)];
        s.settled = " New".into();
        s.resume = Resume::Frame(1);
        s.last_settled = Some(42);
        let last = s.settle(&tok, window(&s, &heard), &config, PassMode::Final);
        assert_eq!(last.settled, " um city");
    }

    #[test]
    fn parakeets_first_words_wait_for_enough_audio() {
        let (mut s, tok) = stream_with(&SpecialWords::new(), 25);
        let config = PassConfig::default();
        // "the cube control" in the first 0.4 s, and a word near the end of 2 s of audio.
        let heard = [(6, 0), (4, 2), (5, 4), (6, 20)];
        s.settle(&tok, window(&s, &heard), &config, PassMode::Tentative);
        let second = s.settle(&tok, window(&s, &heard), &config, PassMode::Tentative);
        assert_eq!(
            (second.settled.as_str(), second.tentative.as_str()),
            ("", " the cube control the")
        );
        s.push(&vec![0.0; 13 * FRAME]);
        let third = s.settle(&tok, window(&s, &heard), &config, PassMode::Tentative);
        assert_eq!(third.settled, " the cube control");
    }

    #[test]
    fn parakeets_words_settle_at_least_four_seconds_after_they_end() {
        let tok = word_tokenizer(&["one", "two", "three"]);
        // 8 s of audio, "one" ending 4 s before its end, "two" 3.92 s before, and a last word.
        let heard = [(1, 49), (2, 50), (3, 90)];
        for (parakeet, want) in [(true, " one"), (false, " one two")] {
            let mut s = stream_for(&SpecialWords::new(), &tok, 100);
            if !parakeet {
                s.resume = Resume::Ctc {
                    frame: 0,
                    prev: u32::MAX,
                };
            }
            let config = PassConfig::default().for_parakeet(parakeet);
            s.settle(&tok, window(&s, &heard), &config, PassMode::Tentative);
            let second = s.settle(&tok, window(&s, &heard), &config, PassMode::Tentative);
            assert_eq!(second.settled, want, "Parakeet: {parakeet}");
        }
        // A longer margin asked for stays.
        let long = PassConfig {
            settle_margin: Duration::from_secs(6),
            ..PassConfig::default()
        };
        assert_eq!(
            long.for_parakeet(true).settle_margin,
            Duration::from_secs(6)
        );
    }

    #[test]
    fn a_copy_of_a_settled_word_goes_in_its_place() {
        let (mut s, tok) = stream_with(&fillers(), 50);
        s.recent = vec![settled(&tok, &[6], 0..1)];
        s.settled = " the".into();
        s.resume = Resume::Frame(1);
        s.last_settled = Some(6);
        // "the" again on frame 10, said after the settled one.
        let heard = [(6, 10), (44, 11), (6, 40)];
        let config = PassConfig::default();
        let out = s.settle(&tok, window(&s, &heard), &config, PassMode::Tentative);
        assert_eq!(out.tentative, " city the");
    }

    #[test]
    fn a_hesitations_sentence_end_follows_the_settled_text() {
        let tok = crate::biasing::tests::tokenizer();
        // "uh." after the settled "the city", then "New York the".
        let heard: Vec<(u32, usize)> = (uh(4).into_iter())
            .chain([(60, 4), (42, 5), (43, 6), (6, 40)])
            .collect();
        for (city, tentative) in [(&[44][..], ". New York the"), (&[44, 7], " New York the")] {
            let (mut s, _) = stream_with(&fillers(), 50);
            s.recent = vec![settled(&tok, &[6], 0..1), settled(&tok, city, 2..3)];
            s.settled = tok.decode_raw(&[&[6][..], city].concat());
            s.resume = Resume::Frame(3);
            s.last_settled = Some(city[city.len() - 1]);
            let config = PassConfig::default();
            let out = s.settle(&tok, window(&s, &heard), &config, PassMode::Tentative);
            // A comma that settled stays, and the sentence end goes with the hesitation.
            assert_eq!(out.tentative, tentative);
        }
    }

    #[test]
    fn a_held_sentence_end_goes_when_the_next_utterance_goes_on_with_the_sentence() {
        let held = (3, '.');
        let settled = |events: &[Event]| -> Vec<String> {
            (events.iter())
                .filter_map(|e| match e {
                    Event::Settled { utterance: 3, text } => Some(text.clone()),
                    _ => None,
                })
                .collect()
        };
        let cleared = Event::Tentative {
            utterance: 3,
            text: String::new(),
        };
        let mut words = SpecialWords::new();
        words.add_code_word("kubectl").unwrap();
        let tok = crate::biasing::tests::tokenizer();
        let words = Compiled::new(&words, &tok).0;
        for (next, mark, text) in [
            // "...speak your mind" then "rather than having to only type."
            (" rather than", None, " rather than"),
            // A copy of the held utterance's last list word that goes on with the sentence is a
            // stutter, and goes with the mark ("...and." then "and then").
            (" and then", None, " then"),
            (", and also", None, ", and also"),
            // A new sentence keeps the mark, with a capital its first word may lack.
            (" detailed design.", Some("."), " Detailed design."),
            (" i think so.", Some("."), " I think so."),
            (" The music came.", Some("."), " The music came."),
            (" kubectl get pods", Some("."), " kubectl get pods"),
            (" 5 minutes.", Some("."), " 5 minutes."),
            // The capital goes on the word, with its punctuation around it.
            (" hello.", Some("."), " Hello."),
            (" hello, world", Some("."), " Hello, world"),
            (" 'hello world", Some("."), " 'Hello world"),
        ] {
            let (events, out) = resolve_held(held, " so we test it and.", next, Some(&words));
            let want: Vec<String> = mark.into_iter().map(String::from).collect();
            assert_eq!((settled(&events), out.as_str()), (want, text), "{next}");
            assert!(events.contains(&cleared), "{next}");
        }
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

    /// The tentative text of a Granite pass that hears `tokens` as (id, frame, end frame), the
    /// first `context` of them on the settled audio, after `ids` settled on `frames`.
    fn granite_pass(
        ids: &[u32],
        frames: Range<usize>,
        tokens: &[(u32, usize, usize)],
        context: usize,
    ) -> String {
        let (mut s, tok) = stream_with(&SpecialWords::new(), 50);
        s.recent = vec![settled(&tok, ids, frames.clone())];
        s.last_settled = ids.last().copied();
        s.resume = Resume::Ctc {
            frame: frames.end,
            prev: ids[ids.len() - 1],
        };
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
        let heard = Heard {
            tokens,
            new: context,
        };
        s.settle(&tok, heard, &PassConfig::default(), PassMode::Tentative)
            .tentative
    }

    #[test]
    fn a_copy_right_after_the_settled_word_heard_on_its_frames_is_the_same_word() {
        // Granite settled "the" on frames 0 and 1, and a later pass aligns it on frames 0 to 2,
        // so the frames after the settled text start with the rest of that run.
        let rest = [(6, 0, 2), (6, 2, 3), (42, 40, 41)];
        assert_eq!(granite_pass(&[6], 0..2, &rest, 1), " New");
        // After a frame of something else, "the" was said again.
        let again = [(6, 0, 2), (6, 3, 4), (42, 40, 41)];
        assert_eq!(granite_pass(&[6], 0..2, &again, 1), " the New");
        // Right after the settled word's comma, it was said again too ("No, no").
        let after_comma = [(6, 0, 2), (7, 2, 3), (6, 3, 4), (42, 40, 41)];
        assert_eq!(granite_pass(&[6, 7], 0..3, &after_comma, 2), " the New");
    }

    #[test]
    fn a_copy_well_after_the_settled_word_was_said_again() {
        // Granite settled "the" on frames 0 and 1, and a later pass no longer aligns it there.
        // A copy a frame later is the settled word placed later, and one three frames later was
        // said again ("if if").
        let later = [(6, 3, 4), (42, 40, 41)];
        assert_eq!(granite_pass(&[6], 0..2, &later, 0), " New");
        let again = [(6, 5, 6), (42, 40, 41)];
        assert_eq!(granite_pass(&[6], 0..2, &again, 0), " the New");
        // A pass that hears only the end of the settled word has no word to compare.
        assert_eq!(granite_pass(&[6], 0..2, &[(60, 2, 3)], 0), "");
    }

    #[test]
    fn a_word_parakeet_emits_twice_on_the_settled_frames_was_said_twice() {
        // "the" settled on frames 2 to 3, then a window emits "the" on frame 2 and again on 3
        // ("to to").
        let (mut s, tok) = stream_with(&SpecialWords::new(), 50);
        s.recent = vec![settled(&tok, &[6], 2..4)];
        s.resume = Resume::Frame(4);
        s.last_settled = Some(6);
        let config = PassConfig::default();
        let twice = [(6, 0), (6, 2), (6, 3), (42, 40)];
        let out = s.settle(&tok, window(&s, &twice), &config, PassMode::Tentative);
        assert_eq!(out.tentative, " the New");
        // Emitted once, a frame later, it's the settled word heard again.
        let once = [(6, 0), (6, 3), (42, 40)];
        let out = s.settle(&tok, window(&s, &once), &config, PassMode::Tentative);
        assert_eq!(out.tentative, " New");
    }

    #[test]
    fn a_settled_word_the_pass_completes_after_the_boundary_was_heard_in_place() {
        // "kubectl" settled on frames 0 to 3. A window places its last token on frame 3, among
        // the tokens that may be new, and hears the word said again from frame 6.
        let (mut s, tok) = stream_with(&SpecialWords::new(), 50);
        s.recent = vec![settled(&tok, &[1, 2, 3], 0..4)];
        s.resume = Resume::Frame(4);
        s.last_settled = Some(3);
        let heard = [(1, 0), (2, 1), (3, 3), (1, 6), (2, 7), (3, 8), (42, 40)];
        let config = PassConfig::default();
        let out = s.settle(&tok, window(&s, &heard), &config, PassMode::Tentative);
        assert_eq!(out.tentative, " kubectl New");
    }

    #[test]
    fn each_settled_word_stands_for_one_word_of_the_pass() {
        // "New New" settled on frames 2 to 4 and 5 to 6. A window hears the first on frames 3
        // to 5, before the boundary, and the second on frames 6 and 7, so nothing is new but
        // "city".
        let (mut s, tok) = stream_with(&SpecialWords::new(), 50);
        s.recent = vec![settled(&tok, &[42], 2..5), settled(&tok, &[42], 5..7)];
        s.resume = Resume::Frame(7);
        s.last_settled = Some(42);
        let heard = [(42, 3, 6), (42, 6, 8), (44, 40, 41)].map(|(id, frame, end_frame)| Hyp {
            id,
            frame,
            end_frame,
            resume: Resume::Frame(end_frame),
            joined: false,
            heard_end: 0,
            entry: None,
        });
        let heard = after_settled(heard.into_iter(), 7);
        let out = s.settle(&tok, heard, &PassConfig::default(), PassMode::Tentative);
        assert_eq!(out.tentative, " city");
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
    fn a_command_shorter_than_voice_activitys_opening_gets_its_final_pass() {
        let dir = std::env::var_os("GRANITE_SPEECH_DIR").expect("set GRANITE_SPEECH_DIR");
        let model = Arc::new(Model::load(dir.as_ref(), &crate::Options::default()).unwrap());
        // 50 ms of quiet and 200 ms of speech-level noise, whole and in 20 ms pieces, at the
        // models' rate and resampled from 48 kHz.
        for rate in [16_000usize, 48_000] {
            let mut seed = 0x2545_f491u64;
            let audio: Vec<f32> = (0..rate / 4)
                .map(|i| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let level = if i < rate / 20 { 0.0005 } else { 0.2 };
                    ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * level
                })
                .collect();
            for piece in [audio.len(), rate / 50] {
                let config = StreamConfig {
                    input_rate: rate as u32,
                    ..StreamConfig::default()
                };
                let mut s = Streamer::new(model.clone(), config);
                let mut events = Vec::new();
                for chunk in audio.chunks(piece) {
                    events.extend(s.push(chunk));
                }
                events.extend(s.finish().unwrap());
                let started = events
                    .iter()
                    .any(|e| matches!(e, Event::SpeechStart { at, .. } if *at < 0.1));
                let ended = events
                    .iter()
                    .any(|e| matches!(e, Event::UtteranceEnd { .. }));
                assert!(started && ended, "{rate} Hz in {piece}: {events:?}");
            }
        }
    }

    #[test]
    #[ignore = "needs a model, named by GRANITE_SPEECH_DIR"]
    fn a_large_push_keeps_the_audio_before_its_onset() {
        let dir = std::env::var_os("GRANITE_SPEECH_DIR").expect("set GRANITE_SPEECH_DIR");
        let model = Arc::new(Model::load(dir.as_ref(), &crate::Options::default()).unwrap());
        // 1 s of quiet, 2.5 s of speech-level noise and 0.5 s of quiet in one push, more than
        // the 2 s a stream keeps before its newest sample.
        let mut seed = 0x9e37_79b9u64;
        let audio: Vec<f32> = (0..4 * SAMPLE_RATE)
            .map(|i| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let speech = (SAMPLE_RATE..7 * SAMPLE_RATE / 2).contains(&i);
                let level = if speech { 0.2 } else { 0.0005 };
                ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * level
            })
            .collect();
        let mut s = Streamer::new(model, StreamConfig::default());
        let events = s.push(&audio);
        assert!(
            matches!(events[..], [Event::SpeechStart { at, .. }] if (at - 1.0).abs() < 0.02),
            "{events:?}"
        );
        // The utterance starts 300 ms before its onset, with every sample from there.
        let a = s.active.as_ref().expect("an utterance is active");
        let from = 7 * SAMPLE_RATE / 10;
        assert_eq!(a.start, from as u64);
        assert!(a.stream.audio() == &audio[from..]);
    }

    #[test]
    #[ignore = "needs a model, named by GRANITE_SPEECH_DIR"]
    fn an_utterance_starts_where_the_last_one_ends_after_a_short_pause() {
        let dir = std::env::var_os("GRANITE_SPEECH_DIR").expect("set GRANITE_SPEECH_DIR");
        let model = Arc::new(Model::load(dir.as_ref(), &crate::Options::default()).unwrap());
        // Speech-level noise from 0.5 s to 2 s and again after a pause, with quiet around it.
        for (pause, contiguous) in [(1.5, true), (3.0, false)] {
            let second = (2.0 + pause) * SAMPLE_RATE as f64;
            let mut seed = 0x9e37_79b9u64;
            let audio: Vec<f32> = (0..(second as usize + 2 * SAMPLE_RATE))
                .map(|i| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let speech = (SAMPLE_RATE / 2..2 * SAMPLE_RATE).contains(&i)
                        || (second as usize..second as usize + SAMPLE_RATE).contains(&i);
                    let level = if speech { 0.2 } else { 0.0005 };
                    ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * level
                })
                .collect();
            let mut s = Streamer::new(model.clone(), StreamConfig::default());
            for chunk in audio.chunks(SAMPLE_RATE / 50) {
                s.push(chunk);
            }
            let first = s.closing.front().expect("the first utterance ended");
            let next = s.active.as_ref().expect("the second utterance is active");
            let end = first.start + first.stream.audio().len() as u64;
            if contiguous {
                // No audio is in both utterances, and none in neither.
                assert_eq!(next.start, end, "a pause of {pause} s");
            } else {
                // After a pause longer than the stream keeps, it starts 300 ms before its onset.
                assert!(next.start > end, "a pause of {pause} s");
                assert!((next.start as f64 - (second - 0.3 * SAMPLE_RATE as f64)).abs() < 1600.0);
            }
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
