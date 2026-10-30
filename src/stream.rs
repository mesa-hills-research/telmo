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

use std::time::Duration;

use crate::error::Result;
use crate::granite::Granite;
use crate::model::Model;
use crate::parakeet::{DecoderState, Parakeet, Timings};
use crate::tokenizer::Tokenizer;

/// Samples per second the models take.
pub const SAMPLE_RATE: usize = 16000;
/// Samples per encoder frame (80 ms) of both models.
const FRAME: usize = 1280;

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
    /// Where the last word the model heard ends, in seconds from the utterance's start.
    pub last_word_end: Option<f64>,
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
    granite: Option<GraniteBlocks>,
    /// Whether any text settled in this utterance.
    settled_any: bool,
    passes: usize,
    pub timings: Timings,
}

/// Granite's encoder state between passes.
struct GraniteBlocks {
    /// CTC argmax of every encoder frame computed so far.
    ids: Vec<u32>,
}

impl UtteranceStream {
    pub fn new(model: &Model) -> UtteranceStream {
        let mut timings = Timings::default();
        let (resume, granite) = match model {
            Model::Parakeet(m) => (Resume::Tdt(m.initial_decoder_state(&mut timings)), None),
            Model::Granite(_) => (
                Resume::Ctc {
                    frame: 0,
                    prev: u32::MAX,
                },
                Some(GraniteBlocks { ids: Vec::new() }),
            ),
        };
        UtteranceStream {
            audio: Vec::new(),
            resume,
            last_settled: None,
            recent: Vec::new(),
            previous: Vec::new(),
            granite,
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
    pub fn truncate(&mut self, samples: usize) {
        self.audio.truncate(samples);
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
    pub fn pass(&mut self, model: &Model, config: &PassConfig, last: bool) -> Result<PassText> {
        self.passes += 1;
        let hyp = match model {
            Model::Parakeet(m) => self.parakeet_hypothesis(m, config)?,
            Model::Granite(m) => self.granite_hypothesis(m, last)?,
        };
        let tokenizer = match model {
            Model::Parakeet(m) => m.tokenizer(),
            Model::Granite(m) => m.tokenizer(),
        };
        let hyp = self.drop_repeats(tokenizer, config, hyp);
        let words = split_words(tokenizer, self.last_settled, &hyp);
        let last_word_end = hyp
            .last()
            .map(|h| (h.end_frame * FRAME) as f64 / SAMPLE_RATE as f64);

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
    /// and decodes from the state saved after the last settled token.
    fn parakeet_hypothesis(&mut self, model: &Parakeet, config: &PassConfig) -> Result<Vec<Hyp>> {
        let Resume::Tdt(state) = &self.resume else {
            unreachable!("Parakeet resumes from a decoder state")
        };
        let context = (config.context.as_secs_f64() * SAMPLE_RATE as f64) as usize;
        let start = (state.frame * FRAME).saturating_sub(context) / FRAME * FRAME;
        let window = &self.audio[start.min(self.audio.len())..];
        // The front end needs two frames of 10 ms, and the encoder at least one frame.
        if window.len() < FRAME {
            return Ok(Vec::new());
        }
        let timings = &mut self.timings;
        let features = timings.time("features", || model.features(window))?;
        let encoded = model.encode(&features.data, features.frames, timings, &mut |_, _, _| {});
        let tokens = model.decode_from(&encoded, start / FRAME, state, timings);
        Ok(tokens
            .into_iter()
            .map(|(t, after)| Hyp {
                id: t.id,
                end_frame: t.frame as usize + (t.duration as usize).max(1),
                resume: Resume::Tdt(after),
            })
            .collect())
    }

    /// Encodes the audio so far and decodes CTC frames from the last settled one.
    fn granite_hypothesis(&mut self, model: &Granite, _last: bool) -> Result<Vec<Hyp>> {
        let blocks = self.granite.as_mut().expect("Granite keeps its blocks");
        if self.audio.len() < FRAME {
            return Ok(Vec::new());
        }
        let timings = &mut self.timings;
        let features = timings.time("features", || model.features(&self.audio))?;
        let encoded = model.encode(&features.data, features.frames, timings, &mut |_, _, _| {})?;
        blocks.ids = timings.time("ctc_head", || model.frame_ids(&encoded));
        let Resume::Ctc { frame, prev } = self.resume else {
            unreachable!("Granite resumes from a CTC frame")
        };
        Ok(ctc_hypothesis(
            &blocks.ids,
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
