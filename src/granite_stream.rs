//! Streaming transcription with Granite Speech. Audio arrives in pieces, and each update encodes
//! what is new without running the encoder over the whole utterance again. Like all streaming
//! in this crate, it is experimental.
//!
//! Granite's attention stays inside blocks of 128 frames, 2.56 s in the first block of layers,
//! 5.12 s in the second and 10.24 s in the rest, so once an attention block is complete, new
//! audio reaches it only through the convolutions at its edge. An update therefore runs the two
//! strided layers over their attention blocks that hold new or changed frames, with the edge
//! frames before them recomputed from cached states, and the other layers over their current
//! 10.24 s block. Blocks before that are frozen as they were when the next one began. Every
//! layer keeps its post-attention states, which the next update's convolutions read at the left
//! edge, and its outputs, which feed the next layer.
//!
//! The front end computes each 10 ms frame once its window is in the audio and floors the
//! features at the loudest frame so far rather than the utterance's. The last update pads the
//! end of the audio as [`Granite::features`] does, so a single update over a whole utterance
//! gives exactly what [`Granite::encode`] and [`Granite::frame_ids`] give.

use crate::granite::{Granite, Scratch, ctc_tokens, timed};
use crate::granite_features::{FrameBuffers, GraniteFrontEnd};
use crate::parakeet::Timings;

/// Encoder frames the next block must have before the one before it freezes, unless set.
const DEFAULT_FREEZE_LAG: usize = 16;

/// One layer's cached activations.
#[derive(Default)]
struct LayerCache {
    /// Post-attention states at the layer's input rate, `[frames, d_model]`.
    states: Vec<f32>,
    /// Outputs at the layer's output rate, with the self-conditioning injection added after the
    /// layer it follows.
    out: Vec<f32>,
}

/// The state of one utterance being transcribed while it is spoken. It holds no reference to
/// the model, which each update takes, so a driver can own both.
pub struct GraniteStream {
    front_end: GraniteFrontEnd,
    buffers: FrameBuffers,
    samples: Vec<f32>,
    /// log10 mel energies of every frame computed, before the floor, `[frames, n_mels]`.
    logs: Vec<f32>,
    peak: f32,
    /// The peak the cached feature rows were floored with.
    floor_peak: f32,
    finished: bool,
    /// Feature rows encoded so far, the first layer's input frames.
    rows: usize,
    /// The feature rows, `[rows, width]`.
    features: Vec<f32>,
    /// The input projection of every row, `[rows, d_model]`.
    input: Vec<f32>,
    layers: Vec<LayerCache>,
    /// The first encoder frame the layers after the strided ones recompute. Earlier frames are
    /// frozen.
    frozen: usize,
    /// Encoder frames the next block must have before the one before it freezes.
    lag: usize,
    ids: Vec<u32>,
    /// The tokens of special words whose logits word spotting reads.
    spotting: Option<Vec<u32>>,
    /// Each frame's row for word spotting (see [`crate::biasing::score_row`]).
    scores: Vec<f32>,
    first_changed: usize,
    scratch: Option<(usize, Scratch)>,
}

impl GraniteStream {
    /// A stream for one utterance. The model's strided blocks must come first, as they do in
    /// Granite Speech 5.0.
    pub fn new(model: &Granite) -> GraniteStream {
        let c = model.config();
        assert!(
            c.subsample_layers
                .iter()
                .enumerate()
                .all(|(i, &layer)| i == layer),
            "streaming needs the strided blocks first"
        );
        GraniteStream {
            front_end: GraniteFrontEnd::new(c.n_mels),
            buffers: FrameBuffers::new(),
            samples: Vec::new(),
            logs: Vec::new(),
            peak: f32::NEG_INFINITY,
            floor_peak: f32::NEG_INFINITY,
            finished: false,
            rows: 0,
            features: Vec::new(),
            input: Vec::new(),
            layers: (0..c.layers).map(|_| LayerCache::default()).collect(),
            frozen: 0,
            lag: DEFAULT_FREEZE_LAG,
            ids: Vec::new(),
            spotting: None,
            scores: Vec::new(),
            first_changed: 0,
            scratch: None,
        }
    }

    /// Keeps, for every frame, the scores word spotting reads for these tokens of special words.
    pub(crate) fn keep_spotting_scores(&mut self, tokens: &[u32]) {
        self.spotting = Some(tokens.to_vec());
    }

    /// The rows word spotting reads, one for each frame so far.
    pub(crate) fn spotting_scores(&self) -> &[f32] {
        &self.scores
    }

    /// Sets how many encoder frames (80 ms each) the next 10.24 s block must have before the one
    /// before it freezes, 16 (1.28 s) by default. With 0, a block freezes at the first update
    /// after the next one starts. Each frame of lag costs an update about one frame of encoding
    /// on average, and buys the frozen block's edge more of the next block as context.
    pub fn set_freeze_lag(&mut self, frames: usize) {
        self.lag = frames;
    }

    /// Appends 16 kHz mono samples of the utterance. No audio may follow a finished update.
    pub fn push(&mut self, samples: &[f32]) {
        assert!(!self.finished, "audio pushed after the stream finished");
        self.samples.extend_from_slice(samples);
    }

    /// Samples pushed so far.
    pub fn samples(&self) -> usize {
        self.samples.len()
    }

    /// Whether the last update was the finished one.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Encoder frames (80 ms each) so far.
    pub fn frames(&self) -> usize {
        self.ids.len()
    }

    /// The most likely token of every frame so far, the CTC blank included.
    pub fn frame_ids(&self) -> &[u32] {
        &self.ids
    }

    /// The first frame the last update may have changed. Earlier frames keep their ids.
    pub fn first_changed(&self) -> usize {
        self.first_changed
    }

    /// The text of the frames so far, with repeats collapsed and blanks dropped.
    pub fn text(&self, model: &Granite) -> String {
        let tokens = ctc_tokens(&self.ids, model.config().blank);
        model
            .tokenizer()
            .decode(&tokens.iter().map(|t| t.id).collect::<Vec<_>>())
    }

    /// The encoder output of frames `first..first + rows`, `[rows, d_model]`.
    pub fn encoder_output(&self, model: &Granite, first: usize, rows: usize) -> &[f32] {
        let d = model.config().d_model;
        let out = &self.layers.last().expect("the model has layers").out;
        &out[first * d..(first + rows) * d]
    }

    /// The feature rows so far, `[rows, 320]`, as the first block read them.
    pub fn features(&self) -> &[f32] {
        &self.features
    }

    /// The outputs of block `layer` so far, at its output rate, `[frames, d_model]`, for
    /// comparing with an offline pass.
    pub fn layer_output(&self, layer: usize) -> &[f32] {
        &self.layers[layer].out
    }

    /// The first encoder frame of the 10.24 s block the next update recomputes. Frames before it
    /// are frozen.
    pub fn frozen(&self) -> usize {
        self.frozen
    }

    /// The CTC logits of frames `first..first + rows`, `[rows, vocab]`, computed from the cached
    /// encoder output.
    pub fn logits(&self, model: &Granite, first: usize, rows: usize) -> Vec<f32> {
        model.logits_of(self.encoder_output(model, first, rows), rows)
    }

    /// Encodes the audio pushed since the last update. With `finished`, the audio is complete:
    /// its end is padded as the offline front end pads it, and no audio may follow.
    pub fn update(&mut self, model: &Granite, finished: bool, timings: &mut Timings) {
        assert!(!self.finished, "the stream already finished");
        let c = model.config();
        let (d, ctx, n_mels, half) = (c.d_model, c.context, c.n_mels, c.conv_kernel / 2);
        let n = self.samples.len();
        self.first_changed = self.ids.len();
        let fe = &self.front_end;
        // The offline front end refuses less than two hops of audio, and then there's nothing.
        let (frames, len) = if !finished {
            (fe.complete_frames(n), n)
        } else if n >= 2 * 160 {
            (fe.frames_for(n), fe.padded_len(n))
        } else {
            (self.logs.len() / n_mels, n)
        };
        self.finished = finished;

        timed(timings, "features", || {
            let done = self.logs.len() / n_mels;
            self.logs.resize(frames * n_mels, 0.0);
            for f in done..frames {
                let out = &mut self.logs[f * n_mels..(f + 1) * n_mels];
                self.front_end
                    .log_frame(&self.samples, len, f, &mut self.buffers, out);
                self.peak = out.iter().fold(self.peak, |m, &v| m.max(v));
            }
        });
        // A row needs the frame after it for its last delta, until the end repeats the last frame.
        let rows = if finished {
            frames / 2
        } else {
            frames.saturating_sub(1) / 2
        };
        if rows <= self.rows {
            return;
        }

        // Rows of the first layer's current attention block are stacked again, so they all take
        // the floor of the loudest frame so far. When that frame got louder, so are the rows that
        // the unfrozen frames of the later blocks read, back through the strided blocks'
        // convolutions and attention blocks.
        let mut first_row = self.rows / ctx * ctx;
        if self.peak > self.floor_peak {
            let mut row = self.frozen;
            for _ in 0..c.subsample_layers.len() {
                row = (2 * row).saturating_sub(half) / ctx * ctx;
            }
            first_row = first_row.min(row);
            self.floor_peak = self.peak;
        }
        let width = self.front_end.width();
        self.features.resize(rows * width, 0.0);
        timed(timings, "features", || {
            self.front_end.stack_rows(
                &self.logs,
                self.peak,
                first_row,
                rows,
                &mut self.features[first_row * width..],
            )
        });
        self.input.resize(rows * d, 0.0);
        timed(timings, "input_linear", || {
            model.input_projection(
                &self.features[first_row * width..],
                rows - first_row,
                &mut self.input[first_row * d..],
            )
        });
        self.rows = rows;

        // Each layer starts at the attention block holding its first changed input frame (the
        // strided layers) or at the first unfrozen frame (the others).
        let mut changed = first_row;
        let mut frames_in = rows;
        for (i, block) in model.blocks().iter().enumerate() {
            let (start, out_start, frames_out) = if block.strided {
                let start = changed / ctx * ctx;
                // Outputs before start / 2 whose convolution reaches the changed block.
                let out_start = start.saturating_sub(half).div_ceil(2);
                (start, out_start, frames_in / 2)
            } else {
                (self.frozen, self.frozen, frames_in)
            };
            if start >= frames_in || out_start >= frames_out {
                // Too little audio yet for this layer to have an output frame.
                return;
            }
            let lo = model.conv_context(block, out_start);
            let need = frames_in - lo;
            if self.scratch.as_ref().is_none_or(|(rows, _)| *rows < need) {
                self.scratch = Some((need, Scratch::new(need, c)));
            }
            let s = &mut self.scratch.as_mut().expect("just made").1;

            let (before, rest) = self.layers.split_at_mut(i);
            let layer = &mut rest[0];
            let input = if i == 0 {
                &self.input
            } else {
                &before[i - 1].out
            };
            let mut x = input[start * d..frames_in * d].to_vec();
            let left = &layer.states[lo * d..start * d];
            let mut out = vec![0f32; (frames_out - out_start) * d];
            model.block_span(
                block, &mut x, start, frames_in, left, out_start, &mut out, s, timings,
            );
            if i == c.self_conditioning_layer() {
                let rows_out = frames_out - out_start;
                let mut injection = vec![0f32; rows_out * d];
                timed(timings, "self_conditioning", || {
                    model.self_conditioning(&out, rows_out, &mut injection)
                });
                for (v, &a) in out.iter_mut().zip(&injection) {
                    *v += a;
                }
            }
            layer.states.resize(frames_in * d, 0.0);
            layer.states[start * d..].copy_from_slice(&x);
            layer.out.resize(frames_out * d, 0.0);
            layer.out[out_start * d..].copy_from_slice(&out);
            changed = out_start;
            frames_in = frames_out;
        }

        let first = changed;
        let (ids, scores) = timed(timings, "ctc_head", || {
            model.frame_scores(
                &self.layers.last().expect("the model has layers").out[first * d..],
                frames_in - first,
                self.spotting.as_deref(),
            )
        });
        self.ids.truncate(first);
        self.ids.extend(ids);
        if let Some(tokens) = &self.spotting {
            self.scores.truncate(first * (2 + tokens.len()));
            self.scores.extend(scores);
        }
        self.first_changed = first;
        if frames_in > self.lag {
            self.frozen = self.frozen.max((frames_in - 1 - self.lag) / ctx * ctx);
        }
    }
}
