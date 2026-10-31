//! Beam search over a token-and-duration transducer's outputs that favours special words. It
//! reads the networks only through [`TdtNetwork`], so its rules are tested with scripted
//! scores.
//!
//! A search state is the text with everything that decides its future: the frame the next joint
//! step reads, the tokens emitted on that frame and the state of the special-word matcher.
//! States that are equal merge, keeping the better score, and the best `beam` survive each step.
//! The score of a hypothesis that can still emit holds the provisional bonus of a special word
//! under way. A hypothesis that reaches the end of the audio takes its final score at once.

use super::{Compiled, MatchState, Matcher};
use crate::parakeet::Token;

/// The networks a TDT beam search reads.
pub(crate) trait TdtNetwork {
    /// Floats in a hypothesis's decoder state.
    fn state_width(&self) -> usize;
    /// Floats in a hypothesis's prediction output.
    fn pred_width(&self) -> usize;
    /// The joint's outputs at encoder frame `frame` (the vocabulary's logits, then the
    /// durations') for each prediction output in `preds`, `[rows, pred_width]`.
    fn joint(&mut self, frame: usize, preds: &[f32], out: &mut [f32]);
    /// Runs the prediction network on `tokens`, updating each row of `states` and writing each
    /// row of `preds`.
    fn predict(&mut self, tokens: &[usize], states: &mut [f32], preds: &mut [f32]);
}

/// How a transducer's outputs are laid out.
pub(crate) struct Shape<'a> {
    /// Tokens, the blank included.
    pub vocab: usize,
    pub blank: usize,
    pub durations: &'a [usize],
    /// The most tokens emitted on one frame before decoding moves on.
    pub max_symbols: usize,
}

/// Where a search starts: the frame, the tokens already emitted on it, and the decoder's state
/// and prediction output.
pub(crate) struct Start {
    pub frame: usize,
    pub on_frame: usize,
    pub state: Vec<f32>,
    pub pred: Vec<f32>,
}

/// What identifies a search state. Equal keys with equal token ids have the same future.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Key {
    /// A hash of the token ids.
    hash: u64,
    len: usize,
    frame: usize,
    on_frame: usize,
    matcher: MatchState,
}

struct Hyp {
    tokens: Vec<Token>,
    key: Key,
    /// The path's log probability plus its bonus, provisional until the hypothesis reaches the
    /// end.
    score: f32,
    state: Vec<f32>,
    pred: Vec<f32>,
}

#[derive(Clone, Copy)]
enum Step {
    /// Nothing, as it waits at a later frame or has reached the end.
    Keep,
    Blank,
    /// Emits a token with a duration.
    Emit(u32, u32),
}

struct Candidate {
    parent: usize,
    step: Step,
    key: Key,
    score: f32,
}

fn log_sum_exp(v: &[f32]) -> f32 {
    let max = v.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
    max + v.iter().map(|&x| (x - max).exp()).sum::<f32>().ln()
}

/// Keeps the `n` best (score, token) pairs in `top`, best first, earlier ones first among equals.
fn push_top(top: &mut Vec<(f32, usize)>, n: usize, score: f32, token: usize) {
    if top.len() == n && score <= top[n - 1].0 {
        return;
    }
    let at = top.partition_point(|&(s, _)| s >= score);
    top.insert(at, (score, token));
    top.truncate(n);
}

/// The ids of a candidate's text.
fn ids<'a>(hyps: &'a [Hyp], c: &'a Candidate) -> impl Iterator<Item = u32> + 'a {
    let emitted = match c.step {
        Step::Emit(id, _) => Some(id),
        _ => None,
    };
    hyps[c.parent].tokens.iter().map(|t| t.id).chain(emitted)
}

/// A candidate that lands on `key` with `score`. One that reaches the end of the audio can't
/// emit again, so it takes its final score there, and all that reach the end with the same text
/// are the same state.
fn land(
    m: &Matcher,
    frames: usize,
    parent: usize,
    step: Step,
    mut key: Key,
    mut score: f32,
) -> Candidate {
    if key.frame >= frames {
        score += m.finish(key.matcher);
        key.matcher = MatchState::START;
        key.frame = frames;
        key.on_frame = 0;
    }
    Candidate {
        parent,
        step,
        key,
        score,
    }
}

/// Decodes the encoder frames `start.frame..frames` with a beam search of `words.boost.beam`
/// hypotheses that favours `words`. Each hypothesis expands to the blank and to the `beam`
/// tokens whose best landing scores highest, each with every duration. A landing's score holds
/// the logit, the change in bonus, the duration and, for a landing at the end of the audio, the
/// bonus of an unfinished word given back.
/// Returns the tokens of the hypothesis with the best final score per token, as NVIDIA's NeMo
/// chooses, since a path that skips speech with long blanks can score as well as the right text.
/// Token frames count from the same origin as `start.frame`.
pub(crate) fn beam_search(
    net: &mut impl TdtNetwork,
    shape: &Shape,
    frames: usize,
    start: Start,
    words: &Compiled,
) -> Vec<Token> {
    if start.frame >= frames || words.is_empty() {
        return Vec::new();
    }
    let m = &words.matcher;
    let beam = words.boost.beam;
    let (vocab, outputs) = (shape.vocab, shape.vocab + shape.durations.len());
    let pred_width = net.pred_width();
    debug_assert_eq!(start.state.len(), net.state_width());
    let mut hyps = vec![Hyp {
        tokens: Vec::new(),
        key: Key {
            hash: 0,
            len: 0,
            frame: start.frame,
            on_frame: start.on_frame,
            matcher: MatchState::START,
        },
        score: 0.0,
        state: start.state,
        pred: start.pred,
    }];
    let mut preds = Vec::new();
    let mut logits = Vec::new();
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut special = vec![false; vocab];
    let mut marked: Vec<usize> = Vec::new();
    let mut top: Vec<(f32, usize)> = Vec::new();
    while let Some(t) = hyps
        .iter()
        .map(|h| h.key.frame)
        .filter(|&f| f < frames)
        .min()
    {
        let group: Vec<usize> = (0..hyps.len())
            .filter(|&i| hyps[i].key.frame == t)
            .collect();
        preds.clear();
        for &i in &group {
            preds.extend_from_slice(&hyps[i].pred);
        }
        logits.resize(group.len() * outputs, 0.0);
        net.joint(t, &preds, &mut logits);
        debug_assert_eq!(preds.len(), group.len() * pred_width);

        candidates.clear();
        for (i, hyp) in hyps.iter().enumerate().filter(|(_, h)| h.key.frame != t) {
            candidates.push(Candidate {
                parent: i,
                step: Step::Keep,
                key: hyp.key,
                score: hyp.score,
            });
        }
        for (g, &i) in group.iter().enumerate() {
            let hyp = &hyps[i];
            let row = &logits[g * outputs..(g + 1) * outputs];
            let token_lse = log_sum_exp(&row[..vocab]);
            let duration_lse = log_sum_exp(&row[vocab..]);
            let durations = shape.durations.iter().zip(&row[vocab..]);
            for (&d, &dl) in durations.clone() {
                let mut key = hyp.key;
                key.frame = t + d.max(1);
                key.on_frame = 0;
                let score = hyp.score + row[shape.blank] - token_lse + dl - duration_lse;
                candidates.push(land(m, frames, i, Step::Blank, key, score));
            }
            // Where a token lands with each duration: the frame, the tokens on it, and whether
            // that's the end of the audio.
            let after: Vec<(usize, usize)> = shape
                .durations
                .iter()
                .map(|&d| {
                    if d > 0 {
                        (t + d, 0)
                    } else if hyp.key.on_frame + 1 >= shape.max_symbols {
                        (t + 1, 0)
                    } else {
                        (t, hyp.key.on_frame + 1)
                    }
                })
                .collect();
            let (mut live_dl, mut end_dl) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
            for (&(frame, _), &dl) in after.iter().zip(&row[vocab..]) {
                let best = if frame >= frames {
                    &mut end_dl
                } else {
                    &mut live_dl
                };
                *best = best.max(dl);
            }
            // The best tokens by the score of their best landing: logit, change in bonus and
            // duration, and the final take-back for a landing at the end. A token that neither
            // continues nor starts a special word changes the bonus by one of two amounts and
            // has nothing to take back, and the matcher lists the rest.
            let state = hyp.key.matcher;
            for tok in m.special_tokens(state) {
                let tok = tok as usize;
                if tok < vocab && !special[tok] {
                    special[tok] = true;
                    marked.push(tok);
                }
            }
            let (boundary_step, other_step) = m.default_steps(state);
            let any_dl = live_dl.max(end_dl);
            top.clear();
            for (tok, &logit) in row[..vocab].iter().enumerate() {
                if tok == shape.blank || special[tok] {
                    continue;
                }
                let delta = if m.is_boundary(tok as u32) {
                    boundary_step
                } else {
                    other_step
                };
                push_top(&mut top, beam, logit + delta + any_dl, tok);
            }
            for &tok in &marked {
                if tok != shape.blank {
                    let (next, delta) = m.step(state, tok as u32);
                    let landing = live_dl.max(end_dl + m.finish(next));
                    push_top(&mut top, beam, row[tok] + delta + landing, tok);
                }
                special[tok] = false;
            }
            marked.clear();
            for &(_, tok) in &top {
                let (matcher, delta) = m.step(state, tok as u32);
                let base = hyp.score + row[tok] - token_lse + delta;
                let hash = (hyp.key.hash ^ (tok as u64 + 1)).wrapping_mul(0x0100_0000_01b3);
                for ((&d, &dl), &(frame, on_frame)) in durations.clone().zip(&after) {
                    let key = Key {
                        hash,
                        len: hyp.key.len + 1,
                        frame,
                        on_frame,
                        matcher,
                    };
                    let step = Step::Emit(tok as u32, d as u32);
                    candidates.push(land(m, frames, i, step, key, base + dl - duration_lse));
                }
            }
        }
        // The best distinct states survive.
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
        let mut kept: Vec<&Candidate> = Vec::with_capacity(beam);
        for c in &candidates {
            if kept.len() == beam {
                break;
            }
            let duplicate = kept
                .iter()
                .any(|k| k.key == c.key && ids(&hyps, k).eq(ids(&hyps, c)));
            if !duplicate {
                kept.push(c);
            }
        }
        let mut next: Vec<Hyp> = Vec::with_capacity(kept.len());
        let mut emitted: Vec<usize> = Vec::new();
        for c in kept {
            let parent = &hyps[c.parent];
            let mut hyp = Hyp {
                tokens: parent.tokens.clone(),
                key: c.key,
                score: c.score,
                state: parent.state.clone(),
                pred: parent.pred.clone(),
            };
            if let Step::Emit(id, duration) = c.step {
                hyp.tokens.push(Token {
                    id,
                    frame: t as u32,
                    duration,
                });
                if hyp.key.frame < frames {
                    emitted.push(next.len());
                }
            }
            next.push(hyp);
        }
        if !emitted.is_empty() {
            let tokens: Vec<usize> = emitted
                .iter()
                .map(|&i| next[i].tokens.last().expect("a token was emitted").id as usize)
                .collect();
            let mut states: Vec<f32> = emitted
                .iter()
                .flat_map(|&i| next[i].state.iter().copied())
                .collect();
            let mut new_preds = vec![0f32; emitted.len() * pred_width];
            net.predict(&tokens, &mut states, &mut new_preds);
            let width = states.len() / emitted.len();
            for (r, &i) in emitted.iter().enumerate() {
                next[i]
                    .state
                    .copy_from_slice(&states[r * width..(r + 1) * width]);
                next[i]
                    .pred
                    .copy_from_slice(&new_preds[r * pred_width..(r + 1) * pred_width]);
            }
        }
        hyps = next;
    }
    let per_token = |h: &Hyp| h.score / (h.tokens.len() + 1) as f32;
    hyps.into_iter()
        .max_by(|a, b| per_token(a).total_cmp(&per_token(b)))
        .map_or_else(Vec::new, |h| h.tokens)
}

#[cfg(test)]
mod tests {
    use super::super::Boost;
    use super::super::tests::{special, tokenizer};
    use super::*;

    const DURATIONS: [usize; 3] = [0, 1, 2];
    /// The token before the first, in the scripted networks.
    const NONE: u32 = u32::MAX;

    /// A network whose joint outputs depend on the frame and the last token, as `script` gives
    /// them. The state and the prediction output are the last token.
    struct Scripted<F: Fn(usize, u32, &mut [f32])> {
        script: F,
        outputs: usize,
    }

    impl<F: Fn(usize, u32, &mut [f32])> TdtNetwork for Scripted<F> {
        fn state_width(&self) -> usize {
            1
        }
        fn pred_width(&self) -> usize {
            1
        }
        fn joint(&mut self, frame: usize, preds: &[f32], out: &mut [f32]) {
            for (row, &last) in out.chunks_exact_mut(self.outputs).zip(preds) {
                row.fill(-20.0);
                (self.script)(frame, last as u32, row);
            }
        }
        fn predict(&mut self, tokens: &[usize], states: &mut [f32], preds: &mut [f32]) {
            for ((s, p), &t) in states.iter_mut().zip(preds.iter_mut()).zip(tokens) {
                *s = t as f32;
                *p = t as f32;
            }
        }
    }

    const VOCAB: usize = 57;

    fn shape() -> Shape<'static> {
        Shape {
            vocab: VOCAB,
            blank: 0,
            durations: &DURATIONS,
            max_symbols: 3,
        }
    }

    fn search(
        script: impl Fn(usize, u32, &mut [f32]),
        frames: usize,
        words: &[&str],
        bonus: f32,
        beam: usize,
    ) -> Vec<u32> {
        let mut list = special(words);
        list.set_boost(Boost::new(bonus, beam, 4.0, 2.0).unwrap());
        let (c, _) = Compiled::new(&list, &tokenizer());
        let mut net = Scripted {
            script,
            outputs: VOCAB + DURATIONS.len(),
        };
        let start = Start {
            frame: 0,
            on_frame: 0,
            state: vec![NONE as f32],
            pred: vec![NONE as f32],
        };
        beam_search(&mut net, &shape(), frames, start, &c)
            .iter()
            .map(|t| t.id)
            .collect()
    }

    /// Sets a token's logit and the durations' logits (for 0, 1 and 2 frames).
    fn set(row: &mut [f32], token: u32, logit: f32, durations: [f32; 3]) {
        row[token as usize] = logit;
        row[VOCAB..].copy_from_slice(&durations);
    }

    fn index(v: &[f32]) -> usize {
        let mut best = 0;
        for (i, &x) in v.iter().enumerate() {
            if x > v[best] {
                best = i;
            }
        }
        best
    }

    #[test]
    fn a_beam_of_one_without_a_bonus_decodes_greedily() {
        // Scores that look random but depend only on the frame and the last token.
        let script = |frame: usize, last: u32, row: &mut [f32]| {
            for (i, x) in row.iter_mut().enumerate() {
                let mut h = (frame as u64 * 1_000_003 + last as u64 * 7919 + i as u64 + 1)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15);
                h ^= h >> 29;
                *x = (h % 10_000) as f32 / 1000.0;
            }
        };
        let frames = 40;
        // The greedy decoding Parakeet does, over the same scores.
        let mut row = vec![0f32; VOCAB + DURATIONS.len()];
        let (mut t, mut on_frame, mut last, mut greedy) = (0, 0, NONE, Vec::new());
        while t < frames {
            row.fill(-20.0);
            script(t, last, &mut row);
            let token = index(&row[..VOCAB]);
            let d = DURATIONS[index(&row[VOCAB..])];
            if token == 0 {
                t += d.max(1);
                on_frame = 0;
                continue;
            }
            greedy.push(token as u32);
            last = token as u32;
            if d > 0 {
                t += d;
                on_frame = 0;
            } else {
                on_frame += 1;
                if on_frame >= 3 {
                    t += 1;
                    on_frame = 0;
                }
            }
        }
        assert!(greedy.len() > 5);
        assert_eq!(search(script, frames, &["kubectl"], 0.0, 1), greedy);
    }

    #[test]
    fn the_same_text_at_different_frames_stays_two_states() {
        // "▁the" one frame long leads to a frame where everything is unlikely, "▁the" two frames
        // long to a confident ",". Merging both as one text keeps only the first.
        let script = |frame: usize, last: u32, row: &mut [f32]| match (frame, last) {
            (0, NONE) => set(row, 6, 0.0, [-20.0, 0.0, -0.5]),
            (1, 6) => row.fill(-3.0),
            (2, 6) => set(row, 7, 0.0, [-20.0, 0.0, -20.0]),
            _ => set(row, 0, 0.0, [-20.0, 0.0, -20.0]),
        };
        assert_eq!(search(script, 3, &["kubectl"], 0.0, 2), vec![6, 7]);
    }

    #[test]
    fn a_word_boundary_that_keeps_a_bonus_is_expanded() {
        // After "kubectl", a letter piece is likelier than a new word but would end the special
        // word and take its bonus back.
        let script = |frame: usize, last: u32, row: &mut [f32]| match (frame, last) {
            (0, NONE) => set(row, 1, 0.0, [-20.0, 0.0, -20.0]),
            (1, 1) => set(row, 2, 0.0, [-20.0, 0.0, -20.0]),
            (2, 2) => set(row, 3, 0.0, [-20.0, 0.0, -20.0]),
            (3, 3) => {
                set(row, 47, 0.0, [-20.0, 0.0, -20.0]);
                row[6] = -0.5;
                row[0] = -5.0;
            }
            _ => set(row, 0, 0.0, [-20.0, 0.0, -20.0]),
        };
        assert_eq!(search(script, 5, &["kubectl"], 1.5, 1), vec![1, 2, 3, 6]);
    }

    #[test]
    fn a_hypothesis_at_the_end_gives_back_an_unfinished_word() {
        // The only token is the start of a special word, which the audio ends before. Its
        // provisional bonus mustn't let it beat the blank.
        let script = |frame: usize, last: u32, row: &mut [f32]| match (frame, last) {
            (0, NONE) => {
                set(row, 1, -1.0, [-20.0, -20.0, 0.0]);
                row[0] = 0.0;
            }
            _ => set(row, 0, 0.0, [-20.0, 0.0, -20.0]),
        };
        assert_eq!(search(script, 2, &["kubectl"], 1.5, 1), Vec::<u32>::new());
    }

    #[test]
    fn an_unfinished_word_at_the_end_doesnt_crowd_out_an_ordinary_token() {
        // On the last frame "▁ku" with its bonus looks better than "▁the", but the audio ends
        // there, so it would give the bonus back, and "▁the" is likelier.
        let script = |frame: usize, last: u32, row: &mut [f32]| match (frame, last) {
            (0, NONE) => {
                set(row, 6, 0.0, [-20.0, 0.0, -20.0]);
                row[1] = -0.5;
            }
            _ => set(row, 0, 0.0, [-20.0, 0.0, -20.0]),
        };
        assert_eq!(search(script, 1, &["kubectl"], 1.5, 1), vec![6]);
    }

    #[test]
    fn a_token_that_may_land_live_or_at_the_end_keeps_its_live_landing() {
        // "▁ku" may land on the next frame, where "kube" can complete, or at the end, where it
        // can't. Its live landing keeps the bonus and beats "▁the".
        let script = |frame: usize, last: u32, row: &mut [f32]| match (frame, last) {
            (0, NONE) => {
                set(row, 1, 0.0, [-20.0, 0.0, 0.0]);
                row[6] = 0.1;
            }
            (1, 1) => set(row, 2, 0.0, [-20.0, 0.0, -20.0]),
            _ => set(row, 0, 0.0, [-20.0, 0.0, -20.0]),
        };
        assert_eq!(search(script, 2, &["kube"], 1.5, 1), vec![1, 2]);
    }
}
