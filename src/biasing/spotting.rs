//! Granite's word spotting: special words aligned to the CTC output, replacing greedy words
//! where they fit the audio well enough.

use std::ops::Range;

use super::{Compiled, Form};
use crate::parakeet::Token;
use crate::tokenizer::Tokenizer;

/// The scores word spotting reads, for each frame: its best logit, the blank's logit and the
/// logits of [`Compiled::tokens`], in that order.
pub(crate) fn score_row(logits: &[f32], blank: usize, tokens: &[u32], out: &mut Vec<f32>) {
    out.push(logits.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x)));
    out.push(logits[blank]);
    out.extend(tokens.iter().map(|&t| logits[t as usize]));
}

/// A stretch of greedy tokens that spotting replaces whole or not at all: a word, or a
/// punctuation token.
#[derive(Debug, Clone, Copy)]
struct Span {
    /// The span's first and last frame.
    first: usize,
    last: usize,
    has_word: bool,
}

/// A replacement word spotting found.
#[derive(Debug, Clone, Copy)]
struct Spot {
    form: usize,
    first: usize,
    last: usize,
    gain: f32,
}

/// Whether two spots share a frame.
fn overlap(a: &Spot, b: &Spot) -> bool {
    a.first <= b.last && b.first <= a.last
}

/// Replaces greedy CTC tokens with special words where an alignment of a word, plus its bonus,
/// beats the greedy path over the same frames by the margin. `greedy` holds the greedy tokens of
/// frames `first..` (each a run of frames), `scores` the rows of [`score_row`] for every frame
/// from 0. A spotted word covers whole greedy words, at least one, and the frames between them.
/// Its last token lasts to the end of those frames. Returns the tokens and, for each spotted
/// word, the range of them it holds, which must stay together.
pub(crate) fn spot(
    words: &Compiled,
    tokenizer: &Tokenizer,
    greedy: &[Token],
    scores: &[f32],
    first: usize,
) -> (Vec<Token>, Vec<Range<usize>>) {
    let width = 2 + words.tokens.len();
    let frames = scores.len() / width;
    if words.is_empty() || first >= frames {
        return (greedy.to_vec(), Vec::new());
    }
    // Words and punctuation of the greedy text, as spans of frames.
    let mut spans: Vec<Span> = Vec::new();
    for t in greedy {
        let last = (t.frame + t.duration.max(1) - 1) as usize;
        match spans.last_mut() {
            Some(s) if !tokenizer.is_word_boundary(t.id) => s.last = last,
            _ => spans.push(Span {
                first: t.frame as usize,
                last,
                has_word: false,
            }),
        }
        let s = spans.last_mut().expect("just pushed");
        s.has_word |= tokenizer.decode(&[t.id]).chars().any(char::is_alphanumeric);
    }
    // An alignment may start where no span is under way, end where none goes on, and must
    // cover a frame of a word.
    let n = frames - first;
    let mut may_start = vec![true; n];
    let mut may_end = vec![true; n];
    let mut in_word = vec![false; n];
    for s in spans.iter().filter(|s| s.first >= first) {
        for f in s.first + 1..=s.last.min(frames - 1) {
            may_start[f - first] = false;
        }
        for f in s.first..s.last.min(frames - 1) {
            may_end[f - first] = false;
        }
        if s.has_word {
            for f in s.first..=s.last.min(frames - 1) {
                in_word[f - first] = true;
            }
        }
    }
    let cost = |f: usize, column: usize| {
        let row = &scores[f * width..(f + 1) * width];
        row[0] - row[column]
    };

    let mut spots = Vec::new();
    for (index, form) in words.forms.iter().enumerate() {
        // The best path to each state, apart for paths that have covered a word's frame and
        // paths that haven't, so a cheaper path over blanks can't hide one that may replace.
        let states = 2 * form.tokens.len() + 1;
        let mut cost_of = vec![[f32::INFINITY; 2]; states];
        let mut start = vec![[0usize; 2]; states];
        let mut next_cost = cost_of.clone();
        let mut next_start = start.clone();
        // Spots of this form that overlap keep only the best, which bounds them by the words.
        let mut last: Option<Spot> = None;
        for f in first..frames {
            let i = f - first;
            for s in 0..states {
                for covered in [false, true] {
                    // Arriving at a word's frame makes a path covered.
                    let mut best = (f32::INFINITY, 0usize);
                    for from in [false, true] {
                        if (from || in_word[i]) != covered {
                            continue;
                        }
                        let c = from as usize;
                        let mut options = [(cost_of[s][c], start[s][c]); 3];
                        if s >= 1 {
                            options[1] = (cost_of[s - 1][c], start[s - 1][c]);
                        }
                        if form.may_skip(s) {
                            options[2] = (cost_of[s - 2][c], start[s - 2][c]);
                        }
                        for o in options {
                            if o.0 < best.0 {
                                best = o;
                            }
                        }
                    }
                    if s <= 1 && may_start[i] && covered == in_word[i] && best.0 > 0.0 {
                        best = (0.0, f);
                    }
                    next_cost[s][covered as usize] = best.0 + cost(f, form.column(s));
                    next_start[s][covered as usize] = best.1;
                }
            }
            std::mem::swap(&mut cost_of, &mut next_cost);
            std::mem::swap(&mut start, &mut next_start);
            if !may_end[i] {
                continue;
            }
            for s in [states - 2, states - 1] {
                let gain = form.bonus - cost_of[s][1] - form.margin;
                if gain > 0.0 {
                    let spot = Spot {
                        form: index,
                        first: start[s][1],
                        last: f,
                        gain,
                    };
                    match last {
                        Some(l) if overlap(&l, &spot) => {
                            if spot.gain > l.gain {
                                last = Some(spot);
                            }
                        }
                        _ => {
                            spots.extend(last);
                            last = Some(spot);
                        }
                    }
                }
            }
        }
        spots.extend(last);
    }
    if spots.is_empty() {
        return (greedy.to_vec(), Vec::new());
    }
    // The best spots that don't overlap win.
    spots.sort_by(|a, b| b.gain.total_cmp(&a.gain));
    let mut chosen: Vec<Spot> = Vec::new();
    for s in spots {
        if chosen.iter().all(|c| !overlap(c, &s)) {
            chosen.push(s);
        }
    }
    chosen.sort_by_key(|s| s.first);
    let mut out = Vec::with_capacity(greedy.len());
    let mut ranges = Vec::with_capacity(chosen.len());
    let mut next = chosen.iter().peekable();
    let mut i = 0;
    while i < greedy.len() {
        let t = greedy[i];
        match next.peek() {
            Some(s) if t.frame as usize >= s.first => {
                let at = out.len();
                out.extend(align(&words.forms[s.form], s.first, s.last, &cost));
                ranges.push(at..out.len());
                while i < greedy.len() && greedy[i].frame as usize <= s.last {
                    i += 1;
                }
                next.next();
            }
            _ => {
                out.push(t);
                i += 1;
            }
        }
    }
    // Every spot covers a greedy word, so none lies after the last greedy token.
    debug_assert!(next.peek().is_none());
    (out, ranges)
}

/// The tokens of the best CTC alignment of `form` over frames `first..=last`, each at the frame
/// its run starts, the last lasting to `last`.
fn align(form: &Form, first: usize, last: usize, cost: &dyn Fn(usize, usize) -> f32) -> Vec<Token> {
    let states = 2 * form.tokens.len() + 1;
    let n = last + 1 - first;
    let mut cost_of = vec![f32::INFINITY; states];
    cost_of[0] = cost(first, form.column(0));
    cost_of[1] = cost(first, form.column(1));
    // back[i][s]: the state at frame first + i - 1 the best path to s at first + i came from.
    let mut back = vec![vec![0u8; states]; n];
    for (i, row) in back.iter_mut().enumerate().skip(1) {
        let mut next = vec![f32::INFINITY; states];
        for s in 0..states {
            let mut best = (cost_of[s], 0u8);
            if s >= 1 && cost_of[s - 1] < best.0 {
                best = (cost_of[s - 1], 1);
            }
            if form.may_skip(s) && cost_of[s - 2] < best.0 {
                best = (cost_of[s - 2], 2);
            }
            next[s] = best.0 + cost(first + i, form.column(s));
            row[s] = best.1;
        }
        cost_of = next;
    }
    let mut s = if cost_of[states - 2] <= cost_of[states - 1] {
        states - 2
    } else {
        states - 1
    };
    let mut path = vec![0usize; n];
    for i in (0..n).rev() {
        path[i] = s;
        s -= back[i][s] as usize;
    }
    let mut tokens: Vec<Token> = Vec::new();
    for (i, &s) in path.iter().enumerate() {
        let new_run = i == 0 || path[i - 1] != s;
        if !s.is_multiple_of(2) && new_run {
            tokens.push(Token {
                id: form.tokens[s / 2],
                frame: (first + i) as u32,
                duration: 1,
            });
        } else if !s.is_multiple_of(2) {
            tokens.last_mut().expect("a run is open").duration += 1;
        }
    }
    if let Some(t) = tokens.last_mut() {
        t.duration = last as u32 + 1 - t.frame;
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::super::Boost;
    use super::super::tests::{special, tokenizer};
    use super::*;

    /// Scores for frames where each frame's best token (or the blank) has logit 0 and every
    /// other token -10, except the logits in `set` (frame, token, logit).
    fn scores(c: &Compiled, best: &[u32], set: &[(usize, u32, f32)]) -> Vec<f32> {
        let mut out = Vec::new();
        for (f, &b) in best.iter().enumerate() {
            let mut logits = vec![-10f32; 64];
            logits[b as usize] = 0.0;
            for &(sf, t, l) in set {
                if sf == f {
                    logits[t as usize] = l;
                }
            }
            score_row(&logits, 0, &c.tokens, &mut out);
        }
        out
    }

    fn ctc(ids: &[u32]) -> Vec<Token> {
        crate::granite::ctc_tokens(ids, 0)
    }

    #[test]
    fn spotting_replaces_whole_greedy_words_when_the_word_is_close() {
        let tok = tokenizer();
        let (c, _) = Compiled::new(&special(&["kubectl"]), &tok);
        // Greedy: "▁the ▁cube ▁control ,".
        let best = [6, 0, 4, 4, 0, 5, 5, 0, 7, 0];
        let greedy = ctc(&best);
        // The word's tokens are a little less likely than the greedy ones on the same frames.
        let close = [(2, 1, -0.5), (3, 2, -0.5), (5, 3, -0.5), (6, 3, -0.5)];
        let (out, ranges) = spot(&c, &tok, &greedy, &scores(&c, &best, &close), 0);
        let ids: Vec<u32> = out.iter().map(|t| t.id).collect();
        assert_eq!(ids, vec![6, 1, 2, 3, 7]);
        assert_eq!(ranges, vec![1..4]);
        assert_eq!((out[3].frame, out[3].duration), (5, 2));
        // Far from it, the greedy words stay.
        let far = [(2, 1, -6.0), (3, 2, -6.0), (5, 3, -6.0), (6, 3, -6.0)];
        assert_eq!(
            spot(&c, &tok, &greedy, &scores(&c, &best, &far), 0).0,
            greedy
        );
        // Frames before `first` aren't touched.
        let from5 = spot(&c, &tok, &greedy[2..], &scores(&c, &best, &close), 5);
        assert_eq!(from5.0, greedy[2..]);
    }

    #[test]
    fn spotting_never_inserts_a_word_into_silence() {
        let tok = tokenizer();
        let (c, _) = Compiled::new(&special(&["kubectl"]), &tok);
        let best = [6, 0, 0, 0, 0, 0];
        let greedy = ctc(&best);
        let strong = [(2, 1, 0.0), (3, 2, 0.0), (4, 3, 0.0)];
        assert_eq!(
            spot(&c, &tok, &greedy, &scores(&c, &best, &strong), 0).0,
            greedy
        );
    }

    #[test]
    fn a_cheap_path_over_blanks_doesnt_hide_a_spot_over_a_word() {
        let tok = tokenizer();
        let (c, _) = Compiled::new(&special(&["kubectl"]), &tok);
        // Greedy "▁the ▁cube" then silence. The word fits the silence better, which may not
        // replace anything, than "▁cube" and the silence after it, which may.
        let best = [6, 0, 4, 4, 0, 0, 0, 0];
        let greedy = ctc(&best);
        // Its last token fits only frame 6, where the path over the silence alone is cheaper.
        let set = [
            (2, 1, -0.5),
            (3, 2, -0.5),
            (4, 1, 0.0),
            (5, 2, 0.0),
            (6, 3, -0.5),
        ];
        let (out, ranges) = spot(&c, &tok, &greedy, &scores(&c, &best, &set), 0);
        let ids: Vec<u32> = out.iter().map(|t| t.id).collect();
        assert_eq!(ids, vec![6, 1, 2, 3]);
        assert_eq!(ranges, vec![1..4]);
    }

    #[test]
    fn a_short_word_needs_a_larger_margin() {
        let tok = tokenizer();
        // "kub" has three letters in two tokens, "▁ku" and "b".
        assert_eq!(tok.encode("kub"), vec![1, 14]);
        let best = [6, 0, 4, 4, 0, 0];
        let greedy = ctc(&best);
        let set = [(2, 1, -1.0), (3, 14, -1.0)];
        let spotted = |short: f32| {
            let mut list = special(&["kub"]);
            list.set_boost(Boost::default().with_short_word_margin(short).unwrap());
            let (c, _) = Compiled::new(&list, &tok);
            spot(&c, &tok, &greedy, &scores(&c, &best, &set), 0).1.len()
        };
        assert_eq!(spotted(0.0), 1);
        assert_eq!(spotted(1.0), 1);
        assert_eq!(spotted(2.5), 0);
    }

    #[test]
    fn a_spot_of_several_words_comes_back_as_one_range() {
        let tok = tokenizer();
        let (c, _) = Compiled::new(&special(&["New York city"]), &tok);
        // Greedy "▁the ▁cube ▁control ▁the", the phrase fits the middle words.
        let best = [6, 0, 4, 0, 5, 0, 6, 0];
        let greedy = ctc(&best);
        let set = [(2, 42, -0.3), (3, 43, -0.3), (4, 44, -0.3)];
        let (out, ranges) = spot(&c, &tok, &greedy, &scores(&c, &best, &set), 0);
        let ids: Vec<u32> = out.iter().map(|t| t.id).collect();
        assert_eq!(ids, vec![6, 42, 43, 44, 6]);
        assert_eq!(ranges, vec![1..4]);
    }
}
