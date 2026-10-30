//! The paired bootstrap over utterances: whether one system's word error rate differs from
//! another's on the same test set by more than chance would explain.
//!
//! Both systems are scored on every utterance. Each resample draws as many utterances as the set
//! has, with replacement, and computes the difference of the two corpus word error rates on the
//! draw. The spread of the differences over many resamples gives the interval.

/// Errors and reference words of one utterance, for systems A and B.
#[derive(Clone, Copy, Debug)]
pub struct Paired {
    pub errors_a: usize,
    pub errors_b: usize,
    pub words: usize,
}

/// The result of the bootstrap, in percentage points of WER (B minus A).
#[derive(Debug)]
pub struct Interval {
    pub difference: f64,
    pub low: f64,
    pub high: f64,
    /// The share of resamples in which B's WER is lower than A's.
    pub b_better: f64,
    pub resamples: usize,
}

/// Runs `resamples` resamples with a fixed seed, so the interval is the same on every run.
pub fn paired(utterances: &[Paired], resamples: usize, seed: u64) -> Interval {
    let n = utterances.len();
    let total = |sel: &mut dyn FnMut() -> usize| {
        let (mut a, mut b, mut w) = (0usize, 0usize, 0usize);
        for _ in 0..n {
            let u = utterances[sel()];
            a += u.errors_a;
            b += u.errors_b;
            w += u.words;
        }
        if w == 0 {
            0.0
        } else {
            100.0 * (b as f64 - a as f64) / w as f64
        }
    };
    let mut i = 0;
    let difference = total(&mut || {
        i += 1;
        i - 1
    });
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % n as u64) as usize
    };
    let mut diffs: Vec<f64> = (0..resamples).map(|_| total(&mut next)).collect();
    diffs.sort_by(|x, y| x.total_cmp(y));
    let at = |q: f64| diffs[((q * (resamples - 1) as f64).round() as usize).min(resamples - 1)];
    Interval {
        difference,
        low: at(0.025),
        high: at(0.975),
        b_better: diffs.iter().filter(|&&d| d < 0.0).count() as f64 / resamples as f64,
        resamples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_systems_differ_by_nothing() {
        let set: Vec<Paired> = (0..100)
            .map(|i| Paired {
                errors_a: i % 3,
                errors_b: i % 3,
                words: 10,
            })
            .collect();
        let r = paired(&set, 1000, 1);
        assert_eq!(
            (r.difference, r.low, r.high, r.b_better),
            (0.0, 0.0, 0.0, 0.0)
        );
    }

    #[test]
    fn a_consistent_gain_has_an_interval_below_zero() {
        // B makes one error fewer on every tenth utterance: 10 fewer errors in 1,000 words.
        let set: Vec<Paired> = (0..100)
            .map(|i| Paired {
                errors_a: 2,
                errors_b: if i % 10 == 0 { 1 } else { 2 },
                words: 10,
            })
            .collect();
        let r = paired(&set, 2000, 7);
        assert!((r.difference + 1.0).abs() < 1e-12);
        assert!(r.high < 0.0 && r.low > -2.5, "{r:?}");
        assert!(r.b_better > 0.99);
    }
}
