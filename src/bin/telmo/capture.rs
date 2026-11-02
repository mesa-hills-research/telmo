//! Where a microphone's audio lands. The audio callback writes into a fixed ring without locks
//! or allocation, and the session thread drains the ring into the take. A take that lost
//! samples or went through a device failure is refused when it stops, never returned.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering::*};
use std::sync::{Mutex, MutexGuard};

/// The ring holds about 6 seconds at 44.1 kHz. The session drains it every 100 ms.
const RING: usize = 1 << 18;

/// Why a take can't be kept.
#[derive(Debug, PartialEq)]
pub enum Refused {
    /// The device stopped or broke, and nothing more will be heard from it.
    Failed(String),
    /// The ring filled, so samples are missing.
    Dropped,
}

pub struct Capture {
    ring: Box<[AtomicU32]>,
    /// Samples ever written to the ring, by the callback only.
    written: AtomicUsize,
    /// Samples ever read from the ring, by the session thread only.
    read: AtomicUsize,
    keeping: AtomicBool,
    dropped: AtomicBool,
    /// The largest sample since `level` last asked, as the bits of a non-negative `f32`.
    peak: AtomicU32,
    failure: Mutex<Option<String>>,
    /// The audio of the current take, touched by the session thread only.
    take: Mutex<Vec<f32>>,
}

impl Capture {
    pub fn new() -> Capture {
        Capture {
            ring: (0..RING).map(|_| AtomicU32::new(0)).collect(),
            written: AtomicUsize::new(0),
            read: AtomicUsize::new(0),
            keeping: AtomicBool::new(false),
            dropped: AtomicBool::new(false),
            peak: AtomicU32::new(0),
            failure: Mutex::new(None),
            take: Mutex::new(Vec::new()),
        }
    }

    /// Takes interleaved samples from the audio callback. It never blocks or allocates. When
    /// the ring is full it drops the samples and marks the take.
    pub fn push(&self, samples: impl Iterator<Item = f32>, channels: usize) {
        let keeping = self.keeping.load(Relaxed);
        let read = self.read.load(Acquire);
        let mut written = self.written.load(Relaxed);
        let (mut peak, mut dropped) = (0.0_f32, false);
        downmix(samples, channels, |sample| {
            peak = peak.max(sample.abs());
            if !keeping {
                return;
            }
            if written - read < RING {
                self.ring[written % RING].store(sample.to_bits(), Relaxed);
                written += 1;
            } else {
                dropped = true;
            }
        });
        self.written.store(written, Release);
        if dropped {
            self.dropped.store(true, Relaxed);
        }
        self.peak.fetch_max(peak.to_bits(), Relaxed);
    }

    /// Records that the device failed, from the audio stream's error callback. Only the first
    /// failure is kept.
    pub fn fail(&self, reason: String) {
        self.failure().get_or_insert(reason);
    }

    fn failure(&self) -> MutexGuard<'_, Option<String>> {
        self.failure.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn take(&self) -> MutexGuard<'_, Vec<f32>> {
        self.take.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Moves what the callback wrote from the ring into the take.
    fn drain(&self) {
        let written = self.written.load(Acquire);
        let mut read = self.read.load(Relaxed);
        let mut take = self.take();
        while read < written {
            take.push(f32::from_bits(self.ring[read % RING].load(Relaxed)));
            read += 1;
        }
        self.read.store(read, Release);
    }

    fn check(&self) -> Result<(), Refused> {
        if let Some(reason) = self.failure().clone() {
            Err(Refused::Failed(reason))
        } else if self.dropped.load(Relaxed) {
            Err(Refused::Dropped)
        } else {
            Ok(())
        }
    }

    /// Begins a new take, discarding any audio heard before. It refuses once the device failed.
    pub fn start(&self) -> Result<(), Refused> {
        self.keeping.store(false, Relaxed);
        self.drain();
        self.take().clear();
        self.dropped.store(false, Relaxed);
        self.level();
        self.check()?;
        self.keeping.store(true, Relaxed);
        Ok(())
    }

    /// Moves the audio into the take, and says if the take can no longer be kept.
    pub fn poll(&self) -> Result<(), Refused> {
        self.drain();
        self.check()
    }

    /// Ends the take and returns its audio, or why it can't be kept. A failure anywhere since
    /// the device opened refuses the take, so no stop can slip past one.
    pub fn stop(&self) -> Result<Vec<f32>, Refused> {
        self.keeping.store(false, Relaxed);
        self.poll()?;
        Ok(std::mem::take(&mut *self.take()))
    }

    /// The largest sample, from 0 to 1, since the last call.
    pub fn level(&self) -> f32 {
        f32::from_bits(self.peak.swap(0, Relaxed))
    }
}

/// Mixes interleaved samples down to one channel, handing each frame's average to `emit`. A
/// partial frame at the end is dropped.
pub fn downmix(samples: impl Iterator<Item = f32>, channels: usize, mut emit: impl FnMut(f32)) {
    let channels = channels.max(1);
    let (mut sum, mut count) = (0.0, 0);
    for sample in samples {
        sum += sample;
        count += 1;
        if count == channels {
            emit(sum / channels as f32);
            (sum, count) = (0.0, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn mono(samples: &[f32]) -> impl Iterator<Item = f32> + '_ {
        samples.iter().copied()
    }

    #[test]
    fn channels_are_averaged_and_a_partial_frame_is_dropped() {
        let mut out = Vec::new();
        downmix([1.0, 0.0, 0.5, 0.5, 0.25].into_iter(), 2, |s| out.push(s));
        assert_eq!(out, [0.5, 0.5]);
        out.clear();
        downmix([0.25, 0.5].into_iter(), 1, |s| out.push(s));
        assert_eq!(out, [0.25, 0.5]);
    }

    #[test]
    fn a_take_holds_what_was_heard_between_start_and_stop() {
        let capture = Capture::new();
        capture.push(mono(&[0.9]), 1);
        capture.start().unwrap();
        capture.push([0.5, -0.5, 0.25, 0.75].into_iter(), 2);
        assert_eq!(capture.level(), 0.5);
        assert_eq!(capture.level(), 0.0, "the level is since the last ask");
        assert_eq!(capture.stop(), Ok(vec![0.0, 0.5]));
        capture.push(mono(&[0.9]), 1);
        capture.start().unwrap();
        assert_eq!(capture.stop(), Ok(vec![]), "the next take starts empty");
    }

    #[test]
    fn a_stop_cannot_slip_past_a_failure() {
        // The device fails while a stop is already waiting, so the session reaches stop
        // without having polled.
        let capture = Capture::new();
        capture.start().unwrap();
        capture.push(mono(&[0.5; 100]), 1);
        capture.fail("device disconnected".into());
        assert_eq!(
            capture.stop(),
            Err(Refused::Failed("device disconnected".into()))
        );
        // A failure between takes refuses the next one before it starts.
        assert_eq!(
            capture.start(),
            Err(Refused::Failed("device disconnected".into()))
        );
        capture.fail("another".into());
        assert_eq!(
            capture.poll(),
            Err(Refused::Failed("device disconnected".into())),
            "the first reason stays"
        );
    }

    #[test]
    fn a_full_ring_drops_without_blocking_and_refuses_the_take() {
        let capture = Capture::new();
        capture.start().unwrap();
        // Nothing drains, as when the session thread is stalled.
        let chunk = vec![0.1_f32; RING / 4];
        for _ in 0..6 {
            capture.push(mono(&chunk), 1);
        }
        assert_eq!(capture.poll(), Err(Refused::Dropped));
        assert_eq!(capture.stop(), Err(Refused::Dropped));
        // The next take is clean.
        capture.start().unwrap();
        capture.push(mono(&chunk), 1);
        assert_eq!(capture.stop().unwrap().len(), chunk.len());
    }

    #[test]
    fn samples_cross_threads_in_order_while_draining() {
        let capture = Arc::new(Capture::new());
        capture.start().unwrap();
        let total = RING - 1000;
        let writer = {
            let capture = Arc::clone(&capture);
            std::thread::spawn(move || {
                for start in (0..total).step_by(480) {
                    let end = (start + 480).min(total);
                    capture.push((start..end).map(|n| n as f32), 1);
                }
            })
        };
        let mut polls = 0;
        while !writer.is_finished() {
            capture.poll().unwrap();
            polls += 1;
            std::thread::yield_now();
        }
        writer.join().unwrap();
        let take = capture.stop().unwrap();
        assert_eq!(take.len(), total, "after {polls} polls");
        assert!(take.iter().enumerate().all(|(n, &s)| s == n as f32));
    }
}
