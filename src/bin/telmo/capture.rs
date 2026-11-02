//! Where a microphone's audio lands. The audio callback writes into a fixed ring without locks
//! or allocation, and the session thread drains the ring into the take. A take that lost
//! samples or went through a device failure is refused when it stops, never returned.
//!
//! A take's boundary is a generation number that the session bumps at every start and stop. It
//! is odd while a take records. The callback reads it once on entry and tags every sample it
//! writes, and any loss it notes, with that number. The session keeps only the samples tagged
//! with its take's number. A callback that was running when a take ended finishes under the old
//! number, so its audio and its loss are ignored, whether they appear before or after the next
//! take starts. Nothing in the callback waits.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Mutex, MutexGuard};

/// The ring holds about 6 seconds at 44.1 kHz. The session drains it every 100 ms. It is a
/// power of two, so an index stays right when the counters wrap around.
const RING: usize = 1 << 18;

/// Why a take can't be kept.
#[derive(Debug, PartialEq)]
pub enum Refused {
    /// The device stopped or broke, and nothing more will be heard from it.
    Failed(String),
    /// The ring filled, so samples are missing.
    Dropped,
}

/// The take being recorded, owned by the session thread.
struct Take {
    /// The generation whose samples belong to it.
    generation: u32,
    samples: Vec<f32>,
}

pub struct Capture {
    /// Each slot holds a generation in its upper half and a sample's bits in its lower half.
    ring: Box<[AtomicU64]>,
    /// Samples ever written to the ring, by the callback only. It wraps around.
    written: AtomicUsize,
    /// Samples ever read from the ring, by the session thread only. It wraps around.
    read: AtomicUsize,
    /// Odd while a take records, even otherwise. Only the session thread changes it.
    generation: AtomicU32,
    /// The newest generation that lost samples.
    dropped: AtomicU32,
    /// The largest sample since `level` last asked, as the bits of a non-negative `f32`.
    peak: AtomicU32,
    failure: Mutex<Option<String>>,
    take: Mutex<Take>,
}

impl Capture {
    pub fn new() -> Capture {
        Capture {
            ring: (0..RING).map(|_| AtomicU64::new(0)).collect(),
            written: AtomicUsize::new(0),
            read: AtomicUsize::new(0),
            generation: AtomicU32::new(0),
            dropped: AtomicU32::new(0),
            peak: AtomicU32::new(0),
            failure: Mutex::new(None),
            take: Mutex::new(Take {
                generation: 0,
                samples: Vec::new(),
            }),
        }
    }

    /// Takes interleaved samples from the audio callback. It never blocks or allocates. When
    /// the ring is full it drops the samples and notes the loss for the take it belongs to.
    pub fn push(&self, samples: impl Iterator<Item = f32>, channels: usize) {
        let generation = self.generation.load(Acquire);
        let keeping = generation % 2 == 1;
        let read = self.read.load(Acquire);
        let mut written = self.written.load(Relaxed);
        let (mut peak, mut dropped) = (0.0_f32, false);
        downmix(samples, channels, |sample| {
            peak = peak.max(sample.abs());
            if !keeping {
                return;
            }
            if written.wrapping_sub(read) < RING {
                let slot = u64::from(generation) << 32 | u64::from(sample.to_bits());
                self.ring[written % RING].store(slot, Relaxed);
                written = written.wrapping_add(1);
            } else {
                dropped = true;
            }
        });
        self.written.store(written, Release);
        if dropped {
            self.dropped.fetch_max(generation, Relaxed);
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

    fn take(&self) -> MutexGuard<'_, Take> {
        self.take.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Moves what the callback wrote from the ring into the take, and drops samples that
    /// belong to other takes.
    fn drain(&self, take: &mut Take) {
        let written = self.written.load(Acquire);
        let mut read = self.read.load(Relaxed);
        for _ in 0..written.wrapping_sub(read) {
            let slot = self.ring[read % RING].load(Relaxed);
            if (slot >> 32) as u32 == take.generation {
                take.samples.push(f32::from_bits(slot as u32));
            }
            read = read.wrapping_add(1);
        }
        self.read.store(read, Release);
    }

    /// Whether the take of `generation` can still be kept.
    fn check(&self, generation: u32) -> Result<(), Refused> {
        if let Some(reason) = self.failure().clone() {
            Err(Refused::Failed(reason))
        } else if generation % 2 == 1 && self.dropped.load(Relaxed) == generation {
            Err(Refused::Dropped)
        } else {
            Ok(())
        }
    }

    /// Begins a new take, discarding any audio heard before. It refuses once the device failed.
    pub fn start(&self) -> Result<(), Refused> {
        let mut take = self.take();
        let ended = self.generation.load(Relaxed);
        // The new number is odd and higher than any callback may still be running under.
        let generation = (ended | 1).wrapping_add(2);
        take.generation = generation;
        take.samples.clear();
        self.drain(&mut take);
        self.level();
        self.check(generation)?;
        self.generation.store(generation, Release);
        Ok(())
    }

    /// Moves the audio into the take, and says if the take can no longer be kept.
    pub fn poll(&self) -> Result<(), Refused> {
        let mut take = self.take();
        self.drain(&mut take);
        self.check(take.generation)
    }

    /// Ends the take and returns its audio, or why it can't be kept. A failure anywhere since
    /// the device opened refuses the take, so no stop can slip past one. Audio that a callback
    /// running at this moment adds afterwards is not part of the take.
    pub fn stop(&self) -> Result<Vec<f32>, Refused> {
        let mut take = self.take();
        let recording = self.generation.load(Relaxed);
        let ended = if recording % 2 == 1 {
            recording.wrapping_add(1)
        } else {
            recording
        };
        self.generation.store(ended, Release);
        self.drain(&mut take);
        let verdict = self.check(take.generation);
        // Later arrivals under the take's number are ignored from here on.
        take.generation = ended;
        let samples = std::mem::take(&mut take.samples);
        verdict.map(|()| samples)
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
    use std::sync::{Arc, Barrier};

    impl Capture {
        /// A capture whose ring counters start at `position`, to reach their wrap-around.
        fn at(position: usize) -> Capture {
            let capture = Capture::new();
            capture.written.store(position, Relaxed);
            capture.read.store(position, Relaxed);
            capture
        }
    }

    /// Pushes `samples` from a thread that stops inside the callback, after it has read its
    /// take, until `release` opens. Returns the thread, which is waiting by the time this returns.
    fn callback_in_flight(
        capture: &Arc<Capture>,
        samples: Vec<f32>,
        release: &Arc<Barrier>,
    ) -> std::thread::JoinHandle<()> {
        let entered = Arc::new(Barrier::new(2));
        let thread = {
            let (capture, entered, release) = (capture.clone(), entered.clone(), release.clone());
            std::thread::spawn(move || {
                let mut first = true;
                capture.push(
                    samples.into_iter().inspect(move |_| {
                        if std::mem::take(&mut first) {
                            entered.wait();
                            release.wait();
                        }
                    }),
                    1,
                );
            })
        };
        entered.wait();
        thread
    }

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

    #[test]
    fn a_callback_running_at_stop_does_not_reach_the_next_take() {
        let capture = Arc::new(Capture::new());
        capture.start().unwrap();
        let release = Arc::new(Barrier::new(2));
        let callback = callback_in_flight(&capture, vec![0.75], &release);
        assert_eq!(capture.stop(), Ok(vec![]), "it hadn't published yet");
        capture.start().unwrap();
        release.wait();
        callback.join().unwrap();
        assert_eq!(
            capture.stop(),
            Ok(vec![]),
            "its audio belongs to the take before"
        );
    }

    #[test]
    fn a_loss_noted_by_a_callback_running_at_stop_does_not_refuse_the_next_take() {
        let capture = Arc::new(Capture::new());
        capture.start().unwrap();
        capture.push(vec![0.1; RING].into_iter(), 1);
        let release = Arc::new(Barrier::new(2));
        // The callback saw a full ring, which the session then drained.
        let callback = callback_in_flight(&capture, vec![0.75], &release);
        assert_eq!(capture.stop().map(|take| take.len()), Ok(RING));
        capture.start().unwrap();
        release.wait();
        callback.join().unwrap();
        assert_eq!(capture.poll(), Ok(()));
        capture.push([0.5].into_iter(), 1);
        assert_eq!(capture.stop(), Ok(vec![0.5]));
    }

    #[test]
    fn a_callback_that_loses_samples_in_its_own_take_refuses_it() {
        let capture = Arc::new(Capture::new());
        capture.start().unwrap();
        capture.push(vec![0.1; RING].into_iter(), 1);
        let release = Arc::new(Barrier::new(2));
        let callback = callback_in_flight(&capture, vec![0.75], &release);
        release.wait();
        callback.join().unwrap();
        assert_eq!(capture.stop(), Err(Refused::Dropped));
    }

    #[test]
    fn counters_that_wrap_around_keep_every_sample() {
        for position in [usize::MAX - 1, usize::MAX - RING / 2, usize::MAX] {
            let capture = Capture::at(position);
            capture.start().unwrap();
            let samples: Vec<f32> = (0..5000).map(|n| n as f32).collect();
            capture.push(mono(&samples), 1);
            capture.poll().unwrap();
            capture.push(mono(&samples), 1);
            let take = capture.stop().unwrap();
            assert_eq!(take.len(), 2 * samples.len(), "from {position}");
            assert!(take.chunks(5000).all(|half| half == samples));
        }
    }

    #[test]
    fn a_full_ring_that_wraps_around_is_detected_and_recovers() {
        let capture = Capture::at(usize::MAX - 100);
        capture.start().unwrap();
        capture.push(vec![0.1; RING + 1].into_iter(), 1);
        assert_eq!(capture.stop(), Err(Refused::Dropped));
        capture.start().unwrap();
        capture.push(vec![0.2; RING].into_iter(), 1);
        assert_eq!(capture.stop().map(|take| take.len()), Ok(RING));
    }
}
