//! A small pool of worker threads that runs numbered tasks.
//!
//! `run(tasks, f)` calls `f(i)` once for every `i` in `0..tasks`, on the calling thread and
//! the workers, and returns when all calls have finished. Callers give each task a fixed set of
//! outputs, so which thread runs a task never changes a result.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How long an idle worker spins before it sleeps.
const SPIN: Duration = Duration::from_micros(50);

type Task<'a> = dyn Fn(usize) + Sync + 'a;

struct Job {
    /// The task function, its lifetime erased. `run` keeps it alive until every worker that
    /// picked up the job has left it.
    f: *const Task<'static>,
    tasks: usize,
}

struct State {
    epoch: u64,
    job: Option<Job>,
    shutdown: bool,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    /// A copy of `state.epoch` that spinning workers can read without the lock.
    epoch: AtomicU64,
    next: AtomicUsize,
    remaining: AtomicUsize,
    /// Workers inside the current job.
    active: AtomicUsize,
    panicked: AtomicBool,
}

// The raw task pointer is only dereferenced while `run` keeps the task alive.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// A fixed set of worker threads. The thread that calls [`Pool::run`] works too, so a pool of
/// `n` threads starts `n - 1` workers.
pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    threads: usize,
    /// Serializes calls to `run` from different threads.
    running: Mutex<()>,
}

impl Pool {
    /// A pool of `threads` threads in all (at least 1).
    pub fn new(threads: usize) -> Pool {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                epoch: 0,
                job: None,
                shutdown: false,
            }),
            wake: Condvar::new(),
            epoch: AtomicU64::new(0),
            next: AtomicUsize::new(0),
            remaining: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
        });
        let workers = (1..threads)
            .map(|i| {
                let shared = shared.clone();
                std::thread::Builder::new()
                    .name(format!("speech-worker-{i}"))
                    .spawn(move || worker(&shared))
                    .expect("couldn't start a worker thread")
            })
            .collect();
        Pool {
            shared,
            workers,
            threads,
            running: Mutex::new(()),
        }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Calls `f(i)` for every `i` in `0..tasks` and waits for all of them.
    ///
    /// A panic in a task is raised again here once the other tasks have finished.
    pub fn run(&self, tasks: usize, f: &(dyn Fn(usize) + Sync)) {
        if tasks == 0 {
            return;
        }
        if self.threads == 1 || tasks == 1 {
            (0..tasks).for_each(f);
            return;
        }
        let _guard = self.running.lock().unwrap_or_else(|e| e.into_inner());
        let shared = &*self.shared;
        // SAFETY: the pointer is cleared from the state and every worker has left the job
        // before this function returns, so `f` outlives every use.
        let erased: *const Task<'static> =
            unsafe { std::mem::transmute::<*const Task<'_>, *const Task<'static>>(f) };
        {
            let mut state = shared.state.lock().unwrap();
            shared.next.store(0, Ordering::Relaxed);
            shared.remaining.store(tasks, Ordering::Relaxed);
            state.job = Some(Job { f: erased, tasks });
            state.epoch += 1;
            shared.epoch.store(state.epoch, Ordering::Release);
            shared.wake.notify_all();
        }

        // The caller works through tasks like any worker.
        run_tasks(shared, f, tasks);

        let start = Instant::now();
        while shared.remaining.load(Ordering::Acquire) != 0 {
            if start.elapsed() < SPIN {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        shared.state.lock().unwrap().job = None;
        while shared.active.load(Ordering::Acquire) != 0 {
            std::thread::yield_now();
        }
        if shared.panicked.swap(false, Ordering::Relaxed) {
            panic!("a task on a worker thread panicked");
        }
    }
}

fn run_tasks(shared: &Shared, f: &Task<'_>, tasks: usize) {
    loop {
        let i = shared.next.fetch_add(1, Ordering::Relaxed);
        if i >= tasks {
            return;
        }
        if catch_unwind(AssertUnwindSafe(|| f(i))).is_err() {
            shared.panicked.store(true, Ordering::Relaxed);
        }
        shared.remaining.fetch_sub(1, Ordering::Release);
    }
}

fn worker(shared: &Shared) {
    let mut seen = 0u64;
    loop {
        // Spin briefly in case the next job follows soon, as it does between kernels.
        let start = Instant::now();
        while shared.epoch.load(Ordering::Acquire) == seen && start.elapsed() < SPIN {
            std::hint::spin_loop();
        }
        let (f, tasks) = {
            let mut state = shared.state.lock().unwrap();
            while state.epoch == seen && !state.shutdown {
                state = shared.wake.wait(state).unwrap();
            }
            if state.shutdown {
                return;
            }
            seen = state.epoch;
            match &state.job {
                Some(job) => {
                    shared.active.fetch_add(1, Ordering::AcqRel);
                    (job.f, job.tasks)
                }
                // The job finished before this worker woke.
                None => continue,
            }
        };
        // SAFETY: `run` waits for `active` to drop to zero before `f` goes away.
        run_tasks(shared, unsafe { &*f }, tasks);
        shared.active.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().shutdown = true;
        self.shared.wake.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// A raw pointer that tasks may share, for writing disjoint parts of one buffer.
#[derive(Clone, Copy)]
pub(crate) struct SharedMut<T>(pub *mut T);

unsafe impl<T> Send for SharedMut<T> {}
unsafe impl<T> Sync for SharedMut<T> {}

impl<T> SharedMut<T> {
    /// The pointer. A method rather than the field, so closures capture the whole wrapper.
    pub(crate) fn get(self) -> *mut T {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_task_runs_once() {
        for threads in [1, 2, 4, 7] {
            let pool = Pool::new(threads);
            for tasks in [0, 1, 3, 100, 1000] {
                let counts: Vec<AtomicUsize> = (0..tasks).map(|_| AtomicUsize::new(0)).collect();
                pool.run(tasks, &|i| {
                    counts[i].fetch_add(1, Ordering::Relaxed);
                });
                assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
            }
        }
    }

    #[test]
    fn many_short_jobs_in_a_row_finish() {
        let pool = Pool::new(4);
        let total = AtomicUsize::new(0);
        for _ in 0..2000 {
            pool.run(8, &|i| {
                total.fetch_add(i, Ordering::Relaxed);
            });
        }
        assert_eq!(total.load(Ordering::Relaxed), 2000 * 28);
    }

    #[test]
    fn a_panicking_task_is_raised_on_the_caller() {
        let pool = Pool::new(3);
        let result = catch_unwind(AssertUnwindSafe(|| {
            pool.run(16, &|i| {
                if i == 7 {
                    panic!("task 7");
                }
            })
        }));
        assert!(result.is_err());
        // The pool still works afterwards.
        let n = AtomicUsize::new(0);
        pool.run(5, &|_| {
            n.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(n.load(Ordering::Relaxed), 5);
    }
}
