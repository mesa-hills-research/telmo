//! Numeric kernels in f32, with a scalar version of each (also the reference in tests) and
//! AVX2 with FMA where the CPU has them.
//!
//! The matrix product splits work between threads by output columns only, so every output is
//! summed in the same order whatever the thread count, and results are bit-identical across
//! thread counts on one instruction set. Row-wise kernels split by rows, which doesn't change
//! any result either.

// Kernel entry points take raw buffers, strides and sizes, which makes for long signatures.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use crate::threads::{Pool, SharedMut};

#[cfg(target_arch = "x86_64")]
mod avx2;
pub(crate) mod scalar;

/// Output columns per weight panel, the width of the matrix product's register tile.
pub const PANEL: usize = 16;
/// Rows of the matrix product's register tile.
pub const TILE_ROWS: usize = 6;

/// The instruction set the kernels use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Isa {
    Scalar,
    /// AVX2 with FMA, on x86-64.
    Avx2,
}

impl Isa {
    /// The best instruction set this CPU supports.
    pub fn detect() -> Isa {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return Isa::Avx2;
        }
        Isa::Scalar
    }

    /// Whether this CPU can run the instruction set.
    pub fn is_supported(self) -> bool {
        match self {
            Isa::Scalar => true,
            Isa::Avx2 => Isa::detect() == Isa::Avx2,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::Avx2 => "avx2+fma",
        }
    }
}

/// A zeroed f32 buffer aligned to 64 bytes.
pub struct AlignedBuf {
    ptr: NonNull<f32>,
    len: usize,
}

unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    pub fn zeroed(len: usize) -> AlignedBuf {
        if len == 0 {
            return AlignedBuf {
                ptr: NonNull::dangling(),
                len: 0,
            };
        }
        let layout = Self::layout(len);
        // SAFETY: the layout has a nonzero size.
        let ptr = unsafe { alloc_zeroed(layout) } as *mut f32;
        let ptr = NonNull::new(ptr).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        AlignedBuf { ptr, len }
    }

    fn layout(len: usize) -> Layout {
        Layout::from_size_align(len * 4, 64).expect("buffer too large")
    }
}

impl Deref for AlignedBuf {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        // SAFETY: the pointer covers `len` initialized floats (or is dangling with len 0).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for AlignedBuf {
    fn deref_mut(&mut self) -> &mut [f32] {
        // SAFETY: as in deref, and `&mut self` makes the access unique.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        if self.len != 0 {
            // SAFETY: allocated in `zeroed` with this layout.
            unsafe { dealloc(self.ptr.as_ptr() as *mut u8, Self::layout(self.len)) }
        }
    }
}

/// A weight matrix of `n` outputs and `k` inputs, stored as panels of 16 outputs with the
/// inputs running inside each panel. Outputs past `n` in the last panel are zero.
pub struct PackedWeights {
    n: usize,
    k: usize,
    data: AlignedBuf,
}

impl PackedWeights {
    /// Packs a row-major `[n, k]` matrix, the layout of a PyTorch `Linear` weight.
    pub fn from_rows(w: &[f32], n: usize, k: usize) -> PackedWeights {
        assert_eq!(w.len(), n * k);
        let panels = n.div_ceil(PANEL);
        let mut data = AlignedBuf::zeroed(panels * k * PANEL);
        for (o, row) in w.chunks_exact(k).enumerate() {
            let (p, j) = (o / PANEL, o % PANEL);
            let panel = &mut data[p * k * PANEL..(p + 1) * k * PANEL];
            for (i, &v) in row.iter().enumerate() {
                panel[i * PANEL + j] = v;
            }
        }
        PackedWeights { n, k, data }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn view(&self) -> BView<'_> {
        BView {
            data: &self.data,
            n: self.n,
            k: self.k,
            panel_stride: self.k * PANEL,
            k_stride: PANEL,
        }
    }

    /// Bytes of weight data.
    pub fn bytes(&self) -> usize {
        self.data.len() * 4
    }
}

/// The right-hand side of a matrix product, `[k, n]`, as a set of 16-column panels: column
/// block `p` at input `i` is the 16 floats at `p * panel_stride + i * k_stride`.
#[derive(Clone, Copy)]
pub struct BView<'a> {
    data: &'a [f32],
    n: usize,
    k: usize,
    panel_stride: usize,
    k_stride: usize,
}

impl<'a> BView<'a> {
    /// A row-major `[k, ld]` matrix whose first `n` columns are used. Columns from `n` up to the
    /// next multiple of 16 are read but their results are dropped, and the buffer must hold
    /// them (`data.len() >= (k - 1) * ld + n.next_multiple_of(16)`).
    pub fn row_major(data: &'a [f32], k: usize, n: usize, ld: usize) -> BView<'a> {
        let view = BView {
            data,
            n,
            k,
            panel_stride: PANEL,
            k_stride: ld,
        };
        view.check();
        view
    }

    fn check(&self) {
        if self.n == 0 || self.k == 0 {
            return;
        }
        let last =
            (self.n.div_ceil(PANEL) - 1) * self.panel_stride + (self.k - 1) * self.k_stride + PANEL;
        assert!(
            last <= self.data.len(),
            "the matrix view reads past its buffer"
        );
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    fn panels(&self) -> usize {
        self.n.div_ceil(PANEL)
    }
}

/// An activation applied in the matrix product's epilogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    None,
    Relu,
    Silu,
}

/// What the matrix product does with each sum before storing it:
/// `c = (accumulate ? c : 0) + scale * act(sum + bias)`.
#[derive(Clone, Copy)]
pub struct Epilogue<'a> {
    pub bias: Option<&'a [f32]>,
    pub act: Act,
    pub scale: f32,
    pub accumulate: bool,
}

impl Epilogue<'_> {
    pub const PLAIN: Epilogue<'static> = Epilogue {
        bias: None,
        act: Act::None,
        scale: 1.0,
        accumulate: false,
    };

    pub fn bias(bias: &[f32]) -> Epilogue<'_> {
        Epilogue {
            bias: Some(bias),
            ..Epilogue::PLAIN
        }
    }
}

/// The per-instruction-set entry points. Every function is safe to call once the table exists,
/// because a table is only made for an instruction set the CPU supports.
#[derive(Clone, Copy)]
pub struct Kernels {
    isa: Isa,
    /// Computes a `rows x 16` tile (rows <= 6) of `a * b` into `out` (row stride 16).
    /// `a` rows are `lda` apart, `b` steps `k_stride` per input.
    tile: unsafe fn(
        k: usize,
        a: *const f32,
        lda: usize,
        rows: usize,
        b: *const f32,
        k_stride: usize,
        out: &mut [f32; 96],
    ),
    /// Applies the epilogue to one tile row and stores `cols` values.
    store: fn(
        acc: &[f32],
        c: &mut [f32],
        bias: Option<&[f32]>,
        act: Act,
        scale: f32,
        accumulate: bool,
    ),
    pub layer_norm: fn(x: &[f32], out: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32),
    /// Softmax of one row in place.
    pub softmax: fn(x: &mut [f32]),
    /// `out = a * sigmoid(b)` element by element.
    pub glu: fn(a: &[f32], b: &[f32], out: &mut [f32]),
    /// SiLU in place.
    pub silu: fn(x: &mut [f32]),
    /// `x += alpha * y`.
    pub axpy: fn(alpha: f32, y: &[f32], x: &mut [f32]),
    /// One output frame of a depthwise 1-D convolution over channels-last input.
    pub dwconv1d_frame: fn(
        input: &[f32],
        channels: usize,
        frames: usize,
        t: usize,
        w: &[f32],
        kernel: usize,
        bias: &[f32],
        out: &mut [f32],
    ),
}

impl Kernels {
    /// The kernels for an instruction set, or `None` if this CPU can't run it.
    pub fn new(isa: Isa) -> Option<Kernels> {
        if !isa.is_supported() {
            return None;
        }
        Some(match isa {
            Isa::Scalar => scalar::kernels(),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => avx2::kernels(),
            #[cfg(not(target_arch = "x86_64"))]
            Isa::Avx2 => unreachable!(),
        })
    }

    pub fn detect() -> Kernels {
        Kernels::new(Isa::detect()).expect("the detected instruction set is supported")
    }

    pub fn isa(&self) -> Isa {
        self.isa
    }

    /// `c[m, n] = epilogue(a[m, k] * b[k, n])` on one thread, for the panels in `panels`.
    ///
    /// # Safety
    /// `c` must be valid for writes of rows `0..m` (stride `ldc`) in the columns of `panels`,
    /// and nothing else may access those elements during the call.
    unsafe fn gemm_panels(
        &self,
        a: &[f32],
        lda: usize,
        m: usize,
        b: &BView,
        panels: std::ops::Range<usize>,
        c: SharedMut<f32>,
        ldc: usize,
        epi: &Epilogue,
    ) {
        let k = b.k;
        let mut acc = [0f32; 96];
        for p in panels {
            let col = p * PANEL;
            let cols = (b.n - col).min(PANEL);
            let bias = epi.bias.map(|bias| &bias[col..col + cols]);
            // SAFETY: `BView::check` keeps the panel's reads inside `b.data`.
            let bp = unsafe { b.data.as_ptr().add(p * b.panel_stride) };
            let mut row = 0;
            while row < m {
                let rows = (m - row).min(TILE_ROWS);
                // SAFETY: rows `row..row + rows` of `a` hold `k` values each, checked by the caller.
                unsafe {
                    (self.tile)(
                        k,
                        a.as_ptr().add(row * lda),
                        lda,
                        rows,
                        bp,
                        b.k_stride,
                        &mut acc,
                    )
                };
                for r in 0..rows {
                    // SAFETY: the caller hands this call the columns of these panels alone.
                    let c_row = unsafe {
                        std::slice::from_raw_parts_mut(c.get().add((row + r) * ldc + col), cols)
                    };
                    (self.store)(
                        &acc[r * PANEL..r * PANEL + cols],
                        c_row,
                        bias,
                        epi.act,
                        epi.scale,
                        epi.accumulate,
                    );
                }
                row += rows;
            }
        }
    }

    /// `c[m, n] = epilogue(a[m, k] * b[k, n])`, with `a` rows `lda` apart and `c` rows `ldc` apart.
    /// The pool splits the output columns.
    pub fn matmul(
        &self,
        pool: &Pool,
        a: &[f32],
        lda: usize,
        m: usize,
        b: &BView,
        c: &mut [f32],
        ldc: usize,
        epi: &Epilogue,
    ) {
        let (k, n) = (b.k, b.n);
        if m == 0 || n == 0 {
            return;
        }
        assert!(lda >= k && ldc >= n);
        assert!(a.len() >= (m - 1) * lda + k, "a is too short");
        assert!(c.len() >= (m - 1) * ldc + n, "c is too short");
        if let Some(bias) = epi.bias {
            assert!(bias.len() >= n);
        }
        let panels = b.panels();
        let c_ptr = SharedMut(c.as_mut_ptr());
        // A few tasks per thread balance uneven threads without making tasks tiny.
        let per_task = panels.div_ceil(pool.threads() * 4).max(1);
        let tasks = panels.div_ceil(per_task);
        pool.run(tasks, &|t| {
            let range = t * per_task..((t + 1) * per_task).min(panels);
            // SAFETY: tasks get disjoint panel ranges, so they write disjoint columns of `c`.
            unsafe { self.gemm_panels(a, lda, m, b, range, c_ptr, ldc, epi) };
        });
    }

    /// `c = epilogue(a * w^T)` for a packed weight, with `a` and `c` contiguous.
    pub fn linear(
        &self,
        pool: &Pool,
        a: &[f32],
        m: usize,
        w: &PackedWeights,
        c: &mut [f32],
        epi: &Epilogue,
    ) {
        self.matmul(pool, a, w.k, m, &w.view(), c, w.n, epi);
    }

    /// The matrix product on the calling thread alone.
    pub fn matmul_serial(
        &self,
        a: &[f32],
        lda: usize,
        m: usize,
        b: &BView,
        c: &mut [f32],
        ldc: usize,
        epi: &Epilogue,
    ) {
        let (k, n) = (b.k, b.n);
        if m == 0 || n == 0 {
            return;
        }
        assert!(lda >= k && ldc >= n);
        assert!(a.len() >= (m - 1) * lda + k, "a is too short");
        assert!(c.len() >= (m - 1) * ldc + n, "c is too short");
        if let Some(bias) = epi.bias {
            assert!(bias.len() >= n);
        }
        // SAFETY: `c` is borrowed mutably for the whole call and is long enough.
        unsafe {
            self.gemm_panels(
                a,
                lda,
                m,
                b,
                0..b.panels(),
                SharedMut(c.as_mut_ptr()),
                ldc,
                epi,
            )
        };
    }
}

/// Runs `f(first_row, rows)` over `rows` rows of a buffer split into chunks, in parallel.
/// `f` gets the chunk's rows of `data` (each `width` long) as one slice.
pub fn for_rows(
    pool: &Pool,
    data: &mut [f32],
    width: usize,
    rows: usize,
    chunk: usize,
    f: &(dyn Fn(usize, &mut [f32]) + Sync),
) {
    assert!(data.len() >= rows * width);
    let ptr = SharedMut(data.as_mut_ptr());
    let tasks = rows.div_ceil(chunk.max(1));
    pool.run(tasks, &|t| {
        let first = t * chunk;
        let n = chunk.min(rows - first);
        // SAFETY: chunks cover disjoint rows of `data`.
        let part =
            unsafe { std::slice::from_raw_parts_mut(ptr.get().add(first * width), n * width) };
        f(first, part);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn random(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// The plain triple loop, in f64.
    fn reference(
        a: &[f32],
        m: usize,
        k: usize,
        w: &[f32],
        n: usize,
        epi: &Epilogue,
        c0: &[f32],
    ) -> Vec<f32> {
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f64;
                for t in 0..k {
                    s += a[i * k + t] as f64 * w[j * k + t] as f64;
                }
                let mut v = s + epi.bias.map_or(0.0, |b| b[j] as f64);
                v = match epi.act {
                    Act::None => v,
                    Act::Relu => v.max(0.0),
                    Act::Silu => v / (1.0 + (-v).exp()),
                };
                v *= epi.scale as f64;
                if epi.accumulate {
                    v += c0[i * n + j] as f64;
                }
                c[i * n + j] = v as f32;
            }
        }
        c
    }

    fn all_kernels() -> Vec<Kernels> {
        [Isa::Scalar, Isa::Avx2]
            .into_iter()
            .filter_map(Kernels::new)
            .collect()
    }

    #[test]
    fn matmul_matches_the_reference_for_every_shape_and_epilogue() {
        let pool = Pool::new(3);
        for kernels in all_kernels() {
            for &(m, k, n) in &[
                (1, 1, 1),
                (1, 7, 5),
                (5, 13, 17),
                (6, 64, 16),
                (7, 33, 31),
                (13, 100, 50),
                (64, 128, 96),
            ] {
                let a = random(m * k, 1);
                let w = random(n * k, 2);
                let bias = random(n, 3);
                let c0 = random(m * n, 4);
                let packed = PackedWeights::from_rows(&w, n, k);
                for (act, accumulate, with_bias) in [
                    (Act::None, false, false),
                    (Act::Relu, false, true),
                    (Act::Silu, true, true),
                    (Act::None, true, false),
                ] {
                    let epi = Epilogue {
                        bias: with_bias.then_some(&bias[..]),
                        act,
                        scale: 0.5,
                        accumulate,
                    };
                    let want = reference(&a, m, k, &w, n, &epi, &c0);
                    let mut got = c0.clone();
                    kernels.linear(&pool, &a, m, &packed, &mut got, &epi);
                    for (g, w) in got.iter().zip(&want) {
                        assert!(
                            (g - w).abs() <= 1e-5 * (k as f32).sqrt() * (1.0 + w.abs()),
                            "{:?} m={m} k={k} n={n}: {g} vs {w}",
                            kernels.isa()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_row_major_right_side_with_padding_works() {
        let pool = Pool::new(2);
        let (m, k, n, ld) = (9, 20, 21, 40);
        let a = random(m * k, 5);
        let b = random(k * ld, 6);
        // w[j][t] = b[t][j]
        let w: Vec<f32> = (0..n)
            .flat_map(|j| (0..k).map(move |t| (j, t)))
            .map(|(j, t)| b[t * ld + j])
            .collect();
        let want = reference(&a, m, k, &w, n, &Epilogue::PLAIN, &[]);
        for kernels in all_kernels() {
            let mut got = vec![0f32; m * 24];
            kernels.matmul(
                &pool,
                &a,
                k,
                m,
                &BView::row_major(&b, k, n, ld),
                &mut got,
                24,
                &Epilogue::PLAIN,
            );
            for i in 0..m {
                for j in 0..n {
                    assert!((got[i * 24 + j] - want[i * n + j]).abs() < 1e-4);
                }
            }
        }
    }

    #[test]
    fn results_are_bit_identical_across_thread_counts() {
        let (m, k, n) = (50, 300, 200);
        let a = random(m * k, 7);
        let w = PackedWeights::from_rows(&random(n * k, 8), n, k);
        for kernels in all_kernels() {
            let mut first: Option<Vec<f32>> = None;
            for threads in [1, 2, 3, 6, 8] {
                let pool = Pool::new(threads);
                let mut c = vec![0f32; m * n];
                kernels.linear(
                    &pool,
                    &a,
                    m,
                    &w,
                    &mut c,
                    &Epilogue {
                        act: Act::Silu,
                        ..Epilogue::PLAIN
                    },
                );
                match &first {
                    None => first = Some(c),
                    Some(f) => assert!(
                        f.iter().zip(&c).all(|(x, y)| x.to_bits() == y.to_bits()),
                        "{threads} threads differ"
                    ),
                }
            }
        }
    }

    #[test]
    fn row_kernels_agree_across_instruction_sets() {
        let scalar = Kernels::new(Isa::Scalar).unwrap();
        for kernels in all_kernels() {
            let x = random(1030, 9);
            let (g, b) = (random(1030, 10), random(1030, 11));
            let (mut o1, mut o2) = (vec![0f32; 1030], vec![0f32; 1030]);
            (scalar.layer_norm)(&x, &mut o1, &g, &b, 1e-5);
            (kernels.layer_norm)(&x, &mut o2, &g, &b, 1e-5);
            assert!(o1.iter().zip(&o2).all(|(p, q)| (p - q).abs() < 1e-5));

            let mut s1: Vec<f32> = x.iter().map(|v| v * 30.0).collect();
            let mut s2 = s1.clone();
            (scalar.softmax)(&mut s1);
            (kernels.softmax)(&mut s2);
            assert!((s2.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            assert!(
                s1.iter()
                    .zip(&s2)
                    .all(|(p, q)| (p - q).abs() <= 1e-6 + 1e-5 * p.abs())
            );

            let mut v1: Vec<f32> = x.iter().map(|v| v * 20.0).collect();
            let mut v2 = v1.clone();
            (scalar.silu)(&mut v1);
            (kernels.silu)(&mut v2);
            assert!(
                v1.iter()
                    .zip(&v2)
                    .all(|(p, q)| (p - q).abs() <= 1e-6 + 1e-6 * p.abs())
            );

            let (mut g1, mut g2) = (vec![0f32; 1030], vec![0f32; 1030]);
            (scalar.glu)(&x, &g, &mut g1);
            (kernels.glu)(&x, &g, &mut g2);
            assert!(g1.iter().zip(&g2).all(|(p, q)| (p - q).abs() < 1e-6));
        }
    }

    #[test]
    fn sigmoid_is_accurate_across_its_range() {
        let x: Vec<f32> = (-1000..=1000).map(|i| i as f32 * 0.0875).collect();
        let want: Vec<f64> = x.iter().map(|&v| 1.0 / (1.0 + (-v as f64).exp())).collect();
        let ones = vec![1.0f32; x.len()];
        for kernels in all_kernels() {
            let mut sig = vec![0f32; x.len()];
            (kernels.glu)(&ones, &x, &mut sig);
            for (s, w) in sig.iter().zip(&want) {
                assert!(
                    (*s as f64 - w).abs() <= 1e-9 + 5e-7 * w,
                    "{:?}: {s} vs {w}",
                    kernels.isa()
                );
            }
        }
    }

    #[test]
    fn depthwise_conv_matches_a_direct_sum() {
        let (frames, channels, kernel) = (11, 24, 9);
        let input = random(frames * channels, 12);
        let w = random(kernel * channels, 13);
        let bias = random(channels, 14);
        for kernels in all_kernels() {
            for t in 0..frames {
                let mut out = vec![0f32; channels];
                (kernels.dwconv1d_frame)(&input, channels, frames, t, &w, kernel, &bias, &mut out);
                for c in 0..channels {
                    let mut s = bias[c] as f64;
                    for j in 0..kernel {
                        let src = t as isize + j as isize - (kernel / 2) as isize;
                        if (0..frames as isize).contains(&src) {
                            s += w[j * channels + c] as f64
                                * input[src as usize * channels + c] as f64;
                        }
                    }
                    assert!((out[c] as f64 - s).abs() < 1e-5);
                }
            }
        }
    }
}
