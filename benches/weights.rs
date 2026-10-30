//! Time and weight traffic of the matrix product per weight precision, on the shapes where
//! reading the weights dominates: Parakeet's decoder steps (one row) and short streaming
//! windows (8 rows), plus a long pass (512 rows) for comparison.
//!
//! ```text
//! cargo bench --bench weights [-- --threads 1,6] [--precisions f32,f16,bf16,int8,q8_0]
//! ```
//!
//! Each line gives the microseconds per call (best of 5 runs of many calls) and the weight
//! bytes read per second. The weights of all the shapes together are larger than the caches, as
//! in a decoder step that runs the LSTM, then the joint, then the LSTM again.

use std::time::Instant;

use notate::Precision;
use notate::kernels::{Epilogue, Kernels, PackedWeights};
use notate::quant::Rows;
use notate::threads::Pool;

fn main() {
    let mut threads = vec![1, 6];
    let mut precisions = Precision::ALL.to_vec();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--threads" => {
                threads = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|t| t.parse().unwrap())
                    .collect()
            }
            "--precisions" => {
                precisions = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|p| Precision::from_name(p).expect("a precision"))
                    .collect()
            }
            _ => {}
        }
    }
    let kernels = Kernels::detect();
    // (name, rows, outputs, inputs). The decoder's LSTM layer joins its two matrices.
    let shapes = [
        ("lstm layer", 1, 2560, 1280),
        ("joint", 1, 8198, 640),
        ("pred_proj", 1, 640, 640),
        ("ff_in, 8 rows", 8, 4096, 1024),
        ("ff_in, 512 rows", 512, 4096, 1024),
    ];
    println!("{} kernels", kernels.isa().name());
    println!(
        "{:<8} {:<16} {:>7} {:>10} {:>12} {:>10}",
        "weights", "shape", "threads", "MB", "us per call", "GB/s"
    );
    for &precision in &precisions {
        // Several copies of each matrix, so consecutive calls read different memory, as the
        // decoder's steps do. 64 MB or more in all, beyond the last-level cache.
        let mats: Vec<(&str, usize, Vec<PackedWeights>)> = shapes
            .iter()
            .map(|&(name, m, n, k)| {
                let w: Vec<f32> = (0..n * k)
                    .map(|i| ((i * 104729) % 1000) as f32 / 1000.0 - 0.5)
                    .collect();
                let one = PackedWeights::encode(&Rows::F32(&w), n, k, precision);
                let copies = (64 << 20) / one.bytes() + 1;
                (
                    name,
                    m,
                    (0..copies)
                        .map(|_| PackedWeights::encode(&Rows::F32(&w), n, k, precision))
                        .collect(),
                )
            })
            .collect();
        for &t in &threads {
            let pool = Pool::new(t);
            for (name, m, copies) in &mats {
                let (n, k) = (copies[0].n(), copies[0].k());
                let a: Vec<f32> = (0..m * k)
                    .map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5)
                    .collect();
                let mut c = vec![0f32; m * n];
                let calls = (copies.len() * 4).max((2e8 / (2 * m * n * k) as f64) as usize);
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let start = Instant::now();
                    for i in 0..calls {
                        let w = &copies[i % copies.len()];
                        kernels.linear(&pool, &a, *m, w, &mut c, &Epilogue::PLAIN);
                    }
                    best = best.min(start.elapsed().as_secs_f64() / calls as f64);
                }
                let bytes = copies[0].bytes() as f64;
                println!(
                    "{:<8} {:<16} {:>7} {:>10.2} {:>12.1} {:>10.1}",
                    precision.name(),
                    name,
                    t,
                    bytes / 1e6,
                    best * 1e6,
                    bytes / best / 1e9
                );
            }
        }
    }
}
