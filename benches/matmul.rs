//! Throughput of the matrix product on the shapes of Parakeet's encoder.
//!
//! ```text
//! cargo bench --bench matmul [-- --threads 1,6] [--isa scalar|avx2|avx512|neon]
//! ```

use std::time::Instant;

use notate::kernels::{Epilogue, Isa, Kernels, PackedWeights};
use notate::threads::Pool;

fn main() {
    let mut threads = vec![1, 6];
    let mut isa = Isa::detect();
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
            "--isa" => {
                isa = args
                    .next()
                    .as_deref()
                    .and_then(Isa::from_name)
                    .expect("--isa takes scalar, avx2, avx512 or neon")
            }
            _ => {}
        }
    }
    let kernels = Kernels::new(isa).expect("this CPU runs the kernels");
    println!("{} kernels, GFLOPS (best of 5)", isa.name());
    print!("{:>5} {:>5} {:>5}", "m", "k", "n");
    for t in &threads {
        print!(" {:>8}", format!("{t} thr"));
    }
    println!();
    for &(k, n) in &[(1024, 4096), (4096, 1024), (1024, 1024)] {
        for m in [8, 32, 128, 512] {
            let a: Vec<f32> = (0..m * k)
                .map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5)
                .collect();
            let w: Vec<f32> = (0..n * k)
                .map(|i| ((i * 104729) % 1000) as f32 / 1000.0 - 0.5)
                .collect();
            let w = PackedWeights::from_rows(&w, n, k);
            let mut c = vec![0f32; m * n];
            print!("{m:>5} {k:>5} {n:>5}");
            for &t in &threads {
                let pool = Pool::new(t);
                let flops = 2.0 * (m * k * n) as f64;
                let reps = ((2e9 / flops) as usize).clamp(1, 1000);
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let start = Instant::now();
                    for _ in 0..reps {
                        kernels.linear(&pool, &a, m, &w, &mut c, &Epilogue::PLAIN);
                    }
                    best = best.min(start.elapsed().as_secs_f64() / reps as f64);
                }
                print!(" {:>8.1}", flops / best / 1e9);
            }
            println!();
        }
    }
}
