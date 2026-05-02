//! Qwen3 benchmark — per-component timing.

use strategos_runtime::loader::ModelWeights;
use strategos_runtime::qwen3::Qwen3;
use std::path::Path;

fn main() {
    let weights = ModelWeights::load(Path::new("models/qwen3")).expect("load");
    let mut model = Qwen3::load(&weights).expect("construct");

    // Warm up.
    let _ = model.forward_one(1);
    model.reset();

    // Bench 5 tokens.
    eprintln!("=== Qwen3 1.7B Benchmark (5 tokens) ===\n");
    let mut times = Vec::new();
    for i in 0..5 {
        let start = std::time::Instant::now();
        let logits = model.forward_one((i + 1) as u32);
        let elapsed = start.elapsed();
        let top = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        eprintln!("  token {}: {:.1}ms → top={}", i, elapsed.as_secs_f64() * 1000.0, top);
        times.push(elapsed.as_secs_f64() * 1000.0);
    }

    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(0.0f64, f64::max);
    eprintln!("\n  avg: {:.1}ms, min: {:.1}ms, max: {:.1}ms", avg, min, max);
    eprintln!("  throughput: {:.1} tok/s", 1000.0 / avg);
}
