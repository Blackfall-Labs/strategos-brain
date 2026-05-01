//! Test: load Qwen3 1.7B weights and run a forward pass.

use strategos_runtime::loader::ModelWeights;
use strategos_runtime::qwen3::Qwen3;
use std::path::Path;

fn main() {
    eprintln!("=== Qwen3 1.7B Load Test ===\n");

    let weights_dir = Path::new("models/qwen3");
    eprintln!("[test] loading weights from {:?}", weights_dir);

    let weights = ModelWeights::load(weights_dir)
        .expect("failed to load weights");

    eprintln!("[test] {} tensors loaded", weights.names().len());

    // Print first few tensor names and shapes.
    let mut names: Vec<&str> = weights.names();
    names.sort();
    for name in names.iter().take(10) {
        let t = weights.get(name).unwrap();
        eprintln!("  {} {:?} ({} elements)", name, t.shape, t.data.len());
    }
    eprintln!("  ... ({} total)", names.len());

    eprintln!("\n[test] constructing Qwen3 model...");
    let mut model = Qwen3::load(&weights).expect("failed to construct model");

    eprintln!("\n[test] running single token forward pass...");
    let start = std::time::Instant::now();
    let logits = model.forward_one(1); // token 1
    let elapsed = start.elapsed();

    eprintln!("[test] forward pass: {:.1}ms", elapsed.as_secs_f64() * 1000.0);
    eprintln!("[test] logits length: {}", logits.len());

    // Find top-5 tokens.
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!("[test] top-5 tokens:");
    for (i, (token, score)) in indexed.iter().take(5).enumerate() {
        eprintln!("  {}: token {} (score {:.4})", i, token, score);
    }

    eprintln!("\n=== Test complete ===");
}
