use strategos_runtime::loader::ModelWeights;
use strategos_runtime::phi4::Phi4;
use std::path::Path;

fn main() {
    eprintln!("=== Phi-4 Mini 3.8B Load Test ===\n");
    let weights = ModelWeights::load(Path::new("models/phi4")).expect("load weights");
    eprintln!("[test] {} tensors", weights.names().len());

    let mut model = Phi4::load(&weights).expect("construct model");

    let start = std::time::Instant::now();
    let logits = model.forward_one(1);
    let elapsed = start.elapsed();

    eprintln!("[test] forward: {:.1}ms, logits: {}", elapsed.as_secs_f64() * 1000.0, logits.len());

    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (i, (tok, score)) in indexed.iter().take(5).enumerate() {
        eprintln!("  {}: token {} ({:.4})", i, tok, score);
    }
    eprintln!("\n=== Done ===");
}
