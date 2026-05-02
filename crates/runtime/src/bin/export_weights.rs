//! Export weights — pure Rust replacement for tools/export_weights.py.
//!
//! Reads safetensors files, converts BF16/F16 → f32, writes
//! manifest.json + .bin files in our format.
//!
//! Usage:
//!   export_weights --input <safetensors_dir> --output <output_dir>
//!
//! Example:
//!   export_weights --input ~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/<hash> --output models/qwen3

use strategos_runtime::safetensors::export_to_bins;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let input = args.iter().position(|a| a == "--input")
        .and_then(|i| args.get(i + 1))
        .expect("usage: export_weights --input <safetensors_dir> --output <output_dir>");

    let output = args.iter().position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .expect("usage: export_weights --input <safetensors_dir> --output <output_dir>");

    eprintln!("=== Safetensors → Strategos Binary Export ===");
    eprintln!("input:  {}", input);
    eprintln!("output: {}\n", output);

    let start = std::time::Instant::now();

    match export_to_bins(Path::new(input), Path::new(output)) {
        Ok(count) => {
            let elapsed = start.elapsed();
            eprintln!("\n{} tensors exported in {:.1}s", count, elapsed.as_secs_f64());
        }
        Err(e) => {
            eprintln!("ERROR: {}", e);
            std::process::exit(1);
        }
    }
}
