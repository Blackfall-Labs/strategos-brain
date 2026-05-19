use clap::Parser;
use std::path::Path;

mod stats;
mod capture;
mod corpus;
mod triggers;
mod l0_lookup;
mod gpt2_probe;
mod grokking_probe;
mod smollm2_probe;
mod qwen3_probe;

#[derive(Parser)]
#[command(name = "probe", about = "Activation prober for neuronal distillation")]
struct Args {
    #[arg(long, default_value = "gpt2")]
    model: String,

    #[arg(long, help = "Path to manifest.json directory (for gpt2)")]
    weights: String,

    #[arg(long, help = "Path to tokenizer.json")]
    tokenizer: String,

    #[arg(long, help = "Path to plain-text corpus file")]
    corpus: String,

    #[arg(long, help = "Path to write activation profile JSON")]
    output: String,

    #[arg(long, default_value = "50000")]
    n_tokens: usize,

    #[arg(long, help = "Path to L0 lookup JSON (enables skip-row FFN distillation at layer 0)")]
    lookup: Option<String>,
}

fn main() {
    let args = Args::parse();

    eprintln!("[probe] model={} corpus={} n_tokens={}", args.model, args.corpus, args.n_tokens);

    match args.model.as_str() {
        "gpt2" => run_gpt2(&args),
        "grokking" => run_grokking(&args),
        "smollm2" => run_smollm2(&args),
        "qwen3" => run_qwen3(&args),
        other => {
            eprintln!("[probe] unknown model: {other}. Supported: gpt2, grokking, smollm2, qwen3");
            std::process::exit(1);
        }
    }
}

fn run_grokking(args: &Args) {
    let model = grokking_probe::GrokkingProbe::load(std::path::Path::new(&args.weights))
        .unwrap_or_else(|e| { eprintln!("[probe] load failed: {e}"); std::process::exit(1); });

    let mut captures = capture::CaptureStore::for_grokking();

    eprintln!("[probe] running all {} pairs through grokking model...", 113 * 113);
    model.run_all_pairs(&mut captures);

    let profile = captures.finalize("grokking-1l", "all-pairs");

    let json = serde_json::to_string_pretty(&profile)
        .unwrap_or_else(|e| { eprintln!("[probe] serialize failed: {e}"); std::process::exit(1); });

    std::fs::write(&args.output, &json)
        .unwrap_or_else(|e| { eprintln!("[probe] write failed: {e}"); std::process::exit(1); });

    eprintln!("[probe] profile written to {}", args.output);
}

fn run_smollm2(args: &Args) {
    let mut model = smollm2_probe::SmolLM2Probe::load(Path::new(&args.weights))
        .unwrap_or_else(|e| { eprintln!("[probe] load failed: {e}"); std::process::exit(1); });

    if let Some(lookup_path) = &args.lookup {
        let lut = l0_lookup::L0Lookup::load(Path::new(lookup_path))
            .unwrap_or_else(|e| { eprintln!("[probe] lookup load failed: {e}"); std::process::exit(1); });
        eprintln!("[probe] L0 lookup active: {} neurons, {} trigger tokens", lut.n_entries, lut.n_tokens);
        model.lookup = Some(lut);
    }

    let tokens = corpus::load_tokens(&args.corpus, &args.tokenizer, args.n_tokens)
        .unwrap_or_else(|e| { eprintln!("[probe] corpus failed: {e}"); std::process::exit(1); });

    let thresholds = load_thresholds(&args.weights, model.n_layer);
    let mut captures = capture::CaptureStore::for_smollm2(model.n_layer, model.n_head, model.ffn_dim, &thresholds);

    let lookup_label = if model.lookup.is_some() { " [L0-lookup]" } else { "" };
    eprintln!("[probe] running {} tokens through SmolLM2-135M ({} layers){}...",
        tokens.len(), model.n_layer, lookup_label);

    let t0 = std::time::Instant::now();
    model.run_corpus(&tokens, &mut captures);
    let elapsed = t0.elapsed();
    let tok_per_sec = tokens.len() as f64 / elapsed.as_secs_f64();
    eprintln!("[probe] {:.0} tok/s  ({:.1}s for {} tokens)", tok_per_sec, elapsed.as_secs_f64(), tokens.len());

    let profile = captures.finalize("smollm2-135m", &args.corpus);
    let json = serde_json::to_string_pretty(&profile)
        .unwrap_or_else(|e| { eprintln!("[probe] serialize failed: {e}"); std::process::exit(1); });
    std::fs::write(&args.output, &json)
        .unwrap_or_else(|e| { eprintln!("[probe] write failed: {e}"); std::process::exit(1); });

    eprintln!("[probe] profile written to {}", args.output);
    eprintln!("[probe] {} tokens, {} layers captured", profile.n_tokens, profile.layers.len());
}

fn load_thresholds(weights_dir: &str, n_layers: usize) -> Vec<f32> {
    let path = Path::new(weights_dir).join("ffn_thresholds.json");
    if let Ok(data) = std::fs::read_to_string(&path) {
        #[derive(serde::Deserialize)]
        struct ThresholdFile { thresholds: Vec<f32> }
        if let Ok(tf) = serde_json::from_str::<ThresholdFile>(&data) {
            eprintln!("[probe] loaded {} per-layer FFN thresholds from {}", tf.thresholds.len(), path.display());
            return tf.thresholds;
        }
    }
    eprintln!("[probe] ffn_thresholds.json not found, using default threshold 0.1");
    vec![0.1f32; n_layers]
}

fn run_qwen3(args: &Args) {
    let model = qwen3_probe::Qwen3Probe::load(Path::new(&args.weights))
        .unwrap_or_else(|e| { eprintln!("[probe] load failed: {e}"); std::process::exit(1); });

    let tokens = corpus::load_tokens(&args.corpus, &args.tokenizer, args.n_tokens)
        .unwrap_or_else(|e| { eprintln!("[probe] corpus failed: {e}"); std::process::exit(1); });

    let thresholds = load_thresholds(&args.weights, model.n_layer);
    let mut captures = capture::CaptureStore::for_qwen3(model.n_layer, model.n_head, model.ffn_dim, &thresholds);

    eprintln!("[probe] running {} tokens through Qwen3-1.7B ({} layers)...", tokens.len(), model.n_layer);
    model.run_corpus(&tokens, &mut captures);

    let profile = captures.finalize("qwen3-1.7b", &args.corpus);
    let json = serde_json::to_string_pretty(&profile)
        .unwrap_or_else(|e| { eprintln!("[probe] serialize failed: {e}"); std::process::exit(1); });
    std::fs::write(&args.output, &json)
        .unwrap_or_else(|e| { eprintln!("[probe] write failed: {e}"); std::process::exit(1); });

    eprintln!("[probe] profile written to {}", args.output);
    eprintln!("[probe] {} tokens, {} layers captured", profile.n_tokens, profile.layers.len());
}

fn run_gpt2(args: &Args) {
    let model = gpt2_probe::GPT2Probe::load(Path::new(&args.weights))
        .unwrap_or_else(|e| { eprintln!("[probe] load failed: {e}"); std::process::exit(1); });

    let tokens = corpus::load_tokens(&args.corpus, &args.tokenizer, args.n_tokens)
        .unwrap_or_else(|e| { eprintln!("[probe] corpus failed: {e}"); std::process::exit(1); });

    let mut captures = capture::CaptureStore::for_gpt2(model.n_layer, model.n_head, model.ffn_dim);

    eprintln!("[probe] running {} tokens through GPT-2...", tokens.len());
    model.run_corpus(&tokens, &mut captures);

    let profile = captures.finalize("gpt2-small", &args.corpus);

    let json = serde_json::to_string_pretty(&profile)
        .unwrap_or_else(|e| { eprintln!("[probe] serialize failed: {e}"); std::process::exit(1); });

    std::fs::write(&args.output, &json)
        .unwrap_or_else(|e| { eprintln!("[probe] write failed: {e}"); std::process::exit(1); });

    eprintln!("[probe] profile written to {}", args.output);
    eprintln!("[probe] {} tokens, {} layers captured", profile.n_tokens, profile.layers.len());
}
