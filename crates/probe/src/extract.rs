use clap::Parser;
use std::collections::HashMap;
use std::path::Path;

mod stats;
mod capture;
mod corpus;
mod triggers;
mod smollm2_probe;
mod qwen3_probe;
// gpt2/grokking probes don't have extract support yet
mod gpt2_probe;
mod grokking_probe;

use triggers::TriggerStore;

#[derive(Parser)]
#[command(name = "extract", about = "Invariant extractor — token trigger analysis per FFN neuron")]
struct Args {
    #[arg(long, default_value = "smollm2")]
    model: String,

    #[arg(long, help = "Path to manifest.json directory")]
    weights: String,

    #[arg(long, help = "Path to tokenizer.json")]
    tokenizer: String,

    #[arg(long, help = "Path to plain-text corpus file")]
    corpus: String,

    #[arg(long, help = "Path to calibrated activation profile JSON")]
    profile: String,

    #[arg(long, help = "Path to write trigger profile JSON")]
    output: String,

    #[arg(long, default_value = "20000")]
    n_tokens: usize,

    #[arg(long, default_value = "50", help = "Top-K triggering tokens to record per neuron")]
    top_k: usize,

    #[arg(long, default_value = "0.01", help = "Min fire rate to include a neuron")]
    min_rate: f32,

    #[arg(long, default_value = "0.95", help = "Max fire rate to include a neuron")]
    max_rate: f32,
}

fn main() {
    let args = Args::parse();

    eprintln!("[extract] model={} corpus={} n_tokens={}", args.model, args.corpus, args.n_tokens);

    let profile_str = std::fs::read_to_string(&args.profile)
        .unwrap_or_else(|e| { eprintln!("[extract] profile read failed: {e}"); std::process::exit(1); });
    let profile: serde_json::Value = serde_json::from_str(&profile_str)
        .unwrap_or_else(|e| { eprintln!("[extract] profile parse failed: {e}"); std::process::exit(1); });

    match args.model.as_str() {
        "smollm2" => run_smollm2(&args, &profile),
        "qwen3"   => run_qwen3(&args, &profile),
        other => {
            eprintln!("[extract] unknown model: {other}. Supported: smollm2, qwen3");
            std::process::exit(1);
        }
    }
}

fn load_thresholds(weights_dir: &str, n_layers: usize) -> Vec<f32> {
    let path = Path::new(weights_dir).join("ffn_thresholds.json");
    if let Ok(data) = std::fs::read_to_string(&path) {
        #[derive(serde::Deserialize)]
        struct ThresholdFile { thresholds: Vec<f32> }
        if let Ok(tf) = serde_json::from_str::<ThresholdFile>(&data) {
            return tf.thresholds;
        }
    }
    vec![0.1f32; n_layers]
}

/// Build a token_id → display string vocab map from the tokenizer JSON.
/// We only need the first-level "vocab" field (token → id), inverted.
fn load_vocab(tokenizer_path: &str) -> HashMap<u32, String> {
    let data = match std::fs::read_to_string(tokenizer_path) {
        Ok(d) => d,
        Err(_) => return HashMap::new(),
    };
    let json: serde_json::Value = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(_) => return HashMap::new(),
    };

    let mut vocab = HashMap::new();
    // HuggingFace tokenizer.json: model.vocab is { "token": id }
    if let Some(model_vocab) = json["model"]["vocab"].as_object() {
        for (token_str, id_val) in model_vocab {
            if let Some(id) = id_val.as_u64() {
                vocab.insert(id as u32, token_str.clone());
            }
        }
    }
    // Also try top-level "vocab" (some tokenizer formats)
    if vocab.is_empty() {
        if let Some(top_vocab) = json["vocab"].as_object() {
            for (token_str, id_val) in top_vocab {
                if let Some(id) = id_val.as_u64() {
                    vocab.insert(id as u32, token_str.clone());
                }
            }
        }
    }
    eprintln!("[extract] vocab loaded: {} tokens", vocab.len());
    vocab
}

fn run_smollm2(args: &Args, profile: &serde_json::Value) {
    let model = smollm2_probe::SmolLM2Probe::load(Path::new(&args.weights))
        .unwrap_or_else(|e| { eprintln!("[extract] load failed: {e}"); std::process::exit(1); });

    let tokens = corpus::load_tokens(&args.corpus, &args.tokenizer, args.n_tokens)
        .unwrap_or_else(|e| { eprintln!("[extract] corpus failed: {e}"); std::process::exit(1); });

    let thresholds = load_thresholds(&args.weights, model.n_layer);
    let mut triggers = TriggerStore::from_profile(profile, thresholds.clone(), args.min_rate, args.max_rate);
    let mut captures = capture::CaptureStore::for_smollm2(model.n_layer, model.n_head, model.ffn_dim, &thresholds);

    eprintln!("[extract] running {} tokens through SmolLM2-135M...", tokens.len());
    model.run_corpus_extract(&tokens, &mut captures, &mut triggers);

    let vocab = load_vocab(&args.tokenizer);
    let trigger_profile = triggers.finalize("smollm2-135m", &args.corpus, Some(&vocab), args.top_k);

    let json = serde_json::to_string_pretty(&trigger_profile)
        .unwrap_or_else(|e| { eprintln!("[extract] serialize failed: {e}"); std::process::exit(1); });
    std::fs::write(&args.output, &json)
        .unwrap_or_else(|e| { eprintln!("[extract] write failed: {e}"); std::process::exit(1); });

    eprintln!("[extract] trigger profile written to {}", args.output);
    eprintln!("[extract] {} tokens, {} layers", trigger_profile.n_tokens, trigger_profile.layers.len());
}

fn run_qwen3(args: &Args, profile: &serde_json::Value) {
    let model = qwen3_probe::Qwen3Probe::load(Path::new(&args.weights))
        .unwrap_or_else(|e| { eprintln!("[extract] load failed: {e}"); std::process::exit(1); });

    let tokens = corpus::load_tokens(&args.corpus, &args.tokenizer, args.n_tokens)
        .unwrap_or_else(|e| { eprintln!("[extract] corpus failed: {e}"); std::process::exit(1); });

    let thresholds = load_thresholds(&args.weights, model.n_layer);
    let mut triggers = TriggerStore::from_profile(profile, thresholds.clone(), args.min_rate, args.max_rate);
    let mut captures = capture::CaptureStore::for_qwen3(model.n_layer, model.n_head, model.ffn_dim, &thresholds);

    eprintln!("[extract] running {} tokens through Qwen3-1.7B...", tokens.len());
    model.run_corpus_extract(&tokens, &mut captures, &mut triggers);

    let vocab = load_vocab(&args.tokenizer);
    let trigger_profile = triggers.finalize("qwen3-1.7b", &args.corpus, Some(&vocab), args.top_k);

    let json = serde_json::to_string_pretty(&trigger_profile)
        .unwrap_or_else(|e| { eprintln!("[extract] serialize failed: {e}"); std::process::exit(1); });
    std::fs::write(&args.output, &json)
        .unwrap_or_else(|e| { eprintln!("[extract] write failed: {e}"); std::process::exit(1); });

    eprintln!("[extract] trigger profile written to {}", args.output);
    eprintln!("[extract] {} tokens, {} layers", trigger_profile.n_tokens, trigger_profile.layers.len());
}
