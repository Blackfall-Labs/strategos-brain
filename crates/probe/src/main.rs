use clap::Parser;
use std::path::Path;

mod stats;
mod capture;
mod corpus;
mod gpt2_probe;
mod grokking_probe;

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
}

fn main() {
    let args = Args::parse();

    eprintln!("[probe] model={} corpus={} n_tokens={}", args.model, args.corpus, args.n_tokens);

    match args.model.as_str() {
        "gpt2" => run_gpt2(&args),
        "grokking" => run_grokking(&args),
        other => {
            eprintln!("[probe] unknown model: {other}. Supported: gpt2, grokking");
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
