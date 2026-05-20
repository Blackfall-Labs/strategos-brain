mod column;
mod brain;
mod builder;
mod tokenizer;

use brain::SpectralBrain;
use builder::{BrainTopology, build_columns, build_token_map};
use tokenizer::Tokenizer;
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    topology: String,
    #[arg(long)]
    tokenizer: String,
    #[arg(long)]
    corpus: String,
    #[arg(long, default_value = "5000")]
    n_tokens: usize,
    #[arg(long, default_value = "8")]
    ticks_per_token: usize,
}

fn main() {
    let args = Args::parse();

    eprintln!("[spectral-brain] loading topology: {}", args.topology);
    let topo: BrainTopology = {
        let s = std::fs::read_to_string(&args.topology)
            .unwrap_or_else(|e| panic!("cannot read topology: {e}"));
        serde_json::from_str(&s).unwrap_or_else(|e| panic!("bad topology JSON: {e}"))
    };

    eprintln!("[spectral-brain] tier0 clusters: {}", topo.tier0_clusters.len());
    eprintln!("[spectral-brain] tier1 sheets:   {}", topo.tier1_sheets.len());
    eprintln!("[spectral-brain] tier2 sheets:   {}", topo.tier2_sheets.len());
    eprintln!("[spectral-brain] band23 tokens:  {}", topo.band23_tokens.len());

    eprintln!("[spectral-brain] loading tokenizer: {}", args.tokenizer);
    let tok = Tokenizer::load(&args.tokenizer);
    eprintln!("[spectral-brain] vocab size: {}", tok.vocab_size());

    let columns   = build_columns(&topo);
    let token_map = build_token_map(&topo);
    let always_on_baseline = columns.iter().filter(|c| c.always_on).count();
    eprintln!("[spectral-brain] columns: {}  always-on: {}  token_map entries: {}",
        columns.len(), always_on_baseline, token_map.len());

    eprintln!("[spectral-brain] tokenizing corpus: {}", args.corpus);
    let text = std::fs::read_to_string(&args.corpus)
        .unwrap_or_else(|e| panic!("cannot read corpus: {e}"));
    let token_ids = tok.encode(&text);
    let limit = if args.n_tokens == 0 { token_ids.len() } else { args.n_tokens.min(token_ids.len()) };
    eprintln!("[spectral-brain] corpus: {} tokens, running first {}", token_ids.len(), limit);

    let mut brain = SpectralBrain::new(columns);
    let mut hits  = 0usize;
    let mut misses = 0usize;
    let mut total_cascade   = [0u64; 3];
    let mut total_always_on = [0u64; 3];

    for (i, &tid) in token_ids[..limit].iter().enumerate() {
        if let Some(&(band_id, magnitude)) = token_map.get(&tid) {
            brain.inject(band_id, magnitude, 0);
            hits += 1;
        } else {
            misses += 1;
        }

        let mut tok_cascade   = [0usize; 3];
        let mut tok_always_on = [0usize; 3];

        for _ in 0..args.ticks_per_token {
            brain.tick();
            let s = brain.last_stats;
            for t in 0..3 {
                tok_cascade[t]   += s.cascade[t];
                tok_always_on[t] += s.always_on[t];
            }
        }

        for t in 0..3 {
            total_cascade[t]   += tok_cascade[t] as u64;
            total_always_on[t] += tok_always_on[t] as u64;
        }

        if (i + 1) % 1000 == 0 {
            eprintln!(
                "[spectral-brain] token {:5}  hits={} miss={}  \
                 T2 cascade={} ao={}  T1={}  T0={}",
                i + 1, hits, misses,
                tok_cascade[2], tok_always_on[2],
                tok_cascade[1], tok_cascade[0],
            );
        }
    }

    eprintln!("\n[spectral-brain] run complete");
    eprintln!("  tokens:           {limit}");
    eprintln!("  hits:             {hits} ({:.1}%)", 100.0 * hits as f64 / limit as f64);
    eprintln!("  misses:           {misses}");
    eprintln!("  T0  cascade:      {}", total_cascade[0]);
    eprintln!("  T1  cascade:      {}", total_cascade[1]);
    eprintln!("  T2  cascade:      {}", total_cascade[2]);
    eprintln!("  T2  always-on:    {}", total_always_on[2]);
    eprintln!("  T2  cascade/hit:  {:.1}", total_cascade[2] as f64 / hits.max(1) as f64);
    eprintln!("  ticks total:      {}", brain.tick_count);
}
