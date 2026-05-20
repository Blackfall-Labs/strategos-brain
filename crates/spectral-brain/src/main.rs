mod column;
mod brain;
mod builder;

use brain::SpectralBrain;
use builder::{BrainTopology, build_columns, build_token_map};
use clap::Parser;
use std::collections::HashMap;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    topology: String,
    #[arg(long)]
    vocab: String,
    #[arg(long)]
    corpus: String,
    #[arg(long, default_value = "1000")]
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

    let columns    = build_columns(&topo);
    let token_map  = build_token_map(&topo);
    eprintln!("[spectral-brain] columns built:  {}", columns.len());
    eprintln!("[spectral-brain] token triggers: {} (band1={} band23={})",
        token_map.len(), topo.tier0_clusters.len(), topo.band23_tokens.len());

    eprintln!("[spectral-brain] loading vocab: {}", args.vocab);
    let vocab: HashMap<String, u32> = {
        let s = std::fs::read_to_string(&args.vocab)
            .unwrap_or_else(|e| panic!("cannot read vocab: {e}"));
        let v: serde_json::Value = serde_json::from_str(&s)
            .unwrap_or_else(|e| panic!("bad vocab JSON: {e}"));
        if let Some(vocab_map) = v.get("model").and_then(|m| m.get("vocab")) {
            vocab_map.as_object().unwrap()
                .iter()
                .filter_map(|(k, v)| v.as_u64().map(|id| (k.clone(), id as u32)))
                .collect()
        } else {
            v.as_object().unwrap()
                .iter()
                .filter_map(|(k, v)| v.as_u64().map(|id| (k.clone(), id as u32)))
                .collect()
        }
    };

    eprintln!("[spectral-brain] loading corpus: {}", args.corpus);
    let text = std::fs::read_to_string(&args.corpus)
        .unwrap_or_else(|e| panic!("cannot read corpus: {e}"));

    let always_on_baseline: usize = columns.iter().filter(|c| c.always_on).count()
        * args.ticks_per_token;
    eprintln!("[spectral-brain] always-on baseline per token: {} fires", always_on_baseline);

    let mut brain = SpectralBrain::new(columns);

    let words: Vec<&str> = text.split_whitespace().collect();
    let limit = if args.n_tokens == 0 { words.len() } else { args.n_tokens.min(words.len()) };

    let mut hits  = 0usize;
    let mut misses = 0usize;
    let mut total_cascade = [0u64; 3];
    let mut total_always_on = [0u64; 3];

    // SmolLM2/BPE vocab uses U+0120 (Ġ) as word-leading-space prefix
    const GPREF: char = '\u{0120}';

    for (i, word) in words[..limit].iter().enumerate() {
        let tok_id = vocab.get(&format!("{GPREF}{word}"))
            .or_else(|| vocab.get(*word))
            .copied();

        if let Some(tid) = tok_id {
            if let Some(&(band_id, magnitude)) = token_map.get(&tid) {
                brain.inject(band_id, magnitude, 0);
                hits += 1;
            } else {
                misses += 1;
            }
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

        if (i + 1) % 100 == 0 {
            eprintln!(
                "[spectral-brain] token {:4}  hits={} miss={}  \
                 T2 cascade={} always-on={}  T1 cascade={}  T0 cascade={}",
                i + 1, hits, misses,
                tok_cascade[2], tok_always_on[2],
                tok_cascade[1], tok_cascade[0],
            );
        }
    }

    let tokens_with_hit = hits;
    eprintln!("\n[spectral-brain] run complete");
    eprintln!("  tokens processed: {limit}");
    eprintln!("  vocab hits:       {hits} ({:.1}%)", 100.0 * hits as f64 / limit as f64);
    eprintln!("  vocab misses:     {misses}");
    eprintln!("  T0  cascade total: {}", total_cascade[0]);
    eprintln!("  T1  cascade total: {}", total_cascade[1]);
    eprintln!("  T2  cascade total: {}", total_cascade[2]);
    eprintln!("  T2  always-on tot: {}", total_always_on[2]);
    eprintln!("  T2  cascade/hit:   {:.1}", total_cascade[2] as f64 / tokens_with_hit.max(1) as f64);
    eprintln!("  ticks total:       {}", brain.tick_count);
}
