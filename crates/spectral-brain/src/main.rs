mod column;
mod brain;
mod builder;

use brain::SpectralBrain;
use builder::{BrainTopology, build_columns, build_token_map};
use clap::Parser;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Parser)]
struct Args {
    /// Brain topology JSON (output of export_band_topology.py)
    #[arg(long)]
    topology: String,

    /// Tokenizer vocab JSON (token_str → token_id map)
    #[arg(long)]
    vocab: String,

    /// Plain text corpus to run through the brain
    #[arg(long)]
    corpus: String,

    /// Number of tokens to process (0 = all)
    #[arg(long, default_value = "1000")]
    n_tokens: usize,

    /// Ticks to run per token before accepting next
    #[arg(long, default_value = "8")]
    ticks_per_token: usize,
}

#[derive(Deserialize)]
struct Vocab {
    #[serde(flatten)]
    map: HashMap<String, u32>,
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

    let columns = build_columns(&topo);
    let token_map = build_token_map(&topo);
    eprintln!("[spectral-brain] columns built:  {}", columns.len());
    eprintln!("[spectral-brain] token triggers: {}", token_map.len());

    eprintln!("[spectral-brain] loading vocab: {}", args.vocab);
    let vocab: HashMap<String, u32> = {
        let s = std::fs::read_to_string(&args.vocab)
            .unwrap_or_else(|e| panic!("cannot read vocab: {e}"));
        let v: serde_json::Value = serde_json::from_str(&s)
            .unwrap_or_else(|e| panic!("bad vocab JSON: {e}"));
        // HuggingFace tokenizer.json: vocab at model.vocab
        // Flat format: {"token": id, ...}
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

    let mut brain = SpectralBrain::new(columns);

    // Simple whitespace tokenizer — good enough for validation against smollm2_triggers.json
    let words: Vec<&str> = text.split_whitespace().collect();
    let limit = if args.n_tokens == 0 { words.len() } else { args.n_tokens.min(words.len()) };

    let mut hits = 0usize;
    let mut misses = 0usize;
    let mut total_fired = [0u64; 3];

    for (i, word) in words[..limit].iter().enumerate() {
        // SmolLM2/BPE vocab uses U+0120 (Ġ) as word-leading-space prefix, not literal ' '
        const GPREF: char = '\u{0120}';
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

        let mut token_fired = [0usize; 3];
        for _ in 0..args.ticks_per_token {
            brain.tick();
            let f = brain.fired_by_tier();
            for t in 0..3 { token_fired[t] += f[t]; }
        }
        for t in 0..3 { total_fired[t] += token_fired[t] as u64; }

        if (i + 1) % 100 == 0 {
            eprintln!(
                "[spectral-brain] token {:4}  hits={} misses={}  fired T0={} T1={} T2={}",
                i + 1, hits, misses, token_fired[0], token_fired[1], token_fired[2]
            );
        }
    }

    eprintln!("\n[spectral-brain] run complete");
    eprintln!("  tokens processed: {}", limit);
    eprintln!("  vocab hits:       {} ({:.1}%)", hits, 100.0 * hits as f64 / limit as f64);
    eprintln!("  vocab misses:     {}", misses);
    eprintln!("  total fired T0:   {}", total_fired[0]);
    eprintln!("  total fired T1:   {}", total_fired[1]);
    eprintln!("  total fired T2:   {}", total_fired[2]);
    eprintln!("  ticks total:      {}", brain.tick_count);
}
