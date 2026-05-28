/// lang-cortex: build a spectral-brain topology from the Blackfall Lexicon.
///
/// Phase 1 of the Cortex Architecture plan:
/// - Tier 0 clusters derived from lexicon word entries, not model neuron probes.
/// - Band assignment: semantic generality rank (most senses = band 0).
/// - Column density: proportional to sense_count (richer words = more columns).
/// - Tier 1/2 sheets: copied from an existing probe-derived topology (unchanged).
///
/// Usage:
///   lang-cortex --word-stats lexicon_word_stats.json \
///               --tokenizer tokenizer.json \
///               --base-topology brain_topology.json \
///               --output lang_topology.json

use std::collections::HashMap;
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Parser)]
struct Args {
    /// JSON dump from `bfl dump`: [{word, word_id, sense_count}, ...]
    #[arg(long)]
    word_stats: String,
    /// SmolLM2 tokenizer.json (HuggingFace BPE format)
    #[arg(long)]
    tokenizer: String,
    /// Existing brain_topology.json — tier1/tier2 sheets are copied from it
    #[arg(long)]
    base_topology: String,
    /// Output topology JSON path
    #[arg(long)]
    output: String,
    /// Minimum sense_count to include a word in tier 0 (default: 1)
    #[arg(long, default_value = "1")]
    min_senses: u32,
    /// Maximum columns per cluster (default: 16)
    #[arg(long, default_value = "16")]
    max_cols: usize,
}

// ── Input formats ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct WordStat {
    word: String,
    #[allow(dead_code)]
    word_id: u32,
    sense_count: u32,
}

#[derive(Deserialize)]
struct TokenizerJson {
    model: BpeModel,
}

#[derive(Deserialize)]
struct BpeModel {
    vocab: HashMap<String, u32>,
}

// ── Output format (matches brain_topology.json schema) ───────────────────────

#[derive(Serialize, Deserialize)]
struct BrainTopology {
    source: String,
    n_cols_total: usize,
    tier0_clusters: Vec<Tier0Cluster>,
    tier1_sheets: Vec<serde_json::Value>,
    tier2_sheets: Vec<serde_json::Value>,
    #[serde(default)]
    band23_tokens: Vec<serde_json::Value>,
}

#[derive(Serialize, Deserialize)]
struct Tier0Cluster {
    cluster_id: usize,
    trigger_token: String,
    token_id: u32,
    band_id: u32,
    members: Vec<Tier0Neuron>,
}

#[derive(Serialize, Deserialize)]
struct Tier0Neuron {
    neuron_idx: usize,
    trigger_rate: f32,
}

// ── Constants (Ġ prefix: U+0120, GPT-2/SmolLM2 BPE word boundary) ───────────
const GPREF: char = '\u{0120}';

fn main() {
    let args = Args::parse();

    // ── Load word stats ───────────────────────────────────────────────────────
    eprintln!("[lang-cortex] loading word stats: {}", args.word_stats);
    let stats_raw = std::fs::read_to_string(&args.word_stats)
        .unwrap_or_else(|e| panic!("cannot read word_stats: {e}"));
    let word_stats: Vec<WordStat> = serde_json::from_str(&stats_raw)
        .unwrap_or_else(|e| panic!("bad word_stats JSON: {e}"));
    eprintln!("[lang-cortex] {} words loaded", word_stats.len());

    // ── Build word → sense_count map (lowercase, deduped by max sense_count) ─
    let mut word_to_senses: HashMap<String, u32> = HashMap::with_capacity(word_stats.len());
    for ws in &word_stats {
        if ws.sense_count < args.min_senses { continue; }
        let key = ws.word.to_lowercase();
        let entry = word_to_senses.entry(key).or_insert(0);
        if ws.sense_count > *entry {
            *entry = ws.sense_count;
        }
    }
    eprintln!("[lang-cortex] {} unique words with >= {} senses", word_to_senses.len(), args.min_senses);

    // ── Assign band_id by semantic generality rank ────────────────────────────
    // Sort words by sense_count descending; rank 0 = most senses = most general = band 0.
    let mut ranked: Vec<(&str, u32)> = word_to_senses.iter()
        .map(|(w, &sc)| (w.as_str(), sc))
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));

    let max_sense_count = ranked.first().map(|r| r.1).unwrap_or(1) as f32;
    let mut word_to_band: HashMap<String, (u32, u32)> = HashMap::with_capacity(ranked.len());
    for (rank, (word, sense_count)) in ranked.iter().enumerate() {
        word_to_band.insert(word.to_string(), (rank as u32, *sense_count));
    }
    eprintln!("[lang-cortex] band range: 0..{}", ranked.len());

    // ── Load tokenizer vocab ──────────────────────────────────────────────────
    eprintln!("[lang-cortex] loading tokenizer: {}", args.tokenizer);
    let tok_raw = std::fs::read_to_string(&args.tokenizer)
        .unwrap_or_else(|e| panic!("cannot read tokenizer: {e}"));
    let tok: TokenizerJson = serde_json::from_str(&tok_raw)
        .unwrap_or_else(|e| panic!("bad tokenizer JSON: {e}"));
    eprintln!("[lang-cortex] vocab size: {}", tok.model.vocab.len());

    // ── Load base topology (for tier1/tier2 sheets) ───────────────────────────
    eprintln!("[lang-cortex] loading base topology: {}", args.base_topology);
    let base_raw = std::fs::read_to_string(&args.base_topology)
        .unwrap_or_else(|e| panic!("cannot read base_topology: {e}"));
    let base: BrainTopology = serde_json::from_str(&base_raw)
        .unwrap_or_else(|e| panic!("bad base topology JSON: {e}"));
    eprintln!("[lang-cortex] base tier1: {}  tier2: {}", base.tier1_sheets.len(), base.tier2_sheets.len());

    // ── Map BPE tokens → lexicon words → clusters ─────────────────────────────
    // For each vocab token: strip Ġ prefix, lowercase → look up in word_to_band.
    // Whole-word tokens (with or without Ġ prefix) that resolve to a lexicon word
    // become tier 0 clusters.
    let mut clusters: Vec<Tier0Cluster> = Vec::new();
    let mut cluster_id = 0usize;
    let mut hits = 0usize;
    let mut misses = 0usize;

    // Sort vocab by token_id for deterministic output
    let mut vocab_sorted: Vec<(&str, u32)> = tok.model.vocab.iter()
        .map(|(k, &v)| (k.as_str(), v))
        .collect();
    vocab_sorted.sort_by_key(|&(_, tid)| tid);

    for (token_str, token_id) in &vocab_sorted {
        // Extract the word: strip Ġ prefix if present, then lowercase.
        let word_raw = if token_str.starts_with(GPREF) {
            &token_str[GPREF.len_utf8()..]
        } else {
            token_str
        };
        // Skip subword tokens (containing Ġ mid-string or other BPE artifacts).
        // A "whole word" token either IS the full token or is Ġ-prefixed.
        // Tokens like "##ing" or those containing Ġ after position 0 are subwords.
        if word_raw.is_empty() { continue; }
        if word_raw.chars().any(|c| c == GPREF) { continue; }

        let word_lower = word_raw.to_lowercase();

        if let Some(&(band_id, sense_count)) = word_to_band.get(&word_lower) {
            // Column density: 1 base + 1 per 4 senses, capped at max_cols.
            let n_cols = (1 + sense_count as usize / 4).min(args.max_cols);
            // trigger_rate: proportional to semantic richness (more senses = fires more reliably).
            let trigger_rate = (sense_count as f32 / max_sense_count).max(0.05).min(0.95);

            let members: Vec<Tier0Neuron> = (0..n_cols)
                .map(|i| Tier0Neuron {
                    neuron_idx: i,
                    trigger_rate,
                })
                .collect();

            clusters.push(Tier0Cluster {
                cluster_id,
                trigger_token: token_str.to_string(),
                token_id: *token_id,
                band_id,
                members,
            });
            cluster_id += 1;
            hits += 1;
        } else {
            misses += 1;
        }
    }

    eprintln!("[lang-cortex] tier0: {} clusters  ({} hits, {} misses)",
        clusters.len(), hits, misses);

    // ── Assemble output topology ──────────────────────────────────────────────
    let n_cols_total = clusters.iter().map(|c| c.members.len()).sum::<usize>()
        + base.tier1_sheets.iter().map(|s| s["n_cols"].as_u64().unwrap_or(0) as usize).sum::<usize>()
        + base.tier2_sheets.iter().map(|s| s["n_cols"].as_u64().unwrap_or(0) as usize).sum::<usize>();

    let out = BrainTopology {
        source: format!("blackfall-lexicon + {} (sense-ranked bands)", args.word_stats),
        n_cols_total,
        tier0_clusters: clusters,
        tier1_sheets: base.tier1_sheets,
        tier2_sheets: base.tier2_sheets,
        band23_tokens: base.band23_tokens,
    };

    eprintln!("[lang-cortex] total columns: {}", n_cols_total);

    let json = serde_json::to_string_pretty(&out)
        .unwrap_or_else(|e| panic!("serialize failed: {e}"));
    std::fs::write(&args.output, json)
        .unwrap_or_else(|e| panic!("write failed: {e}"));

    eprintln!("[lang-cortex] written: {}", args.output);
}
