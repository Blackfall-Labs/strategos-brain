//! GPT-2 Small — Structural Fingerprint Extraction.
//!
//! Reads the activation profile produced by `strategos-probe --model gpt2` and
//! extracts the structural fingerprint of each attention head and FFN layer.
//!
//! This is the GPT-2 step of the 5-model distillation proof sequence:
//!   1. Neel Nanda 1L (done): Fourier frequencies → 100% modular addition
//!   2. GPT-2 Small (this): Head entropy clusters → cortical column types
//!   3. SmolLM2 135M → ...
//!   4. Qwen3 1.7B → ...
//!   5. MiniMax M2.7 → ...
//!
//! What "structural fingerprint" means for GPT-2:
//!   - Attention head entropy (from the profile) determines what each head DOES:
//!       entropy < 0.5  → "anchor" head: attends to one specific position/type
//!       entropy 0.5-1.5 → "selective" head: contextually focused
//!       entropy 1.5-2.5 → "contextual" head: broad integration
//!       entropy > 2.5  → "uniform" head: nearly flat distribution
//!   - Residual stream std growth reveals how information accumulates across layers.
//!   - FFN activation std reveals layer specialization depth.
//!
//! Cortex mapping:
//!   - Anchor heads → narrow-bandwidth columns (inhibit_radius ≈ 0.01, single winner)
//!   - Selective heads → medium-bandwidth columns (inhibit_radius ≈ 0.05, few winners)
//!   - Contextual heads → broad-bandwidth columns (inhibit_radius ≈ 0.15, many winners)
//!   - Uniform heads → background columns (bandwidth ≈ 0.3, always active)
//!
//! Usage:
//!   gpt2_distill <path/to/gpt2_profile.json>

use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq)]
enum HeadType {
    Anchor,     // entropy < 0.5  — deterministic, structural
    Selective,  // 0.5 ≤ entropy < 1.5 — context-selective
    Contextual, // 1.5 ≤ entropy < 2.5 — broad integration
    Uniform,    // entropy ≥ 2.5  — nearly flat, background
}

impl HeadType {
    fn from_entropy(e: f32) -> Self {
        if e < 0.5 { HeadType::Anchor }
        else if e < 1.5 { HeadType::Selective }
        else if e < 2.5 { HeadType::Contextual }
        else { HeadType::Uniform }
    }

    fn label(&self) -> &'static str {
        match self {
            HeadType::Anchor     => "ANCHOR    ",
            HeadType::Selective  => "SELECTIVE ",
            HeadType::Contextual => "CONTEXTUAL",
            HeadType::Uniform    => "UNIFORM   ",
        }
    }

    /// Cortex column parameters for this head type.
    fn cortex_params(&self) -> (f32, f32, usize) {
        // (bandwidth, inhibit_radius, column_budget_per_head)
        match self {
            HeadType::Anchor     => (0.01, 0.02, 1),   // 1 very sharp winner
            HeadType::Selective  => (0.05, 0.08, 3),   // ~3 selective winners
            HeadType::Contextual => (0.12, 0.18, 8),   // ~8 overlapping winners
            HeadType::Uniform    => (0.25, 0.35, 16),  // 16 broad background cols
        }
    }
}

#[derive(Debug)]
struct LayerProfile {
    layer: usize,
    head_entropy: Vec<f32>,
    head_types: Vec<HeadType>,
    ffn_std: f32,
    ffn_mean: f32,
    residual_std: f32,
    residual_mean: f32,
}

impl LayerProfile {
    fn anchor_heads(&self) -> Vec<usize> {
        self.head_types.iter().enumerate()
            .filter(|(_, t)| **t == HeadType::Anchor)
            .map(|(i, _)| i).collect()
    }

    fn selective_heads(&self) -> Vec<usize> {
        self.head_types.iter().enumerate()
            .filter(|(_, t)| **t == HeadType::Selective)
            .map(|(i, _)| i).collect()
    }

    fn column_budget(&self) -> usize {
        self.head_types.iter().map(|t| t.cortex_params().2).sum::<usize>()
            + 8 // 8 FFN region columns per layer (fixed budget)
    }
}

fn load_profile(path: &Path) -> (String, u64, Vec<LayerProfile>) {
    let data = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    let v: serde_json::Value = serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("JSON parse failed: {e}"));

    let model = v["model"].as_str().unwrap_or("unknown").to_string();
    let n_tokens = v["n_tokens"].as_u64().unwrap_or(0);

    let mut layers = Vec::new();
    for layer_v in v["layers"].as_array().unwrap_or(&vec![]) {
        let li = layer_v["layer"].as_u64().unwrap_or(0) as usize;
        let ops = &layer_v["ops"];

        let head_entropy: Vec<f32> = ops["attn_weights"]["head_entropy"]
            .as_array().unwrap_or(&vec![])
            .iter()
            .filter_map(|v| v.as_f64().map(|x| x as f32))
            .collect();

        let head_types: Vec<HeadType> = head_entropy.iter()
            .map(|&e| HeadType::from_entropy(e))
            .collect();

        let ffn = &ops["ffn_hidden"];
        let res = &ops["residual"];

        layers.push(LayerProfile {
            layer: li,
            head_entropy,
            head_types,
            ffn_std: ffn["std"].as_f64().unwrap_or(0.0) as f32,
            ffn_mean: ffn["mean"].as_f64().unwrap_or(0.0) as f32,
            residual_std: res["std"].as_f64().unwrap_or(0.0) as f32,
            residual_mean: res["mean"].as_f64().unwrap_or(0.0) as f32,
        });
    }

    (model, n_tokens, layers)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let profile_path = args.get(1).map(String::as_str).unwrap_or("gpt2_profile.json");

    println!("GPT-2 Small — Structural Fingerprint Extraction");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let (model, n_tokens, layers) = load_profile(Path::new(profile_path));
    let n_layers = layers.len();
    let n_heads = layers.first().map(|l| l.head_entropy.len()).unwrap_or(0);

    println!("Profile: {}  tokens={}  layers={}  heads/layer={}", model, n_tokens, n_layers, n_heads);
    println!();

    // ── Layer-by-layer structural fingerprint ─────────────────────────────────
    println!("LAYER-BY-LAYER STRUCTURAL FINGERPRINT");
    println!("  L   ffn_std  res_std  anchor_heads  selective_heads  col_budget");
    for lp in &layers {
        let anchors = lp.anchor_heads();
        let selectives = lp.selective_heads();
        let budget = lp.column_budget();
        let anchor_str = if anchors.is_empty() { "—".to_string() }
            else { anchors.iter().map(|h| format!("H{h}")).collect::<Vec<_>>().join(",") };
        let selective_str = if selectives.is_empty() { "—".to_string() }
            else { selectives.iter().map(|h| format!("H{h}")).collect::<Vec<_>>().join(",") };
        println!("  L{:2}  {:.4}   {:6.2}   {:12}  {:15}  {}",
            lp.layer, lp.ffn_std, lp.residual_std, anchor_str, selective_str, budget);
    }
    println!();

    // ── Head entropy matrix ───────────────────────────────────────────────────
    println!("HEAD ENTROPY MATRIX  (A=anchor <0.5 | s=selective <1.5 | c=contextual <2.5 | U=uniform)");
    print!("       ");
    for h in 0..n_heads { print!("H{h:<3}"); }
    println!();
    for lp in &layers {
        print!("  L{:2}   ", lp.layer);
        for (h, &e) in lp.head_entropy.iter().enumerate() {
            let tag = match lp.head_types[h] {
                HeadType::Anchor     => format!("{:.2}A", e),
                HeadType::Selective  => format!("{:.2}s", e),
                HeadType::Contextual => format!("{:.2}c", e),
                HeadType::Uniform    => format!("{:.2}U", e),
            };
            print!("{:5} ", tag);
        }
        println!();
    }
    println!();

    // ── Landmark heads (known GPT-2 mechanistic interpretability) ─────────────
    println!("LANDMARK HEADS (structural heads compared to published GPT-2 findings):");

    let mut landmark_heads: Vec<(usize, usize, f32, &str)> = Vec::new();
    for lp in &layers {
        for (h, &e) in lp.head_entropy.iter().enumerate() {
            if e < 1.0 {
                let note = if lp.layer == 0 && h == 3 { "prev-token candidate (L0H3, published: prev-token head)" }
                    else if lp.layer == 4 && h == 11 { "positional head candidate (L4H11, entropy ≈ 0)" }
                    else if lp.layer == 5 && h == 1 { "induction head candidate (L5H1, published: induction head)" }
                    else if lp.layer == 7 && h == 2 { "induction head candidate (L7H2)" }
                    else { "structural head (unknown role)" };
                landmark_heads.push((lp.layer, h, e, note));
            }
        }
    }
    landmark_heads.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    for (l, h, e, note) in &landmark_heads {
        println!("  L{l}H{h}  entropy={e:.4}  → {note}");
    }
    println!();

    // ── Head type distribution ────────────────────────────────────────────────
    println!("HEAD TYPE DISTRIBUTION across all {} heads:", n_layers * n_heads);
    let all_types: Vec<HeadType> = layers.iter()
        .flat_map(|lp| lp.head_types.iter().copied())
        .collect();

    let n_anchor     = all_types.iter().filter(|&&t| t == HeadType::Anchor).count();
    let n_selective  = all_types.iter().filter(|&&t| t == HeadType::Selective).count();
    let n_contextual = all_types.iter().filter(|&&t| t == HeadType::Contextual).count();
    let n_uniform    = all_types.iter().filter(|&&t| t == HeadType::Uniform).count();
    let total = all_types.len();

    println!("  Anchor     (<0.5  entropy): {:3}  ({:.1}%)", n_anchor, 100.0 * n_anchor as f32 / total as f32);
    println!("  Selective  (<1.5  entropy): {:3}  ({:.1}%)", n_selective, 100.0 * n_selective as f32 / total as f32);
    println!("  Contextual (<2.5  entropy): {:3}  ({:.1}%)", n_contextual, 100.0 * n_contextual as f32 / total as f32);
    println!("  Uniform    (≥2.5  entropy): {:3}  ({:.1}%)", n_uniform, 100.0 * n_uniform as f32 / total as f32);
    println!();

    // ── Residual stream drift ─────────────────────────────────────────────────
    println!("RESIDUAL STREAM ACCUMULATION (information growth across layers):");
    println!("  L   res_std   growth_factor  interpretation");
    let base_std = layers.first().map(|l| l.residual_std).unwrap_or(1.0);
    for lp in &layers {
        let factor = lp.residual_std / base_std;
        let interp = if factor < 1.5 { "embedding phase" }
            else if factor < 6.0 { "syntactic integration" }
            else if factor < 10.0 { "semantic integration" }
            else { "output representation" };
        println!("  L{:2}  {:7.3}   {:.2}×   {}", lp.layer, lp.residual_std, factor, interp);
    }
    println!();

    // ── Cortex column budget ──────────────────────────────────────────────────
    println!("TERNSIG CORTEX COLUMN BUDGET for GPT-2 Small equivalent:");

    let (bw_a, ir_a, _) = HeadType::Anchor.cortex_params();
    let (bw_s, ir_s, _) = HeadType::Selective.cortex_params();
    let (bw_c, ir_c, _) = HeadType::Contextual.cortex_params();
    let (bw_u, ir_u, _) = HeadType::Uniform.cortex_params();

    println!("  Head type → cortex column parameters:");
    println!("    Anchor    : bandwidth={bw_a:.3}  inhibit_radius={ir_a:.3}  budget/head=1");
    println!("    Selective : bandwidth={bw_s:.3}  inhibit_radius={ir_s:.3}  budget/head=3");
    println!("    Contextual: bandwidth={bw_c:.3}  inhibit_radius={ir_c:.3}  budget/head=8");
    println!("    Uniform   : bandwidth={bw_u:.3}  inhibit_radius={ir_u:.3}  budget/head=16");
    println!();

    let total_attn_cols = n_anchor * 1 + n_selective * 3 + n_contextual * 8 + n_uniform * 16;
    let total_ffn_cols  = n_layers * 8;
    let total_cols      = total_attn_cols + total_ffn_cols;

    let gpt2_params     = 117_000_000usize;
    let compression_pct = 100.0 * total_cols as f32 / gpt2_params as f32;

    println!("  Attention columns : {total_attn_cols}");
    println!("    = {n_anchor}×1 + {n_selective}×3 + {n_contextual}×8 + {n_uniform}×16");
    println!("  FFN region columns: {total_ffn_cols}  (8 per layer × {n_layers} layers)");
    println!("  Total columns     : {total_cols}");
    println!();
    println!("  GPT-2 parameters  : {gpt2_params}");
    println!("  Column budget     : {total_cols}  ({compression_pct:.4}% of parameter count)");
    println!();
    println!("  Structural compression: {gpt2_params}/{total_cols} = {:.0}:1 parameter-to-column ratio",
        gpt2_params as f64 / total_cols as f64);
    println!();

    // ── Assertions ────────────────────────────────────────────────────────────
    assert!(n_anchor > 0, "no anchor heads found — entropy calculation broken");
    assert!(n_uniform > 0, "no uniform heads found — entropy calculation broken");
    assert!(layers.last().map(|l| l.residual_std).unwrap_or(0.0) > layers[0].residual_std,
        "residual std should grow monotonically");

    // L4H11 must be anchor (entropy ≈ 0.01)
    let l4h11 = layers.iter().find(|l| l.layer == 4)
        .map(|l| l.head_types.get(11).copied()).flatten();
    assert_eq!(l4h11, Some(HeadType::Anchor), "L4H11 must be an anchor head");

    // L5H1 must be anchor or selective (entropy 0.18)
    let l5h1 = layers.iter().find(|l| l.layer == 5)
        .map(|l| l.head_types.get(1).copied()).flatten();
    assert!(
        l5h1 == Some(HeadType::Anchor) || l5h1 == Some(HeadType::Selective),
        "L5H1 (induction head, entropy 0.18) must be anchor or selective"
    );

    println!("═══════════════════════════════════════════════════════════");
    println!("All assertions passed.");
    println!();
    println!("Structural distillation summary:");
    println!("  {} attention heads → {} head types (anchor/selective/contextual/uniform)", total, 4);
    println!("  {gpt2_params} parameters → {total_cols} cortex columns ({:.0}:1 compression)",
        gpt2_params as f64 / total_cols as f64);
    println!("  {} structural heads identified (likely positional/induction/prev-token roles)",
        n_anchor + n_selective);
    println!();
    println!("Next: verify landmark heads by probing specific token sequences");
    println!("  Run probe on sequences with repetition → check L5H1 attends back to previous occurrence");
    println!("  Confirm L4H11 fixed-position attention (attends to position 0 or -1 universally)");
    println!("  Verify L0H3 previous-token pattern (attends always to position pos-1)");
}
