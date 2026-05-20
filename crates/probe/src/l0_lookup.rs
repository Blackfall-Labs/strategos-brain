//! Multi-layer activation lookup table.
//!
//! Supports two JSON formats:
//!   - Multi-layer: `{"layers": [{"layer": N, "entries": [...]}]}` (new)
//!   - Single-layer: `{"entries": [...]}` — legacy L0-only format
//!
//! For Band 1 neurons whose activation can be predicted from the token ID alone
//! (after corpus calibration), this table stores the precomputed mean activation
//! value indexed by (layer, token_id, neuron_id).
//!
//! At inference the caller:
//!   1. Calls `skip_mask(layer, tok)` to get which FFN neurons to skip.
//!   2. Runs `linear_q_skip_rows` for gate and up projections (skipping covered rows).
//!   3. Calls `apply(layer, tok, ffn_hidden)` to overwrite the skipped neurons.

use std::collections::HashMap;
use std::path::Path;

pub struct LayerLookup {
    /// by_layer[li][token_id] = Vec<(neuron_id, activation)>
    by_layer: Vec<HashMap<u32, Vec<(usize, f32)>>>,
    pub n_layers:  usize,
    pub n_entries: usize,
    pub n_tokens:  usize,
}

impl LayerLookup {
    pub fn load(path: &Path, n_model_layers: usize) -> Result<Self, String> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("layer_lookup: cannot read {}: {e}", path.display()))?;
        let json: serde_json::Value = serde_json::from_str(&data)
            .map_err(|e| format!("layer_lookup: JSON error: {e}"))?;

        let mut by_layer: Vec<HashMap<u32, Vec<(usize, f32)>>> =
            (0..n_model_layers).map(|_| HashMap::new()).collect();

        // Multi-layer format: {"layers": [{layer: N, entries: [...]}]}
        if let Some(layers_arr) = json["layers"].as_array() {
            let mut n_entries = 0usize;
            for layer_obj in layers_arr {
                let li = layer_obj["layer"].as_u64().unwrap_or(0) as usize;
                if li >= n_model_layers {
                    continue;
                }
                if let Some(entries) = layer_obj["entries"].as_array() {
                    for e in entries {
                        let neuron   = e["neuron"].as_u64().unwrap_or(0) as usize;
                        let token_id = e["token_id"].as_u64().unwrap_or(0) as u32;
                        let act      = e["activation"].as_f64().unwrap_or(0.0) as f32;
                        by_layer[li].entry(token_id).or_default().push((neuron, act));
                        n_entries += 1;
                    }
                }
            }
            let n_tokens = by_layer.iter().map(|m| m.len()).max().unwrap_or(0);
            eprintln!("[layer-lookup] {} entries across {} layers ({} max tokens/layer)",
                      n_entries, n_model_layers, n_tokens);
            let total_tokens: usize = by_layer.iter().map(|m| m.len()).sum();
            return Ok(Self { by_layer, n_layers: n_model_layers, n_entries, n_tokens: total_tokens });
        }

        // Legacy L0-only format: {"entries": [...]}
        if let Some(entries) = json["entries"].as_array() {
            let mut n_entries = 0usize;
            for e in entries {
                let neuron   = e["neuron"].as_u64().unwrap_or(0) as usize;
                let token_id = e["token_id"].as_u64().unwrap_or(0) as u32;
                let act      = e["activation"].as_f64().unwrap_or(0.0) as f32;
                by_layer[0].entry(token_id).or_default().push((neuron, act));
                n_entries += 1;
            }
            let n_tokens = by_layer[0].len();
            eprintln!("[layer-lookup] legacy L0 format: {} entries, {} tokens", n_entries, n_tokens);
            return Ok(Self { by_layer, n_layers: n_model_layers, n_entries, n_tokens });
        }

        Err("layer_lookup: no 'layers' or 'entries' key found in JSON".to_string())
    }

    /// Returns a skip mask (length `ffn_dim`) for the given layer and token.
    /// Returns `None` if this (layer, token) pair has no lookup entries.
    pub fn skip_mask(&self, layer: usize, tok_id: u32, ffn_dim: usize) -> Option<Vec<bool>> {
        let entries = self.by_layer.get(layer)?.get(&tok_id)?;
        let mut mask = vec![false; ffn_dim];
        for &(ni, _) in entries {
            if ni < ffn_dim {
                mask[ni] = true;
            }
        }
        Some(mask)
    }

    /// Overwrites `ffn_hidden[ni]` with the precomputed lookup value.
    pub fn apply(&self, layer: usize, tok_id: u32, ffn_hidden: &mut [f32]) {
        if let Some(layer_map) = self.by_layer.get(layer) {
            if let Some(entries) = layer_map.get(&tok_id) {
                for &(ni, val) in entries {
                    if ni < ffn_hidden.len() {
                        ffn_hidden[ni] = val;
                    }
                }
            }
        }
    }
}
