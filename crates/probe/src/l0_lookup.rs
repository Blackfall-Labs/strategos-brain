//! Layer-0 embedding-dominant lookup table.
//!
//! Loaded from `smollm2_l0_lookup_filtered.json` (or any compatible file).
//! For Band 1 L0 neurons whose activation can be predicted from the token embedding alone,
//! this table stores the precomputed activation value indexed by (token_id, neuron_id).
//!
//! At inference the caller:
//!   1. Calls `skip_mask(tok)` to get which FFN neurons to skip in gate/up matmuls.
//!   2. Runs `linear_q_skip_rows` for the gate and up projections.
//!   3. Calls `apply(tok, ffn_hidden)` to overwrite the skipped neurons.

use std::collections::HashMap;
use std::path::Path;

pub struct L0Lookup {
    /// token_id → sorted Vec<(neuron_id, activation_value)>
    pub by_token: HashMap<u32, Vec<(usize, f32)>>,
    pub n_entries: usize,
    pub n_tokens:  usize,
}

impl L0Lookup {
    pub fn load(path: &Path) -> Result<Self, String> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("l0_lookup: cannot read {}: {e}", path.display()))?;
        let json: serde_json::Value = serde_json::from_str(&data)
            .map_err(|e| format!("l0_lookup: JSON error: {e}"))?;

        let mut by_token: HashMap<u32, Vec<(usize, f32)>> = HashMap::new();

        if let Some(entries) = json["entries"].as_array() {
            for e in entries {
                let neuron     = e["neuron"].as_u64().unwrap_or(0) as usize;
                let token_id   = e["token_id"].as_u64().unwrap_or(0) as u32;
                let activation = e["activation"].as_f64().unwrap_or(0.0) as f32;
                by_token.entry(token_id).or_default().push((neuron, activation));
            }
        }

        let n_entries = by_token.values().map(|v| v.len()).sum();
        let n_tokens  = by_token.len();

        eprintln!("[l0-lookup] {} entries, {} trigger tokens", n_entries, n_tokens);
        Ok(Self { by_token, n_entries, n_tokens })
    }

    /// Returns a skip mask (length `ffn_dim`) for use with `linear_q_skip_rows`.
    /// Returns `None` if this token has no lookup entries.
    pub fn skip_mask(&self, tok_id: u32, ffn_dim: usize) -> Option<Vec<bool>> {
        let entries = self.by_token.get(&tok_id)?;
        let mut mask = vec![false; ffn_dim];
        for &(ni, _) in entries {
            if ni < ffn_dim {
                mask[ni] = true;
            }
        }
        Some(mask)
    }

    /// Overwrites `ffn_hidden[ni]` with the precomputed lookup value for each covered neuron.
    pub fn apply(&self, tok_id: u32, ffn_hidden: &mut [f32]) {
        if let Some(entries) = self.by_token.get(&tok_id) {
            for &(ni, val) in entries {
                if ni < ffn_hidden.len() {
                    ffn_hidden[ni] = val;
                }
            }
        }
    }
}
