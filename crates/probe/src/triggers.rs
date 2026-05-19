use std::collections::HashMap;
use serde::Serialize;

/// Tracks which input tokens co-activate each selective neuron.
/// "Selective" = fire rate 1%-95% in the calibrated profile.
pub struct TriggerStore {
    /// track_mask[layer][neuron] = true if this neuron should be tracked.
    pub track_mask: Vec<Vec<bool>>,
    /// Per-layer thresholds (same as calibrated probe).
    pub thresholds: Vec<f32>,
    /// fire_counts[layer][neuron] = { token_id → fire_count }
    pub fire_counts: Vec<Vec<HashMap<u32, u32>>>,
    /// total_fires[layer][neuron] = total times this neuron fired.
    pub total_fires: Vec<Vec<u64>>,
    /// token_counts[token_id] = how many times this token appeared as the current token.
    pub token_counts: HashMap<u32, u64>,
    pub n_tokens: u64,
}

impl TriggerStore {
    /// Build from a calibrated activation profile JSON.
    /// Only tracks neurons with fire_rate in [min_rate, max_rate].
    pub fn from_profile(
        profile: &serde_json::Value,
        thresholds: Vec<f32>,
        min_rate: f32,
        max_rate: f32,
    ) -> Self {
        let layers_json = profile["layers"].as_array().expect("layers array");
        let n_layers = layers_json.len();

        let mut track_mask: Vec<Vec<bool>> = Vec::with_capacity(n_layers);
        let mut fire_counts: Vec<Vec<HashMap<u32, u32>>> = Vec::with_capacity(n_layers);
        let mut total_fires: Vec<Vec<u64>> = Vec::with_capacity(n_layers);

        let mut tracked_total = 0usize;

        for layer_val in layers_json {
            let nf_arr = layer_val["ops"]["ffn_hidden"]["neuron_freq"].as_array();
            if let Some(nf) = nf_arr {
                let n = nf.len();
                let mask: Vec<bool> = nf.iter()
                    .map(|v| {
                        let rate = v.as_f64().unwrap_or(0.0) as f32;
                        rate >= min_rate && rate <= max_rate
                    })
                    .collect();
                tracked_total += mask.iter().filter(|&&b| b).count();
                fire_counts.push(mask.iter().map(|_| HashMap::new()).collect());
                total_fires.push(vec![0u64; n]);
                track_mask.push(mask);
            } else {
                track_mask.push(Vec::new());
                fire_counts.push(Vec::new());
                total_fires.push(Vec::new());
            }
        }

        eprintln!("[extract] tracking {} selective neurons across {} layers", tracked_total, n_layers);

        Self {
            track_mask,
            thresholds,
            fire_counts,
            total_fires,
            token_counts: HashMap::new(),
            n_tokens: 0,
        }
    }

    /// Record which neurons fired for the current token at this layer.
    pub fn observe(&mut self, layer: usize, values: &[f32], current_tok: u32) {
        let threshold = self.thresholds.get(layer).copied().unwrap_or(0.1);
        let mask = &self.track_mask[layer];
        let fires = &mut self.fire_counts[layer];
        let totals = &mut self.total_fires[layer];

        for (i, &v) in values.iter().enumerate() {
            if i < mask.len() && mask[i] && v.abs() > threshold {
                *fires[i].entry(current_tok).or_insert(0) += 1;
                totals[i] += 1;
            }
        }
    }

    pub fn observe_token(&mut self, tok: u32) {
        *self.token_counts.entry(tok).or_insert(0) += 1;
        self.n_tokens += 1;
    }

    pub fn make_empty_copy(&self) -> Self {
        let fire_counts = self.track_mask.iter()
            .map(|mask| mask.iter().map(|_| HashMap::new()).collect())
            .collect();
        let total_fires = self.total_fires.iter()
            .map(|v| vec![0u64; v.len()])
            .collect();
        Self {
            track_mask: self.track_mask.clone(),
            thresholds: self.thresholds.clone(),
            fire_counts,
            total_fires,
            token_counts: HashMap::new(),
            n_tokens: 0,
        }
    }

    pub fn merge(&mut self, other: TriggerStore) {
        self.n_tokens += other.n_tokens;
        for (tok, count) in other.token_counts {
            *self.token_counts.entry(tok).or_insert(0) += count;
        }
        for (li, (layer_fires, layer_totals)) in
            other.fire_counts.into_iter().zip(other.total_fires.into_iter()).enumerate()
        {
            for (ni, (neuron_fires, total)) in
                layer_fires.into_iter().zip(layer_totals.into_iter()).enumerate()
            {
                self.total_fires[li][ni] += total;
                for (tok, count) in neuron_fires {
                    *self.fire_counts[li][ni].entry(tok).or_insert(0) += count;
                }
            }
        }
    }

    /// Finalize into a serializable trigger profile.
    /// `vocab`: optional token_id → string mapping for human-readable output.
    pub fn finalize(self, model: &str, corpus: &str, vocab: Option<&HashMap<u32, String>>, top_k: usize) -> TriggerProfile {
        let n_layers = self.track_mask.len();
        let mut layers = Vec::with_capacity(n_layers);

        for li in 0..n_layers {
            let mask = &self.track_mask[li];
            let mut neurons = Vec::new();

            for ni in 0..mask.len() {
                if !mask[ni] { continue; }
                let total = self.total_fires[li][ni];
                if total == 0 { continue; }

                let fire_rate = total as f32 / self.n_tokens.max(1) as f32;

                // Sort token triggers by fire count descending, take top_k.
                let mut hits: Vec<(u32, u32)> = self.fire_counts[li][ni].iter()
                    .map(|(&tok, &cnt)| (tok, cnt))
                    .collect();
                hits.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                hits.truncate(top_k);

                let top_tokens: Vec<TokenHit> = hits.into_iter().map(|(tok, cnt)| {
                    let tok_total = self.token_counts.get(&tok).copied().unwrap_or(1);
                    TokenHit {
                        token_id: tok,
                        token_str: vocab.and_then(|v| v.get(&tok)).cloned().unwrap_or_default(),
                        fire_count: cnt as u64,
                        // Of all times neuron fired, what fraction had this token?
                        neuron_share: cnt as f32 / total as f32,
                        // Of all times this token appeared, what fraction fired neuron?
                        trigger_rate: cnt as f32 / tok_total as f32,
                    }
                }).collect();

                neurons.push(NeuronTrigger {
                    neuron: ni,
                    total_fires: total,
                    fire_rate,
                    top_tokens,
                });
            }

            layers.push(LayerTriggers { layer: li, neurons });
        }

        TriggerProfile {
            model: model.to_string(),
            corpus: corpus.to_string(),
            n_tokens: self.n_tokens,
            layers,
        }
    }
}

#[derive(Serialize)]
pub struct TokenHit {
    pub token_id: u32,
    pub token_str: String,
    /// Of all firings of this neuron, fraction where this token was current.
    pub neuron_share: f32,
    /// Of all occurrences of this token, fraction that fired this neuron.
    pub trigger_rate: f32,
    pub fire_count: u64,
}

#[derive(Serialize)]
pub struct NeuronTrigger {
    pub neuron: usize,
    pub total_fires: u64,
    pub fire_rate: f32,
    pub top_tokens: Vec<TokenHit>,
}

#[derive(Serialize)]
pub struct LayerTriggers {
    pub layer: usize,
    pub neurons: Vec<NeuronTrigger>,
}

#[derive(Serialize)]
pub struct TriggerProfile {
    pub model: String,
    pub corpus: String,
    pub n_tokens: u64,
    pub layers: Vec<LayerTriggers>,
}
