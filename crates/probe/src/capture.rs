use std::collections::HashMap;
use serde::Serialize;
use crate::stats::{ScalarAccum, ScalarStats, NeuronFreqAccum, HeadEntropyAccum};

pub struct OpAccum {
    scalar: ScalarAccum,
    neuron_freq: Option<NeuronFreqAccum>,
    head_entropy: Option<HeadEntropyAccum>,
}

impl OpAccum {
    pub fn scalar() -> Self {
        Self { scalar: ScalarAccum::new(), neuron_freq: None, head_entropy: None }
    }

    pub fn with_neuron_freq(n_neurons: usize, threshold: f32) -> Self {
        Self {
            scalar: ScalarAccum::new(),
            neuron_freq: Some(NeuronFreqAccum::new(n_neurons, threshold)),
            head_entropy: None,
        }
    }

    pub fn with_head_entropy(n_heads: usize) -> Self {
        Self {
            scalar: ScalarAccum::new(),
            neuron_freq: None,
            head_entropy: Some(HeadEntropyAccum::new(n_heads)),
        }
    }

    pub fn observe_values(&mut self, values: &[f32]) {
        self.scalar.observe(values);
        if let Some(nf) = &mut self.neuron_freq {
            nf.observe(values);
        }
    }

    pub fn observe_heads(&mut self, head_scores: &[Vec<f32>]) {
        let flat: Vec<f32> = head_scores.iter().flat_map(|s| s.iter().copied()).collect();
        self.scalar.observe(&flat);
        if let Some(he) = &mut self.head_entropy {
            he.observe(head_scores);
        }
    }

    pub fn make_empty_copy(&self) -> Self {
        Self {
            scalar: ScalarAccum::new(),
            neuron_freq: self.neuron_freq.as_ref()
                .map(|nf| NeuronFreqAccum::new(nf.n_neurons, nf.threshold)),
            head_entropy: self.head_entropy.as_ref()
                .map(|he| HeadEntropyAccum::new(he.n_heads)),
        }
    }

    pub fn merge(&mut self, other: OpAccum) {
        self.scalar.merge(other.scalar);
        if let (Some(nf), Some(o)) = (&mut self.neuron_freq, other.neuron_freq) { nf.merge(o); }
        if let (Some(he), Some(o)) = (&mut self.head_entropy, other.head_entropy) { he.merge(o); }
    }

    pub fn finalize(self) -> OpStats {
        OpStats {
            scalar: self.scalar.finalize(),
            neuron_freq: self.neuron_freq.map(|nf| nf.finalize()),
            head_entropy: self.head_entropy.map(|he| he.finalize()),
        }
    }
}

#[derive(Serialize)]
pub struct OpStats {
    #[serde(flatten)]
    pub scalar: ScalarStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neuron_freq: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_entropy: Option<Vec<f32>>,
}

/// Accumulates activations across all tokens for a full forward-pass model.
pub struct CaptureStore {
    ops: HashMap<(usize, &'static str), OpAccum>,
    pub n_tokens: u64,
}

impl CaptureStore {
    pub fn for_grokking() -> Self {
        let mut ops = HashMap::new();
        // Single layer, captures at "=" position (index 2) throughout.
        let l = 0;
        ops.insert((l, "embed"),               OpAccum::scalar());
        ops.insert((l, "embed_b"),             OpAccum::scalar());
        ops.insert((l, "embed_eq"),            OpAccum::scalar());
        ops.insert((l, "attn_q"),              OpAccum::scalar());
        ops.insert((l, "attn_weights"),        OpAccum::with_head_entropy(4));
        ops.insert((l, "attn_out"),            OpAccum::scalar());
        ops.insert((l, "post_attn_residual"),  OpAccum::scalar());
        ops.insert((l, "mlp_hidden"),          OpAccum::with_neuron_freq(512, 0.01));
        ops.insert((l, "mlp_out"),             OpAccum::scalar());
        ops.insert((l, "residual"),            OpAccum::scalar());
        ops.insert((l, "logits"),              OpAccum::scalar());
        ops.insert((l, "correct"),             OpAccum::scalar());
        Self { ops, n_tokens: 0 }
    }

    pub fn for_gpt2(n_layers: usize, n_heads: usize, ffn_dim: usize) -> Self {
        let mut ops = HashMap::new();
        for l in 0..n_layers {
            ops.insert((l, "attn_pre_ln"),  OpAccum::scalar());
            ops.insert((l, "attn_q"),       OpAccum::scalar());
            ops.insert((l, "attn_k"),       OpAccum::scalar());
            ops.insert((l, "attn_v"),       OpAccum::scalar());
            ops.insert((l, "attn_weights"), OpAccum::with_head_entropy(n_heads));
            ops.insert((l, "attn_out"),     OpAccum::scalar());
            ops.insert((l, "ffn_pre_ln"),   OpAccum::scalar());
            ops.insert((l, "ffn_hidden"),   OpAccum::with_neuron_freq(ffn_dim, 0.01));
            ops.insert((l, "ffn_out"),      OpAccum::scalar());
            ops.insert((l, "residual"),     OpAccum::scalar());
        }
        Self { ops, n_tokens: 0 }
    }

    pub fn for_qwen3(n_layers: usize, n_heads: usize, ffn_dim: usize, thresholds: &[f32]) -> Self {
        let mut ops = HashMap::new();
        for l in 0..n_layers {
            let threshold = thresholds.get(l).copied().unwrap_or(0.1);
            ops.insert((l, "attn_pre_rms"), OpAccum::scalar());
            ops.insert((l, "attn_q"),       OpAccum::scalar());
            ops.insert((l, "attn_k"),       OpAccum::scalar());
            ops.insert((l, "attn_v"),       OpAccum::scalar());
            ops.insert((l, "attn_weights"), OpAccum::with_head_entropy(n_heads));
            ops.insert((l, "attn_out"),     OpAccum::scalar());
            ops.insert((l, "ffn_pre_rms"),  OpAccum::scalar());
            ops.insert((l, "ffn_hidden"),   OpAccum::with_neuron_freq(ffn_dim, threshold));
            ops.insert((l, "ffn_out"),      OpAccum::scalar());
            ops.insert((l, "residual"),     OpAccum::scalar());
        }
        Self { ops, n_tokens: 0 }
    }

    pub fn for_gpt2_extract(n_layers: usize, n_heads: usize, ffn_dim: usize, thresholds: &[f32]) -> Self {
        let mut ops = HashMap::new();
        for l in 0..n_layers {
            let threshold = thresholds.get(l).copied().unwrap_or(0.01);
            ops.insert((l, "attn_pre_ln"),  OpAccum::scalar());
            ops.insert((l, "attn_q"),       OpAccum::scalar());
            ops.insert((l, "attn_k"),       OpAccum::scalar());
            ops.insert((l, "attn_v"),       OpAccum::scalar());
            ops.insert((l, "attn_weights"), OpAccum::with_head_entropy(n_heads));
            ops.insert((l, "attn_out"),     OpAccum::scalar());
            ops.insert((l, "ffn_pre_ln"),   OpAccum::scalar());
            ops.insert((l, "ffn_hidden"),   OpAccum::with_neuron_freq(ffn_dim, threshold));
            ops.insert((l, "ffn_out"),      OpAccum::scalar());
            ops.insert((l, "residual"),     OpAccum::scalar());
        }
        Self { ops, n_tokens: 0 }
    }

    pub fn for_smollm2(n_layers: usize, n_heads: usize, ffn_dim: usize, thresholds: &[f32]) -> Self {
        let mut ops = HashMap::new();
        for l in 0..n_layers {
            let threshold = thresholds.get(l).copied().unwrap_or(0.1);
            ops.insert((l, "attn_pre_rms"), OpAccum::scalar());
            ops.insert((l, "attn_q"),       OpAccum::scalar());
            ops.insert((l, "attn_k"),       OpAccum::scalar());
            ops.insert((l, "attn_v"),       OpAccum::scalar());
            ops.insert((l, "attn_weights"), OpAccum::with_head_entropy(n_heads));
            ops.insert((l, "attn_out"),     OpAccum::scalar());
            ops.insert((l, "ffn_pre_rms"),  OpAccum::scalar());
            ops.insert((l, "ffn_hidden"),   OpAccum::with_neuron_freq(ffn_dim, threshold));
            ops.insert((l, "ffn_out"),      OpAccum::scalar());
            ops.insert((l, "residual"),     OpAccum::scalar());
        }
        Self { ops, n_tokens: 0 }
    }

    /// Create a zero-valued copy with the same op registrations — used for parallel chunk processing.
    pub fn make_empty_copy(&self) -> Self {
        let ops = self.ops.iter()
            .map(|(&key, accum)| (key, accum.make_empty_copy()))
            .collect();
        Self { ops, n_tokens: 0 }
    }

    /// Merge another CaptureStore into this one (parallel Welford combination).
    pub fn merge(&mut self, other: CaptureStore) {
        self.n_tokens += other.n_tokens;
        for (key, other_accum) in other.ops {
            if let Some(accum) = self.ops.get_mut(&key) {
                accum.merge(other_accum);
            }
        }
    }

    pub fn observe(&mut self, layer: usize, op: &'static str, values: &[f32]) {
        if let Some(accum) = self.ops.get_mut(&(layer, op)) {
            accum.observe_values(values);
        }
    }

    pub fn observe_heads(&mut self, layer: usize, op: &'static str, head_scores: &[Vec<f32>]) {
        if let Some(accum) = self.ops.get_mut(&(layer, op)) {
            accum.observe_heads(head_scores);
        }
    }

    pub fn finalize(mut self, model: &str, corpus: &str) -> ActivationProfile {
        let n_layers = self.ops.keys().map(|(l, _)| l + 1).max().unwrap_or(0);

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            // Drain all ops registered for this layer, sorted by name for stable output.
            let mut keys: Vec<&'static str> = self.ops.keys()
                .filter(|(ll, _)| *ll == l)
                .map(|(_, op)| *op)
                .collect();
            keys.sort_unstable();

            let mut ops: HashMap<String, OpStats> = HashMap::new();
            for op_name in keys {
                if let Some(accum) = self.ops.remove(&(l, op_name)) {
                    ops.insert(op_name.to_string(), accum.finalize());
                }
            }
            layers.push(LayerProfile { layer: l, ops });
        }

        ActivationProfile {
            model: model.to_string(),
            n_tokens: self.n_tokens,
            corpus: corpus.to_string(),
            layers,
        }
    }
}

#[derive(Serialize)]
pub struct LayerProfile {
    pub layer: usize,
    pub ops: HashMap<String, OpStats>,
}

#[derive(Serialize)]
pub struct ActivationProfile {
    pub model: String,
    pub n_tokens: u64,
    pub corpus: String,
    pub layers: Vec<LayerProfile>,
}
