//! Qwen3 1.7B — native Rust integer inference.
//!
//! Architecture: 28 layers, hidden 2048, 16 Q heads, 8 KV heads (GQA),
//! head_dim 128, SwiGLU FFN (intermediate 6144), RMSNorm, RoPE.
//! Vocab 151,936, tied embeddings.
//!
//! The Operator/Router expert. Handles tool calling and routing decisions.

use crate::loader::ModelWeights;
use strategos_ops::qtensor::{QTensor, QWeight};
use strategos_ops::linear::linear_q;
use strategos_ops::rmsnorm::{RmsNormWeights, rms_norm};
use strategos_ops::rope::RopeTable;
use strategos_ops::lut::{SwishLut, build_swish_bank, find_swish_lut, softmax_int};

// ---- Architecture constants ----

const HIDDEN: usize = 2048;
const LAYERS: usize = 28;
const Q_HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const INTERMEDIATE: usize = 6144;
const VOCAB: usize = 151936;
const RMS_EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 1_000_000.0;
const MAX_SEQ: usize = 4096; // practical limit for CPU inference

/// Per-layer weights.
struct LayerWeights {
    // Attention
    q_proj: QWeight,      // [HIDDEN, Q_HEADS * HEAD_DIM] = [2048, 2048]
    k_proj: QWeight,      // [HIDDEN, KV_HEADS * HEAD_DIM] = [2048, 1024]
    v_proj: QWeight,      // [HIDDEN, KV_HEADS * HEAD_DIM] = [2048, 1024]
    o_proj: QWeight,      // [Q_HEADS * HEAD_DIM, HIDDEN] = [2048, 2048]

    // FFN (SwiGLU)
    gate_proj: QWeight,   // [HIDDEN, INTERMEDIATE] = [2048, 6144]
    up_proj: QWeight,     // [HIDDEN, INTERMEDIATE] = [2048, 6144]
    down_proj: QWeight,   // [INTERMEDIATE, HIDDEN] = [6144, 2048]

    // Norms
    input_norm: RmsNormWeights,
    post_attn_norm: RmsNormWeights,
}

/// KV cache for autoregressive generation.
struct KVCache {
    /// k[layer][kv_head] = Vec of [HEAD_DIM] vectors, one per position.
    k: Vec<Vec<Vec<Vec<f32>>>>,
    /// v[layer][kv_head] = Vec of [HEAD_DIM] vectors.
    v: Vec<Vec<Vec<Vec<f32>>>>,
    seq_len: usize,
}

impl KVCache {
    fn new() -> Self {
        let k = (0..LAYERS).map(|_| {
            (0..KV_HEADS).map(|_| Vec::new()).collect()
        }).collect();
        let v = (0..LAYERS).map(|_| {
            (0..KV_HEADS).map(|_| Vec::new()).collect()
        }).collect();
        Self { k, v, seq_len: 0 }
    }
}

/// Qwen3 1.7B model.
pub struct Qwen3 {
    embed: Vec<Vec<f32>>,  // [VOCAB, HIDDEN] — float, used for lookup only
    layers: Vec<LayerWeights>,
    final_norm: RmsNormWeights,
    // lm_head tied to embed (same weights transposed)
    rope: RopeTable,
    swish_bank: Vec<SwishLut>,
    cache: KVCache,
}

impl Qwen3 {
    /// Load from exported weights (manifest.json + .bin files).
    pub fn load(weights: &ModelWeights) -> Result<Self, String> {
        eprintln!("[qwen3] loading Qwen3-1.7B...");

        // Embeddings: [VOCAB, HIDDEN]
        let embed_flat = weights.require("model.embed_tokens.weight")?;
        let embed: Vec<Vec<f32>> = (0..VOCAB)
            .map(|i| embed_flat[i * HIDDEN..(i + 1) * HIDDEN].to_vec())
            .collect();

        // Per-layer weights.
        let mut layers = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            let prefix = format!("model.layers.{}", l);

            let q_proj = load_qweight(weights, &format!("{}.self_attn.q_proj.weight", prefix), HIDDEN, Q_HEADS * HEAD_DIM)?;
            let k_proj = load_qweight(weights, &format!("{}.self_attn.k_proj.weight", prefix), HIDDEN, KV_HEADS * HEAD_DIM)?;
            let v_proj = load_qweight(weights, &format!("{}.self_attn.v_proj.weight", prefix), HIDDEN, KV_HEADS * HEAD_DIM)?;
            let o_proj = load_qweight(weights, &format!("{}.self_attn.o_proj.weight", prefix), Q_HEADS * HEAD_DIM, HIDDEN)?;

            let gate_proj = load_qweight(weights, &format!("{}.mlp.gate_proj.weight", prefix), HIDDEN, INTERMEDIATE)?;
            let up_proj = load_qweight(weights, &format!("{}.mlp.up_proj.weight", prefix), HIDDEN, INTERMEDIATE)?;
            let down_proj = load_qweight(weights, &format!("{}.mlp.down_proj.weight", prefix), INTERMEDIATE, HIDDEN)?;

            let input_norm = load_rmsnorm(weights, &format!("{}.input_layernorm.weight", prefix))?;
            let post_attn_norm = load_rmsnorm(weights, &format!("{}.post_attention_layernorm.weight", prefix))?;

            layers.push(LayerWeights {
                q_proj, k_proj, v_proj, o_proj,
                gate_proj, up_proj, down_proj,
                input_norm, post_attn_norm,
            });

            if (l + 1) % 7 == 0 {
                eprintln!("[qwen3] loaded layer {}/{}", l + 1, LAYERS);
            }
        }

        let final_norm = load_rmsnorm(weights, "model.norm.weight")?;

        let rope = RopeTable::new(HEAD_DIM, MAX_SEQ, ROPE_THETA);
        let swish_bank = build_swish_bank(64);

        eprintln!("[qwen3] Qwen3-1.7B loaded — {} layers, {} params",
            LAYERS, VOCAB * HIDDEN + LAYERS * (2048*2048*2 + 2048*1024*2 + 2048*6144*3));

        Ok(Self {
            embed, layers, final_norm, rope, swish_bank,
            cache: KVCache::new(),
        })
    }

    /// Reset the KV cache (start new conversation).
    pub fn reset(&mut self) {
        self.cache = KVCache::new();
    }

    /// Forward pass for a single token at the current position.
    /// Returns logits [VOCAB] as float.
    pub fn forward_one(&mut self, token: u32) -> Vec<f32> {
        let pos = self.cache.seq_len;
        assert!(pos < MAX_SEQ, "sequence length {} exceeds max {}", pos, MAX_SEQ);

        // Token embedding.
        let mut hidden = self.embed[token as usize].clone();

        for l in 0..LAYERS {
            let lw = &self.layers[l];

            // Input norm.
            let normed = rms_norm(&hidden, &lw.input_norm);

            // Q/K/V projections.
            let q_input = QTensor::from_f32(&normed);
            let q_out = linear_q(&q_input, &lw.q_proj, &vec![0.0; Q_HEADS * HEAD_DIM]).to_f32();
            let k_out = linear_q(&q_input, &lw.k_proj, &vec![0.0; KV_HEADS * HEAD_DIM]).to_f32();
            let v_out = linear_q(&q_input, &lw.v_proj, &vec![0.0; KV_HEADS * HEAD_DIM]).to_f32();

            // Split into heads and apply RoPE.
            let mut q_heads: Vec<Vec<f32>> = (0..Q_HEADS)
                .map(|h| q_out[h * HEAD_DIM..(h + 1) * HEAD_DIM].to_vec())
                .collect();
            let mut k_heads: Vec<Vec<f32>> = (0..KV_HEADS)
                .map(|h| k_out[h * HEAD_DIM..(h + 1) * HEAD_DIM].to_vec())
                .collect();
            let v_heads: Vec<Vec<f32>> = (0..KV_HEADS)
                .map(|h| v_out[h * HEAD_DIM..(h + 1) * HEAD_DIM].to_vec())
                .collect();

            // Apply RoPE to Q and K.
            for h in 0..Q_HEADS {
                self.rope.apply(&mut q_heads[h], pos);
            }
            for h in 0..KV_HEADS {
                self.rope.apply(&mut k_heads[h], pos);
            }

            // Cache K and V.
            for h in 0..KV_HEADS {
                self.cache.k[l][h].push(k_heads[h].clone());
                self.cache.v[l][h].push(v_heads[h].clone());
            }

            // Attention: Q attends to all cached K/V.
            // GQA: each KV head serves (Q_HEADS / KV_HEADS) = 2 Q heads.
            let kv_group_size = Q_HEADS / KV_HEADS;
            let mut attn_output = vec![0.0f32; Q_HEADS * HEAD_DIM];

            for qh in 0..Q_HEADS {
                let kvh = qh / kv_group_size; // which KV head this Q head uses
                let seq = self.cache.k[l][kvh].len();

                // Compute attention scores: Q[1,128] × K_cached[seq,128]^T
                let mut scores = Vec::with_capacity(seq);
                for s in 0..seq {
                    let dot: f32 = q_heads[qh].iter()
                        .zip(self.cache.k[l][kvh][s].iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    scores.push(dot / (HEAD_DIM as f32).sqrt());
                }

                // Softmax (float — precision critical).
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                let attn_weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

                // Weighted sum of V.
                for d in 0..HEAD_DIM {
                    let mut val = 0.0f32;
                    for s in 0..seq {
                        val += attn_weights[s] * self.cache.v[l][kvh][s][d];
                    }
                    attn_output[qh * HEAD_DIM + d] = val;
                }
            }

            // Output projection.
            let attn_q = QTensor::from_f32(&attn_output);
            let attn_proj = linear_q(&attn_q, &lw.o_proj, &vec![0.0; HIDDEN]).to_f32();

            // Residual.
            for i in 0..HIDDEN {
                hidden[i] += attn_proj[i];
            }

            // Post-attention norm.
            let normed2 = rms_norm(&hidden, &lw.post_attn_norm);

            // SwiGLU FFN: gate = silu(W_gate × x), up = W_up × x, out = W_down × (gate * up)
            let ffn_input = QTensor::from_f32(&normed2);
            let gate = linear_q(&ffn_input, &lw.gate_proj, &vec![0.0; INTERMEDIATE]).to_f32();
            let up = linear_q(&ffn_input, &lw.up_proj, &vec![0.0; INTERMEDIATE]).to_f32();

            // SiLU (same as Swish): x * sigmoid(x)
            let mut gated = vec![0.0f32; INTERMEDIATE];
            for i in 0..INTERMEDIATE {
                let silu = gate[i] * (1.0 / (1.0 + (-gate[i]).exp()));
                gated[i] = silu * up[i];
            }

            let gated_q = QTensor::from_f32(&gated);
            let ffn_out = linear_q(&gated_q, &lw.down_proj, &vec![0.0; HIDDEN]).to_f32();

            // Residual.
            for i in 0..HIDDEN {
                hidden[i] += ffn_out[i];
            }
        }

        // Final norm.
        let normed = rms_norm(&hidden, &self.final_norm);

        // LM head (tied to embed — matrix multiply with embed weights transposed).
        // output[v] = dot(normed, embed[v])
        let mut logits = vec![0.0f32; VOCAB];
        for v in 0..VOCAB {
            let mut dot = 0.0f32;
            for d in 0..HIDDEN {
                dot += normed[d] * self.embed[v][d];
            }
            logits[v] = dot;
        }

        self.cache.seq_len += 1;
        logits
    }

    /// Generate tokens autoregressively.
    pub fn generate(&mut self, prompt_tokens: &[u32], max_new_tokens: usize) -> Vec<u32> {
        self.reset();
        let mut output = Vec::new();

        // Process prompt.
        for &token in prompt_tokens {
            let _logits = self.forward_one(token);
        }

        // Generate.
        let mut next_token = prompt_tokens.last().copied().unwrap_or(0);
        for _ in 0..max_new_tokens {
            let logits = self.forward_one(next_token);

            // Greedy decode: argmax.
            next_token = logits.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);

            output.push(next_token);

            // Stop on EOS (token 151645 for Qwen3).
            if next_token == 151645 {
                break;
            }
        }

        output
    }
}

// ---- Helper functions ----

fn load_qweight(weights: &ModelWeights, name: &str, in_dim: usize, out_dim: usize) -> Result<QWeight, String> {
    let data = weights.require(name)?;
    if data.len() != in_dim * out_dim {
        return Err(format!("{}: expected {}×{}={} elements, got {}",
            name, out_dim, in_dim, in_dim * out_dim, data.len()));
    }
    Ok(QWeight::from_f32(data, in_dim, out_dim))
}

fn load_rmsnorm(weights: &ModelWeights, name: &str) -> Result<RmsNormWeights, String> {
    let data = weights.require(name)?;
    Ok(RmsNormWeights::from_f32(data, RMS_EPS))
}
