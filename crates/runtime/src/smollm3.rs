//! SmolLM3 3B — native Rust integer inference.
//!
//! Architecture: 36 layers, hidden 2048, 16 Q heads, 4 KV heads (GQA 4:1),
//! head_dim 128, SwiGLU FFN (intermediate 11008), RMSNorm, RoPE with NoPE
//! (skip RoPE every 4th layer). Vocab 128,256, tied embeddings.
//!
//! The Communicator expert. Handles language generation and structured output.

use crate::loader::ModelWeights;
use strategos_ops::qtensor::{QTensor, QWeight};
use strategos_ops::linear::linear_q;
use strategos_ops::rmsnorm::{RmsNormWeights, rms_norm};
use strategos_ops::rope::RopeTable;

const HIDDEN: usize = 2048;
const LAYERS: usize = 36;
const Q_HEADS: usize = 16;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 128;
const INTERMEDIATE: usize = 11008;
const VOCAB: usize = 128256;
const RMS_EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 5_000_000.0;
const MAX_SEQ: usize = 4096;

/// NoPE: which layers skip RoPE. Every 4th layer (0-indexed: 3,7,11,...).
fn has_rope(layer: usize) -> bool {
    (layer + 1) % 4 != 0
}

struct LayerWeights {
    q_proj: QWeight,
    k_proj: QWeight,
    v_proj: QWeight,
    o_proj: QWeight,
    gate_proj: QWeight,
    up_proj: QWeight,
    down_proj: QWeight,
    input_norm: RmsNormWeights,
    post_attn_norm: RmsNormWeights,
}

struct KVCache {
    k: Vec<Vec<Vec<Vec<f32>>>>,
    v: Vec<Vec<Vec<Vec<f32>>>>,
    seq_len: usize,
}

impl KVCache {
    fn new() -> Self {
        Self {
            k: (0..LAYERS).map(|_| (0..KV_HEADS).map(|_| Vec::new()).collect()).collect(),
            v: (0..LAYERS).map(|_| (0..KV_HEADS).map(|_| Vec::new()).collect()).collect(),
            seq_len: 0,
        }
    }
}

pub struct SmolLM3 {
    embed: Vec<Vec<f32>>,
    layers: Vec<LayerWeights>,
    final_norm: RmsNormWeights,
    rope: RopeTable,
    cache: KVCache,
}

impl SmolLM3 {
    pub fn load(weights: &ModelWeights) -> Result<Self, String> {
        eprintln!("[smollm3] loading SmolLM3-3B...");

        let embed_flat = weights.require("model.embed_tokens.weight")?;
        let embed: Vec<Vec<f32>> = (0..VOCAB)
            .map(|i| embed_flat[i * HIDDEN..(i + 1) * HIDDEN].to_vec())
            .collect();

        let mut layers = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            let p = format!("model.layers.{}", l);
            layers.push(LayerWeights {
                q_proj: load_qw(weights, &format!("{}.self_attn.q_proj.weight", p), HIDDEN, Q_HEADS * HEAD_DIM)?,
                k_proj: load_qw(weights, &format!("{}.self_attn.k_proj.weight", p), HIDDEN, KV_HEADS * HEAD_DIM)?,
                v_proj: load_qw(weights, &format!("{}.self_attn.v_proj.weight", p), HIDDEN, KV_HEADS * HEAD_DIM)?,
                o_proj: load_qw(weights, &format!("{}.self_attn.o_proj.weight", p), Q_HEADS * HEAD_DIM, HIDDEN)?,
                gate_proj: load_qw(weights, &format!("{}.mlp.gate_proj.weight", p), HIDDEN, INTERMEDIATE)?,
                up_proj: load_qw(weights, &format!("{}.mlp.up_proj.weight", p), HIDDEN, INTERMEDIATE)?,
                down_proj: load_qw(weights, &format!("{}.mlp.down_proj.weight", p), INTERMEDIATE, HIDDEN)?,
                input_norm: load_rms(weights, &format!("{}.input_layernorm.weight", p))?,
                post_attn_norm: load_rms(weights, &format!("{}.post_attention_layernorm.weight", p))?,
            });
            if (l + 1) % 9 == 0 {
                eprintln!("[smollm3] loaded layer {}/{}", l + 1, LAYERS);
            }
        }

        let final_norm = load_rms(weights, "model.norm.weight")?;
        let rope = RopeTable::new(HEAD_DIM, MAX_SEQ, ROPE_THETA);

        eprintln!("[smollm3] SmolLM3-3B loaded — {} layers", LAYERS);
        Ok(Self { embed, layers, final_norm, rope, cache: KVCache::new() })
    }

    pub fn reset(&mut self) {
        self.cache = KVCache::new();
    }

    pub fn forward_one(&mut self, token: u32) -> Vec<f32> {
        let pos = self.cache.seq_len;
        let mut hidden = self.embed[token as usize].clone();
        let kv_group = Q_HEADS / KV_HEADS; // 4

        for l in 0..LAYERS {
            let lw = &self.layers[l];
            let normed = rms_norm(&hidden, &lw.input_norm);

            let q_in = QTensor::from_f32(&normed);
            let q_out = linear_q(&q_in, &lw.q_proj, &vec![0.0; Q_HEADS * HEAD_DIM]).to_f32();
            let k_out = linear_q(&q_in, &lw.k_proj, &vec![0.0; KV_HEADS * HEAD_DIM]).to_f32();
            let v_out = linear_q(&q_in, &lw.v_proj, &vec![0.0; KV_HEADS * HEAD_DIM]).to_f32();

            let mut q_heads: Vec<Vec<f32>> = (0..Q_HEADS)
                .map(|h| q_out[h * HEAD_DIM..(h + 1) * HEAD_DIM].to_vec()).collect();
            let mut k_heads: Vec<Vec<f32>> = (0..KV_HEADS)
                .map(|h| k_out[h * HEAD_DIM..(h + 1) * HEAD_DIM].to_vec()).collect();
            let v_heads: Vec<Vec<f32>> = (0..KV_HEADS)
                .map(|h| v_out[h * HEAD_DIM..(h + 1) * HEAD_DIM].to_vec()).collect();

            // NoPE: apply RoPE only on layers where has_rope(l) is true.
            if has_rope(l) {
                for h in 0..Q_HEADS { self.rope.apply(&mut q_heads[h], pos); }
                for h in 0..KV_HEADS { self.rope.apply(&mut k_heads[h], pos); }
            }

            for h in 0..KV_HEADS {
                self.cache.k[l][h].push(k_heads[h].clone());
                self.cache.v[l][h].push(v_heads[h].clone());
            }

            // GQA attention.
            let mut attn_output = vec![0.0f32; Q_HEADS * HEAD_DIM];
            let seq = self.cache.k[l][0].len();

            for qh in 0..Q_HEADS {
                let kvh = qh / kv_group;
                let mut scores = Vec::with_capacity(seq);
                for s in 0..seq {
                    let dot: f32 = q_heads[qh].iter()
                        .zip(self.cache.k[l][kvh][s].iter())
                        .map(|(a, b)| a * b).sum();
                    scores.push(dot / (HEAD_DIM as f32).sqrt());
                }

                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
                let sum: f32 = exp.iter().sum();
                let weights: Vec<f32> = exp.iter().map(|e| e / sum).collect();

                for d in 0..HEAD_DIM {
                    let mut val = 0.0f32;
                    for s in 0..seq { val += weights[s] * self.cache.v[l][kvh][s][d]; }
                    attn_output[qh * HEAD_DIM + d] = val;
                }
            }

            let attn_q = QTensor::from_f32(&attn_output);
            let attn_proj = linear_q(&attn_q, &lw.o_proj, &vec![0.0; HIDDEN]).to_f32();
            for i in 0..HIDDEN { hidden[i] += attn_proj[i]; }

            // SwiGLU FFN.
            let normed2 = rms_norm(&hidden, &lw.post_attn_norm);
            let ffn_in = QTensor::from_f32(&normed2);
            let gate = linear_q(&ffn_in, &lw.gate_proj, &vec![0.0; INTERMEDIATE]).to_f32();
            let up = linear_q(&ffn_in, &lw.up_proj, &vec![0.0; INTERMEDIATE]).to_f32();

            let mut gated = vec![0.0f32; INTERMEDIATE];
            for i in 0..INTERMEDIATE {
                let silu = gate[i] * (1.0 / (1.0 + (-gate[i]).exp()));
                gated[i] = silu * up[i];
            }

            let gated_q = QTensor::from_f32(&gated);
            let ffn_out = linear_q(&gated_q, &lw.down_proj, &vec![0.0; HIDDEN]).to_f32();
            for i in 0..HIDDEN { hidden[i] += ffn_out[i]; }
        }

        let normed = rms_norm(&hidden, &self.final_norm);

        // Tied embeddings: lm_head = embed^T.
        let mut logits = vec![0.0f32; VOCAB];
        for v in 0..VOCAB {
            let mut dot = 0.0f32;
            for d in 0..HIDDEN { dot += normed[d] * self.embed[v][d]; }
            logits[v] = dot;
        }

        self.cache.seq_len += 1;
        logits
    }

    pub fn generate(&mut self, prompt_tokens: &[u32], max_new: usize) -> Vec<u32> {
        self.reset();
        for &t in prompt_tokens { let _ = self.forward_one(t); }

        let mut out = Vec::new();
        let mut next = prompt_tokens.last().copied().unwrap_or(0);
        for _ in 0..max_new {
            let logits = self.forward_one(next);
            next = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32).unwrap_or(0);
            out.push(next);
            if next == 128012 { break; } // EOS
        }
        out
    }
}

fn load_qw(w: &ModelWeights, name: &str, in_d: usize, out_d: usize) -> Result<QWeight, String> {
    let data = w.require(name)?;
    if data.len() != in_d * out_d {
        return Err(format!("{}: expected {}×{}={}, got {}", name, out_d, in_d, in_d * out_d, data.len()));
    }
    Ok(QWeight::from_f32(data, in_d, out_d))
}

fn load_rms(w: &ModelWeights, name: &str) -> Result<RmsNormWeights, String> {
    Ok(RmsNormWeights::from_f32(w.require(name)?, RMS_EPS))
}
