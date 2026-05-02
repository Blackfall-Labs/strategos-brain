//! Phi-4 Mini 3.8B — native Rust integer inference.
//!
//! Architecture: 32 layers, hidden 3072, 24 Q heads, 8 KV heads (GQA 3:1),
//! head_dim 128, SwiGLU FFN (intermediate 8192), RMSNorm, RoPE.
//! Vocab 200,064, tied embeddings.
//! Fused weights: qkv_proj (Q+K+V in one matrix), gate_up_proj (gate+up).
//!
//! The Thinker expert. Handles reasoning and analysis.

use crate::loader::ModelWeights;
use strategos_ops::qtensor::{QTensor, QWeight};
use strategos_ops::linear::{linear_q, matmul_embed_parallel};
use strategos_ops::rmsnorm::{RmsNormWeights, rms_norm};
use strategos_ops::rope::RopeTable;
use rayon::prelude::*;

const HIDDEN: usize = 3072;
const LAYERS: usize = 32;
const Q_HEADS: usize = 24;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const INTERMEDIATE: usize = 8192;
const VOCAB: usize = 200064;
const RMS_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 10000.0;
const MAX_SEQ: usize = 4096;

const Q_DIM: usize = Q_HEADS * HEAD_DIM;   // 3072
const KV_DIM: usize = KV_HEADS * HEAD_DIM; // 1024
const QKV_DIM: usize = Q_DIM + KV_DIM + KV_DIM; // 5120

struct LayerWeights {
    qkv_proj: QWeight,     // [HIDDEN, QKV_DIM] = [3072, 5120] — fused Q+K+V
    o_proj: QWeight,       // [Q_DIM, HIDDEN]
    gate_up_proj: QWeight, // [HIDDEN, 2*INTERMEDIATE] = [3072, 16384] — fused gate+up
    down_proj: QWeight,    // [INTERMEDIATE, HIDDEN]
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

pub struct Phi4 {
    embed: Vec<Vec<f32>>,
    layers: Vec<LayerWeights>,
    final_norm: RmsNormWeights,
    rope: RopeTable,
    cache: KVCache,
}

impl Phi4 {
    pub fn load(weights: &ModelWeights) -> Result<Self, String> {
        eprintln!("[phi4] loading Phi-4 Mini 3.8B...");

        let embed_flat = weights.require("model.embed_tokens.weight")?;
        let embed: Vec<Vec<f32>> = (0..VOCAB)
            .map(|i| embed_flat[i * HIDDEN..(i + 1) * HIDDEN].to_vec())
            .collect();

        let mut layers = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            let p = format!("model.layers.{}", l);
            layers.push(LayerWeights {
                qkv_proj: load_qw(weights, &format!("{}.self_attn.qkv_proj.weight", p), HIDDEN, QKV_DIM)?,
                o_proj: load_qw(weights, &format!("{}.self_attn.o_proj.weight", p), Q_DIM, HIDDEN)?,
                gate_up_proj: load_qw(weights, &format!("{}.mlp.gate_up_proj.weight", p), HIDDEN, 2 * INTERMEDIATE)?,
                down_proj: load_qw(weights, &format!("{}.mlp.down_proj.weight", p), INTERMEDIATE, HIDDEN)?,
                input_norm: load_rms(weights, &format!("{}.input_layernorm.weight", p))?,
                post_attn_norm: load_rms(weights, &format!("{}.post_attention_layernorm.weight", p))?,
            });
            if (l + 1) % 8 == 0 {
                eprintln!("[phi4] loaded layer {}/{}", l + 1, LAYERS);
            }
        }

        let final_norm = load_rms(weights, "model.norm.weight")?;
        let rope = RopeTable::new(HEAD_DIM, MAX_SEQ, ROPE_THETA);

        eprintln!("[phi4] Phi-4 Mini loaded — {} layers", LAYERS);
        Ok(Self { embed, layers, final_norm, rope, cache: KVCache::new() })
    }

    pub fn reset(&mut self) {
        self.cache = KVCache::new();
    }

    pub fn forward_one(&mut self, token: u32) -> Vec<f32> {
        let pos = self.cache.seq_len;
        let mut hidden = self.embed[token as usize].clone();
        let kv_group = Q_HEADS / KV_HEADS; // 3

        for l in 0..LAYERS {
            let lw = &self.layers[l];
            let normed = rms_norm(&hidden, &lw.input_norm);

            // Fused QKV projection.
            let qkv_in = QTensor::from_f32(&normed);
            let qkv_out = linear_q(&qkv_in, &lw.qkv_proj, &vec![0.0; QKV_DIM]).to_f32();

            // Split: Q [0..Q_DIM], K [Q_DIM..Q_DIM+KV_DIM], V [Q_DIM+KV_DIM..QKV_DIM]
            let q_flat = &qkv_out[0..Q_DIM];
            let k_flat = &qkv_out[Q_DIM..Q_DIM + KV_DIM];
            let v_flat = &qkv_out[Q_DIM + KV_DIM..QKV_DIM];

            let mut q_heads: Vec<Vec<f32>> = (0..Q_HEADS)
                .map(|h| q_flat[h * HEAD_DIM..(h + 1) * HEAD_DIM].to_vec()).collect();
            let mut k_heads: Vec<Vec<f32>> = (0..KV_HEADS)
                .map(|h| k_flat[h * HEAD_DIM..(h + 1) * HEAD_DIM].to_vec()).collect();
            let v_heads: Vec<Vec<f32>> = (0..KV_HEADS)
                .map(|h| v_flat[h * HEAD_DIM..(h + 1) * HEAD_DIM].to_vec()).collect();

            // RoPE.
            for h in 0..Q_HEADS { self.rope.apply(&mut q_heads[h], pos); }
            for h in 0..KV_HEADS { self.rope.apply(&mut k_heads[h], pos); }

            // Cache.
            for h in 0..KV_HEADS {
                self.cache.k[l][h].push(k_heads[h].clone());
                self.cache.v[l][h].push(v_heads[h].clone());
            }

            // GQA attention. Parallel across Q heads.
            let cache_k = &self.cache.k[l];
            let cache_v = &self.cache.v[l];

            let attn_heads: Vec<Vec<f32>> = (0..Q_HEADS).into_par_iter().map(|qh| {
                let kvh = qh / kv_group;
                let seq = cache_k[kvh].len();
                let scale = 1.0 / (HEAD_DIM as f32).sqrt();

                let mut scores = Vec::with_capacity(seq);
                for s in 0..seq {
                    let dot: f32 = q_heads[qh].iter()
                        .zip(cache_k[kvh][s].iter())
                        .map(|(a, b)| a * b).sum();
                    scores.push(dot * scale);
                }

                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
                let sum: f32 = exp.iter().sum();
                let weights: Vec<f32> = exp.iter().map(|e| e / sum).collect();

                let mut head_out = vec![0.0f32; HEAD_DIM];
                for d in 0..HEAD_DIM {
                    let mut val = 0.0f32;
                    for s in 0..seq { val += weights[s] * cache_v[kvh][s][d]; }
                    head_out[d] = val;
                }
                head_out
            }).collect();

            let mut attn_output = vec![0.0f32; Q_DIM];
            for (qh, head) in attn_heads.iter().enumerate() {
                attn_output[qh * HEAD_DIM..(qh + 1) * HEAD_DIM].copy_from_slice(head);
            }

            let attn_q = QTensor::from_f32(&attn_output);
            let attn_proj = linear_q(&attn_q, &lw.o_proj, &vec![0.0; HIDDEN]).to_f32();
            for i in 0..HIDDEN { hidden[i] += attn_proj[i]; }

            // SwiGLU FFN with fused gate+up.
            let normed2 = rms_norm(&hidden, &lw.post_attn_norm);
            let ffn_in = QTensor::from_f32(&normed2);
            let gate_up = linear_q(&ffn_in, &lw.gate_up_proj, &vec![0.0; 2 * INTERMEDIATE]).to_f32();

            // Split: gate [0..INTERMEDIATE], up [INTERMEDIATE..2*INTERMEDIATE]
            let gated: Vec<f32> = (0..INTERMEDIATE).into_par_iter()
                .map(|i| {
                    let g = gate_up[i];
                    let u = gate_up[INTERMEDIATE + i];
                    let silu = g * (1.0 / (1.0 + (-g).exp()));
                    silu * u
                }).collect();

            let gated_q = QTensor::from_f32(&gated);
            let ffn_out = linear_q(&gated_q, &lw.down_proj, &vec![0.0; HIDDEN]).to_f32();
            for i in 0..HIDDEN { hidden[i] += ffn_out[i]; }
        }

        let normed = rms_norm(&hidden, &self.final_norm);

        // Tied embeddings. Parallel across vocab.
        let logits = matmul_embed_parallel(&normed, &self.embed);

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
            if next == 200018 { break; } // EOS
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
