use std::path::Path;
use rayon::prelude::*;
use strategos_ops::qtensor::{QTensor, QWeight};
use strategos_ops::linear::{linear_q, linear_q_skip_rows};
use strategos_ops::rmsnorm::{RmsNormWeights, rms_norm};
use strategos_ops::rope::RopeTable;
use strategos_runtime::loader::ModelWeights;
use crate::capture::CaptureStore;
use crate::triggers::TriggerStore;
use crate::l0_lookup::L0Lookup;

// Architecture constants from config.json
const HIDDEN:    usize = 576;
const N_Q_HEADS: usize = 9;
const N_KV_HEADS: usize = 3;
const HEAD_DIM:  usize = 64;   // HIDDEN / N_Q_HEADS
const FFN_DIM:   usize = 1536;
const VOCAB:     usize = 49152;
const N_LAYERS:  usize = 30;
const RMS_EPS:   f32   = 1e-5;
const ROPE_BASE: f32   = 100_000.0;
const MAX_SEQ:   usize = 512;

// Each KV head is shared by N_Q_HEADS / N_KV_HEADS query heads.
const KV_GROUP: usize = N_Q_HEADS / N_KV_HEADS;

pub struct SmolLM2Probe {
    embed: Vec<f32>,   // [VOCAB, HIDDEN]
    layers: Vec<SmolLM2Layer>,
    rope: RopeTable,
    pub n_layer: usize,
    pub n_head:  usize,
    pub ffn_dim: usize,
    pub lookup: Option<L0Lookup>,
}

struct SmolLM2Layer {
    input_ln:   RmsNormWeights,
    q_proj:     QWeight,   // [N_Q_HEADS*HEAD_DIM, HIDDEN] = [576, 576]
    k_proj:     QWeight,   // [N_KV_HEADS*HEAD_DIM, HIDDEN] = [192, 576]
    v_proj:     QWeight,   // [N_KV_HEADS*HEAD_DIM, HIDDEN] = [192, 576]
    o_proj:     QWeight,   // [HIDDEN, N_Q_HEADS*HEAD_DIM] = [576, 576]
    post_attn_ln: RmsNormWeights,
    gate_proj:  QWeight,   // [FFN_DIM, HIDDEN] = [1536, 576]
    up_proj:    QWeight,   // [FFN_DIM, HIDDEN] = [1536, 576]
    down_proj:  QWeight,   // [HIDDEN, FFN_DIM] = [576, 1536]
}

struct KVCache {
    k: Vec<Vec<Vec<Vec<f32>>>>,  // [layer][kv_head][seq][head_dim]
    v: Vec<Vec<Vec<Vec<f32>>>>,
}

impl KVCache {
    fn new() -> Self {
        let k = (0..N_LAYERS).map(|_| (0..N_KV_HEADS).map(|_| Vec::new()).collect()).collect();
        let v = (0..N_LAYERS).map(|_| (0..N_KV_HEADS).map(|_| Vec::new()).collect()).collect();
        Self { k, v }
    }
}

impl SmolLM2Probe {
    pub fn load(manifest_dir: &Path) -> Result<Self, String> {
        let weights = ModelWeights::load(manifest_dir)?;
        let get = |name: &str| -> Result<Vec<f32>, String> {
            weights.require(name).map(|s| s.to_vec())
        };

        let embed = get("embed_tokens.weight")?;

        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            let p = format!("layers.{i}");
            // HF stores linear weights [out, in] already (unlike GPT-2 Conv1D).
            layers.push(SmolLM2Layer {
                input_ln:    RmsNormWeights::from_f32(&get(&format!("{p}.input_layernorm.weight"))?, RMS_EPS),
                q_proj:      QWeight::from_f32(&get(&format!("{p}.self_attn.q_proj.weight"))?,   HIDDEN, N_Q_HEADS * HEAD_DIM),
                k_proj:      QWeight::from_f32(&get(&format!("{p}.self_attn.k_proj.weight"))?,   HIDDEN, N_KV_HEADS * HEAD_DIM),
                v_proj:      QWeight::from_f32(&get(&format!("{p}.self_attn.v_proj.weight"))?,   HIDDEN, N_KV_HEADS * HEAD_DIM),
                o_proj:      QWeight::from_f32(&get(&format!("{p}.self_attn.o_proj.weight"))?,   N_Q_HEADS * HEAD_DIM, HIDDEN),
                post_attn_ln: RmsNormWeights::from_f32(&get(&format!("{p}.post_attention_layernorm.weight"))?, RMS_EPS),
                gate_proj:   QWeight::from_f32(&get(&format!("{p}.mlp.gate_proj.weight"))?,      HIDDEN, FFN_DIM),
                up_proj:     QWeight::from_f32(&get(&format!("{p}.mlp.up_proj.weight"))?,        HIDDEN, FFN_DIM),
                down_proj:   QWeight::from_f32(&get(&format!("{p}.mlp.down_proj.weight"))?,      FFN_DIM, HIDDEN),
            });
        }

        let rope = RopeTable::new(HEAD_DIM, MAX_SEQ, ROPE_BASE);
        eprintln!("[smollm2-probe] loaded {} layers", N_LAYERS);

        Ok(Self { embed, layers, rope, n_layer: N_LAYERS, n_head: N_Q_HEADS, ffn_dim: FFN_DIM, lookup: None })
    }

    pub fn run_corpus(&self, tokens: &[u32], captures: &mut CaptureStore) {
        const CONTEXT_LEN: usize = 128;

        let partials: Vec<CaptureStore> = tokens
            .par_chunks(CONTEXT_LEN)
            .map(|chunk| {
                let mut store = captures.make_empty_copy();
                let mut cache = KVCache::new();
                for (ctx_pos, &tok) in chunk.iter().enumerate() {
                    self.forward_token(tok, ctx_pos, &mut cache, &mut store);
                    store.n_tokens += 1;
                }
                store
            })
            .collect();

        let total: u64 = partials.iter().map(|s| s.n_tokens).sum();
        for partial in partials {
            captures.merge(partial);
        }
        eprintln!("[probe] {} tokens processed", total);
    }

    pub fn run_corpus_extract(&self, tokens: &[u32], captures: &mut CaptureStore, triggers: &mut TriggerStore) {
        const CONTEXT_LEN: usize = 128;

        let partials: Vec<(CaptureStore, TriggerStore)> = tokens
            .par_chunks(CONTEXT_LEN)
            .map(|chunk| {
                let mut store = captures.make_empty_copy();
                let mut trig = triggers.make_empty_copy();
                let mut cache = KVCache::new();
                for (ctx_pos, &tok) in chunk.iter().enumerate() {
                    self.forward_token_extract(tok, ctx_pos, &mut cache, &mut store, &mut trig);
                    store.n_tokens += 1;
                    trig.observe_token(tok);
                }
                (store, trig)
            })
            .collect();

        let total: u64 = partials.iter().map(|(s, _)| s.n_tokens).sum();
        for (partial_s, partial_t) in partials {
            captures.merge(partial_s);
            triggers.merge(partial_t);
        }
        eprintln!("[extract] {} tokens processed", total);
    }

    fn forward_token_extract(&self, tok: u32, ctx_pos: usize, cache: &mut KVCache, captures: &mut CaptureStore, triggers: &mut TriggerStore) {
        self.forward_token_inner(tok, ctx_pos, cache, captures, Some((tok, triggers)));
    }

    fn forward_token(&self, tok: u32, ctx_pos: usize, cache: &mut KVCache, captures: &mut CaptureStore) {
        self.forward_token_inner(tok, ctx_pos, cache, captures, None);
    }

    fn forward_token_inner(&self, tok: u32, ctx_pos: usize, cache: &mut KVCache, captures: &mut CaptureStore, mut trigger_hook: Option<(u32, &mut TriggerStore)>) {
        let tok_off = (tok as usize).min(VOCAB - 1) * HIDDEN;
        let mut hidden = self.embed[tok_off..tok_off + HIDDEN].to_vec();

        for (li, layer) in self.layers.iter().enumerate() {
            // --- Attention ---
            let pre_attn = rms_norm(&hidden, &layer.input_ln);
            captures.observe(li, "attn_pre_rms", &pre_attn);

            let x_q = QTensor::from_f32(&pre_attn);
            let no_bias = vec![0.0f32; N_Q_HEADS * HEAD_DIM];
            let kv_no_bias = vec![0.0f32; N_KV_HEADS * HEAD_DIM];

            let q_raw = linear_q(&x_q, &layer.q_proj, &no_bias).to_f32();
            let k_raw = linear_q(&x_q, &layer.k_proj, &kv_no_bias).to_f32();
            let v_raw = linear_q(&x_q, &layer.v_proj, &kv_no_bias).to_f32();

            captures.observe(li, "attn_q", &q_raw);
            captures.observe(li, "attn_k", &k_raw);
            captures.observe(li, "attn_v", &v_raw);

            // Apply RoPE (non-interleaved: pairs i and i+HEAD_DIM/2)
            let mut q = q_raw.clone();
            let mut k = k_raw.clone();
            for h in 0..N_Q_HEADS {
                apply_rope_half(&mut q[h * HEAD_DIM..(h + 1) * HEAD_DIM], &self.rope, ctx_pos);
            }
            for h in 0..N_KV_HEADS {
                apply_rope_half(&mut k[h * HEAD_DIM..(h + 1) * HEAD_DIM], &self.rope, ctx_pos);
            }

            // Store KV for this position
            for kv_h in 0..N_KV_HEADS {
                cache.k[li][kv_h].push(k[kv_h * HEAD_DIM..(kv_h + 1) * HEAD_DIM].to_vec());
                cache.v[li][kv_h].push(v_raw[kv_h * HEAD_DIM..(kv_h + 1) * HEAD_DIM].to_vec());
            }

            let seq_len = cache.k[li][0].len();
            let scale = (HEAD_DIM as f32).sqrt();
            let mut attn_out = vec![0.0f32; HIDDEN];
            let mut head_scores_snap: Vec<Vec<f32>> = Vec::with_capacity(N_Q_HEADS);

            for q_h in 0..N_Q_HEADS {
                let kv_h = q_h / KV_GROUP;
                let mut scores = vec![0.0f32; seq_len];
                for ki in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..HEAD_DIM {
                        dot += q[q_h * HEAD_DIM + d] * cache.k[li][kv_h][ki][d];
                    }
                    scores[ki] = dot / scale;
                }
                softmax(&mut scores);
                head_scores_snap.push(scores.clone());
                for d in 0..HEAD_DIM {
                    let mut sum = 0.0f32;
                    for ki in 0..seq_len { sum += scores[ki] * cache.v[li][kv_h][ki][d]; }
                    attn_out[q_h * HEAD_DIM + d] = sum;
                }
            }

            captures.observe_heads(li, "attn_weights", &head_scores_snap);
            captures.observe(li, "attn_out", &attn_out);

            let a_q = QTensor::from_f32(&attn_out);
            let o_bias = vec![0.0f32; HIDDEN];
            let proj_f = linear_q(&a_q, &layer.o_proj, &o_bias).to_f32();
            for i in 0..HIDDEN { hidden[i] += proj_f[i]; }

            // --- FFN (SwiGLU) ---
            let pre_ffn = rms_norm(&hidden, &layer.post_attn_ln);
            captures.observe(li, "ffn_pre_rms", &pre_ffn);

            let x2 = QTensor::from_f32(&pre_ffn);
            let ffn_no_bias = vec![0.0f32; FFN_DIM];

            // At L0: use skip-row matmul for lookup-covered neurons, then overwrite.
            let ffn_hidden = if li == 0 {
                if let Some(ref lut) = self.lookup {
                    if let Some(skip) = lut.skip_mask(tok, FFN_DIM) {
                        let gate_f = linear_q_skip_rows(&x2, &layer.gate_proj, &ffn_no_bias, &skip).to_f32();
                        let up_f   = linear_q_skip_rows(&x2, &layer.up_proj,   &ffn_no_bias, &skip).to_f32();
                        let mut h = vec![0.0f32; FFN_DIM];
                        for i in 0..FFN_DIM {
                            if !skip[i] { h[i] = silu(gate_f[i]) * up_f[i]; }
                        }
                        lut.apply(tok, &mut h);
                        h
                    } else {
                        ffn_standard(&x2, layer, &ffn_no_bias)
                    }
                } else {
                    ffn_standard(&x2, layer, &ffn_no_bias)
                }
            } else {
                ffn_standard(&x2, layer, &ffn_no_bias)
            };

            captures.observe(li, "ffn_hidden", &ffn_hidden);
            if let Some((current_tok, ref mut trig)) = trigger_hook {
                trig.observe(li, &ffn_hidden, current_tok);
            }

            let h_q = QTensor::from_f32(&ffn_hidden);
            let down_bias = vec![0.0f32; HIDDEN];
            let ff_f = linear_q(&h_q, &layer.down_proj, &down_bias).to_f32();
            captures.observe(li, "ffn_out", &ff_f);

            for i in 0..HIDDEN { hidden[i] += ff_f[i]; }
            captures.observe(li, "residual", &hidden);
        }
    }
}

/// SwiGLU FFN without any skip — full gate and up projections.
fn ffn_standard(x2: &QTensor, layer: &SmolLM2Layer, bias: &[f32]) -> Vec<f32> {
    let gate_f = linear_q(x2, &layer.gate_proj, bias).to_f32();
    let up_f   = linear_q(x2, &layer.up_proj,   bias).to_f32();
    let mut h = vec![0.0f32; FFN_DIM];
    for i in 0..FFN_DIM {
        h[i] = silu(gate_f[i]) * up_f[i];
    }
    h
}

/// LLaMA-style (non-interleaved) RoPE: rotate pairs (i, i+half) instead of (2i, 2i+1).
/// SmolLM2 uses rope_interleaved=false — first half and second half are the two components.
fn apply_rope_half(vec: &mut [f32], rope: &RopeTable, pos: usize) {
    let half = vec.len() / 2;
    // Access the precomputed cos/sin through the public apply method on a scratch buffer,
    // but we need the non-interleaved variant. We replicate the math directly here.
    // theta[i] = pos / rope_base^(2i / head_dim)
    let head_dim = vec.len();
    for i in 0..half {
        let theta = (pos as f32) / ROPE_BASE.powf(2.0 * i as f32 / head_dim as f32);
        let cos = theta.cos();
        let sin = theta.sin();
        let x0 = vec[i];
        let x1 = vec[i + half];
        vec[i]        = x0 * cos - x1 * sin;
        vec[i + half] = x0 * sin + x1 * cos;
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() { *v = (*v - max).exp(); sum += *v; }
    if sum > 0.0 { for v in x.iter_mut() { *v /= sum; } }
}
