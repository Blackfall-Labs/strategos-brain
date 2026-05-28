use std::path::Path;
use rayon::prelude::*;
use strategos_ops::qtensor::{QTensor, QWeight};
use strategos_ops::linear::linear_q;
use strategos_ops::rmsnorm::{RmsNormWeights, rms_norm};
use strategos_ops::rope::RopeTable;
use strategos_runtime::loader::ModelWeights;
use crate::capture::CaptureStore;
use crate::triggers::TriggerStore;

// Qwen2.5-0.5B architecture constants (config.json)
const HIDDEN:       usize = 896;
const Q_HEADS:      usize = 14;
const KV_HEADS:     usize = 2;
const HEAD_DIM:     usize = 64;
const INTERMEDIATE: usize = 4864;
const VOCAB:        usize = 151936;
const N_LAYERS:     usize = 24;
const RMS_EPS:      f32   = 1e-6;
const ROPE_THETA:   f32   = 1_000_000.0;
const MAX_SEQ:      usize = 512;
const KV_GROUP:     usize = Q_HEADS / KV_HEADS;

pub struct Qwen25Probe {
    embed:     Vec<f32>,  // [VOCAB * HIDDEN]
    layers:    Vec<Qwen25Layer>,
    rope:      RopeTable,
    pub n_layer:  usize,
    pub n_head:   usize,
    pub ffn_dim:  usize,
}

struct Qwen25Layer {
    input_norm:     RmsNormWeights,
    q_proj:         QWeight,  // [Q_HEADS*HEAD_DIM, HIDDEN]
    k_proj:         QWeight,  // [KV_HEADS*HEAD_DIM, HIDDEN]
    v_proj:         QWeight,  // [KV_HEADS*HEAD_DIM, HIDDEN]
    o_proj:         QWeight,  // [HIDDEN, Q_HEADS*HEAD_DIM]
    post_attn_norm: RmsNormWeights,
    gate_proj:      QWeight,  // [INTERMEDIATE, HIDDEN]
    up_proj:        QWeight,  // [INTERMEDIATE, HIDDEN]
    down_proj:      QWeight,  // [HIDDEN, INTERMEDIATE]
}

struct KVCache {
    k: Vec<Vec<Vec<Vec<f32>>>>,  // [layer][kv_head][seq][head_dim]
    v: Vec<Vec<Vec<Vec<f32>>>>,
}

impl KVCache {
    fn new() -> Self {
        let k = (0..N_LAYERS).map(|_| (0..KV_HEADS).map(|_| Vec::new()).collect()).collect();
        let v = (0..N_LAYERS).map(|_| (0..KV_HEADS).map(|_| Vec::new()).collect()).collect();
        Self { k, v }
    }
}

impl Qwen25Probe {
    pub fn load(manifest_dir: &Path) -> Result<Self, String> {
        let weights = ModelWeights::load(manifest_dir)?;
        let get = |name: &str| -> Result<Vec<f32>, String> {
            weights.require(name).map(|s| s.to_vec())
        };

        let embed = get("embed_tokens.weight")?;

        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            // Exporter aliases strip "model." prefix → "layers.{i}.*"
            let p = format!("layers.{i}");
            layers.push(Qwen25Layer {
                input_norm:     rmsnorm(&get(&format!("{p}.input_layernorm.weight"))?)?,
                q_proj:         qweight(&get(&format!("{p}.self_attn.q_proj.weight"))?,  HIDDEN, Q_HEADS * HEAD_DIM)?,
                k_proj:         qweight(&get(&format!("{p}.self_attn.k_proj.weight"))?,  HIDDEN, KV_HEADS * HEAD_DIM)?,
                v_proj:         qweight(&get(&format!("{p}.self_attn.v_proj.weight"))?,  HIDDEN, KV_HEADS * HEAD_DIM)?,
                o_proj:         qweight(&get(&format!("{p}.self_attn.o_proj.weight"))?,  Q_HEADS * HEAD_DIM, HIDDEN)?,
                post_attn_norm: rmsnorm(&get(&format!("{p}.post_attention_layernorm.weight"))?)?,
                gate_proj:      qweight(&get(&format!("{p}.mlp.gate_proj.weight"))?,     HIDDEN, INTERMEDIATE)?,
                up_proj:        qweight(&get(&format!("{p}.mlp.up_proj.weight"))?,       HIDDEN, INTERMEDIATE)?,
                down_proj:      qweight(&get(&format!("{p}.mlp.down_proj.weight"))?,     INTERMEDIATE, HIDDEN)?,
            });

            if (i + 1) % 8 == 0 {
                eprintln!("[qwen25-probe] loaded layer {}/{}", i + 1, N_LAYERS);
            }
        }

        let rope = RopeTable::new(HEAD_DIM, MAX_SEQ, ROPE_THETA);
        eprintln!("[qwen25-probe] loaded {} layers", N_LAYERS);

        Ok(Self { embed, layers, rope, n_layer: N_LAYERS, n_head: Q_HEADS, ffn_dim: INTERMEDIATE })
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
            let pre_attn = rms_norm(&hidden, &layer.input_norm);
            captures.observe(li, "attn_pre_rms", &pre_attn);

            let x_q = QTensor::from_f32(&pre_attn);
            let q_raw = linear_q(&x_q, &layer.q_proj, &vec![0.0f32; Q_HEADS * HEAD_DIM]).to_f32();
            let k_raw = linear_q(&x_q, &layer.k_proj, &vec![0.0f32; KV_HEADS * HEAD_DIM]).to_f32();
            let v_raw = linear_q(&x_q, &layer.v_proj, &vec![0.0f32; KV_HEADS * HEAD_DIM]).to_f32();

            captures.observe(li, "attn_q", &q_raw);
            captures.observe(li, "attn_k", &k_raw);
            captures.observe(li, "attn_v", &v_raw);

            // Qwen2.5 has no QK-norm — apply RoPE directly to Q and K heads.
            let mut q = q_raw;
            let mut k = k_raw;
            for h in 0..Q_HEADS {
                self.rope.apply(&mut q[h * HEAD_DIM..(h + 1) * HEAD_DIM], ctx_pos);
            }
            for h in 0..KV_HEADS {
                self.rope.apply(&mut k[h * HEAD_DIM..(h + 1) * HEAD_DIM], ctx_pos);
            }

            for kv_h in 0..KV_HEADS {
                cache.k[li][kv_h].push(k[kv_h * HEAD_DIM..(kv_h + 1) * HEAD_DIM].to_vec());
                cache.v[li][kv_h].push(v_raw[kv_h * HEAD_DIM..(kv_h + 1) * HEAD_DIM].to_vec());
            }

            let seq_len = cache.k[li][0].len();
            let scale = (HEAD_DIM as f32).sqrt();
            let mut attn_out = vec![0.0f32; HIDDEN];
            let mut head_scores_snap: Vec<Vec<f32>> = Vec::with_capacity(Q_HEADS);

            for q_h in 0..Q_HEADS {
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
            let proj = linear_q(&a_q, &layer.o_proj, &vec![0.0f32; HIDDEN]).to_f32();
            for i in 0..HIDDEN { hidden[i] += proj[i]; }

            // --- FFN (SwiGLU) ---
            let pre_ffn = rms_norm(&hidden, &layer.post_attn_norm);
            captures.observe(li, "ffn_pre_rms", &pre_ffn);

            let x2 = QTensor::from_f32(&pre_ffn);
            let gate = linear_q(&x2, &layer.gate_proj, &vec![0.0f32; INTERMEDIATE]).to_f32();
            let up   = linear_q(&x2, &layer.up_proj,   &vec![0.0f32; INTERMEDIATE]).to_f32();

            let mut ffn_hidden = vec![0.0f32; INTERMEDIATE];
            for i in 0..INTERMEDIATE {
                ffn_hidden[i] = silu(gate[i]) * up[i];
            }
            captures.observe(li, "ffn_hidden", &ffn_hidden);
            if let Some((current_tok, ref mut trig)) = trigger_hook {
                trig.observe(li, &ffn_hidden, current_tok);
            }

            let h_q = QTensor::from_f32(&ffn_hidden);
            let ff = linear_q(&h_q, &layer.down_proj, &vec![0.0f32; HIDDEN]).to_f32();
            captures.observe(li, "ffn_out", &ff);

            for i in 0..HIDDEN { hidden[i] += ff[i]; }
            captures.observe(li, "residual", &hidden);
        }
    }
}

fn qweight(data: &[f32], in_dim: usize, out_dim: usize) -> Result<QWeight, String> {
    if data.len() != in_dim * out_dim {
        return Err(format!("qweight: expected {}×{}={}, got {}", out_dim, in_dim, in_dim * out_dim, data.len()));
    }
    Ok(QWeight::from_f32(data, in_dim, out_dim))
}

fn rmsnorm(data: &[f32]) -> Result<RmsNormWeights, String> {
    Ok(RmsNormWeights::from_f32(data, RMS_EPS))
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
