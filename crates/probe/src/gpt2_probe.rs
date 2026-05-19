use std::path::Path;
use strategos_ops::qtensor::{QTensor, QWeight};
use strategos_ops::linear::linear_q;
use strategos_ops::lut::SwishLut;
use strategos_runtime::loader::ModelWeights;
use crate::capture::CaptureStore;

pub struct GPT2Probe {
    wte: Vec<f32>,
    wpe: Vec<f32>,
    layers: Vec<GPT2ProbeLayer>,
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub ffn_dim: usize,
    gelu_luts: Vec<SwishLut>,
}

struct GPT2ProbeLayer {
    ln_1_w: Vec<f32>,
    ln_1_b: Vec<f32>,
    c_attn_w: QWeight,
    c_attn_b: Vec<f32>,
    c_proj_w: QWeight,
    c_proj_b: Vec<f32>,
    ln_2_w: Vec<f32>,
    ln_2_b: Vec<f32>,
    c_fc_w: QWeight,
    c_fc_b: Vec<f32>,
    mlp_proj_w: QWeight,
    mlp_proj_b: Vec<f32>,
}

struct KVCache {
    k: Vec<Vec<Vec<Vec<f32>>>>,
    v: Vec<Vec<Vec<Vec<f32>>>>,
}

impl KVCache {
    fn new(n_layer: usize, n_head: usize) -> Self {
        let k = (0..n_layer).map(|_| (0..n_head).map(|_| Vec::new()).collect()).collect();
        let v = (0..n_layer).map(|_| (0..n_head).map(|_| Vec::new()).collect()).collect();
        Self { k, v }
    }
}

/// Transpose a matrix stored [rows, cols] → [cols, rows].
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(data.len(), rows * cols);
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

impl GPT2Probe {
    pub fn load(manifest_dir: &Path) -> Result<Self, String> {
        let weights = ModelWeights::load(manifest_dir)?;

        let get = |name: &str| -> Result<Vec<f32>, String> {
            weights.require(name).map(|s| s.to_vec())
        };

        // Exported without the "transformer." prefix.
        let wte = get("wte.weight")?;
        let wpe = get("wpe.weight")?;

        let mut layers = Vec::with_capacity(12);
        for i in 0..12 {
            let p = format!("h.{i}");
            // Conv1D weights are stored [in_dim, out_dim] (HuggingFace convention).
            // linear_q expects [out_dim, in_dim] row-major, so transpose before quantizing.
            let c_attn_raw = get(&format!("{p}.attn.c_attn.weight"))?; // [768, 2304]
            let c_proj_raw = get(&format!("{p}.attn.c_proj.weight"))?; // [768, 768]
            let c_fc_raw   = get(&format!("{p}.mlp.c_fc.weight"))?;    // [768, 3072]
            let c_mp_raw   = get(&format!("{p}.mlp.c_proj.weight"))?;  // [3072, 768]

            layers.push(GPT2ProbeLayer {
                ln_1_w: get(&format!("{p}.ln_1.weight"))?,
                ln_1_b: get(&format!("{p}.ln_1.bias"))?,
                c_attn_w: QWeight::from_f32(&transpose(&c_attn_raw, 768, 2304), 768, 2304),
                c_attn_b: get(&format!("{p}.attn.c_attn.bias"))?,
                c_proj_w: QWeight::from_f32(&transpose(&c_proj_raw, 768, 768), 768, 768),
                c_proj_b: get(&format!("{p}.attn.c_proj.bias"))?,
                ln_2_w: get(&format!("{p}.ln_2.weight"))?,
                ln_2_b: get(&format!("{p}.ln_2.bias"))?,
                c_fc_w:    QWeight::from_f32(&transpose(&c_fc_raw, 768, 3072), 768, 3072),
                c_fc_b: get(&format!("{p}.mlp.c_fc.bias"))?,
                mlp_proj_w: QWeight::from_f32(&transpose(&c_mp_raw, 3072, 768), 3072, 768),
                mlp_proj_b: get(&format!("{p}.mlp.c_proj.bias"))?,
            });
        }

        // 64 GELU LUTs covering the range of scales seen during inference.
        let gelu_luts: Vec<SwishLut> = (0..64)
            .map(|i| build_gelu_lut(1e-6 * (1.35f32).powi(i)))
            .collect();

        eprintln!("[gpt2-probe] loaded 12 layers");
        Ok(Self {
            wte, wpe, layers,
            vocab_size: 50257, n_embd: 768, n_head: 12, n_layer: 12, ffn_dim: 3072,
            gelu_luts,
        })
    }

    /// Run a token sequence through the model, capturing activations into `captures`.
    /// KV cache is reset every CONTEXT_LEN tokens to keep attention O(1) amortized.
    pub fn run_corpus(&self, tokens: &[u32], captures: &mut CaptureStore) {
        const CONTEXT_LEN: usize = 256;

        let d = self.n_embd;
        let n_head = self.n_head;
        let head_dim = d / n_head;
        let mut cache = KVCache::new(self.n_layer, n_head);

        for (pos, &tok) in tokens.iter().enumerate() {
            // Reset KV cache at context boundaries so attention stays bounded.
            if pos % CONTEXT_LEN == 0 {
                cache = KVCache::new(self.n_layer, n_head);
            }
            let ctx_pos = pos % CONTEXT_LEN;

            let tok_off = (tok as usize).min(self.vocab_size - 1) * d;
            let pos_off = ctx_pos.min(1023) * d;
            let mut hidden = vec![0.0f32; d];
            for i in 0..d {
                hidden[i] = self.wte[tok_off + i] + self.wpe[pos_off + i];
            }

            for (li, layer) in self.layers.iter().enumerate() {
                // Attention block
                let pre_attn = layer_norm(&hidden, &layer.ln_1_w, &layer.ln_1_b, 1e-5);
                captures.observe(li, "attn_pre_ln", &pre_attn);

                let x_q = QTensor::from_f32(&pre_attn);
                let qkv = linear_q(&x_q, &layer.c_attn_w, &layer.c_attn_b);
                let qkv_f = qkv.to_f32();

                captures.observe(li, "attn_q", &qkv_f[..d]);
                captures.observe(li, "attn_k", &qkv_f[d..2 * d]);
                captures.observe(li, "attn_v", &qkv_f[2 * d..]);

                // Update KV cache
                for h in 0..n_head {
                    let mut k = vec![0.0f32; head_dim];
                    let mut v = vec![0.0f32; head_dim];
                    for di in 0..head_dim {
                        k[di] = qkv_f[d + h * head_dim + di];
                        v[di] = qkv_f[2 * d + h * head_dim + di];
                    }
                    cache.k[li][h].push(k);
                    cache.v[li][h].push(v);
                }

                let seq_len = cache.k[li][0].len();
                let scale = (head_dim as f32).sqrt();
                let mut attn_out = vec![0.0f32; d];
                let mut head_scores_snap: Vec<Vec<f32>> = Vec::with_capacity(n_head);

                for h in 0..n_head {
                    let mut scores = vec![0.0f32; seq_len];
                    for ki in 0..seq_len {
                        let mut dot = 0.0f32;
                        for di in 0..head_dim {
                            dot += qkv_f[h * head_dim + di] * cache.k[li][h][ki][di];
                        }
                        scores[ki] = dot / scale;
                    }
                    softmax(&mut scores);
                    head_scores_snap.push(scores.clone());

                    for di in 0..head_dim {
                        let mut sum = 0.0f32;
                        for ki in 0..seq_len {
                            sum += scores[ki] * cache.v[li][h][ki][di];
                        }
                        attn_out[h * head_dim + di] = sum;
                    }
                }

                captures.observe_heads(li, "attn_weights", &head_scores_snap);
                captures.observe(li, "attn_out", &attn_out);

                // Attention residual
                let a_q = QTensor::from_f32(&attn_out);
                let proj = linear_q(&a_q, &layer.c_proj_w, &layer.c_proj_b);
                let proj_f = proj.to_f32();
                for i in 0..d { hidden[i] += proj_f[i]; }

                // FFN block
                let pre_ffn = layer_norm(&hidden, &layer.ln_2_w, &layer.ln_2_b, 1e-5);
                captures.observe(li, "ffn_pre_ln", &pre_ffn);

                let x2 = QTensor::from_f32(&pre_ffn);
                let mut h_ff = linear_q(&x2, &layer.c_fc_w, &layer.c_fc_b);

                let gelu = self.find_gelu(h_ff.scale);
                for v in h_ff.data.iter_mut() { *v = gelu.lookup(*v); }

                let ffn_hidden_f = h_ff.to_f32();
                captures.observe(li, "ffn_hidden", &ffn_hidden_f);

                let ff_out = linear_q(&h_ff, &layer.mlp_proj_w, &layer.mlp_proj_b);
                let ff_f = ff_out.to_f32();
                captures.observe(li, "ffn_out", &ff_f);

                for i in 0..d { hidden[i] += ff_f[i]; }
                captures.observe(li, "residual", &hidden);
            }

            captures.n_tokens += 1;

            if captures.n_tokens % 1000 == 0 {
                eprintln!("[probe] {} tokens processed", captures.n_tokens);
            }
        }
    }

    fn find_gelu(&self, scale: f32) -> &SwishLut {
        let mut best = 0;
        let mut best_d = f32::MAX;
        for (i, l) in self.gelu_luts.iter().enumerate() {
            let d = (l.scale - scale).abs();
            if d < best_d { best_d = d; best = i; }
        }
        &self.gelu_luts[best]
    }
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter().zip(w.iter()).zip(b.iter())
        .map(|((xi, wi), bi)| (xi - mean) * inv * wi + bi)
        .collect()
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() { *v = (*v - max).exp(); sum += *v; }
    if sum > 0.0 { for v in x.iter_mut() { *v /= sum; } }
}

fn build_gelu_lut(scale: f32) -> SwishLut {
    let range = 32767.0 * scale;
    let mut table = [0i16; 1024];
    for i in 0..1024 {
        let x = (i as f32 / 1023.0) * 2.0 * range - range;
        let inner = (2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
        let gelu = 0.5 * x * (1.0 + inner.tanh());
        table[i] = (gelu / scale).round().clamp(-32767.0, 32767.0) as i16;
    }
    SwishLut { table, scale }
}
