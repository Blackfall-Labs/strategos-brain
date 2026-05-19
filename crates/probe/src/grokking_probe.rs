use std::path::Path;
use strategos_ops::qtensor::{QTensor, QWeight};
use strategos_ops::linear::linear_q;
use strategos_runtime::loader::ModelWeights;
use crate::capture::CaptureStore;

// Architecture constants from the checkpoint.
const P: usize = 113;   // modulus
const D: usize = 128;   // d_model
const N_HEADS: usize = 4;
const HEAD_DIM: usize = 32; // D / N_HEADS
const D_MLP: usize = 512;
const VOCAB: usize = 114;   // 0..112 numbers + 113 for "="
const EQ_TOKEN: usize = P;  // index of the "=" token

pub struct GrokkingProbe {
    w_e:     Vec<f32>,   // [vocab, d]     token embeddings
    w_pos:   Vec<f32>,   // [3, d]         position embeddings
    w_k:     Vec<f32>,   // [n_heads, head_dim, d]
    w_q:     Vec<f32>,   // [n_heads, head_dim, d]
    w_v:     Vec<f32>,   // [n_heads, head_dim, d]
    w_o:     QWeight,    // [d, d]
    w_in:    QWeight,    // [d_mlp, d]
    b_in:    Vec<f32>,
    w_out:   QWeight,    // [d, d_mlp]
    b_out:   Vec<f32>,
    w_u:     Vec<f32>,   // [d, vocab]     unembed
}

impl GrokkingProbe {
    pub fn load(manifest_dir: &Path) -> Result<Self, String> {
        let weights = ModelWeights::load(manifest_dir)?;
        let get = |name: &str| -> Result<Vec<f32>, String> {
            weights.require(name).map(|s| s.to_vec())
        };

        Ok(Self {
            w_e:   get("embed.W_E")?,
            w_pos: get("pos_embed.W_pos")?,
            w_k:   get("blocks.0.attn.W_K")?,
            w_q:   get("blocks.0.attn.W_Q")?,
            w_v:   get("blocks.0.attn.W_V")?,
            // W_O: [128, 128] → out_dim=128, in_dim=128
            w_o:   QWeight::from_f32(&get("blocks.0.attn.W_O")?, D, D),
            // W_in: [512, 128] → out_dim=512, in_dim=128
            w_in:  QWeight::from_f32(&get("blocks.0.mlp.W_in")?, D, D_MLP),
            b_in:  get("blocks.0.mlp.b_in")?,
            // W_out: [128, 512] → out_dim=128, in_dim=512
            w_out: QWeight::from_f32(&get("blocks.0.mlp.W_out")?, D_MLP, D),
            b_out: get("blocks.0.mlp.b_out")?,
            w_u:   get("unembed.W_U")?,
        })
    }

    /// Run all (a, b) pairs through the model, capturing activations.
    /// This is the complete dataset: P*P = 12769 pairs.
    pub fn run_all_pairs(&self, captures: &mut CaptureStore) {
        for a in 0..P {
            for b in 0..P {
                self.forward(a, b, captures);
                captures.n_tokens += 1;
            }
        }
        eprintln!("[grokking-probe] {} pairs processed", captures.n_tokens);
    }

    fn forward(&self, a: usize, b: usize, captures: &mut CaptureStore) {
        // Embed: W_E is [d_model, vocab] so W_E[i, tok] = w_e[i * VOCAB + tok]
        let mut x = [[0.0f32; D]; 3];  // [seq=3, d]
        let tokens = [a, b, EQ_TOKEN];
        for pos in 0..3 {
            let tok = tokens[pos];
            for i in 0..D {
                x[pos][i] = self.w_e[i * VOCAB + tok] + self.w_pos[pos * D + i];
            }
        }

        captures.observe(0, "embed", &x[0]);
        captures.observe(0, "embed_b", &x[1]);
        captures.observe(0, "embed_eq", &x[2]);

        // Attention — no LayerNorm in this model.
        // W_K/Q/V: [n_heads, head_dim, d_model]
        // For each head h: Q[h] = W_Q[h] @ x[pos]  (head_dim vector per position)
        let scale = (HEAD_DIM as f32).sqrt();
        let mut attn_out = [[0.0f32; D]; 3];
        let mut head_scores_snap: Vec<Vec<f32>> = Vec::with_capacity(N_HEADS);

        for h in 0..N_HEADS {
            let wq_h = &self.w_q[h * HEAD_DIM * D..(h + 1) * HEAD_DIM * D];
            let wk_h = &self.w_k[h * HEAD_DIM * D..(h + 1) * HEAD_DIM * D];
            let wv_h = &self.w_v[h * HEAD_DIM * D..(h + 1) * HEAD_DIM * D];

            // Q, K, V for all 3 positions: [3, head_dim]
            let mut q = [[0.0f32; HEAD_DIM]; 3];
            let mut k = [[0.0f32; HEAD_DIM]; 3];
            let mut v = [[0.0f32; HEAD_DIM]; 3];
            for pos in 0..3 {
                for hd in 0..HEAD_DIM {
                    for i in 0..D {
                        q[pos][hd] += wq_h[hd * D + i] * x[pos][i];
                        k[pos][hd] += wk_h[hd * D + i] * x[pos][i];
                        v[pos][hd] += wv_h[hd * D + i] * x[pos][i];
                    }
                }
            }

            // Attention scores [3, 3] — full (no causal mask: Neel's model is bidirectional)
            let mut scores = [[0.0f32; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let mut dot = 0.0f32;
                    for hd in 0..HEAD_DIM { dot += q[i][hd] * k[j][hd]; }
                    scores[i][j] = dot / scale;
                }
                softmax(&mut scores[i]);
            }

            // Capture per-head scores for "=" position (index 2) — that's where we read the answer
            head_scores_snap.push(scores[2].to_vec());

            // Weighted sum of values
            let mut out_h = [[0.0f32; HEAD_DIM]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    for hd in 0..HEAD_DIM {
                        out_h[i][hd] += scores[i][j] * v[j][hd];
                    }
                }
            }

            // Concatenate heads into d-dim vector per position
            for pos in 0..3 {
                for hd in 0..HEAD_DIM {
                    attn_out[pos][h * HEAD_DIM + hd] = out_h[pos][hd];
                }
            }
        }

        // Capture attention outputs and head scores at the "=" position
        captures.observe(0, "attn_q", &{
            let mut flat = vec![0.0f32; N_HEADS * HEAD_DIM];
            for h in 0..N_HEADS {
                let wq_h = &self.w_q[h * HEAD_DIM * D..(h + 1) * HEAD_DIM * D];
                for hd in 0..HEAD_DIM {
                    for i in 0..D { flat[h * HEAD_DIM + hd] += wq_h[hd * D + i] * x[2][i]; }
                }
            }
            flat
        });
        captures.observe_heads(0, "attn_weights", &head_scores_snap);
        captures.observe(0, "attn_out", &attn_out[2]);

        // W_O projection: [d, d] applied to attn_out at "=" position
        let attn_q = QTensor::from_f32(&attn_out[2]);
        let proj = linear_q(&attn_q, &self.w_o, &vec![0.0f32; D]);
        let proj_f = proj.to_f32();

        // Residual
        let mut h_res = [0.0f32; D];
        for i in 0..D { h_res[i] = x[2][i] + proj_f[i]; }
        captures.observe(0, "post_attn_residual", &h_res);

        // MLP: W_in [d_mlp, d], ReLU, W_out [d, d_mlp]
        let h_q = QTensor::from_f32(&h_res);
        let mut mlp_hidden = linear_q(&h_q, &self.w_in, &self.b_in);

        // ReLU in-place
        for v in mlp_hidden.data.iter_mut() { if *v < 0 { *v = 0; } }
        let mlp_hidden_f = mlp_hidden.to_f32();
        captures.observe(0, "mlp_hidden", &mlp_hidden_f);

        let mlp_out = linear_q(&mlp_hidden, &self.w_out, &self.b_out);
        let mlp_out_f = mlp_out.to_f32();
        captures.observe(0, "mlp_out", &mlp_out_f);

        // Final residual
        let mut final_h = [0.0f32; D];
        for i in 0..D { final_h[i] = h_res[i] + mlp_out_f[i]; }
        captures.observe(0, "residual", &final_h);

        // Unembed logits: W_U [d, vocab=114]
        // We only care about logits 0..P (the answer space)
        let mut logits = vec![0.0f32; P];
        for v in 0..P {
            let mut sum = 0.0f32;
            for i in 0..D { sum += final_h[i] * self.w_u[i * VOCAB + v]; }
            logits[v] = sum;
        }
        let predicted = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let correct = (a + b) % P;
        captures.observe(0, "logits", &logits);

        // Record correctness as a 1/0 signal in a dedicated capture
        captures.observe(0, "correct", &[if predicted == correct { 1.0 } else { 0.0 }]);
    }
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() { *v = (*v - max).exp(); sum += *v; }
    if sum > 0.0 { for v in x.iter_mut() { *v /= sum; } }
}
