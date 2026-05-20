//! Grokking 1L Transformer — Spectral Fingerprint Extraction.
//!
//! Loads Neel Nanda's trained 1L modular addition model weights and extracts
//! the Fourier frequencies each MLP neuron responds to.
//!
//! Neel Nanda's key finding: the model grokks by learning to compute
//!   cos(2πk·a/p) × cos(2πk·b/p) − sin(2πk·a/p) × sin(2πk·b/p) = cos(2πk·(a+b)/p)
//! for a small set of "key frequencies" k. This binary identifies those k values
//! from the weight matrices — no training, just reading the learned structure.
//!
//! Algorithm:
//!   1. Load W_E, W_pos, W_Q/K/V, W_O, W_in, b_in, W_out, b_out, W_U
//!   2. Run all p² = 12769 (a,b) pairs through the forward pass
//!   3. For each of 512 MLP neurons, collect activations across all pairs
//!   4. Sum activations along antidiagonals: neuron_by_sum[j][c] = Σ_{a+b≡c mod p} act(a,b)
//!   5. DFT of neuron_by_sum[j] → dominant Fourier frequency k_j for each neuron
//!   6. Report the "key frequency" histogram and per-frequency neuron groups

use std::path::Path;
use std::f32::consts::PI;

const P: usize = 113;
const D: usize = 128;
const N_HEADS: usize = 4;
const HEAD_DIM: usize = 32;
const D_MLP: usize = 512;
const VOCAB: usize = 114;
const EQ_TOKEN: usize = P;

fn load_f32(dir: &Path, filename: &str) -> Vec<f32> {
    let path = dir.join(filename);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

struct Weights {
    w_e:   Vec<f32>,  // [d, vocab]
    w_pos: Vec<f32>,  // [3, d]
    w_q:   Vec<f32>,  // [n_heads, head_dim, d]
    w_k:   Vec<f32>,  // [n_heads, head_dim, d]
    w_v:   Vec<f32>,  // [n_heads, head_dim, d]
    w_o:   Vec<f32>,  // [d, d]
    w_in:  Vec<f32>,  // [d_mlp, d]
    b_in:  Vec<f32>,  // [d_mlp]
    w_out: Vec<f32>,  // [d, d_mlp]
    b_out: Vec<f32>,  // [d]
    w_u:   Vec<f32>,  // [d, vocab]
}

impl Weights {
    fn load(dir: &Path) -> Self {
        Self {
            w_e:   load_f32(dir, "embed_W_E.bin"),
            w_pos: load_f32(dir, "pos_embed_W_pos.bin"),
            w_q:   load_f32(dir, "blocks_0_attn_W_Q.bin"),
            w_k:   load_f32(dir, "blocks_0_attn_W_K.bin"),
            w_v:   load_f32(dir, "blocks_0_attn_W_V.bin"),
            w_o:   load_f32(dir, "blocks_0_attn_W_O.bin"),
            w_in:  load_f32(dir, "blocks_0_mlp_W_in.bin"),
            b_in:  load_f32(dir, "blocks_0_mlp_b_in.bin"),
            w_out: load_f32(dir, "blocks_0_mlp_W_out.bin"),
            b_out: load_f32(dir, "blocks_0_mlp_b_out.bin"),
            w_u:   load_f32(dir, "unembed_W_U.bin"),
        }
    }

    /// Returns (mlp_hidden [D_MLP], predicted_sum, correct_sum).
    fn forward(&self, a: usize, b: usize) -> (Vec<f32>, usize, usize) {
        // Embed tokens at positions 0 (a), 1 (b), 2 (=)
        let tokens = [a, b, EQ_TOKEN];
        let mut x = [[0.0f32; D]; 3];
        for pos in 0..3 {
            let tok = tokens[pos];
            for i in 0..D {
                x[pos][i] = self.w_e[i * VOCAB + tok] + self.w_pos[pos * D + i];
            }
        }

        // Attention (bidirectional, no layernorm, no causal mask)
        let scale = (HEAD_DIM as f32).sqrt();
        let mut attn_out = [[0.0f32; D]; 3];

        for h in 0..N_HEADS {
            let base = h * HEAD_DIM * D;
            let wq = &self.w_q[base..base + HEAD_DIM * D];
            let wk = &self.w_k[base..base + HEAD_DIM * D];
            let wv = &self.w_v[base..base + HEAD_DIM * D];

            let mut q = [[0.0f32; HEAD_DIM]; 3];
            let mut k = [[0.0f32; HEAD_DIM]; 3];
            let mut v = [[0.0f32; HEAD_DIM]; 3];
            for pos in 0..3 {
                for hd in 0..HEAD_DIM {
                    for i in 0..D {
                        q[pos][hd] += wq[hd * D + i] * x[pos][i];
                        k[pos][hd] += wk[hd * D + i] * x[pos][i];
                        v[pos][hd] += wv[hd * D + i] * x[pos][i];
                    }
                }
            }

            let mut scores = [[0.0f32; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let mut dot = 0.0f32;
                    for hd in 0..HEAD_DIM { dot += q[i][hd] * k[j][hd]; }
                    scores[i][j] = dot / scale;
                }
                softmax(&mut scores[i]);
            }

            for pos in 0..3 {
                let mut out_h = [0.0f32; HEAD_DIM];
                for j in 0..3 {
                    for hd in 0..HEAD_DIM { out_h[hd] += scores[pos][j] * v[j][hd]; }
                }
                for hd in 0..HEAD_DIM {
                    attn_out[pos][h * HEAD_DIM + hd] = out_h[hd];
                }
            }
        }

        // W_O projection on "=" position (pos=2), then residual
        let eq_attn = &attn_out[2];
        let mut proj = [0.0f32; D];
        for i in 0..D {
            let mut sum = 0.0f32;
            for j in 0..D { sum += self.w_o[i * D + j] * eq_attn[j]; }
            proj[i] = sum;
        }
        let mut h = [0.0f32; D];
        for i in 0..D { h[i] = x[2][i] + proj[i]; }

        // MLP: W_in @ h + b_in → ReLU → W_out @ hidden + b_out
        let mut hidden = vec![0.0f32; D_MLP];
        for j in 0..D_MLP {
            let mut v = self.b_in[j];
            for i in 0..D { v += self.w_in[j * D + i] * h[i]; }
            hidden[j] = v.max(0.0); // ReLU
        }

        let mut mlp_out = [0.0f32; D];
        for i in 0..D {
            let mut v = self.b_out[i];
            for j in 0..D_MLP { v += self.w_out[i * D_MLP + j] * hidden[j]; }
            mlp_out[i] = v;
        }

        let mut final_h = [0.0f32; D];
        for i in 0..D { final_h[i] = h[i] + mlp_out[i]; }

        // Unembed: logits over 0..P
        let mut logits = vec![0.0f32; P];
        for c in 0..P {
            let mut s = 0.0f32;
            for i in 0..D { s += final_h[i] * self.w_u[i * VOCAB + c]; }
            logits[c] = s;
        }
        let predicted = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        (hidden, predicted, (a + b) % P)
    }
}

/// Discrete Fourier Transform of a real signal — returns (magnitude, phase) for each freq k.
fn dft(signal: &[f32]) -> Vec<(f32, f32)> {
    let n = signal.len() as f32;
    (0..signal.len()).map(|k| {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (t, &x) in signal.iter().enumerate() {
            let angle = 2.0 * PI * k as f32 * t as f32 / n;
            re += x * angle.cos();
            im -= x * angle.sin();
        }
        ((re * re + im * im).sqrt() / n, im.atan2(re))
    }).collect()
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() { *v = (*v - max).exp(); sum += *v; }
    if sum > 0.0 { for v in x.iter_mut() { *v /= sum; } }
}

fn main() {
    // Accept path as first CLI arg or default to models/grokking
    let args: Vec<String> = std::env::args().collect();
    let model_dir = args.get(1).map(String::as_str).unwrap_or("models/grokking");
    let dir = Path::new(model_dir);

    println!("Grokking 1L Transformer — Spectral Fingerprint Extraction");
    println!("═══════════════════════════════════════════════════════════");
    println!("Model dir: {}", dir.display());
    println!("  p={P}  d_model={D}  d_mlp={D_MLP}  n_heads={N_HEADS}  head_dim={HEAD_DIM}");
    println!();

    let weights = Weights::load(dir);
    println!("Weights loaded.");
    println!();

    // ── Pass 1: run all pairs, collect accuracy + neuron activations ──────────
    println!("Running all {}×{} = {} pairs...", P, P, P * P);

    // neuron_by_sum[j][c] = total activation of neuron j when a+b≡c (mod p)
    let mut neuron_by_sum = vec![vec![0.0f32; P]; D_MLP];
    // count[c] = number of (a,b) pairs with a+b≡c
    let mut count = vec![0u32; P];
    let mut correct = 0usize;
    let total = P * P;

    for a in 0..P {
        for b in 0..P {
            let (hidden, predicted, expected) = weights.forward(a, b);
            if predicted == expected { correct += 1; }
            let c = (a + b) % P;
            count[c] += 1;
            for j in 0..D_MLP {
                neuron_by_sum[j][c] += hidden[j];
            }
        }
    }

    println!("  Done. Accuracy: {}/{} = {:.2}%", correct, total,
        100.0 * correct as f32 / total as f32);
    println!();

    // Normalize by count (each c appears exactly p times)
    for j in 0..D_MLP {
        for c in 0..P {
            if count[c] > 0 {
                neuron_by_sum[j][c] /= count[c] as f32;
            }
        }
    }

    // ── Pass 2: DFT each neuron → find dominant Fourier frequency ────────────
    println!("Extracting spectral fingerprints (DFT over {} sum values)...", P);

    // For each neuron: dominant_k = argmax |DFT[k]| for k in 1..P/2
    // k=0 is DC (mean activation), skip. k > P/2 are mirrors.
    let mut dominant_k = vec![0usize; D_MLP];
    let mut dominant_mag = vec![0.0f32; D_MLP];
    let mut neuron_mean = vec![0.0f32; D_MLP];

    for j in 0..D_MLP {
        neuron_mean[j] = neuron_by_sum[j].iter().sum::<f32>() / P as f32;
        let spectrum = dft(&neuron_by_sum[j]);
        // Search k=1..P/2 (skip DC, skip mirror half)
        let (k, mag) = spectrum[1..=P/2].iter().enumerate()
            .map(|(i, &(m, _))| (i + 1, m))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or((0, 0.0));
        dominant_k[j] = k;
        dominant_mag[j] = mag;
    }

    // ── Frequency histogram ───────────────────────────────────────────────────
    println!();
    println!("Dominant Fourier frequency histogram (top 15 frequencies by neuron count):");
    println!("  k   neurons  avg_mag");

    let mut freq_neurons: Vec<Vec<usize>> = vec![Vec::new(); P];
    for j in 0..D_MLP {
        freq_neurons[dominant_k[j]].push(j);
    }

    // Sort by neuron count descending
    let mut freq_counts: Vec<(usize, usize)> = freq_neurons.iter().enumerate()
        .map(|(k, ns)| (k, ns.len()))
        .filter(|&(k, n)| k > 0 && n > 0)
        .collect();
    freq_counts.sort_by(|a, b| b.1.cmp(&a.1));

    let mut key_freqs: Vec<usize> = Vec::new();
    for (k, n) in freq_counts.iter().take(15) {
        let avg_mag: f32 = freq_neurons[*k].iter()
            .map(|&j| dominant_mag[j])
            .sum::<f32>() / (*n).max(1) as f32;
        let bar: String = "#".repeat((*n).min(40));
        println!("  k={:3}  n={:3}  mag={:.4}  {}", k, n, avg_mag, bar);
        if *n >= 3 { key_freqs.push(*k); } // threshold: at least 3 neurons = meaningful
    }

    // ── Key frequency analysis ────────────────────────────────────────────────
    println!();
    println!("Key Fourier frequencies (≥3 neurons): {:?}", key_freqs);
    println!("  These are the k values where sin(2πk·a/p) × cos(2πk·b/p) terms");
    println!("  compute modular addition via the product-to-sum identity.");
    println!();

    // ── Neuron activation stats ───────────────────────────────────────────────
    let active_neurons = neuron_mean.iter().filter(|&&m| m > 0.01).count();
    let dead_neurons   = neuron_mean.iter().filter(|&&m| m <= 0.01).count();
    println!("Neuron activity:");
    println!("  Active (mean > 0.01):  {:3}", active_neurons);
    println!("  Dead   (mean ≤ 0.01):  {:3}", dead_neurons);
    println!();

    // ── Per key-frequency: which neurons, what magnitude ─────────────────────
    println!("Per key-frequency neuron groups (top 5 neurons by DFT magnitude):");
    for &k in key_freqs.iter().take(10) {
        let mut ns: Vec<(usize, f32)> = freq_neurons[k].iter()
            .map(|&j| (j, dominant_mag[j]))
            .collect();
        ns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top5: Vec<String> = ns.iter().take(5)
            .map(|&(j, m)| format!("n{}({:.3})", j, m))
            .collect();
        println!("  k={:3}: {} neurons  top: {}", k, ns.len(), top5.join(", "));
    }
    println!();

    // ── Cortex seeding summary ────────────────────────────────────────────────
    println!("TernSig Cortex Tier-1 Seeding Plan:");
    println!("  One tier-1 column per key frequency k × per phase (sin/cos) × per sum value c.");
    println!("  Total tier-1 columns needed: {} key_freqs × 2 phases × {} values = {}",
        key_freqs.len(), P, key_freqs.len() * 2 * P);
    println!();
    println!("  Each tier-1 column at freq(c) encodes:");
    println!("    'is_active when a+b ≡ c (mod p) at Fourier frequency k'");
    println!("  Column weight seeding: cell.m ∝ |DFT_k[neuron_j]| for the assigned frequency");
    println!();
    println!("Next step: implement tier-1 conjunction detection");
    println!("  A tier-1 column must receive from BOTH freq(a) AND freq(b) tier-0 columns.");
    println!("  This requires extending TernSigColumn to support multi-source gating.");
    println!("  The spectral fingerprint above provides the target weights.");
}
