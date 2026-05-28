//! GPT-2 Small — integer matmul via SignalLinear, float activations.
//!
//! Activations stay as f32 throughout. Each weight matmul encodes the input
//! as Signals, calls forward(), then recovers float via:
//!   float_out[i] = acc[i] × weight_scale × input_scale / M_MAX²
//!
//! Integer exceptions: softmax, GELU, layernorm (these stay float).
//! The integer path replaces every matmul with sign×M accumulation in i64.
//!
//! Validates against a pure-float reference. Target: top-1 match, delta < 0.10.

use std::collections::HashMap;
use std::time::Instant;

use ternsig_cortex::linear::SignalLinear;
use signode::{Signal, K_MAX, M_MAX};

const N_VOCAB:  usize = 50257;
const N_EMBD:   usize = 768;
const N_HEAD:   usize = 12;
const N_LAYER:  usize = 12;
const HEAD_DIM: usize = N_EMBD / N_HEAD;
const FFN_DIM:  usize = N_EMBD * 4;

const MODEL_DIR: &str = "E:/repos/blackfall-labs/strategos-brain/models/gpt2";

// ── Manifest loader ───────────────────────────────────────────────────────────

struct ManifestEntry { file: String }

fn load_manifest(dir: &str) -> HashMap<String, ManifestEntry> {
    let text = std::fs::read_to_string(format!("{dir}/manifest.json"))
        .unwrap_or_else(|e| panic!("manifest: {e}"));
    let v: serde_json::Value = serde_json::from_str(&text).expect("manifest JSON");
    v.as_array().unwrap().iter().map(|e| {
        (e["name"].as_str().unwrap().to_string(),
         ManifestEntry { file: e["file"].as_str().unwrap().to_string() })
    }).collect()
}

fn load_f32(dir: &str, m: &HashMap<String, ManifestEntry>, name: &str) -> Vec<f32> {
    let entry = m.get(name).unwrap_or_else(|| panic!("weight not found: {name}"));
    let bytes = std::fs::read(format!("{dir}/{}", entry.file))
        .unwrap_or_else(|e| panic!("read {}: {e}", entry.file));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

fn p95(w: &[f32]) -> f32 {
    if w.is_empty() { return 1.0; }
    let mut a: Vec<f32> = w.iter().map(|v| v.abs()).collect();
    a.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
    let p = a[((a.len() as f32 * 0.95) as usize).min(a.len() - 1)];
    if p < f32::EPSILON { 1.0 } else { p }
}

// ── Integer matmul helper ─────────────────────────────────────────────────────
//
// Encodes x_f as Signals, calls forward, recovers float.
// Correctly accounts for the input normalization scale.

fn imatmul(lin: &SignalLinear, x_f: &[f32]) -> Vec<f32> {
    let x_scale = x_f.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(f32::EPSILON);
    let inv = 1.0 / x_scale;
    let x_sig: Vec<Signal> = x_f.iter().map(|&v| {
        let p: i16 = if v >= 0.0 { 1000 } else { -1000 };
        let m = (v.abs() * inv * M_MAX as f32).min(M_MAX as f32) as u16;
        Signal { p, m, k: K_MAX }
    }).collect();
    let acc = lin.forward(&x_sig, 0);
    let recovery = lin.weight_scale() * x_scale / (M_MAX as f32 * M_MAX as f32);
    acc.iter().map(|&a| a as f32 * recovery).collect()
}

// Same but returns raw i64 (for lm_head — comparison of relative order only).
fn imatmul_raw(lin: &SignalLinear, x_f: &[f32]) -> Vec<i64> {
    let x_scale = x_f.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(f32::EPSILON);
    let inv = 1.0 / x_scale;
    let x_sig: Vec<Signal> = x_f.iter().map(|&v| {
        let p: i16 = if v >= 0.0 { 1000 } else { -1000 };
        let m = (v.abs() * inv * M_MAX as f32).min(M_MAX as f32) as u16;
        Signal { p, m, k: K_MAX }
    }).collect();
    lin.forward(&x_sig, 0)
}

// Embedding row decoded to float.
fn emb_row(lin: &SignalLinear, idx: usize) -> Vec<f32> {
    let sc = lin.weight_scale() / M_MAX as f32;
    lin.row(idx).iter().map(|s| s.p.signum() as f32 * s.m as f32 * sc).collect()
}

// ── Integer model ─────────────────────────────────────────────────────────────

struct Layer {
    ln_1_w: Vec<f32>, ln_1_b: Vec<f32>,
    c_attn:  SignalLinear,  c_attn_b: Vec<f32>,
    c_proj:  SignalLinear,  c_proj_b: Vec<f32>,
    ln_2_w: Vec<f32>, ln_2_b: Vec<f32>,
    c_fc:    SignalLinear,  c_fc_b:   Vec<f32>,
    c_proj2: SignalLinear,  c_proj2_b: Vec<f32>,
}

struct Gpt2Int {
    wte: SignalLinear,
    wpe: SignalLinear,
    layers: Vec<Layer>,
    ln_f_w: Vec<f32>, ln_f_b: Vec<f32>,
}

impl Gpt2Int {
    fn load(dir: &str, m: &HashMap<String, ManifestEntry>) -> Self {
        let lf = |name: &str| load_f32(dir, m, name);
        macro_rules! enc_t {
            ($name:expr, $in:expr, $out:expr) => {{
                let w = lf($name); let sc = p95(&w);
                SignalLinear::from_weights_t(&w, $in, $out, sc)
            }};
        }

        print!("  wte... ");
        let wte_w = lf("wte.weight"); let sc = p95(&wte_w);
        let wte = SignalLinear::from_weights(&wte_w, N_VOCAB, N_EMBD, sc);

        print!("wpe... ");
        let wpe_w = lf("wpe.weight"); let sc = p95(&wpe_w);
        let wpe = SignalLinear::from_weights(&wpe_w, 1024, N_EMBD, sc);

        let layers = (0..N_LAYER).map(|l| {
            print!("L{l}... "); std::io::Write::flush(&mut std::io::stdout()).ok();
            Layer {
                ln_1_w:    lf(&format!("h.{l}.ln_1.weight")),
                ln_1_b:    lf(&format!("h.{l}.ln_1.bias")),
                c_attn:    enc_t!(&format!("h.{l}.attn.c_attn.weight"), N_EMBD, 3*N_EMBD),
                c_attn_b:  lf(&format!("h.{l}.attn.c_attn.bias")),
                c_proj:    enc_t!(&format!("h.{l}.attn.c_proj.weight"), N_EMBD, N_EMBD),
                c_proj_b:  lf(&format!("h.{l}.attn.c_proj.bias")),
                ln_2_w:    lf(&format!("h.{l}.ln_2.weight")),
                ln_2_b:    lf(&format!("h.{l}.ln_2.bias")),
                c_fc:      enc_t!(&format!("h.{l}.mlp.c_fc.weight"),   N_EMBD, FFN_DIM),
                c_fc_b:    lf(&format!("h.{l}.mlp.c_fc.bias")),
                c_proj2:   enc_t!(&format!("h.{l}.mlp.c_proj.weight"), FFN_DIM, N_EMBD),
                c_proj2_b: lf(&format!("h.{l}.mlp.c_proj.bias")),
            }
        }).collect();

        println!("done.");
        Gpt2Int { wte, wpe, layers, ln_f_w: lf("ln_f.weight"), ln_f_b: lf("ln_f.bias") }
    }

    fn forward(&self, tokens: &[u32]) -> Vec<i64> {
        let t_len = tokens.len();

        // Embeddings: decode from SignalLinear rows → f32
        let mut h: Vec<Vec<f32>> = tokens.iter().enumerate().map(|(i, &t)| {
            let tok = emb_row(&self.wte, t as usize);
            let pos = emb_row(&self.wpe, i);
            tok.iter().zip(&pos).map(|(a, b)| a + b).collect()
        }).collect();

        for layer in &self.layers {
            // Pre-attn layernorm
            let h_norm: Vec<Vec<f32>> = h.iter()
                .map(|x| layernorm(x, &layer.ln_1_w, &layer.ln_1_b))
                .collect();

            // QKV via integer matmul
            let qkv: Vec<Vec<f32>> = h_norm.iter().map(|x| {
                let mut out = imatmul(&layer.c_attn, x);
                for (o, &b) in out.iter_mut().zip(&layer.c_attn_b) { *o += b; }
                out
            }).collect();

            let q:   Vec<Vec<Vec<f32>>> = qkv.iter().map(|v| heads(&v[..N_EMBD])).collect();
            let k:   Vec<Vec<Vec<f32>>> = qkv.iter().map(|v| heads(&v[N_EMBD..2*N_EMBD])).collect();
            let v_h: Vec<Vec<Vec<f32>>> = qkv.iter().map(|v| heads(&v[2*N_EMBD..])).collect();

            // Causal multi-head attention
            let mut ao: Vec<Vec<f32>> = vec![vec![0.0f32; N_EMBD]; t_len];
            for hi in 0..N_HEAD {
                for t in 0..t_len {
                    let sc = (HEAD_DIM as f32).sqrt();
                    let scores: Vec<f32> = (0..=t).map(|s| {
                        q[t][hi].iter().zip(&k[s][hi]).map(|(a, b)| a * b).sum::<f32>() / sc
                    }).collect();
                    let wts = softmax(&scores);
                    let off = hi * HEAD_DIM;
                    for d in 0..HEAD_DIM {
                        ao[t][off + d] = (0..=t).map(|s| wts[s] * v_h[s][hi][d]).sum();
                    }
                }
            }

            // Output proj + residual
            for t in 0..t_len {
                let proj = imatmul(&layer.c_proj, &ao[t]);
                for j in 0..N_EMBD {
                    h[t][j] += proj[j] + layer.c_proj_b[j];
                }
            }

            // Pre-FFN layernorm
            let h2: Vec<Vec<f32>> = h.iter()
                .map(|x| layernorm(x, &layer.ln_2_w, &layer.ln_2_b))
                .collect();

            // FFN
            for t in 0..t_len {
                let fc = {
                    let mut v = imatmul(&layer.c_fc, &h2[t]);
                    for (o, &b) in v.iter_mut().zip(&layer.c_fc_b) { *o += b; }
                    v.iter().map(|&x| gelu(x)).collect::<Vec<f32>>()
                };
                let p2 = imatmul(&layer.c_proj2, &fc);
                for j in 0..N_EMBD {
                    h[t][j] += p2[j] + layer.c_proj2_b[j];
                }
            }
        }

        // Final layernorm + lm_head (tied wte)
        let h_last = layernorm(&h[t_len - 1], &self.ln_f_w, &self.ln_f_b);
        imatmul_raw(&self.wte, &h_last)
    }
}

// ── Float reference ───────────────────────────────────────────────────────────

fn float_forward(dir: &str, m: &HashMap<String, ManifestEntry>, tokens: &[u32]) -> Vec<f32> {
    let lf = |name: &str| load_f32(dir, m, name);
    let t_len = tokens.len();
    let wte_w = lf("wte.weight");
    let wpe_w = lf("wpe.weight");

    let mut h: Vec<Vec<f32>> = tokens.iter().enumerate().map(|(i, &t)| {
        let te = &wte_w[t as usize * N_EMBD..(t as usize + 1) * N_EMBD];
        let pe = &wpe_w[i * N_EMBD..(i + 1) * N_EMBD];
        te.iter().zip(pe).map(|(a, b)| a + b).collect()
    }).collect();

    for l in 0..N_LAYER {
        let ln1_w = lf(&format!("h.{l}.ln_1.weight"));
        let ln1_b = lf(&format!("h.{l}.ln_1.bias"));
        let ca_w  = lf(&format!("h.{l}.attn.c_attn.weight"));  // [N_EMBD, 3*N_EMBD]
        let ca_b  = lf(&format!("h.{l}.attn.c_attn.bias"));
        let cp_w  = lf(&format!("h.{l}.attn.c_proj.weight"));  // [N_EMBD, N_EMBD]
        let cp_b  = lf(&format!("h.{l}.attn.c_proj.bias"));
        let ln2_w = lf(&format!("h.{l}.ln_2.weight"));
        let ln2_b = lf(&format!("h.{l}.ln_2.bias"));
        let fc_w  = lf(&format!("h.{l}.mlp.c_fc.weight"));     // [N_EMBD, FFN_DIM]
        let fc_b  = lf(&format!("h.{l}.mlp.c_fc.bias"));
        let p2_w  = lf(&format!("h.{l}.mlp.c_proj.weight"));   // [FFN_DIM, N_EMBD]
        let p2_b  = lf(&format!("h.{l}.mlp.c_proj.bias"));

        let h_norm: Vec<Vec<f32>> = h.iter().map(|x| layernorm(x, &ln1_w, &ln1_b)).collect();

        // Conv1D style: weight [in, out] → out[i] = Σ_j w[j,i] * x[j]
        let qkv: Vec<Vec<f32>> = h_norm.iter().map(|x| {
            let mut out = ca_b.clone();
            for i in 0..3*N_EMBD {
                for j in 0..N_EMBD { out[i] += ca_w[j * 3*N_EMBD + i] * x[j]; }
            }
            out
        }).collect();

        let q:  Vec<Vec<Vec<f32>>> = qkv.iter().map(|v| heads(&v[..N_EMBD])).collect();
        let k:  Vec<Vec<Vec<f32>>> = qkv.iter().map(|v| heads(&v[N_EMBD..2*N_EMBD])).collect();
        let vv: Vec<Vec<Vec<f32>>> = qkv.iter().map(|v| heads(&v[2*N_EMBD..])).collect();

        let mut ao: Vec<Vec<f32>> = vec![vec![0.0; N_EMBD]; t_len];
        for hi in 0..N_HEAD {
            for t in 0..t_len {
                let sc = (HEAD_DIM as f32).sqrt();
                let scores: Vec<f32> = (0..=t).map(|s|
                    q[t][hi].iter().zip(&k[s][hi]).map(|(a,b)| a*b).sum::<f32>() / sc
                ).collect();
                let wts = softmax(&scores);
                let off = hi * HEAD_DIM;
                for d in 0..HEAD_DIM {
                    ao[t][off + d] = (0..=t).map(|s| wts[s] * vv[s][hi][d]).sum();
                }
            }
        }

        for t in 0..t_len {
            let mut proj = cp_b.clone();
            for i in 0..N_EMBD {
                for j in 0..N_EMBD { proj[i] += cp_w[j * N_EMBD + i] * ao[t][j]; }
            }
            for i in 0..N_EMBD { h[t][i] += proj[i]; }
        }

        let h2: Vec<Vec<f32>> = h.iter().map(|x| layernorm(x, &ln2_w, &ln2_b)).collect();

        for t in 0..t_len {
            let mut fc = fc_b.clone();
            for i in 0..FFN_DIM {
                for j in 0..N_EMBD { fc[i] += fc_w[j * FFN_DIM + i] * h2[t][j]; }
            }
            let fc_g: Vec<f32> = fc.iter().map(|&x| gelu(x)).collect();
            let mut p2 = p2_b.clone();
            for i in 0..N_EMBD {
                for j in 0..FFN_DIM { p2[i] += p2_w[j * N_EMBD + i] * fc_g[j]; }
            }
            for i in 0..N_EMBD { h[t][i] += p2[i]; }
        }
    }

    let ln_fw = lf("ln_f.weight");
    let ln_fb = lf("ln_f.bias");
    let h_last = layernorm(&h[t_len - 1], &ln_fw, &ln_fb);
    let wte_w2 = lf("wte.weight");
    (0..N_VOCAB).map(|v| {
        h_last.iter().zip(&wte_w2[v*N_EMBD..(v+1)*N_EMBD]).map(|(a,b)| a*b).sum::<f32>()
    }).collect()
}

// ── Ops ───────────────────────────────────────────────────────────────────────

fn layernorm(x: &[f32], gamma: &[f32], beta: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var  = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std  = (var + 1e-5).sqrt();
    x.iter().enumerate().map(|(i, v)| (v - mean) / std * gamma[i] + beta[i]).collect()
}

fn gelu(x: f32) -> f32 {
    const C: f32 = 0.7978845608028654;
    x * 0.5 * (1.0 + (C * (x + 0.044715 * x.powi(3))).tanh())
}

fn softmax(s: &[f32]) -> Vec<f32> {
    let max = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = s.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.iter().map(|x| x / sum).collect()
}

fn heads(x: &[f32]) -> Vec<Vec<f32>> {
    (0..N_HEAD).map(|h| x[h*HEAD_DIM..(h+1)*HEAD_DIM].to_vec()).collect()
}

// ── Top-K ─────────────────────────────────────────────────────────────────────

fn top_k_f32(v: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut p: Vec<(usize, f32)> = v.iter().enumerate().map(|(i,&x)| (i,x)).collect();
    p.sort_unstable_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
    p.truncate(k); p
}

fn top_k_i64(v: &[i64], k: usize) -> Vec<(usize, i64)> {
    let mut p: Vec<(usize, i64)> = v.iter().enumerate().map(|(i,&x)| (i,x)).collect();
    p.sort_unstable_by(|a,b| b.1.cmp(&a.1));
    p.truncate(k); p
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let tokens: Vec<u32> = vec![15496, 11, 995];
    println!("GPT-2 Ternify — integer matmul via SignalLinear");
    println!("═══════════════════════════════════════════════════════════");
    println!("Tokens: {:?}  (\"Hello, world\")\n", tokens);

    let t0 = Instant::now();
    let manifest = load_manifest(MODEL_DIR);
    println!("Manifest: {} entries in {:.1}s\n", manifest.len(), t0.elapsed().as_secs_f32());

    println!("Float reference...");
    let t1 = Instant::now();
    let float_logits = float_forward(MODEL_DIR, &manifest, &tokens);
    println!("  Done in {:.3}s", t1.elapsed().as_secs_f32());
    let ft5 = top_k_f32(&float_logits, 5);
    println!("  Float top-5:");
    for (tok, sc) in &ft5 { println!("    token {:6}  {:.3}", tok, sc); }
    println!();

    println!("Encoding weights as Signodes...");
    let t2 = Instant::now();
    let model = Gpt2Int::load(MODEL_DIR, &manifest);
    println!("  Encoded in {:.1}s\n", t2.elapsed().as_secs_f32());

    println!("Integer forward pass...");
    let t3 = Instant::now();
    let int_logits = model.forward(&tokens);
    println!("  Done in {:.3}s", t3.elapsed().as_secs_f32());
    let it5 = top_k_i64(&int_logits, 5);
    println!("  Integer top-5:");
    for (tok, sc) in &it5 { println!("    token {:6}  {}", tok, sc); }
    println!();

    println!("═══════════════════════════════════════════════════════════");
    let top1_match = ft5[0].0 == it5[0].0;
    println!("Top-1 match:   {}", if top1_match { "YES ✓" } else { "NO ✗" });
    println!("Float top-1:   token {}", ft5[0].0);
    println!("Integer top-1: token {}", it5[0].0);
    println!();

    let f_max = float_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let f_min = float_logits.iter().cloned().fold(f32::INFINITY,     f32::min);
    let f_rng = (f_max - f_min).max(f32::EPSILON);
    let i_max = *int_logits.iter().max().unwrap_or(&1) as f64;
    let i_min = *int_logits.iter().min().unwrap_or(&0) as f64;
    let i_rng = (i_max - i_min).max(f64::EPSILON);

    let max_delta = float_logits.iter().zip(int_logits.iter()).map(|(&f, &iv)| {
        let fn_ = (f - f_min) / f_rng;
        let in_ = ((iv as f64 - i_min) / i_rng) as f32;
        (fn_ - in_).abs()
    }).fold(0.0f32, f32::max);

    println!("Max norm logit delta: {max_delta:.4}  (target < 0.10)");

    let ft10: std::collections::HashSet<usize> = top_k_f32(&float_logits, 10).into_iter().map(|(t,_)| t).collect();
    let it10: std::collections::HashSet<usize> = top_k_i64(&int_logits, 10).into_iter().map(|(t,_)| t).collect();
    println!("Top-10 overlap: {}/10", ft10.intersection(&it10).count());

    if top1_match && max_delta < 0.10 {
        println!("\nPASSED ✓");
    } else {
        println!("\nFAILED ✗  (delta {max_delta:.4})");
    }
}
