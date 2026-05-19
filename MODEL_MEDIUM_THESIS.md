# Neuronal Distillation — Phase 1: Activation Prober

## Context

The thesis: LLM parameters are a medium for intelligence, not the intelligence itself.
We can distill the functional behavior of each layer/expert into cortical wiring structure
that produces equivalent outputs without carrying the matrices at all.

The activation prober is the first tool in that pipeline. Before we can encode a layer's
behavior as neuronal structure, we must characterize what it *does* — its functional
fingerprint across a real corpus. This is the "probe" phase of probe → extract → synthesize → validate.

We start with the smallest, most mechanistically understood models and work up. Every
successful distillation is a proof that extends the method to larger scale.

---

## 5-Model Proof Sequence

| Step | Model | Params | Why |
|------|-------|--------|-----|
| 1 | Neel Nanda's 1L Transformer (modular addition) | ~500K | Only model where every circuit is 100% published ground truth |
| 2 | GPT-2 Small | 117M | Already in codebase (gpt2.rs, 17 tok/s). Most interpreted LLM ever. |
| 3 | SmolLM2 135M | 135M | Modern arch: RMSNorm + SwiGLU + GQA + RoPE. Tests modern patterns. |
| 4 | Qwen3 1.7B | 1.7B | Already in strategos plan (qwen3.rs). First real capable LLM scale. |
| 5 | MiniMax M2.7 | 230B (10B active) | End target. 256 experts = 256 cortical regions to distill. |

---

## What We Are Building: `strategos-probe`

A new crate in strategos-brain. A CLI binary that:
1. Loads a model (GPT-2 first)
2. Runs a text corpus through it token by token
3. Captures intermediate activations at every layer boundary
4. Aggregates statistical fingerprints per layer, per op, per neuron/head
5. Writes a structured JSON activation profile

This profile becomes the input to the next tool: invariant extractor → circuit synthesizer.

---

## New Crate: `strategos-probe`

**Location**: `E:\repos\blackfall-labs\strategos-brain\crates\probe\`

**Add to workspace**: `E:\repos\blackfall-labs\strategos-brain\Cargo.toml`
```toml
members = ["crates/runtime", "crates/ops", "crates/frame-catalog", "crates/probe"]
```

**Dependencies**:
- `strategos-ops` — QTensor, QWeight (reuse existing)
- `strategos-runtime` — ModelWeights loader (manifest.json+.bin), safetensors, tokenizer
- `serde`, `serde_json` — output format
- `clap` — CLI args

---

## Source Files

### `crates/probe/src/capture.rs`
`LayerCapture` struct — accumulates activation stats across all tokens for one op point.
Fields: op name, layer index, running mean/variance (Welford online), min/max, per-neuron
activation frequency (count of |x| > threshold), count of zero-range activations (sparsity).

`ActivationProfile` — full model profile, serializes to JSON.

### `crates/probe/src/stats.rs`
Welford online mean/variance over `QTensor` slices.
Produces: mean, std, min, max, p5/p95 (reservoir sample), sparsity ratio.
For attention layers: per-head entropy from softmax weights.
For FFN hidden: per-neuron activation frequency (what % of tokens fire each neuron).

### `crates/probe/src/gpt2_probe.rs`
GPT-2 forward pass with capture hooks injected at every op boundary.
Based on `ternsig/crates/deepspeech/src/gpt2.rs` (394 lines) — same weight loading
(manifest.json + .bin), same QWeight/QTensor pipeline.
Capture points per layer:
- `attn_pre_ln` — residual stream entering attention LayerNorm
- `attn_q`, `attn_k`, `attn_v` — post-projection
- `attn_weights` — softmax attention matrix (per head)
- `attn_out` — post-output projection
- `ffn_pre_ln` — residual stream entering FFN LayerNorm
- `ffn_hidden` — post-GELU hidden (4×768 = 3072 neurons)
- `ffn_out` — post-output projection
- `residual` — residual stream after each full layer

### `crates/probe/src/corpus.rs`
Reads a plain `.txt` file line by line. Tokenizes via GPT-2 BPE vocab
(reuse `strategos-runtime::tokenizer`). Yields token sequences.
Configurable: max tokens, shuffle, seed.

### `crates/probe/src/main.rs`
CLI binary: `probe`
```
probe --model gpt2 --weights <path/to/manifest.json> --corpus <corpus.txt>
      --output <profile.json> --n-tokens 50000
```
Loops corpus tokens, calls `gpt2_probe::forward_with_capture()`, feeds captures
to aggregator, writes JSON at end.

---

## Output Format

```json
{
  "model": "gpt2-small",
  "n_tokens": 50000,
  "corpus": "wikitext2_subset.txt",
  "layers": [
    {
      "layer": 0,
      "ops": {
        "attn_pre_ln":  { "mean": 0.0, "std": 1.0, "min": -4.2, "max": 3.8, "sparsity": 0.02 },
        "attn_q":       { "mean": 0.1, "std": 0.9, ... },
        "attn_weights": { "per_head_entropy": [2.1, 1.8, 3.0, ...] },
        "ffn_hidden": {
          "mean": 0.05, "std": 0.8, "sparsity": 0.42,
          "neuron_freq": [0.12, 0.88, 0.03, ...]
        },
        "residual": { "mean": 0.0, "std": 1.1, "drift_from_layer0": 0.08 }
      }
    }
  ]
}
```

`neuron_freq` is the key output: the fraction of tokens that activated each FFN neuron
above threshold. Dead neurons (freq ≈ 0) and universal neurons (freq ≈ 1) are the first
proof of functional structure — they are the ones most directly expressible as cortical
column wiring constraints.

---

## Critical Files to Modify / Create

| File | Action |
|------|--------|
| `E:\repos\blackfall-labs\strategos-brain\Cargo.toml` | Add `crates/probe` to workspace |
| `crates/probe/Cargo.toml` | New — deps: strategos-ops, strategos-runtime, serde_json, clap |
| `crates/probe/src/main.rs` | New — CLI entrypoint |
| `crates/probe/src/capture.rs` | New — LayerCapture, ActivationProfile |
| `crates/probe/src/stats.rs` | New — Welford accumulator, neuron_freq |
| `crates/probe/src/gpt2_probe.rs` | New — instrumented GPT-2 forward (from gpt2.rs) |
| `crates/probe/src/corpus.rs` | New — corpus tokenization |

**Reuse (do not reimplement)**:
- `strategos-ops::qtensor::{QTensor, QWeight}` — quantized tensor types
- `strategos-runtime::loader::ModelWeights` — manifest.json+.bin loader
- `strategos-runtime::tokenizer` — BPE tokenizer
- Integer op validation pattern from `ternsig/crates/deepspeech/src/ops_int.rs` lines 247-254

---

## Verification

1. `cargo check -p strategos-probe` — compiles without errors
2. Convert GPT-2 weights to manifest.json+.bin using existing Python conversion script
   (same format as conformer — already proven)
3. Download WikiText-2 raw text as corpus (public domain, small, well-known)
4. Run: `cargo run -p strategos-probe --bin probe -- --model gpt2 --weights ... --corpus wikitext2.txt --output gpt2_profile.json --n-tokens 50000`
5. Inspect `gpt2_profile.json`:
   - FFN hidden sparsity should be ~40-60% (known from GPT-2 interpretability work)
   - `attn_weights` per-head entropy should vary significantly between heads (induction heads have low entropy = spiky; uniform heads have high entropy)
   - `neuron_freq` distribution should be bimodal: many neurons nearly-dead, some near-universal
6. Milestone: if the per-head entropy pattern matches published GPT-2 mechanistic interpretability findings, the prober is working correctly

---

## What Comes Next (not in scope yet)

- **Invariant extractor**: takes activation profile → finds input patterns that reliably activate specific neurons/heads → functional fingerprint per op
- **Circuit synthesizer**: takes fingerprints → outputs cortical region seed (column count, layer profile, tract topology, inter-column wiring weights)
- **Validator**: runs cortical circuit vs original model on held-out inputs, measures output divergence
- **SmolLM2 135M**: extend prober to handle safetensors format + GQA/SwiGLU/RoPE capture points
