"""Measure SmolLM2 perplexity with and without L0 lookup table substitution.

Runs the full SmolLM2 forward pass in Python/numpy on a small validation corpus.
Two modes:
  baseline  — standard forward pass, all weight matrices intact
  patched   — Band 1 L0 neurons substituted with precomputed lookup values

Reports: perplexity for both modes and the percentage change.

Usage:
    python tools/measure_perplexity.py --weights models/smollm2
        --lookup smollm2_l0_lookup.json --tokenizer models/smollm2/tokenizer.json
        --corpus data/wikitext2_train.txt --n-tokens 500
"""

import json
import struct
import argparse
import math
import os
import numpy as np
from collections import defaultdict


# SmolLM2 architecture
HIDDEN     = 576
N_Q_HEADS  = 9
N_KV_HEADS = 3
HEAD_DIM   = 64
FFN_DIM    = 1536
VOCAB      = 49152
N_LAYERS   = 30
RMS_EPS    = 1e-5
ROPE_BASE  = 100_000.0
KV_GROUP   = N_Q_HEADS // N_KV_HEADS   # 3


# ── weight loading ────────────────────────────────────────────────────────────

def load_manifest(d):
    with open(os.path.join(d, "manifest.json")) as f:
        return json.load(f)


def find_tensor(manifest, name):
    for e in manifest:
        if e["name"] == name:
            return e
    raise KeyError(f"tensor not found: {name}")


def load_np(manifest_dir, entry):
    path = os.path.join(manifest_dir, entry["file"])
    numel = entry["numel"]
    with open(path, "rb") as f:
        data = f.read(numel * 4)
    flat = np.frombuffer(data, dtype=np.float32).copy()
    if entry.get("shape"):
        flat = flat.reshape(entry["shape"])
    return flat


# ── ops ───────────────────────────────────────────────────────────────────────

def rms_norm(x, w, eps=RMS_EPS):
    scale = 1.0 / math.sqrt(float(np.mean(x * x)) + eps)
    return x * scale * w


def silu_np(x):
    return x / (1.0 + np.exp(-np.clip(x, -30, 30)))


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def build_rope_table(max_seq=512):
    cos_t = np.zeros((max_seq, HEAD_DIM // 2), dtype=np.float32)
    sin_t = np.zeros((max_seq, HEAD_DIM // 2), dtype=np.float32)
    for pos in range(max_seq):
        for i in range(HEAD_DIM // 2):
            theta = pos / (ROPE_BASE ** (2.0 * i / HEAD_DIM))
            cos_t[pos, i] = math.cos(theta)
            sin_t[pos, i] = math.sin(theta)
    return cos_t, sin_t


def apply_rope(v, pos, cos_t, sin_t):
    half = HEAD_DIM // 2
    x0, x1 = v[:half].copy(), v[half:].copy()
    v[:half] = x0 * cos_t[pos] - x1 * sin_t[pos]
    v[half:] = x0 * sin_t[pos] + x1 * cos_t[pos]


# ── model weights ─────────────────────────────────────────────────────────────

class SmolLM2Weights:
    def __init__(self, manifest_dir):
        m = load_manifest(manifest_dir)
        g = lambda name: load_np(manifest_dir, find_tensor(m, name))
        print("[ppl] loading embed...")
        self.embed   = g("embed_tokens.weight")           # [VOCAB, HIDDEN]
        self.norm_f  = g("norm.weight")                   # [HIDDEN] final norm
        self.lm_head = None                               # tied to embed

        self.layers = []
        for i in range(N_LAYERS):
            p = f"layers.{i}"
            if i % 6 == 0:
                print(f"[ppl]   loading layer {i}/{N_LAYERS}...")
            self.layers.append({
                "input_ln":    g(f"{p}.input_layernorm.weight"),
                "q_proj":      g(f"{p}.self_attn.q_proj.weight"),
                "k_proj":      g(f"{p}.self_attn.k_proj.weight"),
                "v_proj":      g(f"{p}.self_attn.v_proj.weight"),
                "o_proj":      g(f"{p}.self_attn.o_proj.weight"),
                "post_attn_ln": g(f"{p}.post_attention_layernorm.weight"),
                "gate_proj":   g(f"{p}.mlp.gate_proj.weight"),
                "up_proj":     g(f"{p}.mlp.up_proj.weight"),
                "down_proj":   g(f"{p}.mlp.down_proj.weight"),
            })

        self.cos, self.sin = build_rope_table()
        print(f"[ppl] {N_LAYERS} layers loaded")


# ── forward pass ─────────────────────────────────────────────────────────────

def forward(tok_id, ctx_pos, kv_cache, weights, lookup_patch=None):
    """Full SmolLM2 forward for one token. Returns log-prob vector over vocab.

    lookup_patch: optional dict {neuron_id: activation_val} to substitute at L0 FFN.
    """
    hidden = weights.embed[tok_id].copy()

    for li, layer in enumerate(weights.layers):
        # --- Attention ---
        pre_attn = rms_norm(hidden, layer["input_ln"])
        q = (layer["q_proj"] @ pre_attn).reshape(N_Q_HEADS, HEAD_DIM)
        k = (layer["k_proj"] @ pre_attn).reshape(N_KV_HEADS, HEAD_DIM)
        v = (layer["v_proj"] @ pre_attn).reshape(N_KV_HEADS, HEAD_DIM)

        for h in range(N_Q_HEADS):
            apply_rope(q[h], ctx_pos, weights.cos, weights.sin)
        for h in range(N_KV_HEADS):
            apply_rope(k[h], ctx_pos, weights.cos, weights.sin)

        kv_cache["k"][li].append(k)
        kv_cache["v"][li].append(v)

        seq_len = len(kv_cache["k"][li])
        scale   = math.sqrt(HEAD_DIM)
        attn_out = np.zeros(N_Q_HEADS * HEAD_DIM, dtype=np.float32)

        for q_h in range(N_Q_HEADS):
            kv_h   = q_h // KV_GROUP
            scores = np.array([np.dot(q[q_h], kv_cache["k"][li][ki][kv_h]) / scale
                               for ki in range(seq_len)])
            wts    = softmax(scores)
            head_v = sum(wts[ki] * kv_cache["v"][li][ki][kv_h] for ki in range(seq_len))
            attn_out[q_h * HEAD_DIM:(q_h + 1) * HEAD_DIM] = head_v

        hidden += layer["o_proj"] @ attn_out

        # --- FFN ---
        pre_ffn    = rms_norm(hidden, layer["post_attn_ln"])
        gate       = layer["gate_proj"] @ pre_ffn
        up         = layer["up_proj"]   @ pre_ffn
        ffn_hidden = silu_np(gate) * up

        # Patch L0 with lookup values if provided
        if li == 0 and lookup_patch is not None:
            for ni, val in lookup_patch.items():
                ffn_hidden[ni] = val

        hidden += layer["down_proj"] @ ffn_hidden

    # LM head: SmolLM2 ties weights (embed == lm_head)
    logits = rms_norm(hidden, weights.norm_f) @ weights.embed.T
    log_probs = logits - np.log(np.sum(np.exp(logits - logits.max())) + 1e-30) - logits.max()
    return log_probs


def make_kv_cache():
    return {"k": [[] for _ in range(N_LAYERS)],
            "v": [[] for _ in range(N_LAYERS)]}


# ── tokenizer ─────────────────────────────────────────────────────────────────

def load_tokens(corpus_path, tokenizer_path, max_tokens):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    with open(corpus_path, encoding="utf-8") as f:
        text = f.read(max_tokens * 8)
    enc = tok.encode(text)
    ids = enc.ids[:max_tokens]
    print(f"[ppl] tokenized: {len(ids)} tokens")
    return ids


# ── perplexity ────────────────────────────────────────────────────────────────

def compute_perplexity(tokens, weights, lookup_by_token=None, label="baseline"):
    """Compute per-token cross-entropy loss (→ perplexity) over a token sequence."""
    CONTEXT_LEN = 128
    total_nll   = 0.0
    n_pred      = 0

    for chunk_start in range(0, len(tokens) - 1, CONTEXT_LEN):
        chunk = tokens[chunk_start:chunk_start + CONTEXT_LEN + 1]
        kv    = make_kv_cache()

        for ctx_pos in range(len(chunk) - 1):
            tok_id  = chunk[ctx_pos]
            next_id = chunk[ctx_pos + 1]

            patch = lookup_by_token.get(tok_id) if lookup_by_token is not None else None
            log_probs = forward(tok_id, ctx_pos, kv, weights, lookup_patch=patch)
            total_nll += -float(log_probs[next_id])
            n_pred    += 1

        pct = (chunk_start + len(chunk)) / len(tokens) * 100
        if (chunk_start // CONTEXT_LEN) % 2 == 0:
            avg_nll = total_nll / n_pred if n_pred > 0 else 0
            print(f"  [{label}] {pct:.0f}%  nll={avg_nll:.4f}  ppl={math.exp(avg_nll):.2f}")

    avg_nll = total_nll / n_pred if n_pred > 0 else 0
    ppl     = math.exp(avg_nll)
    return ppl


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    required=True)
    parser.add_argument("--lookup",     required=True)
    parser.add_argument("--tokenizer",  required=True)
    parser.add_argument("--corpus",     required=True)
    parser.add_argument("--n-tokens",   type=int, default=500)
    parser.add_argument("--scale",      type=float, default=1.0,
                        help="Scale lookup values (0.82 corrects for attention dampening)")
    parser.add_argument("--clamp",      type=float, default=0.0,
                        help="Clamp lookup |value| to this max after scaling (0=no clamp)")
    args = parser.parse_args()

    with open(args.lookup, encoding="utf-8") as f:
        lookup_data = json.load(f)

    # Build index: token_id → {neuron_id → scaled/clamped activation}
    lookup_by_token = defaultdict(dict)
    for entry in lookup_data["entries"]:
        val = entry["activation"] * args.scale
        if args.clamp > 0:
            import math
            val = math.copysign(min(abs(val), args.clamp), val)
        lookup_by_token[entry["token_id"]][entry["neuron"]] = val

    print(f"[ppl] lookup: {lookup_data['n_neurons']} neurons, scale={args.scale}, clamp={args.clamp or 'none'}")

    weights = SmolLM2Weights(args.weights)
    tokens  = load_tokens(args.corpus, args.tokenizer, args.n_tokens)

    print(f"\n[ppl] === BASELINE (no patch) ===")
    ppl_base = compute_perplexity(tokens, weights, lookup_by_token=None, label="base")

    clamp_str = f", clamp={args.clamp}" if args.clamp else ""
    print(f"\n[ppl] === PATCHED (L0 lookup, scale={args.scale}{clamp_str}) ===")
    ppl_patch = compute_perplexity(tokens, weights, lookup_by_token=dict(lookup_by_token), label="patch")

    delta_pct = (ppl_patch - ppl_base) / ppl_base * 100
    print(f"\n[ppl] Results:")
    print(f"  Baseline perplexity: {ppl_base:.4f}")
    print(f"  Patched perplexity:  {ppl_patch:.4f}")
    print(f"  Delta:               {delta_pct:+.3f}%")

    if delta_pct < -0.5:
        verdict = f"IMPROVEMENT — perplexity decreased {delta_pct:.1f}%; lookup regularizes L0 signal"
    elif abs(delta_pct) < 0.5:
        verdict = "PASS — perplexity change < 0.5%; lookup substitution is lossless in practice"
    elif abs(delta_pct) < 2.0:
        verdict = f"NEAR-LOSSLESS — perplexity change {delta_pct:+.1f}%; within acceptable range"
    elif abs(delta_pct) < 5.0:
        verdict = f"ACCEPTABLE — perplexity change {delta_pct:+.1f}%; minor quality loss"
    else:
        verdict = f"DEGRADATION — perplexity change {delta_pct:+.1f}%; substitution hurts quality"
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
