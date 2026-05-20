"""Build a calibrated multi-layer lookup table for all Band 1 neurons in SmolLM2.

For each Band 1 (layer, neuron, trigger_token) triplet, runs the full SmolLM2
forward pass on a corpus and records the mean FFN hidden activation whenever
the trigger token is seen at that position.

The resulting lookup table stores:
    lookup[layer][token_id] = [(neuron_id, mean_activation), ...]

This is the corpus-calibrated approach (vs. embedding-alone for L0). It works
for all 30 layers because the mean activation is measured at runtime, including
the attention contribution that embedding-alone misses.

Outlier filter: any entry with abs(mean_activation) > OUTLIER_THRESHOLD is
excluded. These neurons have high context sensitivity and their corpus mean is
not a reliable substitute.

Output JSON schema:
{
  "model": "smollm2",
  "n_tokens": <int>,
  "outlier_threshold": <float>,
  "layers": [
    {
      "layer": <int>,
      "n_entries": <int>,
      "entries": [
        {"neuron": <int>, "token_id": <int>, "token_str": <str>, "activation": <float>}
      ]
    }
  ]
}

Usage:
    python tools/build_multilayer_lookup.py \\
        --weights models/smollm2 \\
        --bands smollm2_bands.json \\
        --tokenizer models/smollm2/tokenizer.json \\
        --corpus data/wikitext2_train.txt \\
        --n-tokens 5000 \\
        --output smollm2_multilayer_lookup.json
"""

import json
import math
import os
import argparse
from collections import defaultdict

import numpy as np


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
KV_GROUP   = N_Q_HEADS // N_KV_HEADS
OUTLIER_THRESHOLD = 2.0


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


def softmax_np(x):
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
        print("[lookup] loading embed...")
        self.embed = g("embed_tokens.weight")
        self.norm_f = g("norm.weight")

        self.layers = []
        for i in range(N_LAYERS):
            p = f"layers.{i}"
            if i % 6 == 0:
                print(f"[lookup]   loading layer {i}/{N_LAYERS}...")
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
        print(f"[lookup] {N_LAYERS} layers loaded")


def make_kv_cache():
    return {"k": [[] for _ in range(N_LAYERS)],
            "v": [[] for _ in range(N_LAYERS)]}


# ── forward pass with FFN capture ────────────────────────────────────────────

def forward_capture_ffn(tok_id, ctx_pos, kv_cache, weights):
    """Full SmolLM2 forward for one token. Returns ffn_hidden for all layers."""
    hidden = weights.embed[tok_id].copy()
    ffn_hiddens = []

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
            wts    = softmax_np(scores)
            head_v = sum(wts[ki] * kv_cache["v"][li][ki][kv_h] for ki in range(seq_len))
            attn_out[q_h * HEAD_DIM:(q_h + 1) * HEAD_DIM] = head_v

        hidden += layer["o_proj"] @ attn_out

        # --- FFN ---
        pre_ffn    = rms_norm(hidden, layer["post_attn_ln"])
        gate       = layer["gate_proj"] @ pre_ffn
        up         = layer["up_proj"]   @ pre_ffn
        ffn_hidden = silu_np(gate) * up
        ffn_hiddens.append(ffn_hidden)

        hidden += layer["down_proj"] @ ffn_hidden

    return ffn_hiddens


# ── tokenizer ─────────────────────────────────────────────────────────────────

def load_tokens(corpus_path, tokenizer_path, max_tokens):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    with open(corpus_path, encoding="utf-8") as f:
        text = f.read(max_tokens * 8)
    enc = tok.encode(text)
    tokens = enc.ids[:max_tokens]
    print(f"[lookup] tokenized: {len(tokens)} tokens")
    return tokens


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",   required=True)
    parser.add_argument("--bands",     required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--corpus",    required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--n-tokens",  type=int, default=5000)
    parser.add_argument("--outlier-threshold", type=float, default=OUTLIER_THRESHOLD)
    parser.add_argument("--min-obs", type=int, default=5,
                        help="Minimum observations to include an entry (default: 5)")
    args = parser.parse_args()

    weights = SmolLM2Weights(args.weights)
    tokens  = load_tokens(args.corpus, args.tokenizer, args.n_tokens)

    # Load Band 1 neuron list from bands file
    with open(args.bands, encoding="utf-8") as f:
        bands_data = json.load(f)
    band1 = next(b for b in bands_data["bands"] if b["band"] == 1)

    # Build: token_id -> str
    with open(args.tokenizer, encoding="utf-8") as f:
        tok_json = json.load(f)
    id_to_str = {v: k for k, v in tok_json["model"]["vocab"].items()}

    # Build: (layer, neuron) -> trigger_token_id set
    # Also: trigger_token_id -> token_str
    trigger_neurons = defaultdict(set)   # token_id -> set of (layer, neuron)
    token_id_to_str = {}
    for cluster in band1["clusters"]:
        tok_str = cluster["trigger_token"]
        tok_id  = tok_json["model"]["vocab"].get(tok_str)
        if tok_id is None:
            continue
        token_id_to_str[tok_id] = tok_str
        for n in cluster["neurons"]:
            trigger_neurons[tok_id].add((n["layer"], n["neuron"]))

    print(f"[lookup] {len(trigger_neurons)} trigger token IDs, "
          f"{sum(len(v) for v in trigger_neurons.values())} (layer,neuron) pairs")

    # Accumulators: (layer, neuron, token_id) -> [sum, count]
    accum = defaultdict(lambda: [0.0, 0])

    # Run forward pass — 128-token chunks, reset KV cache each chunk
    CHUNK = 128
    n_chunks = (len(tokens) + CHUNK - 1) // CHUNK
    print(f"[lookup] running {len(tokens)} tokens in {n_chunks} chunks of {CHUNK}...")

    for ci in range(n_chunks):
        chunk = tokens[ci * CHUNK:(ci + 1) * CHUNK]
        kv    = make_kv_cache()
        for ctx_pos, tok_id in enumerate(chunk):
            if tok_id not in trigger_neurons:
                # Still need to run full forward to maintain KV cache state
                forward_capture_ffn(tok_id, ctx_pos, kv, weights)
                continue

            ffn_hiddens = forward_capture_ffn(tok_id, ctx_pos, kv, weights)
            for (li, ni) in trigger_neurons[tok_id]:
                val = float(ffn_hiddens[li][ni])
                acc = accum[(li, ni, tok_id)]
                acc[0] += val
                acc[1] += 1

        if (ci + 1) % 10 == 0 or ci == n_chunks - 1:
            print(f"[lookup]   chunk {ci+1}/{n_chunks} done")

    print(f"[lookup] accumulation complete. {len(accum)} (layer,neuron,token) entries")

    # Compute means, group by layer
    layer_entries = defaultdict(list)
    n_outliers = 0
    n_low_obs = 0
    for (li, ni, tok_id), (s, c) in accum.items():
        if c == 0:
            continue
        if c < args.min_obs:
            n_low_obs += 1
            continue
        mean_val = s / c
        if abs(mean_val) > args.outlier_threshold:
            n_outliers += 1
            continue
        layer_entries[li].append({
            "neuron":     ni,
            "token_id":   tok_id,
            "token_str":  token_id_to_str.get(tok_id, str(tok_id)),
            "activation": round(float(mean_val), 6),
            "n_obs":      c,
        })

    print(f"[lookup] {n_low_obs} low-obs entries excluded (n_obs < {args.min_obs})")
    print(f"[lookup] {n_outliers} outlier entries excluded (|mean| > {args.outlier_threshold})")
    total_entries = sum(len(v) for v in layer_entries.values())
    print(f"[lookup] {total_entries} valid entries across {len(layer_entries)} layers")

    # Print per-layer summary
    for li in sorted(layer_entries.keys()):
        toks = len({e["token_id"] for e in layer_entries[li]})
        print(f"  L{li:2d}: {len(layer_entries[li]):4d} entries, {toks} tokens")

    out = {
        "model":              "smollm2",
        "n_tokens":           len(tokens),
        "outlier_threshold":  args.outlier_threshold,
        "min_obs":            args.min_obs,
        "n_total_entries":    total_entries,
        "layers": [
            {
                "layer":    li,
                "n_entries": len(layer_entries[li]),
                "entries":  sorted(layer_entries[li], key=lambda e: (e["neuron"], e["token_id"])),
            }
            for li in sorted(layer_entries.keys())
        ],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n[lookup] written to {args.output}")
    original_params = total_entries * 2 * HIDDEN
    print(f"  Compression: {total_entries} entries replaces {original_params:,} weight params  ({original_params // max(total_entries, 1)}× per entry)")


if __name__ == "__main__":
    main()
