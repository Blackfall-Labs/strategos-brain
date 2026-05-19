"""Build a lookup table for Band 1 L0 embedding-dominant neurons in SmolLM2.

For each Band 1 L0 cluster (trigger_token, neuron_list), precomputes:
    activation[neuron] = silu(W_gate[n,:] @ rms_norm(embed[tok])) * W_up[n,:] @ rms_norm(embed[tok])

Writes only the neurons where abs(activation) > L0_THRESHOLD (the embedding-dominant
subset — 64% of Band 1 L0 neurons based on validate_distill.py results).

Output JSON schema:
{
  "model": "smollm2",
  "layer": 0,
  "threshold": 0.1321,
  "n_neurons": <int>,          # total neurons in lookup
  "n_tokens_covered": <int>,   # number of distinct trigger tokens
  "entries": [
    {
      "neuron": <int>,
      "token_id": <int>,
      "token_str": <str>,
      "activation": <float>    # precomputed silu(gate) * up value
    }
  ]
}

Usage:
    python tools/build_lookup.py --weights models/smollm2 --bands smollm2_bands.json
        --tokenizer models/smollm2/tokenizer.json --output smollm2_l0_lookup.json
"""

import json
import struct
import argparse
import math
import os
import numpy as np


def load_manifest(manifest_dir):
    with open(os.path.join(manifest_dir, "manifest.json")) as f:
        return json.load(f)


def load_tensor_np(manifest_dir, entry):
    path = os.path.join(manifest_dir, entry["file"])
    numel = entry["numel"]
    with open(path, "rb") as f:
        data = f.read(numel * 4)
    flat = np.frombuffer(data, dtype=np.float32).copy()
    if entry.get("shape"):
        flat = flat.reshape(entry["shape"])
    return flat


def find_tensor(manifest, name):
    for entry in manifest:
        if entry["name"] == name:
            return entry
    raise KeyError(f"tensor not found: {name}")


def rms_norm_np(x, weight, eps=1e-5):
    ms = np.mean(x * x)
    scale = 1.0 / math.sqrt(float(ms) + eps)
    return x * scale * weight


def silu(x):
    return x / (1.0 + np.exp(-x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    required=True, help="Path to smollm2 manifest dir")
    parser.add_argument("--bands",      required=True, help="Path to smollm2_bands.json")
    parser.add_argument("--tokenizer",  required=True, help="Path to tokenizer.json")
    parser.add_argument("--output",     required=True, help="Output lookup JSON path")
    args = parser.parse_args()

    HIDDEN   = 576
    FFN_DIM  = 1536
    VOCAB    = 49152
    L0_THRESHOLD = 0.1321

    print("[lookup] loading weights...")
    manifest = load_manifest(args.weights)

    embed = load_tensor_np(args.weights, find_tensor(manifest, "embed_tokens.weight"))
    assert embed.shape == (VOCAB, HIDDEN), f"embed shape mismatch: {embed.shape}"

    norm_w = load_tensor_np(args.weights, find_tensor(manifest, "layers.0.post_attention_layernorm.weight"))
    gate_w = load_tensor_np(args.weights, find_tensor(manifest, "layers.0.mlp.gate_proj.weight"))
    up_w   = load_tensor_np(args.weights, find_tensor(manifest, "layers.0.mlp.up_proj.weight"))
    # gate_w shape: [FFN_DIM, HIDDEN]
    assert gate_w.shape == (FFN_DIM, HIDDEN), f"gate_w shape: {gate_w.shape}"

    print(f"  embed: {embed.shape}, gate: {gate_w.shape}, up: {up_w.shape}")

    with open(args.tokenizer, encoding="utf-8") as f:
        tok_json = json.load(f)
    tok_str_to_id = tok_json["model"]["vocab"]

    with open(args.bands, encoding="utf-8") as f:
        bands = json.load(f)
    b1 = next(b for b in bands["bands"] if b["band"] == 1)

    # Collect all L0 clusters
    l0_clusters = []
    for cluster in b1["clusters"]:
        l0_neurons = [n for n in cluster["neurons"] if n["layer"] == 0]
        if l0_neurons:
            l0_clusters.append((cluster["trigger_token"], l0_neurons))

    print(f"\n[lookup] {len(l0_clusters)} clusters with L0 neurons")
    print(f"  threshold: {L0_THRESHOLD}")

    entries = []
    n_total = 0
    n_added = 0
    tokens_covered = set()

    for trigger_tok_str, l0_neurons in l0_clusters:
        tok_id = tok_str_to_id.get(trigger_tok_str)
        if tok_id is None:
            continue

        # Compute embedding-alone activation for this token
        x = rms_norm_np(embed[tok_id], norm_w)      # [HIDDEN]
        gate = gate_w @ x                             # [FFN_DIM]
        up   = up_w   @ x                             # [FFN_DIM]
        ffn_hidden = silu(gate) * up                  # [FFN_DIM]

        for n in l0_neurons:
            ni = n["neuron"]
            val = float(ffn_hidden[ni])
            n_total += 1
            if abs(val) > L0_THRESHOLD:
                entries.append({
                    "neuron":     ni,
                    "token_id":   tok_id,
                    "token_str":  trigger_tok_str,
                    "activation": round(val, 6),
                })
                tokens_covered.add(tok_id)
                n_added += 1

    precision = n_added / n_total * 100 if n_total > 0 else 0
    print(f"\n[lookup] {n_added}/{n_total} neurons qualify ({precision:.1f}%)")
    print(f"  {len(tokens_covered)} distinct trigger tokens covered")

    out = {
        "model":             "smollm2",
        "layer":             0,
        "threshold":         L0_THRESHOLD,
        "n_neurons":         n_added,
        "n_tokens_covered":  len(tokens_covered),
        "entries":           entries,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n[lookup] written to {args.output}")
    # Compression ratio
    original_params = n_added * 2 * HIDDEN
    lookup_params   = n_added
    print(f"  Original params: {n_added} neurons × 2 × {HIDDEN} = {original_params:,} floats")
    print(f"  Lookup params:   {n_added} scalars")
    print(f"  Compression:     {original_params / max(lookup_params,1):.0f}×")


if __name__ == "__main__":
    main()
