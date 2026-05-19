"""First distillation proof: validate L0 pure-token neuron lookup approximation.

Claim: For Band 1 (pure-token) neurons at Layer 0 in SmolLM2, the FFN activation
value when the trigger token appears can be computed from the token embedding alone,
without running the attention layer.

If true: the weight matrix rows for those neurons (gate_proj, up_proj) can be
replaced by a precomputed scalar table:
    activation[neuron] = silu(W_gate[neuron,:] @ rms_norm(embed[tok])) * W_up[neuron,:] @ rms_norm(embed[tok])

This replaces a 576-dim dot product with a single float lookup per (neuron, token).

Validation method:
  1. For each Band 1 L0 cluster (trigger token T, neuron list [N]):
     a. Compute lookup value from embedding alone (no attention)
     b. Check all predicted neurons fire (value > L0 threshold = 0.1321)
  2. Report precision: what fraction of predicted neurons actually fire?

Usage:
    python tools/validate_distill.py --weights models/smollm2 --bands smollm2_bands.json
"""

import json
import struct
import argparse
import math
import os


def load_manifest(manifest_dir):
    path = os.path.join(manifest_dir, "manifest.json")
    with open(path) as f:
        return json.load(f)


def load_tensor(manifest_dir, entry):
    path = os.path.join(manifest_dir, entry["file"])
    numel = entry["numel"]
    with open(path, "rb") as f:
        data = f.read(numel * 4)
    return list(struct.unpack(f"{numel}f", data))


def find_tensor(manifest, name):
    for entry in manifest:
        if entry["name"] == name:
            return entry
    raise KeyError(f"tensor not found: {name}")


def rms_norm(x, weight, eps=1e-5):
    n = len(x)
    ms = sum(v * v for v in x) / n
    scale = 1.0 / math.sqrt(ms + eps)
    return [x[i] * scale * weight[i] for i in range(n)]


def dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))


def silu(x):
    return x / (1.0 + math.exp(-x))


def compute_l0_ffn_no_attn(tok_id, embed_flat, hidden, input_norm_w, gate_w, up_w, ffn_dim):
    """Compute L0 FFN hidden activations for a token with zero attention context.

    embed_flat: flat [VOCAB * HIDDEN] embedding matrix
    Returns: list of FFN hidden activations [FFN_DIM]
    """
    # Token embedding
    tok_embed = embed_flat[tok_id * hidden : (tok_id + 1) * hidden]

    # L0 input RMSNorm (eps from SmolLM2 config)
    x = rms_norm(tok_embed, input_norm_w, eps=1e-5)

    # Gate and up projections: gate_w is [FFN_DIM, HIDDEN] row-major
    ffn_hidden = []
    for ni in range(ffn_dim):
        row_start = ni * hidden
        row = gate_w[row_start : row_start + hidden]
        g = dot(row, x)
        row_up = up_w[row_start : row_start + hidden]
        u = dot(row_up, x)
        ffn_hidden.append(silu(g) * u)

    return ffn_hidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to smollm2 manifest dir")
    parser.add_argument("--bands", required=True, help="Path to smollm2_bands.json")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer.json (optional)")
    args = parser.parse_args()

    # SmolLM2 architecture constants
    HIDDEN   = 576
    FFN_DIM  = 1536
    VOCAB    = 49152
    L0_THRESHOLD = 0.1321  # from ffn_thresholds.json layer 0

    print("[validate] loading weights...")
    manifest = load_manifest(args.weights)

    embed_entry = find_tensor(manifest, "embed_tokens.weight")
    print(f"  embedding: {embed_entry['shape']}")
    embed_flat = load_tensor(args.weights, embed_entry)

    l0_norm_entry = find_tensor(manifest, "layers.0.post_attention_layernorm.weight")
    input_norm_w = load_tensor(args.weights, l0_norm_entry)

    l0_gate_entry = find_tensor(manifest, "layers.0.mlp.gate_proj.weight")
    l0_up_entry   = find_tensor(manifest, "layers.0.mlp.up_proj.weight")
    print(f"  gate_proj: {l0_gate_entry['shape']}  up_proj: {l0_up_entry['shape']}")
    gate_w = load_tensor(args.weights, l0_gate_entry)
    up_w   = load_tensor(args.weights, l0_up_entry)

    # Load band assignments
    with open(args.bands, encoding="utf-8") as f:
        bands = json.load(f)

    # Optional vocab for display
    vocab_inv = {}
    tok_str_to_id = {}
    if args.tokenizer:
        with open(args.tokenizer, encoding="utf-8") as f:
            tok_json = json.load(f)
        if "model" in tok_json and "vocab" in tok_json["model"]:
            vocab_inv = {v: k for k, v in tok_json["model"]["vocab"].items()}
            tok_str_to_id = tok_json["model"]["vocab"]

    # Get Band 1 L0 clusters
    b1 = next(b for b in bands["bands"] if b["band"] == 1)
    l0_clusters = []
    for cluster in b1["clusters"]:
        l0_neurons = [n for n in cluster["neurons"] if n["layer"] == 0]
        if len(l0_neurons) >= 3:
            l0_clusters.append((cluster["trigger_token"], l0_neurons))

    print(f"\n[validate] {len(l0_clusters)} Band-1 L0 clusters with >= 3 neurons")
    print(f"  L0 threshold: {L0_THRESHOLD}")
    print()

    total_neurons = 0
    total_firing = 0
    total_correct_sign = 0

    print(f"{'Token':25s}  {'n_neurons':>10}  {'fired':>7}  {'precision':>10}  {'near':>8}  {'mean_act':>10}")
    print("-" * 80)

    for trigger_tok_str, l0_neurons in l0_clusters[:20]:
        tok_id = tok_str_to_id.get(trigger_tok_str)
        if tok_id is None:
            print(f"  {trigger_tok_str!r:23s}  (token not in vocab, skipping)")
            continue

        # Compute L0 FFN hidden for this token with no attention context
        ffn_hidden = compute_l0_ffn_no_attn(
            tok_id, embed_flat, HIDDEN, input_norm_w, gate_w, up_w, FFN_DIM
        )

        # Check each predicted neuron
        neuron_ids = [n["neuron"] for n in l0_neurons]
        activations = [ffn_hidden[ni] for ni in neuron_ids]
        firing      = [abs(v) > L0_THRESHOLD for v in activations]
        near_miss   = [not f and abs(v) > L0_THRESHOLD * 0.5 for v, f in zip(activations, firing)]
        n_fired     = sum(firing)
        n_near      = sum(near_miss)
        mean_act    = sum(activations) / len(activations)

        tok_display = trigger_tok_str.replace("Ġ", "_").replace("\xa0", "_").replace("\n", "\\n").encode("ascii", "replace").decode("ascii")
        precision = n_fired / len(neuron_ids) * 100
        print(f"  {tok_display!r:23s}  {len(neuron_ids):>10}  {n_fired:>7}  {precision:>9.1f}%  {n_near:>8}  {mean_act:>10.4f}")

        total_neurons += len(neuron_ids)
        total_firing  += n_fired
        total_correct_sign += n_near

    print("-" * 80)
    overall_precision = total_firing / total_neurons * 100 if total_neurons > 0 else 0
    near_pct = (total_firing + total_correct_sign) / total_neurons * 100 if total_neurons > 0 else 0
    print(f"  {'TOTAL':23s}  {total_neurons:>10}  {total_firing:>7}  {overall_precision:>9.1f}%  {total_correct_sign:>8}  ({near_pct:.1f}% fired+near)")

    print()
    if overall_precision >= 90.0:
        print("[validate] PASS: >=90% of Band-1 L0 neurons fire from embedding alone.")
        print("           Lookup table distillation is valid.")
    elif overall_precision >= 70.0:
        print("[validate] PARTIAL: 70-90% precision. Embedding drives most neurons;")
        print("           attention adds final push for remainder.")
    else:
        print(f"[validate] RESULT: {overall_precision:.1f}% fire from embedding alone.")
        print(f"           {near_pct - overall_precision:.1f}% more are near-threshold (within 2× threshold).")
        print(f"           Embedding is significant driver but attention contributes ~{100-overall_precision:.0f}%.")

    print()
    print(f"[validate] Compression ratio for these neurons:")
    replaced_neurons = total_neurons
    # Original storage: replaced_neurons neurons × 2 (gate + up) × HIDDEN floats
    original_params = replaced_neurons * 2 * HIDDEN
    # Lookup storage: one float per (neuron, trigger_token) — for these neurons, 1 trigger each
    lookup_params = replaced_neurons  # one scalar per neuron
    print(f"  Original:  {replaced_neurons} neurons × 2 × {HIDDEN} = {original_params:,} floats")
    print(f"  Lookup:    {replaced_neurons} neurons × 1 scalar = {lookup_params:,} floats")
    print(f"  Reduction: {original_params / lookup_params:.0f}× fewer parameters")


if __name__ == "__main__":
    main()
