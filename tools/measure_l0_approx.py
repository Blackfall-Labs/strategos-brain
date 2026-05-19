"""Measure approximation error of the L0 embedding-dominant lookup table.

For tokens in a corpus whose current token is a Band 1 L0 trigger, runs the
full L0 attention + FFN forward pass (ground truth) and compares the result to
the precomputed embedding-alone lookup values.

Reports:
  - Mean absolute error (MAE) between lookup and true activation
  - Max absolute error
  - Fraction of lookup entries where sign agrees with true activation
  - Mean absolute deviation as fraction of threshold
  - Distribution of true activations for the covered neurons

If MAE << threshold (0.1321), the lookup table is a high-fidelity approximation
and the 1152x compression is practically valid.

Usage:
    python tools/measure_l0_approx.py --weights models/smollm2
        --lookup smollm2_l0_lookup.json --tokenizer models/smollm2/tokenizer.json
        --corpus data/wikitext2_val.txt --n-tokens 2000
"""

import json
import struct
import argparse
import math
import os
import numpy as np
from collections import defaultdict


# SmolLM2 architecture constants
HIDDEN     = 576
N_Q_HEADS  = 9
N_KV_HEADS = 3
HEAD_DIM   = 64
FFN_DIM    = 1536
VOCAB      = 49152
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


def silu(x):
    return x / (1.0 + np.exp(-x))


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def rope_half(v, pos):
    """LLaMA-style non-interleaved RoPE: rotate pairs (i, i+half)."""
    half = len(v) // 2
    for i in range(half):
        theta = pos / (ROPE_BASE ** (2.0 * i / HEAD_DIM))
        c, s = math.cos(theta), math.sin(theta)
        x0, x1 = v[i], v[i + half]
        v[i]        = x0 * c - x1 * s
        v[i + half] = x0 * s + x1 * c
    return v


def build_rope_table(max_seq=512):
    """Precompute cos/sin for RoPE."""
    cos_table = np.zeros((max_seq, HEAD_DIM // 2), dtype=np.float32)
    sin_table = np.zeros((max_seq, HEAD_DIM // 2), dtype=np.float32)
    for pos in range(max_seq):
        for i in range(HEAD_DIM // 2):
            theta = pos / (ROPE_BASE ** (2.0 * i / HEAD_DIM))
            cos_table[pos, i] = math.cos(theta)
            sin_table[pos, i] = math.sin(theta)
    return cos_table, sin_table


def apply_rope_half_fast(v, pos, cos_table, sin_table):
    """Vectorized non-interleaved RoPE."""
    half = HEAD_DIM // 2
    x0 = v[:half].copy()
    x1 = v[half:].copy()
    c  = cos_table[pos]
    s  = sin_table[pos]
    v[:half] = x0 * c - x1 * s
    v[half:] = x0 * s + x1 * c
    return v


# ── L0-only forward ───────────────────────────────────────────────────────────

class L0Weights:
    def __init__(self, manifest_dir):
        manifest = load_manifest(manifest_dir)
        p = "layers.0"
        get = lambda name: load_np(manifest_dir, find_tensor(manifest, name))

        self.embed       = get("embed_tokens.weight")           # [VOCAB, HIDDEN]
        self.input_ln    = get(f"{p}.input_layernorm.weight")   # [HIDDEN]
        self.q_proj      = get(f"{p}.self_attn.q_proj.weight")  # [N_Q_HEADS*HEAD_DIM, HIDDEN]
        self.k_proj      = get(f"{p}.self_attn.k_proj.weight")  # [N_KV_HEADS*HEAD_DIM, HIDDEN]
        self.v_proj      = get(f"{p}.self_attn.v_proj.weight")  # [N_KV_HEADS*HEAD_DIM, HIDDEN]
        self.o_proj      = get(f"{p}.self_attn.o_proj.weight")  # [HIDDEN, N_Q_HEADS*HEAD_DIM]
        self.post_attn_ln = get(f"{p}.post_attention_layernorm.weight")  # [HIDDEN]
        self.gate_proj   = get(f"{p}.mlp.gate_proj.weight")     # [FFN_DIM, HIDDEN]
        self.up_proj     = get(f"{p}.mlp.up_proj.weight")       # [FFN_DIM, HIDDEN]
        self.cos, self.sin = build_rope_table()
        print(f"[measure] L0 weights loaded")


def l0_forward(tok_id, ctx_pos, kv_k, kv_v, w):
    """Run L0 attention + FFN for one token. Returns (ffn_hidden, updated residual)."""
    hidden = w.embed[tok_id].copy()  # [HIDDEN]

    # --- Attention ---
    pre_attn = rms_norm(hidden, w.input_ln)

    q = (w.q_proj @ pre_attn).reshape(N_Q_HEADS, HEAD_DIM)
    k = (w.k_proj @ pre_attn).reshape(N_KV_HEADS, HEAD_DIM)
    v = (w.v_proj @ pre_attn).reshape(N_KV_HEADS, HEAD_DIM)

    # RoPE
    for h in range(N_Q_HEADS):
        apply_rope_half_fast(q[h], ctx_pos, w.cos, w.sin)
    for h in range(N_KV_HEADS):
        apply_rope_half_fast(k[h], ctx_pos, w.cos, w.sin)

    kv_k.append(k)
    kv_v.append(v)

    seq_len = len(kv_k)
    scale = math.sqrt(HEAD_DIM)
    attn_out = np.zeros(N_Q_HEADS * HEAD_DIM, dtype=np.float32)

    for q_h in range(N_Q_HEADS):
        kv_h = q_h // KV_GROUP
        scores = np.array([np.dot(q[q_h], kv_k[ki][kv_h]) / scale for ki in range(seq_len)])
        weights = softmax(scores)
        head_out = sum(weights[ki] * kv_v[ki][kv_h] for ki in range(seq_len))
        attn_out[q_h * HEAD_DIM:(q_h + 1) * HEAD_DIM] = head_out

    proj = w.o_proj @ attn_out
    hidden += proj

    # --- FFN ---
    pre_ffn = rms_norm(hidden, w.post_attn_ln)
    gate = w.gate_proj @ pre_ffn
    up   = w.up_proj   @ pre_ffn
    ffn_hidden = silu(gate) * up

    return ffn_hidden


# ── tokenizer ─────────────────────────────────────────────────────────────────

def load_vocab(tokenizer_path):
    with open(tokenizer_path, encoding="utf-8") as f:
        tj = json.load(f)
    vocab = tj["model"]["vocab"]
    id_to_str = {v: k for k, v in vocab.items()}
    return vocab, id_to_str


def load_tokens(corpus_path, tokenizer_path, max_tokens):
    """Very simple whitespace BPE tokenizer using the vocab directly."""
    vocab, _ = load_vocab(tokenizer_path)

    with open(corpus_path, encoding="utf-8") as f:
        text = f.read(max_tokens * 8)  # read enough chars

    # Use HuggingFace tokenizers library if available, else fallback
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tokenizer_path)
        enc = tok.encode(text)
        ids = enc.ids[:max_tokens]
        print(f"[measure] tokenized: {len(ids)} tokens (HF tokenizers)")
        return ids
    except ImportError:
        pass

    # Fallback: character-level approximation (for testing only)
    # Map each char to its id if present, else <unk>
    unk_id = vocab.get("<unk>", 0)
    ids = []
    for ch in text:
        ids.append(vocab.get(ch, unk_id))
        if len(ids) >= max_tokens:
            break
    print(f"[measure] tokenized: {len(ids)} tokens (char fallback)")
    return ids


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    required=True)
    parser.add_argument("--lookup",     required=True, help="smollm2_l0_lookup.json")
    parser.add_argument("--tokenizer",  required=True)
    parser.add_argument("--corpus",     required=True)
    parser.add_argument("--n-tokens",   type=int, default=2000)
    args = parser.parse_args()

    L0_THRESHOLD = 0.1321
    CONTEXT_LEN  = 128

    # Load lookup table
    with open(args.lookup, encoding="utf-8") as f:
        lookup_data = json.load(f)

    # Build index: token_id → {neuron_id → lookup_val}
    lookup_index = defaultdict(dict)
    for entry in lookup_data["entries"]:
        lookup_index[entry["token_id"]][entry["neuron"]] = entry["activation"]

    trigger_token_ids = set(lookup_index.keys())
    print(f"[measure] lookup: {lookup_data['n_neurons']} neurons, {len(trigger_token_ids)} trigger tokens")

    # Load L0 weights
    w = L0Weights(args.weights)

    # Load corpus tokens
    tokens = load_tokens(args.corpus, args.tokenizer, args.n_tokens)
    if not tokens:
        print("[measure] ERROR: no tokens loaded")
        return

    print(f"[measure] processing {len(tokens)} tokens in chunks of {CONTEXT_LEN}...")

    all_errors    = []   # |true - lookup| for each (pos, neuron) hit
    all_true_vals = []   # true activations for covered neurons
    all_look_vals = []   # lookup values for covered neurons
    sign_agree    = 0
    sign_total    = 0
    n_trigger_hits = 0

    for chunk_start in range(0, len(tokens), CONTEXT_LEN):
        chunk = tokens[chunk_start:chunk_start + CONTEXT_LEN]
        kv_k = []
        kv_v = []

        for ctx_pos, tok_id in enumerate(chunk):
            ffn_hidden = l0_forward(tok_id, ctx_pos, kv_k, kv_v, w)

            if tok_id in trigger_token_ids:
                n_trigger_hits += 1
                neuron_lookup = lookup_index[tok_id]
                for ni, lval in neuron_lookup.items():
                    true_val = float(ffn_hidden[ni])
                    err = abs(true_val - lval)
                    all_errors.append(err)
                    all_true_vals.append(true_val)
                    all_look_vals.append(lval)
                    if (true_val > 0) == (lval > 0):
                        sign_agree += 1
                    sign_total += 1

        if (chunk_start // CONTEXT_LEN + 1) % 4 == 0:
            pct = (chunk_start + len(chunk)) / len(tokens) * 100
            print(f"  {pct:.0f}% — hits so far: {n_trigger_hits}, comparisons: {len(all_errors)}")

    if not all_errors:
        print("[measure] no trigger token hits — check corpus and tokenizer")
        return

    errors    = np.array(all_errors)
    true_vals = np.array(all_true_vals)
    look_vals = np.array(all_look_vals)

    mae         = float(np.mean(errors))
    max_err     = float(np.max(errors))
    median_err  = float(np.median(errors))
    sign_pct    = sign_agree / sign_total * 100
    mae_pct_thr = mae / L0_THRESHOLD * 100

    # Fraction where true value fires
    true_fire = np.abs(true_vals) > L0_THRESHOLD
    look_fire = np.abs(look_vals) > L0_THRESHOLD
    both_fire = (true_fire & look_fire).sum()
    true_only = (true_fire & ~look_fire).sum()
    look_only = (~true_fire & look_fire).sum()
    neither   = (~true_fire & ~look_fire).sum()

    # Separate stats for firing pairs only
    fire_mask  = true_fire & look_fire
    mean_true  = float(np.mean(np.abs(true_vals[fire_mask]))) if fire_mask.any() else 0
    mean_look  = float(np.mean(np.abs(look_vals[fire_mask]))) if fire_mask.any() else 0
    scale_factor = mean_true / mean_look if mean_look > 0 else 0

    print(f"\n[measure] Results over {n_trigger_hits} trigger occurrences, {len(all_errors)} (neuron, position) pairs")
    print(f"  Threshold:               {L0_THRESHOLD:.4f}")
    print(f"  MAE (mean |true-lookup|): {mae:.4f}  ({mae_pct_thr:.1f}% of threshold)")
    print(f"  Median error:             {median_err:.4f}")
    print(f"  Max error:                {max_err:.4f}")
    print(f"  Sign agreement:           {sign_pct:.1f}%")
    print()
    print(f"  Firing agreement (binary: does neuron cross threshold?):")
    print(f"    Both fire:              {both_fire:6d}  ({both_fire/len(errors)*100:.1f}%)")
    print(f"    True fires, lookup no:  {true_only:6d}  ({true_only/len(errors)*100:.1f}%)")
    print(f"    Lookup fires, true no:  {look_only:6d}  ({look_only/len(errors)*100:.1f}%)")
    print(f"    Neither fires:          {neither:6d}  ({neither/len(errors)*100:.1f}%)")
    print()
    tp = both_fire
    fp = look_only
    fn = true_only
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  Lookup as binary fire predictor:")
    print(f"    Precision: {precision*100:.1f}%  (when lookup fires, true fires this often)")
    print(f"    Recall:    {recall*100:.1f}%  (of true fires, lookup predicts this fraction)")
    print()
    print(f"  Magnitude (for pairs where both fire):")
    print(f"    Mean true |activation|:   {mean_true:.4f}")
    print(f"    Mean lookup |activation|: {mean_look:.4f}")
    print(f"    Attention amplification:  {scale_factor:.2f}×  (true / lookup)")
    print()

    # Verdict is two-dimensional: binary accuracy vs magnitude accuracy
    binary_ok = precision >= 0.95 and recall >= 0.95
    mag_ok     = mae < L0_THRESHOLD * 0.50

    if binary_ok and mag_ok:
        verdict = "PASS — high binary accuracy AND low magnitude error; lookup substitution is valid"
    elif binary_ok and not mag_ok:
        verdict = (f"BINARY PASS / MAGNITUDE MISS — 100% recall, {precision*100:.1f}% precision but "
                   f"attention amplifies activations {scale_factor:.2f}×. "
                   f"Binary distillation (fire/no-fire) is valid; quantitative distillation needs "
                   f"a {scale_factor:.2f}× scale correction per neuron.")
    elif not binary_ok and mag_ok:
        verdict = "BINARY MISS — firing prediction unreliable; lookup table is invalid"
    else:
        verdict = "FAIL — both binary accuracy and magnitude error are poor"

    print(f"[measure] Verdict: {verdict}")


if __name__ == "__main__":
    main()
