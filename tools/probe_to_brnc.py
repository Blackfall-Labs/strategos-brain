"""Write a BRNC v5 checkpoint seeded from Qwen2.5-0.5B weights.

The brain IS the model. Every weight matrix in the transformer is used.
No approximations. The full 24-layer forward pass runs for each vocab token.

Architecture:
  AfferentField  = top-K dimensions of h_final (post-24-layer hidden state) by
                   contrast score (deviation from population mean). Maximally
                   discriminative: each word selects dimensions unique to it.
                   Near-zero overlap between words by construction.
  Cross-sheet    = Direct L0→L23 skip connections: X's afferent neurons → Y's
                   dedicated L23 neurons, wired by the model's actual logit
                   predictions (embed_tokens.T @ h_final scoring).
  EfferentField  = Dedicated non-overlapping neuron blocks per word in L23.
                   Word Y occupies [HIDDEN + Y*n .. HIDDEN + (Y+1)*n].
  KCa per cell   = Band selectivity tier (B1=0, B2=128, B3=300, B4=450).

Single-token forward pass simplifications (exact, not approximations):
  - RoPE at position 0: cos(0)=1, sin(0)=0 → rotation = identity (skipped)
  - Attention with 1 token: softmax([x]) = 1.0 → attn_out = V (Q/K skipped)
  - num_kv_heads=1: all 7 Q heads share the single KV pair → tile(v, 7)

Weight file layout (manifest.json):
  weight_00000: embed_tokens.weight [151936, 896]
  weight_00001: model.norm.weight [896]  ← final RMSNorm (before lm_head)
  Layer i (0-23): weights at indices (2+9*i) through (10+9*i):
    +0 input_layernorm [896]
    +1 q_proj [896, 896]  (not loaded — Q has no effect on single-token output)
    +2 k_proj [128, 896]  (not loaded)
    +3 v_proj [128, 896]  (bias loaded separately from v_bias_{layer:02d}.bin)
    +4 o_proj [896, 896]
    +5 post_attention_layernorm [896]
    +6 gate_proj [4864, 896]
    +7 up_proj [4864, 896]
    +8 down_proj [896, 4864]

Reads:
  qwen25_bands.json  -- band assignment (KCa seeding)
  models/qwen25_05b/ -- model weight files

Writes:
  <output>.brnc -- BRNC v5 loadable by sigbrain::checkpoint::load()
"""

import json
import struct
import argparse
import sys
import os
from collections import defaultdict

N_SHEETS     = 24
# N_COLS is computed dynamically in main() from actual vocab size.
# It must be >= max(FFN_DIM, EFF_COL_OFFSET + V * neurons_per_word).
FFN_DIM      = 4864   # FFN intermediate size (also KCa neuron index range)
HIDDEN       = 896
NUM_HEADS    = 14      # Qwen2.5-0.5B: 14 Q attention heads
NUM_KV_HEADS = 2       # 2 KV heads (GQA: 7 Q groups per KV head)
HEAD_DIM     = 64      # HIDDEN / NUM_HEADS = 896 / 14

# eff_cols start after the hidden-state range so aff and eff never overlap.
EFF_COL_OFFSET = HIDDEN   # 896

KCA_BY_BAND = {1: 0, 2: 128, 3: 300, 4: 450}

MAGIC              = b"BRNC"
VERSION            = 5
PROJECTION_AMPL    = 3_000
CELL_THRESHOLD     = 10_000
KCA_DECAY          = 128
RESET_KCA_ON_TRAIN = 0
GATED              = 0


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _silu(x):
    import numpy as np
    return x / (1.0 + np.exp(-x))


def _rms_norm_batch(H, w):
    """Batched RMSNorm. H: [V, D], w: [D] → [V, D]."""
    import numpy as np
    rms = np.sqrt((H * H).mean(axis=1, keepdims=True) + 1e-6)
    return H / rms * w


# ---------------------------------------------------------------------------
# BPE helpers
# ---------------------------------------------------------------------------

def strip_bpe(token: str) -> str:
    return token.replace("Ġ", "").lower().strip()


def load_bpe_vocab(model_dir):
    tok_path = os.path.join(model_dir, "tokenizer.json")
    return json.load(open(tok_path, encoding="utf-8"))["model"]["vocab"]


def resolve_token_ids(vocab, bpe_vocab):
    return [bpe_vocab.get("Ġ" + w, bpe_vocab.get(w)) for w in vocab]


# ---------------------------------------------------------------------------
# Probe data helpers
# ---------------------------------------------------------------------------

def build_band_map(bands_data):
    band_map = {}
    for band_info in bands_data["bands"]:
        band_num = band_info["band"]
        if band_num == 0:
            continue
        for cluster in band_info.get("clusters", []):
            for n in cluster["neurons"]:
                key = (n["layer"], n["neuron"])
                if key not in band_map:
                    band_map[key] = band_num
    return band_map


def build_layer_token_neurons(bands_data, min_trigger_rate):
    by_layer_token = defaultdict(list)
    for band_info in bands_data["bands"]:
        if band_info["band"] == 0:
            continue
        for cluster in band_info.get("clusters", []):
            token = strip_bpe(cluster["trigger_token"])
            if not token or not token.isalpha():
                continue
            for n in cluster["neurons"]:
                if n["neuron"] >= FFN_DIM:
                    continue
                if n["trigger_rate"] < min_trigger_rate:
                    continue
                by_layer_token[(n["layer"], token)].append(
                    (n["neuron"], n["trigger_rate"])
                )
    cleaned = {}
    for key, entries in by_layer_token.items():
        best = {}
        for neuron, tr in entries:
            if neuron not in best or tr > best[neuron]:
                best[neuron] = tr
        cleaned[key] = sorted(best.items(), key=lambda x: -x[1])
    return cleaned


def build_vocab(layer_token_neurons, max_vocab):
    token_weight = defaultdict(int)
    for (layer, token), neurons in layer_token_neurons.items():
        token_weight[token] += len(neurons)
    top = sorted(token_weight, key=lambda w: -token_weight[w])[:max_vocab]
    top.sort()
    return top


# ---------------------------------------------------------------------------
# Full forward pass
# ---------------------------------------------------------------------------

def run_full_forward_pass(vocab, model_dir):
    """Run the exact Qwen2.5-0.5B single-token forward pass for each vocab word.

    Uses all 24 transformer layers: attention (V→o_proj) + SwiGLU FFN.
    Q and K projections are skipped because at position 0 with 1 token the
    attention output is exactly V regardless of Q and K values.

    Returns:
        h_finals: [V, HIDDEN] array of final hidden states per vocab word
        token_ids: list of BPE token IDs per word (None if not in tokenizer)
    """
    import numpy as np

    bpe_vocab = load_bpe_vocab(model_dir)
    token_ids = resolve_token_ids(vocab, bpe_vocab)
    V         = len(vocab)

    print(f"  Loading embed_tokens [151936x{HIDDEN}]...")
    embed      = np.fromfile(os.path.join(model_dir, "weight_00000.bin"),
                             dtype=np.float32).reshape(151936, HIDDEN)
    final_norm = np.fromfile(os.path.join(model_dir, "weight_00001.bin"),
                             dtype=np.float32)

    H = np.zeros((V, HIDDEN), dtype=np.float32)
    for i, tid in enumerate(token_ids):
        if tid is not None:
            H[i] = embed[tid]

    for layer in range(N_SHEETS):
        base = 2 + 9 * layer
        print(f"  Layer {layer:2d}...", end="", flush=True)

        def wt(offset, shape):
            return np.fromfile(
                os.path.join(model_dir, f"weight_{base + offset:05d}.bin"),
                dtype=np.float32,
            ).reshape(shape)

        ln1_w  = wt(0, (HIDDEN,))
        v_proj = wt(3, (NUM_KV_HEADS * HEAD_DIM, HIDDEN))
        o_proj = wt(4, (HIDDEN,   HIDDEN))
        ln2_w  = wt(5, (HIDDEN,))
        gate_w = wt(6, (FFN_DIM,  HIDDEN))
        up_w   = wt(7, (FFN_DIM,  HIDDEN))
        down_w = wt(8, (HIDDEN,   FFN_DIM))

        v_bias_path = os.path.join(model_dir, f"v_bias_{layer:02d}.bin")
        v_bias = np.fromfile(v_bias_path, dtype=np.float32) if os.path.exists(v_bias_path) else None

        h1         = _rms_norm_batch(H, ln1_w)
        v_out      = h1 @ v_proj.T                                    # [V, NUM_KV_HEADS*HEAD_DIM]
        if v_bias is not None:
            v_out  = v_out + v_bias
        # GQA: repeat each KV head for its 7 Q-groups → [V, NUM_HEADS*HEAD_DIM]
        num_kv_groups = NUM_HEADS // NUM_KV_HEADS                     # = 7
        v_expanded = np.repeat(
            v_out.reshape(V, NUM_KV_HEADS, HEAD_DIM),
            num_kv_groups, axis=1
        ).reshape(V, HIDDEN)
        attn_out   = v_expanded @ o_proj.T
        H          = H + attn_out

        h2           = _rms_norm_batch(H, ln2_w)
        gate         = h2 @ gate_w.T
        up           = h2 @ up_w.T
        intermediate = _silu(gate) * up

        H = H + (intermediate @ down_w.T)
        print(" done")

    h_finals = _rms_norm_batch(H, final_norm)  # [V, HIDDEN]

    covered = sum(1 for tid in token_ids if tid is not None)
    print(f"  {covered}/{V} words have tokenizer coverage")
    return h_finals, token_ids


# ---------------------------------------------------------------------------
# Discriminative afferent selection
# ---------------------------------------------------------------------------

def select_exclusive_hfinal_aff(h_finals, token_ids, max_neurons):
    """Assign each of the HIDDEN dimensions to exactly one word (its argmax owner).
    Guarantees zero overlap between any two words' afferent neurons.

    Each dimension d is owned by the word with the highest h_finals[i][d].
    If a word owns more than max_neurons dims, keep the top-max by activation.

    Returns: list of lists of column IDs, indices in [0, HIDDEN)
    """
    import numpy as np

    V = h_finals.shape[0]
    valid_rows = [i for i, tid in enumerate(token_ids) if tid is not None]

    if not valid_rows:
        return [[] for _ in range(V)]

    valid_h = h_finals[valid_rows]              # [N_valid, HIDDEN]
    owner_pos = np.argmax(valid_h, axis=0)      # [HIDDEN] — index into valid_rows

    # Bucket each dimension into its owner
    owned_dims = [[] for _ in range(len(valid_rows))]
    for d in range(HIDDEN):
        owned_dims[owner_pos[d]].append(d)

    # rank_matrix[pos, d] = rank of word pos in dimension d (0 = highest activation)
    rank_matrix = np.argsort(np.argsort(-valid_h, axis=0), axis=0)  # [N_valid, HIDDEN]

    aff_cols = [[] for _ in range(V)]
    for pos, i in enumerate(valid_rows):
        dims = owned_dims[pos]
        if len(dims) >= max_neurons:
            dims.sort(key=lambda d: -h_finals[i][d])
            aff_cols[i] = dims[:max_neurons]
        elif dims:
            aff_cols[i] = dims
        else:
            # Fallback: pick dims where this word has the best (lowest) rank
            ranks = rank_matrix[pos]                         # [HIDDEN]
            best  = np.argsort(ranks)[:max_neurons].tolist()
            aff_cols[i] = best

    owned_counts = [len(d) for d in owned_dims]
    print(f"  Dimension ownership: min={min(owned_counts)} max={max(owned_counts)} "
          f"mean={sum(owned_counts)/len(owned_counts):.1f} per word", flush=True)

    return aff_cols


# ---------------------------------------------------------------------------
# Logit-based efferent + cross-axon builder
# ---------------------------------------------------------------------------

def build_logit_predictions(vocab, h_finals, token_ids, model_dir, top_k=3):
    """Score each vocab word X against the model's predicted next-token
    distribution, returning the top-K within-vocab predictions per word.

    Uses embed_tokens as lm_head (tied weights in Qwen2.5).

    Returns: list of lists — predictions[X] = [(Y_idx, rank), ...] top-K
    """
    import numpy as np

    print(f"  Loading embed_tokens for logit scoring...")
    embed = np.fromfile(
        os.path.join(model_dir, "weight_00000.bin"), dtype=np.float32
    ).reshape(151936, HIDDEN)

    V = len(vocab)
    valid_indices = [(i, tid) for i, tid in enumerate(token_ids) if tid is not None]
    vocab_embed   = np.stack([embed[tid] for _, tid in valid_indices])  # [V_valid, HIDDEN]

    predictions = [[] for _ in range(V)]
    for X_idx in range(V):
        if token_ids[X_idx] is None:
            continue
        h       = h_finals[X_idx]
        scores  = vocab_embed @ h
        top_rel = np.argsort(scores)[-top_k:][::-1]
        for rank, rel_idx in enumerate(top_rel):
            Y_idx = valid_indices[rel_idx][0]
            predictions[X_idx].append((Y_idx, rank))

    return predictions


def build_logit_efferent_and_axons(vocab, token_ids, predictions, aff_cols,
                                   neurons_per_word, strength):
    """Build dedicated, non-overlapping eff_cols per word + logit-driven axons.

    Architecture:
      eff_cols[Y]: neurons [EFF_COL_OFFSET + Y*n .. +n) — uniquely assigned per word,
        starting after the HIDDEN-dim afferent range so there is no index collision.
      cross_axons: for each prediction X→Y (rank 0-2), wire every neuron in
        aff_cols[X] to every neuron in eff_cols[Y]. Strength is rank-weighted:
          rank-0 = strength, rank-1 = 2/3 strength, rank-2 = 1/3 strength.

    Since aff_cols are chosen by contrast scoring from h_finals, they are nearly
    word-unique. Combined with exclusive eff_cols, the signal is clean: only word X
    can charge its predicted Y's neurons.
    """
    V = len(vocab)

    eff_cols = [
        list(range(EFF_COL_OFFSET + Y_idx * neurons_per_word,
                   EFF_COL_OFFSET + (Y_idx + 1) * neurons_per_word))
        for Y_idx in range(V)
    ]

    strengths = [strength, max(1, strength * 2 // 3), max(1, strength // 3)]

    axons = []
    for X_idx in range(V):
        if token_ids[X_idx] is None:
            continue
        src_neurons = aff_cols[X_idx]
        if not src_neurons:
            continue
        for (Y_idx, rank) in predictions[X_idx]:
            s           = strengths[min(rank, len(strengths) - 1)]
            tgt_neurons = eff_cols[Y_idx]
            for src_cell in src_neurons:
                for tgt_cell in tgt_neurons:
                    axons.append((0, src_cell, N_SHEETS - 1, tgt_cell, s))

    return eff_cols, axons


# ---------------------------------------------------------------------------
# Binary writers
# ---------------------------------------------------------------------------

def write_u8(f, v):  f.write(struct.pack("<B", v))
def write_u16(f, v): f.write(struct.pack("<H", v))
def write_u32(f, v): f.write(struct.pack("<I", v))
def write_i32(f, v): f.write(struct.pack("<i", v))


def write_checkpoint(path, vocab, band_map, aff_cols, eff_cols, cross_axons, n_cols):
    with open(path, "wb") as f:
        f.write(MAGIC)
        write_u32(f, VERSION)
        write_u32(f, n_cols)
        write_u32(f, N_SHEETS)
        write_i32(f, PROJECTION_AMPL)

        write_u32(f, len(vocab))
        for word in vocab:
            b = word.encode("utf-8")
            write_u32(f, len(b))
            f.write(b)

        for aff, eff in zip(aff_cols, eff_cols):
            write_u16(f, len(aff))
            for col in aff:
                write_u32(f, col)
            write_u16(f, len(eff))
            for col in eff:
                write_u32(f, col)

        for layer_idx in range(N_SHEETS):
            write_i32(f, CELL_THRESHOLD)
            write_u16(f, KCA_DECAY)
            write_u8(f, RESET_KCA_ON_TRAIN)
            write_u8(f, GATED)

            kca = [0] * n_cols
            for (layer, neuron), band_num in band_map.items():
                if layer == layer_idx and neuron < n_cols:
                    kca[neuron] = KCA_BY_BAND.get(band_num, 0)
            write_u32(f, n_cols)
            for k in kca:
                write_u16(f, k)

            write_u32(f, 0)  # dir_count
            write_u32(f, 0)  # rel_count

        write_u32(f, len(cross_axons))
        for src_sheet, src_cell, tgt_sheet, tgt_cell, strength in cross_axons:
            write_u32(f, src_sheet)
            write_u32(f, src_cell)
            write_u32(f, tgt_sheet)
            write_u32(f, tgt_cell)
            write_u16(f, strength)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bands",            default="qwen25_bands.json")
    parser.add_argument("--output",           required=True)
    parser.add_argument("--max-vocab",        type=int,   default=4000)
    parser.add_argument("--min-trigger-rate", type=float, default=0.4)
    parser.add_argument("--max-aff-neurons",  type=int,   default=16,
                        help="Exclusive h_final dimensions per word for afferent (default: 16)")
    parser.add_argument("--max-eff-neurons",  type=int,   default=50,
                        help="Dedicated L23 neurons per word for efferent (default: 50)")
    parser.add_argument("--cross-strength",   type=int,   default=2000)
    parser.add_argument("--model-dir",        default="models/qwen25_05b")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    if args.max_aff_neurons > HIDDEN:
        print(f"ERROR: --max-aff-neurons {args.max_aff_neurons} exceeds HIDDEN={HIDDEN}.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.bands}...")
    with open(args.bands, encoding="utf-8") as f:
        bands_data = json.load(f)

    print("Building band map...")
    band_map = build_band_map(bands_data)
    print(f"  {len(band_map):,} band-assigned neurons")

    print(f"Building layer-token-neuron lookup (min_tr={args.min_trigger_rate})...")
    layer_token_neurons = build_layer_token_neurons(bands_data, args.min_trigger_rate)
    unique_tokens       = len(set(tok for _, tok in layer_token_neurons.keys()))
    total_entries       = sum(len(v) for v in layer_token_neurons.values())
    print(f"  {total_entries:,} entries, {unique_tokens} unique tokens")

    print(f"Building vocab (max {args.max_vocab})...")
    vocab = build_vocab(layer_token_neurons, args.max_vocab)
    print(f"  {len(vocab)} words: {', '.join(vocab[:10])}{'...' if len(vocab) > 10 else ''}")

    # N_COLS must accommodate both FFN KCa neurons and dedicated eff blocks
    n_cols = max(FFN_DIM, EFF_COL_OFFSET + len(vocab) * args.max_eff_neurons)
    eff_end = EFF_COL_OFFSET + len(vocab) * args.max_eff_neurons
    print(f"  Column layout: aff [0,{HIDDEN}), eff [{EFF_COL_OFFSET},{eff_end}), N_COLS={n_cols}")

    print(f"\nRunning full Qwen2.5-0.5B forward pass ({len(vocab)} tokens, {N_SHEETS} layers)...")
    h_finals, token_ids = run_full_forward_pass(vocab, args.model_dir)

    print(f"\nSelecting discriminative afferent neurons from final hidden states "
          f"(top-{args.max_aff_neurons} by contrast)...")
    aff_cols    = select_exclusive_hfinal_aff(h_finals, token_ids, args.max_aff_neurons)
    aff_covered = sum(1 for c in aff_cols if c)
    print(f"  {aff_covered}/{len(vocab)} words have afferent coverage")

    # Diagnose overlap
    all_aff_sets = [set(c) for c in aff_cols if c]
    if len(all_aff_sets) >= 2:
        import itertools
        max_overlap = 0
        for a, b in itertools.combinations(all_aff_sets, 2):
            max_overlap = max(max_overlap, len(a & b))
        print(f"  Max afferent overlap between any two words: {max_overlap} neurons")

    print(f"\nComputing within-vocab logit predictions (top-3)...")
    predictions = build_logit_predictions(vocab, h_finals, token_ids, args.model_dir, top_k=3)
    pred_covered = sum(1 for p in predictions if p)
    print(f"  {pred_covered}/{len(vocab)} words have model predictions")
    for w, preds in zip(vocab, predictions):
        if preds:
            pred_words = [vocab[Y_idx] for Y_idx, _ in preds[:3]]
            print(f"    {w} -> {pred_words}")

    print(f"\nBuilding logit-based efferent + axons "
          f"({args.max_eff_neurons} neurons/word, strength={args.cross_strength})...")
    eff_cols, cross_axons = build_logit_efferent_and_axons(
        vocab, token_ids, predictions, aff_cols,
        args.max_eff_neurons, args.cross_strength)
    eff_covered = sum(1 for c in eff_cols if c)
    print(f"  {eff_covered}/{len(vocab)} words have dedicated eff neurons")
    print(f"  {len(cross_axons):,} cross-axons (L0→L23)")

    print(f"\nWriting {args.output}...")
    write_checkpoint(args.output, vocab, band_map, aff_cols, eff_cols, cross_axons, n_cols)

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Done — {size_mb:.1f} MB")

    from collections import Counter
    band_counts = Counter(v for v in band_map.values())
    print("\nKCa seeding:")
    for band_num in sorted(band_counts):
        print(f"  Band {band_num}: {band_counts[band_num]:,} neurons → KCa={KCA_BY_BAND[band_num]}")


if __name__ == "__main__":
    main()
