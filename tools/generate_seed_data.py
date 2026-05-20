"""Generate sigcortex seed files from the cross-model spectral profile.

Converts cross_model_spectral_profile.json into the exact seed JSON format
that sigcortex reads: pv_thresholds and lang_affinities.

The three brain regions map to depth ranges of the cross-model profile:
  wernicke -> early depth (0.0-0.33): broad attention, uniform sparse coding
  semantic -> middle depth (0.33-0.67): functional zone, tighter attention
  pfc      -> late depth (0.67-1.0): compositional, always-on emerging

pv_scale per column (threshold scaling):
  Sampled from the neuron_freq histogram at the region's depth midpoint.
  High fire_rate -> low pv_scale (easier to fire, low threshold)
  Low fire_rate  -> high pv_scale (harder to fire, high threshold)
  Formula: pv_scale = 2.0 - 1.5 * sampled_fire_rate  [range 0.5-2.0]

attn_k per column (plasticity window width):
  Derived from mean head entropy at the region's depth.
  Low entropy (tight/functional)  -> low k (tight plasticity, fast convergence)
  High entropy (broad/structural) -> high k (broad plasticity, slow convergence)
  Formula: k = int(k_min + (k_max - k_min) * entropy_mean)

No LLM-dimension-specific connections are written. Columns learn specific
dimensional affinities through plasticity once the brain runs on text.

Output files (written to --out-dir):
  brain_seed_pv_thresholds_xmodel.json
  brain_seed_lang_affinities_xmodel.json

Usage:
    python tools/generate_seed_data.py \\
        --profile cross_model_spectral_profile.json \\
        --out-dir E:/repos/blackfall-labs/ternsig \\
        --wernicke-cols 2000 \\
        --semantic-cols 2000 \\
        --pfc-cols 4000
"""

import json
import math
import argparse
import random
import os


REGION_DEPTH_RANGES = {
    "wernicke": (0.00, 0.33),
    "semantic":  (0.33, 0.67),
    "pfc":       (0.67, 1.00),
}

K_MIN = 200
K_MAX = 5000
PV_MIN = 0.5
PV_MAX = 2.0


def load_profile(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def bins_for_depth_range(profile, lo, hi):
    """Return all depth bins whose midpoint falls within [lo, hi)."""
    return [b for b in profile["depth_profile"]
            if lo <= b["depth_mid"] < hi]


def weighted_sample_fire_rate(hist_norm):
    """Sample a fire_rate value from the 10-bucket histogram."""
    if not hist_norm:
        return 0.10
    r = random.random()
    cumulative = 0.0
    for i, w in enumerate(hist_norm):
        cumulative += w
        if r <= cumulative:
            # Return midpoint of bucket i (buckets span 0.0-1.0 in steps of 0.1)
            return (i + 0.5) / 10.0
    return 0.95


def pv_scale_from_fire_rate(fire_rate):
    """High fire_rate -> low threshold (pv_scale); low fire_rate -> high threshold."""
    scale = PV_MAX - (PV_MAX - PV_MIN) * fire_rate
    return round(max(PV_MIN, min(PV_MAX, scale)), 4)


def attn_k_from_entropy(entropy_mean):
    """Low entropy -> tight plasticity (low k); high entropy -> broad (high k)."""
    if entropy_mean is None:
        return (K_MIN + K_MAX) // 2
    k = K_MIN + int((K_MAX - K_MIN) * entropy_mean)
    return max(K_MIN, min(K_MAX, k))


def region_stats(bins):
    """Aggregate stats for a set of depth bins."""
    entropies = [b["head_entropy_mean"] for b in bins if b["head_entropy_mean"] is not None]
    min_entropies = [b["head_entropy_min"] for b in bins if b["head_entropy_min"] is not None]
    freqs = [b["neuron_freq_mean"] for b in bins if b["neuron_freq_mean"] is not None]
    always_on = [b["always_on_frac"] for b in bins if b["always_on_frac"] is not None]
    intervals = [b["sigcortex"]["interval"] for b in bins if b.get("sigcortex")]
    decays = [b["sigcortex"]["charge_decay"] for b in bins if b.get("sigcortex")]

    # Aggregate histogram across bins
    hist_sum = None
    hist_count = 0
    for b in bins:
        h = b.get("neuron_freq_hist")
        if h:
            if hist_sum is None:
                hist_sum = [0.0] * len(h)
            for i, v in enumerate(h):
                hist_sum[i] += v
            hist_count += 1
    hist_avg = [v / hist_count for v in hist_sum] if hist_sum else None

    def avg(lst):
        return sum(lst) / len(lst) if lst else None

    return {
        "entropy_mean":  avg(entropies),
        "entropy_min":   avg(min_entropies),
        "freq_mean":     avg(freqs),
        "always_on":     avg(always_on),
        "interval":      round(avg(intervals)) if intervals else 4,
        "charge_decay":  round(avg(decays), 3) if decays else 0.2,
        "hist":          hist_avg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile",       required=True, help="cross_model_spectral_profile.json")
    parser.add_argument("--out-dir",       required=True, help="Output directory (ternsig root)")
    parser.add_argument("--wernicke-cols", type=int, default=2000)
    parser.add_argument("--semantic-cols", type=int, default=2000)
    parser.add_argument("--pfc-cols",      type=int, default=4000)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    profile = load_profile(args.profile)

    region_col_counts = {
        "wernicke": args.wernicke_cols,
        "semantic":  args.semantic_cols,
        "pfc":       args.pfc_cols,
    }

    total_cols = sum(region_col_counts.values())
    print("[seed] total columns: %d" % total_cols)

    # --- Build pv_thresholds ---
    pv_scales = {}
    pv_stats = {}
    for region, (lo, hi) in REGION_DEPTH_RANGES.items():
        bins = bins_for_depth_range(profile, lo, hi)
        stats = region_stats(bins)
        n_cols = region_col_counts[region]
        pv_stats[region] = stats

        # Sample pv_scale for each column from the region's neuron_freq histogram
        scales = []
        for _ in range(n_cols):
            fire_rate = weighted_sample_fire_rate(stats["hist"])
            scales.append(pv_scale_from_fire_rate(fire_rate))

        pv_scales[region] = scales
        mean_scale = sum(scales) / len(scales)
        print("[seed] %s: %d cols, depth [%.2f-%.2f], entropy=%.3f, "
              "mean_pv_scale=%.3f, interval=%d, decay=%.3f" % (
              region, n_cols, lo, hi,
              stats["entropy_mean"] or 0, mean_scale,
              stats["interval"], stats["charge_decay"]))

    # dim_remap: sequential mapping (column i -> dimension i)
    # Plasticity will refine this; we start with full-coverage uniform assignment
    dim_remap = list(range(total_cols))

    pv_out = {
        "format":       "astromind_pv_thresholds_v1",
        "source":       "cross_model_spectral_xmodel",
        "hidden_dim":   total_cols,
        "wernicke_cells": args.wernicke_cols,
        "scale_min":    PV_MIN,
        "scale_max":    PV_MAX,
        "nope_layers":  [],
        "groups": {
            "wernicke": list(range(args.wernicke_cols)),
            "semantic":  list(range(args.semantic_cols)),
            "pfc":       list(range(args.pfc_cols)),
        },
        "dim_remap": dim_remap,
        "scales": pv_scales,
        "depth_ranges": {r: list(v) for r, v in REGION_DEPTH_RANGES.items()},
        "cross_model_stats": {r: {k: v for k, v in s.items() if k != "hist"}
                               for r, s in pv_stats.items()},
    }

    pv_path = os.path.join(args.out_dir, "brain_seed_pv_thresholds_xmodel.json")
    with open(pv_path, "w", encoding="utf-8") as f:
        json.dump(pv_out, f, indent=2)
    print("[seed] pv_thresholds written: %s" % pv_path)

    # --- Build lang_affinities ---
    # No LLM-dimension-specific connections. Instead, write uniform coverage:
    # each column is assigned top_k=1 "connection" pointing to its own index
    # at the region's depth-derived attn_k. Plasticity replaces these.
    connections = {}
    for region, (lo, hi) in REGION_DEPTH_RANGES.items():
        bins = bins_for_depth_range(profile, lo, hi)
        stats = region_stats(bins)
        k = attn_k_from_entropy(stats["entropy_mean"])
        n_cols = region_col_counts[region]

        # [col_idx, source_dim, attn_k] — uniform self-seeding as placeholder
        conns = [[i, i % total_cols, k] for i in range(n_cols)]
        connections[region] = conns
        print("[seed] %s: attn_k=%d (from entropy=%.3f)" % (
              region, k, stats["entropy_mean"] or 0))

    la_out = {
        "format":        "astromind_lang_affinities_v1",
        "source":        "cross_model_spectral_xmodel",
        "hidden_dim":    total_cols,
        "wernicke_cells": args.wernicke_cols,
        "top_k":         1,
        "k_min":         K_MIN,
        "nope_layers":   [],
        "groups": {
            "wernicke": list(range(args.wernicke_cols)),
            "semantic":  list(range(args.semantic_cols)),
            "pfc":       list(range(args.pfc_cols)),
        },
        "dim_remap": dim_remap,
        "connections": connections,
        "note": ("Connections are uniform placeholder — col_idx=source_dim, "
                 "attn_k from cross-model head entropy. Plasticity will reshape."),
    }

    la_path = os.path.join(args.out_dir, "brain_seed_lang_affinities_xmodel.json")
    with open(la_path, "w", encoding="utf-8") as f:
        json.dump(la_out, f, indent=2)
    print("[seed] lang_affinities written: %s" % la_path)

    print("\n[seed] done. Load with:")
    print("  brain_seed_pv_thresholds_xmodel.json")
    print("  brain_seed_lang_affinities_xmodel.json")
    print("  (replace existing brain_seed_*.json references in brain_daemon.rs)")


if __name__ == "__main__":
    main()
