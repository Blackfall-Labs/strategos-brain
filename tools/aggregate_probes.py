"""Cross-model probe aggregator.

Normalizes activation profiles from GPT-2, SmolLM2, Qwen3, and Qwen2.5-0.5B
to relative depth [0.0, 1.0] and averages the structural metrics at each
depth point.

The output is a depth-indexed spectral profile: the architecture-invariant
laws of language computation that all models converge on. This profile seeds
the sigcortex brain regions — not copying any model's weights, but encoding
the structural attractor that training finds regardless of architecture.

Structural laws extracted:
  - mean_head_entropy(depth): how diffuse attention is (broad=early, narrow=mid)
  - mean_neuron_freq(depth): mean FFN activation rate (~10% everywhere)
  - freq_distribution(depth): histogram of neuron fire rates per depth band
  - residual_std(depth): signal energy in residual stream
  - always_on_fraction(depth): fraction of always-on neurons (>90% fire rate)

Output: cross_model_spectral_profile.json

Usage:
    python tools/aggregate_probes.py \\
        --gpt2    gpt2_profile.json \\
        --smollm2 smollm2_profile_calibrated.json \\
        --qwen3   qwen3_profile_calibrated.json \\
        --qwen25  qwen25_profile.json \\
        --output  cross_model_spectral_profile.json \\
        --depth-bins 20
"""

import json
import math
import argparse
from collections import defaultdict


def load_profile(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalized_layers(profile):
    """Yield (relative_depth, layer_data) for each layer in the profile."""
    layers = profile["layers"]
    n = len(layers)
    for i, layer in enumerate(layers):
        depth = i / max(n - 1, 1)   # 0.0 at first layer, 1.0 at last
        yield depth, layer


def head_entropy_stats(layer):
    """Mean and min head entropy for this layer, normalized to [0,1].

    Entropy stored in attn_weights.head_entropy as list of per-head values (nats).
    Normalize by ln(64) ~ 4.16 (approximate max entropy for ~64-token windows).
    """
    ops = layer.get("ops", {})
    attn = ops.get("attn_weights", {})
    entropies = attn.get("head_entropy", [])
    if not entropies:
        return None, None
    norm = math.log(64)
    normed = [min(e / norm, 1.0) for e in entropies]
    return sum(normed) / len(normed), min(normed)


def neuron_freq_stats(layer):
    """Distribution of FFN neuron fire rates for this layer."""
    ops = layer.get("ops", {})
    ffn = ops.get("ffn_hidden", {})
    freqs = ffn.get("neuron_freq", [])
    if not freqs:
        return None
    n = len(freqs)
    mean_freq = sum(freqs) / n
    always_on = sum(1 for f in freqs if f > 0.90) / n
    near_dead  = sum(1 for f in freqs if f < 0.01) / n
    # Histogram: 10 buckets 0.0-1.0
    hist = [0] * 10
    for f in freqs:
        bucket = min(int(f * 10), 9)
        hist[bucket] += 1
    hist_norm = [h / n for h in hist]
    return {
        "mean": mean_freq,
        "always_on": always_on,
        "near_dead": near_dead,
        "hist": hist_norm,
    }


def residual_std(layer):
    ops = layer.get("ops", {})
    return ops.get("residual", {}).get("std", None)


def assign_depth_bin(depth, n_bins):
    return min(int(depth * n_bins), n_bins - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt2",    required=True)
    parser.add_argument("--smollm2", required=True)
    parser.add_argument("--qwen3",   required=True)
    parser.add_argument("--qwen25",  required=False, default=None,
                        help="Path to Qwen2.5-0.5B profile JSON (optional, adds 4th model)")
    parser.add_argument("--output",  required=True)
    parser.add_argument("--depth-bins", type=int, default=20,
                        help="Number of depth bins to aggregate into (default: 20)")
    args = parser.parse_args()

    profiles = [
        ("gpt2",    load_profile(args.gpt2)),
        ("smollm2", load_profile(args.smollm2)),
        ("qwen3",   load_profile(args.qwen3)),
    ]
    if args.qwen25:
        profiles.append(("qwen25", load_profile(args.qwen25)))

    # Detect uncalibrated neuron_freq: if the global mean fire rate is > 0.5,
    # the threshold was not properly calibrated (GPT-2 GELU fires on ~97% of neurons
    # with threshold=0.01). Exclude those models from neuron_freq aggregation but
    # keep their head entropy (which is always calibration-independent).
    calibrated_neuron_freq = set()
    for model_name, profile in profiles:
        all_freqs = []
        for _, layer in normalized_layers(profile):
            freqs = layer.get("ops", {}).get("ffn_hidden", {}).get("neuron_freq", [])
            if freqs:
                all_freqs.extend(freqs)
        if all_freqs:
            global_mean = sum(all_freqs) / len(all_freqs)
            calibrated = global_mean < 0.5
            print("[agg] %s: global mean neuron_freq=%.3f -> %s" % (
                model_name, global_mean,
                "calibrated (included)" if calibrated else "UNCALIBRATED (neuron_freq excluded)"))
            if calibrated:
                calibrated_neuron_freq.add(model_name)

    N = args.depth_bins

    # Accumulators per depth bin
    head_entropy_mean_acc = defaultdict(list)   # bin -> [values]
    head_entropy_min_acc  = defaultdict(list)
    neuron_freq_acc       = defaultdict(list)   # bin -> [mean_freq values]
    always_on_acc         = defaultdict(list)
    near_dead_acc         = defaultdict(list)
    hist_acc              = defaultdict(list)   # bin -> [[hist values]]
    residual_acc          = defaultdict(list)

    for model_name, profile in profiles:
        n_layers = len(profile["layers"])
        use_nf = model_name in calibrated_neuron_freq
        print("[agg] %s: %d layers, neuron_freq=%s" % (model_name, n_layers, "yes" if use_nf else "no"))
        for depth, layer in normalized_layers(profile):
            b = assign_depth_bin(depth, N)

            hem, he_min = head_entropy_stats(layer)
            if hem is not None:
                head_entropy_mean_acc[b].append(hem)
                head_entropy_min_acc[b].append(he_min)

            if use_nf:
                nf = neuron_freq_stats(layer)
                if nf is not None:
                    neuron_freq_acc[b].append(nf["mean"])
                    always_on_acc[b].append(nf["always_on"])
                    near_dead_acc[b].append(nf["near_dead"])
                    hist_acc[b].append(nf["hist"])

            rs = residual_std(layer)
            if rs is not None:
                residual_acc[b].append(rs)

    def avg(lst):
        return sum(lst) / len(lst) if lst else None

    def avg_hist(hists):
        if not hists:
            return None
        n_buckets = len(hists[0])
        return [sum(h[i] for h in hists) / len(hists) for i in range(n_buckets)]

    depth_profile = []
    for b in range(N):
        depth_mid = (b + 0.5) / N
        entry = {
            "bin":        b,
            "depth_mid":  round(depth_mid, 3),
            "n_layers":   len(head_entropy_mean_acc[b]),

            # Head attention entropy [0=spiky, 1=uniform]
            "head_entropy_mean": avg(head_entropy_mean_acc[b]),
            "head_entropy_min":  avg(head_entropy_min_acc[b]),

            # FFN activation frequency
            "neuron_freq_mean":  avg(neuron_freq_acc[b]),
            "always_on_frac":    avg(always_on_acc[b]),
            "near_dead_frac":    avg(near_dead_acc[b]),
            "neuron_freq_hist":  avg_hist(hist_acc[b]),

            # Residual stream energy
            "residual_std_mean": avg(residual_acc[b]),
        }
        depth_profile.append(entry)

        if entry["head_entropy_mean"] is not None:
            print("  bin %2d (depth %.2f): entropy=%.3f  min_entropy=%.3f  neuron_freq=%.3f  always_on=%.4f" % (
                b, depth_mid,
                entry["head_entropy_mean"] or 0,
                entry["head_entropy_min"] or 0,
                entry["neuron_freq_mean"] or 0,
                entry["always_on_frac"] or 0,
            ))

    # Derive sigcortex seeding parameters for each depth bin
    print("\n[agg] deriving sigcortex parameters...")

    for entry in depth_profile:
        hem = entry["head_entropy_mean"]
        nfm = entry["neuron_freq_mean"]
        aof = entry["always_on_frac"]

        if hem is None:
            entry["sigcortex"] = None
            continue

        # homeostasis target: match the observed mean neuron fire rate
        homeostasis = round(nfm or 0.10, 4)

        # charge_decay: tight decay for high-entropy (diffuse) layers — they need
        # faster turnover. Spiky attention layers (low entropy) sustain signal longer.
        # Map: entropy 0.0 -> decay 0.3 (slow, sustained), entropy 1.0 -> decay 0.1 (fast, reset)
        charge_decay = round(0.3 - 0.2 * (hem or 0.5), 3)
        charge_decay = max(0.05, min(0.7, charge_decay))

        # region interval: functional zone (mid depth, low entropy) gets tighter interval
        # Map: entropy < 0.3 -> interval 2, entropy < 0.5 -> interval 3, else -> interval 4
        hem_min = entry["head_entropy_min"] or hem
        if hem_min < 0.25:
            interval = 2   # tight functional zone
        elif hem_min < 0.45:
            interval = 3
        else:
            interval = 4   # broad early/late layers

        # spontaneous rate: higher where always_on neurons are dense (late layers)
        # Scale 0-400 mapped from always_on 0-5%
        spontaneous_rate = int(min(aof or 0, 0.05) * 8000)

        # pv_scale distribution: sample from the observed neuron_freq histogram
        # High-fire neurons -> low pv_scale (0.5 = easier to fire)
        # Low-fire neurons  -> high pv_scale (2.0 = harder to fire)
        # We encode this as a (min, max, shape) tuple; the seeder will sample from it
        entry["sigcortex"] = {
            "homeostasis":       homeostasis,
            "charge_decay":      charge_decay,
            "interval":          interval,
            "spontaneous_rate":  spontaneous_rate,
            # pv_scale distribution parameters derived from neuron_freq_hist
            "pv_scale_min":      0.5,
            "pv_scale_max":      2.0,
            # Use the neuron_freq histogram inverted as the pv_scale sampling distribution:
            # neurons that fire rarely get high pv_scale (high threshold = hard to fire)
            "pv_scale_hist": [round(entry["neuron_freq_hist"][9 - i], 4)
                               if entry["neuron_freq_hist"] else 0.1
                               for i in range(10)],
        }

    out = {
        "models":       [name for name, _ in profiles],
        "n_depth_bins": N,
        "depth_profile": depth_profile,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n[agg] written to %s" % args.output)
    print("[agg] %d depth bins, cross-model structural profile complete" % N)


if __name__ == "__main__":
    main()
