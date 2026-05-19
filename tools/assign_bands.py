"""Spectral band assignment for FFN neurons.

Reads a trigger profile (from extract binary) and calibrated activation profile
(from probe binary) and assigns each selective neuron to a frequency band.

Band taxonomy (by trigger_rate of top token and fire_rate):
  Band 0 — always-on:     fire_rate > 0.95  (fires on nearly every token)
  Band 1 — pure-token:    top trigger_rate > 0.90  (deterministic on specific tokens)
  Band 2 — selective:     top trigger_rate 0.40-0.90  (strong preference, context modulated)
  Band 3 — contextual:    top trigger_rate 0.10-0.40  (token matters, context matters more)
  Band 4 — diffuse:       top trigger_rate < 0.10  (responds weakly to many tokens)

For Band 1 neurons, clusters are defined by top trigger token identity —
neurons that fire on the same token belong to the same cluster.

For Bands 2-4, clusters are defined by Jaccard similarity of their top-10
token sets. Neurons sharing >= 3 of their top-10 tokens are in the same cluster.

Usage:
    python tools/assign_bands.py --model smollm2 \\
        --triggers smollm2_triggers.json \\
        --profile smollm2_profile_calibrated.json \\
        --output smollm2_bands.json
"""

import json
import argparse
from collections import defaultdict


BAND_ALWAYS_ON   = 0
BAND_PURE_TOKEN  = 1
BAND_SELECTIVE   = 2
BAND_CONTEXTUAL  = 3
BAND_DIFFUSE     = 4

BAND_NAMES = {
    0: "always-on",
    1: "pure-token",
    2: "selective",
    3: "contextual",
    4: "diffuse",
}

BAND_THRESHOLDS = {
    "always_on_fire_rate":   0.95,
    "pure_token_tr":         0.90,
    "selective_tr_low":      0.40,
    "contextual_tr_low":     0.10,
}


def assign_band(fire_rate: float, top_tr: float) -> int:
    t = BAND_THRESHOLDS
    if fire_rate > t["always_on_fire_rate"]:
        return BAND_ALWAYS_ON
    if top_tr > t["pure_token_tr"]:
        return BAND_PURE_TOKEN
    if top_tr > t["selective_tr_low"]:
        return BAND_SELECTIVE
    if top_tr > t["contextual_tr_low"]:
        return BAND_CONTEXTUAL
    return BAND_DIFFUSE


def cluster_band1(neurons_by_layer):
    """Cluster Band 1 neurons by top trigger token."""
    clusters = defaultdict(list)
    for layer, neuron_id, fire_rate, top_tr, top_tokens in neurons_by_layer:
        if not top_tokens:
            continue
        tok_str = top_tokens[0]["token_str"]
        clusters[tok_str].append({
            "layer": layer,
            "neuron": neuron_id,
            "fire_rate": round(fire_rate, 4),
            "trigger_rate": round(top_tr, 4),
        })
    return [
        {"trigger_token": tok, "n_neurons": len(ns), "neurons": ns}
        for tok, ns in sorted(clusters.items(), key=lambda x: -len(x[1]))
    ]


def cluster_by_top_token(neurons_by_layer):
    """Cluster neurons by their top trigger token.

    Simple and interpretable: two neurons in the same cluster share the same
    primary preferred token. Avoids false merges from shared common function words.
    """
    clusters = defaultdict(list)
    for layer, neuron_id, fire_rate, top_tr, top_tokens in neurons_by_layer:
        tok_str = top_tokens[0]["token_str"] if top_tokens else "<none>"
        clusters[tok_str].append({
            "layer": layer,
            "neuron": neuron_id,
            "fire_rate": round(fire_rate, 4),
            "trigger_rate": round(top_tr, 4),
        })
    return [
        {"trigger_token": tok, "n_neurons": len(ns), "neurons": ns}
        for tok, ns in sorted(clusters.items(), key=lambda x: -len(x[1]))
    ]


def build_fire_rate_index(profile):
    """Build (layer, neuron) → fire_rate from calibrated profile."""
    index = {}
    for layer in profile["layers"]:
        l = layer["layer"]
        nf = layer["ops"].get("ffn_hidden", {}).get("neuron_freq")
        if nf:
            for ni, rate in enumerate(nf):
                index[(l, ni)] = rate
    return index


def main():
    parser = argparse.ArgumentParser(description="Spectral band assignment")
    parser.add_argument("--model", required=True)
    parser.add_argument("--triggers", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"[bands] loading {args.triggers}...")
    with open(args.triggers, encoding="utf-8") as f:
        triggers = json.load(f)

    print(f"[bands] loading {args.profile}...")
    with open(args.profile, encoding="utf-8") as f:
        profile = json.load(f)

    fire_rate_index = build_fire_rate_index(profile)
    n_tokens = triggers["n_tokens"]

    # Collect all neurons with their assignments
    band_neurons = {b: [] for b in range(5)}

    for layer in triggers["layers"]:
        l = layer["layer"]
        for n in layer["neurons"]:
            ni = n["neuron"]
            fire_rate = fire_rate_index.get((l, ni), n["fire_rate"])
            top_tr = n["top_tokens"][0]["trigger_rate"] if n["top_tokens"] else 0.0
            band = assign_band(fire_rate, top_tr)
            band_neurons[band].append((l, ni, fire_rate, top_tr, n["top_tokens"]))

    # Report counts
    total = sum(len(v) for v in band_neurons.values())
    print(f"\n[bands] {total} total selective neurons across {len(triggers['layers'])} layers")
    for b, name in BAND_NAMES.items():
        cnt = len(band_neurons[b])
        print(f"  Band {b} ({name:12s}): {cnt:6d}  ({cnt/total*100:.1f}%)")

    # Build output
    print("\n[bands] clustering...")

    band0_out = [
        {"layer": l, "neuron": ni, "fire_rate": round(fr, 4)}
        for l, ni, fr, tr, _ in band_neurons[BAND_ALWAYS_ON]
    ]

    band1_clusters = cluster_band1(band_neurons[BAND_PURE_TOKEN])
    print(f"  Band 1: {len(band1_clusters)} clusters from {len(band_neurons[BAND_PURE_TOKEN])} neurons")

    band2_clusters = cluster_by_top_token(band_neurons[BAND_SELECTIVE])
    print(f"  Band 2: {len(band2_clusters)} clusters from {len(band_neurons[BAND_SELECTIVE])} neurons")

    band3_clusters = cluster_by_top_token(band_neurons[BAND_CONTEXTUAL])
    print(f"  Band 3: {len(band3_clusters)} clusters from {len(band_neurons[BAND_CONTEXTUAL])} neurons")

    band4_clusters = cluster_by_top_token(band_neurons[BAND_DIFFUSE])
    print(f"  Band 4: {len(band4_clusters)} clusters from {len(band_neurons[BAND_DIFFUSE])} neurons")

    out = {
        "model": args.model,
        "n_tokens": n_tokens,
        "total_selective_neurons": total,
        "bands": [
            {
                "band": BAND_ALWAYS_ON,
                "name": BAND_NAMES[BAND_ALWAYS_ON],
                "description": "fire_rate > 0.95 — unconditional, fires on nearly every token",
                "n_neurons": len(band0_out),
                "neurons": band0_out,
            },
            {
                "band": BAND_PURE_TOKEN,
                "name": BAND_NAMES[BAND_PURE_TOKEN],
                "description": "top trigger_rate > 0.90 — deterministic on specific token(s)",
                "n_neurons": len(band_neurons[BAND_PURE_TOKEN]),
                "n_clusters": len(band1_clusters),
                "clusters": band1_clusters,
            },
            {
                "band": BAND_SELECTIVE,
                "name": BAND_NAMES[BAND_SELECTIVE],
                "description": "top trigger_rate 0.40-0.90 — strong preference, context modulated",
                "n_neurons": len(band_neurons[BAND_SELECTIVE]),
                "n_clusters": len(band2_clusters),
                "clusters": band2_clusters,
            },
            {
                "band": BAND_CONTEXTUAL,
                "name": BAND_NAMES[BAND_CONTEXTUAL],
                "description": "top trigger_rate 0.10-0.40 — token matters, context matters more",
                "n_neurons": len(band_neurons[BAND_CONTEXTUAL]),
                "n_clusters": len(band3_clusters),
                "clusters": band3_clusters,
            },
            {
                "band": BAND_DIFFUSE,
                "name": BAND_NAMES[BAND_DIFFUSE],
                "description": "top trigger_rate < 0.10 — responds weakly to many tokens, no strong preference",
                "n_neurons": len(band_neurons[BAND_DIFFUSE]),
                "n_clusters": len(band4_clusters),
                "clusters": band4_clusters,
            },
        ],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n[bands] written to {args.output}")


if __name__ == "__main__":
    main()
