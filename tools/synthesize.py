"""Cortical column synthesis — frequency band assignment.

For each selective neuron (Bands 1-3), assigns a frequency band based on the
vocabulary frequency rank of its trigger tokens. This directly encodes the
spectral model:

  rare trigger tokens  (low BPE rank)  -> high oscillation frequency band
  common trigger tokens (high BPE rank) -> low oscillation frequency band

Band assignment is O(n) per neuron — no clustering required. The frequency
band number is the quantile bucket of the neuron's mean trigger-token rank
across N_BANDS equally-spaced buckets.

Output: smollm2_column_spec.json
  {
    "model": "smollm2",
    "n_freq_bands": <int>,
    "layers": [
      {
        "layer": <int>,
        "n_neurons": <int>,
        "neurons": [
          {
            "neuron":      <int>,
            "input_band":  <int>,   // spectral band (1=pure-token, 2=selective, 3=contextual)
            "freq_band":   <int>,   // oscillation frequency band (0=lowest, N-1=highest)
            "vocab_rank":  <float>, // mean BPE-order rank of top trigger tokens (0=rare, 1=common)
            "fire_rate":   <float>,
            "n_triggers":  <int>
          }
        ]
      }
    ]
  }

Usage:
    python tools/synthesize.py \\
        --triggers smollm2_triggers.json \\
        --bands smollm2_bands.json \\
        --tokenizer models/smollm2/tokenizer.json \\
        --output smollm2_column_spec.json \\
        --n-bands 32
"""

import json
import math
import argparse


def build_corpus_freq_rank(triggers):
    """Map token_id -> normalized corpus frequency rank [0=rarest, 1=most-common].

    Estimates token frequency from the trigger profile: for each (neuron, token)
    pair, fire_count / trigger_rate approximates how many times that token appeared
    in the corpus. We take the max estimate across all neurons that observed the
    token, then rank all tokens.
    """
    token_freq = {}  # token_id -> estimated corpus occurrences
    for layer in triggers["layers"]:
        for n in layer["neurons"]:
            for t in n["top_tokens"]:
                tid = t["token_id"]
                tr = t["trigger_rate"]
                fc = t["fire_count"]
                if tr > 0:
                    est = fc / tr
                    if tid not in token_freq or est > token_freq[tid]:
                        token_freq[tid] = est

    # Sort by frequency: rank 0 = rarest, rank n-1 = most common
    sorted_tokens = sorted(token_freq, key=lambda t: token_freq[t])
    n = len(sorted_tokens)
    return {tok_id: rank / max(n - 1, 1) for rank, tok_id in enumerate(sorted_tokens)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triggers",   required=True)
    parser.add_argument("--bands",      required=True)
    parser.add_argument("--tokenizer",  default=None, help="(unused, reserved)")
    parser.add_argument("--output",     required=True)
    parser.add_argument("--n-bands",    type=int, default=32,
                        help="Number of oscillation frequency bands (default: 32)")
    parser.add_argument("--input-bands", nargs="*", type=int, default=[1, 2, 3],
                        help="Spectral input bands to include (default: 1 2 3)")
    args = parser.parse_args()

    with open(args.triggers, encoding="utf-8") as f:
        triggers = json.load(f)
    with open(args.bands, encoding="utf-8") as f:
        bands_data = json.load(f)

    freq_rank = build_corpus_freq_rank(triggers)
    print("[synth] corpus freq rank: %d distinct tokens, n_bands: %d" % (len(freq_rank), args.n_bands))

    # Build (layer, neuron) -> spectral input band
    neuron_input_band = {}
    for band in bands_data["bands"]:
        bn = band["band"]
        if bn not in args.input_bands:
            continue
        for cluster in band.get("clusters", []):
            for n in cluster.get("neurons", []):
                neuron_input_band[(n["layer"], n["neuron"])] = bn

    print("[synth] %d neurons in input bands %s" % (len(neuron_input_band), args.input_bands))

    output_layers = []
    total_neurons = 0

    for layer_data in triggers["layers"]:
        li = layer_data["layer"]
        out_neurons = []

        for n in layer_data["neurons"]:
            ni = n["neuron"]
            key = (li, ni)
            if key not in neuron_input_band:
                continue
            if not n["top_tokens"]:
                continue

            # Mean vocab rank of top trigger tokens (weighted by trigger_rate)
            total_tr = sum(t["trigger_rate"] for t in n["top_tokens"])
            if total_tr == 0:
                continue

            mean_rank = sum(
                freq_rank.get(t["token_id"], 0.5) * t["trigger_rate"]
                for t in n["top_tokens"]
            ) / total_tr

            # Assign frequency band: rare tokens -> high band, common tokens -> low band
            # mean_rank near 1 = rare, near 0 = common
            freq_band = min(int(mean_rank * args.n_bands), args.n_bands - 1)

            out_neurons.append({
                "neuron":     ni,
                "input_band": neuron_input_band[key],
                "freq_band":  freq_band,
                "vocab_rank": round(mean_rank, 4),
                "fire_rate":  round(n["fire_rate"], 5),
                "n_triggers": len(n["top_tokens"]),
            })

        if not out_neurons:
            continue

        out_neurons.sort(key=lambda x: (x["freq_band"], x["neuron"]))

        # Count neurons per freq band
        band_counts = [0] * args.n_bands
        for x in out_neurons:
            band_counts[x["freq_band"]] += 1
        occupied = sum(1 for c in band_counts if c > 0)

        output_layers.append({
            "layer":         li,
            "n_neurons":     len(out_neurons),
            "n_occupied_bands": occupied,
            "band_counts":   band_counts,
            "neurons":       out_neurons,
        })
        total_neurons += len(out_neurons)

    print("[synth] %d neurons assigned across %d layers" % (total_neurons, len(output_layers)))

    # Aggregate band_counts across all layers
    global_band_counts = [0] * args.n_bands
    for layer in output_layers:
        for b, c in enumerate(layer["band_counts"]):
            global_band_counts[b] += c

    print("[synth] global freq band distribution:")
    for b in range(args.n_bands):
        c = global_band_counts[b]
        if c > 0:
            bar = "#" * (c * 40 // max(global_band_counts))
            rarity = "rare " if b >= args.n_bands // 2 else "common"
            print("  band %2d (%s): %5d  %s" % (b, rarity, c, bar))

    out = {
        "model":          "smollm2",
        "n_freq_bands":   args.n_bands,
        "n_neurons":      total_neurons,
        "input_bands":    args.input_bands,
        "global_band_counts": global_band_counts,
        "layers":         output_layers,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("[synth] written to %s" % args.output)


if __name__ == "__main__":
    main()
