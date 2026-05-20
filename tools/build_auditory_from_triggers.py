"""Build brain_seed_auditory.json from SmolLM2 trigger analysis.

Converts the Band 1 (pure-token detector) clusters from the invariant extractor
into the SparseTokenResonator format that sigcortex reads.

The mapping:
  Each Band 1 cluster = a specific trigger token with high-confidence L0 neurons.
  Each L0 neuron in the cluster = a wernicke column that fires for that token.
  Column index = neuron index within the FFN hidden layer (0..1535 → 0..1535 in 2K).
  Magnitude = trigger_rate scaled to u16 range [0, 30000].
  Polarity = +1 (always excitatory — pure-token detectors are positive activations).

This directly encodes what SmolLM2 actually does at L0: when token T arrives,
the Band 1 neurons that fire for T correspond to wernicke columns that activate.
The wiring is not invented — it is measured from the real model.

Output format (SparseTokenResonator):
  {
    "n_columns": 2000,
    "token_strings": {"tok_id": "str", ...},
    "tokens": {"tok_id": [(col_idx, polarity, magnitude), ...], ...}
  }

Usage:
    python tools/build_auditory_from_triggers.py \\
        --triggers smollm2_triggers.json \\
        --bands smollm2_bands.json \\
        --output E:/repos/blackfall-labs/ternsig/brain_seed_auditory_xmodel.json \\
        --n-columns 2000 \\
        --l0-only
"""

import json
import argparse
from collections import defaultdict


MAG_SCALE = 30_000   # trigger_rate=1.0 -> magnitude 30000 (fits u16)
TOP_N = 40           # max columns per token burst (SparseTokenResonator reads top-N)


def build_token_str_map(triggers):
    """Build token_str -> token_id map from trigger data."""
    tok_map = {}
    for layer in triggers["layers"]:
        for n in layer["neurons"]:
            for t in n["top_tokens"]:
                s = t["token_str"]
                tid = t["token_id"]
                if s not in tok_map:
                    tok_map[s] = tid
    return tok_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triggers",   required=True)
    parser.add_argument("--bands",      required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--n-columns",  type=int, default=2000)
    parser.add_argument("--l0-only",    action="store_true",
                        help="Only use L0 neurons (pure embedding-dominant detectors)")
    parser.add_argument("--max-layer",  type=int, default=3,
                        help="Max layer to include neurons from (default 3, early recognition only)")
    args = parser.parse_args()

    print("[auditory] loading triggers...")
    with open(args.triggers, encoding="utf-8") as f:
        triggers = json.load(f)

    print("[auditory] loading bands...")
    with open(args.bands, encoding="utf-8") as f:
        bands = json.load(f)

    # Build token string -> token_id lookup from trigger data
    tok_str_to_id = build_token_str_map(triggers)
    print("[auditory] %d distinct token strings in trigger data" % len(tok_str_to_id))

    # Build neuron -> trigger_rate at L0 lookup
    # Format: (layer, neuron) -> {token_id -> trigger_rate}
    neuron_top_tr = defaultdict(dict)
    max_layer = 0 if args.l0_only else args.max_layer
    for layer in triggers["layers"]:
        li = layer["layer"]
        if li > max_layer:
            continue
        for n in layer["neurons"]:
            for t in n["top_tokens"]:
                neuron_top_tr[(li, n["neuron"])][t["token_id"]] = t["trigger_rate"]

    # Find Band 1 clusters
    band1 = next((b for b in bands["bands"] if b["band"] == 1), None)
    if band1 is None:
        print("[auditory] ERROR: no Band 1 found in bands file")
        return

    print("[auditory] %d Band 1 clusters" % len(band1["clusters"]))

    token_bursts = {}   # token_id -> [(col_idx, polarity, magnitude)]
    token_strings = {}  # token_id -> str
    n_mapped = 0
    n_missing = 0

    for cluster in band1["clusters"]:
        trigger_str = cluster["trigger_token"]

        # Look up token_id
        token_id = tok_str_to_id.get(trigger_str)
        # Also try with leading space
        if token_id is None:
            token_id = tok_str_to_id.get(" " + trigger_str)
        if token_id is None:
            # Search by suffix
            for s, tid in tok_str_to_id.items():
                if s.strip() == trigger_str.strip():
                    token_id = tid
                    break
        if token_id is None:
            safe = trigger_str.encode("ascii", "replace").decode("ascii")
            print("[auditory]   SKIP: no token_id for '%s'" % safe)
            n_missing += 1
            continue

        token_strings[token_id] = trigger_str

        # Collect neurons from early layers (recognition neurons only)
        burst_entries = []
        for n in cluster["neurons"]:
            li = n["layer"]
            ni = n["neuron"]
            if li > max_layer:
                continue
            if ni >= args.n_columns:
                continue  # column out of range for 2K wernicke

            tr = n["trigger_rate"]
            mag = min(int(tr * MAG_SCALE), 65535)
            if mag < 1000:
                continue  # skip very weak entries

            burst_entries.append((ni, 1, mag))

        if not burst_entries:
            safe = trigger_str.encode("ascii", "replace").decode("ascii")
            print("[auditory]   SKIP: no valid columns for '%s' (token %d)" % (safe, token_id))
            n_missing += 1
            continue

        # Sort by magnitude descending, keep top-N
        burst_entries.sort(key=lambda x: -x[2])
        burst_entries = burst_entries[:TOP_N]

        token_bursts[token_id] = burst_entries
        n_mapped += 1
        safe = trigger_str.encode("ascii", "replace").decode("ascii")
        print("[auditory]   token %5d '%s': %d columns, max_mag=%d" % (
              token_id, safe, len(burst_entries), burst_entries[0][2]))

    print("[auditory] %d tokens mapped, %d skipped" % (n_mapped, n_missing))

    out = {
        "n_columns":     args.n_columns,
        "source":        "smollm2_band1_triggers",
        "n_tokens":      n_mapped,
        "max_layer":     max_layer,
        "token_strings": {str(k): v for k, v in token_strings.items()},
        "tokens":        {str(k): v for k, v in token_bursts.items()},
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("[auditory] written: %s" % args.output)
    print("[auditory] %d tokens, up to %d cols each, source: Band 1 L0-%d neurons" % (
          n_mapped, TOP_N, max_layer))


if __name__ == "__main__":
    main()
