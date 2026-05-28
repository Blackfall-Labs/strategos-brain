"""Export smollm2_bands.json + cross_model_spectral_profile.json → brain_topology.json.

The topology JSON is what spectral-brain/src/builder.rs reads.

Tier 0 (Band 1): one cluster per trigger token.
  Each neuron in the cluster → one column.
  band_id = token vocabulary frequency rank (lower rank = more common = lower band ID).
  token_id pulled from smollm2_triggers.json.

Tier 1 (Bands 2-3): one sheet per spectral depth-bin covering depth 0.33-0.67.
  Columns per sheet derived from band neuron counts at that depth.
  band range = (depth_bin_idx * BAND_SPAN, depth_bin_idx * BAND_SPAN + BAND_SPAN).
  pv_scales sampled from neuron_freq_hist at that depth.

Tier 2 (Band 4 + always-on): sheets covering depth 0.67-1.00.
  always_on_frac from profile's always_on_frac field.
  Wide band (responds to anything Tier 1 broadcasts).

Usage:
    python tools/export_band_topology.py \\
        --bands smollm2_bands.json \\
        --profile cross_model_spectral_profile.json \\
        --triggers smollm2_triggers.json \\
        --output brain_topology.json
"""

import json
import argparse
from collections import defaultdict


MAX_COLS_PER_SHEET = 512   # cap per sheet
BAND_SPAN          = 32    # frequency range width per depth bin


def build_token_rank_map(triggers):
    token_counts = defaultdict(int)
    for layer in triggers["layers"]:
        for n in layer["neurons"]:
            for t in n["top_tokens"]:
                token_counts[t["token_id"]] += 1
    ranked = sorted(token_counts.keys(), key=lambda tid: -token_counts[tid])
    return {tid: rank for rank, tid in enumerate(ranked)}


def build_token_id_map(triggers):
    m = {}
    for layer in triggers["layers"]:
        for n in layer["neurons"]:
            for t in n["top_tokens"]:
                if t["token_str"] not in m:
                    m[t["token_str"]] = t["token_id"]
    return m


def hist_to_pv_scales(hist_fracs, n):
    """Sample n pv_scale values from a 10-bin neuron_freq histogram.

    hist_fracs: list of 10 fractions summing to ~1.0
    bin i covers fire_rate range [i/10, (i+1)/10].
    Low fire_rate → high pv_scale (harder to fire). Range [0.5, 2.0].
    """
    if not hist_fracs:
        return [1.0] * n

    pv_list = []
    for i, frac in enumerate(hist_fracs):
        mid_freq = (i + 0.5) / len(hist_fracs)
        pv = 2.0 - 1.5 * mid_freq    # freq=0 → pv=2.0, freq=1 → pv=0.5
        count = max(1, round(frac * n))
        pv_list.extend([pv] * count)

    if len(pv_list) >= n:
        return pv_list[:n]
    return pv_list + [1.0] * (n - len(pv_list))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bands",    required=True)
    parser.add_argument("--profile",  required=True)
    parser.add_argument("--triggers", required=True)
    parser.add_argument("--output",   required=True)
    args = parser.parse_args()

    print("[export] loading bands...")
    with open(args.bands, encoding="utf-8") as f:
        bands = json.load(f)

    print("[export] loading profile...")
    with open(args.profile, encoding="utf-8") as f:
        profile = json.load(f)

    print("[export] loading triggers...")
    with open(args.triggers, encoding="utf-8") as f:
        triggers = json.load(f)

    depth_profile = profile["depth_profile"]
    tok_str_to_id = build_token_id_map(triggers)
    tok_rank      = build_token_rank_map(triggers)

    # ── All L0 trigger tokens (3443 unique) ───────────────────────────────────
    # For every token that is a top_token for any L0 neuron with high TR,
    # record its best trigger_rate. Used later for extended injection coverage.
    l0_token_best_tr = {}   # token_id → best trigger_rate at L0
    for layer in triggers["layers"]:
        if layer["layer"] != 0:
            continue
        for n in layer["neurons"]:
            for t in n["top_tokens"]:
                tid = t["token_id"]
                tr  = t["trigger_rate"]
                if tr > l0_token_best_tr.get(tid, 0.0):
                    l0_token_best_tr[tid] = tr
    print("[export] L0 trigger tokens: %d" % len(l0_token_best_tr))

    # ── Band neuron counts per depth bin from smollm2 calibrated profile ─────
    # Compute approximate neuron counts per band at each depth bin by looking at
    # where Band 1/2/3/4 neurons actually sit (by layer), normalised to depth.
    n_layers = 30   # SmolLM2
    band_by_depth = defaultdict(lambda: defaultdict(int))  # depth_bin → band → count
    for band in bands["bands"]:
        b = band["band"]
        if b == 0:
            continue
        for cluster in band.get("clusters", []):
            for n in cluster.get("neurons", []):
                li = n["layer"]
                depth_bin = min(int(li / n_layers * len(depth_profile)), len(depth_profile) - 1)
                band_by_depth[depth_bin][b] += 1

    # ── Tier 0: Band 1 clusters ──────────────────────────────────────────────
    band1 = next(b for b in bands["bands"] if b["band"] == 1)
    tier0_clusters = []
    cluster_id = 0

    for cluster in band1["clusters"]:
        tstr = cluster["trigger_token"]
        tid  = tok_str_to_id.get(tstr) or tok_str_to_id.get(" " + tstr)
        if tid is None:
            for s, i in tok_str_to_id.items():
                if s.strip() == tstr.strip():
                    tid = i; break
        if tid is None:
            safe = tstr.encode("ascii", "replace").decode("ascii")
            print("[export]   SKIP tier0: no token_id for '%s'" % safe)
            continue

        band_id = tok_rank.get(tid, 9999)

        members = [
            {"neuron_idx": n["neuron"], "trigger_rate": n["trigger_rate"]}
            for n in cluster["neurons"]
            if n["trigger_rate"] >= 0.40
        ]
        if not members:
            continue

        safe = tstr.encode("ascii", "replace").decode("ascii")
        print("[export]   tier0 '%s' tok=%d band=%d cols=%d" % (safe, tid, band_id, len(members)))

        tier0_clusters.append({
            "cluster_id":    cluster_id,
            "trigger_token": tstr,
            "token_id":      tid,
            "band_id":       band_id,
            "members":       members,
        })
        cluster_id += 1

    print("[export] tier0: %d clusters" % len(tier0_clusters))

    # ── Band range for the whole system ─────────────────────────────────────
    # T0 emits at band_id = token_rank (0=most common, up to ~max_rank).
    # T1 must be tuned across [0, max_T0_band] to receive T0 broadcasts.
    # T2 is wide-band: band_lo=0, band_hi covers the full range.
    max_t0_band = max((c["band_id"] for c in tier0_clusters), default=256)
    MAX_BAND = max_t0_band + BAND_SPAN
    print("[export] T0 band range [0, %d]  MAX_BAND=%d" % (max_t0_band, MAX_BAND))

    # ── Tier 1: Bands 2-3, depth 0.33-0.67 ──────────────────────────────────
    # Divide [0, MAX_BAND] into slices across Tier 1 depth bins so T1 covers
    # the same frequency range that T0 emits into.
    tier1_bins = [(i, b) for i, b in enumerate(depth_profile)
                  if 0.33 <= b["depth_mid"] < 0.67]
    n_t1_bins = max(len(tier1_bins), 1)
    tier1_sheets = []
    sheet_id = 0

    for rank, (i, bin_) in enumerate(tier1_bins):
        hist = bin_.get("neuron_freq_hist", [])
        pv_scales = hist_to_pv_scales(hist, MAX_COLS_PER_SHEET)

        slice_lo = int(rank * MAX_BAND / n_t1_bins)
        slice_hi = int((rank + 1) * MAX_BAND / n_t1_bins)

        n2 = min(band_by_depth[i].get(2, 64), MAX_COLS_PER_SHEET)
        if n2 > 0:
            tier1_sheets.append({
                "sheet_id":       sheet_id,
                "tier":           1,
                "band_lo":        slice_lo,
                "band_hi":        slice_hi,
                "n_cols":         n2,
                "pv_scales":      pv_scales[:n2],
                "always_on_frac": 0.0,
            })
            sheet_id += 1

        n3 = min(band_by_depth[i].get(3, 128), MAX_COLS_PER_SHEET)
        if n3 > 0:
            overlap = (slice_hi - slice_lo) // 2
            tier1_sheets.append({
                "sheet_id":       sheet_id,
                "tier":           1,
                "band_lo":        max(0, slice_lo - overlap),
                "band_hi":        slice_hi + overlap,
                "n_cols":         n3,
                "pv_scales":      pv_scales[:n3],
                "always_on_frac": 0.0,
            })
            sheet_id += 1

    print("[export] tier1: %d sheets" % len(tier1_sheets))

    # ── Tier 2: Band 4 + always-on, depth 0.67-1.00 ─────────────────────────
    tier2_sheets = []

    for i, bin_ in enumerate(depth_profile):
        depth = bin_["depth_mid"]
        if depth < 0.67:
            continue

        hist = bin_.get("neuron_freq_hist", [])
        pv_scales = hist_to_pv_scales(hist, MAX_COLS_PER_SHEET)
        always_on_frac = float(bin_.get("always_on_frac", 0.0))

        n4 = min(band_by_depth[i].get(4, 32), MAX_COLS_PER_SHEET)
        if n4 > 0:
            tier2_sheets.append({
                "sheet_id":       sheet_id,
                "tier":           2,
                "band_lo":        0,
                "band_hi":        MAX_BAND,   # wide-band — receives from any tier
                "n_cols":         n4,
                "pv_scales":      pv_scales[:n4],
                "always_on_frac": always_on_frac,
            })
            sheet_id += 1

    print("[export] tier2: %d sheets" % len(tier2_sheets))

    # ── Band 2/3 token injection entries ────────────────────────────────────
    # For tokens that appear in Band 2 or 3 clusters but NOT in Band 1,
    # we still inject them into the bus at the appropriate T1 band range.
    # magnitude is scaled down (Band 2: 0.4-0.9 TR → ~15000; Band 3: 0.1-0.4 → ~6000).
    # band_id is set to the midpoint of the T1 slice for the token's depth bin.
    band1_token_ids = {c["token_id"] for c in tier0_clusters}
    band23_tokens = []
    seen_tids = set(band1_token_ids)

    # ── Extended L0 coverage: all tokens with strong L0 trigger_rate ─────────
    # Tokens that didn't make Band 1 clusters still carry L0 signal.
    # Inject them at band_id=tok_rank (same scheme as Band 1) but lower magnitude.
    for tid, tr in l0_token_best_tr.items():
        if tid in seen_tids or tr < 0.5:
            continue
        seen_tids.add(tid)
        band_id   = tok_rank.get(tid, 9999)
        magnitude = int(tr * 30000 * 0.7)   # 70% of Band 1 magnitude
        if magnitude < 500:
            continue
        band23_tokens.append({
            "token_id":  tid,
            "band":      1,      # treated as Band 1 signal, lower magnitude
            "band_id":   band_id,
            "magnitude": magnitude,
        })
    print("[export] extended L0 tokens added: %d" % len(band23_tokens))

    t1_slice_mids = []
    for rank, (i, _) in enumerate(tier1_bins):
        slice_lo = int(rank * MAX_BAND / n_t1_bins)
        slice_hi = int((rank + 1) * MAX_BAND / n_t1_bins)
        t1_slice_mids.append((i, (slice_lo + slice_hi) // 2))

    # Build depth_bin → T1 band_id midpoint lookup
    depth_bin_to_band = {i: mid for i, mid in t1_slice_mids}

    for band_obj in bands["bands"]:
        b = band_obj["band"]
        if b not in (2, 3):
            continue
        for cluster in band_obj.get("clusters", []):
            tstr = cluster["trigger_token"]
            tid  = tok_str_to_id.get(tstr) or tok_str_to_id.get(" " + tstr)
            if tid is None:
                for s, i in tok_str_to_id.items():
                    if s.strip() == tstr.strip():
                        tid = i; break
            if tid is None or tid in seen_tids:
                continue
            seen_tids.add(tid)

            # Pick the dominant depth bin for this cluster's neurons
            depth_counts = defaultdict(int)
            total_tr = 0.0
            for n in cluster.get("neurons", []):
                li = n["layer"]
                dbin = min(int(li / n_layers * len(depth_profile)), len(depth_profile) - 1)
                depth_counts[dbin] += 1
                total_tr += n.get("trigger_rate", 0.5)
            if not depth_counts:
                continue
            dominant_bin = max(depth_counts, key=depth_counts.get)
            band_id = depth_bin_to_band.get(dominant_bin, MAX_BAND // 2)

            n_neurons = len(cluster.get("neurons", []))
            mean_tr = total_tr / n_neurons if n_neurons else 0.5
            # Scale magnitude lower than Band 1 (Band 1 uses 30000 * TR)
            mag_scale = 0.5 if b == 2 else 0.25
            magnitude = int(mean_tr * 30000 * mag_scale)
            if magnitude < 500:
                continue

            band23_tokens.append({
                "token_id":  tid,
                "band":      b,
                "band_id":   band_id,
                "magnitude": magnitude,
            })

    print("[export] band23 tokens: %d" % len(band23_tokens))

    total_cols = (sum(len(c["members"]) for c in tier0_clusters)
                + sum(s["n_cols"] for s in tier1_sheets)
                + sum(s["n_cols"] for s in tier2_sheets))

    out = {
        "source":         "smollm2_bands + cross_model_spectral_profile",
        "n_cols_total":   total_cols,
        "tier0_clusters": tier0_clusters,
        "tier1_sheets":   tier1_sheets,
        "tier2_sheets":   tier2_sheets,
        "band23_tokens":  band23_tokens,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("[export] written: %s  (%d total columns)" % (args.output, total_cols))


if __name__ == "__main__":
    main()
