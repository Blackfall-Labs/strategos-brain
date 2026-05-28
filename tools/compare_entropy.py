"""Quick cross-model entropy depth comparison."""
import json, math, sys

profiles = [
    ("smollm2", "smollm2_profile.json"),
    ("qwen3",   "qwen3_profile_calibrated.json"),
    ("qwen25",  "qwen25_profile_calibrated.json"),
]

norm = math.log(64)

for name, path in profiles:
    try:
        with open(path) as f:
            p = json.load(f)
    except FileNotFoundError:
        print(f"{name}: NOT FOUND ({path})")
        continue

    layers = p["layers"]
    n = len(layers)
    min_entropy_by_layer = []
    for layer in layers:
        l = layer["layer"]
        depth = l / max(n - 1, 1)
        ents = layer["ops"].get("attn_weights", {}).get("head_entropy", [])
        if ents:
            normed = [min(e / norm, 1.0) for e in ents]
            min_entropy_by_layer.append((l, depth, min(normed)))

    if not min_entropy_by_layer:
        print(f"{name}: no head entropy data")
        continue

    # Find the layer with the global minimum
    best = min(min_entropy_by_layer, key=lambda x: x[2])
    print(f"{name} ({n} layers): depth_min={best[1]:.2f} at L{best[0]:02d}  min_entropy={best[2]:.4f}")
    print(f"  Top-5 spikiest layers:")
    for l, d, me in sorted(min_entropy_by_layer, key=lambda x: x[2])[:5]:
        print(f"    L{l:02d} (depth={d:.2f})  min_entropy={me:.4f}")
    print()
