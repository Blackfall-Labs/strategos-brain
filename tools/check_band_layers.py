import json, sys
from collections import defaultdict

with open("qwen25_bands.json") as f:
    bands = json.load(f)

print(f"Total selective neurons: {bands['total_selective_neurons']}")
print(f"{'Layer':<6} {'B0':>5} {'B1':>5} {'B2':>6} {'B3':>6} {'B4':>5}  B1%   B2%   B3%")

layer_counts = {b: defaultdict(int) for b in range(5)}

for band_idx, band in enumerate(bands["bands"]):
    key = "neurons" if band_idx == 0 else "clusters"
    if band_idx == 0:
        for n in band["neurons"]:
            layer_counts[0][n["layer"]] += 1
    else:
        for cluster in band["clusters"]:
            for n in cluster["neurons"]:
                layer_counts[band_idx][n["layer"]] += 1

n_layers = 24
for l in range(n_layers):
    b0 = layer_counts[0][l]
    b1 = layer_counts[1][l]
    b2 = layer_counts[2][l]
    b3 = layer_counts[3][l]
    b4 = layer_counts[4][l]
    total = b0 + b1 + b2 + b3 + b4
    if total == 0:
        continue
    p1 = b1 / total * 100
    p2 = b2 / total * 100
    p3 = b3 / total * 100
    print(f"L{l:02d}   {b0:5d} {b1:5d} {b2:6d} {b3:6d} {b4:5d}  {p1:4.1f}% {p2:4.1f}% {p3:4.1f}%")
