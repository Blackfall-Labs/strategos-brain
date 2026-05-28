"""Show Band 1 (pure-token) neuron concentration per layer."""
import json

with open("qwen25_bands.json") as f:
    bands = json.load(f)

band1 = bands["bands"][1]  # Band 1
print(f"Band 1 total: {band1['n_neurons']} neurons, {band1['n_clusters']} token clusters")

# Count neurons per layer
from collections import defaultdict
layer_counts = defaultdict(int)
for cluster in band1["clusters"]:
    for n in cluster["neurons"]:
        layer_counts[n["layer"]] += 1

print("\nLayer  Band1_count")
for l in sorted(layer_counts):
    bar = "#" * (layer_counts[l] // 20)
    print(f"  L{l:02d}   {layer_counts[l]:4d}  {bar}")

print("\nTop 10 token clusters:")
for c in band1["clusters"][:10]:
    print(f"  '{c['trigger_token']}': {c['n_neurons']} neurons")
