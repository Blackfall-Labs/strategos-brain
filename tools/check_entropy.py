import json, math
with open("qwen25_profile_calibrated.json") as f:
    p = json.load(f)
norm = math.log(64)
print("Layer  min_entropy  mean_entropy  depth")
n = len(p["layers"])
for layer in p["layers"]:
    l = layer["layer"]
    depth = l / max(n - 1, 1)
    attn = layer["ops"].get("attn_weights", {})
    ents = attn.get("head_entropy", [])
    if ents:
        normed = [min(e / norm, 1.0) for e in ents]
        mn = min(normed)
        me = sum(normed) / len(normed)
        print(f"  L{l:02d}  ({depth:.2f})  min={mn:.4f}  mean={me:.4f}")
