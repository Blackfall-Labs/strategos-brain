"""
Export the grokking model checkpoint to manifest.json + .bin format.
Source: progress-measures-paper/saved_runs/low_precision_mod_addition.pth

Usage: C:\Python313\python.exe tools/export_grokking.py
"""
import torch, json, struct, os

SRC = "E:/repos/progress-measures-paper/saved_runs/low_precision_mod_addition.pth"
OUT = "E:/repos/blackfall-labs/strategos-brain/models/grokking"
os.makedirs(OUT, exist_ok=True)

d = torch.load(SRC, map_location="cpu", weights_only=False)
model = d["model"]

manifest = []
for name, tensor in model.items():
    t = tensor.float().cpu()
    flat = t.reshape(-1).tolist()
    fname = name.replace(".", "_") + ".bin"
    with open(os.path.join(OUT, fname), "wb") as f:
        f.write(struct.pack(f"{len(flat)}f", *flat))
    manifest.append({"name": name, "file": fname, "shape": list(t.shape), "numel": len(flat)})
    print(f"  {name}: {list(t.shape)}")

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

config = {
    "model": "grokking-1l",
    "P": 113,
    "d_model": 128,
    "d_mlp": 512,
    "n_heads": 4,
    "head_dim": 32,
    "vocab_size": 114,
    "seq_len": 3,
    "has_ln": False,
    "final_train_loss": d["train_losses"][-1],
    "final_test_loss": d["test_losses"][-1],
}
with open(os.path.join(OUT, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"\nExported {len(manifest)} tensors to {OUT}/")
