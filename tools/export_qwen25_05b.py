"""Export Qwen2.5-0.5B weights to strategos manifest+bin format.

Downloads from HuggingFace (Qwen/Qwen2.5-0.5B) and exports the tensors
needed for the activation prober. All weights kept as float32.

Usage:
    C:\\Python313\\python.exe tools\\export_qwen25_05b.py
"""

import json, os, glob, shutil
import numpy as np
from safetensors import safe_open
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-0.5B"
OUT_DIR  = "models/qwen25_05b"
N_LAYERS = 24

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Downloading {MODEL_ID}...")
local_dir = snapshot_download(
    repo_id=MODEL_ID,
    ignore_patterns=["*.msgpack", "*.h5", "*.pt", "flax_*", "tf_*"],
)
print(f"Local dir: {local_dir}")

shutil.copy(os.path.join(local_dir, "tokenizer.json"), os.path.join(OUT_DIR, "tokenizer.json"))
print("Tokenizer copied.")

# Build tensor → shard-file index across all safetensors shards.
shard_paths = sorted(glob.glob(os.path.join(local_dir, "*.safetensors")))
print(f"Shards found: {len(shard_paths)}")
tensor_index: dict[str, str] = {}
for shard in shard_paths:
    with safe_open(shard, framework="pt", device="cpu") as st:
        for key in st.keys():
            tensor_index[key] = shard

manifest = []
file_idx = 0

def export_tensor(name: str, alias: str | None = None) -> None:
    global file_idx
    shard = tensor_index[name]
    with safe_open(shard, framework="pt", device="cpu") as st:
        t = st.get_tensor(name).float().numpy()
    flat = t.flatten().astype(np.float32)
    out_name = alias if alias else name
    bin_file = f"weight_{file_idx:05d}.bin"
    with open(os.path.join(OUT_DIR, bin_file), "wb") as f:
        f.write(flat.tobytes())
    manifest.append({
        "file":  bin_file,
        "name":  out_name,
        "numel": int(flat.size),
        "shape": list(t.shape),
    })
    file_idx += 1
    print(f"  {out_name}: {list(t.shape)}")

print("\nExporting embeddings...")
export_tensor("model.embed_tokens.weight", "embed_tokens.weight")
export_tensor("model.norm.weight",         "norm.weight")

print("\nExporting layers...")
for i in range(N_LAYERS):
    p = f"model.layers.{i}"
    a = f"layers.{i}"
    export_tensor(f"{p}.input_layernorm.weight",          f"{a}.input_layernorm.weight")
    export_tensor(f"{p}.self_attn.q_proj.weight",         f"{a}.self_attn.q_proj.weight")
    export_tensor(f"{p}.self_attn.k_proj.weight",         f"{a}.self_attn.k_proj.weight")
    export_tensor(f"{p}.self_attn.v_proj.weight",         f"{a}.self_attn.v_proj.weight")
    export_tensor(f"{p}.self_attn.o_proj.weight",         f"{a}.self_attn.o_proj.weight")
    export_tensor(f"{p}.post_attention_layernorm.weight", f"{a}.post_attention_layernorm.weight")
    export_tensor(f"{p}.mlp.gate_proj.weight",            f"{a}.mlp.gate_proj.weight")
    export_tensor(f"{p}.mlp.up_proj.weight",              f"{a}.mlp.up_proj.weight")
    export_tensor(f"{p}.mlp.down_proj.weight",            f"{a}.mlp.down_proj.weight")

manifest_path = os.path.join(OUT_DIR, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

total_params = sum(e["numel"] for e in manifest)
print(f"\nDone. {len(manifest)} tensors, {total_params:,} params")
print(f"Manifest: {manifest_path}")
