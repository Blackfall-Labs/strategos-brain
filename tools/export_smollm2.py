"""Export SmolLM2-135M weights to strategos manifest+bin format.

Downloads from HuggingFace (HuggingFaceTB/SmolLM2-135M) and exports the tensors
needed for the activation prober. All weights kept as float32.

Usage:
    C:\\Python313\\python.exe tools\\export_smollm2.py
"""

import json, os, struct
import numpy as np
from safetensors import safe_open
from huggingface_hub import hf_hub_download

MODEL_ID   = "HuggingFaceTB/SmolLM2-135M"
OUT_DIR    = "models/smollm2"
N_LAYERS   = 30

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Downloading {MODEL_ID}...")
st_path = hf_hub_download(repo_id=MODEL_ID, filename="model.safetensors")
tok_path = hf_hub_download(repo_id=MODEL_ID, filename="tokenizer.json")
print(f"Weights: {st_path}")
print(f"Tokenizer: {tok_path}")

# Copy tokenizer
import shutil
shutil.copy(tok_path, os.path.join(OUT_DIR, "tokenizer.json"))
print("Tokenizer copied.")

manifest = []
file_idx = 0

def export_tensor(st, name, alias=None):
    global file_idx
    t = st.get_tensor(name).float().numpy()
    flat = t.flatten().astype(np.float32)
    out_name = alias if alias else name
    bin_file = f"weight_{file_idx:05d}.bin"
    with open(os.path.join(OUT_DIR, bin_file), "wb") as f:
        f.write(flat.tobytes())
    manifest.append({
        "file": bin_file,
        "name": out_name,
        "numel": int(flat.size),
        "shape": list(t.shape),
    })
    file_idx += 1
    print(f"  {out_name}: {list(t.shape)}")

with safe_open(st_path, framework="pt", device="cpu") as st:
    print("\nExporting embeddings...")
    export_tensor(st, "model.embed_tokens.weight", "embed_tokens.weight")
    export_tensor(st, "model.norm.weight", "norm.weight")

    print("\nExporting layers...")
    for i in range(N_LAYERS):
        p = f"model.layers.{i}"
        a = f"layers.{i}"
        export_tensor(st, f"{p}.input_layernorm.weight",          f"{a}.input_layernorm.weight")
        export_tensor(st, f"{p}.self_attn.q_proj.weight",         f"{a}.self_attn.q_proj.weight")
        export_tensor(st, f"{p}.self_attn.k_proj.weight",         f"{a}.self_attn.k_proj.weight")
        export_tensor(st, f"{p}.self_attn.v_proj.weight",         f"{a}.self_attn.v_proj.weight")
        export_tensor(st, f"{p}.self_attn.o_proj.weight",         f"{a}.self_attn.o_proj.weight")
        export_tensor(st, f"{p}.post_attention_layernorm.weight",  f"{a}.post_attention_layernorm.weight")
        export_tensor(st, f"{p}.mlp.gate_proj.weight",            f"{a}.mlp.gate_proj.weight")
        export_tensor(st, f"{p}.mlp.up_proj.weight",              f"{a}.mlp.up_proj.weight")
        export_tensor(st, f"{p}.mlp.down_proj.weight",            f"{a}.mlp.down_proj.weight")

manifest_path = os.path.join(OUT_DIR, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

total_params = sum(e["numel"] for e in manifest)
print(f"\nDone. {len(manifest)} tensors, {total_params:,} params")
print(f"Manifest: {manifest_path}")
