"""Export QKV attention biases that were omitted from the original export.

The original export_qwen25_05b.py exported only weight matrices, skipping
bias=True parameters on q_proj, k_proj, v_proj. For single-token forward
pass, q/k biases don't affect the output (softmax([scalar]) = 1.0), but
v_proj.bias directly shifts every layer's V computation.

Exports one file per layer:
  models/qwen25_05b/v_bias_{layer:02d}.bin  -- [num_kv_heads * head_dim = 128]

Q/K biases are also exported for completeness but unused by probe_to_brnc.py.
"""

import json, os, glob
import numpy as np
from safetensors import safe_open
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-0.5B"
OUT_DIR  = "models/qwen25_05b"
N_LAYERS = 24

print(f"Locating {MODEL_ID} in HuggingFace cache...")
local_dir = snapshot_download(
    repo_id=MODEL_ID,
    ignore_patterns=["*.msgpack", "*.h5", "*.pt", "flax_*", "tf_*"],
)
print(f"Local dir: {local_dir}")

shard_paths = sorted(glob.glob(os.path.join(local_dir, "*.safetensors")))
tensor_index: dict[str, str] = {}
for shard in shard_paths:
    with safe_open(shard, framework="pt", device="cpu") as st:
        for key in st.keys():
            tensor_index[key] = shard

def export_bias(name: str, out_file: str) -> None:
    if name not in tensor_index:
        print(f"  MISSING: {name}")
        return
    shard = tensor_index[name]
    with safe_open(shard, framework="pt", device="cpu") as st:
        t = st.get_tensor(name).float().numpy()
    path = os.path.join(OUT_DIR, out_file)
    t.flatten().astype(np.float32).tofile(path)
    print(f"  {name}: {list(t.shape)} -> {out_file}")

print("\nExporting QKV biases...")
for i in range(N_LAYERS):
    p = f"model.layers.{i}"
    export_bias(f"{p}.self_attn.q_proj.bias", f"q_bias_{i:02d}.bin")
    export_bias(f"{p}.self_attn.k_proj.bias", f"k_bias_{i:02d}.bin")
    export_bias(f"{p}.self_attn.v_proj.bias", f"v_bias_{i:02d}.bin")

print(f"\nDone. {N_LAYERS * 3} bias files written to {OUT_DIR}/")
