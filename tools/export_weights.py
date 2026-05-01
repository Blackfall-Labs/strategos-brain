#!/usr/bin/env python3
"""
Export HuggingFace model weights to Strategos binary format.

Reads safetensors weights and writes:
  manifest.json — tensor metadata (name, file, shape, numel)
  *.bin — raw f32 little-endian bytes per tensor

Usage:
  python export_weights.py --model Qwen/Qwen3-1.7B --output models/qwen3
  python export_weights.py --model HuggingFaceTB/SmolLM3-3B --output models/smollm3
  python export_weights.py --model microsoft/Phi-4-mini-instruct --output models/phi4

Requires: pip install safetensors torch huggingface_hub
"""

import argparse
import json
import os
import sys


def export_safetensors(model_path: str, output_dir: str):
    """Export from safetensors format using torch (handles bfloat16)."""
    import torch
    from safetensors import safe_open

    os.makedirs(output_dir, exist_ok=True)

    # Find all safetensors files.
    st_files = []
    if os.path.isdir(model_path):
        for f in sorted(os.listdir(model_path)):
            if f.endswith('.safetensors'):
                st_files.append(os.path.join(model_path, f))
    elif model_path.endswith('.safetensors'):
        st_files = [model_path]

    if not st_files:
        print(f"No safetensors files found in {model_path}")
        sys.exit(1)

    manifest = []
    tensor_idx = 0

    for st_path in st_files:
        print(f"Reading {st_path}...")
        with safe_open(st_path, framework="pt") as f:
            for name in sorted(f.keys()):
                tensor = f.get_tensor(name)
                shape = list(tensor.shape)
                numel = int(tensor.numel())

                # Convert to float32 contiguous.
                tensor = tensor.float().contiguous().cpu()

                # Write raw f32 LE bytes.
                bin_name = f"weight_{tensor_idx:05d}.bin"
                bin_path = os.path.join(output_dir, bin_name)
                with open(bin_path, 'wb') as bf:
                    bf.write(tensor.numpy().tobytes())

                manifest.append({
                    "name": name,
                    "file": bin_name,
                    "shape": shape,
                    "numel": numel,
                })

                tensor_idx += 1
                if tensor_idx % 50 == 0:
                    print(f"  {tensor_idx} tensors exported...")

    # Write manifest.
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nExported {tensor_idx} tensors to {output_dir}/")
    print(f"Manifest: {manifest_path}")

    # Summary.
    total_params = sum(e['numel'] for e in manifest)
    total_bytes = total_params * 4
    print(f"Total parameters: {total_params:,}")
    print(f"Total size (f32): {total_bytes / 1e9:.2f} GB")
    print(f"Total size (i16): {total_params * 2 / 1e9:.2f} GB")


def export_from_hub(model_id: str, output_dir: str):
    """Download from HuggingFace Hub and export."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {model_id}...")
    local_path = snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "config.json", "tokenizer*"],
    )
    print(f"Downloaded to {local_path}")

    # Copy config.json.
    config_src = os.path.join(local_path, "config.json")
    if os.path.exists(config_src):
        os.makedirs(output_dir, exist_ok=True)
        import shutil
        shutil.copy2(config_src, os.path.join(output_dir, "config.json"))
        print("Copied config.json")

    # Copy tokenizer files.
    for f in os.listdir(local_path):
        if f.startswith("tokenizer"):
            import shutil
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy2(os.path.join(local_path, f), os.path.join(output_dir, f))
            print(f"Copied {f}")

    export_safetensors(local_path, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export HF weights to Strategos format")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--output", required=True, help="Output directory for manifest + bins")
    args = parser.parse_args()

    if os.path.isdir(args.model) or args.model.endswith('.safetensors'):
        export_safetensors(args.model, args.output)
    else:
        export_from_hub(args.model, args.output)
