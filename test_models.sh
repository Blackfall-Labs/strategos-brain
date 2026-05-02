#!/usr/bin/env bash
# Load-test each expert model: verifies weights load and one forward pass runs.
# Prints timing + top-5 logits per model. No generation, no text output.
set -e
cd "$(dirname "$0")"

echo "=== Qwen3 1.7B ==="
cargo run --release --bin qwen3_test 2>&1

echo ""
echo "=== SmolLM3 3B ==="
cargo run --release --bin smollm3_test 2>&1

echo ""
echo "=== Phi-4 Mini 3.8B ==="
cargo run --release --bin phi4_test 2>&1

echo ""
echo "All models OK."
