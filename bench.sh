#!/usr/bin/env bash
# Run the Qwen3 throughput benchmark.
set -e
cd "$(dirname "$0")"
cargo run --release --bin qwen3_bench 2>&1
