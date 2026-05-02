#!/usr/bin/env bash
# Build all binaries in release mode.
set -e
cd "$(dirname "$0")"
cargo build --release -p strategos-runtime
echo "Build complete."
