#!/usr/bin/env bash
# Run the Strategos REPL.
set -e
cd "$(dirname "$0")"
./target/release/strategos.exe
