@echo off
cd /d "%~dp0"
cargo run --release --bin qwen3_bench
