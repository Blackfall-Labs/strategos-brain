@echo off
cd /d "%~dp0"
cargo build --release -p strategos-runtime
echo Build complete.
