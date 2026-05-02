@echo off
cd /d "%~dp0"

echo === Qwen3 1.7B ===
cargo run --release --bin qwen3_test
if %errorlevel% neq 0 ( echo FAILED & exit /b 1 )

echo.
echo === SmolLM3 3B ===
cargo run --release --bin smollm3_test
if %errorlevel% neq 0 ( echo FAILED & exit /b 1 )

echo.
echo === Phi-4 Mini 3.8B ===
cargo run --release --bin phi4_test
if %errorlevel% neq 0 ( echo FAILED & exit /b 1 )

echo.
echo All models OK.
