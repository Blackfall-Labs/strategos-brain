//! Strategos Runtime — model loading and inference.
//!
//! Loads model weights from manifest.json + .bin files (same format
//! as ternsig/deepspeech/conformer_native.rs), runs forward passes
//! using strategos-ops integer math.

pub mod loader;
pub mod qwen3;
pub mod smollm3;
pub mod phi4;
