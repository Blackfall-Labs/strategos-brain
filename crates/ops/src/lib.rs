//! Strategos Integer Ops — proven math from ternsig/deepspeech.
//!
//! All computation is i16 weights × i16 inputs → i64 accumulator.
//! No floats in the hot path except attention softmax (precision-critical).
//!
//! Ported from:
//!   E:\repos\blackfall-labs\ternsig\crates\deepspeech\src\pipeline_int.rs
//!   E:\repos\blackfall-labs\ternsig\crates\deepspeech\src\ops_int.rs
//!   E:\repos\blackfall-labs\ternsig\crates\deepspeech\src\lut.rs

pub mod qtensor;
pub mod linear;
pub mod lut;
pub mod rope;
pub mod rmsnorm;
