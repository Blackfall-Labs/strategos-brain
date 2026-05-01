//! RMSNorm — Root Mean Square Layer Normalization.
//!
//! NEW — does not exist in ternsig/deepspeech (which uses LayerNorm).
//! Required by Qwen3, SmolLM3, and Phi-4.
//!
//! Simpler than LayerNorm: no mean subtraction.
//!   rms = sqrt(mean(x^2) + eps)
//!   output = (x / rms) * weight
//!
//! The sqrt is computed via InvSqrtLut (proven in lut.rs).
//! The weight multiply uses integer (i16 × i16).

use crate::qtensor::QTensor;

/// RMSNorm weights — pre-quantized scale parameters.
pub struct RmsNormWeights {
    /// Float weights (kept float because normalization
    /// involves float division anyway).
    pub weight: Vec<f32>,
    pub dim: usize,
    pub eps: f32,
}

impl RmsNormWeights {
    pub fn from_f32(weight: &[f32], eps: f32) -> Self {
        Self {
            weight: weight.to_vec(),
            dim: weight.len(),
            eps,
        }
    }
}

/// Apply RMSNorm to a float vector. Returns normalized float.
/// This is the version used in the forward pass — mean/sqrt stay float.
pub fn rms_norm(x: &[f32], w: &RmsNormWeights) -> Vec<f32> {
    let dim = w.dim;
    assert_eq!(x.len(), dim);

    // Compute RMS.
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
    let rms = (mean_sq + w.eps).sqrt();
    let inv_rms = 1.0 / rms;

    // Normalize and scale.
    x.iter().zip(w.weight.iter())
        .map(|(&xi, &wi)| xi * inv_rms * wi)
        .collect()
}

/// Apply RMSNorm to a QTensor. Returns QTensor.
/// Dequantizes to float for the normalization (sqrt is transcendental),
/// then requantizes the output.
pub fn rms_norm_q(x: &QTensor, w: &RmsNormWeights) -> QTensor {
    let float_input = x.to_f32();
    let normalized = rms_norm(&float_input, w);
    QTensor::from_f32(&normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_unit_weights() {
        let x = vec![3.0f32, 4.0]; // rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
        let w = RmsNormWeights::from_f32(&[1.0, 1.0], 1e-6);
        let out = rms_norm(&x, &w);
        // 3.0 / 3.536 ≈ 0.849, 4.0 / 3.536 ≈ 1.131
        assert!((out[0] - 0.849).abs() < 0.01, "got {}", out[0]);
        assert!((out[1] - 1.131).abs() < 0.01, "got {}", out[1]);
    }

    #[test]
    fn rms_norm_with_weights() {
        let x = vec![1.0f32; 8];
        let weight = vec![2.0f32; 8];
        let w = RmsNormWeights::from_f32(&weight, 1e-6);
        let out = rms_norm(&x, &w);
        // rms of all-ones = 1.0, so output = x * 1.0 * weight = 2.0
        for &v in &out {
            assert!((v - 2.0).abs() < 0.01, "got {}", v);
        }
    }

    #[test]
    fn rms_norm_q_roundtrip() {
        let x_float = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = RmsNormWeights::from_f32(&[1.0; 4], 1e-6);

        let ref_out = rms_norm(&x_float, &w);
        let q_in = QTensor::from_f32(&x_float);
        let q_out = rms_norm_q(&q_in, &w);
        let q_float = q_out.to_f32();

        let max_diff: f32 = ref_out.iter().zip(q_float.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 0.05, "rms_norm_q max_diff = {}", max_diff);
    }
}
