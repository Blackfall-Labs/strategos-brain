//! QTensor / QWeight — quantized tensor types.
//!
//! Proven in ternsig/deepspeech/pipeline_int.rs.
//! i16 data with a single f32 scale factor per tensor.
//! value[i] ≈ data[i] * scale

/// Quantized activation tensor.
#[derive(Clone)]
pub struct QTensor {
    pub data: Vec<i16>,
    pub scale: f32,
}

impl QTensor {
    /// Dynamic quantization: scan for max, scale to fill i16 range.
    pub fn from_f32(f: &[f32]) -> Self {
        if f.is_empty() {
            return Self { data: Vec::new(), scale: 1.0 };
        }
        let max_abs = f.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        if max_abs < 1e-10 {
            return Self { data: vec![0i16; f.len()], scale: 1e-10 };
        }
        let inv_scale = 32767.0 / max_abs;
        let data: Vec<i16> = f.iter()
            .map(|v| (v * inv_scale).round().clamp(-32767.0, 32767.0) as i16)
            .collect();
        Self { data, scale: max_abs / 32767.0 }
    }

    /// Fixed-scale quantization (for known ranges like after layer norm).
    pub fn from_f32_fixed(f: &[f32], scale: f32) -> Self {
        let inv = 1.0 / scale;
        let data: Vec<i16> = f.iter()
            .map(|v| (v * inv).round().clamp(-32767.0, 32767.0) as i16)
            .collect();
        Self { data, scale }
    }

    /// Dequantize back to float.
    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|&v| v as f32 * self.scale).collect()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Quantized weight matrix — pre-quantized at load time.
#[derive(Clone)]
pub struct QWeight {
    pub data: Vec<i16>,
    pub scale: f32,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl QWeight {
    /// Quantize f32 weights to i16. Called once at model load.
    /// Layout: [out_dim, in_dim] row-major.
    pub fn from_f32(weights: &[f32], in_dim: usize, out_dim: usize) -> Self {
        assert_eq!(weights.len(), in_dim * out_dim);
        let max_abs = weights.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let inv_scale = if max_abs > 1e-10 { 32767.0 / max_abs } else { 1.0 };
        let data: Vec<i16> = weights.iter()
            .map(|v| (v * inv_scale).round().clamp(-32767.0, 32767.0) as i16)
            .collect();
        let scale = if max_abs > 1e-10 { max_abs / 32767.0 } else { 1e-10 };
        Self { data, scale, in_dim, out_dim }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qtensor_roundtrip() {
        let input: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.5).collect();
        let q = QTensor::from_f32(&input);
        let output = q.to_f32();
        let max_diff: f32 = input.iter().zip(output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 0.01, "roundtrip max_diff = {}", max_diff);
    }

    #[test]
    fn qweight_quantize() {
        let weights: Vec<f32> = (0..256).map(|i| ((i as f32) - 128.0) / 128.0).collect();
        let qw = QWeight::from_f32(&weights, 16, 16);
        assert_eq!(qw.data.len(), 256);
        assert!(qw.scale > 0.0);
        // Check max value maps near 32767
        let max_q = qw.data.iter().map(|v| v.abs()).max().unwrap();
        assert!(max_q > 32000, "should use most of i16 range, got {}", max_q);
    }
}
