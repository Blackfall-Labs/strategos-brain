//! Integer linear (matmul) — the core operation.
//!
//! i16 weights × i16 inputs → i64 accumulator → rescale to i16 output.
//! Proven in ternsig/deepspeech/pipeline_int.rs (linear_q).

use crate::qtensor::{QTensor, QWeight};

/// Integer matrix-vector multiply with bias.
///
/// x: QTensor [in_dim]
/// w: QWeight [out_dim, in_dim] (row-major)
/// bias: [out_dim] in original float units
///
/// Returns QTensor [out_dim].
pub fn linear_q(x: &QTensor, w: &QWeight, bias: &[f32]) -> QTensor {
    assert_eq!(x.len(), w.in_dim);
    assert_eq!(bias.len(), w.out_dim);

    let in_dim = w.in_dim;
    let out_dim = w.out_dim;

    // Accumulate in i64 — no overflow possible.
    let mut acc = vec![0i64; out_dim];

    for j in 0..out_dim {
        let row_offset = j * in_dim;
        let mut sum = 0i64;
        for i in 0..in_dim {
            sum += x.data[i] as i64 * w.data[row_offset + i] as i64;
        }
        acc[j] = sum;
    }

    // Add bias in accumulator scale units.
    let acc_scale = x.scale * w.scale;
    if acc_scale > 1e-20 {
        let bias_inv = 1.0 / acc_scale;
        for j in 0..out_dim {
            acc[j] += (bias[j] * bias_inv).round() as i64;
        }
    }

    // Two-pass rescale: find max, compute shift to fit i16.
    let acc_max = acc.iter().map(|v| v.unsigned_abs()).max().unwrap_or(1);
    let shift = (acc_max / 32767).max(1) as i64;
    let out_scale = acc_scale * shift as f32;

    let data: Vec<i16> = acc.iter()
        .map(|&v| (v / shift).clamp(-32767, 32767) as i16)
        .collect();

    QTensor { data, scale: out_scale }
}

/// Integer matrix multiply for batched time steps.
/// x: QTensor [t * in_dim] (t frames, each in_dim)
/// Returns QTensor [t * out_dim].
pub fn linear_q_batch(x: &QTensor, w: &QWeight, bias: &[f32], t: usize) -> QTensor {
    assert_eq!(x.len(), t * w.in_dim);

    let in_dim = w.in_dim;
    let out_dim = w.out_dim;
    let mut all_data = Vec::with_capacity(t * out_dim);
    let mut max_scale: f32 = 0.0;

    for ti in 0..t {
        let frame = QTensor {
            data: x.data[ti * in_dim..(ti + 1) * in_dim].to_vec(),
            scale: x.scale,
        };
        let out = linear_q(&frame, w, bias);
        if out.scale > max_scale {
            max_scale = out.scale;
        }
        all_data.push(out);
    }

    // Rescale all frames to the same scale for consistent output.
    let mut data = Vec::with_capacity(t * out_dim);
    for frame in &all_data {
        let ratio = frame.scale / max_scale;
        for &v in &frame.data {
            data.push((v as f32 * ratio).round().clamp(-32767.0, 32767.0) as i16);
        }
    }

    QTensor { data, scale: max_scale }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_small() {
        // 4-dim input, 2-dim output
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let weights = vec![
            1.0, 0.0, 0.0, 0.0, // row 0: picks input[0]
            0.0, 0.0, 0.0, 1.0, // row 1: picks input[3]
        ];
        let bias = vec![0.0, 0.0];

        let x = QTensor::from_f32(&input);
        let w = QWeight::from_f32(&weights, 4, 2);
        let out = linear_q(&x, &w, &bias);
        let result = out.to_f32();

        assert!((result[0] - 1.0).abs() < 0.1, "expected ~1.0, got {}", result[0]);
        assert!((result[1] - 4.0).abs() < 0.1, "expected ~4.0, got {}", result[1]);
    }

    #[test]
    fn linear_with_bias() {
        let input = vec![1.0f32; 8];
        let weights = vec![1.0f32; 8 * 4]; // all ones, 8→4
        let bias = vec![10.0, 20.0, 30.0, 40.0];

        let x = QTensor::from_f32(&input);
        let w = QWeight::from_f32(&weights, 8, 4);
        let out = linear_q(&x, &w, &bias);
        let result = out.to_f32();

        // Each output = sum(1.0 * 1.0, 8 times) + bias = 8 + bias
        assert!((result[0] - 18.0).abs() < 1.0, "expected ~18, got {}", result[0]);
        assert!((result[3] - 48.0).abs() < 1.0, "expected ~48, got {}", result[3]);
    }
}
