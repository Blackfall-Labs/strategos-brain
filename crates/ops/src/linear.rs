//! Integer linear (matmul) — the core operation.
//!
//! i16 weights × i16 inputs → i64 accumulator → rescale to i16 output.
//! Parallelized with Rayon across output rows.
//! Cache-blocked inner loop for L1 locality.

use rayon::prelude::*;
use crate::qtensor::{QTensor, QWeight};

/// Cache block size for inner loop. 64 elements × 2 bytes = 128 bytes,
/// fits in a single cache line pair.
const BLOCK: usize = 64;

/// Integer matrix-vector multiply with bias. Parallelized.
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
    let acc_scale = x.scale * w.scale;

    // Parallel accumulation across output rows.
    let accs: Vec<i64> = (0..out_dim).into_par_iter().map(|j| {
        let row_offset = j * in_dim;
        let mut sum = 0i64;

        // Cache-blocked inner loop.
        let mut i = 0;
        while i + BLOCK <= in_dim {
            let mut block_sum = 0i64;
            for k in i..i + BLOCK {
                block_sum += x.data[k] as i64 * w.data[row_offset + k] as i64;
            }
            sum += block_sum;
            i += BLOCK;
        }
        // Remainder.
        for k in i..in_dim {
            sum += x.data[k] as i64 * w.data[row_offset + k] as i64;
        }

        // Add bias.
        if acc_scale > 1e-20 {
            sum += (bias[j] / acc_scale).round() as i64;
        }

        sum
    }).collect();

    // Two-pass rescale.
    let acc_max = accs.iter().map(|v| v.unsigned_abs()).max().unwrap_or(1);
    let shift = (acc_max / 32767).max(1) as i64;
    let out_scale = acc_scale * shift as f32;

    let data: Vec<i16> = accs.iter()
        .map(|&v| (v / shift).clamp(-32767, 32767) as i16)
        .collect();

    QTensor { data, scale: out_scale }
}

/// Batch matrix multiply: x [t, in_dim] × w [out_dim, in_dim] → [t, out_dim].
/// Each time step computed in parallel.
pub fn linear_q_batch(x: &QTensor, w: &QWeight, bias: &[f32], t: usize) -> QTensor {
    assert_eq!(x.len(), t * w.in_dim);

    let in_dim = w.in_dim;
    let out_dim = w.out_dim;

    // Process all frames in parallel.
    let frames: Vec<QTensor> = (0..t).into_par_iter().map(|ti| {
        let frame = QTensor {
            data: x.data[ti * in_dim..(ti + 1) * in_dim].to_vec(),
            scale: x.scale,
        };
        linear_q(&frame, w, bias)
    }).collect();

    // Find max scale.
    let max_scale = frames.iter().map(|f| f.scale).fold(0.0f32, f32::max);

    // Rescale all frames to same scale.
    let mut data = Vec::with_capacity(t * out_dim);
    for frame in &frames {
        let ratio = frame.scale / max_scale;
        for &v in &frame.data {
            data.push((v as f32 * ratio).round().clamp(-32767.0, 32767.0) as i16);
        }
    }

    QTensor { data, scale: max_scale }
}

/// Fast dot product of f32 slices. Used for tied-embedding lm_head
/// where we can't avoid f32 (embeddings are stored as f32 for lookup).
/// Parallelized across vocab entries.
pub fn matmul_embed_parallel(hidden: &[f32], embed: &[Vec<f32>]) -> Vec<f32> {
    embed.par_iter().map(|row| {
        let mut dot = 0.0f32;
        let mut i = 0;
        // Unrolled by 4 for ILP.
        while i + 4 <= hidden.len() {
            dot += hidden[i] * row[i]
                + hidden[i+1] * row[i+1]
                + hidden[i+2] * row[i+2]
                + hidden[i+3] * row[i+3];
            i += 4;
        }
        while i < hidden.len() {
            dot += hidden[i] * row[i];
            i += 1;
        }
        dot
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_small() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let weights = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
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
        let weights = vec![1.0f32; 8 * 4];
        let bias = vec![10.0, 20.0, 30.0, 40.0];

        let x = QTensor::from_f32(&input);
        let w = QWeight::from_f32(&weights, 8, 4);
        let out = linear_q(&x, &w, &bias);
        let result = out.to_f32();

        assert!((result[0] - 18.0).abs() < 1.0, "expected ~18, got {}", result[0]);
        assert!((result[3] - 48.0).abs() < 1.0, "expected ~48, got {}", result[3]);
    }

    #[test]
    fn embed_parallel() {
        let hidden = vec![1.0f32; 64];
        let embed: Vec<Vec<f32>> = (0..100).map(|i| {
            vec![if i == 42 { 1.0 } else { 0.0 }; 64]
        }).collect();
        let logits = matmul_embed_parallel(&hidden, &embed);
        assert_eq!(logits.len(), 100);
        assert!((logits[42] - 64.0).abs() < 0.01);
        assert!((logits[0] - 0.0).abs() < 0.01);
    }
}
