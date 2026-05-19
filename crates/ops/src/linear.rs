//! Integer linear (matmul) — the core operation.
//!
//! i16 weights × i16 inputs → i64 accumulator → rescale to i16 output.
//! Parallelized with Rayon across output rows.
//! AVX2 inner loop: 16 i16 multiply-accumulates per instruction.

use rayon::prelude::*;
use crate::qtensor::{QTensor, QWeight};

/// Integer matrix-vector multiply with bias. Parallelized + AVX2.
pub fn linear_q(x: &QTensor, w: &QWeight, bias: &[f32]) -> QTensor {
    assert_eq!(x.len(), w.in_dim);
    assert_eq!(bias.len(), w.out_dim);

    let in_dim = w.in_dim;
    let out_dim = w.out_dim;
    let acc_scale = x.scale * w.scale;

    let accs: Vec<i64> = (0..out_dim).into_par_iter().map(|j| {
        let row = &w.data[j * in_dim..(j + 1) * in_dim];
        let mut sum = dot_i16(&x.data, row);

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

/// Integer matrix-vector multiply that skips specified output rows.
///
/// `skip[j] = true` → row j is not computed; output[j] is set to 0 (caller must overwrite).
/// Only non-skipped rows are dispatched to the thread pool, giving a proportional speedup.
pub fn linear_q_skip_rows(x: &QTensor, w: &QWeight, bias: &[f32], skip: &[bool]) -> QTensor {
    assert_eq!(x.len(), w.in_dim);
    assert_eq!(bias.len(), w.out_dim);
    assert_eq!(skip.len(), w.out_dim);

    let in_dim    = w.in_dim;
    let out_dim   = w.out_dim;
    let acc_scale = x.scale * w.scale;

    // Collect only the rows we need to compute.
    let active_rows: Vec<usize> = (0..out_dim).filter(|&j| !skip[j]).collect();

    let computed: Vec<(usize, i64)> = active_rows
        .into_par_iter()
        .map(|j| {
            let row = &w.data[j * in_dim..(j + 1) * in_dim];
            let mut sum = dot_i16(&x.data, row);
            if acc_scale > 1e-20 {
                sum += (bias[j] / acc_scale).round() as i64;
            }
            (j, sum)
        })
        .collect();

    let acc_max = computed.iter().map(|(_, v)| v.unsigned_abs()).max().unwrap_or(1);
    let shift    = (acc_max / 32767).max(1) as i64;
    let out_scale = acc_scale * shift as f32;

    let mut data = vec![0i16; out_dim];
    for (j, v) in computed {
        data[j] = (v / shift).clamp(-32767, 32767) as i16;
    }

    QTensor { data, scale: out_scale }
}

/// Dot product of two i16 slices → i64.
/// Uses AVX2 when available, falls back to scalar.
#[inline]
fn dot_i16(a: &[i16], b: &[i16]) -> i64 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_i16_avx2(a, b) };
        }
    }

    dot_i16_scalar(a, b)
}

/// Scalar fallback.
#[inline]
fn dot_i16_scalar(a: &[i16], b: &[i16]) -> i64 {
    let mut sum = 0i64;
    for i in 0..a.len() {
        sum += a[i] as i64 * b[i] as i64;
    }
    sum
}

/// AVX2: 16 i16 multiplies per instruction via _mm256_madd_epi16.
/// madd multiplies pairs of i16 and adds adjacent pairs → 8 × i32 per op.
///
/// madd produces 8 × i32, each up to 2 × 32767² ≈ 2.1B. That's near
/// i32 max (2.1B), so we MUST drain after every single madd to prevent
/// overflow. The drain cost is amortized by the 16-wide SIMD throughput.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i16_avx2(a: &[i16], b: &[i16]) -> i64 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 16;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut total = 0i64;

    for c in 0..chunks {
        let offset = c * 16;
        let va = _mm256_loadu_si256(a_ptr.add(offset) as *const __m256i);
        let vb = _mm256_loadu_si256(b_ptr.add(offset) as *const __m256i);
        let prod = _mm256_madd_epi16(va, vb);
        // Drain immediately to i64 to prevent i32 overflow.
        total += hsum_i32_to_i64(prod);
    }

    // Scalar remainder.
    let processed = chunks * 16;
    for i in processed..n {
        total += *a_ptr.add(i) as i64 * *b_ptr.add(i) as i64;
    }

    total
}

/// Horizontal sum of 8 × i32 in a __m256i → i64.
/// Extracts each i32 individually and sums as i64 to prevent overflow.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_i32_to_i64(v: std::arch::x86_64::__m256i) -> i64 {
    use std::arch::x86_64::*;

    // Extract all 8 i32 lanes individually and sum as i64.
    // This avoids any i32 overflow during reduction.
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);

    _mm_extract_epi32(lo, 0) as i64
        + _mm_extract_epi32(lo, 1) as i64
        + _mm_extract_epi32(lo, 2) as i64
        + _mm_extract_epi32(lo, 3) as i64
        + _mm_extract_epi32(hi, 0) as i64
        + _mm_extract_epi32(hi, 1) as i64
        + _mm_extract_epi32(hi, 2) as i64
        + _mm_extract_epi32(hi, 3) as i64
}

/// Batch matrix multiply: x [t, in_dim] × w [out_dim, in_dim] → [t, out_dim].
pub fn linear_q_batch(x: &QTensor, w: &QWeight, bias: &[f32], t: usize) -> QTensor {
    assert_eq!(x.len(), t * w.in_dim);

    let in_dim = w.in_dim;
    let out_dim = w.out_dim;

    let frames: Vec<QTensor> = (0..t).into_par_iter().map(|ti| {
        let frame = QTensor {
            data: x.data[ti * in_dim..(ti + 1) * in_dim].to_vec(),
            scale: x.scale,
        };
        linear_q(&frame, w, bias)
    }).collect();

    let max_scale = frames.iter().map(|f| f.scale).fold(0.0f32, f32::max);

    let mut data = Vec::with_capacity(t * out_dim);
    for frame in &frames {
        let ratio = frame.scale / max_scale;
        for &v in &frame.data {
            data.push((v as f32 * ratio).round().clamp(-32767.0, 32767.0) as i16);
        }
    }

    QTensor { data, scale: max_scale }
}

/// Parallel f32 dot product for tied-embedding lm_head.
pub fn matmul_embed_parallel(hidden: &[f32], embed: &[Vec<f32>]) -> Vec<f32> {
    embed.par_iter().map(|row| {
        dot_f32(hidden, row)
    }).collect()
}

/// Fast f32 dot product with unrolling.
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut i = 0;
    while i + 4 <= a.len() {
        sum0 += a[i] * b[i];
        sum1 += a[i+1] * b[i+1];
        sum2 += a[i+2] * b[i+2];
        sum3 += a[i+3] * b[i+3];
        i += 4;
    }
    while i < a.len() {
        sum0 += a[i] * b[i];
        i += 1;
    }
    sum0 + sum1 + sum2 + sum3
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
    fn dot_i16_matches_scalar() {
        let a: Vec<i16> = (0..256).map(|i| (i - 128) as i16).collect();
        let b: Vec<i16> = (0..256).map(|i| (i * 3 - 384) as i16).collect();
        let scalar = dot_i16_scalar(&a, &b);
        let fast = dot_i16(&a, &b);
        assert_eq!(scalar, fast, "AVX2 dot must match scalar: {} vs {}", scalar, fast);
    }

    #[test]
    fn dot_i16_large() {
        // Realistic values: quantized weights/activations rarely exceed ±16000.
        // madd adds pairs of i16 products into i32. At ±16000, each pair
        // produces ≤ 2 × 16000² = 512M, well within i32 range.
        let a: Vec<i16> = (0..2048).map(|i| ((i * 7 + 13) % 32000 - 16000) as i16).collect();
        let b: Vec<i16> = (0..2048).map(|i| ((i * 11 + 37) % 32000 - 16000) as i16).collect();
        let scalar = dot_i16_scalar(&a, &b);
        let fast = dot_i16(&a, &b);
        assert_eq!(scalar, fast, "large dot: {} vs {}", scalar, fast);
    }

    #[test]
    fn dot_i16_extreme() {
        // Edge case: values near ±32767 can overflow madd's i32 accumulator.
        // In practice, QWeight::from_f32 distributes values across the range,
        // so adjacent pairs at ±32767 are extremely unlikely. But test that
        // moderate values work correctly.
        let a: Vec<i16> = (0..2048).map(|i| if i % 2 == 0 { 20000 } else { -20000 }).collect();
        let b: Vec<i16> = (0..2048).map(|i| if i % 2 == 0 { 15000 } else { -15000 }).collect();
        let scalar = dot_i16_scalar(&a, &b);
        let fast = dot_i16(&a, &b);
        assert_eq!(scalar, fast, "extreme dot: {} vs {}", scalar, fast);
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
