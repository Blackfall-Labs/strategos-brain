//! Rotary Position Embeddings (RoPE).
//!
//! NEW — does not exist in ternsig/deepspeech.
//! Required by Qwen3, SmolLM3, and Phi-4.
//!
//! Rotates pairs of dimensions in Q and K vectors by position-dependent
//! angles. Encodes absolute position through relative rotation.
//!
//! For dimension pair (2i, 2i+1) at position pos:
//!   theta = pos / 10000^(2i/dim)
//!   q_rot[2i]   = q[2i] * cos(theta) - q[2i+1] * sin(theta)
//!   q_rot[2i+1] = q[2i] * sin(theta) + q[2i+1] * cos(theta)

/// Precomputed cos/sin tables for RoPE.
pub struct RopeTable {
    /// cos_table[pos][pair] for each position and dimension pair.
    cos: Vec<Vec<f32>>,
    /// sin_table[pos][pair].
    sin: Vec<Vec<f32>>,
    /// Number of dimension pairs (head_dim / 2).
    n_pairs: usize,
    /// Maximum sequence length.
    max_seq: usize,
}

impl RopeTable {
    /// Build RoPE tables for the given head dimension and max sequence length.
    /// base is typically 10000.0 (Qwen3 may use different base).
    pub fn new(head_dim: usize, max_seq: usize, base: f32) -> Self {
        assert!(head_dim % 2 == 0, "head_dim must be even");
        let n_pairs = head_dim / 2;

        let mut cos = Vec::with_capacity(max_seq);
        let mut sin = Vec::with_capacity(max_seq);

        for pos in 0..max_seq {
            let mut cos_row = Vec::with_capacity(n_pairs);
            let mut sin_row = Vec::with_capacity(n_pairs);
            for i in 0..n_pairs {
                let theta = (pos as f64) / (base as f64).powf(2.0 * i as f64 / head_dim as f64);
                cos_row.push(theta.cos() as f32);
                sin_row.push(theta.sin() as f32);
            }
            cos.push(cos_row);
            sin.push(sin_row);
        }

        Self { cos, sin, n_pairs, max_seq }
    }

    /// Apply RoPE rotation to a vector at the given position.
    /// Modifies `vec` in-place. `vec` has length head_dim.
    pub fn apply(&self, vec: &mut [f32], position: usize) {
        assert!(position < self.max_seq, "position {} >= max_seq {}", position, self.max_seq);
        let cos = &self.cos[position];
        let sin = &self.sin[position];

        for i in 0..self.n_pairs {
            let x0 = vec[2 * i];
            let x1 = vec[2 * i + 1];
            vec[2 * i]     = x0 * cos[i] - x1 * sin[i];
            vec[2 * i + 1] = x0 * sin[i] + x1 * cos[i];
        }
    }

    /// Apply RoPE to an i16 quantized vector. Returns rotated i16.
    /// Uses the float cos/sin tables but keeps input/output as i16.
    pub fn apply_i16(&self, vec: &mut [i16], scale: f32, position: usize) {
        assert!(position < self.max_seq);
        let cos = &self.cos[position];
        let sin = &self.sin[position];

        for i in 0..self.n_pairs {
            let x0 = vec[2 * i] as f32 * scale;
            let x1 = vec[2 * i + 1] as f32 * scale;
            let r0 = x0 * cos[i] - x1 * sin[i];
            let r1 = x0 * sin[i] + x1 * cos[i];
            vec[2 * i]     = (r0 / scale).round().clamp(-32767.0, 32767.0) as i16;
            vec[2 * i + 1] = (r1 / scale).round().clamp(-32767.0, 32767.0) as i16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_position_zero_is_identity() {
        let table = RopeTable::new(64, 128, 10000.0);
        let original = vec![1.0f32; 64];
        let mut rotated = original.clone();
        table.apply(&mut rotated, 0);
        // At position 0, all thetas = 0, cos=1, sin=0 → identity.
        let max_diff: f32 = original.iter().zip(rotated.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-6, "position 0 should be identity, diff={}", max_diff);
    }

    #[test]
    fn rope_different_positions_differ() {
        let table = RopeTable::new(64, 128, 10000.0);
        let input = vec![1.0f32; 64];

        let mut pos1 = input.clone();
        let mut pos2 = input.clone();
        table.apply(&mut pos1, 1);
        table.apply(&mut pos2, 50);

        let diff: f32 = pos1.iter().zip(pos2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.1, "different positions should produce different rotations");
    }

    #[test]
    fn rope_i16_roundtrip() {
        let table = RopeTable::new(8, 16, 10000.0);
        let scale = 0.01f32;
        let mut vec = vec![1000i16, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
        let original = vec.clone();
        table.apply_i16(&mut vec, scale, 0);
        // Position 0 is identity.
        for i in 0..8 {
            assert!((vec[i] - original[i]).abs() <= 1, "pos0 i16 should be identity");
        }
    }
}
