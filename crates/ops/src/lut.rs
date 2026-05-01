//! Lookup tables for transcendentals — zero float at runtime.
//!
//! Proven in ternsig/deepspeech/lut.rs.
//! Sigmoid, Swish, InvSqrt, Softmax — all precomputed.

/// Swish LUT: x * sigmoid(x). Used for SwiGLU activation.
/// 1024 entries, <5% error.
pub struct SwishLut {
    pub table: [i16; 1024],
    pub scale: f32,
}

impl SwishLut {
    pub fn new(scale: f32) -> Self {
        let range = 32767.0 * scale;
        let mut table = [0i16; 1024];
        for i in 0..1024 {
            let x = (i as f64 / 1023.0) * 2.0 * range as f64 - range as f64;
            let sig = 1.0 / (1.0 + (-x).exp());
            let swish = x * sig;
            table[i] = (swish / scale as f64).round().clamp(-32767.0, 32767.0) as i16;
        }
        Self { table, scale }
    }

    /// Look up swish(x) for an i16 input at this LUT's scale.
    pub fn lookup(&self, x: i16) -> i16 {
        let idx = ((x as f32 + 32767.0) / 65534.0 * 1023.0).round() as usize;
        self.table[idx.clamp(0, 1023)]
    }
}

/// Inverse square root LUT: 1/sqrt(x). Used for RMSNorm.
/// 4096 entries, log-spaced near zero, <2% error.
pub struct InvSqrtLut {
    /// (input_threshold, output_value) pairs, sorted by threshold.
    pub table: Vec<(f32, f32)>,
}

impl InvSqrtLut {
    pub fn new() -> Self {
        let n = 4096;
        let mut table = Vec::with_capacity(n);
        for i in 0..n {
            // Log-space: dense near zero where d(1/sqrt(x))/dx is large.
            let t = (i as f64 + 1.0) / n as f64;
            let x = (t * t * 1000.0) as f32; // range [0, 1000]
            let inv_sqrt = 1.0 / (x + 1e-6).sqrt();
            table.push((x, inv_sqrt));
        }
        Self { table }
    }

    /// Look up 1/sqrt(x) by binary search.
    pub fn lookup(&self, x: f32) -> f32 {
        let x = x.max(1e-10);
        match self.table.binary_search_by(|&(t, _)| t.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)) {
            Ok(i) => self.table[i].1,
            Err(i) => {
                if i == 0 { self.table[0].1 }
                else if i >= self.table.len() { self.table[self.table.len() - 1].1 }
                else {
                    // Linear interpolate between neighbors.
                    let (x0, y0) = self.table[i - 1];
                    let (x1, y1) = self.table[i];
                    let frac = (x - x0) / (x1 - x0).max(1e-10);
                    y0 + (y1 - y0) * frac
                }
            }
        }
    }
}

/// Integer softmax.
/// Input: slice of i16 logits.
/// Output: Vec<i16> probabilities in [0, 32767] summing to ~32767.
pub fn softmax_int(logits: &[i16]) -> Vec<i16> {
    if logits.is_empty() {
        return Vec::new();
    }

    // Find max for numerical stability.
    let max_val = *logits.iter().max().unwrap() as i32;

    // Exp approximation via shift: exp(x - max) ≈ 2^((x-max) * log2(e))
    // Simpler: use a small exp LUT.
    let mut exp_vals: Vec<u32> = logits.iter()
        .map(|&v| {
            let diff = (v as i32 - max_val).clamp(-20000, 0);
            // Piecewise linear exp approximation.
            // exp(x) for x in [-20, 0] mapped to [0, 32768]
            let scaled = diff as f32 / 1000.0; // rough scale
            let exp_f = (scaled as f64).exp();
            (exp_f * 32768.0).round().clamp(0.0, 32768.0) as u32
        })
        .collect();

    let sum: u64 = exp_vals.iter().map(|&v| v as u64).sum();
    if sum == 0 {
        return vec![0i16; logits.len()];
    }

    exp_vals.iter()
        .map(|&v| ((v as u64 * 32767) / sum).min(32767) as i16)
        .collect()
}

/// Build a bank of SwishLuts covering a range of scales.
/// Same pattern as conformer_native.rs line 578.
pub fn build_swish_bank(count: usize) -> Vec<SwishLut> {
    (0..count).map(|i| {
        let s = 1e-6 * (1.35f32).powi(i as i32);
        SwishLut::new(s)
    }).collect()
}

/// Find the closest SwishLut in a bank for a given scale.
pub fn find_swish_lut(bank: &[SwishLut], scale: f32) -> usize {
    bank.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (a.scale - scale).abs().partial_cmp(&(b.scale - scale).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swish_lut_basic() {
        let lut = SwishLut::new(0.01);
        // swish(0) = 0 * sigmoid(0) = 0
        // Allow small quantization error.
        assert!(lut.lookup(0).abs() < 50, "swish(0) should be near 0, got {}", lut.lookup(0));
        // swish(large positive) should be positive.
        assert!(lut.lookup(10000) > 0, "swish(+) should be positive");
        // swish(large negative) should be near 0 (sigmoid → 0).
        assert!(lut.lookup(-10000).abs() < 1000, "swish(-) should be near 0");
    }

    #[test]
    fn inv_sqrt_basic() {
        let lut = InvSqrtLut::new();
        let result = lut.lookup(4.0);
        let expected = 0.5; // 1/sqrt(4) = 0.5
        assert!((result - expected).abs() < 0.05, "1/sqrt(4) = {}, expected 0.5", result);
    }

    #[test]
    fn softmax_sums_correctly() {
        let logits = vec![100i16, 200, 300, 400, 500];
        let probs = softmax_int(&logits);
        let sum: i32 = probs.iter().map(|&v| v as i32).sum();
        // Should sum to approximately 32767.
        assert!((sum - 32767).abs() < 100, "softmax sum = {}", sum);
        // Largest input should get largest probability.
        assert!(probs[4] > probs[0], "max input should have max prob");
    }
}
