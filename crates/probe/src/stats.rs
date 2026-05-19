use serde::Serialize;

const RESERVOIR_SIZE: usize = 2000;

/// Welford online mean/variance + min/max + reservoir sample for percentiles.
pub struct ScalarAccum {
    count: u64,
    mean: f64,
    m2: f64,
    min: f32,
    max: f32,
    reservoir: Vec<f32>,
    reservoir_cursor: usize,
}

impl ScalarAccum {
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f32::MAX,
            max: f32::MIN,
            reservoir: Vec::with_capacity(RESERVOIR_SIZE),
            reservoir_cursor: 0,
        }
    }

    pub fn observe(&mut self, values: &[f32]) {
        for &v in values {
            self.count += 1;
            let delta = v as f64 - self.mean;
            self.mean += delta / self.count as f64;
            self.m2 += delta * (v as f64 - self.mean);
            if v < self.min { self.min = v; }
            if v > self.max { self.max = v; }
            if self.reservoir.len() < RESERVOIR_SIZE {
                self.reservoir.push(v);
            } else {
                self.reservoir[self.reservoir_cursor % RESERVOIR_SIZE] = v;
                self.reservoir_cursor += 1;
            }
        }
    }

    pub fn finalize(self) -> ScalarStats {
        let std = if self.count > 1 {
            (self.m2 / (self.count - 1) as f64).sqrt() as f32
        } else {
            0.0
        };
        let mut sorted = self.reservoir.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p5 = sorted.get(sorted.len() / 20).copied().unwrap_or(0.0);
        let p95 = sorted.get(sorted.len() * 19 / 20).copied().unwrap_or(0.0);
        ScalarStats {
            mean: self.mean as f32,
            std,
            min: if self.min == f32::MAX { 0.0 } else { self.min },
            max: if self.max == f32::MIN { 0.0 } else { self.max },
            p5,
            p95,
            n: self.count,
        }
    }
}

#[derive(Serialize)]
pub struct ScalarStats {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub p5: f32,
    pub p95: f32,
    pub n: u64,
}

/// Per-neuron activation frequency tracker.
/// Counts how often each neuron fires above `threshold` (absolute value).
pub struct NeuronFreqAccum {
    pub n_neurons: usize,
    pub fire_count: Vec<u64>,
    pub total: u64,
    pub threshold: f32,
}

impl NeuronFreqAccum {
    pub fn new(n_neurons: usize, threshold: f32) -> Self {
        Self {
            n_neurons,
            fire_count: vec![0u64; n_neurons],
            total: 0,
            threshold,
        }
    }

    pub fn observe(&mut self, values: &[f32]) {
        self.total += 1;
        for (i, &v) in values.iter().enumerate().take(self.n_neurons) {
            if v.abs() > self.threshold {
                self.fire_count[i] += 1;
            }
        }
    }

    pub fn finalize(self) -> Vec<f32> {
        if self.total == 0 {
            return vec![0.0; self.n_neurons];
        }
        self.fire_count.iter()
            .map(|&c| c as f32 / self.total as f32)
            .collect()
    }
}

/// Per-head Shannon entropy tracker over softmax attention weights.
/// High entropy = diffuse attention (uniform head). Low entropy = spiky (induction-style head).
pub struct HeadEntropyAccum {
    pub n_heads: usize,
    entropy_sum: Vec<f64>,
    count: u64,
}

impl HeadEntropyAccum {
    pub fn new(n_heads: usize) -> Self {
        Self {
            n_heads,
            entropy_sum: vec![0.0; n_heads],
            count: 0,
        }
    }

    /// head_scores[h] = softmax attention weights for head h over the current sequence.
    pub fn observe(&mut self, head_scores: &[Vec<f32>]) {
        self.count += 1;
        for (h, scores) in head_scores.iter().enumerate().take(self.n_heads) {
            let mut entropy = 0.0f64;
            for &p in scores {
                if p > 1e-12 {
                    entropy -= p as f64 * (p as f64).ln();
                }
            }
            self.entropy_sum[h] += entropy;
        }
    }

    pub fn finalize(self) -> Vec<f32> {
        if self.count == 0 {
            return vec![0.0; self.n_heads];
        }
        self.entropy_sum.iter()
            .map(|&s| (s / self.count as f64) as f32)
            .collect()
    }
}
