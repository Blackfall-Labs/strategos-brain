//! SignalLinear — integer matrix multiply via Signode grid.
//!
//! Weight matrix stored as Signode cells. Forward pass: K-gated signed
//! accumulation in i64. No floats. No conduct() — direct P-sign × M multiply
//! avoids K-attenuation distorting weight values.
//!
//! Encoding:
//!   P = sign(w) × 1000        (i16, direction only — never 0)
//!   M = |w| / scale × M_MAX  (u16, magnitude)
//!   K = K_MAX                 (fully confident — gating is the pruning knob)
//!
//! Forward (per output row i):
//!   acc = 0i64
//!   for j in 0..cols:
//!     if cell[i,j].k < gate_k: skip
//!     sign_w = sign(cell[i,j].p)
//!     sign_x = sign(input[j].p)   (0 input → skip)
//!     acc += sign_w * cell[i,j].m as i64 * sign_x * input[j].m as i64
//!   output[i] = acc
//!
//! The caller rescales: actual_value = acc × weight_scale × input_scale / M_MAX²

use signode::{Signal, Signode, SigUint, K_MAX, M_MAX, SIGNODE_SIZE};

pub struct SignalLinear {
    rows: usize,
    cols: usize,
    cells: Vec<Signode>,
    weight_scale: f32,
}

impl SignalLinear {
    /// Encode a float weight matrix into a Signode grid.
    ///
    /// `scale` = max(|w|) / (M_MAX × 0.9) — caller computes this so it can
    /// store it in the scales manifest.
    pub fn from_weights(w: &[f32], rows: usize, cols: usize, scale: f32) -> Self {
        assert_eq!(w.len(), rows * cols, "weight slice length must equal rows×cols");
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 1.0 };
        let cells: Vec<Signode> = w.iter().map(|&v| {
            let p: i16 = if v >= 0.0 { 1000 } else { -1000 };
            let m = ((v.abs() * inv_scale) * M_MAX as f32).clamp(0.0, M_MAX as f32) as SigUint;
            Signode {
                signal: Signal { p, m, k: K_MAX },
                accumulator: 0,
                sensitivity: K_MAX,
            }
        }).collect();
        Self { rows, cols, cells, weight_scale: scale }
    }

    /// K-gated signed integer accumulation.
    ///
    /// Returns raw i64 per output row. Caller rescales using `weight_scale()`
    /// and the input's own scale to recover float-equivalent values.
    pub fn forward(&self, input: &[Signal], gate_k: SigUint) -> Vec<i64> {
        assert_eq!(input.len(), self.cols, "input length must match cols");
        let mut out = vec![0i64; self.rows];
        for i in 0..self.rows {
            let base = i * self.cols;
            let mut acc = 0i64;
            for j in 0..self.cols {
                let cell = &self.cells[base + j];
                if cell.signal.k < gate_k { continue; }
                let xm = input[j].m;
                if xm == 0 { continue; }
                let xp = input[j].p;
                if xp == 0 { continue; }
                let sign_w: i64 = if cell.signal.p > 0 { 1 } else { -1 };
                let sign_x: i64 = if xp > 0 { 1 } else { -1 };
                acc += sign_w * cell.signal.m as i64 * sign_x * xm as i64;
            }
            out[i] = acc;
        }
        out
    }

    /// Conv1D transpose: stored as [in_dim, out_dim] → forward as [out_dim, in_dim].
    pub fn from_weights_t(w: &[f32], in_dim: usize, out_dim: usize, scale: f32) -> Self {
        assert_eq!(w.len(), in_dim * out_dim);
        let mut wt = vec![0.0f32; out_dim * in_dim];
        for i in 0..in_dim {
            for j in 0..out_dim {
                wt[j * in_dim + i] = w[i * out_dim + j];
            }
        }
        Self::from_weights(&wt, out_dim, in_dim, scale)
    }

    /// Return the raw Signals for row `idx` — for embedding table lookups.
    pub fn row(&self, idx: usize) -> Vec<Signal> {
        assert!(idx < self.rows, "row index out of bounds");
        let base = idx * self.cols;
        self.cells[base..base + self.cols].iter().map(|n| n.signal).collect()
    }

    pub fn weight_scale(&self) -> f32 { self.weight_scale }
    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }

    /// Serialize: 8B rows + 8B cols + 4B scale + N×SIGNODE_SIZE cells (LE).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(20 + self.cells.len() * SIGNODE_SIZE);
        buf.extend_from_slice(&(self.rows as u64).to_le_bytes());
        buf.extend_from_slice(&(self.cols as u64).to_le_bytes());
        buf.extend_from_slice(&self.weight_scale.to_le_bytes());
        for cell in &self.cells {
            buf.extend_from_slice(&cell.to_bytes());
        }
        buf
    }

    /// Deserialize from bytes written by `to_bytes`.
    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() < 20 { return None; }
        let rows = u64::from_le_bytes(b[0..8].try_into().ok()?) as usize;
        let cols = u64::from_le_bytes(b[8..16].try_into().ok()?) as usize;
        let weight_scale = f32::from_le_bytes(b[16..20].try_into().ok()?);
        let n = rows.checked_mul(cols)?;
        if b.len() < 20 + n * SIGNODE_SIZE { return None; }
        let mut cells = Vec::with_capacity(n);
        for i in 0..n {
            let off = 20 + i * SIGNODE_SIZE;
            cells.push(Signode::from_bytes(&b[off..off + SIGNODE_SIZE])?);
        }
        Some(Self { rows, cols, cells, weight_scale })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use signode::Signal;

    fn sig(p: i16, m: u16) -> Signal {
        Signal { p, m, k: K_MAX }
    }

    #[test]
    fn identity_row() {
        // Single row [1.0], input [1.0] → acc = M_MAX * M_MAX
        let scale = 1.0f32 / (M_MAX as f32 * 0.9);
        let lin = SignalLinear::from_weights(&[1.0f32], 1, 1, scale);
        let input = [sig(1000, M_MAX)];
        let out = lin.forward(&input, 0);
        assert_eq!(out.len(), 1);
        assert!(out[0] > 0, "positive weight × positive input must be positive");
    }

    #[test]
    fn sign_flip() {
        let scale = 1.0f32 / (M_MAX as f32 * 0.9);
        let lin = SignalLinear::from_weights(&[-1.0f32], 1, 1, scale);
        let input = [sig(1000, M_MAX)];
        let out = lin.forward(&input, 0);
        assert!(out[0] < 0, "negative weight × positive input must be negative");
    }

    #[test]
    fn gate_k_prunes() {
        let scale = 1.0f32 / (M_MAX as f32 * 0.9);
        let lin = SignalLinear::from_weights(&[1.0f32], 1, 1, scale);
        let input = [sig(1000, M_MAX)];
        // Gate above K_MAX → all cells pruned → acc = 0
        let out = lin.forward(&input, K_MAX);
        assert_eq!(out[0], 0, "gate_k = K_MAX must prune everything");
    }

    #[test]
    fn roundtrip_bytes() {
        let w = vec![0.5f32, -0.3, 1.0, -1.0];
        let scale = 1.0 / (M_MAX as f32 * 0.9);
        let lin = SignalLinear::from_weights(&w, 2, 2, scale);
        let bytes = lin.to_bytes();
        let lin2 = SignalLinear::from_bytes(&bytes).unwrap();
        assert_eq!(lin2.rows(), 2);
        assert_eq!(lin2.cols(), 2);
        let input = [sig(1000, 10000), sig(-1000, 5000)];
        assert_eq!(lin.forward(&input, 0), lin2.forward(&input, 0));
    }

    #[test]
    fn multi_row_accumulation() {
        // 2×2 matrix [[1, -1], [1, 1]], input [1, 1]
        // row 0: +M*M - M*M = 0
        // row 1: +M*M + M*M > 0
        let scale = 1.0 / (M_MAX as f32 * 0.9);
        let w = [1.0f32, -1.0, 1.0, 1.0];
        let lin = SignalLinear::from_weights(&w, 2, 2, scale);
        let input = [sig(1000, 30000), sig(1000, 30000)];
        let out = lin.forward(&input, 0);
        assert_eq!(out[0], 0, "cancellation row must sum to 0");
        assert!(out[1] > 0, "additive row must be positive");
    }
}
