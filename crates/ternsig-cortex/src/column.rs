//! TernSigColumn — the cortical column primitive.
//!
//! Replaces the integer-band Column with a continuous resonant_freq + Signode memory.
//!
//! Key properties over the old Column:
//!   - resonant_freq: continuous f32 in [0.0, 1.0], not an integer band
//!   - Gaussian falloff: 500.0 and 500.3 Hz interfere — the falloff IS semantic similarity
//!   - Polarity: +1 excitatory (reinforces downstream), -1 inhibitory (inverts downstream)
//!   - Multi-latch: each Signode cell has its own K dynamics — no central gate
//!   - K confidence: cells that consistently fire harden; surprised cells soften

use signode::{Signal, Signode, SigUint, SigInt, M_MAX, K_MAX, M_SCALE, circuit};

/// Output of a column firing.
#[derive(Clone, Debug)]
pub struct ColumnOutput {
    /// Resonant frequency of the column that fired.
    pub resonant_freq: f32,
    /// Output polarity: +1000 excitatory, -1000 inhibitory (reflects column polarity).
    pub p: SigInt,
    /// Output magnitude after resonance scaling and cell conduction.
    pub m: SigUint,
    /// Average K across cells — confidence in this activation.
    pub avg_k: SigUint,
}

impl ColumnOutput {
    /// True if this output is excitatory (P > 0).
    pub fn is_excitatory(&self) -> bool { self.p > 0 }
}

/// A cortical column: a resonant unit backed by Signode multi-latch memory.
///
/// Tier structure maps to cortical depth:
///   0 = recognition  (L4 equivalent: receives thalamic input)
///   1 = integration   (L2/3 equivalent: lateral association, pattern formation)
///   2 = composition   (L5 equivalent: committed output)
pub struct TernSigColumn {
    /// Continuous semantic frequency in [0.0, 1.0].
    /// Synonyms at nearby values. Antonyms: same carrier, opposite polarity.
    /// Most general concepts near 0.0, specific concepts near 1.0.
    pub resonant_freq: f32,

    /// Gaussian bandwidth — how far resonance extends from resonant_freq.
    /// Narrow (0.01): sharp discrimination. Wide (0.1): broad tuning.
    pub bandwidth: f32,

    /// +1 excitatory: reinforces downstream. -1 inhibitory: inverts downstream.
    pub polarity: i8,

    /// Cortical depth. Tier N can only receive from tier N-1 (hop limit = 1).
    pub tier: u8,

    /// Frequency radius for lateral inhibition fanout.
    /// When this column fires: columns within this radius receive inhibitory signal.
    pub inhibit_radius: f32,

    /// The memory cells. Each has independent K-latch dynamics.
    /// Multiple cells = harder to miss, more confident activation.
    cells: Vec<Signode>,
}

impl TernSigColumn {
    /// Create a column with `n_cells` Signode memory cells.
    ///
    /// Cells are initialized with a baseline state:
    ///   P = ±1000 (excitatory/inhibitory direction)
    ///   M = M_SCALE (= 255 for u16) — transparent gain: output.m = resonance × input.m exactly
    ///   K = K_MAX/2 (moderate confidence — can conduct and can learn)
    ///
    /// M = M_SCALE is the "transparent" value: (input.m × stored.m / M_SCALE) = input.m.
    /// This makes resonance the sole discriminator — no artificial amplification hides selectivity.
    pub fn new(
        resonant_freq: f32,
        bandwidth: f32,
        polarity: i8,
        tier: u8,
        inhibit_radius: f32,
        n_cells: usize,
    ) -> Self {
        let p: SigInt = if polarity >= 0 { 1000 } else { -1000 };
        let cells = (0..n_cells).map(|_| {
            let mut node = Signode::ZERO;
            node.set(p, M_SCALE as SigUint, K_MAX / 2);
            node
        }).collect();
        Self { resonant_freq, bandwidth, polarity, tier, inhibit_radius, cells }
    }

    /// Gaussian resonance: how strongly does this column respond to signal at frequency `f`?
    ///
    /// Returns 1.0 at resonant_freq, falls off exponentially with distance.
    /// At f = resonant_freq ± bandwidth: response ≈ 0.607 (1/e).
    /// At f = resonant_freq ± 3×bandwidth: response ≈ 0.011 (negligible).
    pub fn resonance(&self, f: f32) -> f32 {
        let delta = (f - self.resonant_freq).abs();
        let b = self.bandwidth.max(1e-6);
        (-delta * delta / (2.0 * b * b)).exp()
    }

    /// Inject a signal at `input_freq` with `raw_m` magnitude.
    ///
    /// 1. Compute resonance (gaussian falloff from resonant_freq)
    /// 2. Scale magnitude by resonance — far-away signals pass weakly
    /// 3. Pass through each Signode cell (gate → amplify → scale)
    /// 4. Reduce via K-weighted blend (confident cells dominate)
    /// 5. Return None if column didn't fire (below threshold or K blocked)
    pub fn inject(&self, input_freq: f32, raw_m: SigUint, gate_threshold: SigUint) -> Option<ColumnOutput> {
        let r = self.resonance(input_freq);
        if r < 0.01 { return None; }

        let scaled_m = ((raw_m as f32 * r) as u32).min(M_MAX as u32) as SigUint;
        if scaled_m == 0 { return None; }

        // Input signal carries the column's own polarity direction.
        // Excitatory column: P=+1000 → amplify() preserves input direction → reinforces
        // Inhibitory column: P=-1000 → amplify() negates input direction → inverts/suppresses
        let input_p: SigInt = if self.polarity >= 0 { 1000 } else { -1000 };
        let input_sig = Signal { p: input_p, m: scaled_m, k: K_MAX };

        let cell_outputs: Vec<Signal> = self.cells.iter()
            .map(|cell| cell.conduct(input_sig, gate_threshold))
            .collect();

        let out = circuit::reduce(&cell_outputs);
        if out.m == 0 { return None; }

        Some(ColumnOutput {
            resonant_freq: self.resonant_freq,
            p: out.p,
            m: out.m,
            avg_k: self.avg_k(),
        })
    }

    /// Average K across all cells — the column's confidence level.
    /// High = confirmed pattern, resists change. Low = plastic, open to learning.
    pub fn avg_k(&self) -> SigUint {
        if self.cells.is_empty() { return 0; }
        let sum: u64 = self.cells.iter().map(|c| c.signal.k as u64).sum();
        (sum / self.cells.len() as u64) as SigUint
    }

    pub fn n_cells(&self) -> usize { self.cells.len() }

    /// Mutable access to cells for direct seeding or weight loading.
    pub fn cells_mut(&mut self) -> &mut Vec<Signode> { &mut self.cells }
}
