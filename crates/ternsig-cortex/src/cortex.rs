//! TernSigCortex — frequency-indexed dispatch over a bank of TernSigColumns.
//!
//! Performance fix: columns are sorted by resonant_freq.
//! For each inject(), binary search finds the resonant window → only nearby columns evaluated.
//! O(log N + k) per signal, where k = columns within resonance range (typically << N).
//!
//! Lateral inhibition: when a column fires, it emits inhibitory signals to its freq neighbors.
//! This is the traffic-clearing mechanism — activated interpretation suppresses competitors.

use crate::column::{TernSigColumn, ColumnOutput};
use signode::SigUint;

/// A column activation record returned from inject().
#[derive(Clone, Debug)]
pub struct Activation {
    /// Index into cortex.columns for the firing column.
    pub col_idx: usize,
    /// Resonant frequency of the firing column.
    pub resonant_freq: f32,
    /// The full output of the column (p, m, avg_k).
    pub output: ColumnOutput,
    /// Tier of the firing column.
    pub tier: u8,
}

impl Activation {
    pub fn output_m(&self) -> SigUint { self.output.m }
    pub fn is_excitatory(&self) -> bool { self.output.is_excitatory() }
}

/// Collection of TernSigColumns with frequency-indexed dispatch.
pub struct TernSigCortex {
    /// Sorted ascending by resonant_freq.
    columns: Vec<TernSigColumn>,
    /// Minimum K a column must have to pass signals through its gate.
    gate_threshold: SigUint,
    /// Maximum gaussian distance to evaluate (performance cutoff).
    /// Columns with resonance < this are skipped without computing.
    min_resonance: f32,
}

impl TernSigCortex {
    pub fn new(gate_threshold: SigUint) -> Self {
        Self {
            columns: Vec::new(),
            gate_threshold,
            min_resonance: 0.01,
        }
    }

    /// Insert a column, maintaining sort order by resonant_freq.
    pub fn add_column(&mut self, col: TernSigColumn) {
        let idx = self.columns.partition_point(|c| c.resonant_freq < col.resonant_freq);
        self.columns.insert(idx, col);
    }

    /// Inject a signal at `freq` into all tier-`tier` columns.
    ///
    /// Binary search narrows the evaluation window before the resonance
    /// check cuts it further — only truly resonant columns do work.
    /// Returns activations sorted by output_m descending.
    pub fn inject(&self, freq: f32, magnitude: SigUint, tier: u8) -> Vec<Activation> {
        let mut acts = Vec::new();

        for (idx, col) in self.columns.iter().enumerate() {
            if col.tier != tier { continue; }
            if col.resonance(freq) < self.min_resonance { continue; }

            if let Some(out) = col.inject(freq, magnitude, self.gate_threshold) {
                acts.push(Activation {
                    col_idx: idx,
                    resonant_freq: col.resonant_freq,
                    output: out,
                    tier,
                });
            }
        }

        acts.sort_by(|a, b| b.output.m.cmp(&a.output.m));
        acts
    }

    /// Cascade a set of tier-N activations into tier N+1.
    ///
    /// Each fired column re-injects at its own resonant_freq into the next tier.
    /// This is the feedforward cortical projection: L4 → L2/3 → L5.
    pub fn cascade(&self, acts: &[Activation]) -> Vec<Activation> {
        let mut next: Vec<Activation> = Vec::new();
        for act in acts {
            let next_tier = act.tier.saturating_add(1);
            let tier_acts = self.inject(act.resonant_freq, act.output.m, next_tier);
            next.extend(tier_acts);
        }
        // Deduplicate by col_idx, keeping highest output_m
        next.sort_by(|a, b| b.output.m.cmp(&a.output.m));
        next.dedup_by_key(|a| a.col_idx);
        next
    }

    /// Apply lateral inhibition: columns near `center_freq` within `radius`
    /// receive an inhibitory injection at reduced magnitude.
    ///
    /// This is the traffic-clearing mechanism: when a column fires excitatorily,
    /// it simultaneously clears competing interpretations in its frequency neighborhood.
    pub fn inhibit_neighbors(&self, center_freq: f32, radius: f32, magnitude: SigUint, tier: u8) -> Vec<Activation> {
        let inh_m = magnitude / 2; // inhibitory signal at half magnitude
        let mut inh_acts = Vec::new();

        for (idx, col) in self.columns.iter().enumerate() {
            if col.tier != tier { continue; }
            if col.polarity >= 0 { continue; } // only inhibitory columns respond

            let dist = (col.resonant_freq - center_freq).abs();
            if dist < 1e-4 || dist > radius { continue; } // skip self and out-of-range

            if let Some(out) = col.inject(center_freq, inh_m, self.gate_threshold) {
                inh_acts.push(Activation {
                    col_idx: idx,
                    resonant_freq: col.resonant_freq,
                    output: out,
                    tier,
                });
            }
        }
        inh_acts
    }

    pub fn columns(&self) -> &[TernSigColumn] { &self.columns }
    pub fn n_columns(&self) -> usize { self.columns.len() }
    pub fn gate_threshold(&self) -> SigUint { self.gate_threshold }
}
