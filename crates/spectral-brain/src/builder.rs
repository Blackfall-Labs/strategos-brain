use serde::Deserialize;
use std::collections::HashMap;
use crate::column::Column;

/// Output of export_band_topology.py
#[derive(Deserialize, Debug)]
pub struct BrainTopology {
    pub n_cols_total:    usize,
    pub tier0_clusters:  Vec<Tier0Cluster>,
    pub tier1_sheets:    Vec<SpectralSheet>,
    pub tier2_sheets:    Vec<SpectralSheet>,
    /// Band 2/3 token entries for broader injection coverage.
    #[serde(default)]
    pub band23_tokens:   Vec<Band23Token>,
}

/// A Band 2 or 3 token trigger — context-dependent, so magnitude is lower.
#[derive(Deserialize, Debug)]
pub struct Band23Token {
    pub token_id:      u32,
    pub band:          u8,    // 2 or 3
    pub band_id:       u32,   // where to broadcast (within T1 range)
    pub magnitude:     u32,   // scaled from mean trigger_rate
}

/// One Band 1 cluster = one recognition sheet in Tier 0.
/// Each member neuron becomes one column tuned to token_band_id.
#[derive(Deserialize, Debug)]
pub struct Tier0Cluster {
    pub cluster_id:   usize,
    pub trigger_token: String,
    pub token_id:     u32,
    pub band_id:      u32,   // token frequency rank → band ID
    pub members:      Vec<Tier0Neuron>,
}

#[derive(Deserialize, Debug)]
pub struct Tier0Neuron {
    pub neuron_idx:   usize,
    pub trigger_rate: f32,   // → threshold (1 - trigger_rate maps to how easy to fire)
}

/// Tier 1 and Tier 2 sheets are populations of columns covering a band range.
#[derive(Deserialize, Debug)]
pub struct SpectralSheet {
    pub sheet_id:  usize,
    pub tier:      u8,
    pub band_lo:   u32,
    pub band_hi:   u32,
    pub n_cols:    usize,
    pub pv_scales: Vec<f32>,  // per-column, from neuron_freq histogram
    pub always_on_frac: f32,  // fraction of columns that are always-on
}

pub fn build_columns(topo: &BrainTopology) -> Vec<Column> {
    let mut cols: Vec<Column> = Vec::with_capacity(topo.n_cols_total);
    let mut id = 0usize;

    // Tier 0: one column per neuron in each Band 1 cluster
    for cluster in &topo.tier0_clusters {
        for neuron in &cluster.members {
            let threshold = trigger_rate_to_threshold(neuron.trigger_rate);
            cols.push(Column {
                id,
                tier:      0,
                band_lo:   cluster.band_id,
                band_hi:   cluster.band_id,  // exact single-band tuning
                threshold,
                k:         0,
                k_reset:   3,   // 3-tick refractory for recognition columns
                always_on: false,
            });
            id += 1;
        }
    }

    // Tier 1: integration sheets
    for sheet in &topo.tier1_sheets {
        add_sheet_columns(&mut cols, &mut id, sheet);
    }

    // Tier 2: composition sheets (some always-on)
    for sheet in &topo.tier2_sheets {
        add_sheet_columns(&mut cols, &mut id, sheet);
    }

    cols
}

fn add_sheet_columns(cols: &mut Vec<Column>, id: &mut usize, sheet: &SpectralSheet) {
    let always_on_count = (sheet.n_cols as f32 * sheet.always_on_frac).round() as usize;
    for i in 0..sheet.n_cols {
        let pv = sheet.pv_scales.get(i).copied().unwrap_or(1.0);
        let threshold = pv_scale_to_threshold(pv);
        let always_on = i < always_on_count;
        cols.push(Column {
            id: *id,
            tier:     sheet.tier,
            band_lo:  sheet.band_lo,
            band_hi:  sheet.band_hi,
            threshold,
            k:        0,
            k_reset:  if sheet.tier == 2 { 5 } else { 4 },
            always_on,
        });
        *id += 1;
    }
}

/// trigger_rate=1.0 → very easy to fire (low threshold). trigger_rate=0.1 → harder.
/// Threshold range: 2000 (easy) to 20000 (hard). Scaled to u32 (fits u16 in practice).
fn trigger_rate_to_threshold(tr: f32) -> u32 {
    let t = 2000.0 + (1.0 - tr.clamp(0.0, 1.0)) * 18000.0;
    t as u32
}

/// pv_scale: sampled from neuron_freq histogram. Low freq → high pv_scale → higher threshold.
/// pv_scale range [0.5, 2.0] → threshold [5000, 20000].
fn pv_scale_to_threshold(pv: f32) -> u32 {
    let t = (pv.clamp(0.5, 2.0) / 2.0) * 20000.0;
    t as u32
}

/// Build the token → (band_id, magnitude) lookup.
/// Tier 0 (Band 1) takes priority. Band 2/3 fill in tokens not in Band 1.
pub fn build_token_map(topo: &BrainTopology) -> HashMap<u32, (u32, u32)> {
    let mut map = HashMap::new();

    // Band 2/3 first (lower priority — overwritten by Band 1)
    for t in &topo.band23_tokens {
        map.insert(t.token_id, (t.band_id, t.magnitude));
    }

    // Band 1 overwrites — strongest signal takes precedence
    for cluster in &topo.tier0_clusters {
        let mean_tr = if cluster.members.is_empty() {
            0.5
        } else {
            cluster.members.iter().map(|n| n.trigger_rate).sum::<f32>() / cluster.members.len() as f32
        };
        let magnitude = (mean_tr * 30000.0) as u32;
        map.insert(cluster.token_id, (cluster.band_id, magnitude));
    }

    map
}
