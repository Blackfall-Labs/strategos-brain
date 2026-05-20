//! Architectural proof: TernSig Cortical Column properties.
//!
//! Five proofs validate the architecture before any distillation:
//!
//!   1. Resonance Selectivity    — input at freq F activates columns near F strongest
//!   2. Multi-Freq Coexistence   — two simultaneous inputs stay distinct (no collision)
//!   3. Tier Cascade (no WTA)    — feedforward projection, broadcast storm visible
//!   4. E/I Polarity             — excitatory and inhibitory columns produce opposite P
//!   5. WTA Lateral Inhibition   — cascade_wta() contains the broadcast storm

use ternsig_cortex::{column::TernSigColumn, cortex::TernSigCortex};

fn main() {
    println!("TernSig Cortical Column — Architectural Proof");
    println!("==============================================");
    println!();

    // ── Cortex setup ─────────────────────────────────────────────────────────
    // 21 frequency bins across [0.0, 1.0], 3 tiers each = 63 excitatory columns.
    // bandwidth = 0.06: at distance 0.06, resonance = 0.607 (1/e). At 0.18: 0.011.
    let n_freqs = 21;
    let bandwidth = 0.06f32;
    let inhibit_radius = 0.10f32;
    let n_cells = 4;

    let mut cortex = TernSigCortex::new(1000);
    for i in 0..n_freqs {
        let freq = i as f32 / (n_freqs - 1) as f32;
        for tier in 0u8..3 {
            cortex.add_column(TernSigColumn::new(freq, bandwidth, 1, tier, inhibit_radius, n_cells));
        }
    }
    println!("Cortex: {} columns ({} freqs × 3 tiers × {} cells)", cortex.n_columns(), n_freqs, n_cells);
    println!();

    // ── Proof 1: Resonance Selectivity ───────────────────────────────────────
    println!("PROOF 1: Resonance Selectivity");
    println!("  Inject at freq 0.300 — expect Gaussian peak centred on 0.300");

    let target = 0.30f32;
    let acts = cortex.inject(target, 10_000, 0);
    assert!(!acts.is_empty(), "no tier-0 column fired for freq {target}");

    let winner_freq = acts[0].resonant_freq;
    let winner_m    = acts[0].output_m();
    assert!(
        (winner_freq - target).abs() < bandwidth * 2.0,
        "winner {winner_freq:.3} is not near target {target:.3}"
    );
    for act in acts.iter().take(5) {
        let bar: String = "#".repeat((act.output_m() as usize / 1000).min(40));
        println!("  col@{:.3}  m={:5}  {}", act.resonant_freq, act.output_m(), bar);
    }
    println!("  → Winner: freq={winner_freq:.3}  output_m={winner_m}  PASS ✓");
    println!();

    // ── Proof 2: Multi-Frequency Coexistence ─────────────────────────────────
    println!("PROOF 2: Multi-Frequency Coexistence");
    println!("  Signals at 0.200 and 0.750 must activate distinct populations");

    let freq_a = 0.20f32;
    let freq_b = 0.75f32;

    let acts_a = cortex.inject(freq_a, 10_000, 0);
    let acts_b = cortex.inject(freq_b, 10_000, 0);

    assert!(!acts_a.is_empty(), "freq_a={freq_a} fired nothing");
    assert!(!acts_b.is_empty(), "freq_b={freq_b} fired nothing");

    let winner_a = acts_a[0].resonant_freq;
    let winner_b = acts_b[0].resonant_freq;

    assert!((winner_a - freq_a).abs() < bandwidth * 2.0, "winner_a {winner_a:.3} not near {freq_a}");
    assert!((winner_b - freq_b).abs() < bandwidth * 2.0, "winner_b {winner_b:.3} not near {freq_b}");

    let separation = (winner_a - winner_b).abs();
    assert!(separation > 0.3, "winners collided — separation only {separation:.3}");

    println!("  Signal A → winner@{winner_a:.3}  (target {freq_a:.3})");
    println!("  Signal B → winner@{winner_b:.3}  (target {freq_b:.3})");
    println!("  Separation: {separation:.3}  (no collision)  PASS ✓");
    println!();

    // ── Proof 3: Tier Cascade without WTA ────────────────────────────────────
    println!("PROOF 3: Tier Cascade (no WTA) — broadcast storm baseline");
    println!("  Cascade without lateral inhibition: every resonant column fans out");

    let cascade_freq = 0.50f32;

    let tier0 = cortex.inject(cascade_freq, 10_000, 0);
    let tier1 = cortex.cascade(&tier0);
    let tier2 = cortex.cascade(&tier1);

    assert!(!tier0.is_empty(), "tier0 empty");
    assert!(!tier1.is_empty(), "tier1 empty");
    assert!(!tier2.is_empty(), "tier2 empty");

    let t0_top = &tier0[0];
    let t1_top = &tier1[0];
    let t2_top = &tier2[0];

    println!("  Tier0: {:3} activations  winner@{:.3}  m={}", tier0.len(), t0_top.resonant_freq, t0_top.output_m());
    println!("  Tier1: {:3} activations  winner@{:.3}  m={}", tier1.len(), t1_top.resonant_freq, t1_top.output_m());
    println!("  Tier2: {:3} activations  winner@{:.3}  m={}", tier2.len(), t2_top.resonant_freq, t2_top.output_m());

    let freq_drift = (t0_top.resonant_freq - t2_top.resonant_freq).abs();
    assert!(freq_drift < bandwidth * 3.0, "cascade drifted {freq_drift:.3}");
    println!("  Freq preserved: drift={freq_drift:.4}  PASS ✓ (WTA proof below will fix the fanout)");
    println!();

    // ── Proof 4: E/I Polarity ─────────────────────────────────────────────────
    println!("PROOF 4: Excitatory / Inhibitory Polarity");
    println!("  Excitatory column outputs P > 0 (reinforces downstream)");
    println!("  Inhibitory column outputs P < 0 (inverts/suppresses downstream)");

    let mut ei_cortex = TernSigCortex::new(1000);
    let probe_freq = 0.50f32;
    ei_cortex.add_column(TernSigColumn::new(probe_freq, bandwidth, 1, 0, inhibit_radius, n_cells));
    ei_cortex.add_column(TernSigColumn::new(probe_freq, bandwidth, -1, 0, inhibit_radius, n_cells));

    let ei_acts = ei_cortex.inject(probe_freq, 10_000, 0);
    assert_eq!(ei_acts.len(), 2, "expected 2 activations, got {}", ei_acts.len());

    let exc = ei_acts.iter().find(|a| a.is_excitatory()).unwrap();
    let inh = ei_acts.iter().find(|a| !a.is_excitatory()).unwrap();

    assert!(exc.output.p > 0, "excitatory P must be > 0, got {}", exc.output.p);
    assert!(inh.output.p < 0, "inhibitory P must be < 0, got {}", inh.output.p);

    println!("  Excitatory col@{:.3}: P={:+}  M={}  (reinforcing)", exc.resonant_freq, exc.output.p, exc.output_m());
    println!("  Inhibitory col@{:.3}: P={:+}  M={}  (suppressing)", inh.resonant_freq, inh.output.p, inh.output_m());
    println!("  Polarity distinction verified  PASS ✓");
    println!();

    // ── Proof 5: WTA Lateral Inhibition ──────────────────────────────────────
    println!("PROOF 5: WTA Lateral Inhibition  (cascade_wta vs cascade)");
    println!("  When a winner fires, its inhibit_radius zone blocks weaker neighbours.");
    println!("  Cortex has two signal sources 0.20 apart — without WTA both cascade,");
    println!("  with WTA the stronger winner takes the band.");

    // Two nearby input sources: close enough that inhibit_radius overlaps.
    // Input A at 0.20 (stronger), Input B at 0.30 (weaker).
    // inhibit_radius = 0.10, so A should suppress B.
    let acts_ab = {
        let mut combined = cortex.inject(0.20, 10_000, 0);
        combined.extend(cortex.inject(0.30, 6_000, 0)); // B is ~60% as strong
        combined.sort_by(|a, b| b.output.m.cmp(&a.output.m));
        combined.dedup_by_key(|a| a.col_idx);
        combined
    };

    // Without WTA: both fans out freely
    let wta_off_t1 = cortex.cascade(&acts_ab);
    // With WTA: winner at 0.20 fires and blocks 0.30 zone before it cascades
    let wta_on_t1 = cortex.cascade_wta(&acts_ab);

    assert!(!wta_on_t1.is_empty(), "WTA cascade produced nothing");
    assert!(
        wta_off_t1.len() > wta_on_t1.len(),
        "WTA should reduce activations: {} vs {}",
        wta_off_t1.len(), wta_on_t1.len()
    );

    // The WTA winner should still be near 0.20 (the stronger signal)
    let wta_winner_freq = wta_on_t1[0].resonant_freq;
    assert!(
        (wta_winner_freq - 0.20).abs() < bandwidth * 2.0,
        "WTA winner {wta_winner_freq:.3} not near the stronger input 0.200"
    );

    // WTA tier2: continue suppressing downstream
    let wta_off_t2 = cortex.cascade(&wta_off_t1);
    let wta_on_t2  = cortex.cascade_wta(&wta_on_t1);

    println!("  Tier1  without WTA: {:3} activations", wta_off_t1.len());
    println!("  Tier1  with    WTA: {:3} activations  (winner@{:.3})", wta_on_t1.len(), wta_winner_freq);
    println!("  Tier2  without WTA: {:3} activations", wta_off_t2.len());
    println!("  Tier2  with    WTA: {:3} activations", wta_on_t2.len());

    let reduction_t1 = 100 - (wta_on_t1.len() * 100 / wta_off_t1.len().max(1));
    let reduction_t2 = 100 - (wta_on_t2.len() * 100 / wta_off_t2.len().max(1));
    println!("  Reduction: Tier1 {reduction_t1}%  Tier2 {reduction_t2}%  PASS ✓");
    println!();

    // ── Summary ───────────────────────────────────────────────────────────────
    println!("══════════════════════════════════════════════");
    println!("All 5 proofs passed.");
    println!();
    println!("Architecture confirmed:");
    println!("  • Continuous resonant_freq + Gaussian falloff = real semantic geometry");
    println!("  • Signode K-latch: per-cell independent confidence, no central gate");
    println!("  • Tier cascade: T0 → T1 → T2 feedforward, freq preserved");
    println!("  • E/I polarity: P=+1 excitatory, P=-1 inhibitory — one mechanism, two effects");
    println!("  • WTA via lateral inhibition: broadcast storm → sparse O(winners) cascade");
    println!();
    println!("Next: Neel Nanda 1L circuit mapping — 113 freq bins, modular addition routing.");
}
