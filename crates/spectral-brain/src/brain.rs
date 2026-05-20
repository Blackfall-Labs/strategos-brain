use crate::column::Column;

#[derive(Clone, Debug)]
pub struct BroadcastEvent {
    pub source_col:  usize,
    pub source_tier: u8,
    pub band_id:     u32,
    pub magnitude:   u32,
}

/// Per-tick firing stats, split by cause.
#[derive(Default, Clone, Copy, Debug)]
pub struct FireStats {
    pub always_on: [usize; 3],  // always-on fires by tier
    pub cascade:   [usize; 3],  // cascade-driven fires by tier
}

impl FireStats {
    pub fn total(&self, tier: usize) -> usize { self.always_on[tier] + self.cascade[tier] }
}

pub struct SpectralBrain {
    pub columns:    Vec<Column>,
    pub always_on_count: [usize; 3],  // pre-computed always-on per tier

    pending:  Vec<BroadcastEvent>,
    next:     Vec<BroadcastEvent>,

    pub tick_count: u64,
    pub last_stats: FireStats,
}

impl SpectralBrain {
    pub fn new(columns: Vec<Column>) -> Self {
        let mut always_on_count = [0usize; 3];
        for col in &columns {
            if col.always_on {
                always_on_count[col.tier as usize] += 1;
            }
        }
        Self {
            columns,
            always_on_count,
            pending: Vec::new(),
            next:    Vec::new(),
            tick_count: 0,
            last_stats: FireStats::default(),
        }
    }

    pub fn inject(&mut self, band_id: u32, magnitude: u32, source_tier: u8) {
        self.pending.push(BroadcastEvent {
            source_col: usize::MAX,
            source_tier,
            band_id,
            magnitude,
        });
    }

    pub fn tick(&mut self) {
        let mut stats = FireStats::default();

        // Always-on columns fire unconditionally
        for col in &mut self.columns {
            if col.always_on && col.k == 0 {
                col.k = col.k_reset;
                self.next.push(BroadcastEvent {
                    source_col:  col.id,
                    source_tier: col.tier,
                    band_id:     col.band_lo,
                    magnitude:   col.threshold * 2,
                });
                stats.always_on[col.tier as usize] += 1;
            }
        }

        // Dispatch pending into columns
        for event in &self.pending {
            for col in &mut self.columns {
                let fired_m = col.evaluate(event.band_id, event.magnitude, event.source_tier);
                if fired_m > 0 {
                    self.next.push(BroadcastEvent {
                        source_col:  col.id,
                        source_tier: col.tier,
                        band_id:     col.band_lo + (col.band_hi - col.band_lo) / 2,
                        magnitude:   fired_m,
                    });
                    if col.always_on {
                        stats.always_on[col.tier as usize] += 1;
                    } else {
                        stats.cascade[col.tier as usize] += 1;
                    }
                }
            }
        }

        std::mem::swap(&mut self.pending, &mut self.next);
        self.next.clear();

        for col in &mut self.columns { col.tick(); }

        self.last_stats = stats;
        self.tick_count += 1;
    }
}
