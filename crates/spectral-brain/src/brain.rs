use crate::column::Column;

/// A broadcast event emitted when any column fires.
#[derive(Clone, Debug)]
pub struct BroadcastEvent {
    pub source_col:  usize,
    pub source_tier: u8,
    pub band_id:     u32,
    pub magnitude:   u32,
}

pub struct SpectralBrain {
    pub columns: Vec<Column>,

    /// Pending events for this tick. Double-buffered so columns fired this tick
    /// don't recursively re-evaluate in the same tick.
    pending:  Vec<BroadcastEvent>,
    next:     Vec<BroadcastEvent>,

    pub tick_count: u64,
    pub fired_this_tick: Vec<usize>,  // column IDs that fired
}

impl SpectralBrain {
    pub fn new(columns: Vec<Column>) -> Self {
        Self {
            columns,
            pending: Vec::new(),
            next:    Vec::new(),
            tick_count: 0,
            fired_this_tick: Vec::new(),
        }
    }

    /// Inject an external event (e.g. token recognition from Band 1 trigger lookup).
    /// source_tier=0, band_id derived from token cluster, magnitude from trigger_rate.
    pub fn inject(&mut self, band_id: u32, magnitude: u32, source_tier: u8) {
        self.pending.push(BroadcastEvent {
            source_col: usize::MAX, // external input
            source_tier,
            band_id,
            magnitude,
        });
    }

    /// One brain tick:
    /// 1. Always-on columns fire unconditionally → add to pending.
    /// 2. Dispatch pending events to all columns.
    /// 3. Columns that fire add events to next buffer.
    /// 4. Swap buffers. Tick K counters.
    pub fn tick(&mut self) {
        self.fired_this_tick.clear();

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
                self.fired_this_tick.push(col.id);
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
                    self.fired_this_tick.push(col.id);
                }
            }
        }

        // Swap buffers
        std::mem::swap(&mut self.pending, &mut self.next);
        self.next.clear();

        // Tick refractory counters
        for col in &mut self.columns {
            col.tick();
        }

        self.tick_count += 1;
    }

    pub fn fired_by_tier(&self) -> [usize; 3] {
        let mut counts = [0usize; 3];
        for &id in &self.fired_this_tick {
            let tier = self.columns[id].tier as usize;
            if tier < 3 { counts[tier] += 1; }
        }
        counts
    }
}
