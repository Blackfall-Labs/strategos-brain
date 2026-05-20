/// A frequency-tuned column. No axons. No wiring. Just tuning + threshold.
///
/// When a broadcast event arrives:
///   1. Is source_tier within hop_limit of my tier? If not, skip.
///   2. Does band_id fall within [band_lo, band_hi]? If not, skip.
///   3. Compute effective_m = magnitude * tuning_gain(band_id). Compare to threshold.
///   4. If fires: bump K (refractory), publish own broadcast.
///
/// K field: refractory counter. Decrements each tick. Column cannot fire while K > 0.

#[derive(Debug, Clone)]
pub struct Column {
    pub id:       usize,
    pub tier:     u8,           // 0=recognition, 1=integration, 2=composition
    pub band_lo:  u32,
    pub band_hi:  u32,
    pub threshold: u32,         // scaled 0..65535
    pub k:        u16,          // refractory countdown (ticks remaining)
    pub k_reset:  u16,          // refractory duration when fired
    pub always_on: bool,        // fires every tick regardless of input
}

impl Column {
    /// Returns fired magnitude (0 = did not fire).
    pub fn evaluate(&mut self, band_id: u32, magnitude: u32, source_tier: u8) -> u32 {
        if self.k > 0 { return 0; }
        if !self.hop_ok(source_tier) { return 0; }
        let gain = self.tuning_gain(band_id);
        if gain == 0 { return 0; }
        let effective = (magnitude as u64 * gain as u64 / 65536) as u32;
        if effective >= self.threshold {
            self.k = self.k_reset;
            effective
        } else {
            0
        }
    }

    pub fn tick(&mut self) {
        if self.k > 0 { self.k -= 1; }
    }

    /// Soft tuning: full gain at band centre, falls off toward edges. Zero outside.
    fn tuning_gain(&self, band_id: u32) -> u32 {
        if band_id < self.band_lo || band_id > self.band_hi {
            return 0;
        }
        let span = (self.band_hi - self.band_lo).max(1) as u64;
        let centre = (self.band_lo + self.band_hi) as u64 / 2;
        let dist = (band_id as i64 - centre as i64).unsigned_abs() as u64;
        // gain = 65536 at centre, 16384 at band edge (1/4 roll-off)
        let gain = 65536u64 - (dist * 49152 / span);
        gain.min(65536) as u32
    }

    /// Columns more than 1 tier away from source don't receive the signal.
    fn hop_ok(&self, source_tier: u8) -> bool {
        (self.tier as i8 - source_tier as i8).unsigned_abs() <= 1
    }
}
