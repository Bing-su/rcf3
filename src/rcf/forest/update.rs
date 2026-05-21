use rand::prelude::*;

use super::Forest;
use crate::error::Result;
use crate::rcf::sampler::reservoir_weight;

impl Forest {
    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /// Incorporate a new observation into the forest.
    ///
    /// When `internal_shingling` is true, pass one base observation of length
    /// `input_dim`.  Otherwise pass the full shingled vector of length
    /// `input_dim * shingle_size`.
    pub fn update(&mut self, base: &[f32]) -> Result<()> {
        if self.config.internal_shingling() {
            self.point_store.advance_shingle(base)?;
        } else {
            self.point_store.validate_full_point(base)?;
        }
        self.entries_seen += 1;

        // Only update the trees once the shingle buffer is primed.
        // With internal shingling the first shingle_size - 1 observations
        // only fill the buffer.
        let shingle_lag = if self.config.internal_shingling() {
            self.config.shingle_size().saturating_sub(1)
        } else {
            0
        };
        if self.entries_seen as usize <= shingle_lag {
            return Ok(());
        }

        let time_decay = self.config.effective_time_decay();
        let initial_frac = self.config.initial_accept_fraction();

        self.update_scratch.clear();

        for t in 0..self.trees.len() {
            let u: f64 = self.rng.random::<f64>();
            let weight = reservoir_weight(u, time_decay, self.entries_seen);

            // Determine initial-phase acceptance probability.
            let fill = self.samplers[t].fill_fraction();
            let is_initial = if self.samplers[t].is_full() {
                false
            } else {
                let prob = if fill < initial_frac {
                    1.0
                } else if initial_frac >= 1.0 {
                    0.0
                } else {
                    1.0 - (fill - initial_frac) / (1.0 - initial_frac)
                };
                self.rng.random::<f64>() < prob
            };

            let result = self.samplers[t].accept(is_initial, weight);
            if result.accepted {
                self.update_scratch.push((t, result.evicted));
            }
        }

        if self.update_scratch.is_empty() {
            self.point_store.record_skipped_add();
            return Ok(());
        }

        // Add point to the shared store only after at least one tree accepts it.
        let point_idx = if self.config.internal_shingling() {
            self.point_store.add_current_shingled()?
        } else {
            self.point_store.add_validated(base)?
        };

        for i in 0..self.update_scratch.len() {
            let (t, evicted) = self.update_scratch[i];

            // Evict old point if necessary.
            if let Some(evicted_idx) = evicted {
                self.trees[t].delete(evicted_idx, &self.point_store)?;
                self.point_store.dec_ref(evicted_idx);
            }

            // Insert new point.
            self.trees[t].insert(point_idx, &self.point_store)?;
            self.point_store.inc_ref(point_idx);
            self.samplers[t].add_point(point_idx);
        }

        Ok(())
    }
}
