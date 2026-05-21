use rand::prelude::*;

use super::{AcceptedUpdate, Forest};
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
        self.prepare_update_input(base)?;
        self.entries_seen += 1;

        if !self.has_primed_shingle() {
            return Ok(());
        }

        self.collect_accepted_updates();

        if self.update_scratch.is_empty() {
            self.point_store.record_logical_add_without_storage();
            return Ok(());
        }

        let point_idx = self.store_update_point(base)?;
        self.apply_accepted_updates(point_idx)
    }

    fn prepare_update_input(&mut self, base: &[f32]) -> Result<()> {
        if self.config.internal_shingling() {
            self.point_store.advance_shingle(base)?;
        } else {
            self.point_store.validate_full_point(base)?;
        }
        Ok(())
    }

    fn has_primed_shingle(&self) -> bool {
        let shingle_lag = if self.config.internal_shingling() {
            self.config.shingle_size().saturating_sub(1)
        } else {
            0
        };

        self.entries_seen as usize > shingle_lag
    }

    fn collect_accepted_updates(&mut self) {
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
                self.update_scratch.push(AcceptedUpdate {
                    tree_index: t,
                    evicted_point: result.evicted,
                });
            }
        }
    }

    fn store_update_point(&mut self, base: &[f32]) -> Result<usize> {
        if self.config.internal_shingling() {
            self.point_store.add_current_shingled()
        } else {
            self.point_store.add_validated(base)
        }
    }

    fn apply_accepted_updates(&mut self, point_idx: usize) -> Result<()> {
        for i in 0..self.update_scratch.len() {
            let update = self.update_scratch[i];
            let t = update.tree_index;

            // Evict old point if necessary.
            if let Some(evicted_idx) = update.evicted_point {
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
