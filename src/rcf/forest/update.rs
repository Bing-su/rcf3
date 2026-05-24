use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;

use super::{AcceptedUpdate, Forest};
use crate::error::{RcfError, Result};
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
        self.validate_update_input(base)?;

        let next_entries_seen = self.entries_seen + 1;
        let mut rng = self.rng.clone();
        let has_primed_shingle = self.has_primed_shingle_at(next_entries_seen);
        self.collect_accepted_updates_with(&mut rng, next_entries_seen, has_primed_shingle);

        if !self.staged_accepted_updates.is_empty() {
            self.validate_staged_updates()?;
            self.point_store.ensure_can_allocate_slot()?;
        }

        self.prepare_update_input(base)?;
        self.entries_seen = next_entries_seen;
        self.rng = rng;

        core::mem::swap(
            &mut self.accepted_updates,
            &mut self.staged_accepted_updates,
        );

        if !has_primed_shingle {
            self.point_store.record_logical_add_without_storage();
            return Ok(());
        }

        if self.accepted_updates.is_empty() {
            self.point_store.record_logical_add_without_storage();
            return Ok(());
        }

        let point_idx = self.store_update_point(base)?;
        self.apply_accepted_updates(point_idx)
    }

    /// Check that all staged evictions can be applied before committing the update.
    fn validate_staged_updates(&self) -> Result<()> {
        for update in &self.staged_accepted_updates {
            if let Some(evicted_idx) = update.evicted_point {
                self.trees[update.tree_index].validate_delete(evicted_idx, &self.point_store)?;
            }
        }
        Ok(())
    }

    /// Validate the update input before sampling decisions consume RNG state.
    fn validate_update_input(&self, base: &[f32]) -> Result<()> {
        if self.config.internal_shingling() {
            if base.len() != self.config.input_dim() {
                return Err(RcfError::DimensionMismatch {
                    expected: self.config.input_dim(),
                    got: base.len(),
                });
            }
            Ok(())
        } else {
            self.point_store.validate_full_point(base)
        }
    }

    /// Commit the input into the rolling shingle buffer after preflight checks pass.
    fn prepare_update_input(&mut self, base: &[f32]) -> Result<()> {
        if self.config.internal_shingling() {
            self.point_store.advance_shingle(base)?;
        }
        Ok(())
    }

    /// Return whether the logical update count has filled the shingle window.
    fn has_primed_shingle_at(&self, entries_seen: u64) -> bool {
        let shingle_lag = if self.config.internal_shingling() {
            self.config.shingle_size().saturating_sub(1)
        } else {
            0
        };

        entries_seen as usize > shingle_lag
    }

    /// Stage sampler decisions against cloned RNG state before mutating the forest.
    fn collect_accepted_updates_with(
        &mut self,
        rng: &mut Xoshiro256PlusPlus,
        entries_seen: u64,
        has_primed_shingle: bool,
    ) {
        self.staged_accepted_updates.clear();
        if !has_primed_shingle {
            return;
        }

        let time_decay = self.config.effective_time_decay();
        let initial_frac = self.config.initial_accept_fraction();

        for t in 0..self.trees.len() {
            let u: f64 = rng.random::<f64>();
            let weight = reservoir_weight(u, time_decay, entries_seen);

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
                rng.random::<f64>() < prob
            };

            let result = self.samplers[t].accept(is_initial, weight);
            if result.accepted {
                self.staged_accepted_updates.push(AcceptedUpdate {
                    tree_index: t,
                    evicted_point: result.evicted,
                    weight,
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
        let mut new_point_refs = 0usize;

        for i in 0..self.accepted_updates.len() {
            let update = self.accepted_updates[i];
            let t = update.tree_index;

            // Evict old point if necessary.
            if let Some(evicted_idx) = update.evicted_point {
                self.trees[t].delete(evicted_idx, &self.point_store)?;
                self.point_store.dec_ref(evicted_idx);
            }

            // Insert new point.
            let tree_point_idx = self.trees[t].insert(point_idx, &self.point_store)?;
            if tree_point_idx == point_idx {
                new_point_refs += 1;
            }
            self.point_store.inc_ref(tree_point_idx);
            self.samplers[t].add_point(tree_point_idx, update.weight);
        }

        if new_point_refs == 0 {
            self.point_store.dec_ref(point_idx);
        }

        Ok(())
    }
}
