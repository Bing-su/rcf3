#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;

use super::Forest;
use crate::error::{RcfError, Result};

pub(super) fn make_missing_flags(missing: &[usize], dim: usize) -> Result<Vec<bool>> {
    let mut missing_flags = vec![false; dim];
    fill_missing_flags(missing, &mut missing_flags)?;
    Ok(missing_flags)
}

fn fill_missing_flags(missing: &[usize], missing_flags: &mut [bool]) -> Result<()> {
    missing_flags.fill(false);
    for &i in missing {
        if i >= missing_flags.len() {
            return Err(RcfError::IndexOutOfBounds(i));
        }
        missing_flags[i] = true;
    }
    Ok(())
}

pub(super) fn median_in_place(vals: &mut [f32]) -> f32 {
    debug_assert!(!vals.is_empty(), "median_in_place requires non-empty input");
    let n = vals.len();
    let mid = n / 2;
    vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    if n % 2 == 1 {
        vals[mid]
    } else {
        let lo = vals[..mid]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        (lo + vals[mid]) / 2.0
    }
}

impl Forest {
    // -----------------------------------------------------------------------
    // Imputation
    // -----------------------------------------------------------------------

    /// Impute the `missing` positions of `query`.
    ///
    /// `query` must have the full dimensionality (`input_dim * shingle_size`).
    /// Values at `missing` indices are ignored; the returned vector fills them
    /// with the median of the nearest-neighbour estimates across all trees.
    ///
    /// When `centrality` = 1.0 the nearest neighbour in each tree is selected
    /// deterministically; lower values introduce randomness.
    pub fn impute(&self, query: &[f32], missing: &[usize], centrality: f64) -> Result<Vec<f32>> {
        if missing.is_empty() {
            return Err(RcfError::InvalidArgument("missing list is empty".into()));
        }
        let dim = self.config.dim();
        if query.len() != dim {
            return Err(RcfError::DimensionMismatch {
                expected: dim,
                got: query.len(),
            });
        }

        let missing_flags = make_missing_flags(missing, dim)?;
        let mut seed_rng = self.rng.clone();
        let mut candidate_idxs = Vec::with_capacity(self.trees.len());
        self.collect_conditional_candidate_indices_into(
            query,
            &missing_flags,
            centrality,
            seed_rng.next_u64(),
            &mut candidate_idxs,
        );

        if candidate_idxs.is_empty() {
            return Err(RcfError::NotReady);
        }

        let mut result = query.to_vec();
        let mut values = Vec::with_capacity(candidate_idxs.len());
        self.impute_dimensions_from_candidates(&mut result, missing, &candidate_idxs, &mut values);

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Extrapolation
    // -----------------------------------------------------------------------

    /// Predict the next `look_ahead` base observations beyond the current
    /// shingle buffer.
    ///
    /// Requires `internal_shingling = true`, `shingle_size > 1`,
    /// and `look_ahead <= shingle_size`.
    /// Returns a vector of length `look_ahead * input_dim`.
    pub fn extrapolate(&self, look_ahead: usize) -> Result<Vec<f32>> {
        if !self.config.internal_shingling() {
            return Err(RcfError::InvalidArgument(
                "extrapolation requires internal_shingling = true".into(),
            ));
        }
        if self.config.shingle_size() <= 1 {
            return Err(RcfError::InvalidArgument(
                "extrapolation requires shingle_size > 1".into(),
            ));
        }
        if look_ahead == 0 {
            return Ok(Vec::new());
        }
        let shingle_size = self.config.shingle_size();
        if look_ahead > shingle_size {
            return Err(RcfError::InvalidArgument(format!(
                "extrapolation requires look_ahead <= shingle_size (got {look_ahead}, shingle_size={})",
                shingle_size
            )));
        }

        let input_dim = self.config.input_dim();
        let dim = self.config.dim();
        let mut fictitious = self.point_store.current_shingled().to_vec();
        let mut result = Vec::with_capacity(look_ahead * input_dim);
        let mut missing_indices = Vec::with_capacity(input_dim);
        let mut missing_flags = vec![false; dim];
        let mut candidate_idxs = Vec::with_capacity(self.trees.len());
        let mut values = Vec::with_capacity(self.trees.len());

        let mut rng = self.rng.clone();
        let _ = rng.next_u64();

        for step in 0..look_ahead {
            self.point_store
                .next_indices_into(step, &mut missing_indices);
            fill_missing_flags(&missing_indices, &mut missing_flags)?;
            let seed = rng.next_u64();
            self.collect_conditional_candidate_indices_into(
                &fictitious,
                &missing_flags,
                1.0,
                seed,
                &mut candidate_idxs,
            );

            if candidate_idxs.is_empty() {
                return Err(RcfError::NotReady);
            }

            for &mi in &missing_indices {
                let median = self.median_for_dimension_into(&candidate_idxs, mi, &mut values);
                fictitious[mi] = median;
                result.push(median);
            }
        }

        Ok(result)
    }
    fn collect_conditional_candidate_indices_into(
        &self,
        query: &[f32],
        missing_flags: &[bool],
        centrality: f64,
        seed: u64,
        output: &mut Vec<usize>,
    ) {
        output.clear();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        output.extend(self.trees.iter().filter_map(|tree| {
            let tree_seed = rng.next_u64();
            tree.conditional_field(
                query,
                missing_flags,
                &self.point_store,
                centrality,
                tree_seed,
            )
            .map(|c| c.point_idx)
        }));
    }

    fn impute_dimensions_from_candidates(
        &self,
        result: &mut [f32],
        missing: &[usize],
        candidate_idxs: &[usize],
        values: &mut Vec<f32>,
    ) {
        for &mi in missing {
            result[mi] = self.median_for_dimension_into(candidate_idxs, mi, values);
        }
    }

    fn median_for_dimension_into(
        &self,
        candidate_idxs: &[usize],
        dim_idx: usize,
        values: &mut Vec<f32>,
    ) -> f32 {
        values.clear();
        values.extend(
            candidate_idxs
                .iter()
                .map(|&i| self.point_store.get(i)[dim_idx]),
        );
        median_in_place(values)
    }
}
