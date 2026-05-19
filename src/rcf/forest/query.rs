#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap;
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(feature = "std")]
use std::collections::BTreeMap;

use super::{Forest, NeighborCandidate, NeighborResult};
use crate::error::{RcfError, Result};
use crate::rcf::score::{Attribution, ScoreMode};

impl Forest {
    // -----------------------------------------------------------------------
    // Scoring
    // -----------------------------------------------------------------------

    /// Anomaly score for `query`. Higher means more anomalous.
    pub fn score(&self, query: &[f32]) -> Result<f64> {
        let q = self.prepare_query(query)?;
        Ok(self.forest_score(&q, &ScoreMode::standard()))
    }

    /// Displacement-based anomaly score.
    pub fn displacement_score(&self, query: &[f32]) -> Result<f64> {
        let q = self.prepare_query(query)?;
        Ok(self.forest_score(&q, &ScoreMode::displacement()))
    }

    /// Per-dimension attribution of the anomaly score.
    ///
    /// Returns a `Vec<Attribution>` of length `input_dim * shingle_size`.
    pub fn attribution(&self, query: &[f32]) -> Result<Vec<Attribution>> {
        self.attribution_sequential(query)
    }

    fn attribution_sequential(&self, query: &[f32]) -> Result<Vec<Attribution>> {
        let q = self.prepare_query(query)?;
        let dim = self.config.dim();
        let mode = ScoreMode::standard();
        let n = self.trees.len() as f64;
        let total_attr = self
            .trees
            .iter()
            .map(|tree| tree.attribution(&q, &mode))
            .fold(vec![Attribution::default(); dim], |mut acc, tree_attr| {
                for i in 0..dim {
                    acc[i] += tree_attr[i];
                }
                acc
            });
        Ok(total_attr.into_iter().map(|a| a.scale(1.0 / n)).collect())
    }

    // -----------------------------------------------------------------------
    // Density
    // -----------------------------------------------------------------------

    /// Density estimate at `query`. Higher means a denser neighbourhood.
    pub fn density(&self, query: &[f32]) -> Result<f64> {
        self.density_sequential(query)
    }

    fn density_sequential(&self, query: &[f32]) -> Result<f64> {
        let q = self.prepare_query(query)?;
        let raw: f64 = self
            .trees
            .iter()
            .map(|t| t.density(&q, &self.point_store))
            .sum::<f64>()
            / self.trees.len() as f64;
        Ok(raw)
    }

    // -----------------------------------------------------------------------
    // Near-neighbour retrieval
    // -----------------------------------------------------------------------

    /// Find approximate near-neighbours of `query`.
    ///
    /// `percentile` controls per-tree traversal aggressiveness in `[0, 100]`;
    /// lower values visit more branches and usually return more candidates.
    ///
    /// Returns a `Vec<NeighborResult>` sorted by distance (ascending), with
    /// duplicate points across trees merged by point index. At most `top_k`
    /// results are returned.
    pub fn near_neighbors(
        &self,
        query: &[f32],
        top_k: usize,
        percentile: usize,
    ) -> Result<Vec<NeighborResult>> {
        let q = self.prepare_query(query)?;
        let mode = ScoreMode::standard();
        let candidates = self.collect_neighbor_candidates(&q, &mode, percentile);
        Ok(self.aggregate_neighbor_candidates(candidates, top_k))
    }
    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Validate and shingle `query`.  Returns the full-dimensional vector.
    pub(super) fn prepare_query(&self, query: &[f32]) -> Result<Vec<f32>> {
        let base_dim = self.config.input_dim;
        let full_dim = self.config.dim();
        if query.len() == base_dim && self.config.internal_shingling {
            // Caller passed a base observation; apply the current shingle state.
            let mut buf = self.point_store.current_shingled().to_vec();
            let start = full_dim - base_dim;
            buf[start..].copy_from_slice(query);
            Ok(buf)
        } else if query.len() == full_dim {
            Ok(query.to_vec())
        } else {
            Err(RcfError::DimensionMismatch {
                expected: full_dim,
                got: query.len(),
            })
        }
    }

    fn forest_score(&self, query: &[f32], mode: &ScoreMode) -> f64 {
        self.forest_score_sequential(query, mode)
    }

    /// Average score across trees using sequential traversal.
    fn forest_score_sequential(&self, query: &[f32], mode: &ScoreMode) -> f64 {
        if self.trees.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .trees
            .iter()
            .map(|t| t.raw_score(query, &self.point_store, mode))
            .sum();
        sum / self.trees.len() as f64
    }

    fn collect_neighbor_candidates(
        &self,
        query: &[f32],
        mode: &ScoreMode,
        percentile: usize,
    ) -> Vec<NeighborCandidate> {
        self.collect_neighbor_candidates_sequential(query, mode, percentile)
    }

    /// Collect neighbor candidates by traversing trees sequentially.
    fn collect_neighbor_candidates_sequential(
        &self,
        query: &[f32],
        mode: &ScoreMode,
        percentile: usize,
    ) -> Vec<NeighborCandidate> {
        // Reuse one output buffer across trees to avoid per-tree temporary Vec allocations.
        let mut candidates = Vec::with_capacity(self.trees.len() * 2);
        for tree in &self.trees {
            tree.near_neighbors_into(query, &self.point_store, mode, percentile, &mut candidates);
        }
        candidates
    }

    pub(super) fn aggregate_neighbor_candidates(
        &self,
        candidates: Vec<NeighborCandidate>,
        top_k: usize,
    ) -> Vec<NeighborResult> {
        if candidates.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let n = self.trees.len() as f64;
        let mut merged: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
        for item in candidates {
            let entry = merged.entry(item.point_idx).or_insert((0.0, f64::MAX));
            entry.0 += item.score;
            entry.1 = entry.1.min(item.distance);
        }

        let mut aggregated: Vec<NeighborCandidate> = merged
            .into_iter()
            .map(|(point_idx, (score_sum, dist_min))| NeighborCandidate {
                score: score_sum / n,
                point_idx,
                distance: dist_min,
            })
            .collect();

        let limit = top_k.min(aggregated.len());
        if limit < aggregated.len() {
            // Partition once around kth element, then sort only the kept prefix.
            aggregated
                .select_nth_unstable_by(limit - 1, |a, b| cmp_distance(a.distance, b.distance));
            aggregated.truncate(limit);
        }
        aggregated.sort_unstable_by(|a, b| cmp_distance(a.distance, b.distance));

        aggregated
            .into_iter()
            .map(|item| NeighborResult {
                score: item.score,
                point: self.point_store.copy_point(item.point_idx),
                distance: item.distance,
            })
            .collect()
    }
}

fn cmp_distance(a: f64, b: f64) -> core::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => core::cmp::Ordering::Equal,
        (true, false) => core::cmp::Ordering::Greater,
        (false, true) => core::cmp::Ordering::Less,
        (false, false) => a.total_cmp(&b),
    }
}
