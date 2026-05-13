#[cfg(all(not(feature = "std"), feature = "serde"))]
use alloc::string::{String, ToString};
#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

use itertools::Itertools;
use ordered_float::NotNan;
use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    config::RcfConfig,
    error::{RcfError, Result},
    point_store::PointStore,
    sampler::{Sampler, reservoir_weight},
    score::{Attribution, ScoreMode},
    tree::RcfTree,
};

// ---------------------------------------------------------------------------
// Forest
// ---------------------------------------------------------------------------

/// A Random Cut Forest: an ensemble of [`RcfTree`]s sharing a [`PointStore`].
///
/// # Typical usage
/// ```ignore
/// let mut forest = Forest::builder(2, 1).num_trees(50).capacity(256).build()?;
/// for point in stream {
///     forest.update(&point)?;
///     if forest.is_ready() {
///         let score = forest.score(&point)?;
///     }
/// }
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Forest {
    pub(crate) config: RcfConfig,
    trees: Vec<RcfTree>,
    samplers: Vec<Sampler>,
    pub(crate) point_store: PointStore,
    entries_seen: u64,
    rng: Xoshiro256PlusPlus,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NeighborCandidate {
    pub score: f64,
    pub point_idx: usize,
    pub distance: f64,
}

impl From<(f64, usize, f64)> for NeighborCandidate {
    fn from(value: (f64, usize, f64)) -> Self {
        Self {
            score: value.0,
            point_idx: value.1,
            distance: value.2,
        }
    }
}

impl From<NeighborCandidate> for (f64, usize, f64) {
    fn from(value: NeighborCandidate) -> Self {
        (value.score, value.point_idx, value.distance)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NeighborResult {
    pub score: f64,
    pub point: Vec<f32>,
    pub distance: f64,
}

impl From<(f64, Vec<f32>, f64)> for NeighborResult {
    fn from(value: (f64, Vec<f32>, f64)) -> Self {
        Self {
            score: value.0,
            point: value.1,
            distance: value.2,
        }
    }
}

impl From<NeighborResult> for (f64, Vec<f32>, f64) {
    fn from(value: NeighborResult) -> Self {
        (value.score, value.point, value.distance)
    }
}

fn make_missing_flags(missing: &[usize], dim: usize) -> Result<Vec<bool>> {
    let mut missing_flags = vec![false; dim];
    for &i in missing {
        if i >= dim {
            return Err(RcfError::IndexOutOfBounds(i));
        }
        missing_flags[i] = true;
    }
    Ok(missing_flags)
}

fn median_in_place(vals: &mut [f32]) -> f32 {
    debug_assert!(!vals.is_empty(), "median_in_place requires non-empty input");
    let n = vals.len();
    let mid = n / 2;
    vals.select_nth_unstable_by(mid, |a, b| {
        NotNan::new(*a)
            .unwrap_or(NotNan::new(f32::MAX).unwrap())
            .cmp(&NotNan::new(*b).unwrap_or(NotNan::new(f32::MAX).unwrap()))
    });
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
    // Construction
    // -----------------------------------------------------------------------

    fn new_internal(config: RcfConfig, seed: u64) -> Result<Self> {
        if config.input_dim == 0 {
            return Err(RcfError::InvalidArgument("input_dim must be > 0".into()));
        }
        if config.shingle_size == 0 {
            return Err(RcfError::InvalidArgument("shingle_size must be > 0".into()));
        }
        if config.capacity == 0 {
            return Err(RcfError::InvalidArgument("capacity must be > 0".into()));
        }
        if config.num_trees == 0 {
            return Err(RcfError::InvalidArgument("num_trees must be > 0".into()));
        }

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let dim = config.dim();
        let capacity = config.capacity;
        let num_trees = config.num_trees;

        let store_capacity = (capacity * num_trees + 1).max(2 * capacity);

        let trees: Vec<RcfTree> = (0..num_trees)
            .map(|_| RcfTree::new(dim, capacity, rng.next_u64()))
            .collect();

        let samplers: Vec<Sampler> = (0..num_trees).map(|_| Sampler::new(capacity)).collect();

        let point_store = PointStore::new(
            config.input_dim,
            config.shingle_size,
            store_capacity,
            config.internal_shingling,
        );

        Ok(Forest {
            config,
            trees,
            samplers,
            point_store,
            entries_seen: 0,
            rng: Xoshiro256PlusPlus::seed_from_u64(rng.next_u64()),
        })
    }

    /// Create a forest from a [`RcfConfig`] with a random seed.
    pub fn from_config(config: &RcfConfig) -> Result<Self> {
        let mut seed_rng: Xoshiro256PlusPlus = rand::make_rng();
        Self::new_internal(config.clone(), seed_rng.next_u64())
    }

    /// Create a forest from a [`RcfConfig`] with a deterministic seed.
    pub fn from_config_seeded(config: &RcfConfig, seed: u64) -> Result<Self> {
        Self::new_internal(config.clone(), seed)
    }

    // -----------------------------------------------------------------------
    // Builder
    // -----------------------------------------------------------------------

    /// Convenience entry point.
    pub fn builder(input_dim: usize, shingle_size: usize) -> ForestBuilder {
        ForestBuilder::new(input_dim, shingle_size)
    }

    // -----------------------------------------------------------------------
    // Update
    // -----------------------------------------------------------------------

    /// Incorporate a new observation into the forest.
    ///
    /// When `internal_shingling` is true, pass one base observation of length
    /// `input_dim`.  Otherwise pass the full shingled vector of length
    /// `input_dim * shingle_size`.
    pub fn update(&mut self, base: &[f32]) -> Result<()> {
        let shingled = self.point_store.shingled_point(base)?;
        self.entries_seen += 1;

        // Only update the trees once the shingle buffer is primed.
        // With internal shingling the first shingle_size - 1 observations
        // only fill the buffer.
        let shingle_lag = if self.config.internal_shingling {
            self.config.shingle_size.saturating_sub(1)
        } else {
            0
        };
        if self.entries_seen as usize <= shingle_lag {
            return Ok(());
        }

        // Add point to the shared store.
        let point_idx = self.point_store.add(&shingled)?;

        let time_decay = self.config.effective_time_decay();
        let initial_frac = self.config.initial_accept_fraction;

        let mut any_accepted = false;

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

            let result = self.samplers[t].accept(is_initial, weight, point_idx);

            if result.accepted {
                any_accepted = true;

                // Evict old point if necessary.
                if let Some(evicted_idx) = result.evicted {
                    self.trees[t].delete(evicted_idx, &self.point_store)?;
                    self.point_store.dec_ref(evicted_idx);
                }

                // Insert new point.
                self.trees[t].insert(point_idx, &self.point_store)?;
                self.point_store.inc_ref(point_idx);
                self.samplers[t].add_point(point_idx);
            }
        }

        // If no tree accepted, dec ref immediately (point is unused).
        if !any_accepted {
            self.point_store.dec_ref(point_idx);
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Readiness
    // -----------------------------------------------------------------------

    /// Returns `true` once enough observations have been processed that the
    /// scoring functions return meaningful values.
    pub fn is_ready(&self) -> bool {
        let needed = self.config.effective_output_after()
            + if self.config.internal_shingling {
                self.config.shingle_size.saturating_sub(1)
            } else {
                0
            };
        self.entries_seen as usize > needed
    }

    // -----------------------------------------------------------------------
    // Scoring
    // -----------------------------------------------------------------------

    /// Anomaly score for `query`.  Higher → more anomalous.
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

    /// Density estimate at `query`.  Higher → denser neighbourhood.
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
    /// Returns a `Vec<(score, point, distance)>` sorted by distance
    /// (ascending), with duplicates removed.  At most `top_k` results are
    /// returned.
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
        let candidate_idxs = self.collect_conditional_candidate_indices(
            query,
            &missing_flags,
            centrality,
            seed_rng.next_u64(),
        );

        if candidate_idxs.is_empty() {
            return Err(RcfError::NotReady);
        }

        let mut result = query.to_vec();
        self.impute_dimensions_from_candidates(&mut result, missing, &candidate_idxs);

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
        if !self.config.internal_shingling {
            return Err(RcfError::InvalidArgument(
                "extrapolation requires internal_shingling = true".into(),
            ));
        }
        if self.config.shingle_size <= 1 {
            return Err(RcfError::InvalidArgument(
                "extrapolation requires shingle_size > 1".into(),
            ));
        }
        if look_ahead == 0 {
            return Ok(Vec::new());
        }
        let shingle_size = self.config.shingle_size;
        if look_ahead > shingle_size {
            return Err(RcfError::InvalidArgument(format!(
                "extrapolation requires look_ahead <= shingle_size (got {look_ahead}, shingle_size={})",
                shingle_size
            )));
        }

        let input_dim = self.config.input_dim;
        let dim = self.config.dim();
        let mut fictitious = self.point_store.current_shingled().to_vec();
        let mut result = Vec::with_capacity(look_ahead * input_dim);

        let mut rng = self.rng.clone();
        let _ = rng.next_u64();

        for step in 0..look_ahead {
            let missing_indices = self.point_store.next_indices(step);
            let missing_flags = make_missing_flags(&missing_indices, dim)?;
            let seed = rng.next_u64();
            let candidate_idxs =
                self.collect_conditional_candidate_indices(&fictitious, &missing_flags, 1.0, seed);

            if candidate_idxs.is_empty() {
                return Err(RcfError::NotReady);
            }

            for &mi in &missing_indices {
                let median = self.median_for_dimension(&candidate_idxs, mi);
                fictitious[mi] = median;
                result.push(median);
            }
        }

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Save / Load
    // -----------------------------------------------------------------------

    /// Serialise the entire forest state to a JSON string.
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| RcfError::Io(e.to_string()))
    }

    /// Deserialise a forest from a JSON string previously written by [`to_json`].
    #[cfg(feature = "serde")]
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| RcfError::Io(e.to_string()))
    }

    /// Serialise the entire forest state to a JSON file.
    #[cfg(all(feature = "serde", feature = "std"))]
    pub fn save_json(&self, path: impl Into<std::path::PathBuf>) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path.into(), json).map_err(|e| RcfError::Io(e.to_string()))
    }

    /// Deserialise a forest from a JSON file previously written by
    /// [`save_json`].
    #[cfg(all(feature = "serde", feature = "std"))]
    pub fn load_json(path: impl Into<std::path::PathBuf>) -> Result<Self> {
        let data = std::fs::read_to_string(path.into()).map_err(|e| RcfError::Io(e.to_string()))?;
        Self::from_json(&data)
    }

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------

    pub fn entries_seen(&self) -> u64 {
        self.entries_seen
    }

    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    pub fn config(&self) -> &RcfConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Validate and shingle `query`.  Returns the full-dimensional vector.
    fn prepare_query(&self, query: &[f32]) -> Result<Vec<f32>> {
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
        self.trees
            .iter()
            .flat_map(|tree| tree.near_neighbors(query, &self.point_store, mode, percentile))
            .collect()
    }

    fn aggregate_neighbor_candidates(
        &self,
        candidates: Vec<NeighborCandidate>,
        top_k: usize,
    ) -> Vec<NeighborResult> {
        let n = self.trees.len() as f64;
        candidates
            .into_iter()
            .sorted_by_key(|item| item.point_idx)
            .chunk_by(|item| item.point_idx)
            .into_iter()
            .map(|(idx, group)| {
                let (score_sum, dist_min) = group.fold((0.0f64, f64::MAX), |(s, d), item| {
                    (s + item.score, d.min(item.distance))
                });
                NeighborCandidate {
                    score: score_sum / n,
                    point_idx: idx,
                    distance: dist_min,
                }
            })
            .sorted_by_key(|item| {
                NotNan::new(item.distance).unwrap_or(NotNan::new(f64::MAX).unwrap())
            })
            .take(top_k)
            .map(|item| NeighborResult {
                score: item.score,
                point: self.point_store.copy_point(item.point_idx),
                distance: item.distance,
            })
            .collect()
    }

    fn collect_conditional_candidate_indices(
        &self,
        query: &[f32],
        missing_flags: &[bool],
        centrality: f64,
        seed: u64,
    ) -> Vec<usize> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        self.trees
            .iter()
            .filter_map(|tree| {
                let tree_seed = rng.next_u64();
                tree.conditional_field(
                    query,
                    missing_flags,
                    &self.point_store,
                    centrality,
                    tree_seed,
                )
                .map(|c| c.point_idx)
            })
            .collect()
    }

    fn impute_dimensions_from_candidates(
        &self,
        result: &mut [f32],
        missing: &[usize],
        candidate_idxs: &[usize],
    ) {
        for &mi in missing {
            result[mi] = self.median_for_dimension(candidate_idxs, mi);
        }
    }

    fn median_for_dimension(&self, candidate_idxs: &[usize], dim_idx: usize) -> f32 {
        let mut vals: Vec<f32> = candidate_idxs
            .iter()
            .map(|&i| self.point_store.get(i)[dim_idx])
            .collect();
        median_in_place(&mut vals)
    }
}

// ---------------------------------------------------------------------------
// ForestBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for [`Forest`].
pub struct ForestBuilder {
    config: RcfConfig,
    seed: Option<u64>,
}

impl ForestBuilder {
    pub fn new(input_dim: usize, shingle_size: usize) -> Self {
        let config = RcfConfig::new(input_dim).with_shingle_size(shingle_size);
        ForestBuilder { config, seed: None }
    }

    pub fn num_trees(mut self, n: usize) -> Self {
        self.config = self.config.with_num_trees(n);
        self
    }

    pub fn capacity(mut self, c: usize) -> Self {
        self.config = self.config.with_capacity(c);
        self
    }

    pub fn time_decay(mut self, d: f64) -> Self {
        self.config = self.config.with_time_decay(d);
        self
    }

    pub fn output_after(mut self, n: usize) -> Self {
        self.config = self.config.with_output_after(n);
        self
    }

    pub fn internal_shingling(mut self, v: bool) -> Self {
        self.config = self.config.with_internal_shingling(v);
        self
    }

    pub fn initial_accept_fraction(mut self, f: f64) -> Self {
        self.config = self.config.with_initial_accept_fraction(f);
        self
    }

    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    pub fn build(self) -> Result<Forest> {
        match self.seed {
            Some(s) => Forest::from_config_seeded(&self.config, s),
            None => Forest::from_config(&self.config),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use rstest::*;

    use super::*;
    use crate::score::attribution_total;

    fn make_forest() -> Forest {
        Forest::builder(2, 1)
            .num_trees(10)
            .capacity(64)
            .output_after(10)
            .seed(42)
            .build()
            .unwrap()
    }

    #[test]
    fn forest_not_ready_initially() {
        let f = make_forest();
        assert!(!f.is_ready());
    }

    #[test]
    fn forest_ready_after_enough_updates() {
        let mut f = make_forest();
        for i in 0..100 {
            f.update(&[i as f32, 0.5]).unwrap();
        }
        assert!(f.is_ready());
    }

    #[rstest]
    #[case([100.0f32, 100.0])]
    #[case([-50.0f32, -50.0])]
    #[case([0.0f32, 500.0])]
    fn outlier_scores_higher_than_inlier(#[case] outlier: [f32; 2]) {
        let mut f = make_forest();
        // Warm up on a tight cluster.
        for _ in 0..200 {
            f.update(&[0.5f32, 0.5]).unwrap();
        }
        let inlier = f.score(&[0.5f32, 0.5]).unwrap();
        let out = f.score(&outlier).unwrap();
        assert!(
            out > inlier,
            "outlier={out:.4} should be > inlier={inlier:.4}"
        );
    }

    #[test]
    fn score_zero_before_ready() {
        let f = make_forest();
        let s = f.score(&[1.0f32, 2.0]).unwrap();
        assert_abs_diff_eq!(s, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn attribution_sums_close_to_score() {
        let mut f = make_forest();
        for i in 0..200 {
            f.update(&[(i % 5) as f32 * 0.1, 0.5]).unwrap();
        }
        let query = &[5.0f32, 0.5];
        let score = f.score(query).unwrap();
        let attr = f.attribution(query).unwrap();
        let attr_total: f64 = attribution_total(&attr);
        // Attribution total should be ≤ score (leaf contributions are unattributed)
        // and at least 5% of the score (some signal must come from internal nodes).
        let ratio = attr_total / score;
        assert!(
            ratio <= 1.01 && ratio >= 0.05,
            "attr_total={attr_total:.4} score={score:.4} ratio={ratio:.4}"
        );
    }

    #[test]
    #[cfg(all(feature = "serde", feature = "std"))]
    fn save_load_roundtrip() {
        let mut f = make_forest();
        for i in 0..200 {
            f.update(&[i as f32 * 0.01, 0.5]).unwrap();
        }
        let query = &[0.5f32, 0.5];
        let score_before = f.score(query).unwrap();

        let tmpdir = tempfile::tempdir().unwrap();
        let path = tmpdir.path().join("forest.json");
        f.save_json(&path).unwrap();
        let f2 = Forest::load_json(&path).unwrap();
        let score_after = f2.score(query).unwrap();

        assert_abs_diff_eq!(score_before, score_after, epsilon = 1e-10);
    }

    #[test]
    fn shingling_forest_update_and_score() {
        let mut f = Forest::builder(1, 4)
            .num_trees(10)
            .capacity(64)
            .output_after(10)
            .internal_shingling(true)
            .seed(7)
            .build()
            .unwrap();
        for i in 0..200 {
            let v = (i as f32 * 0.1).sin();
            f.update(&[v]).unwrap();
        }
        assert!(f.is_ready());
        let s = f.score(&[0.0f32]).unwrap();
        assert!(s >= 0.0);
    }

    #[test]
    fn extrapolate_returns_expected_length() {
        let mut f = Forest::builder(1, 4)
            .num_trees(10)
            .capacity(64)
            .output_after(10)
            .internal_shingling(true)
            .seed(17)
            .build()
            .unwrap();

        for i in 0..200 {
            let v = (i as f32 * 0.1).sin();
            f.update(&[v]).unwrap();
        }

        let look_ahead = 3;
        let out = f.extrapolate(look_ahead).unwrap();
        assert_eq!(out.len(), look_ahead * f.config().input_dim);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn extrapolate_requires_internal_shingling() {
        let mut f = Forest::builder(1, 4)
            .num_trees(10)
            .capacity(64)
            .output_after(10)
            .internal_shingling(false)
            .seed(19)
            .build()
            .unwrap();

        for i in 0..200 {
            let v = (i as f32 * 0.1).sin();
            f.update(&[v, v, v, v]).unwrap();
        }

        let err = f.extrapolate(1).unwrap_err();
        assert!(
            matches!(err, RcfError::InvalidArgument(msg) if msg.contains("internal_shingling"))
        );
    }

    #[test]
    fn extrapolate_rejects_look_ahead_beyond_shingle_size() {
        let mut f = Forest::builder(1, 4)
            .num_trees(10)
            .capacity(64)
            .output_after(10)
            .internal_shingling(true)
            .seed(23)
            .build()
            .unwrap();

        for i in 0..200 {
            let v = (i as f32 * 0.1).sin();
            f.update(&[v]).unwrap();
        }

        let err = f.extrapolate(5).unwrap_err();
        assert!(matches!(err, RcfError::InvalidArgument(msg) if msg.contains("look_ahead")));
    }

    #[rstest]
    #[case::top1(1)]
    #[case::top5(5)]
    #[case::top7(7)]
    #[case::top15(15)]
    fn near_neighbors_sorted_and_bounded(#[case] top_k: usize) {
        let mut f = make_forest();
        for i in 0..300 {
            let x = (i as f32 * 0.07).sin();
            let y = (i as f32 * 0.11).cos();
            f.update(&[x, y]).unwrap();
        }

        let neighbors = f.near_neighbors(&[0.1, -0.2], top_k, 0).unwrap();
        assert!(neighbors.len() <= top_k);

        for w in neighbors.windows(2) {
            assert!(
                w[0].distance <= w[1].distance,
                "neighbors are not sorted by distance"
            );
        }
    }

    #[rstest]
    #[case::odd_3(vec![7.0f32, 1.0, 5.0], 5.0f32)]
    #[case::even_4(vec![8.0f32, 2.0, 6.0, 4.0], 5.0f32)]
    #[case::single(vec![3.0f32], 3.0f32)]
    #[case::two(vec![2.0f32, 8.0], 5.0f32)]
    fn median_in_place_handles_odd_and_even_lengths(
        #[case] mut data: Vec<f32>,
        #[case] expected: f32,
    ) {
        let m = median_in_place(&mut data);
        assert_abs_diff_eq!(m, expected, epsilon = f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Anomaly-detection simulation
    // -----------------------------------------------------------------------

    /// Build a forest tuned for anomaly simulation: 2-D input, 50 trees,
    /// large capacity so the window never rolls over during the test.
    fn make_anomaly_forest() -> Forest {
        Forest::builder(2, 4)
            .num_trees(50)
            .capacity(512)
            .output_after(50)
            .internal_shingling(true)
            .seed(1234)
            .build()
            .unwrap()
    }

    /// Generate `n` tight 2-D cluster points near (0.5, 0.5).
    ///
    /// Uses per-dimension uniform noise to keep the cluster within ±0.15
    /// of centre and avoid any dependency on a Gaussian sampler.
    /// All values are fully determined by `seed` and `n`.
    fn normal_cluster_points(n: usize, seed: u64) -> Vec<[f32; 2]> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let dx: f32 = rng.random_range(-0.15f32..0.15);
                let dy: f32 = rng.random_range(-0.15f32..0.15);
                [0.5 + dx, 0.5 + dy]
            })
            .collect()
    }

    /// Phase A+B: warm up the forest and return the normal-point score
    /// baseline that subsequent anomaly assertions are relative to.
    fn warm_up_forest(f: &mut Forest) -> f64 {
        for pt in normal_cluster_points(250, 42) {
            f.update(&pt).unwrap();
        }
        assert!(f.is_ready(), "forest must be ready after warm-up");
        f.score(&[0.5f32, 0.5]).unwrap()
    }

    // Phase C: each case is an anomalous query.
    // `dominant_direction`:
    //   0 = attr[dim].below (query > cut_val) should dominate the total
    //   1 = attr[dim].above (query < cut_val) should dominate the total
    // This is determined purely by whether the anomaly is above or below the
    // normal cluster, independent of which specific dimension the tree chose
    // to cut in.
    #[rstest]
    #[case::far_positive([10.0f32, 10.0], 0)] // both dims above cuts → index-0 direction
    #[case::far_negative([-8.0f32, -8.0], 1)] // both dims below cuts → index-1 direction
    #[case::axis_spike([0.5f32, 15.0], 0)] // dim 1 far above cuts → index-0 direction
    fn anomaly_detection_simulation(
        #[case] anomaly: [f32; 2],
        // 0 = below component should dominate; 1 = above component should dominate
        #[case] dominant_direction: usize,
    ) {
        let mut f = make_anomaly_forest();

        // ── Phase B: warm-up & normal baseline ────────────────────────────
        let normal_score = warm_up_forest(&mut f);

        // Normal point's attribution must be within plausible bounds.
        let normal_attr = f.attribution(&[0.5f32, 0.5]).unwrap();
        let normal_attr_total: f64 = attribution_total(&normal_attr);
        let normal_ratio = if normal_score > 0.0 {
            normal_attr_total / normal_score
        } else {
            1.0
        };
        assert!(
            normal_ratio <= 1.01,
            "attribution total {normal_attr_total:.4} exceeds score {normal_score:.4}"
        );

        // ── Phase C: anomaly score must be meaningfully higher ─────────────
        let anomaly_score = f.score(&anomaly).unwrap();
        assert!(
            anomaly_score > normal_score * 2.0,
            "anomaly score {anomaly_score:.4} not > 2× normal {normal_score:.4} \
             for point {anomaly:?}"
        );

        // Displacement score must be positive for a genuine outlier.
        let disp = f.displacement_score(&anomaly).unwrap();
        assert!(
            disp > 0.0,
            "displacement score {disp:.4} should be positive for {anomaly:?}"
        );

        // Attribution: verify the dominant direction for the *current* shingle
        // slot (last `input_dim` dimensions). With shingle_size > 1 the earlier
        // slots may still hold normal-range values and dilute the total.
        let attr = f.attribution(&anomaly).unwrap();
        let input_dim = f.config().input_dim;
        let current_slot = &attr[attr.len() - input_dim..];
        let total_dir0: f64 = current_slot.iter().map(|a| a.below).sum();
        let total_dir1: f64 = current_slot.iter().map(|a| a.above).sum();

        // Keep a small margin to avoid borderline ties from floating-point noise.
        let direction_margin = 1.01;

        if dominant_direction == 0 {
            assert!(
                total_dir0 > total_dir1 * direction_margin,
                "expected 'below' direction to dominate for {anomaly:?}: \
                 dir0={total_dir0:.4} dir1={total_dir1:.4}. attr={attr:?}"
            );
        } else {
            assert!(
                total_dir1 > total_dir0 * direction_margin,
                "expected 'above' direction to dominate for {anomaly:?}: \
                 dir1={total_dir1:.4} dir0={total_dir0:.4}. attr={attr:?}"
            );
        }

        // ── Phase D: near-neighbour sanity ────────────────────────────────
        let anomaly_neighbors = f.near_neighbors(&anomaly, 5, 0).unwrap();
        assert!(
            !anomaly_neighbors.is_empty(),
            "near_neighbors must return at least 1 result for {anomaly:?}"
        );

        // Neighbours must be sorted ascending by distance.
        for w in anomaly_neighbors.windows(2) {
            assert!(
                w[0].distance <= w[1].distance,
                "neighbors not sorted by distance for {anomaly:?}"
            );
        }

        // Anomalous query is far from the cluster: its nearest neighbour
        // distance must exceed the nearest neighbour distance of the cluster
        // centre.
        let normal_neighbors = f.near_neighbors(&[0.5f32, 0.5], 5, 0).unwrap();
        let normal_nn_dist = normal_neighbors.first().map(|r| r.distance).unwrap_or(0.0);
        let anomaly_nn_dist = anomaly_neighbors.first().map(|r| r.distance).unwrap_or(0.0);
        assert!(
            anomaly_nn_dist > normal_nn_dist,
            "anomaly nn-distance {anomaly_nn_dist:.4} should exceed \
             normal nn-distance {normal_nn_dist:.4} for {anomaly:?}"
        );
    }
}
