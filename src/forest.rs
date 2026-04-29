use itertools::Itertools;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Forest {
    pub(crate) config: RcfConfig,
    trees: Vec<RcfTree>,
    samplers: Vec<Sampler>,
    pub(crate) point_store: PointStore,
    entries_seen: u64,
    rng_seed: u64,
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

        let mut rng = StdRng::seed_from_u64(seed);
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
            rng_seed: rng.next_u64(),
        })
    }

    /// Create a forest from a [`RcfConfig`] with a random seed.
    pub fn from_config(config: &RcfConfig) -> Result<Self> {
        let mut seed_rng = rand::make_rng::<StdRng>();
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

        let mut rng = StdRng::seed_from_u64(self.rng_seed);
        self.rng_seed = rng.next_u64();

        let mut any_accepted = false;

        for t in 0..self.trees.len() {
            let u: f64 = rng.random::<f64>();
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
                rng.random::<f64>() < prob
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
    /// Returns a `Vec<[f64; 2]>` of length `input_dim * shingle_size` where
    /// `[below, above]` are the contributions from cuts below and above the
    /// query value in each dimension.
    pub fn attribution(&self, query: &[f32]) -> Result<Attribution> {
        let q = self.prepare_query(query)?;
        let dim = self.config.dim();
        let mode = ScoreMode::standard();
        let n = self.trees.len() as f64;
        let total_attr = self
            .trees
            .par_iter()
            .map(|tree| tree.attribution(&q, &self.point_store, &mode))
            .reduce(
                || vec![[0.0f64; 2]; dim],
                |mut acc, tree_attr| {
                    for i in 0..dim {
                        acc[i][0] += tree_attr[i][0];
                        acc[i][1] += tree_attr[i][1];
                    }
                    acc
                },
            );
        Ok(total_attr
            .into_iter()
            .map(|[a, b]| [a / n, b / n])
            .collect())
    }

    // -----------------------------------------------------------------------
    // Density
    // -----------------------------------------------------------------------

    /// Density estimate at `query`.  Higher → denser neighbourhood.
    pub fn density(&self, query: &[f32]) -> Result<f64> {
        let q = self.prepare_query(query)?;
        let raw: f64 = self
            .trees
            .par_iter()
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
    ) -> Result<Vec<(f64, Vec<f32>, f64)>> {
        let q = self.prepare_query(query)?;
        let mode = ScoreMode::standard();
        let n = self.trees.len() as f64;

        // Collect all candidates, deduplicate by idx (sum scores, min dist), sort by distance.
        let results: Vec<(f64, Vec<f32>, f64)> = self
            .trees
            .par_iter()
            .flat_map(|tree| tree.near_neighbors(&q, &self.point_store, &mode, percentile))
            .collect::<Vec<_>>()
            .into_iter()
            .sorted_by_key(|(_, idx, _)| *idx)
            .chunk_by(|(_, idx, _)| *idx)
            .into_iter()
            .map(|(idx, group)| {
                let (score_sum, dist_min) = group
                    .fold((0.0f64, f64::MAX), |(s, d), (score, _, dist)| {
                        (s + score, d.min(dist))
                    });
                (score_sum / n, self.point_store.copy_point(idx), dist_min)
            })
            .sorted_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
            .take(top_k)
            .collect();

        Ok(results)
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

        let mut missing_flags = vec![false; dim];
        for &i in missing {
            if i >= dim {
                return Err(RcfError::IndexOutOfBounds(i));
            }
            missing_flags[i] = true;
        }

        let mut candidates: Vec<Vec<f32>> = Vec::new();
        let mut rng = StdRng::seed_from_u64(self.rng_seed ^ 0xdead_beef);

        for tree in &self.trees {
            let seed = rng.next_u64();
            if let Some((_, idx, _)) =
                tree.conditional_field(query, &missing_flags, &self.point_store, centrality, seed)
            {
                candidates.push(self.point_store.copy_point(idx));
            }
        }

        if candidates.is_empty() {
            return Err(RcfError::NotReady);
        }

        // Component-wise median over the candidate points.
        let mut result = query.to_vec();
        for &mi in missing {
            let mut vals: Vec<f32> = candidates.iter().map(|c| c[mi]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = if vals.len() % 2 == 1 {
                vals[vals.len() / 2]
            } else {
                (vals[vals.len() / 2 - 1] + vals[vals.len() / 2]) / 2.0
            };
            result[mi] = median;
        }

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Extrapolation
    // -----------------------------------------------------------------------

    /// Predict the next `look_ahead` base observations beyond the current
    /// shingle buffer.
    ///
    /// Requires `internal_shingling = true` and `shingle_size > 1`.
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

        let input_dim = self.config.input_dim;
        let dim = self.config.dim();
        let mut fictitious = self.point_store.current_shingled().to_vec();
        let mut result = Vec::with_capacity(look_ahead * input_dim);

        let mut rng = StdRng::seed_from_u64(self.rng_seed ^ 0xcafe_babe);

        for step in 0..look_ahead {
            let missing_indices = self.point_store.next_indices(step);
            let mut missing_flags = vec![false; dim];
            for &i in &missing_indices {
                missing_flags[i] = true;
            }

            let mut candidates: Vec<Vec<f32>> = Vec::new();
            for tree in &self.trees {
                let seed = rng.next_u64();
                if let Some((_, idx, _)) = tree.conditional_field(
                    &fictitious,
                    &missing_flags,
                    &self.point_store,
                    1.0,
                    seed,
                ) {
                    candidates.push(self.point_store.copy_point(idx));
                }
            }

            if candidates.is_empty() {
                return Err(RcfError::NotReady);
            }

            // Median per missing dimension.
            for &mi in &missing_indices {
                let mut vals: Vec<f32> = candidates.iter().map(|c| c[mi]).collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let med = if vals.len() % 2 == 1 {
                    vals[vals.len() / 2]
                } else {
                    (vals[vals.len() / 2 - 1] + vals[vals.len() / 2]) / 2.0
                };
                fictitious[mi] = med;
                result.push(med);
            }
        }

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Save / Load
    // -----------------------------------------------------------------------

    /// Serialise the entire forest state to a JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| RcfError::Io(e.to_string()))
    }

    /// Deserialise a forest from a JSON string previously written by [`to_json`].
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| RcfError::Io(e.to_string()))
    }

    /// Serialise the entire forest state to a JSON file.
    pub fn save_json(&self, path: &str) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path, json).map_err(|e| RcfError::Io(e.to_string()))
    }

    /// Deserialise a forest from a JSON file previously written by
    /// [`save_json`].
    pub fn load_json(path: &str) -> Result<Self> {
        let data = std::fs::read_to_string(path).map_err(|e| RcfError::Io(e.to_string()))?;
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
        if self.trees.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .trees
            .par_iter()
            .map(|t| t.raw_score(query, &self.point_store, mode))
            .sum();
        sum / self.trees.len() as f64
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
    use super::*;
    use rstest::*;

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
        assert_eq!(s, 0.0);
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
        let attr_total: f64 = attr.iter().map(|[a, b]| a + b).sum();
        // Attribution total should be ≤ score (leaf contributions are unattributed)
        // and at least 5% of the score (some signal must come from internal nodes).
        let ratio = attr_total / score;
        assert!(
            ratio <= 1.01 && ratio >= 0.05,
            "attr_total={attr_total:.4} score={score:.4} ratio={ratio:.4}"
        );
    }

    #[test]
    fn save_load_roundtrip() {
        let mut f = make_forest();
        for i in 0..200 {
            f.update(&[i as f32 * 0.01, 0.5]).unwrap();
        }
        let query = &[0.5f32, 0.5];
        let score_before = f.score(query).unwrap();

        let path = "/tmp/arcf_test_forest.json";
        f.save_json(path).unwrap();
        let f2 = Forest::load_json(path).unwrap();
        let score_after = f2.score(query).unwrap();

        assert!(
            (score_before - score_after).abs() < 1e-10,
            "score changed after round-trip: {score_before} vs {score_after}"
        );
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
}
