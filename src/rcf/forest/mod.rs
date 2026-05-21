#[cfg(all(not(feature = "std"), feature = "serde"))]
use alloc::string::{String, ToString};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{config::RcfConfig, point_store::PointStore, sampler::Sampler, tree::RcfTree};
#[cfg(any(feature = "serde", test))]
use crate::error::RcfError;
use crate::error::Result;

mod impute;
mod query;
mod update;

/// Intermediate candidate collected from a single tree during near-neighbour search.
///
/// Gathered per-tree inside [`Forest::near_neighbors`] and then aggregated
/// and converted into [`NeighborResult`].
#[derive(Clone, Debug, PartialEq)]
pub(in crate::rcf) struct NeighborCandidate {
    /// Anomaly score of this candidate point.
    pub(in crate::rcf) score: f64,
    /// Index of the point in the [`PointStore`].
    pub(in crate::rcf) point_idx: usize,
    /// L1 distance to the query point.
    pub(in crate::rcf) distance: f64,
}

/// Per-tree update decision kept between sampler acceptance and point storage.
#[derive(Clone, Copy, Debug)]
pub(super) struct AcceptedUpdate {
    pub(super) tree_index: usize,
    pub(super) evicted_point: Option<usize>,
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

/// A near-neighbour result returned by [`Forest::near_neighbors`].
///
/// Candidates collected across all trees are deduplicated and
/// aggregated by point index, then returned sorted by distance (ascending).
#[derive(Clone, Debug, PartialEq)]
pub struct NeighborResult {
    /// Anomaly score of this point.
    pub score: f64,
    /// Coordinate vector of the point (length = `input_dim * shingle_size`).
    pub point: Vec<f32>,
    /// L1 distance to the query point.
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

// ---------------------------------------------------------------------------
// Forest
// ---------------------------------------------------------------------------

/// A Random Cut Forest: an ensemble of random-cut trees sharing point storage.
///
/// # Typical usage
/// ```ignore
/// let mut forest = Forest::builder(2)
///     .shingle_size(1)
///     .num_trees(50)
///     .capacity(256)
///     .build()?;
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
    config: RcfConfig,
    trees: Vec<RcfTree>,
    samplers: Vec<Sampler>,
    point_store: PointStore,
    entries_seen: u64,
    rng: Xoshiro256PlusPlus,
    #[cfg_attr(feature = "serde", serde(skip, default))]
    update_scratch: Vec<AcceptedUpdate>,
}

impl Forest {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    fn new_internal(config: RcfConfig, seed: u64) -> Result<Self> {
        config.validate()?;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let dim = config.dim();
        let capacity = config.capacity();
        let num_trees = config.num_trees();

        let store_capacity = (capacity * num_trees + 1).max(2 * capacity);

        let trees: Vec<RcfTree> = (0..num_trees)
            .map(|_| RcfTree::new(dim, capacity, rng.next_u64()))
            .collect();

        let samplers: Vec<Sampler> = (0..num_trees).map(|_| Sampler::new(capacity)).collect();

        let point_store = PointStore::new(
            config.input_dim(),
            config.shingle_size(),
            store_capacity,
            config.internal_shingling(),
        );

        Ok(Forest {
            config,
            trees,
            samplers,
            point_store,
            entries_seen: 0,
            rng: Xoshiro256PlusPlus::seed_from_u64(rng.next_u64()),
            update_scratch: Vec::with_capacity(num_trees),
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
    pub fn builder(input_dim: usize) -> ForestBuilder {
        ForestBuilder::new(input_dim)
    }
    // -----------------------------------------------------------------------
    // Readiness
    // -----------------------------------------------------------------------

    /// Returns `true` once enough observations have been processed that the
    /// scoring functions return meaningful values.
    pub fn is_ready(&self) -> bool {
        let needed = self.config.effective_output_after()
            + if self.config.internal_shingling() {
                self.config.shingle_size().saturating_sub(1)
            } else {
                0
            };
        self.entries_seen as usize > needed
    }
    // -----------------------------------------------------------------------
    // Save / Load
    // -----------------------------------------------------------------------

    /// Serialize the entire forest state to a JSON string.
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| RcfError::Io(e.to_string()))
    }

    /// Deserialize a forest from a JSON string previously written by
    /// [`Self::to_json`].
    #[cfg(feature = "serde")]
    pub fn from_json(json: impl AsRef<[u8]>) -> Result<Self> {
        serde_json::from_slice(json.as_ref()).map_err(|e| RcfError::Io(e.to_string()))
    }

    /// Serialize the entire forest state to a JSON file.
    #[cfg(all(feature = "serde", feature = "std"))]
    pub fn save_json(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path.as_ref(), json).map_err(|e| RcfError::Io(e.to_string()))
    }

    /// Deserialize a forest from a JSON file previously written by
    /// [`Self::save_json`].
    #[cfg(all(feature = "serde", feature = "std"))]
    pub fn load_json(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let data = std::fs::read(path.as_ref()).map_err(|e| RcfError::Io(e.to_string()))?;
        Self::from_json(&data)
    }

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------

    /// Number of observations processed so far.
    pub fn entries_seen(&self) -> u64 {
        self.entries_seen
    }

    /// Number of trees in the ensemble.
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    pub fn config(&self) -> &RcfConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// ForestBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing a [`Forest`] with custom hyperparameters.
pub struct ForestBuilder {
    config: RcfConfig,
    seed: Option<u64>,
}

impl ForestBuilder {
    /// Create a builder for the given base dimension.
    pub fn new(input_dim: usize) -> Self {
        let config = RcfConfig::new(input_dim);
        ForestBuilder { config, seed: None }
    }

    /// Set the shingle size.
    pub fn shingle_size(mut self, n: usize) -> Self {
        self.config = self.config.with_shingle_size(n);
        self
    }

    /// Set the number of trees in the ensemble.
    pub fn num_trees(mut self, n: usize) -> Self {
        self.config = self.config.with_num_trees(n);
        self
    }

    /// Set the maximum number of points retained per tree.
    pub fn capacity(mut self, c: usize) -> Self {
        self.config = self.config.with_capacity(c);
        self
    }

    /// Set the exponential time-decay rate for sampling weights.
    pub fn time_decay(mut self, d: f64) -> Self {
        self.config = self.config.with_time_decay(d);
        self
    }

    /// Set the minimum updates before non-trivial scores are returned.
    pub fn output_after(mut self, n: usize) -> Self {
        self.config = self.config.with_output_after(n);
        self
    }

    /// Enable or disable internal shingle buffer management.
    pub fn internal_shingling(mut self, v: bool) -> Self {
        self.config = self.config.with_internal_shingling(v);
        self
    }

    /// Set the warm-up acceptance fraction for the sampler.
    pub fn initial_accept_fraction(mut self, f: f64) -> Self {
        self.config = self.config.with_initial_accept_fraction(f);
        self
    }

    /// Set a deterministic seed for all trees in the forest.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// Build a forest from the accumulated configuration.
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
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use approx::assert_abs_diff_eq;
    use rstest::*;

    use super::impute::{make_missing_flags, median_in_place};
    use super::*;
    use crate::rcf::score::attribution_total;

    fn make_forest() -> Forest {
        Forest::builder(2)
            .shingle_size(1)
            .num_trees(10)
            .capacity(64)
            .output_after(10)
            .seed(42)
            .build()
            .unwrap()
    }

    #[test]
    fn builder_uses_default_shingle_size() {
        let f = Forest::builder(2).build().unwrap();
        assert_eq!(f.config().shingle_size(), 1);
    }

    #[test]
    fn builder_applies_explicit_shingle_size() {
        let f = Forest::builder(2).shingle_size(4).build().unwrap();
        assert_eq!(f.config().shingle_size(), 4);
    }

    #[test]
    fn prepare_query_uses_current_shingle_for_base_observation() {
        let mut f = Forest::builder(1)
            .shingle_size(3)
            .internal_shingling(true)
            .seed(11)
            .build()
            .unwrap();
        f.update(&[1.0]).unwrap();
        f.update(&[2.0]).unwrap();

        let prepared = f.prepare_query(&[9.0]).unwrap();
        assert_eq!(prepared.as_ref(), &[0.0, 1.0, 9.0]);
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
    fn duplicate_updates_share_canonical_point_storage() {
        let mut f = Forest::builder(2)
            .shingle_size(1)
            .num_trees(1)
            .capacity(8)
            .output_after(0)
            .initial_accept_fraction(1.0)
            .seed(42)
            .build()
            .unwrap();

        for _ in 0..8 {
            f.update(&[1.0f32, 2.0]).unwrap();
        }

        assert_eq!(f.point_store.num_points(), 1);
        assert_eq!(f.samplers[0].points(), &[0; 8]);
        assert_eq!(f.point_store.ref_count(0), f.samplers[0].points().len());
        assert!(f.score(&[1.0f32, 2.0]).unwrap().is_finite());
    }

    #[test]
    fn internal_shingling_priming_counts_logical_point_store_updates() {
        let mut f = Forest::builder(1)
            .shingle_size(4)
            .num_trees(1)
            .capacity(8)
            .output_after(0)
            .initial_accept_fraction(1.0)
            .seed(42)
            .build()
            .unwrap();

        for i in 0..3 {
            f.update(&[i as f32]).unwrap();
        }

        assert_eq!(f.entries_seen(), 3);
        assert_eq!(f.point_store.entries_seen(), 3);
        assert_eq!(f.point_store.num_points(), 0);

        f.update(&[3.0]).unwrap();

        assert_eq!(f.entries_seen(), 4);
        assert_eq!(f.point_store.entries_seen(), 4);
        assert_eq!(f.point_store.num_points(), 1);
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
            (0.05..=1.01).contains(&ratio),
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
    #[cfg(all(feature = "serde", feature = "std"))]
    fn serde_omits_internal_scratch_buffers() {
        let mut f = make_forest();
        for i in 0..100 {
            f.update(&[i as f32 * 0.01, 0.5]).unwrap();
        }

        let json = f.to_json().unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert!(value.get("update_scratch").is_none());
        for tree in value["trees"].as_array().unwrap() {
            assert!(tree.get("path_scratch").is_none());
        }
    }

    #[test]
    fn shingling_forest_update_and_score() {
        let mut f = Forest::builder(1)
            .shingle_size(4)
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
        let mut f = Forest::builder(1)
            .shingle_size(4)
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
        assert_eq!(out.len(), look_ahead * f.config().input_dim());
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn extrapolate_requires_internal_shingling() {
        let mut f = Forest::builder(1)
            .shingle_size(4)
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
        let mut f = Forest::builder(1)
            .shingle_size(4)
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

    #[test]
    fn missing_flags_reject_out_of_bounds_indices() {
        let err = make_missing_flags(&[0, 2], 2).unwrap_err();
        assert!(matches!(err, RcfError::IndexOutOfBounds(2)));
    }

    #[test]
    fn aggregate_neighbor_candidates_merges_duplicates_and_sorts_by_distance() {
        let mut f = make_forest();
        let idx_a = f.point_store.add(&[1.0, 1.0]).unwrap();
        let idx_b = f.point_store.add(&[2.0, 2.0]).unwrap();

        let aggregated = f.aggregate_neighbor_candidates(
            vec![
                NeighborCandidate {
                    score: 4.0,
                    point_idx: idx_a,
                    distance: 3.0,
                },
                NeighborCandidate {
                    score: 6.0,
                    point_idx: idx_a,
                    distance: 1.0,
                },
                NeighborCandidate {
                    score: 8.0,
                    point_idx: idx_b,
                    distance: 0.5,
                },
            ],
            2,
        );

        assert_eq!(aggregated.len(), 2);
        assert_eq!(aggregated[0].point, vec![2.0, 2.0]);
        assert_abs_diff_eq!(aggregated[0].score, 0.8, epsilon = 1e-12);
        assert_abs_diff_eq!(aggregated[0].distance, 0.5, epsilon = 1e-12);
        assert_eq!(aggregated[1].point, vec![1.0, 1.0]);
        assert_abs_diff_eq!(aggregated[1].score, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(aggregated[1].distance, 1.0, epsilon = 1e-12);
    }

    // -----------------------------------------------------------------------
    // Anomaly-detection simulation
    // -----------------------------------------------------------------------

    /// Build a forest tuned for anomaly simulation: 2-D input, 50 trees,
    /// large capacity so the window never rolls over during the test.
    fn make_anomaly_forest() -> Forest {
        Forest::builder(2)
            .shingle_size(4)
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
        let input_dim = f.config().input_dim();
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

    #[cfg(feature = "std")]
    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn entries_seen_monotonic(n in 1usize..=20) {
                let mut f = Forest::builder(2)
                    .num_trees(10)
                    .capacity(64)
                    .output_after(1)
                    .seed(42)
                    .build()
                    .unwrap();
                for i in 0..n {
                    f.update(&[i as f32, 0.0]).unwrap();
                }
                prop_assert_eq!(f.entries_seen(), n as u64);
            }
        }
    }
}
