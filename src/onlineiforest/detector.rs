#[cfg(all(not(feature = "std"), feature = "serde"))]
use alloc::string::{String, ToString};
#[cfg(not(feature = "std"))]
use alloc::{collections::VecDeque, vec::Vec};
#[cfg(feature = "std")]
use std::collections::VecDeque;

use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{RcfError, Result};

use super::config::OnlineIForestConfig;
use super::tree::OnlineITree;

/// Builder for [`OnlineIForest`].
#[derive(Clone, Debug)]
pub struct OnlineIForestBuilder {
    config: OnlineIForestConfig,
    seed: Option<u64>,
}

impl OnlineIForestBuilder {
    pub(crate) fn new(config: OnlineIForestConfig) -> Self {
        Self { config, seed: None }
    }

    /// Set a random seed for deterministic trees.
    pub fn seed(mut self, value: u64) -> Self {
        self.seed = Some(value);
        self
    }

    /// Set the number of trees in the ensemble.
    pub fn num_trees(mut self, value: usize) -> Self {
        self.config = self.config.with_num_trees(value);
        self
    }

    /// Set the number of recent points retained by the sliding window.
    pub fn window_size(mut self, value: usize) -> Self {
        self.config = self.config.with_window_size(value);
        self
    }

    /// Set the base leaf-splitting threshold.
    pub fn max_leaf_samples(mut self, value: usize) -> Self {
        self.config = self.config.with_max_leaf_samples(value);
        self
    }

    /// Build the detector.
    pub fn build(self) -> Result<OnlineIForest> {
        match self.seed {
            Some(seed) => OnlineIForest::from_config_seeded(&self.config, seed),
            None => OnlineIForest::from_config(&self.config),
        }
    }
}

/// Online Isolation Forest detector for numerical streams.
///
/// Use [`update`](Self::update) or [`update_and_score`](Self::update_and_score)
/// to ingest observations. Use [`score`](Self::score) to preview the current
/// anomaly score for a point without mutating detector state.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OnlineIForest {
    config: OnlineIForestConfig,
    trees: Vec<OnlineITree>,
    window: VecDeque<Vec<f32>>,
    entries_seen: u64,
}

impl OnlineIForest {
    /// Create a builder for points with the required dimensionality.
    pub fn builder(input_dim: usize) -> OnlineIForestBuilder {
        OnlineIForestBuilder::new(OnlineIForestConfig::new(input_dim))
    }

    /// Build directly from config with a random seed.
    pub fn from_config(config: &OnlineIForestConfig) -> Result<Self> {
        let mut seed_rng: Xoshiro256PlusPlus = rand::make_rng();
        Self::new_internal(config.clone(), seed_rng.next_u64())
    }

    /// Build directly from config with an explicit deterministic seed.
    pub fn from_config_seeded(config: &OnlineIForestConfig, seed: u64) -> Result<Self> {
        Self::new_internal(config.clone(), seed)
    }

    fn new_internal(config: OnlineIForestConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let trees = (0..config.num_trees())
            .map(|_| OnlineITree::new(rng.next_u64()))
            .collect();
        let window_capacity = config.window_size() + 1;
        Ok(Self {
            config,
            trees,
            window: VecDeque::with_capacity(window_capacity),
            entries_seen: 0,
        })
    }

    /// Return the detector configuration.
    pub fn config(&self) -> &OnlineIForestConfig {
        &self.config
    }

    /// Number of points processed so far.
    pub fn entries_seen(&self) -> u64 {
        self.entries_seen
    }

    /// Number of trees in the ensemble.
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Return `true` once at least one point has been processed.
    pub fn is_ready(&self) -> bool {
        self.entries_seen > 0
    }

    /// Ingest a point and return its anomaly score under the updated forest.
    pub fn update_and_score(&mut self, point: &[f32]) -> Result<f64> {
        self.update(point)?;
        Ok(self.score_validated(point))
    }

    /// Ingest a point without returning its score.
    pub fn update(&mut self, point: &[f32]) -> Result<()> {
        self.validate_point(point)?;
        let point = point.to_vec();
        let depth_limit = self.config.depth_limit();
        let max_leaf_samples = self.config.max_leaf_samples();

        // Match Algorithm 1: learn the incoming point, then forget the oldest
        // point only after the window overflows.
        self.window.push_back(point);
        let point = self.window.back().expect("just pushed a point");
        for tree in &mut self.trees {
            tree.learn(point, max_leaf_samples, depth_limit);
        }

        if self.window.len() > self.config.window_size() {
            let forgotten = self
                .window
                .pop_front()
                .expect("window just exceeded capacity");
            for tree in &mut self.trees {
                tree.forget(&forgotten, max_leaf_samples);
            }
        }

        self.entries_seen += 1;
        Ok(())
    }

    /// Preview the current anomaly score for `point` without mutating state.
    ///
    /// This can differ from [`update_and_score`](Self::update_and_score)
    /// because the preview is computed before `point` is learned by the
    /// forest. This is specific to the Online Isolation Forest algorithm: the
    /// streamed update score is computed after the new point has been learned,
    /// unlike the preview-style scoring semantics used by `Forest` and
    /// `MStream`. By contrast, `update_and_score(point)` returns the same value
    /// as calling [`update`](Self::update) with `point` and then
    /// [`score`](Self::score) for that same point.
    ///
    /// Calling this before [`is_ready`](Self::is_ready) is allowed, but the value
    /// is not a stable anomaly estimate yet. In an empty forest, all trees have
    /// depth zero, so the preview score is the maximum score.
    pub fn score(&self, point: &[f32]) -> Result<f64> {
        self.validate_point(point)?;
        Ok(self.score_validated(point))
    }

    #[cfg(feature = "serde")]
    /// Serialize detector state to JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|err| RcfError::Io(err.to_string()))
    }

    #[cfg(feature = "serde")]
    /// Deserialize detector state from JSON previously written by [`Self::to_json`].
    pub fn from_json(json: impl AsRef<[u8]>) -> Result<Self> {
        let detector: Self =
            serde_json::from_slice(json.as_ref()).map_err(|err| RcfError::Io(err.to_string()))?;
        detector
            .config
            .validate()
            .map_err(RcfError::invalid_serialized_config)?;
        Ok(detector)
    }

    #[cfg(all(feature = "serde", feature = "std"))]
    /// Serialize detector state to a JSON file.
    pub fn save_json(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path.as_ref(), json).map_err(|err| RcfError::Io(err.to_string()))
    }

    #[cfg(all(feature = "serde", feature = "std"))]
    /// Deserialize detector state from a JSON file previously written by [`Self::save_json`].
    pub fn load_json(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let data = std::fs::read(path.as_ref()).map_err(|err| RcfError::Io(err.to_string()))?;
        Self::from_json(data)
    }

    fn validate_point(&self, point: &[f32]) -> Result<()> {
        if point.len() != self.config.input_dim() {
            return Err(RcfError::DimensionMismatch {
                expected: self.config.input_dim(),
                got: point.len(),
            });
        }
        if point.iter().any(|value| !value.is_finite()) {
            return Err(RcfError::InvalidArgument(
                "point values must be finite".into(),
            ));
        }
        Ok(())
    }

    fn score_validated(&self, point: &[f32]) -> f64 {
        if self.trees.is_empty() {
            return 0.0;
        }
        let average_depth = self
            .trees
            .iter()
            .map(|tree| tree.point_depth(point, self.config.max_leaf_samples()))
            .sum::<f64>()
            / self.trees.len() as f64;
        libm::pow(2.0, -average_depth / self.config.normalization_factor())
    }

    #[cfg(test)]
    pub(crate) fn window_len(&self) -> usize {
        self.window.len()
    }

    #[cfg(test)]
    pub(crate) fn trees(&self) -> &[OnlineITree] {
        &self.trees
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;
    use rstest::rstest;

    use super::*;

    #[test]
    fn score_preview_does_not_mutate_state() {
        let mut detector = OnlineIForest::builder(2)
            .num_trees(4)
            .window_size(16)
            .max_leaf_samples(2)
            .seed(7)
            .build()
            .unwrap();
        detector.update(&[0.0, 0.0]).unwrap();
        let entries_before = detector.entries_seen();
        let window_before = detector.window_len();
        let first = detector.score(&[1.0, 1.0]).unwrap();
        let second = detector.score(&[1.0, 1.0]).unwrap();
        assert_abs_diff_eq!(first, second, epsilon = 1e-12);
        assert_eq!(detector.entries_seen(), entries_before);
        assert_eq!(detector.window_len(), window_before);
    }

    #[test]
    fn becomes_ready_after_first_update() {
        let mut detector = OnlineIForest::builder(1).build().unwrap();
        assert!(!detector.is_ready());
        detector.update(&[0.0]).unwrap();
        assert!(detector.is_ready());
    }

    #[test]
    fn score_before_ready_is_allowed_but_untrained() {
        let detector = OnlineIForest::builder(1).build().unwrap();
        assert!(!detector.is_ready());
        assert_abs_diff_eq!(detector.score(&[0.0]).unwrap(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn update_and_score_matches_update_then_score_for_seeded_detectors() {
        let mut commit = OnlineIForest::builder(1)
            .num_trees(4)
            .window_size(16)
            .max_leaf_samples(2)
            .seed(99)
            .build()
            .unwrap();
        let mut split = commit.clone();
        let point = [3.0];
        let committed = commit.update_and_score(&point).unwrap();
        split.update(&point).unwrap();
        let preview = split.score(&point).unwrap();
        assert_abs_diff_eq!(committed, preview, epsilon = 1e-12);
    }

    fn clustered_detector() -> OnlineIForest {
        let mut detector = OnlineIForest::builder(2)
            .num_trees(16)
            .window_size(64)
            .max_leaf_samples(4)
            .seed(42)
            .build()
            .unwrap();
        for idx in 0..96 {
            let dx = (idx % 8) as f32 * 0.02 - 0.07;
            let dy = (idx / 8 % 8) as f32 * 0.02 - 0.07;
            detector.update(&[dx, dy]).unwrap();
        }
        detector
    }

    #[test]
    fn far_outlier_scores_above_cluster_point() {
        let detector = clustered_detector();
        let inlier = detector.score(&[0.0, 0.0]).unwrap();
        let outlier = detector.score(&[8.0, -8.0]).unwrap();
        assert!(outlier > inlier, "outlier={outlier} inlier={inlier}");
    }

    #[test]
    fn sliding_window_adapts_to_new_region_after_drift() {
        let mut detector = OnlineIForest::builder(1)
            .num_trees(16)
            .window_size(32)
            .max_leaf_samples(4)
            .seed(7)
            .build()
            .unwrap();

        for idx in 0..32 {
            detector.update(&[(idx % 4) as f32 * 0.02]).unwrap();
        }
        let new_region_before = detector.score(&[10.02]).unwrap();

        for idx in 0..48 {
            detector.update(&[10.0 + (idx % 4) as f32 * 0.02]).unwrap();
        }

        let old_region_after = detector.score(&[0.02]).unwrap();
        let new_region_after = detector.score(&[10.02]).unwrap();
        assert!(
            new_region_after < new_region_before,
            "new region should become less anomalous after drift adaptation: before={new_region_before} after={new_region_after}"
        );
        assert!(
            old_region_after > new_region_after,
            "forgotten old region should become more anomalous than the current region: old={old_region_after} new={new_region_after}"
        );
    }

    #[rstest]
    #[case::too_short(vec![1.0], 2, 1)]
    #[case::too_long(vec![1.0, 2.0, 3.0], 2, 3)]
    fn rejects_dimension_mismatch(
        #[case] point: Vec<f32>,
        #[case] expected: usize,
        #[case] got: usize,
    ) {
        let detector = OnlineIForest::builder(2).build().unwrap();
        assert!(matches!(
            detector.score(&point),
            Err(RcfError::DimensionMismatch { expected: e, got: g })
            if e == expected && g == got
        ));
    }

    #[rstest]
    #[case::nan(f32::NAN)]
    #[case::positive_infinity(f32::INFINITY)]
    #[case::negative_infinity(f32::NEG_INFINITY)]
    fn rejects_non_finite_values(#[case] value: f32) {
        let detector = OnlineIForest::builder(2).build().unwrap();
        assert!(matches!(
            detector.score(&[value, 0.0]),
            Err(RcfError::InvalidArgument(_))
        ));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trip_preserves_scoring() {
        let mut detector = OnlineIForest::builder(2)
            .num_trees(4)
            .window_size(16)
            .max_leaf_samples(2)
            .seed(123)
            .build()
            .unwrap();
        for idx in 0..8 {
            detector.update(&[idx as f32, idx as f32 * 0.5]).unwrap();
        }
        let json = detector.to_json().unwrap();
        let restored = OnlineIForest::from_json(json).unwrap();
        assert_abs_diff_eq!(
            detector.score(&[10.0, 10.0]).unwrap(),
            restored.score(&[10.0, 10.0]).unwrap(),
            epsilon = 1e-12
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn from_json_rejects_invalid_serialized_config() {
        let detector = OnlineIForest::builder(1).build().unwrap();
        let mut value: serde_json::Value =
            serde_json::from_str(&detector.to_json().unwrap()).unwrap();
        value["config"]["window_size"] = serde_json::json!(32);
        value["config"]["max_leaf_samples"] = serde_json::json!(32);

        let err = OnlineIForest::from_json(value.to_string()).unwrap_err();

        assert!(
            matches!(
                err,
                RcfError::InvalidSerializedConfig(ref msg)
                    if msg == "window_size must be greater than max_leaf_samples"
            ),
            "unexpected error: {err:?}"
        );
        assert_eq!(
            err.to_string(),
            "invalid serialized config: window_size must be greater than max_leaf_samples"
        );
    }

    proptest! {
        #[test]
        fn entries_seen_and_window_bounds_hold(points in prop::collection::vec(-10.0f32..10.0, 1..40)) {
            let mut detector = OnlineIForest::builder(1)
                .num_trees(3)
                .window_size(8)
                .max_leaf_samples(2)
                .seed(3)
                .build()
                .unwrap();
            for (idx, point) in points.iter().enumerate() {
                detector.update(&[*point]).unwrap();
                prop_assert_eq!(detector.entries_seen(), (idx + 1) as u64);
                prop_assert!(detector.window_len() <= 8);
                let score = detector.score(&[*point]).unwrap();
                prop_assert!(score.is_finite());
                prop_assert!((0.0..=1.0).contains(&score));
                prop_assert!(
                    detector
                        .trees()
                        .iter()
                        .all(|tree| tree.root_height() == detector.window_len())
                );
                prop_assert!(detector.trees().iter().all(OnlineITree::supports_are_nested));
            }
        }
    }
}
