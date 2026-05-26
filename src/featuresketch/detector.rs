#[cfg(all(not(feature = "std"), feature = "serde"))]
use alloc::format;
#[cfg(all(not(feature = "std"), feature = "serde"))]
use alloc::string::String;

use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{RcfError, Result};

use super::chain::ChainLayout;
use super::config::FeatureSketchConfig;
use super::input::normalize;
use super::projection::{ProjectedEvent, ProjectionSeeds, project};
use super::sketch::EnsembleSketch;

const FORMAT_VERSION: u32 = 1;

/// Builder for [`FeatureSketch`].
#[derive(Clone, Debug)]
pub struct FeatureSketchBuilder {
    config: FeatureSketchConfig,
    seed: Option<u64>,
}

impl FeatureSketchBuilder {
    pub(crate) fn new(config: FeatureSketchConfig) -> Self {
        Self { config, seed: None }
    }

    /// Set a random seed for deterministic projection, chain, and sketch layout.
    pub fn seed(mut self, value: u64) -> Self {
        self.seed = Some(value);
        self
    }

    pub fn value_projection_dims(mut self, value: usize) -> Self {
        self.config = self.config.with_value_projection_dims(value);
        self
    }

    pub fn presence_projection_dims(mut self, value: usize) -> Self {
        self.config = self.config.with_presence_projection_dims(value);
        self
    }

    pub fn chains_per_ensemble(mut self, value: usize) -> Self {
        self.config = self.config.with_chains_per_ensemble(value);
        self
    }

    pub fn chain_depth(mut self, value: usize) -> Self {
        self.config = self.config.with_chain_depth(value);
        self
    }

    pub fn sketch_rows(mut self, value: usize) -> Self {
        self.config = self.config.with_sketch_rows(value);
        self
    }

    pub fn sketch_buckets(mut self, value: usize) -> Self {
        self.config = self.config.with_sketch_buckets(value);
        self
    }

    pub fn decay_half_life(mut self, value: u64) -> Self {
        self.config = self.config.with_decay_half_life(value);
        self
    }

    /// Build the detector.
    pub fn build(self) -> Result<FeatureSketch> {
        match self.seed {
            Some(seed) => FeatureSketch::from_config_seeded(&self.config, seed),
            None => FeatureSketch::from_config(&self.config),
        }
    }
}

/// Sparse feature-name sketch detector for schema-evolving streams.
///
/// `score()` previews the current anomaly score without mutating detector
/// state. `update()` teaches the detector and advances the internal event
/// counter. Scores are higher for more anomalous events.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FeatureSketch {
    #[cfg_attr(not(feature = "serde"), allow(dead_code))]
    format_version: u32,
    config: FeatureSketchConfig,
    projection_seeds: ProjectionSeeds,
    value_layout: ChainLayout,
    presence_layout: ChainLayout,
    value_sketch: EnsembleSketch,
    presence_sketch: EnsembleSketch,
    current_epoch: u64,
    entries_seen: u64,
}

impl FeatureSketch {
    /// Create a builder with the documented default configuration.
    pub fn builder() -> FeatureSketchBuilder {
        FeatureSketchBuilder::new(FeatureSketchConfig::new())
    }

    /// Build directly from config with a random seed.
    pub fn from_config(config: &FeatureSketchConfig) -> Result<Self> {
        let mut seed_rng: Xoshiro256PlusPlus = rand::make_rng();
        Self::new_internal(config.clone(), seed_rng.next_u64())
    }

    /// Build directly from config with an explicit deterministic seed.
    pub fn from_config_seeded(config: &FeatureSketchConfig, seed: u64) -> Result<Self> {
        Self::new_internal(config.clone(), seed)
    }

    fn new_internal(config: FeatureSketchConfig, seed: u64) -> Result<Self> {
        config.validate()?;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let projection_seeds = ProjectionSeeds::new(&mut rng);
        let value_layout = ChainLayout::new(
            config.chains_per_ensemble(),
            config.chain_depth(),
            config.value_projection_dims(),
            false,
            &mut rng,
        );
        let presence_layout = ChainLayout::new(
            config.chains_per_ensemble(),
            config.chain_depth(),
            config.presence_projection_dims(),
            true,
            &mut rng,
        );
        let value_sketch = EnsembleSketch::new(&config, value_layout.len(), &mut rng);
        let presence_sketch = EnsembleSketch::new(&config, presence_layout.len(), &mut rng);

        Ok(Self {
            format_version: FORMAT_VERSION,
            config,
            projection_seeds,
            value_layout,
            presence_layout,
            value_sketch,
            presence_sketch,
            current_epoch: 0,
            entries_seen: 0,
        })
    }

    /// Return the detector configuration.
    pub fn config(&self) -> &FeatureSketchConfig {
        &self.config
    }

    /// Return the number of processed events.
    pub fn entries_seen(&self) -> u64 {
        self.entries_seen
    }

    /// Return `true` once the detector has processed at least one event.
    pub fn is_ready(&self) -> bool {
        self.entries_seen > 0
    }

    /// Preview the anomaly score for an event without mutating detector state.
    pub fn score<I, N>(&self, features: I) -> Result<f64>
    where
        I: IntoIterator<Item = (N, f64)>,
        N: AsRef<str>,
    {
        let projected = self.project_features(features)?;
        Ok(self.score_projected(&projected))
    }

    /// Ingest an event without returning its score.
    pub fn update<I, N>(&mut self, features: I) -> Result<()>
    where
        I: IntoIterator<Item = (N, f64)>,
        N: AsRef<str>,
    {
        let projected = self.project_features(features)?;
        let next_epoch = self
            .current_epoch
            .checked_add(1)
            .ok_or_else(|| RcfError::Overflow("FeatureSketch epoch overflow".into()))?;
        let next_entries_seen = self
            .entries_seen
            .checked_add(1)
            .ok_or_else(|| RcfError::Overflow("FeatureSketch entries_seen overflow".into()))?;
        self.value_sketch.update(
            &self.value_layout,
            &projected.value,
            next_epoch,
            &self.config,
        );
        self.presence_sketch.update(
            &self.presence_layout,
            &projected.presence,
            next_epoch,
            &self.config,
        );
        self.current_epoch = next_epoch;
        self.entries_seen = next_entries_seen;
        Ok(())
    }

    /// Serialize detector state to JSON.
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self)
            .map_err(|err| RcfError::Runtime(format!("failed to serialize FeatureSketch: {err}")))
    }

    /// Deserialize detector state from JSON previously written by [`Self::to_json`].
    #[cfg(feature = "serde")]
    pub fn from_json(json: impl AsRef<[u8]>) -> Result<Self> {
        let detector: Self = serde_json::from_slice(json.as_ref()).map_err(|err| {
            RcfError::InvalidArgument(format!("invalid FeatureSketch JSON: {err}"))
        })?;
        if detector.format_version != FORMAT_VERSION {
            return Err(RcfError::InvalidArgument(
                "invalid FeatureSketch JSON: unsupported format version".into(),
            ));
        }
        detector.config.validate()?;
        Ok(detector)
    }

    /// Serialize detector state to a JSON file.
    #[cfg(all(feature = "serde", feature = "std"))]
    pub fn save_json(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path.as_ref(), json).map_err(|err| RcfError::Io(err.to_string()))
    }

    /// Deserialize detector state from a JSON file previously written by [`Self::save_json`].
    #[cfg(all(feature = "serde", feature = "std"))]
    pub fn load_json(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let data = std::fs::read(path.as_ref()).map_err(|err| RcfError::Io(err.to_string()))?;
        Self::from_json(data)
    }

    fn project_features<I, N>(&self, features: I) -> Result<ProjectedEvent>
    where
        I: IntoIterator<Item = (N, f64)>,
        N: AsRef<str>,
    {
        let normalized = normalize(features)?;
        Ok(project(
            &normalized,
            self.config.value_projection_dims(),
            self.config.presence_projection_dims(),
            &self.projection_seeds,
        ))
    }

    fn score_projected(&self, projected: &ProjectedEvent) -> f64 {
        let value = self.value_sketch.score(
            &self.value_layout,
            &projected.value,
            self.current_epoch,
            &self.config,
        );
        let presence = self.presence_sketch.score(
            &self.presence_layout,
            &projected.presence,
            self.current_epoch,
            &self.config,
        );
        (value + presence) / 2.0
    }
}

#[cfg(test)]
mod tests {
    #[cfg(all(not(feature = "std"), feature = "serde"))]
    use alloc::format;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    use crate::error::RcfError;

    use super::*;

    fn small_detector(seed: u64) -> FeatureSketch {
        FeatureSketch::builder()
            .value_projection_dims(8)
            .presence_projection_dims(8)
            .chains_per_ensemble(8)
            .chain_depth(4)
            .sketch_rows(2)
            .sketch_buckets(256)
            .decay_half_life(256)
            .seed(seed)
            .build()
            .unwrap()
    }

    fn tiny_detector(seed: u64) -> FeatureSketch {
        FeatureSketch::builder()
            .value_projection_dims(4)
            .presence_projection_dims(4)
            .chains_per_ensemble(4)
            .chain_depth(3)
            .sketch_rows(2)
            .sketch_buckets(64)
            .decay_half_life(64)
            .seed(seed)
            .build()
            .unwrap()
    }

    fn p95(mut values: Vec<f64>) -> f64 {
        values.sort_by(|left, right| left.partial_cmp(right).unwrap());
        values[((values.len() * 95) / 100).min(values.len() - 1)]
    }

    #[test]
    fn duplicate_names_match_precombined_input() {
        let mut left = tiny_detector(11);
        let mut right = tiny_detector(11);
        for _ in 0..100 {
            left.update([("a", 1.0), ("a", 2.0), ("b", -1.0)]).unwrap();
            right.update([("a", 3.0), ("b", -1.0)]).unwrap();
        }

        let left_score = left.score([("a", 1.5), ("a", 1.5), ("b", -1.0)]).unwrap();
        let right_score = right.score([("a", 3.0), ("b", -1.0)]).unwrap();
        assert_eq!(left_score, right_score);
    }

    #[test]
    fn zero_valued_present_feature_differs_from_absence() {
        let mut detector = small_detector(12);
        for _ in 0..300 {
            detector.update([("a", 0.0)]).unwrap();
        }

        let present = detector.score([("a", 0.0)]).unwrap();
        let absent = detector.score([] as [(&str, f64); 0]).unwrap();
        assert_ne!(present, absent);
    }

    #[test]
    fn empty_event_scores_and_updates_deterministically() {
        let mut left = tiny_detector(13);
        let mut right = tiny_detector(13);
        assert_eq!(
            left.score([] as [(&str, f64); 0]).unwrap(),
            right.score([] as [(&str, f64); 0]).unwrap()
        );
        left.update([] as [(&str, f64); 0]).unwrap();
        right.update([] as [(&str, f64); 0]).unwrap();
        assert_eq!(left.entries_seen(), 1);
        assert_eq!(
            left.score([] as [(&str, f64); 0]).unwrap(),
            right.score([] as [(&str, f64); 0]).unwrap()
        );
    }

    #[test]
    fn signed_values_are_accepted_and_non_finite_values_rejected() {
        let mut detector = tiny_detector(14);
        detector.update([("neg", -10.0), ("pos", 10.0)]).unwrap();
        assert!(detector.score([("neg", -1.0)]).unwrap().is_finite());
        assert!(matches!(
            detector.update([("bad", f64::NAN)]),
            Err(RcfError::InvalidArgument(_))
        ));
        assert!(matches!(
            detector.update([("bad", f64::MAX), ("bad", f64::MAX)]),
            Err(RcfError::InvalidArgument(_))
        ));
    }

    #[test]
    #[cfg(feature = "serde")]
    fn score_is_pure() {
        let mut detector = tiny_detector(15);
        for _ in 0..80 {
            detector.update([("a", 1.0), ("b", 2.0)]).unwrap();
        }
        let before = detector.to_json().unwrap();
        let first = detector.score([("a", 1.0), ("b", 2.0)]).unwrap();
        let second = detector.score([("a", 1.0), ("b", 2.0)]).unwrap();
        let after = detector.to_json().unwrap();
        assert_eq!(first, second);
        assert_eq!(before, after);
    }

    #[test]
    fn update_advances_state_and_affects_later_scores() {
        let mut detector = tiny_detector(16);
        let before = detector.score([("a", 1.0)]).unwrap();
        detector.update([("a", 1.0)]).unwrap();
        let after = detector.score([("a", 1.0)]).unwrap();
        assert_eq!(detector.entries_seen(), 1);
        assert!(after < before);
    }

    #[test]
    fn entries_seen_overflow_does_not_mutate_state() {
        let mut detector = tiny_detector(18);
        detector.update([("a", 1.0)]).unwrap();
        detector.entries_seen = u64::MAX;

        let epoch_before = detector.current_epoch;
        let score_before = detector.score([("a", 1.0)]).unwrap();
        assert!(matches!(
            detector.update([("a", 1.0)]),
            Err(RcfError::Overflow(_))
        ));

        assert_eq!(detector.current_epoch, epoch_before);
        assert_eq!(detector.entries_seen, u64::MAX);
        assert_eq!(detector.score([("a", 1.0)]).unwrap(), score_before);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn score_after_unrelated_updates_applies_decay_without_mutation() {
        let mut detector = tiny_detector(17);
        detector.update([("a", 1.0)]).unwrap();
        for i in 0..20 {
            detector.update([(format!("other:{i}"), 1.0)]).unwrap();
        }

        let before = detector.to_json().unwrap();
        let _ = detector.score([("a", 1.0)]).unwrap();
        let after = detector.to_json().unwrap();
        assert_eq!(before, after);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn serde_roundtrip_preserves_future_scores_and_updates() {
        let mut left = tiny_detector(19);
        for _ in 0..60 {
            left.update([("a", 1.0), ("b", 2.0)]).unwrap();
        }
        let json = left.to_json().unwrap();
        let mut right = FeatureSketch::from_json(json).unwrap();

        assert_eq!(
            left.score([("future", 42.0)]).unwrap(),
            right.score([("future", 42.0)]).unwrap()
        );
        left.update([("future", 42.0)]).unwrap();
        right.update([("future", 42.0)]).unwrap();
        assert_eq!(
            left.score([("another_future", -7.0)]).unwrap(),
            right.score([("another_future", -7.0)]).unwrap()
        );
    }

    #[test]
    fn feature_growth_scores_above_warm_baseline() {
        let mut detector = small_detector(20);
        for _ in 0..512 {
            detector.update([("a", 1.0), ("b", 1.0)]).unwrap();
        }
        let baseline: Vec<_> = (0..64)
            .map(|_| detector.score([("a", 1.0), ("b", 1.0)]).unwrap())
            .collect();

        let grown = detector
            .score([("a", 1.0), ("b", 1.0), ("new_feature", 1.0)])
            .unwrap();
        assert!(grown > p95(baseline));
    }

    #[test]
    fn feature_shrink_scores_above_warm_baseline_and_then_adapts() {
        let mut detector = small_detector(21);
        for _ in 0..512 {
            detector
                .update([("a", 1.0), ("b", 1.0), ("c", 1.0)])
                .unwrap();
        }
        let baseline: Vec<_> = (0..64)
            .map(|_| {
                detector
                    .score([("a", 1.0), ("b", 1.0), ("c", 1.0)])
                    .unwrap()
            })
            .collect();

        let shrunk_before = detector.score([("a", 1.0), ("b", 1.0)]).unwrap();
        assert!(shrunk_before > p95(baseline));

        for _ in 0..512 {
            detector.update([("a", 1.0), ("b", 1.0)]).unwrap();
        }
        let shrunk_after = detector.score([("a", 1.0), ("b", 1.0)]).unwrap();
        assert!(shrunk_after < shrunk_before);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn high_cardinality_stream_does_not_serialize_feature_names() {
        let mut detector = tiny_detector(22);
        for i in 0..200 {
            detector.update([(format!("feature:{i}"), 1.0)]).unwrap();
        }
        let json = detector.to_json().unwrap();
        assert!(!json.contains("feature:"));
    }
}
