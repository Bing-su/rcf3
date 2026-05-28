#[cfg(all(not(feature = "std"), feature = "serde"))]
use alloc::format;
#[cfg(all(not(feature = "std"), feature = "serde"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use itertools::izip;
use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;

use crate::error::{RcfError, Result};
use crate::math;

use super::clock::StreamClock;
use super::config::MStreamConfig;
use super::normalization::{NormalizedRecord, NumericRangeNormalizer};
use super::scoring::{counts_to_anom, preview_insert_score};
use super::sketch::{CategoricalSketch, NumericSketch, RecordSketch};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct SketchCounts<S> {
    current: S,
    total: S,
}

impl<S> SketchCounts<S> {
    fn new(current: S, total: S) -> Self {
        Self { current, total }
    }
}

/// Decomposed anomaly score for one streamed record.
///
/// The final mStream score is the sum of one record-level score and one score
/// per feature. Exposing the components keeps that explainability available to
/// callers.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MStreamScore {
    /// Sum of the record-level score and all feature-level scores.
    pub total: f64,
    /// Score contributed by the entire record hash.
    pub record: f64,
    /// Scores contributed by numerical features in input order.
    pub numeric_features: Vec<f64>,
    /// Scores contributed by categorical features in input order.
    pub categorical_features: Vec<f64>,
}

/// Builder for [`MStream`].
#[derive(Clone, Debug)]
pub struct MStreamBuilder {
    config: MStreamConfig,
    seed: Option<u64>,
}

impl MStreamBuilder {
    pub(crate) fn new(config: MStreamConfig) -> Self {
        Self { config, seed: None }
    }

    /// Set a random seed for deterministic hashing.
    pub fn seed(mut self, value: u64) -> Self {
        self.seed = Some(value);
        self
    }

    /// Set the number of hash rows.
    pub fn num_rows(mut self, value: usize) -> Self {
        self.config = self.config.with_num_rows(value);
        self
    }

    /// Set the number of buckets per hash row.
    pub fn num_buckets(mut self, value: usize) -> Self {
        self.config = self.config.with_num_buckets(value);
        self
    }

    /// Set the temporal decay factor.
    pub fn alpha(mut self, value: f64) -> Self {
        self.config = self.config.with_alpha(value);
        self
    }

    /// Build the detector.
    pub fn build(self) -> Result<MStream> {
        match self.seed {
            Some(seed) => MStream::from_config_seeded(&self.config, seed),
            None => MStream::from_config(&self.config),
        }
    }
}

/// mStream detector for mixed numerical/categorical records.
///
/// `timestamp` is interpreted as a logical time tick, not as wall-clock time.
/// Scores are invariant to adding a constant offset to all timestamps, while a
/// gap of `k` ticks applies the temporal decay factor `alpha` exactly `k` times.
///
/// Use [`update`](Self::update) or [`update_and_score`](Self::update_and_score)
/// to ingest records. Use [`score`](Self::score) or
/// [`score_detailed`](Self::score_detailed) to preview the next score without
/// mutating detector state.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MStream {
    config: MStreamConfig,
    clock: StreamClock,
    entries_seen: u64,

    record_counts: SketchCounts<RecordSketch>,
    numeric_counts: Vec<SketchCounts<NumericSketch>>,
    categorical_counts: Vec<SketchCounts<CategoricalSketch>>,

    numeric_normalizer: NumericRangeNormalizer,
}

impl MStream {
    /// Create a builder for records with the required dimensions.
    pub fn builder(numeric_dim: usize, categorical_dim: usize) -> MStreamBuilder {
        MStreamBuilder::new(MStreamConfig::new(numeric_dim, categorical_dim))
    }

    /// Build directly from config with a random seed.
    pub fn from_config(config: &MStreamConfig) -> Result<Self> {
        let mut seed_rng: Xoshiro256PlusPlus = rand::make_rng();
        Self::new_internal(config.clone(), seed_rng.next_u64())
    }

    /// Build directly from config with an explicit deterministic seed.
    pub fn from_config_seeded(config: &MStreamConfig, seed: u64) -> Result<Self> {
        Self::new_internal(config.clone(), seed)
    }

    pub(crate) fn new_internal(config: MStreamConfig, seed: u64) -> Result<Self> {
        config.validate()?;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

        let record_current = RecordSketch::new(
            config.num_rows(),
            config.num_buckets(),
            config.numeric_dim(),
            config.categorical_dim(),
            &mut rng,
        )?;
        let record_total = record_current.zeroed_like();
        let record_counts = SketchCounts::new(record_current, record_total);

        let mut numeric_counts = Vec::with_capacity(config.numeric_dim());
        for _ in 0..config.numeric_dim() {
            numeric_counts.push(SketchCounts::new(
                NumericSketch::new(config.num_buckets()),
                NumericSketch::new(config.num_buckets()),
            ));
        }

        let mut categorical_counts = Vec::with_capacity(config.categorical_dim());
        for _ in 0..config.categorical_dim() {
            let score_sketch =
                CategoricalSketch::new(config.num_rows(), config.num_buckets(), &mut rng);
            let total_sketch = score_sketch.zeroed_like();
            categorical_counts.push(SketchCounts::new(score_sketch, total_sketch));
        }

        let numeric_dim = config.numeric_dim();

        debug_assert_eq!(numeric_counts.len(), numeric_dim);
        debug_assert_eq!(categorical_counts.len(), config.categorical_dim());

        Ok(Self {
            config,
            clock: StreamClock::default(),
            entries_seen: 0,
            record_counts,
            numeric_counts,
            categorical_counts,
            numeric_normalizer: NumericRangeNormalizer::new(numeric_dim),
        })
    }

    /// Return the detector configuration.
    pub fn config(&self) -> &MStreamConfig {
        &self.config
    }

    /// Return the number of processed records.
    pub fn entries_seen(&self) -> u64 {
        self.entries_seen
    }

    /// Return the last timestamp observed by the detector.
    pub fn current_time(&self) -> Option<u64> {
        self.clock.current_time()
    }

    /// Return `true` once the detector has processed at least one record.
    pub fn is_ready(&self) -> bool {
        self.entries_seen > 0
    }

    /// Ingest a record and return its anomaly score.
    ///
    /// The `timestamp` argument must be a monotonically non-decreasing tick
    /// index. Only tick differences matter: shifting all timestamps by the same
    /// constant does not change the scores.
    pub fn update_and_score(
        &mut self,
        numeric: &[f64],
        categorical: &[i64],
        timestamp: u64,
    ) -> Result<f64> {
        Ok(self
            .update_and_score_detailed(numeric, categorical, timestamp)?
            .total)
    }

    /// Ingest a record and return the decomposed score used to form the final
    /// anomaly score.
    pub fn update_and_score_detailed(
        &mut self,
        numeric: &[f64],
        categorical: &[i64],
        timestamp: u64,
    ) -> Result<MStreamScore> {
        self.validate_record(numeric, categorical)?;

        debug_assert_eq!(self.numeric_counts.len(), self.config.numeric_dim());
        debug_assert_eq!(
            self.numeric_normalizer.min_numeric.len(),
            self.config.numeric_dim()
        );
        debug_assert_eq!(
            self.numeric_normalizer.max_numeric.len(),
            self.config.numeric_dim()
        );
        debug_assert_eq!(self.categorical_counts.len(), self.config.categorical_dim());

        let tick_gap = self.clock.advance(timestamp)?;
        if tick_gap > 0 {
            self.lower_current_counts(math::powf(self.config.alpha(), tick_gap as f64));
        }

        let cur_t = self.clock.current_tick().max(1);
        let normalized_numeric = self
            .numeric_normalizer
            .normalize(numeric, self.entries_seen)?;
        let numeric_features = self.score_numeric_features(&normalized_numeric, cur_t);

        self.record_counts
            .current
            .insert(&normalized_numeric.raw, categorical, 1.0);
        self.record_counts
            .total
            .insert(&normalized_numeric.raw, categorical, 1.0);

        let categorical_features = self.score_categorical_features(categorical, cur_t);

        let record_score = counts_to_anom(
            self.record_counts
                .total
                .get_count(&normalized_numeric.raw, categorical),
            self.record_counts
                .current
                .get_count(&normalized_numeric.raw, categorical),
            cur_t,
        );
        self.entries_seen += 1;
        let total = record_score
            + numeric_features.iter().sum::<f64>()
            + categorical_features.iter().sum::<f64>();

        Ok(MStreamScore {
            total,
            record: record_score,
            numeric_features,
            categorical_features,
        })
    }

    /// Preview the anomaly score for a record without mutating detector state.
    ///
    /// The preview answers “what would this record score if it were ingested
    /// next?” using the same timestamp semantics as
    /// [`update_and_score`](Self::update_and_score).
    pub fn score(&self, numeric: &[f64], categorical: &[i64], timestamp: u64) -> Result<f64> {
        Ok(self.score_detailed(numeric, categorical, timestamp)?.total)
    }

    /// Ingest a record without returning its score.
    pub fn update(&mut self, numeric: &[f64], categorical: &[i64], timestamp: u64) -> Result<()> {
        let _ = self.update_and_score(numeric, categorical, timestamp)?;
        Ok(())
    }

    /// Preview the decomposed anomaly score without mutating detector state.
    pub fn score_detailed(
        &self,
        numeric: &[f64],
        categorical: &[i64],
        timestamp: u64,
    ) -> Result<MStreamScore> {
        self.validate_record(numeric, categorical)?;

        let clock_step = self.clock.preview(timestamp)?;
        let decay_factor = math::powf(self.config.alpha(), clock_step.tick_gap as f64);
        let normalized_numeric = self
            .numeric_normalizer
            .preview(numeric, self.entries_seen)?;

        let numeric_features = self.preview_numeric_feature_scores(
            &normalized_numeric,
            decay_factor,
            clock_step.current_tick,
        );
        let categorical_features = self.preview_categorical_feature_scores(
            categorical,
            decay_factor,
            clock_step.current_tick,
        );
        let record_score = preview_insert_score(
            self.record_counts
                .total
                .get_count(&normalized_numeric.raw, categorical),
            self.record_counts
                .current
                .get_count(&normalized_numeric.raw, categorical),
            decay_factor,
            clock_step.current_tick,
        );
        let total = record_score
            + numeric_features.iter().sum::<f64>()
            + categorical_features.iter().sum::<f64>();

        Ok(MStreamScore {
            total,
            record: record_score,
            numeric_features,
            categorical_features,
        })
    }

    // -----------------------------------------------------------------------
    // Save / Load
    // -----------------------------------------------------------------------

    /// Serialize the detector state to JSON.
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self)
            .map_err(|e| RcfError::Runtime(format!("failed to serialize MStream: {e}")))
    }

    /// Deserialize detector state from JSON previously written by
    /// [`Self::to_json`].
    #[cfg(feature = "serde")]
    pub fn from_json(json: impl AsRef<[u8]>) -> Result<Self> {
        serde_json::from_slice(json.as_ref())
            .map_err(|e| RcfError::InvalidArgument(format!("invalid MStream JSON: {e}")))
    }

    /// Serialize the detector state to a JSON file.
    #[cfg(all(feature = "serde", feature = "std"))]
    pub fn save_json(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path.as_ref(), json).map_err(|e| RcfError::Io(e.to_string()))
    }

    /// Deserialize detector state from a JSON file previously written by
    /// [`Self::save_json`].
    #[cfg(all(feature = "serde", feature = "std"))]
    pub fn load_json(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let data = std::fs::read(path.as_ref()).map_err(|e| RcfError::Io(e.to_string()))?;
        Self::from_json(&data)
    }

    fn validate_record(&self, numeric: &[f64], categorical: &[i64]) -> Result<()> {
        if numeric.len() != self.config.numeric_dim() {
            return Err(RcfError::DimensionMismatch {
                expected: self.config.numeric_dim(),
                got: numeric.len(),
            });
        }
        if categorical.len() != self.config.categorical_dim() {
            return Err(RcfError::DimensionMismatch {
                expected: self.config.categorical_dim(),
                got: categorical.len(),
            });
        }
        Ok(())
    }

    fn lower_current_counts(&mut self, factor: f64) {
        debug_assert!(factor.is_finite());
        debug_assert!((0.0..=1.0).contains(&factor));

        self.record_counts.current.lower(factor);
        for counts in &mut self.numeric_counts {
            counts.current.lower(factor);
        }
        for counts in &mut self.categorical_counts {
            counts.current.lower(factor);
        }
    }

    fn score_numeric_features(
        &mut self,
        numeric: &NormalizedRecord,
        current_tick: u64,
    ) -> Vec<f64> {
        let mut scores = Vec::with_capacity(self.config.numeric_dim());

        for (counts, value) in izip!(&mut self.numeric_counts, &numeric.normalized) {
            counts.current.insert(*value, 1.0);
            counts.total.insert(*value, 1.0);
            scores.push(counts_to_anom(
                counts.total.get_count(*value),
                counts.current.get_count(*value),
                current_tick,
            ));
        }

        scores
    }

    fn score_categorical_features(&mut self, categorical: &[i64], current_tick: u64) -> Vec<f64> {
        let mut scores = Vec::with_capacity(self.config.categorical_dim());

        for (counts, value) in izip!(&mut self.categorical_counts, categorical) {
            counts.current.insert(*value, 1.0);
            counts.total.insert(*value, 1.0);
            scores.push(counts_to_anom(
                counts.total.get_count(*value),
                counts.current.get_count(*value),
                current_tick,
            ));
        }

        scores
    }

    fn preview_numeric_feature_scores(
        &self,
        numeric: &NormalizedRecord,
        decay_factor: f64,
        current_tick: u64,
    ) -> Vec<f64> {
        izip!(&self.numeric_counts, &numeric.normalized)
            .map(|(counts, value)| {
                preview_insert_score(
                    counts.total.get_count(*value),
                    counts.current.get_count(*value),
                    decay_factor,
                    current_tick,
                )
            })
            .collect()
    }

    fn preview_categorical_feature_scores(
        &self,
        categorical: &[i64],
        decay_factor: f64,
        current_tick: u64,
    ) -> Vec<f64> {
        izip!(&self.categorical_counts, categorical)
            .map(|(counts, value)| {
                preview_insert_score(
                    counts.total.get_count(*value),
                    counts.current.get_count(*value),
                    decay_factor,
                    current_tick,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    use approx::{abs_diff_eq, assert_abs_diff_eq};
    use proptest::prelude::*;

    use crate::error::RcfError;

    use super::*;

    fn run_scores(timestamps: &[u64]) -> Vec<f64> {
        let mut detector = MStream::builder(0, 1)
            .seed(7)
            .alpha(0.8)
            .num_rows(2)
            .num_buckets(256)
            .build()
            .unwrap();

        timestamps
            .iter()
            .enumerate()
            .map(|(index, timestamp)| {
                let value = if index < 2 { 1 } else { 2 };
                detector
                    .update_and_score(&[], &[value], *timestamp)
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn counts_to_anom_penalizes_negative_deviation() {
        let score = counts_to_anom(10.0, 0.0, 2);
        assert!(score > 0.0);
        assert_abs_diff_eq!(score, 10.0, epsilon = 1e-12);
    }

    #[test]
    fn rejects_non_monotonic_timestamps() {
        let mut d = MStream::builder(1, 1).seed(7).build().unwrap();
        d.update_and_score(&[0.1], &[1], 10).unwrap();
        let err = d.update_and_score(&[0.2], &[1], 9).unwrap_err();
        assert!(matches!(err, RcfError::InvalidArgument(_)));
    }

    #[test]
    fn score_is_finite_and_non_negative() {
        let mut d = MStream::builder(2, 1)
            .seed(42)
            .alpha(0.8)
            .num_rows(2)
            .num_buckets(256)
            .build()
            .unwrap();

        for _ in 0..200 {
            let _ = d.update_and_score(&[1.0, 1.2], &[2], 1).unwrap();
        }

        let score = d.update_and_score(&[100.0, 120.0], &[9], 2).unwrap();
        assert!(score.is_finite());
        assert!(score >= 0.0);
    }

    #[test]
    fn seeded_from_config_is_deterministic() {
        let cfg = crate::mstream::MStreamConfig::new(2, 1)
            .with_alpha(0.8)
            .with_num_rows(2)
            .with_num_buckets(256);

        let mut a = MStream::from_config_seeded(&cfg, 123).unwrap();
        let mut b = MStream::from_config_seeded(&cfg, 123).unwrap();

        for _ in 0..64 {
            let sa = a.update_and_score(&[1.0, 1.2], &[2], 1).unwrap();
            let sb = b.update_and_score(&[1.0, 1.2], &[2], 1).unwrap();
            assert_abs_diff_eq!(sa, sb, epsilon = 1e-12);
        }
    }

    #[test]
    fn becomes_ready_after_first_update() {
        let mut d = MStream::builder(1, 1).seed(7).build().unwrap();
        assert!(!d.is_ready());
        d.update(&[0.1], &[1], 1).unwrap();
        assert!(d.is_ready());
    }

    #[test]
    fn scores_are_invariant_to_timestamp_offset() {
        let base = run_scores(&[1, 1, 2, 2]);
        let shifted = run_scores(&[100, 100, 101, 101]);

        assert_eq!(base.len(), shifted.len());
        for (lhs, rhs) in base.iter().zip(shifted.iter()) {
            assert_abs_diff_eq!(lhs, rhs, epsilon = 1e-12);
        }
    }

    #[test]
    fn gap_between_timestamps_applies_decay_per_tick() {
        let alpha = 0.8;
        let mut d = MStream::builder(1, 0)
            .seed(7)
            .alpha(alpha)
            .num_rows(2)
            .num_buckets(256)
            .build()
            .unwrap();

        d.update(&[9.0], &[], 1).unwrap();
        d.update(&[9.0], &[], 4).unwrap();

        let count = d.numeric_counts[0].current.get_count(0.0);
        let expected = 1.0 + alpha.powi(3);
        assert_abs_diff_eq!(count, expected, epsilon = 1e-12);
    }

    #[test]
    fn score_returns_raw_sum_without_log_compression() {
        let mut d = MStream::builder(0, 1)
            .seed(7)
            .alpha(0.8)
            .num_rows(2)
            .num_buckets(256)
            .build()
            .unwrap();

        d.update(&[], &[1], 1).unwrap();
        let score = d.update_and_score(&[], &[2], 2).unwrap();

        let expected = counts_to_anom(
            d.categorical_counts[0].total.get_count(2),
            d.categorical_counts[0].current.get_count(2),
            2,
        ) + counts_to_anom(
            d.record_counts.total.get_count(&[], &[2]),
            d.record_counts.current.get_count(&[], &[2]),
            2,
        );
        assert_abs_diff_eq!(score, expected, epsilon = 1e-12);
    }

    #[test]
    fn detailed_score_exposes_the_paper_score_decomposition() {
        let mut d = MStream::builder(2, 1)
            .seed(7)
            .alpha(0.8)
            .num_rows(2)
            .num_buckets(256)
            .build()
            .unwrap();

        d.update(&[0.1, 0.2], &[1], 1).unwrap();
        let score = d.update_and_score_detailed(&[1.0, 2.0], &[2], 2).unwrap();

        assert_eq!(score.numeric_features.len(), 2);
        assert_eq!(score.categorical_features.len(), 1);
        let recomposed = score.record
            + score.numeric_features.iter().sum::<f64>()
            + score.categorical_features.iter().sum::<f64>();
        assert_abs_diff_eq!(score.total, recomposed, epsilon = 1e-12);
    }

    #[test]
    fn score_previews_without_mutating_state() {
        let mut d = MStream::builder(1, 1).seed(7).build().unwrap();
        d.update(&[0.1], &[1], 1).unwrap();

        let before_entries = d.entries_seen();
        let before_time = d.current_time();
        let preview = d.score(&[1.0], &[2], 2).unwrap();

        assert_eq!(d.entries_seen(), before_entries);
        assert_eq!(d.current_time(), before_time);

        let actual = d.update_and_score(&[1.0], &[2], 2).unwrap();

        assert_eq!(d.entries_seen(), before_entries + 1);
        assert_eq!(before_time, Some(1));
        assert_abs_diff_eq!(preview, actual, epsilon = 1e-12);
    }

    #[test]
    fn detailed_score_previews_without_mutating_state() {
        let mut d = MStream::builder(2, 1).seed(7).build().unwrap();
        d.update(&[0.1, 0.2], &[1], 1).unwrap();

        let before_entries = d.entries_seen();
        let before_time = d.current_time();
        let preview = d.score_detailed(&[1.0, 2.0], &[2], 2).unwrap();

        assert_eq!(d.entries_seen(), before_entries);
        assert_eq!(d.current_time(), before_time);

        let actual = d.update_and_score_detailed(&[1.0, 2.0], &[2], 2).unwrap();
        assert_eq!(
            preview.numeric_features.len(),
            actual.numeric_features.len()
        );
        assert_eq!(
            preview.categorical_features.len(),
            actual.categorical_features.len()
        );
        assert_abs_diff_eq!(preview.total, actual.total, epsilon = 1e-12);
    }

    #[test]
    fn rejects_non_finite_numeric_values() {
        let mut d = MStream::builder(1, 0).seed(7).build().unwrap();
        let err = d.update_and_score(&[f64::NAN], &[], 1).unwrap_err();
        assert!(matches!(err, RcfError::InvalidArgument(_)));
    }

    #[test]
    fn accepts_negative_numeric_values_below_minus_one() {
        let mut d = MStream::builder(1, 0).seed(7).build().unwrap();

        d.update(&[-2.0], &[], 1).unwrap();
        let preview = d.score(&[-10.0], &[], 2).unwrap();
        let detailed = d.score_detailed(&[-10.0], &[], 2).unwrap();
        let committed = d.update_and_score(&[-10.0], &[], 2).unwrap();

        assert!(preview.is_finite());
        assert!(detailed.total.is_finite());
        assert_abs_diff_eq!(preview, committed, epsilon = 1e-12);
    }

    #[test]
    fn repeated_anomalous_group_scores_above_baseline_tick() {
        let mut d = MStream::builder(0, 1)
            .seed(11)
            .alpha(0.8)
            .num_rows(2)
            .num_buckets(256)
            .build()
            .unwrap();

        let mut baseline_max = 0.0_f64;
        for tick in 1..=8 {
            let score = d.update_and_score(&[], &[1], tick).unwrap();
            baseline_max = baseline_max.max(score);
        }

        let mut anomaly_max = 0.0_f64;
        for _ in 0..6 {
            let score = d.update_and_score(&[], &[2], 9).unwrap();
            anomaly_max = anomaly_max.max(score);
        }

        assert!(anomaly_max > baseline_max);
    }

    proptest! {
        #[test]
        fn detailed_score_total_matches_component_sum(
            records in prop::collection::vec(
                ((-10_000.0f64..10_000.0), (-10_000.0f64..10_000.0), -8i64..=8, 0u64..=3),
                1..=32,
            ),
        ) {
            let mut detector = MStream::builder(2, 1)
                .seed(17)
                .num_rows(2)
                .num_buckets(256)
                .build()
                .unwrap();
            let mut timestamp = 1;

            for (left, right, category, gap) in records {
                timestamp += gap;
                let score = detector
                    .update_and_score_detailed(&[left, right], &[category], timestamp)
                    .unwrap();
                let recomposed = score.record
                    + score.numeric_features.iter().sum::<f64>()
                    + score.categorical_features.iter().sum::<f64>();

                prop_assert!(abs_diff_eq!(score.total, recomposed, epsilon = 1e-12));
            }
        }

        #[test]
        fn seeded_detectors_match_for_same_sequence(
            records in prop::collection::vec(
                ((-10_000.0f64..10_000.0), -8i64..=8, 0u64..=3),
                1..=32,
            ),
        ) {
            let mut left = MStream::builder(1, 1).seed(91).build().unwrap();
            let mut right = MStream::builder(1, 1).seed(91).build().unwrap();
            let mut timestamp = 1;

            for (numeric, category, gap) in records {
                timestamp += gap;
                let left_score = left
                    .update_and_score(&[numeric], &[category], timestamp)
                    .unwrap();
                let right_score = right
                    .update_and_score(&[numeric], &[category], timestamp)
                    .unwrap();

                prop_assert!(abs_diff_eq!(left_score, right_score, epsilon = 1e-12));
            }
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn json_roundtrip_preserves_state() {
        let mut d = MStream::builder(1, 1).seed(7).build().unwrap();
        d.update(&[0.1], &[1], 1).unwrap();
        d.update(&[0.2], &[2], 2).unwrap();

        let json = d.to_json().unwrap();
        let restored = MStream::from_json(json).unwrap();

        assert_eq!(restored.entries_seen(), d.entries_seen());
        assert_eq!(restored.current_time(), d.current_time());
        assert_eq!(restored.config().num_rows(), d.config().num_rows());
        assert_eq!(restored.config().num_buckets(), d.config().num_buckets());
        assert_eq!(restored.config().numeric_dim(), d.config().numeric_dim());
        assert_eq!(
            restored.config().categorical_dim(),
            d.config().categorical_dim()
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn from_json_rejects_invalid_json_as_invalid_argument() {
        let err = MStream::from_json(b"not json").unwrap_err();

        assert!(
            matches!(err, RcfError::InvalidArgument(msg) if msg.contains("invalid MStream JSON"))
        );
    }
}
