#[cfg(all(not(feature = "std"), feature = "serde"))]
use alloc::string::{String, ToString};
#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;

use crate::error::{RcfError, Result};

use super::config::MStreamConfig;
use super::sketch::{CategoricalSketch, NumericSketch, RecordSketch};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

fn counts_to_anom(total: f64, current: f64, current_time: u64) -> f64 {
    let cur_t = (current_time as f64).max(1.0);
    let cur_mean = total / cur_t;
    if cur_mean <= f64::EPSILON {
        return 0.0;
    }

    let sqerr = current - cur_mean;
    let sqerr = sqerr * sqerr;

    sqerr / cur_mean + sqerr / (cur_mean * (cur_t - 1.0).max(1.0))
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

    /// Set random seed for deterministic hashing.
    pub fn seed(mut self, value: u64) -> Self {
        self.seed = Some(value);
        self
    }

    /// Set number of hash rows.
    pub fn num_rows(mut self, value: usize) -> Self {
        self.config = self.config.with_num_rows(value);
        self
    }

    /// Set number of buckets.
    pub fn num_buckets(mut self, value: usize) -> Self {
        self.config = self.config.with_num_buckets(value);
        self
    }

    /// Set temporal decay factor.
    pub fn alpha(mut self, value: f64) -> Self {
        self.config = self.config.with_alpha(value);
        self
    }

    /// Build detector.
    pub fn build(self) -> Result<MStream> {
        match self.seed {
            Some(seed) => MStream::from_config_seeded(&self.config, seed),
            None => MStream::from_config(&self.config),
        }
    }
}

/// mStream detector for mixed numerical/categorical records.
///
/// `timestamp` is interpreted as the paper's time tick, not as wall-clock time.
/// Scores are invariant to adding a constant offset to all timestamps, while a
/// gap of `k` ticks applies the temporal decay factor `alpha` exactly `k` times.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MStream {
    config: MStreamConfig,
    current_time: Option<u64>,
    current_tick: u64,
    entries_seen: u64,

    cur_count: RecordSketch,
    total_count: RecordSketch,

    numeric_score: Vec<NumericSketch>,
    numeric_total: Vec<NumericSketch>,
    categorical_score: Vec<CategoricalSketch>,
    categorical_total: Vec<CategoricalSketch>,

    min_numeric: Vec<f64>,
    max_numeric: Vec<f64>,
}

impl MStream {
    /// Create a builder with the required dimensions.
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

        let cur_count = RecordSketch::new(
            config.num_rows,
            config.num_buckets,
            config.numeric_dim,
            config.categorical_dim,
            &mut rng,
        )?;
        let total_count = RecordSketch::new(
            config.num_rows,
            config.num_buckets,
            config.numeric_dim,
            config.categorical_dim,
            &mut rng,
        )?;

        let mut numeric_score = Vec::with_capacity(config.numeric_dim);
        let mut numeric_total = Vec::with_capacity(config.numeric_dim);
        for _ in 0..config.numeric_dim {
            numeric_score.push(NumericSketch::new(config.num_buckets));
            numeric_total.push(NumericSketch::new(config.num_buckets));
        }

        let mut categorical_score = Vec::with_capacity(config.categorical_dim);
        let mut categorical_total = Vec::with_capacity(config.categorical_dim);
        for _ in 0..config.categorical_dim {
            categorical_score.push(CategoricalSketch::new(
                config.num_rows,
                config.num_buckets,
                &mut rng,
            ));
            categorical_total.push(CategoricalSketch::new(
                config.num_rows,
                config.num_buckets,
                &mut rng,
            ));
        }

        let numeric_dim = config.numeric_dim;

        debug_assert_eq!(numeric_score.len(), numeric_dim);
        debug_assert_eq!(numeric_total.len(), numeric_dim);
        debug_assert_eq!(categorical_score.len(), config.categorical_dim);
        debug_assert_eq!(categorical_total.len(), config.categorical_dim);

        Ok(Self {
            config,
            current_time: None,
            current_tick: 0,
            entries_seen: 0,
            cur_count,
            total_count,
            numeric_score,
            numeric_total,
            categorical_score,
            categorical_total,
            min_numeric: vec![f64::INFINITY; numeric_dim],
            max_numeric: vec![f64::NEG_INFINITY; numeric_dim],
        })
    }

    /// Returns configuration.
    pub fn config(&self) -> &MStreamConfig {
        &self.config
    }

    /// Number of processed records.
    pub fn entries_seen(&self) -> u64 {
        self.entries_seen
    }

    /// Last timestamp observed by the detector.
    pub fn current_time(&self) -> Option<u64> {
        self.current_time
    }

    /// Returns `true` once the detector has processed at least one record.
    pub fn is_ready(&self) -> bool {
        self.entries_seen > 0
    }

    /// Update the detector and return anomaly score for this record.
    ///
    /// The `timestamp` argument must be a monotonically non-decreasing tick
    /// index. Only tick differences matter: shifting all timestamps by the same
    /// constant does not change the scores.
    pub fn update_and_score(
        &mut self,
        numeric: &[f32],
        categorical: &[i64],
        timestamp: u64,
    ) -> Result<f64> {
        self.validate_record(numeric, categorical)?;

        debug_assert_eq!(self.numeric_score.len(), self.config.numeric_dim);
        debug_assert_eq!(self.numeric_total.len(), self.config.numeric_dim);
        debug_assert_eq!(self.min_numeric.len(), self.config.numeric_dim);
        debug_assert_eq!(self.max_numeric.len(), self.config.numeric_dim);
        debug_assert_eq!(self.categorical_score.len(), self.config.categorical_dim);
        debug_assert_eq!(self.categorical_total.len(), self.config.categorical_dim);

        if timestamp == 0 {
            return Err(RcfError::InvalidArgument("timestamp must be > 0".into()));
        }

        match self.current_time {
            None => {
                self.current_time = Some(timestamp);
                self.current_tick = 1;
            }
            Some(t) if timestamp > t => {
                let tick_gap = timestamp - t;
                self.lower_current_counts(self.config.alpha.powf(tick_gap as f64));
                self.current_time = Some(timestamp);
                self.current_tick += tick_gap;
            }
            Some(t) if timestamp < t => {
                return Err(RcfError::InvalidArgument(format!(
                    "timestamps must be non-decreasing: previous={t}, got={timestamp}"
                )));
            }
            _ => {}
        }

        let cur_t = self.current_tick.max(1);
        let mut normalized = vec![0.0_f64; self.config.numeric_dim];
        let mut record_numeric = vec![0.0_f64; self.config.numeric_dim];
        let mut sum = 0.0_f64;

        for i in 0..self.config.numeric_dim {
            let raw = f64::from(numeric[i]);
            if raw <= -1.0 {
                return Err(RcfError::InvalidArgument(
                    "numeric value must be > -1.0 for log10(1+x) transform".into(),
                ));
            }

            record_numeric[i] = raw;

            let transformed = (1.0 + raw).log10();
            if self.entries_seen == 0 {
                self.min_numeric[i] = transformed;
                self.max_numeric[i] = transformed;
                normalized[i] = 0.0;
            } else {
                if transformed < self.min_numeric[i] {
                    self.min_numeric[i] = transformed;
                }
                if transformed > self.max_numeric[i] {
                    self.max_numeric[i] = transformed;
                }

                let span = self.max_numeric[i] - self.min_numeric[i];
                normalized[i] = if span <= f64::EPSILON {
                    0.0
                } else {
                    (transformed - self.min_numeric[i]) / span
                };
            }

            self.numeric_score[i].insert(normalized[i], 1.0);
            self.numeric_total[i].insert(normalized[i], 1.0);

            let t = counts_to_anom(
                self.numeric_total[i].get_count(normalized[i]),
                self.numeric_score[i].get_count(normalized[i]),
                cur_t,
            );
            sum += t;
        }

        self.cur_count.insert(&record_numeric, categorical, 1.0);
        self.total_count.insert(&record_numeric, categorical, 1.0);

        for i in 0..self.config.categorical_dim {
            let v = categorical[i];
            self.categorical_score[i].insert(v, 1.0);
            self.categorical_total[i].insert(v, 1.0);

            let t = counts_to_anom(
                self.categorical_total[i].get_count(v),
                self.categorical_score[i].get_count(v),
                cur_t,
            );
            sum += t;
        }

        let record_score = counts_to_anom(
            self.total_count.get_count(&record_numeric, categorical),
            self.cur_count.get_count(&record_numeric, categorical),
            cur_t,
        );
        sum += record_score;

        self.entries_seen += 1;
        Ok(sum)
    }

    /// mStream computes score online; this method is an alias to
    /// [`MStream::update_and_score`].
    pub fn score(&mut self, numeric: &[f32], categorical: &[i64], timestamp: u64) -> Result<f64> {
        self.update_and_score(numeric, categorical, timestamp)
    }

    /// Update detector state without using the score.
    pub fn update(&mut self, numeric: &[f32], categorical: &[i64], timestamp: u64) -> Result<()> {
        let _ = self.update_and_score(numeric, categorical, timestamp)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Save / Load
    // -----------------------------------------------------------------------

    /// Serialise the entire mStream state to a JSON string.
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| RcfError::Io(e.to_string()))
    }

    /// Deserialise mStream state from a JSON string previously written by
    /// [`to_json`].
    #[cfg(feature = "serde")]
    pub fn from_json(json: impl AsRef<[u8]>) -> Result<Self> {
        serde_json::from_slice(json.as_ref()).map_err(|e| RcfError::Io(e.to_string()))
    }

    /// Serialise the entire mStream state to a JSON file.
    #[cfg(all(feature = "serde", feature = "std"))]
    pub fn save_json(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path.as_ref(), json).map_err(|e| RcfError::Io(e.to_string()))
    }

    /// Deserialise mStream state from a JSON file previously written by
    /// [`save_json`].
    #[cfg(all(feature = "serde", feature = "std"))]
    pub fn load_json(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let data = std::fs::read(path.as_ref()).map_err(|e| RcfError::Io(e.to_string()))?;
        Self::from_json(&data)
    }

    fn validate_record(&self, numeric: &[f32], categorical: &[i64]) -> Result<()> {
        if numeric.len() != self.config.numeric_dim {
            return Err(RcfError::DimensionMismatch {
                expected: self.config.numeric_dim,
                got: numeric.len(),
            });
        }
        if categorical.len() != self.config.categorical_dim {
            return Err(RcfError::DimensionMismatch {
                expected: self.config.categorical_dim,
                got: categorical.len(),
            });
        }
        Ok(())
    }

    fn lower_current_counts(&mut self, factor: f64) {
        debug_assert!(factor.is_finite());
        debug_assert!((0.0..=1.0).contains(&factor));

        self.cur_count.lower(factor);
        for sketch in &mut self.numeric_score {
            sketch.lower(factor);
        }
        for sketch in &mut self.categorical_score {
            sketch.lower(factor);
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

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
        assert!((score - 10.0).abs() < 1e-12);
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
            assert!((sa - sb).abs() < 1e-12);
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
            assert!((lhs - rhs).abs() < 1e-12);
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

        let count = d.numeric_score[0].get_count(0.0);
        let expected = 1.0 + alpha.powi(3);
        assert!((count - expected).abs() < 1e-12);
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
            d.categorical_total[0].get_count(2),
            d.categorical_score[0].get_count(2),
            2,
        ) + counts_to_anom(
            d.total_count.get_count(&[], &[2]),
            d.cur_count.get_count(&[], &[2]),
            2,
        );
        assert!((score - expected).abs() < 1e-12);
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
        assert_eq!(restored.config().num_rows, d.config().num_rows);
        assert_eq!(restored.config().num_buckets, d.config().num_buckets);
        assert_eq!(restored.config().numeric_dim, d.config().numeric_dim);
        assert_eq!(
            restored.config().categorical_dim,
            d.config().categorical_dim
        );
    }
}
