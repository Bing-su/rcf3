#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{RcfError, Result};

/// Hyperparameters for a Random Cut Forest.
///
/// Use [`RcfConfig::new`] then chain the builder methods, or deserialise from JSON.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RcfConfig {
    /// Number of base feature dimensions per observation (before shingling).
    input_dim: usize,

    /// Temporal window size. When `internal_shingling` is true the forest
    /// maintains a rolling buffer and the effective model dimension is
    /// `input_dim * shingle_size`.
    #[cfg_attr(feature = "serde", serde(default = "default_shingle_size"))]
    shingle_size: usize,

    /// Maximum number of points stored per tree.
    #[cfg_attr(feature = "serde", serde(default = "default_capacity"))]
    capacity: usize,

    /// Number of trees in the forest.
    #[cfg_attr(feature = "serde", serde(default = "default_num_trees"))]
    num_trees: usize,

    /// Exponential time-decay rate applied to sampling weights.
    /// `0.0` means "use the default `0.1 / capacity`".
    #[cfg_attr(feature = "serde", serde(default))]
    time_decay: f64,

    /// Minimum number of updates before `score` / `attribution` / etc. return
    /// non-trivial results.  `0` means "use `1 + capacity / 4`".
    #[cfg_attr(feature = "serde", serde(default))]
    output_after: usize,

    /// When true the forest manages the shingle buffer automatically so callers
    /// pass one base observation at a time.
    #[cfg_attr(feature = "serde", serde(default = "default_internal_shingling"))]
    internal_shingling: bool,

    /// Controls how quickly the sampler fills to capacity during warm-up.
    #[cfg_attr(feature = "serde", serde(default = "default_initial_accept_fraction"))]
    initial_accept_fraction: f64,
}

fn default_shingle_size() -> usize {
    1
}
fn default_capacity() -> usize {
    256
}
fn default_num_trees() -> usize {
    50
}
fn default_internal_shingling() -> bool {
    true
}
fn default_initial_accept_fraction() -> f64 {
    0.125
}

impl RcfConfig {
    /// Create a config with defaults for all optional parameters.
    pub fn new(input_dim: usize) -> Self {
        Self {
            input_dim,
            shingle_size: default_shingle_size(),
            capacity: default_capacity(),
            num_trees: default_num_trees(),
            time_decay: 0.0,
            output_after: 0,
            internal_shingling: default_internal_shingling(),
            initial_accept_fraction: default_initial_accept_fraction(),
        }
    }

    /// Set the temporal window size. `1` disables shingling.
    pub fn with_shingle_size(mut self, v: usize) -> Self {
        self.shingle_size = v;
        self
    }

    /// Set the maximum number of points retained per tree.
    pub fn with_capacity(mut self, v: usize) -> Self {
        self.capacity = v;
        self
    }

    /// Set the number of trees in the forest ensemble.
    pub fn with_num_trees(mut self, v: usize) -> Self {
        self.num_trees = v;
        self
    }

    /// Set the exponential time-decay rate for sampling weights.
    ///
    /// Use `0.0` to keep the default behavior (`0.1 / capacity`).
    pub fn with_time_decay(mut self, v: f64) -> Self {
        self.time_decay = v;
        self
    }

    /// Set the minimum updates before non-trivial scores are returned.
    ///
    /// Use `0` to keep the default behavior (`1 + capacity / 4`).
    pub fn with_output_after(mut self, v: usize) -> Self {
        self.output_after = v;
        self
    }

    /// Enable or disable internal shingle buffer management.
    pub fn with_internal_shingling(mut self, v: bool) -> Self {
        self.internal_shingling = v;
        self
    }

    /// Set the warm-up acceptance fraction for the sampler.
    pub fn with_initial_accept_fraction(mut self, v: f64) -> Self {
        self.initial_accept_fraction = v;
        self
    }

    /// Number of base feature dimensions per observation.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Temporal window size.
    pub fn shingle_size(&self) -> usize {
        self.shingle_size
    }

    /// Maximum number of points stored per tree.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of trees in the forest.
    pub fn num_trees(&self) -> usize {
        self.num_trees
    }

    /// Configured exponential time-decay rate.
    pub fn time_decay(&self) -> f64 {
        self.time_decay
    }

    /// Configured minimum number of updates before non-trivial outputs.
    pub fn output_after(&self) -> usize {
        self.output_after
    }

    /// Whether the forest manages the shingle buffer automatically.
    pub fn internal_shingling(&self) -> bool {
        self.internal_shingling
    }

    /// Warm-up acceptance fraction for the sampler.
    pub fn initial_accept_fraction(&self) -> f64 {
        self.initial_accept_fraction
    }

    /// Effective time-decay (resolves the `0.0 → default` convention).
    pub fn effective_time_decay(&self) -> f64 {
        if self.time_decay == 0.0 {
            0.1 / self.capacity as f64
        } else {
            self.time_decay
        }
    }

    /// Effective output threshold (resolves the `0 → default` convention).
    pub fn effective_output_after(&self) -> usize {
        if self.output_after == 0 {
            1 + self.capacity / 4
        } else {
            self.output_after
        }
    }

    /// Full dimensionality seen by each tree.
    pub fn dim(&self) -> usize {
        self.input_dim * self.shingle_size
    }

    pub(in crate::rcf) fn validate(&self) -> Result<()> {
        if self.input_dim == 0 {
            return Err(RcfError::InvalidArgument("input_dim must be > 0".into()));
        }
        if self.shingle_size == 0 {
            return Err(RcfError::InvalidArgument("shingle_size must be > 0".into()));
        }
        if self.input_dim.checked_mul(self.shingle_size).is_none() {
            return Err(RcfError::InvalidArgument(
                "input_dim * shingle_size overflows usize".into(),
            ));
        }
        if self.capacity == 0 {
            return Err(RcfError::InvalidArgument("capacity must be > 0".into()));
        }
        if self.num_trees == 0 {
            return Err(RcfError::InvalidArgument("num_trees must be > 0".into()));
        }
        if !self.time_decay.is_finite() || self.time_decay < 0.0 {
            return Err(RcfError::InvalidArgument(
                "time_decay must be finite and >= 0.0".into(),
            ));
        }
        if !self.initial_accept_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.initial_accept_fraction)
        {
            return Err(RcfError::InvalidArgument(
                "initial_accept_fraction must be finite and in [0.0, 1.0]".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::error::RcfError;
    use proptest::prelude::*;
    use rstest::rstest;

    proptest! {
        #[test]
        fn dim_equals_input_times_shingle(
            input_dim in 1usize..=32,
            shingle_size in 1usize..=16,
        ) {
            let cfg = RcfConfig::new(input_dim).with_shingle_size(shingle_size);
            prop_assert_eq!(cfg.dim(), input_dim * shingle_size);
        }

        #[test]
        fn effective_time_decay_positive(capacity in 1usize..=1000) {
            // time_decay == 0.0 triggers the default formula: 0.1 / capacity
            let cfg = RcfConfig::new(1).with_capacity(capacity);
            prop_assert!(cfg.effective_time_decay() > 0.0);
        }

        #[test]
        fn effective_output_after_positive(capacity in 1usize..=1000) {
            // output_after == 0 triggers the default formula: 1 + capacity/4
            let cfg = RcfConfig::new(1).with_capacity(capacity).with_output_after(0);
            prop_assert!(cfg.effective_output_after() >= 1);
        }

        #[test]
        fn setters_reflect_values(n in 1usize..=100) {
            let cfg = RcfConfig::new(1).with_num_trees(n);
            prop_assert_eq!(cfg.num_trees(), n);
        }
    }

    #[test]
    fn validate_accepts_default_config() {
        RcfConfig::new(1).validate().unwrap();
    }

    #[test]
    fn getters_reflect_all_config_fields() {
        let config = RcfConfig::new(3)
            .with_shingle_size(4)
            .with_capacity(128)
            .with_num_trees(17)
            .with_time_decay(0.25)
            .with_output_after(9)
            .with_internal_shingling(false)
            .with_initial_accept_fraction(0.5);

        assert_eq!(config.input_dim(), 3);
        assert_eq!(config.shingle_size(), 4);
        assert_eq!(config.capacity(), 128);
        assert_eq!(config.num_trees(), 17);
        assert_eq!(config.time_decay(), 0.25);
        assert_eq!(config.output_after(), 9);
        assert!(!config.internal_shingling());
        assert_eq!(config.initial_accept_fraction(), 0.5);
        assert_eq!(config.dim(), 12);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_preserves_private_field_wire_shape_and_defaults() {
        let config = RcfConfig::new(3)
            .with_shingle_size(4)
            .with_capacity(128)
            .with_num_trees(17)
            .with_time_decay(0.25)
            .with_output_after(9)
            .with_internal_shingling(false)
            .with_initial_accept_fraction(0.5);

        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(value["input_dim"], 3);
        assert_eq!(value["shingle_size"], 4);
        assert_eq!(value["capacity"], 128);
        assert_eq!(value["num_trees"], 17);
        assert_eq!(value["time_decay"], 0.25);
        assert_eq!(value["output_after"], 9);
        assert_eq!(value["internal_shingling"], false);
        assert_eq!(value["initial_accept_fraction"], 0.5);

        let minimal: RcfConfig = serde_json::from_str(r#"{"input_dim":2}"#).unwrap();
        assert_eq!(minimal.input_dim(), 2);
        assert_eq!(minimal.shingle_size(), 1);
        assert_eq!(minimal.capacity(), 256);
        assert_eq!(minimal.num_trees(), 50);
        assert_eq!(minimal.time_decay(), 0.0);
        assert_eq!(minimal.output_after(), 0);
        assert!(minimal.internal_shingling());
        assert_eq!(minimal.initial_accept_fraction(), 0.125);
    }

    #[rstest]
    #[case::zero_input_dim(RcfConfig::new(0), "input_dim")]
    #[case::zero_shingle_size(RcfConfig::new(1).with_shingle_size(0), "shingle_size")]
    #[case::zero_capacity(RcfConfig::new(1).with_capacity(0), "capacity")]
    #[case::zero_num_trees(RcfConfig::new(1).with_num_trees(0), "num_trees")]
    #[case::negative_time_decay(RcfConfig::new(1).with_time_decay(-0.1), "time_decay")]
    #[case::nan_time_decay(RcfConfig::new(1).with_time_decay(f64::NAN), "time_decay")]
    #[case::infinite_time_decay(
        RcfConfig::new(1).with_time_decay(f64::INFINITY),
        "time_decay"
    )]
    #[case::negative_initial_accept_fraction(
        RcfConfig::new(1).with_initial_accept_fraction(-0.1),
        "initial_accept_fraction"
    )]
    #[case::nan_initial_accept_fraction(
        RcfConfig::new(1).with_initial_accept_fraction(f64::NAN),
        "initial_accept_fraction"
    )]
    #[case::infinite_initial_accept_fraction(
        RcfConfig::new(1).with_initial_accept_fraction(f64::INFINITY),
        "initial_accept_fraction"
    )]
    #[case::too_large_initial_accept_fraction(
        RcfConfig::new(1).with_initial_accept_fraction(1.1),
        "initial_accept_fraction"
    )]
    fn validate_rejects_invalid_core_fields(
        #[case] config: RcfConfig,
        #[case] expected_message: &str,
    ) {
        let err = config.validate().unwrap_err();

        assert!(
            matches!(err, RcfError::InvalidArgument(ref msg) if msg.contains(expected_message)),
            "unexpected error variant: {err:?}"
        );
    }

    #[rstest]
    #[case::zero_time_decay_and_zero_initial_fraction(
        RcfConfig::new(1)
            .with_time_decay(0.0)
            .with_initial_accept_fraction(0.0)
    )]
    #[case::positive_time_decay_and_full_initial_fraction(
        RcfConfig::new(1)
            .with_time_decay(0.1)
            .with_initial_accept_fraction(1.0)
    )]
    fn validate_accepts_float_boundaries(#[case] config: RcfConfig) {
        config.validate().unwrap();
    }
}
