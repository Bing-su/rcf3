#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{RcfError, Result};

/// Configuration for an [`OnlineIForest`](super::OnlineIForest) detector.
///
/// Values are validated when the detector is built. Use the `with_*` methods to
/// override defaults while keeping the fields themselves read-only.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OnlineIForestConfig {
    input_dim: usize,
    num_trees: usize,
    window_size: usize,
    max_leaf_samples: usize,
}

impl OnlineIForestConfig {
    /// Create a config with the paper's default hyperparameters.
    pub fn new(input_dim: usize) -> Self {
        Self {
            input_dim,
            num_trees: 32,
            window_size: 2048,
            max_leaf_samples: 32,
        }
    }

    /// Set the number of trees in the ensemble.
    pub fn with_num_trees(mut self, value: usize) -> Self {
        self.num_trees = value;
        self
    }

    /// Set the number of recent points retained by the sliding window.
    pub fn with_window_size(mut self, value: usize) -> Self {
        self.window_size = value;
        self
    }

    /// Set the base leaf-splitting threshold.
    pub fn with_max_leaf_samples(mut self, value: usize) -> Self {
        self.max_leaf_samples = value;
        self
    }

    /// Number of numerical features in each point.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Number of trees in the ensemble.
    pub fn num_trees(&self) -> usize {
        self.num_trees
    }

    /// Number of recent points retained by the sliding window.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Base leaf-splitting threshold.
    pub fn max_leaf_samples(&self) -> usize {
        self.max_leaf_samples
    }

    /// Maximum split depth from the paper: `δ = log2(window_size / max_leaf_samples)`.
    ///
    /// Keep the fractional value intact. A non-integer `δ` still allows any
    /// integer depth `k` satisfying `k < δ`.
    pub(crate) fn depth_limit(&self) -> f64 {
        self.normalization_factor()
    }

    /// Forest-level score normalizer `c(window_size, max_leaf_samples)`.
    pub(crate) fn normalization_factor(&self) -> f64 {
        libm::log2(self.window_size as f64 / self.max_leaf_samples as f64)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.input_dim == 0 {
            return Err(RcfError::InvalidArgument("input_dim must be > 0".into()));
        }
        if self.num_trees == 0 {
            return Err(RcfError::InvalidArgument("num_trees must be > 0".into()));
        }
        if self.window_size == 0 {
            return Err(RcfError::InvalidArgument("window_size must be > 0".into()));
        }
        if self.window_size.checked_add(1).is_none() {
            return Err(RcfError::InvalidArgument("window_size is too large".into()));
        }
        if self.max_leaf_samples == 0 {
            return Err(RcfError::InvalidArgument(
                "max_leaf_samples must be > 0".into(),
            ));
        }
        if self.window_size <= self.max_leaf_samples {
            return Err(RcfError::InvalidArgument(
                "window_size must be greater than max_leaf_samples".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn defaults_match_paper_configuration() {
        let config = OnlineIForestConfig::new(3);
        assert_eq!(config.input_dim(), 3);
        assert_eq!(config.num_trees(), 32);
        assert_eq!(config.window_size(), 2048);
        assert_eq!(config.max_leaf_samples(), 32);
    }

    #[rstest]
    #[case::zero_input_dim(OnlineIForestConfig::new(0))]
    #[case::zero_trees(OnlineIForestConfig::new(1).with_num_trees(0))]
    #[case::zero_window(OnlineIForestConfig::new(1).with_window_size(0))]
    #[case::large_window(OnlineIForestConfig::new(1).with_window_size(usize::MAX))]
    #[case::zero_leaf_samples(OnlineIForestConfig::new(1).with_max_leaf_samples(0))]
    #[case::window_not_larger_than_leaf_samples(
        OnlineIForestConfig::new(1)
            .with_window_size(32)
            .with_max_leaf_samples(32)
    )]
    fn rejects_invalid_values(#[case] config: OnlineIForestConfig) {
        assert!(matches!(
            config.validate(),
            Err(RcfError::InvalidArgument(_))
        ));
    }

    #[test]
    fn depth_limit_preserves_fractional_values() {
        let config = OnlineIForestConfig::new(1)
            .with_window_size(10)
            .with_max_leaf_samples(3);
        assert!(config.depth_limit() > 1.0);
        assert!(config.depth_limit() < 2.0);
    }
}
