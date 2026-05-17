#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{RcfError, Result};

/// Configuration for an [`MStream`](super::MStream) detector.
///
/// Values are validated when a detector is built. Use the `with_*` methods to
/// override defaults while keeping the fields themselves read-only.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MStreamConfig {
    /// Number of numerical aspects in each record.
    numeric_dim: usize,
    /// Number of categorical aspects in each record.
    categorical_dim: usize,
    /// Number of hash rows.
    num_rows: usize,
    /// Number of buckets per hash row.
    num_buckets: usize,
    /// Temporal decay factor in `(0, 1)`.
    alpha: f64,
}

impl MStreamConfig {
    /// Create a config with default hash parameters.
    pub fn new(numeric_dim: usize, categorical_dim: usize) -> Self {
        Self {
            numeric_dim,
            categorical_dim,
            num_rows: 2,
            num_buckets: 1024,
            alpha: 0.8,
        }
    }

    /// Set the number of hash rows.
    pub fn with_num_rows(mut self, value: usize) -> Self {
        self.num_rows = value;
        self
    }

    /// Set the number of buckets per hash row.
    pub fn with_num_buckets(mut self, value: usize) -> Self {
        self.num_buckets = value;
        self
    }

    /// Set the temporal decay factor.
    pub fn with_alpha(mut self, value: f64) -> Self {
        self.alpha = value;
        self
    }

    /// Number of numerical aspects in each record.
    pub fn numeric_dim(&self) -> usize {
        self.numeric_dim
    }

    /// Number of categorical aspects in each record.
    pub fn categorical_dim(&self) -> usize {
        self.categorical_dim
    }

    /// Number of hash rows.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Number of buckets per hash row.
    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }

    /// Temporal decay factor.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.numeric_dim == 0 && self.categorical_dim == 0 {
            return Err(RcfError::InvalidArgument(
                "at least one of numeric_dim or categorical_dim must be > 0".into(),
            ));
        }
        if self.num_rows == 0 {
            return Err(RcfError::InvalidArgument("num_rows must be > 0".into()));
        }
        if self.num_buckets < 2 {
            return Err(RcfError::InvalidArgument("num_buckets must be >= 2".into()));
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 || self.alpha >= 1.0 {
            return Err(RcfError::InvalidArgument(
                "alpha must be in range (0, 1)".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::error::RcfError;

    use super::*;

    #[test]
    fn rejects_empty_dimensions() {
        let err = MStreamConfig::new(0, 0).validate().unwrap_err();
        assert!(matches!(err, RcfError::InvalidArgument(_)));
    }

    #[rstest]
    #[case(0.0)]
    #[case(1.0)]
    #[case(f64::NAN)]
    fn rejects_invalid_alpha(#[case] alpha: f64) {
        let err = MStreamConfig::new(1, 0)
            .with_alpha(alpha)
            .validate()
            .unwrap_err();
        assert!(matches!(err, RcfError::InvalidArgument(_)));
    }
}
