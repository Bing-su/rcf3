#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{RcfError, Result};

const MAX_CHAIN_DEPTH_FOR_POSITIVE_WIDTH: usize = 1024;

/// Configuration for a [`FeatureSketch`](super::FeatureSketch) detector.
///
/// Values are validated when a detector is built. Use [`Self::builder`] to
/// override defaults while keeping the fields themselves read-only.
///
/// ```rust
/// use rcf3::FeatureSketchConfig;
/// let config = FeatureSketchConfig::builder().sketch_buckets(256).build();
/// ```
#[derive(Clone, Debug, bon::Builder)]
#[builder(derive(Clone, Debug))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FeatureSketchConfig {
    #[builder(default = FeatureSketchConfig::default_value_projection_dims())]
    value_projection_dims: usize,
    #[builder(default = FeatureSketchConfig::default_presence_projection_dims())]
    presence_projection_dims: usize,
    #[builder(default = FeatureSketchConfig::default_chains_per_ensemble())]
    chains_per_ensemble: usize,
    #[builder(default = FeatureSketchConfig::default_chain_depth())]
    chain_depth: usize,
    #[builder(default = FeatureSketchConfig::default_sketch_rows())]
    sketch_rows: usize,
    #[builder(default = FeatureSketchConfig::default_sketch_buckets())]
    sketch_buckets: usize,
    #[builder(default = FeatureSketchConfig::default_decay_half_life())]
    decay_half_life: u64,
}

impl FeatureSketchConfig {
    /// Default value projection dimension.
    pub const fn default_value_projection_dims() -> usize {
        32
    }

    /// Default presence projection dimension.
    pub const fn default_presence_projection_dims() -> usize {
        32
    }

    /// Default number of chains per ensemble.
    pub const fn default_chains_per_ensemble() -> usize {
        16
    }

    /// Default chain depth.
    pub const fn default_chain_depth() -> usize {
        8
    }

    /// Default number of sketch rows.
    pub const fn default_sketch_rows() -> usize {
        2
    }

    /// Default number of buckets per sketch row.
    pub const fn default_sketch_buckets() -> usize {
        2048
    }

    /// Default event-count decay half-life.
    pub const fn default_decay_half_life() -> u64 {
        2048
    }

    /// Create a config with the documented FeatureSketch defaults.
    pub fn new() -> Self {
        Self::builder().build()
    }

    #[deprecated(note = "Use FeatureSketchConfig::builder() instead.")]
    pub fn with_value_projection_dims(mut self, value: usize) -> Self {
        self.value_projection_dims = value;
        self
    }

    #[deprecated(note = "Use FeatureSketchConfig::builder() instead.")]
    pub fn with_presence_projection_dims(mut self, value: usize) -> Self {
        self.presence_projection_dims = value;
        self
    }

    #[deprecated(note = "Use FeatureSketchConfig::builder() instead.")]
    pub fn with_chains_per_ensemble(mut self, value: usize) -> Self {
        self.chains_per_ensemble = value;
        self
    }

    #[deprecated(note = "Use FeatureSketchConfig::builder() instead.")]
    pub fn with_chain_depth(mut self, value: usize) -> Self {
        self.chain_depth = value;
        self
    }

    #[deprecated(note = "Use FeatureSketchConfig::builder() instead.")]
    pub fn with_sketch_rows(mut self, value: usize) -> Self {
        self.sketch_rows = value;
        self
    }

    #[deprecated(note = "Use FeatureSketchConfig::builder() instead.")]
    pub fn with_sketch_buckets(mut self, value: usize) -> Self {
        self.sketch_buckets = value;
        self
    }

    #[deprecated(note = "Use FeatureSketchConfig::builder() instead.")]
    pub fn with_decay_half_life(mut self, value: u64) -> Self {
        self.decay_half_life = value;
        self
    }

    pub fn value_projection_dims(&self) -> usize {
        self.value_projection_dims
    }

    pub fn presence_projection_dims(&self) -> usize {
        self.presence_projection_dims
    }

    pub fn chains_per_ensemble(&self) -> usize {
        self.chains_per_ensemble
    }

    pub fn chain_depth(&self) -> usize {
        self.chain_depth
    }

    pub fn sketch_rows(&self) -> usize {
        self.sketch_rows
    }

    pub fn sketch_buckets(&self) -> usize {
        self.sketch_buckets
    }

    pub fn decay_half_life(&self) -> u64 {
        self.decay_half_life
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.value_projection_dims == 0 {
            return Err(RcfError::InvalidArgument(
                "value_projection_dims must be > 0".into(),
            ));
        }
        if self.presence_projection_dims == 0 {
            return Err(RcfError::InvalidArgument(
                "presence_projection_dims must be > 0".into(),
            ));
        }
        if self.chains_per_ensemble == 0 {
            return Err(RcfError::InvalidArgument(
                "chains_per_ensemble must be > 0".into(),
            ));
        }
        if self.chain_depth == 0 {
            return Err(RcfError::InvalidArgument("chain_depth must be > 0".into()));
        }
        if self.chain_depth > MAX_CHAIN_DEPTH_FOR_POSITIVE_WIDTH {
            return Err(RcfError::InvalidArgument(
                "chain_depth is too large to keep chain bin widths positive".into(),
            ));
        }
        if self.sketch_rows == 0 {
            return Err(RcfError::InvalidArgument("sketch_rows must be > 0".into()));
        }
        if self.sketch_buckets < 2 {
            return Err(RcfError::InvalidArgument(
                "sketch_buckets must be >= 2".into(),
            ));
        }
        if self.decay_half_life == 0 {
            return Err(RcfError::InvalidArgument(
                "decay_half_life must be > 0".into(),
            ));
        }
        self.presence_projection_dims
            .checked_add(1)
            .ok_or_else(|| {
                RcfError::InvalidArgument(
                    "presence_projection_dims plus the feature-count signal must fit in usize"
                        .into(),
                )
            })?;
        self.chains_per_ensemble
            .checked_mul(self.chain_depth)
            .ok_or_else(|| {
                RcfError::InvalidArgument(
                    "chains_per_ensemble * chain_depth must fit in usize".into(),
                )
            })?;
        self.sketch_rows
            .checked_mul(self.sketch_buckets)
            .ok_or_else(|| {
                RcfError::InvalidArgument("sketch_rows * sketch_buckets must fit in usize".into())
            })?;
        self.chains_per_ensemble
            .checked_mul(self.chain_depth)
            .and_then(|value| value.checked_mul(self.sketch_rows))
            .and_then(|value| value.checked_mul(self.sketch_buckets))
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                RcfError::InvalidArgument(
                    "total FeatureSketch sketch cells must fit in usize".into(),
                )
            })?;
        Ok(())
    }
}

impl Default for FeatureSketchConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::error::RcfError;

    use super::*;

    #[test]
    fn defaults_match_research_document() {
        let config = FeatureSketchConfig::new();
        assert_eq!(config.value_projection_dims(), 32);
        assert_eq!(config.presence_projection_dims(), 32);
        assert_eq!(config.chains_per_ensemble(), 16);
        assert_eq!(config.chain_depth(), 8);
        assert_eq!(config.sketch_rows(), 2);
        assert_eq!(config.sketch_buckets(), 2048);
        assert_eq!(config.decay_half_life(), 2048);
    }

    #[rstest]
    #[case(FeatureSketchConfig::builder().value_projection_dims(1).build())]
    #[case(FeatureSketchConfig::builder().presence_projection_dims(1).build())]
    #[case(FeatureSketchConfig::builder().chains_per_ensemble(1).build())]
    #[case(FeatureSketchConfig::builder().chain_depth(1).build())]
    #[case(FeatureSketchConfig::builder().chain_depth(MAX_CHAIN_DEPTH_FOR_POSITIVE_WIDTH).build())]
    #[case(FeatureSketchConfig::builder().sketch_rows(1).build())]
    #[case(FeatureSketchConfig::builder().sketch_buckets(2).build())]
    #[case(FeatureSketchConfig::builder().decay_half_life(1).build())]
    fn accepts_minimum_valid_boundaries(#[case] config: FeatureSketchConfig) {
        config.validate().unwrap();
    }

    #[rstest]
    #[case::value_dims(FeatureSketchConfig::builder().value_projection_dims(0).build())]
    #[case::presence_dims(FeatureSketchConfig::builder().presence_projection_dims(0).build())]
    #[case::chains(FeatureSketchConfig::builder().chains_per_ensemble(0).build())]
    #[case::depth(FeatureSketchConfig::builder().chain_depth(0).build())]
    #[case::rows(FeatureSketchConfig::builder().sketch_rows(0).build())]
    #[case::buckets(FeatureSketchConfig::builder().sketch_buckets(1).build())]
    #[case::half_life(FeatureSketchConfig::builder().decay_half_life(0).build())]
    #[case::too_deep(
        FeatureSketchConfig::builder()
            .chain_depth(MAX_CHAIN_DEPTH_FOR_POSITIVE_WIDTH + 1)
            .build()
    )]
    fn rejects_invalid_values(#[case] config: FeatureSketchConfig) {
        assert!(matches!(
            config.validate(),
            Err(RcfError::InvalidArgument(_))
        ));
    }

    #[rstest]
    #[case::presence_feature_count_dimension(
        FeatureSketchConfig::builder().presence_projection_dims(usize::MAX).build()
    )]
    #[case::chain_layout_size(
        FeatureSketchConfig::builder()
            .chains_per_ensemble(usize::MAX)
            .chain_depth(2)
            .build()
    )]
    #[case::sketch_cell_count(
        FeatureSketchConfig::builder()
            .sketch_rows(usize::MAX)
            .sketch_buckets(2)
            .build()
    )]
    #[case::total_sketch_cells(
        FeatureSketchConfig::builder()
            .chains_per_ensemble(usize::MAX / 8 + 1)
            .chain_depth(2)
            .sketch_rows(2)
            .sketch_buckets(2)
            .build()
    )]
    fn rejects_overflowing_values(#[case] config: FeatureSketchConfig) {
        assert!(matches!(
            config.validate(),
            Err(RcfError::InvalidArgument(_))
        ));
    }
}
