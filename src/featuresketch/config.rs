#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{RcfError, Result};

const MAX_CHAIN_DEPTH_FOR_POSITIVE_WIDTH: usize = 1024;

/// Configuration for a [`FeatureSketch`](super::FeatureSketch) detector.
///
/// Values are validated when a detector is built. Use the `with_*` methods to
/// override defaults while keeping the fields themselves read-only.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FeatureSketchConfig {
    value_projection_dims: usize,
    presence_projection_dims: usize,
    chains_per_ensemble: usize,
    chain_depth: usize,
    sketch_rows: usize,
    sketch_buckets: usize,
    decay_half_life: u64,
}

impl FeatureSketchConfig {
    /// Create a config with the documented FeatureSketch defaults.
    pub fn new() -> Self {
        Self {
            value_projection_dims: 32,
            presence_projection_dims: 32,
            chains_per_ensemble: 16,
            chain_depth: 8,
            sketch_rows: 2,
            sketch_buckets: 2048,
            decay_half_life: 2048,
        }
    }

    pub fn with_value_projection_dims(mut self, value: usize) -> Self {
        self.value_projection_dims = value;
        self
    }

    pub fn with_presence_projection_dims(mut self, value: usize) -> Self {
        self.presence_projection_dims = value;
        self
    }

    pub fn with_chains_per_ensemble(mut self, value: usize) -> Self {
        self.chains_per_ensemble = value;
        self
    }

    pub fn with_chain_depth(mut self, value: usize) -> Self {
        self.chain_depth = value;
        self
    }

    pub fn with_sketch_rows(mut self, value: usize) -> Self {
        self.sketch_rows = value;
        self
    }

    pub fn with_sketch_buckets(mut self, value: usize) -> Self {
        self.sketch_buckets = value;
        self
    }

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
    #[case(FeatureSketchConfig::new().with_value_projection_dims(1))]
    #[case(FeatureSketchConfig::new().with_presence_projection_dims(1))]
    #[case(FeatureSketchConfig::new().with_chains_per_ensemble(1))]
    #[case(FeatureSketchConfig::new().with_chain_depth(1))]
    #[case(FeatureSketchConfig::new().with_chain_depth(MAX_CHAIN_DEPTH_FOR_POSITIVE_WIDTH))]
    #[case(FeatureSketchConfig::new().with_sketch_rows(1))]
    #[case(FeatureSketchConfig::new().with_sketch_buckets(2))]
    #[case(FeatureSketchConfig::new().with_decay_half_life(1))]
    fn accepts_minimum_valid_boundaries(#[case] config: FeatureSketchConfig) {
        config.validate().unwrap();
    }

    #[rstest]
    #[case::value_dims(FeatureSketchConfig::new().with_value_projection_dims(0))]
    #[case::presence_dims(FeatureSketchConfig::new().with_presence_projection_dims(0))]
    #[case::chains(FeatureSketchConfig::new().with_chains_per_ensemble(0))]
    #[case::depth(FeatureSketchConfig::new().with_chain_depth(0))]
    #[case::rows(FeatureSketchConfig::new().with_sketch_rows(0))]
    #[case::buckets(FeatureSketchConfig::new().with_sketch_buckets(1))]
    #[case::half_life(FeatureSketchConfig::new().with_decay_half_life(0))]
    #[case::too_deep(
        FeatureSketchConfig::new()
            .with_chain_depth(MAX_CHAIN_DEPTH_FOR_POSITIVE_WIDTH + 1)
    )]
    fn rejects_invalid_values(#[case] config: FeatureSketchConfig) {
        assert!(matches!(
            config.validate(),
            Err(RcfError::InvalidArgument(_))
        ));
    }

    #[rstest]
    #[case::presence_feature_count_dimension(
        FeatureSketchConfig::new().with_presence_projection_dims(usize::MAX)
    )]
    #[case::chain_layout_size(
        FeatureSketchConfig::new()
            .with_chains_per_ensemble(usize::MAX)
            .with_chain_depth(2)
    )]
    #[case::sketch_cell_count(
        FeatureSketchConfig::new()
            .with_sketch_rows(usize::MAX)
            .with_sketch_buckets(2)
    )]
    #[case::total_sketch_cells(
        FeatureSketchConfig::new()
            .with_chains_per_ensemble(usize::MAX / 8 + 1)
            .with_chain_depth(2)
            .with_sketch_rows(2)
            .with_sketch_buckets(2)
    )]
    fn rejects_overflowing_values(#[case] config: FeatureSketchConfig) {
        assert!(matches!(
            config.validate(),
            Err(RcfError::InvalidArgument(_))
        ));
    }
}
