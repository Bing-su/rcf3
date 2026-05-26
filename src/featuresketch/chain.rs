#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;

use crate::math;

const PROJECTION_BASE_BIN_WIDTH: f64 = 4.0;
const FEATURE_COUNT_BASE_BIN_WIDTH: f64 = 2.0;

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub(crate) enum DimensionFamily {
    Projection,
    FeatureCount,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub(crate) struct ChainLevel {
    pub(crate) dimension: usize,
    pub(crate) width: f64,
    pub(crate) offset: f64,
    pub(crate) bin_volume_ratio: f64,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub(crate) struct ChainLayout {
    levels: Vec<ChainLevel>,
    chain_depth: usize,
}

impl ChainLayout {
    pub(crate) fn new(
        chains: usize,
        depth: usize,
        projection_dims: usize,
        include_feature_count: bool,
        rng: &mut Xoshiro256PlusPlus,
    ) -> Self {
        let dims = projection_dims + usize::from(include_feature_count);
        let mut levels = Vec::with_capacity(chains * depth);
        for _ in 0..chains {
            for level in 0..depth {
                let dimension = (rng.next_u64() % dims as u64) as usize;
                let family = if dimension == projection_dims && include_feature_count {
                    DimensionFamily::FeatureCount
                } else {
                    DimensionFamily::Projection
                };
                let base_width = match family {
                    DimensionFamily::Projection => PROJECTION_BASE_BIN_WIDTH,
                    DimensionFamily::FeatureCount => FEATURE_COUNT_BASE_BIN_WIDTH,
                };
                let scale = math::powf(2.0, level as f64);
                let width = base_width / scale;
                let offset = unit_f64(rng.next_u64()) * width;
                levels.push(ChainLevel {
                    dimension,
                    width,
                    offset,
                    bin_volume_ratio: width / base_width,
                });
            }
        }
        Self {
            levels,
            chain_depth: depth,
        }
    }

    pub(crate) fn levels(&self) -> &[ChainLevel] {
        &self.levels
    }

    pub(crate) fn chain_depth(&self) -> usize {
        self.chain_depth
    }

    pub(crate) fn len(&self) -> usize {
        self.levels.len()
    }
}

impl ChainLevel {
    pub(crate) fn bin(&self, vector: &[f64]) -> i64 {
        math::floor((vector[self.dimension] + self.offset) / self.width) as i64
    }
}

fn unit_f64(value: u64) -> f64 {
    const DENOM: f64 = (1u64 << 53) as f64;
    ((value >> 11) as f64) / DENOM
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn generated_layout_obeys_chain_invariants(
            seed: u64,
            chains in 1usize..8,
            depth in 1usize..8,
            projection_dims in 1usize..16,
            include_feature_count: bool,
        ) {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
            let layout = ChainLayout::new(
                chains,
                depth,
                projection_dims,
                include_feature_count,
                &mut rng,
            );
            let dims = projection_dims + usize::from(include_feature_count);

            prop_assert_eq!(layout.len(), chains * depth);
            prop_assert_eq!(layout.chain_depth(), depth);

            for (index, level) in layout.levels().iter().enumerate() {
                let depth_level = index % depth;
                let is_feature_count = include_feature_count && level.dimension == projection_dims;
                let base_width = if is_feature_count {
                    FEATURE_COUNT_BASE_BIN_WIDTH
                } else {
                    PROJECTION_BASE_BIN_WIDTH
                };
                let expected_width = base_width / math::powf(2.0, depth_level as f64);

                prop_assert!(level.dimension < dims);
                prop_assert!(level.width.is_finite());
                prop_assert!(level.width > 0.0);
                prop_assert!(level.offset >= 0.0);
                prop_assert!(level.offset < level.width);
                prop_assert!((level.width - expected_width).abs() < 1.0e-12);
                prop_assert!((level.bin_volume_ratio - (level.width / base_width)).abs() < 1.0e-12);
            }
        }
    }
}
