#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use ahash::RandomState;
use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;

use super::input::NormalizedFeature;
use crate::math;

const SQRT_3: f64 = 1.732050807568877293527446341505872367_f64;

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub(crate) struct ProjectionSeeds {
    value: Seed4,
    presence: Seed4,
    feature_lo: Seed4,
    feature_hi: Seed4,
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub(crate) struct Seed4(
    pub(crate) u64,
    pub(crate) u64,
    pub(crate) u64,
    pub(crate) u64,
);

impl ProjectionSeeds {
    pub(crate) fn new(rng: &mut Xoshiro256PlusPlus) -> Self {
        Self {
            value: Seed4::from_rng(rng),
            presence: Seed4::from_rng(rng),
            feature_lo: Seed4::from_rng(rng),
            feature_hi: Seed4::from_rng(rng),
        }
    }
}

impl Seed4 {
    pub(crate) fn from_rng(rng: &mut Xoshiro256PlusPlus) -> Self {
        Self(
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FeatureHash {
    lo: u64,
    hi: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct ProjectedEvent {
    pub(crate) value: Vec<f64>,
    pub(crate) presence: Vec<f64>,
}

pub(crate) fn project(
    features: &[NormalizedFeature],
    value_dims: usize,
    presence_dims: usize,
    seeds: &ProjectionSeeds,
) -> ProjectedEvent {
    let mut value = vec![0.0; value_dims];
    let mut presence = vec![0.0; presence_dims + 1];

    for feature in features {
        let hash = feature_hash(&feature.name, seeds);
        for (dim, projected) in value.iter_mut().enumerate() {
            *projected += feature.value * coefficient(seeds.value, hash, dim);
        }
        for (dim, projected) in presence.iter_mut().take(presence_dims).enumerate() {
            *projected += coefficient(seeds.presence, hash, dim);
        }
    }
    presence[presence_dims] = math::ln(1.0 + features.len() as f64);

    ProjectedEvent { value, presence }
}

fn feature_hash(name: &str, seeds: &ProjectionSeeds) -> FeatureHash {
    FeatureHash {
        lo: random_state(seeds.feature_lo).hash_one(name),
        hi: random_state(seeds.feature_hi).hash_one(name),
    }
}

fn coefficient(seed: Seed4, feature: FeatureHash, dim: usize) -> f64 {
    let state = random_state(seed);
    match state.hash_one((feature.lo, feature.hi, dim as u64)) % 6 {
        0 => SQRT_3,
        1 => -SQRT_3,
        _ => 0.0,
    }
}

pub(crate) fn random_state(seed: Seed4) -> RandomState {
    RandomState::with_seeds(seed.0, seed.1, seed.2, seed.3)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use rstest::rstest;

    use super::*;

    #[test]
    fn projection_is_deterministic() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let seeds = ProjectionSeeds::new(&mut rng);
        let features = crate::featuresketch::input::normalize([("a", 1.0), ("b", -2.0)]).unwrap();
        let left = project(&features, 8, 8, &seeds);
        let right = project(&features, 8, 8, &seeds);
        for (left, right) in left.value.iter().zip(right.value.iter()) {
            assert_abs_diff_eq!(left, right, epsilon = 1.0e-12);
        }
        for (left, right) in left.presence.iter().zip(right.presence.iter()) {
            assert_abs_diff_eq!(left, right, epsilon = 1.0e-12);
        }
    }

    #[rstest]
    #[case::empty(Vec::new(), 4, 3, 0.0)]
    #[case::two_features(
        crate::featuresketch::input::normalize([("a", 1.0), ("b", -2.0)]).unwrap(),
        8,
        5,
        math::ln(3.0)
    )]
    fn projection_shape_and_feature_count_signal(
        #[case] features: Vec<NormalizedFeature>,
        #[case] value_dims: usize,
        #[case] presence_dims: usize,
        #[case] expected_count_signal: f64,
    ) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(19);
        let seeds = ProjectionSeeds::new(&mut rng);
        let projected = project(&features, value_dims, presence_dims, &seeds);

        assert_eq!(projected.value.len(), value_dims);
        assert_eq!(projected.presence.len(), presence_dims + 1);
        assert_abs_diff_eq!(
            projected.presence[presence_dims],
            expected_count_signal,
            epsilon = 1.0e-12
        );
        assert!(projected.value.iter().all(|value| value.is_finite()));
        assert!(projected.presence.iter().all(|value| value.is_finite()));
    }
}
