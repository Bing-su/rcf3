#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

use itertools::izip;
use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Axis-aligned support rectangle for an online isolation-tree bin.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(crate) struct Support {
    min: Vec<f32>,
    max: Vec<f32>,
}

impl Support {
    pub(crate) fn from_point(point: &[f32]) -> Self {
        Self {
            min: point.to_vec(),
            max: point.to_vec(),
        }
    }

    pub(crate) fn from_points(points: &[Vec<f32>]) -> Option<Self> {
        let first = points.first()?;
        let mut support = Self::from_point(first);
        for point in &points[1..] {
            support.expand(point);
        }
        Some(support)
    }

    pub(crate) fn expand(&mut self, point: &[f32]) {
        for (lo, hi, value) in izip!(&mut self.min, &mut self.max, point) {
            *lo = lo.min(*value);
            *hi = hi.max(*value);
        }
    }

    pub(crate) fn merged(left: &Self, right: &Self) -> Self {
        let mut support = left.clone();
        for (lo, hi, right_lo, right_hi) in
            izip!(&mut support.min, &mut support.max, &right.min, &right.max)
        {
            *lo = lo.min(*right_lo);
            *hi = hi.max(*right_hi);
        }
        support
    }

    pub(crate) fn split_regions(&self, dimension: usize, value: f32) -> (Self, Self) {
        let mut left = self.clone();
        let mut right = self.clone();
        left.max[dimension] = value;
        right.min[dimension] = value;
        (left, right)
    }

    pub(crate) fn sample_point(&self, rng: &mut Xoshiro256PlusPlus) -> Vec<f32> {
        izip!(&self.min, &self.max)
            .map(|(&lo, &hi)| {
                if lo == hi {
                    lo
                } else {
                    rng.random_range(lo..hi)
                }
            })
            .collect()
    }

    pub(crate) fn sample_split(&self, rng: &mut Xoshiro256PlusPlus) -> Option<(usize, f32)> {
        // The paper samples from every feature dimension. We intentionally skip
        // zero-width dimensions here: they cannot produce a useful split, and a
        // fully degenerate support should remain an unsplit leaf.
        let active_dims: Vec<usize> = izip!(&self.min, &self.max)
            .enumerate()
            .filter_map(|(idx, (&lo, &hi))| (lo < hi).then_some(idx))
            .collect();
        if active_dims.is_empty() {
            return None;
        }
        let dimension = *active_dims.get(rng.random_range(0..active_dims.len()))?;
        let value = rng.random_range(self.min[dimension]..self.max[dimension]);
        Some((dimension, value))
    }

    #[cfg(test)]
    pub(crate) fn contains_support(&self, other: &Self) -> bool {
        izip!(&self.min, &self.max, &other.min, &other.max)
            .all(|(&lo, &hi, &other_lo, &other_hi)| lo <= other_lo && other_hi <= hi)
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(crate) struct Split {
    pub(crate) dimension: usize,
    pub(crate) value: f32,
    pub(crate) left: Box<Node>,
    pub(crate) right: Box<Node>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(crate) struct Node {
    pub(crate) height: usize,
    pub(crate) support: Support,
    pub(crate) split: Option<Split>,
}

impl Node {
    pub(crate) fn new(height: usize, support: Support) -> Self {
        Self {
            height,
            support,
            split: None,
        }
    }

    pub(crate) fn is_leaf(&self) -> bool {
        self.split.is_none()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use rstest::rstest;

    use super::*;

    #[test]
    fn support_expands_and_merges() {
        let mut support = Support::from_point(&[1.0, 3.0]);
        support.expand(&[-2.0, 5.0]);
        assert_eq!(
            support,
            Support {
                min: vec![-2.0, 3.0],
                max: vec![1.0, 5.0]
            }
        );

        let merged = Support::merged(
            &Support::from_point(&[-3.0, 4.0]),
            &Support::from_point(&[2.0, -1.0]),
        );
        assert_eq!(
            merged,
            Support {
                min: vec![-3.0, -1.0],
                max: vec![2.0, 4.0]
            }
        );
    }

    #[rstest]
    #[case::first_dimension(0, 1.5)]
    #[case::second_dimension(1, 0.5)]
    fn split_regions_stay_inside_parent(#[case] dimension: usize, #[case] value: f32) {
        let parent = Support {
            min: vec![0.0, -1.0],
            max: vec![4.0, 3.0],
        };
        let (left, right) = parent.split_regions(dimension, value);
        assert!(parent.contains_support(&left));
        assert!(parent.contains_support(&right));
    }

    #[test]
    fn split_sampling_ignores_zero_width_dimensions() {
        let support = Support {
            min: vec![2.0, -1.0, 5.0],
            max: vec![2.0, 3.0, 5.0],
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(9);

        for _ in 0..16 {
            let (dimension, value) = support.sample_split(&mut rng).unwrap();
            assert_eq!(dimension, 1);
            assert!((-1.0..3.0).contains(&value));
        }
    }

    #[test]
    fn fully_degenerate_support_has_no_split_candidate() {
        let support = Support::from_point(&[1.0, 1.0]);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(9);
        assert_eq!(support.sample_split(&mut rng), None);
    }
}
