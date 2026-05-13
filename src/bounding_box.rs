#[cfg(all(not(feature = "std"), test))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use itertools::izip;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

fn excess_component(p: f32, lo: f32, hi: f32) -> f32 {
    if p < lo {
        lo - p
    } else if p > hi {
        p - hi
    } else {
        0.0
    }
}

fn excess_outside_box(point: &[f32], min: &[f32], max: &[f32]) -> f32 {
    izip!(point, min, max)
        .map(|(&p, &lo, &hi)| excess_component(p, lo, hi))
        .sum()
}

fn excess_outside_box_with_missing(
    point: &[f32],
    min: &[f32],
    max: &[f32],
    missing: &[bool],
) -> f32 {
    izip!(point, min, max, missing)
        .filter(|(_, _, _, is_missing)| !*is_missing)
        .map(|(&p, &lo, &hi, _)| excess_component(p, lo, hi))
        .sum()
}

fn active_range_sum_with_missing(min: &[f32], max: &[f32], missing: &[bool]) -> f64 {
    izip!(min, max, missing)
        .filter(|(_, _, is_missing)| !*is_missing)
        .map(|(&lo, &hi, _)| (hi - lo) as f64)
        .sum()
}

fn componentwise_min_max(a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>) {
    izip!(a, b).map(|(&x, &y)| (x.min(y), x.max(y))).unzip()
}

fn merge_bounds_in_place(min: &mut [f32], max: &mut [f32], other_min: &[f32], other_max: &[f32]) {
    debug_assert_eq!(min.len(), max.len());
    debug_assert_eq!(min.len(), other_min.len());
    debug_assert_eq!(max.len(), other_max.len());

    for (lo, hi, olo, ohi) in izip!(min.iter_mut(), max.iter_mut(), other_min, other_max) {
        *lo = lo.min(*olo);
        *hi = hi.max(*ohi);
    }
}

/// Axis-aligned bounding box in `dim`-dimensional space.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BoundingBox {
    pub min: Vec<f32>,
    pub max: Vec<f32>,
    /// Sum of per-dimension ranges; cached for efficiency.
    range_sum: f64,
}

impl BoundingBox {
    /// Degenerate box containing a single point.
    pub fn from_point(p: &[f32]) -> Self {
        BoundingBox {
            min: p.to_vec(),
            max: p.to_vec(),
            range_sum: 0.0,
        }
    }

    /// Smallest box containing both `a` and `b`.
    pub fn from_two_points(a: &[f32], b: &[f32]) -> Self {
        debug_assert_eq!(a.len(), b.len());
        let (min, max) = componentwise_min_max(a, b);
        let range_sum = range_sum(&min, &max);
        BoundingBox {
            min,
            max,
            range_sum,
        }
    }

    /// Expand this box to also contain `point`.
    /// Returns `true` if the box expanded.
    pub fn expand_with_point(&mut self, point: &[f32]) -> bool {
        let old = self.range_sum;
        merge_bounds_in_place(&mut self.min, &mut self.max, point, point);
        self.range_sum = range_sum(&self.min, &self.max);
        (self.range_sum - old).abs() >= 1e-12
    }

    /// Expand this box to also contain all of `other`.
    pub fn merge(&mut self, other: &BoundingBox) {
        merge_bounds_in_place(&mut self.min, &mut self.max, &other.min, &other.max);
        self.range_sum = range_sum(&self.min, &self.max);
    }

    /// Probability that a random cut separating `point` from this box would
    /// be made at some dimension when cutting on the merged (box ∪ point) box.
    ///
    /// Returns 0.0 when `point` is inside the box.
    pub fn probability_of_cut(&self, point: &[f32]) -> f64 {
        let excess = excess_outside_box(point, &self.min, &self.max);

        if excess == 0.0 {
            return 0.0;
        }
        if self.range_sum == 0.0 {
            return 1.0;
        }
        excess as f64 / (self.range_sum + excess as f64)
    }

    /// Probability of cut ignoring dimensions listed in `missing`.
    pub fn probability_of_cut_with_missing(&self, point: &[f32], missing: &[bool]) -> f64 {
        let excess = excess_outside_box_with_missing(point, &self.min, &self.max, missing);
        let active_range = active_range_sum_with_missing(&self.min, &self.max, missing);
        if excess == 0.0 {
            return 0.0;
        }
        if active_range == 0.0 {
            return 1.0;
        }
        excess as f64 / (active_range + excess as f64)
    }

    pub fn range_sum(&self) -> f64 {
        self.range_sum
    }

    pub fn merge_with(mut self, other: &BoundingBox) -> Self {
        self.merge(other);
        self
    }
}

fn range_sum(min: &[f32], max: &[f32]) -> f64 {
    izip!(min, max).map(|(&lo, &hi)| (hi - lo) as f64).sum()
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::inside([0.5, 0.5], 0.0)]
    #[case::outside([5.0, 0.5], 4.0 / 6.0)]
    fn probability_of_cut_matches_expected(#[case] point: [f32; 2], #[case] expected: f64) {
        let bbox = BoundingBox::from_two_points(&[0.0, 0.0], &[1.0, 1.0]);
        let prob = bbox.probability_of_cut(&point);
        assert_abs_diff_eq!(prob, expected, epsilon = 1e-12);
    }

    #[test]
    fn extreme_outlier_approaches_one() {
        let bbox = BoundingBox::from_two_points(&[0.0, 0.0], &[1.0, 1.0]);
        let prob = bbox.probability_of_cut(&[1000.0, 0.5]);
        assert!(prob > 0.99);
    }

    #[test]
    fn merge_expands_box() {
        let mut a = BoundingBox::from_point(&[0.0, 0.0]);
        let b = BoundingBox::from_point(&[2.0, 3.0]);
        a.merge(&b);
        assert_eq!(a.min, vec![0.0f32, 0.0]);
        assert_eq!(a.max, vec![2.0f32, 3.0]);
    }

    #[test]
    fn excess_outside_box_no_missing() {
        let min = [0.0f32, 0.0, -1.0];
        let max = [1.0f32, 2.0, 1.0];
        let point = [-2.0f32, 1.5, 3.0];
        // dim0: 0-(-2)=2, dim1: inside, dim2: 3-1=2 => total 4
        assert_abs_diff_eq!(
            excess_outside_box(&point, &min, &max),
            4.0f32,
            epsilon = 1e-6f32
        );
    }

    #[rstest]
    #[case::no_mask([-2.0, 1.5, 3.0], [false, false, false], 4.0)]
    #[case::mask_first([-2.0, 1.5, 3.0], [true, false, false], 2.0)]
    #[case::all_masked([-2.0, 1.5, 3.0], [true, true, true], 0.0)]
    #[case::all_inside([0.5, 1.5, 0.0], [false, false, false], 0.0)]
    fn excess_outside_box_with_missing_work(
        #[case] point: [f32; 3],
        #[case] missing: [bool; 3],
        #[case] expected: f32,
    ) {
        let min = [0.0f32, 0.0, -1.0];
        let max = [1.0f32, 2.0, 1.0];

        let excess = excess_outside_box_with_missing(&point, &min, &max, &missing);
        assert_abs_diff_eq!(excess, expected, epsilon = 1e-6f32);
    }

    #[rstest]
    #[case::masked_outside_dim([5.0, 0.5], [true, false], 0.0)]
    #[case::active_outside_dim([0.5, 5.0], [true, false], 4.0 / 5.0)]
    #[case::boundary_is_inside([1.0, 1.0], [false, false], 0.0)]
    fn probability_with_missing_ignores_masked_dims(
        #[case] point: [f32; 2],
        #[case] missing: [bool; 2],
        #[case] expected: f64,
    ) {
        let bbox = BoundingBox::from_two_points(&[0.0, 0.0], &[1.0, 1.0]);
        let prob = bbox.probability_of_cut_with_missing(&point, &missing);
        assert_abs_diff_eq!(prob, expected, epsilon = 1e-12);
    }

    #[test]
    fn from_two_points_uses_componentwise_bounds() {
        let bbox = BoundingBox::from_two_points(&[3.0, -2.0, 1.0], &[1.0, 4.0, -5.0]);
        assert_eq!(bbox.min, vec![1.0f32, -2.0, -5.0]);
        assert_eq!(bbox.max, vec![3.0f32, 4.0, 1.0]);
        assert_abs_diff_eq!(bbox.range_sum(), 14.0, epsilon = 1e-12);
    }

    #[test]
    fn componentwise_min_max_matches_expected() {
        let (min, max) = componentwise_min_max(&[2.0, -1.0, 7.0], &[3.0, -4.0, 5.0]);
        assert_eq!(min, vec![2.0f32, -4.0, 5.0]);
        assert_eq!(max, vec![3.0f32, -1.0, 7.0]);
    }

    #[test]
    fn merge_bounds_in_place_updates_both_sides() {
        let mut min = vec![0.0f32, -1.0, 2.0];
        let mut max = vec![1.0f32, 2.0, 4.0];
        let other_min = vec![-3.0f32, 0.0, 3.0];
        let other_max = vec![0.5f32, 5.0, 10.0];

        merge_bounds_in_place(&mut min, &mut max, &other_min, &other_max);

        assert_eq!(min, vec![-3.0f32, -1.0, 2.0]);
        assert_eq!(max, vec![1.0f32, 5.0, 10.0]);
    }

    #[test]
    fn expand_with_point_matches_merge_with_point_box() {
        let mut via_expand = BoundingBox::from_two_points(&[0.0, 0.0], &[1.0, 1.0]);
        let mut via_merge = via_expand.clone();
        let point = [3.0f32, -2.0];

        via_expand.expand_with_point(&point);
        via_merge.merge(&BoundingBox::from_point(&point));

        assert_eq!(via_expand.min, via_merge.min);
        assert_eq!(via_expand.max, via_merge.max);
        assert_abs_diff_eq!(
            via_expand.range_sum(),
            via_merge.range_sum(),
            epsilon = 1e-12
        );
    }
}
