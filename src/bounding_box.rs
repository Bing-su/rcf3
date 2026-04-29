use serde::{Deserialize, Serialize};

/// Axis-aligned bounding box in `dim`-dimensional space.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min: Vec<f32>,
    pub max: Vec<f32>,
    /// Sum of per-dimension ranges; cached for efficiency.
    pub range_sum: f64,
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
        let min: Vec<f32> = a.iter().zip(b).map(|(x, y)| x.min(*y)).collect();
        let max: Vec<f32> = a.iter().zip(b).map(|(x, y)| x.max(*y)).collect();
        let range_sum = range_sum(&min, &max);
        BoundingBox {
            min,
            max,
            range_sum,
        }
    }

    /// Expand this box to also contain `point`.
    /// Returns `true` if `point` was already inside (box unchanged).
    pub fn expand_with_point(&mut self, point: &[f32]) -> bool {
        let old = self.range_sum;
        for i in 0..self.min.len() {
            if point[i] < self.min[i] {
                self.min[i] = point[i];
            }
            if point[i] > self.max[i] {
                self.max[i] = point[i];
            }
        }
        self.range_sum = range_sum(&self.min, &self.max);
        (self.range_sum - old).abs() < 1e-12
    }

    /// Expand this box to also contain all of `other`.
    pub fn merge(&mut self, other: &BoundingBox) {
        for i in 0..self.min.len() {
            if other.min[i] < self.min[i] {
                self.min[i] = other.min[i];
            }
            if other.max[i] > self.max[i] {
                self.max[i] = other.max[i];
            }
        }
        self.range_sum = range_sum(&self.min, &self.max);
    }

    /// Probability that a random cut separating `point` from this box would
    /// be made at some dimension when cutting on the merged (box ∪ point) box.
    ///
    /// Returns 0.0 when `point` is inside the box.
    pub fn probability_of_cut(&self, point: &[f32]) -> f64 {
        let excess: f32 = point
            .iter()
            .zip(&self.min)
            .zip(&self.max)
            .map(|((&p, &lo), &hi)| {
                if p < lo {
                    lo - p
                } else if p > hi {
                    p - hi
                } else {
                    0.0
                }
            })
            .sum();

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
        let mut excess = 0.0f32;
        let mut active_range = 0.0f64;
        for i in 0..self.min.len() {
            if missing[i] {
                continue;
            }
            let p = point[i];
            if p < self.min[i] {
                excess += self.min[i] - p;
            } else if p > self.max[i] {
                excess += p - self.max[i];
            }
            active_range += (self.max[i] - self.min[i]) as f64;
        }
        if excess == 0.0 {
            return 0.0;
        }
        if active_range == 0.0 {
            return 1.0;
        }
        excess as f64 / (active_range + excess as f64)
    }
}

#[inline]
fn range_sum(min: &[f32], max: &[f32]) -> f64 {
    min.iter().zip(max).map(|(lo, hi)| (hi - lo) as f64).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_inside_has_zero_cut_prob() {
        let bbox = BoundingBox::from_two_points(&[0.0, 0.0], &[1.0, 1.0]);
        assert_eq!(bbox.probability_of_cut(&[0.5, 0.5]), 0.0);
    }

    #[test]
    fn point_outside_has_positive_cut_prob() {
        let bbox = BoundingBox::from_two_points(&[0.0, 0.0], &[1.0, 1.0]);
        assert!(bbox.probability_of_cut(&[5.0, 0.5]) > 0.0);
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
        assert_eq!(a.min, vec![0.0, 0.0]);
        assert_eq!(a.max, vec![2.0, 3.0]);
    }
}
