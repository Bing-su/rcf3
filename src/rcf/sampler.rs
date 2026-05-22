#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::math;

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------

/// Weighted reservoir sampler backed by a max-heap.
///
/// Points are weighted by `w = ln(-ln(U)) - time_decay * entries_seen` where
/// `U ~ Uniform(0,1)`.  The heap stores the *maximum* weight at the root so
/// that eviction removes the point most likely to be replaced.
///
/// A point is accepted when:
/// - the caller marks it as accepted by the initial warm-up policy, OR
/// - the sampler is full and the new weight is *less than* the current maximum
///   weight.
///
/// This matches the exponential-reservoir scheme used in the reference
/// implementation.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(super) struct Sampler {
    capacity: usize,
    weights: Vec<f64>,
    point_indices: Vec<usize>,
    size: usize,
}

/// Outcome of a call to [`Sampler::accept`].
#[derive(Debug)]
pub(super) struct AcceptResult {
    /// Whether the new point was accepted into the sampler.
    pub(super) accepted: bool,
    /// If a previously-sampled point was evicted, its index.
    pub(super) evicted: Option<usize>,
}

impl Sampler {
    pub(super) fn new(capacity: usize) -> Self {
        Sampler {
            capacity,
            weights: vec![0.0f64; capacity],
            point_indices: vec![usize::MAX; capacity],
            size: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Heap helpers (max-heap on weights)
    // -----------------------------------------------------------------------

    fn swap(&mut self, a: usize, b: usize) {
        self.weights.swap(a, b);
        self.point_indices.swap(a, b);
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.weights[i] > self.weights[parent] {
                self.swap(i, parent);
                i = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        loop {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            let mut largest = i;
            if left < self.size && self.weights[left] > self.weights[largest] {
                largest = left;
            }
            if right < self.size && self.weights[right] > self.weights[largest] {
                largest = right;
            }
            if largest == i {
                break;
            }
            self.swap(i, largest);
            i = largest;
        }
    }

    /// Current maximum weight (root of max-heap).
    fn max_weight(&self) -> f64 {
        if self.size == 0 {
            f64::MAX
        } else {
            self.weights[0]
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Try to accept a candidate point with the given weight.
    ///
    /// `is_initial` should be `true` during the warm-up phase when the sampler
    /// is below capacity; the caller controls this flag to implement the
    /// initial-accept-fraction policy.
    ///
    /// Returns an [`AcceptResult`] describing whether the point was accepted
    /// and whether an existing point was evicted.
    ///
    /// The sampler does not stage accepted state internally; callers pass the
    /// same accepted `weight` to [`Self::add_point`] after the point has been
    /// materialized in the shared point store.
    pub(super) fn accept(&self, is_initial: bool, weight: f64) -> AcceptResult {
        if self.size < self.capacity {
            // Warm-up policy accepted this point; no eviction before capacity.
            if is_initial {
                return AcceptResult {
                    accepted: true,
                    evicted: None,
                };
            }
            return AcceptResult {
                accepted: false,
                evicted: None,
            };
        }

        if weight < self.max_weight() {
            // Replace the current maximum-weight point.
            let evicted_idx = self.point_indices[0];
            AcceptResult {
                accepted: true,
                evicted: Some(evicted_idx),
            }
        } else {
            AcceptResult {
                accepted: false,
                evicted: None,
            }
        }
    }

    /// Finalise insertion after the tree has accepted a point and possibly
    /// resolved duplicates.
    ///
    /// `tree_point_idx` is the index that the tree assigned to the new point
    /// (may differ from the original if it was merged with a duplicate leaf).
    /// `weight` must be the same reservoir weight that produced the accepted decision.
    pub(super) fn add_point(&mut self, tree_point_idx: usize, weight: f64) {
        if self.size < self.capacity {
            // Still filling: append.
            let i = self.size;
            self.weights[i] = weight;
            self.point_indices[i] = tree_point_idx;
            self.size += 1;
            self.sift_up(i);
        } else {
            // Replace heap root (the evicted point).
            self.weights[0] = weight;
            self.point_indices[0] = tree_point_idx;
            self.sift_down(0);
        }
    }

    /// Whether the sampler has reached capacity.
    pub(super) fn is_full(&self) -> bool {
        self.size == self.capacity
    }

    /// Fraction of capacity currently used, in the range `[0.0, 1.0]`.
    pub(super) fn fill_fraction(&self) -> f64 {
        self.size as f64 / self.capacity as f64
    }

    /// All point indices currently in the sampler.
    #[cfg(test)]
    pub(super) fn points(&self) -> &[usize] {
        &self.point_indices[..self.size]
    }
}

// ---------------------------------------------------------------------------
// Weight formula
// ---------------------------------------------------------------------------

/// Compute the exponential-reservoir weight for a new point.
///
/// `u` must be in `(0, 1)`.  Values at or outside this range are clamped to
/// avoid NaN/infinity.
pub(super) fn reservoir_weight(u: f64, time_decay: f64, entries_seen: u64) -> f64 {
    let u = u.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    math::ln(-math::ln(u)) - time_decay * entries_seen as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use rstest::*;

    use super::*;

    #[rstest]
    #[case::cap_1(1)]
    #[case::cap_4(4)]
    #[case::cap_8(8)]
    #[case::cap_16(16)]
    #[case::cap_32(32)]
    fn sampler_fills_to_capacity(#[case] capacity: usize) {
        let mut s = Sampler::new(capacity);
        for i in 0..capacity as u64 {
            let w = reservoir_weight(0.5, 0.0, i);
            let result = s.accept(true, w);
            assert!(result.accepted);
            s.add_point(i as usize, w);
        }
        assert!(s.is_full());
        assert_eq!(s.points().len(), capacity);
    }

    #[rstest]
    #[case::cap_2(2)]
    #[case::cap_4(4)]
    #[case::cap_8(8)]
    fn sampler_evicts_max_weight(#[case] capacity: usize) {
        let mut s = Sampler::new(capacity);
        // Fill with ascending high weights so the max is deterministic.
        for i in 0..capacity {
            let weight = 100.0 + i as f64;
            s.accept(true, weight);
            s.add_point(i, weight);
        }
        assert!(s.is_full());

        // A new point with very low weight should evict the current max.
        let result = s.accept(false, -100.0f64);
        assert!(result.accepted);
        assert!(result.evicted.is_some());
        s.add_point(capacity, -100.0);
    }

    #[test]
    fn sampler_rejects_higher_weight_when_full() {
        let mut s = Sampler::new(2);
        let w = -10.0f64;
        s.accept(true, w);
        s.add_point(0, w);
        s.accept(true, w - 1.0);
        s.add_point(1, w - 1.0);

        // A point with weight > current max is rejected.
        let result = s.accept(false, 999.0f64);
        assert!(!result.accepted);
    }

    #[test]
    fn sampler_rejects_unapproved_warmup_candidate() {
        let s = Sampler::new(2);

        let result = s.accept(false, -100.0);

        assert!(!result.accepted);
        assert!(s.points().is_empty());
    }

    #[test]
    fn rejected_accept_does_not_change_sampler_state() {
        let mut s = Sampler::new(1);
        let accepted_weight = -10.0;
        s.accept(true, accepted_weight);
        s.add_point(0, accepted_weight);

        let result = s.accept(false, 999.0);

        assert!(!result.accepted);
        assert_eq!(s.points(), &[0]);
    }

    #[cfg(feature = "std")]
    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn fill_fraction_in_unit_interval(capacity in 1usize..=64, n_adds in 0usize..=64) {
                let n = n_adds.min(capacity);
                let mut s = Sampler::new(capacity);
                for i in 0..n as u64 {
                    let w = reservoir_weight(0.5, 0.0, i);
                    s.accept(true, w);
                    s.add_point(i as usize, w);
                }
                let frac = s.fill_fraction();
                prop_assert!((0.0..=1.0).contains(&frac), "fill_fraction={frac}");
            }

            #[test]
            fn sampler_never_exceeds_capacity(capacity in 1usize..=32, n_adds in 1usize..=128) {
                let mut s = Sampler::new(capacity);
                for i in 0..n_adds as u64 {
                    let w = reservoir_weight(0.5, 0.0, i);
                    let result = s.accept(!s.is_full(), w);
                    if result.accepted {
                        s.add_point(i as usize, w);
                    }
                }
                prop_assert!(
                    s.points().len() <= capacity,
                    "len={} > capacity={capacity}",
                    s.points().len()
                );
            }

            #[test]
            fn reservoir_weight_finite(
                u in 0.001f64..0.999,
                time_decay in 0.0f64..1.0,
                entries_seen in 0u64..10_000,
            ) {
                let w = reservoir_weight(u, time_decay, entries_seen);
                prop_assert!(w.is_finite(), "reservoir_weight={w} is not finite");
            }
        }
    }
}
