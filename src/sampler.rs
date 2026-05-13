#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

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
/// - the sampler is not yet full (initial warm-up), OR
/// - the new weight is *less than* the current maximum weight.
///
/// This matches the exponential-reservoir scheme used in the reference
/// implementation.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Sampler {
    capacity: usize,
    weights: Vec<f64>,
    point_indices: Vec<usize>,
    size: usize,
    /// State saved between `accept` and `add_point`.
    pending_weight: f64,
    pending_point: usize,
}

/// Outcome of a call to [`Sampler::accept`].
#[derive(Debug)]
pub struct AcceptResult {
    /// Whether the new point was accepted into the sampler.
    pub accepted: bool,
    /// If a previously-sampled point was evicted, its index.
    pub evicted: Option<usize>,
}

impl Sampler {
    pub fn new(capacity: usize) -> Self {
        Sampler {
            capacity,
            weights: vec![0.0f64; capacity],
            point_indices: vec![usize::MAX; capacity],
            size: 0,
            pending_weight: 0.0,
            pending_point: usize::MAX,
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

    /// Try to accept a point with the given weight.
    ///
    /// `is_initial` should be `true` during the warm-up phase when the sampler
    /// is below capacity; the caller controls this flag to implement the
    /// initial-accept-fraction policy.
    ///
    /// Returns an [`AcceptResult`] describing whether the point was accepted
    /// and whether an existing point was evicted.
    ///
    /// If accepted, the caller must follow up with [`add_point`] once the
    /// actual point index is known (after possible de-duplication by the tree).
    pub fn accept(&mut self, is_initial: bool, weight: f64, point_idx: usize) -> AcceptResult {
        if is_initial || (self.size < self.capacity) {
            // Warm-up: always accept; no eviction yet.
            self.pending_weight = weight;
            self.pending_point = point_idx;
            return AcceptResult {
                accepted: true,
                evicted: None,
            };
        }

        if weight < self.max_weight() {
            // Replace the current maximum-weight point.
            let evicted_idx = self.point_indices[0];
            self.pending_weight = weight;
            self.pending_point = point_idx;
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
    pub fn add_point(&mut self, tree_point_idx: usize) {
        debug_assert!(self.pending_point != usize::MAX);

        if self.size < self.capacity {
            // Still filling: append.
            let i = self.size;
            self.weights[i] = self.pending_weight;
            self.point_indices[i] = tree_point_idx;
            self.size += 1;
            self.sift_up(i);
        } else {
            // Replace heap root (the evicted point).
            self.weights[0] = self.pending_weight;
            self.point_indices[0] = tree_point_idx;
            self.sift_down(0);
        }

        self.pending_point = usize::MAX;
    }

    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }

    pub fn fill_fraction(&self) -> f64 {
        self.size as f64 / self.capacity as f64
    }

    /// All point indices currently in the sampler.
    pub fn points(&self) -> &[usize] {
        &self.point_indices[..self.size]
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ---------------------------------------------------------------------------
// Weight formula
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
fn ln_f64(x: f64) -> f64 {
    x.ln()
}

#[cfg(not(feature = "std"))]
fn ln_f64(x: f64) -> f64 {
    libm::log(x)
}

/// Compute the exponential-reservoir weight for a new point.
///
/// `u` must be in `(0, 1)`.  Values at or outside this range are clamped to
/// avoid NaN/infinity.
pub fn reservoir_weight(u: f64, time_decay: f64, entries_seen: u64) -> f64 {
    let u = u.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    ln_f64(-ln_f64(u)) - time_decay * entries_seen as f64
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
            let result = s.accept(true, w, i as usize);
            assert!(result.accepted);
            s.add_point(i as usize);
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
            s.accept(true, 100.0 + i as f64, i);
            s.add_point(i);
        }
        assert!(s.is_full());

        // A new point with very low weight should evict the current max.
        let result = s.accept(false, -100.0f64, capacity);
        assert!(result.accepted);
        assert!(result.evicted.is_some());
        s.add_point(capacity);
    }

    #[test]
    fn sampler_rejects_higher_weight_when_full() {
        let mut s = Sampler::new(2);
        let w = -10.0f64;
        s.accept(true, w, 0);
        s.add_point(0);
        s.accept(true, w - 1.0, 1);
        s.add_point(1);

        // A point with weight > current max is rejected.
        let result = s.accept(false, 999.0f64, 2);
        assert!(!result.accepted);
    }
}
