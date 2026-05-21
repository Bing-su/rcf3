#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use itertools::izip;
use ndarray::{Array2, ArrayView1, s};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{RcfError, Result};

fn lookahead_offset(dim: usize, input_dim: usize, look_ahead: usize) -> usize {
    dim - input_dim * (1 + look_ahead)
}

fn l1_distance_slices(query: &[f32], stored: &[f32]) -> f64 {
    query
        .iter()
        .zip(stored)
        .map(|(a, b)| (a - b).abs() as f64)
        .sum()
}

fn l1_distance_slices_ignore_missing(query: &[f32], stored: &[f32], missing: &[bool]) -> f64 {
    izip!(query, stored, missing)
        .filter(|(_, _, m)| !*m)
        .map(|(a, b, _)| (a - b).abs() as f64)
        .sum()
}

// ---------------------------------------------------------------------------
// PointStore
// ---------------------------------------------------------------------------

/// Row-indexed point matrix: each row `i` holds the `dim`-dimensional vector
/// for point slot `i`.  In C (row-major) layout rows are contiguous, so
/// `row(idx).as_slice()` is always `Some`.
type PointMatrix = Array2<f32>;

/// Row-indexed point storage shared across all trees in the forest.
///
/// When `internal_shingling` is enabled, callers pass one base observation at
/// a time and the store automatically maintains the rolling shingle buffer,
/// exposing a full `input_dim * shingle_size`-dimensional vector.
///
/// Memory layout: an `Array2<f32>` of shape `(capacity, dim)` in C order.
/// Each row is a stored point; rows are contiguous so `row(idx)` gives a
/// zero-copy `ArrayView1<f32>` or `&[f32]` slice.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(super) struct PointStore {
    /// Full dimensionality: `input_dim * shingle_size`.
    dim: usize,
    /// Base input dimensionality (before shingling).
    input_dim: usize,
    /// Shingle size (temporal window).
    ///
    /// Kept in serialized state even though current production code derives
    /// the same information from `dim` and `input_dim`.
    #[allow(dead_code)]
    shingle_size: usize,
    /// Whether the store manages the rolling shingle buffer.
    ///
    /// Kept in serialized state to preserve historical Forest snapshots.
    #[allow(dead_code)]
    internal_shingling: bool,

    /// Row-indexed point matrix (shape: `capacity × dim`).
    store: PointMatrix,
    /// Whether each slot is occupied.
    occupied: Vec<bool>,
    /// Reference count for each slot (how many trees reference it).
    ref_count: Vec<usize>,
    /// Next slot to use when free_list is empty.
    next_free: usize,
    /// Recycled slot indices.
    free_list: Vec<usize>,
    /// Current number of occupied slots.
    size: usize,
    /// Maximum number of slots.
    capacity: usize,

    /// Rolling shingle buffer (used when `internal_shingling = true`).
    shingle_buf: Vec<f32>,
    /// Total number of `add` calls (for shingling state tracking).
    entries_seen: u64,
}

impl PointStore {
    /// Create a new store.
    ///
    /// `input_dim` × `shingle_size` = full model dimension.
    pub(super) fn new(
        input_dim: usize,
        shingle_size: usize,
        capacity: usize,
        internal_shingling: bool,
    ) -> Self {
        let dim = input_dim * shingle_size;
        PointStore {
            dim,
            input_dim,
            shingle_size,
            internal_shingling,
            store: Array2::zeros((capacity, dim)),
            occupied: vec![false; capacity],
            ref_count: vec![0usize; capacity],
            next_free: 0,
            free_list: Vec::new(),
            size: 0,
            capacity,
            shingle_buf: vec![0.0f32; dim],
            entries_seen: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Shingling
    // -----------------------------------------------------------------------

    /// Build the full-dimensional shingled point from `base`.
    ///
    /// When `internal_shingling` is false, `base` must already have length
    /// `dim`; it is returned as-is (cloned).
    ///
    /// When `internal_shingling` is true, `base` must have length `input_dim`;
    /// the store shifts the rolling buffer and fills in the new observation.
    #[cfg(test)]
    fn shingled_point(&mut self, base: &[f32]) -> Result<Vec<f32>> {
        if self.internal_shingling {
            self.advance_shingle(base)?;
            Ok(self.shingle_buf.clone())
        } else {
            self.validate_full_point(base)?;
            Ok(base.to_vec())
        }
    }

    pub(super) fn advance_shingle(&mut self, base: &[f32]) -> Result<()> {
        if base.len() != self.input_dim {
            return Err(RcfError::DimensionMismatch {
                expected: self.input_dim,
                got: base.len(),
            });
        }
        self.shingle_buf.copy_within(self.input_dim.., 0);
        let start = self.dim - self.input_dim;
        self.shingle_buf[start..].copy_from_slice(base);
        Ok(())
    }

    /// Peek at the current shingled point without advancing the shingle buffer.
    pub(super) fn current_shingled(&self) -> &[f32] {
        &self.shingle_buf
    }

    // -----------------------------------------------------------------------
    // Allocation / deallocation
    // -----------------------------------------------------------------------

    /// Store `point` and return its index.
    ///
    /// The reference count is initialised to 0; callers must call
    /// [`Self::inc_ref`] for each tree that accepts this point.
    #[cfg(test)]
    pub(super) fn add(&mut self, point: &[f32]) -> Result<usize> {
        self.validate_full_point(point)?;
        self.add_validated(point)
    }

    pub(super) fn add_validated(&mut self, point: &[f32]) -> Result<usize> {
        debug_assert_eq!(point.len(), self.dim);
        self.store_point(point)
    }

    pub(super) fn add_current_shingled(&mut self) -> Result<usize> {
        let idx = self.allocate_slot()?;
        self.store
            .row_mut(idx)
            .assign(&ArrayView1::from(&self.shingle_buf[..]));
        self.finish_add(idx);
        Ok(idx)
    }

    pub(super) fn validate_full_point(&self, point: &[f32]) -> Result<()> {
        if point.len() != self.dim {
            return Err(RcfError::DimensionMismatch {
                expected: self.dim,
                got: point.len(),
            });
        }
        Ok(())
    }

    fn store_point(&mut self, point: &[f32]) -> Result<usize> {
        let idx = self.allocate_slot()?;
        self.store.row_mut(idx).assign(&ArrayView1::from(point));
        self.finish_add(idx);
        Ok(idx)
    }

    fn finish_add(&mut self, idx: usize) {
        self.occupied[idx] = true;
        self.ref_count[idx] = 0;
        self.size += 1;
        self.entries_seen += 1;
    }

    fn allocate_slot(&mut self) -> Result<usize> {
        if let Some(idx) = self.free_list.pop() {
            return Ok(idx);
        }
        if self.next_free < self.capacity {
            let idx = self.next_free;
            self.next_free += 1;
            return Ok(idx);
        }
        // Grow the store: allocate a larger matrix and copy the existing rows.
        let new_cap = self.capacity * 2 + 4;
        let mut new_store = Array2::zeros((new_cap, self.dim));
        new_store
            .slice_mut(s![..self.capacity, ..])
            .assign(&self.store);
        self.store = new_store;
        self.occupied.resize(new_cap, false);
        self.ref_count.resize(new_cap, 0);
        let idx = self.next_free;
        self.next_free += 1;
        self.capacity = new_cap;
        Ok(idx)
    }

    /// Increment reference count for slot `idx`.
    pub(super) fn inc_ref(&mut self, idx: usize) {
        self.ref_count[idx] += 1;
    }

    /// Decrement reference count; free the slot when it reaches zero.
    pub(super) fn dec_ref(&mut self, idx: usize) {
        if self.ref_count[idx] > 0 {
            self.ref_count[idx] -= 1;
        }
        if self.ref_count[idx] == 0 && self.occupied[idx] {
            self.occupied[idx] = false;
            self.size -= 1;
            self.free_list.push(idx);
        }
    }

    // -----------------------------------------------------------------------
    // Read access
    // -----------------------------------------------------------------------

    /// Return the point at `idx`.  Panics if `idx` is not occupied.
    pub(super) fn get(&self, idx: usize) -> &[f32] {
        debug_assert!(self.occupied[idx], "accessing unoccupied slot {idx}");
        self.store
            .row(idx)
            .to_slice()
            .expect("store must be contiguous")
    }

    /// Check whether the stored point at `idx` equals `point` component-wise.
    pub(super) fn is_equal(&self, point: &[f32], idx: usize) -> bool {
        self.store.row(idx) == ArrayView1::from(point)
    }

    /// L1 distance between `query` and the stored point at `idx`.
    pub(super) fn l1_distance(&self, query: &[f32], idx: usize) -> f64 {
        let stored = self.get(idx);
        l1_distance_slices(query, stored)
    }

    /// L1 distance ignoring `missing` dimensions.
    pub(super) fn l1_distance_ignore_missing(
        &self,
        query: &[f32],
        idx: usize,
        missing: &[bool],
    ) -> f64 {
        let stored = self.get(idx);
        l1_distance_slices_ignore_missing(query, stored, missing)
    }

    /// Return a copy of the point at `idx`.
    pub(super) fn copy_point(&self, idx: usize) -> Vec<f32> {
        self.get(idx).to_vec()
    }

    // -----------------------------------------------------------------------
    // Imputation helpers
    // -----------------------------------------------------------------------

    /// Compute the indices of the base dimensions that would be filled at
    /// look-ahead step `look_ahead` inside the shingle buffer.
    ///
    /// For a shingle buffer `[t-k+1, …, t]` (k slots of `input_dim` each),
    /// `next_indices(0)` returns the positions that the *next* base observation
    /// would fill (the newest slot in the buffer).
    #[cfg(test)]
    fn next_indices(&self, look_ahead: usize) -> Vec<usize> {
        let offset = lookahead_offset(self.dim, self.input_dim, look_ahead);
        (0..self.input_dim).map(|i| offset + i).collect()
    }

    pub(super) fn next_indices_into(&self, look_ahead: usize, indices: &mut Vec<usize>) {
        indices.clear();
        let offset = lookahead_offset(self.dim, self.input_dim, look_ahead);
        indices.extend((0..self.input_dim).map(|i| offset + i));
    }

    /// Convert `missing` base-dimension indices into full-dimension indices
    /// within the shingled vector, shifted by `look_ahead` steps.
    #[cfg(test)]
    fn missing_indices_with_lookahead(
        &self,
        look_ahead: usize,
        missing_base: &[usize],
    ) -> Vec<usize> {
        let offset = lookahead_offset(self.dim, self.input_dim, look_ahead);
        missing_base.iter().map(|&i| offset + i).collect()
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Number of points currently retained in the store.
    #[cfg(test)]
    fn num_points(&self) -> usize {
        self.size
    }

    #[cfg(test)]
    fn dim(&self) -> usize {
        self.dim
    }

    #[cfg(test)]
    fn input_dim(&self) -> usize {
        self.input_dim
    }

    #[cfg(test)]
    fn entries_seen(&self) -> u64 {
        self.entries_seen
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use rstest::rstest;

    use super::*;

    #[test]
    fn add_and_get() {
        let mut ps = PointStore::new(2, 1, 8, false);
        let idx = ps.add(&[1.0, 2.0]).unwrap();
        assert_eq!(ps.get(idx), &[1.0f32, 2.0]);
    }

    #[test]
    fn add_validated_stores_full_point() {
        let mut ps = PointStore::new(2, 2, 8, false);

        let idx = ps.add_validated(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        assert_eq!(idx, 0);
        assert_eq!(ps.get(idx), &[1.0f32, 2.0, 3.0, 4.0]);
        assert_eq!(ps.num_points(), 1);
        assert_eq!(ps.entries_seen(), 1);
    }

    #[test]
    fn ref_count_frees_slot() {
        let mut ps = PointStore::new(2, 1, 8, false);
        let idx = ps.add(&[1.0, 2.0]).unwrap();
        ps.inc_ref(idx);
        assert_eq!(ps.size, 1);
        ps.dec_ref(idx);
        assert_eq!(ps.size, 0);
    }

    #[rstest]
    #[case::window_2(vec![1.0, 2.0], vec![1.0, 2.0])]
    #[case::window_3(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0])]
    #[case::window_4(vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 2.0, 3.0, 4.0])]
    fn shingling_shifts_buffer(#[case] series: Vec<f32>, #[case] expected: Vec<f32>) {
        let shingle_size = series.len();
        let mut ps = PointStore::new(1, shingle_size, 8, true);
        for point in &series[..shingle_size - 1] {
            let _ = ps.shingled_point(&[*point]).unwrap();
        }

        let full = ps.shingled_point(&[series[shingle_size - 1]]).unwrap();
        assert_eq!(full, expected);
    }

    #[test]
    fn is_equal_works() {
        let mut ps = PointStore::new(2, 1, 8, false);
        let idx = ps.add(&[3.0, 4.0]).unwrap();
        assert!(ps.is_equal(&[3.0, 4.0], idx));
        assert!(!ps.is_equal(&[3.0, 5.0], idx));
    }

    #[rstest]
    #[case::no_missing([false, false, false, false], 8.0)]
    #[case::alternate_missing([false, true, false, true], 3.0)]
    #[case::all_missing([true, true, true, true], 0.0)]
    fn l1_helpers_match_expected(#[case] missing: [bool; 4], #[case] expected_partial: f64) {
        let q = [1.0f32, -1.0, 3.5, 0.0];
        let s = [0.0f32, 2.0, 1.5, -2.0];

        let full = l1_distance_slices(&q, &s);
        let partial = l1_distance_slices_ignore_missing(&q, &s, &missing);

        assert_abs_diff_eq!(full, 8.0, epsilon = 1e-12);
        assert_abs_diff_eq!(partial, expected_partial, epsilon = 1e-12);
    }

    #[rstest]
    #[case::lookahead_0(0, vec![4, 5])]
    #[case::lookahead_1(1, vec![2, 3])]
    #[case::lookahead_2(2, vec![0, 1])]
    fn lookahead_offset_and_indices_are_consistent(
        #[case] look_ahead: usize,
        #[case] expected_indices: Vec<usize>,
    ) {
        let ps = PointStore::new(2, 3, 8, true);
        let offset = lookahead_offset(ps.dim(), ps.input_dim(), look_ahead);

        assert_eq!(offset, expected_indices[0]);
        assert_eq!(ps.next_indices(look_ahead), expected_indices);
    }

    #[test]
    fn missing_indices_with_lookahead_maps_base_indices() {
        let ps = PointStore::new(2, 3, 8, true);
        assert_eq!(ps.missing_indices_with_lookahead(1, &[0, 1]), vec![2, 3]);
    }

    #[cfg(feature = "std")]
    mod proptest_tests {
        use super::*;
        use approx::abs_diff_eq;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn l1_distance_symmetric(
                a0 in -100f32..100f32,
                a1 in -100f32..100f32,
                b0 in -100f32..100f32,
                b1 in -100f32..100f32,
            ) {
                let a = [a0, a1];
                let b = [b0, b1];
                let d_ab = l1_distance_slices(&a, &b);
                let d_ba = l1_distance_slices(&b, &a);
                prop_assert!(
                    abs_diff_eq!(d_ab, d_ba, epsilon = 1e-9),
                    "d_ab={d_ab} d_ba={d_ba}"
                );
            }

            #[test]
            fn l1_distance_non_negative(
                a0 in -100f32..100f32,
                a1 in -100f32..100f32,
                b0 in -100f32..100f32,
                b1 in -100f32..100f32,
            ) {
                let a = [a0, a1];
                let b = [b0, b1];
                let d = l1_distance_slices(&a, &b);
                prop_assert!(d >= 0.0, "distance={d}");
            }

            #[test]
            fn l1_missing_leq_full(
                a0 in -100f32..100f32,
                a1 in -100f32..100f32,
                b0 in -100f32..100f32,
                b1 in -100f32..100f32,
                m0 in any::<bool>(),
                m1 in any::<bool>(),
            ) {
                let a = [a0, a1];
                let b = [b0, b1];
                let missing = [m0, m1];
                let full = l1_distance_slices(&a, &b);
                let partial = l1_distance_slices_ignore_missing(&a, &b, &missing);
                prop_assert!(partial <= full + 1e-9, "partial={partial} > full={full}");
            }

            #[test]
            fn l1_all_missing_is_zero(
                a0 in -100f32..100f32,
                a1 in -100f32..100f32,
                b0 in -100f32..100f32,
                b1 in -100f32..100f32,
            ) {
                let a = [a0, a1];
                let b = [b0, b1];
                let d = l1_distance_slices_ignore_missing(&a, &b, &[true, true]);
                prop_assert_eq!(d, 0.0);
            }
        }
    }
}
