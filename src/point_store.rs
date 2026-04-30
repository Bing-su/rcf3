use crate::error::{RcfError, Result};
use itertools::izip;
use ndarray::{Array2, ArrayView1, s};
use serde::{Deserialize, Serialize};

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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PointStore {
    /// Full dimensionality: `input_dim * shingle_size`.
    pub dim: usize,
    /// Base input dimensionality (before shingling).
    pub input_dim: usize,
    /// Shingle size (temporal window).
    pub shingle_size: usize,
    /// Whether the store manages the rolling shingle buffer.
    pub internal_shingling: bool,

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
    pub entries_seen: u64,
}

impl PointStore {
    /// Create a new store.
    ///
    /// `input_dim` × `shingle_size` = full model dimension.
    pub fn new(
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
    pub fn shingled_point(&mut self, base: &[f32]) -> Result<Vec<f32>> {
        if self.internal_shingling {
            if base.len() != self.input_dim {
                return Err(RcfError::DimensionMismatch {
                    expected: self.input_dim,
                    got: base.len(),
                });
            }
            // Shift buffer left by input_dim, then append new values.
            self.shingle_buf.copy_within(self.input_dim.., 0);
            let start = self.dim - self.input_dim;
            self.shingle_buf[start..].copy_from_slice(base);
            Ok(self.shingle_buf.clone())
        } else {
            if base.len() != self.dim {
                return Err(RcfError::DimensionMismatch {
                    expected: self.dim,
                    got: base.len(),
                });
            }
            Ok(base.to_vec())
        }
    }

    /// Peek at the current shingled point without advancing the shingle buffer.
    pub fn current_shingled(&self) -> &[f32] {
        &self.shingle_buf
    }

    // -----------------------------------------------------------------------
    // Allocation / deallocation
    // -----------------------------------------------------------------------

    /// Store `point` and return its index.
    ///
    /// The reference count is initialised to 0; callers must call
    /// [`inc_ref`] for each tree that accepts this point.
    pub fn add(&mut self, point: &[f32]) -> Result<usize> {
        if point.len() != self.dim {
            return Err(RcfError::DimensionMismatch {
                expected: self.dim,
                got: point.len(),
            });
        }

        let idx = self.allocate_slot()?;
        self.store.row_mut(idx).assign(&ArrayView1::from(point));
        self.occupied[idx] = true;
        self.ref_count[idx] = 0;
        self.size += 1;
        self.entries_seen += 1;
        Ok(idx)
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
    pub fn inc_ref(&mut self, idx: usize) {
        self.ref_count[idx] += 1;
    }

    /// Decrement reference count; free the slot when it reaches zero.
    pub fn dec_ref(&mut self, idx: usize) {
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
    pub fn get(&self, idx: usize) -> &[f32] {
        debug_assert!(self.occupied[idx], "accessing unoccupied slot {idx}");
        // `row()` returns a temporary ArrayView1 whose lifetime Rust can't
        // propagate through `as_slice()`.  Instead, grab the underlying
        // contiguous flat slice and index into it directly — same data,
        // lifetime correctly tied to `&self`.
        let flat = self.store.as_slice().expect("store must be contiguous");
        &flat[idx * self.dim..(idx + 1) * self.dim]
    }

    /// Return the point at `idx` as an ndarray view (zero-copy).
    pub fn point_view(&self, idx: usize) -> ArrayView1<'_, f32> {
        debug_assert!(self.occupied[idx], "accessing unoccupied slot {idx}");
        self.store.row(idx)
    }

    /// Check whether the stored point at `idx` equals `point` component-wise.
    pub fn is_equal(&self, point: &[f32], idx: usize) -> bool {
        self.store.row(idx) == ArrayView1::from(point)
    }

    /// L1 distance between `query` and the stored point at `idx`.
    pub fn l1_distance(&self, query: &[f32], idx: usize) -> f64 {
        let stored = self.get(idx);
        l1_distance_slices(query, stored)
    }

    /// L1 distance ignoring `missing` dimensions.
    pub fn l1_distance_ignore_missing(&self, query: &[f32], idx: usize, missing: &[bool]) -> f64 {
        let stored = self.get(idx);
        l1_distance_slices_ignore_missing(query, stored, missing)
    }

    /// Return a copy of the point at `idx`.
    pub fn copy_point(&self, idx: usize) -> Vec<f32> {
        self.get(idx).to_vec()
    }

    // -----------------------------------------------------------------------
    // Imputation helpers
    // -----------------------------------------------------------------------

    /// Compute the indices of the `missing` base-dimensions for look-ahead
    /// step `look_ahead` inside the shingle buffer.
    ///
    /// For a shingle buffer `[t-k+1, …, t]` (k slots of `input_dim` each),
    /// `next_indices(0)` returns the positions that the *next* base observation
    /// would fill (the newest slot in the buffer).
    pub fn next_indices(&self, look_ahead: usize) -> Vec<usize> {
        let offset = lookahead_offset(self.dim, self.input_dim, look_ahead);
        (0..self.input_dim).map(|i| offset + i).collect()
    }

    /// Convert `missing` base-dimension indices into full-dimension indices
    /// within the shingled vector, shifted by `look_ahead` steps.
    pub fn missing_indices_with_lookahead(
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

    pub fn num_points(&self) -> usize {
        self.size
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_get() {
        let mut ps = PointStore::new(2, 1, 8, false);
        let idx = ps.add(&[1.0, 2.0]).unwrap();
        assert_eq!(ps.get(idx), &[1.0f32, 2.0]);
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

    #[test]
    fn shingling_shifts_buffer() {
        let mut ps = PointStore::new(1, 3, 8, true);
        let _ = ps.shingled_point(&[1.0]).unwrap();
        let _ = ps.shingled_point(&[2.0]).unwrap();
        let full = ps.shingled_point(&[3.0]).unwrap();
        assert_eq!(full, vec![1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn is_equal_works() {
        let mut ps = PointStore::new(2, 1, 8, false);
        let idx = ps.add(&[3.0, 4.0]).unwrap();
        assert!(ps.is_equal(&[3.0, 4.0], idx));
        assert!(!ps.is_equal(&[3.0, 5.0], idx));
    }

    #[test]
    fn l1_helpers_match_expected() {
        let q = [1.0f32, -1.0, 3.5, 0.0];
        let s = [0.0f32, 2.0, 1.5, -2.0];
        let missing = [false, true, false, true];

        let full = l1_distance_slices(&q, &s);
        let partial = l1_distance_slices_ignore_missing(&q, &s, &missing);

        assert!((full - 8.0).abs() < 1e-12);
        assert!((partial - 3.0).abs() < 1e-12);
    }

    #[test]
    fn lookahead_offset_and_indices_are_consistent() {
        let ps = PointStore::new(2, 3, 8, true);
        let offset0 = lookahead_offset(ps.dim, ps.input_dim, 0);
        let offset1 = lookahead_offset(ps.dim, ps.input_dim, 1);

        assert_eq!(offset0, 4);
        assert_eq!(offset1, 2);
        assert_eq!(ps.next_indices(0), vec![4, 5]);
        assert_eq!(ps.next_indices(1), vec![2, 3]);
        assert_eq!(ps.missing_indices_with_lookahead(1, &[0, 1]), vec![2, 3]);
    }
}
