use super::bounding_box::BoundingBox;

/// Return the dimension index and local position that `pos` falls into when walking ranges.
///
/// Dimensions with `ranges[i] == 0.0` are skipped. If `pos` falls at or after
/// the end of the cumulative range, it maps to the upper edge of the last
/// nonzero range.
fn select_cut_bucket(ranges: impl IntoIterator<Item = f64>, mut pos: f64) -> (usize, f64) {
    let mut last_nonzero = 0;
    let mut last_nonzero_range = 0.0;
    for (i, range) in ranges.into_iter().enumerate() {
        if range > 0.0 {
            last_nonzero = i;
            last_nonzero_range = range;
        }
        if pos < range {
            return (i, pos);
        }
        pos -= range;
    }
    (last_nonzero, last_nonzero_range)
}

#[cfg(test)]
fn select_cut_dim(ranges: &[f64], pos: f64) -> usize {
    select_cut_bucket(ranges.iter().copied(), pos).0
}

/// Select a random cut dimension and return its position within that dimension.
fn select_cut_dim_from_box(bbox: &BoundingBox, point: &[f32], pos: f64) -> (usize, f64) {
    debug_assert_eq!(bbox.min.len(), bbox.max.len());
    debug_assert_eq!(bbox.min.len(), point.len());

    select_cut_bucket(
        (0..point.len()).map(|i| (bbox.max[i].max(point[i]) - bbox.min[i].min(point[i])) as f64),
        pos,
    )
}

/// Clamp `raw` into the half-open interval `[lo, hi)`.
fn clamp_cut_val(raw: f32, lo: f32, hi: f32) -> f32 {
    raw.clamp(lo, hi.next_down().max(lo))
}

/// A single random cut: split on `dim` at threshold `val`.
#[derive(Clone, Debug)]
pub(super) struct Cut {
    pub(super) dim: usize,
    pub(super) val: f32,
}

/// Choose a random cut that separates `point` from `bbox`.
///
/// `factor` is a uniform random value in `[0, 1)` supplied by the caller.
///
/// Returns `(Cut, separation)` where `separation = true` means the cut places
/// `point` on the opposite side from the existing box content.
/// Returns `None` when `bbox` is a degenerate single-point box that already
/// equals `point` (no cut possible).
pub(super) fn random_cut(bbox: &BoundingBox, point: &[f32], factor: f64) -> Option<(Cut, bool)> {
    // Per-dimension extended range: covers both the existing box and the new point.
    let total: f64 = (0..point.len())
        .map(|i| (bbox.max[i].max(point[i]) - bbox.min[i].min(point[i])) as f64)
        .sum();

    if total == 0.0 {
        // point coincides with a degenerate single-point box; nothing to cut.
        return None;
    }

    // Walk the dimensions to find which one the random position falls into.
    let pos = factor * total;
    let (cut_dim, pos_in_dim) = select_cut_dim_from_box(bbox, point, pos);

    let lo = bbox.min[cut_dim].min(point[cut_dim]);
    let hi = bbox.max[cut_dim].max(point[cut_dim]);

    let raw_val = lo + pos_in_dim as f32;
    let cut_val = clamp_cut_val(raw_val, lo, hi);

    // Separation: the new cut isolates `point` from the existing box content.
    let in_box_lo = bbox.min[cut_dim];
    let in_box_hi = bbox.max[cut_dim];
    let separation = (point[cut_dim] <= cut_val && cut_val < in_box_lo)
        || (in_box_hi <= cut_val && cut_val < point[cut_dim]);

    Some((
        Cut {
            dim: cut_dim,
            val: cut_val,
        },
        separation,
    ))
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use rstest::rstest;

    use super::*;

    fn simple_bbox() -> BoundingBox {
        BoundingBox::from_two_points(&[0.0, 0.0], &[1.0, 1.0])
    }

    /// factor=0.0: cut lands at lo of extended range, still inside box → no separation.
    /// factor>0: cut rises above box max → separates the outside point.
    #[rstest]
    #[case::factor_zero_no_sep(0.0, false)]
    #[case::factor_mid_separates(0.5, true)]
    #[case::factor_high_separates(0.8, true)]
    fn cut_outside_point_separation_by_factor(#[case] factor: f64, #[case] expected_sep: bool) {
        let bbox = simple_bbox();
        let point = &[5.0f32, 0.5];
        let (cut, sep) = random_cut(&bbox, point, factor).unwrap();
        assert_eq!(cut.dim, 0);
        assert_eq!(sep, expected_sep);
    }

    #[test]
    fn degenerate_box_equal_to_point_returns_none() {
        let bbox = BoundingBox::from_point(&[1.0f32, 2.0]);
        assert!(random_cut(&bbox, &[1.0, 2.0], 0.5).is_none());
    }

    #[rstest]
    #[case::first_non_zero([0.0, 2.0, 3.0, 1.0], 0.0, 1)]
    #[case::middle_bucket([0.0, 2.0, 3.0, 1.0], 1.5, 1)]
    #[case::boundary_next_bucket([0.0, 2.0, 3.0, 1.0], 2.0, 2)]
    #[case::late_in_bucket([0.0, 2.0, 3.0, 1.0], 4.9, 2)]
    #[case::skip_zero_ranges([0.0, 0.0, 5.0, 0.0], 0.0, 2)]
    fn select_cut_dim_handles_bucket_boundaries(
        #[case] ranges: [f64; 4],
        #[case] pos: f64,
        #[case] expected_dim: usize,
    ) {
        assert_eq!(select_cut_dim(&ranges, pos), expected_dim);
    }

    #[rstest]
    #[case::below_lower(-1.0, 0.0)]
    #[case::at_upper(1.0, 0.999_999_9)]
    #[case::inside_range(0.5, 0.5)]
    fn clamp_cut_val_stays_in_range(#[case] raw: f32, #[case] expected: f32) {
        assert_abs_diff_eq!(
            clamp_cut_val(raw, 0.0, 1.0),
            expected,
            epsilon = f32::EPSILON
        );
    }

    #[rstest]
    #[case::f0(0.0)]
    #[case::f25(0.25)]
    #[case::f50(0.5)]
    #[case::f75(0.75)]
    #[case::f99(0.99)]
    fn cut_value_inside_extended_range(#[case] factor: f64) {
        let bbox = simple_bbox();
        let point = &[0.5f32, 0.5];
        let (cut, _) = random_cut(&bbox, point, factor).unwrap();
        let lo = bbox.min[cut.dim].min(point[cut.dim]);
        let hi = bbox.max[cut.dim].max(point[cut.dim]);
        assert!(
            cut.val >= lo && cut.val < hi,
            "cut.val={} not in [{lo},{hi})",
            cut.val
        );
    }

    #[test]
    fn random_cut_factor_one_clamps_last_bucket_to_upper_edge() {
        let bbox = simple_bbox();
        let point = &[0.5f32, 5.0];

        let (cut, separation) = random_cut(&bbox, point, 1.0).unwrap();

        assert_eq!(cut.dim, 1);
        assert_abs_diff_eq!(cut.val, 5.0f32.next_down(), epsilon = f32::EPSILON);
        assert!(separation);
    }

    #[cfg(feature = "std")]
    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn clamp_cut_val_within_bounds(
                lo in -1000f32..0f32,
                hi in 0f32..1000f32,
                raw in -2000f32..2000f32,
            ) {
                let result = clamp_cut_val(raw, lo, hi);
                prop_assert!(result >= lo, "result={result} < lo={lo}");
                prop_assert!(result < hi, "result={result} >= hi={hi}");
            }

            #[test]
            fn clamp_cut_val_idempotent(
                lo in -1000f32..0f32,
                hi in 0f32..1000f32,
                raw in -2000f32..2000f32,
            ) {
                let once = clamp_cut_val(raw, lo, hi);
                let twice = clamp_cut_val(once, lo, hi);
                prop_assert_eq!(once, twice);
            }

            #[test]
            fn select_cut_dim_in_valid_range(
                r0 in 0.1f64..10.0,
                r1 in 0.1f64..10.0,
                r2 in 0.1f64..10.0,
                factor in 0.0f64..1.0,
            ) {
                let ranges = [r0, r1, r2];
                let total = r0 + r1 + r2;
                let pos = factor * total;
                let dim = select_cut_dim(&ranges, pos);
                prop_assert!(dim < ranges.len(), "dim={dim} >= len={}", ranges.len());
            }
        }
    }
}
