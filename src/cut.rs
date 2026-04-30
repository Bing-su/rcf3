use itertools::izip;

use crate::bounding_box::BoundingBox;

/// Return the dimension index that `pos` falls into when walking the cumulative range array.
///
/// `pos` is in `[0, total)`.  Dimensions with `ranges[i] == 0.0` are skipped.
fn select_cut_dim(ranges: &[f64], mut pos: f64) -> usize {
    let mut last_nonzero = ranges.len() - 1;
    for (i, &r) in ranges.iter().enumerate() {
        if r > 0.0 {
            last_nonzero = i;
        }
        if pos < r {
            return i;
        }
        pos -= r;
    }
    last_nonzero
}

/// Clamp `raw` into the half-open interval `[lo, hi)`.
fn clamp_cut_val(raw: f32, lo: f32, hi: f32) -> f32 {
    if raw <= lo {
        lo
    } else if raw >= hi {
        lo + (hi - lo) * 0.999_999_9
    } else {
        raw
    }
}

/// A single random cut: split on `dim` at threshold `val`.
#[derive(Clone, Debug)]
pub struct Cut {
    pub dim: usize,
    pub val: f32,
}

/// Choose a random cut that separates `point` from `bbox`.
///
/// `factor` is a uniform random value in `[0, 1)` supplied by the caller.
///
/// Returns `(Cut, separation)` where `separation = true` means the cut places
/// `point` on the opposite side from the existing box content.
/// Returns `None` when `bbox` is a degenerate single-point box that already
/// equals `point` (no cut possible).
pub fn random_cut(bbox: &BoundingBox, point: &[f32], factor: f64) -> Option<(Cut, bool)> {
    // Per-dimension extended range: covers both the existing box and the new point.
    let extended: Vec<f64> = izip!(&bbox.min, &bbox.max, point)
        .map(|(&bmin, &bmax, &p)| (bmax.max(p) - bmin.min(p)) as f64)
        .collect();

    let total: f64 = extended.iter().sum();

    if total == 0.0 {
        // point coincides with a degenerate single-point box; nothing to cut.
        return None;
    }

    // Walk the dimensions to find which one the random position falls into.
    let pos = factor * total;
    let cut_dim = select_cut_dim(&extended, pos);
    // pos after subtracting preceding dimensions — recompute for the chosen dim.
    let preceding: f64 = extended[..cut_dim].iter().sum();
    let pos_in_dim = (pos - preceding).max(0.0);

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
    use super::*;

    fn simple_bbox() -> BoundingBox {
        BoundingBox::from_two_points(&[0.0, 0.0], &[1.0, 1.0])
    }

    #[test]
    fn cut_outside_point_is_separating() {
        let bbox = simple_bbox();
        let point = &[5.0f32, 0.5];
        // point is far outside; all random factors should give a separating cut
        let (cut, sep) = random_cut(&bbox, point, 0.5).unwrap();
        // dim 0 has the excess, so cut should be on dim 0
        assert_eq!(cut.dim, 0);
        assert!(sep);
    }

    #[test]
    fn degenerate_box_equal_to_point_returns_none() {
        let bbox = BoundingBox::from_point(&[1.0f32, 2.0]);
        assert!(random_cut(&bbox, &[1.0, 2.0], 0.5).is_none());
    }

    #[test]
    fn select_cut_dim_picks_correct_bucket() {
        let ranges = [0.0f64, 2.0, 3.0, 1.0];
        assert_eq!(select_cut_dim(&ranges, 0.0), 1); // pos=0 falls in bucket 1
        assert_eq!(select_cut_dim(&ranges, 1.5), 1); // still bucket 1
        assert_eq!(select_cut_dim(&ranges, 2.0), 2); // boundary: enters bucket 2
        assert_eq!(select_cut_dim(&ranges, 4.9), 2); // still bucket 2 (ranges[2]=3 covers [2,5))
    }

    #[test]
    fn select_cut_dim_skips_zero_ranges() {
        // All-zero except last: must return last non-zero fallback.
        let ranges = [0.0f64, 0.0, 5.0];
        assert_eq!(select_cut_dim(&ranges, 0.0), 2);
    }

    #[test]
    fn clamp_cut_val_stays_in_range() {
        assert_eq!(clamp_cut_val(-1.0, 0.0, 1.0), 0.0);
        assert!(clamp_cut_val(1.0, 0.0, 1.0) < 1.0);
        assert_eq!(clamp_cut_val(0.5, 0.0, 1.0), 0.5);
    }

    #[test]
    fn cut_value_inside_extended_range() {
        let bbox = simple_bbox();
        let point = &[0.5f32, 0.5];
        // point is inside box; cut must be inside the box range
        for factor in [0.0, 0.25, 0.5, 0.75, 0.99] {
            let (cut, _) = random_cut(&bbox, point, factor).unwrap();
            let lo = bbox.min[cut.dim].min(point[cut.dim]);
            let hi = bbox.max[cut.dim].max(point[cut.dim]);
            assert!(
                cut.val >= lo && cut.val < hi,
                "cut.val={} not in [{lo},{hi})",
                cut.val
            );
        }
    }
}
