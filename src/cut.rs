use crate::bounding_box::BoundingBox;

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
    let dim = bbox.min.len();

    // Per-dimension extended range: covers both the existing box and the new point.
    let extended: Vec<f64> = (0..dim)
        .map(|i| {
            let lo = bbox.min[i].min(point[i]);
            let hi = bbox.max[i].max(point[i]);
            (hi - lo) as f64
        })
        .collect();

    let total: f64 = extended.iter().sum();

    if total == 0.0 {
        // point coincides with a degenerate single-point box; nothing to cut.
        return None;
    }

    // Walk the dimensions to find which one the random position falls into.
    let mut pos = factor * total;
    let mut cut_dim = dim - 1; // fallback to last non-zero dimension
    for i in 0..dim {
        if extended[i] > 0.0 {
            cut_dim = i;
        }
        if pos < extended[i] {
            cut_dim = i;
            break;
        }
        pos -= extended[i];
    }

    let lo = bbox.min[cut_dim].min(point[cut_dim]);
    let hi = bbox.max[cut_dim].max(point[cut_dim]);
    let range = hi - lo;

    // Clamp cut value into [lo, hi).
    let raw_val = lo + pos as f32;
    let cut_val = if raw_val <= lo {
        lo
    } else if raw_val >= hi {
        lo + range * 0.999_999_9
    } else {
        raw_val
    };

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
