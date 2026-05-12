/// How anomaly scores are computed and normalised.
///
/// This replaces the original `VisitorInfo` struct and its collection of
/// function-pointer fields with a simple enum.  Custom modes can be
/// constructed via [`ScoreMode::custom`].
#[derive(Clone, Debug)]
pub struct ScoreMode {
    pub score_seen: fn(usize, usize) -> f64,
    pub score_unseen: fn(usize, usize) -> f64,
    pub damp: fn(usize, usize) -> f64,
    pub normalizer: fn(f64, usize) -> f64,
}

// ---------------------------------------------------------------------------
// Score functions (matching the AWS reference implementation)
// ---------------------------------------------------------------------------

/// Standard isolation score for a point already in the tree.
///
/// `x` = depth, `y` = leaf mass.
pub fn score_seen(x: usize, y: usize) -> f64 {
    1.0 / (x as f64 + f64::log2(1.0 + y as f64))
}

/// Standard isolation score for a point *not* already in the tree.
pub fn score_unseen(x: usize, _y: usize) -> f64 {
    1.0 / (x as f64 + 1.0)
}

/// Standard normalizer: multiplies by log₂(1 + tree_mass).
pub fn normalizer(x: f64, y: usize) -> f64 {
    x * f64::log2(1.0 + y as f64)
}

/// Damping applied when the query is a duplicate of a leaf point.
pub fn damp(x: usize, y: usize) -> f64 {
    if y == 0 {
        return 1.0;
    }
    1.0 - (x as f64) / (2.0 * y as f64)
}

/// Displacement score for *seen* points (ignores depth, uses mass).
pub fn score_seen_displacement(_x: usize, y: usize) -> f64 {
    1.0 / (1.0 + y as f64)
}

/// Displacement score for *unseen* points.
pub fn score_unseen_displacement(_x: usize, y: usize) -> f64 {
    y as f64
}

/// Displacement normalizer: `score / (1 + tree_mass)`.
pub fn displacement_normalizer(x: f64, y: usize) -> f64 {
    x / (1.0 + y as f64)
}

/// Identity normalizer (used for density mode).
pub fn identity(x: f64, _y: usize) -> f64 {
    x
}

// ---------------------------------------------------------------------------
// ScoreMode constructors
// ---------------------------------------------------------------------------

impl ScoreMode {
    /// Standard anomaly-score mode (matches the AWS reference default).
    pub fn standard() -> Self {
        ScoreMode {
            score_seen,
            score_unseen,
            damp,
            normalizer,
        }
    }

    /// Displacement-based score (density-sensitive).
    pub fn displacement() -> Self {
        ScoreMode {
            score_seen: score_seen_displacement,
            score_unseen: score_unseen_displacement,
            damp,
            normalizer: displacement_normalizer,
        }
    }

    /// Density estimation mode.
    pub fn density() -> Self {
        ScoreMode {
            score_seen: score_unseen_displacement,
            score_unseen: score_unseen_displacement,
            damp,
            normalizer: identity,
        }
    }

    /// Build a fully custom mode.
    pub fn custom(
        score_seen: fn(usize, usize) -> f64,
        score_unseen: fn(usize, usize) -> f64,
        damp: fn(usize, usize) -> f64,
        normalizer: fn(f64, usize) -> f64,
    ) -> Self {
        ScoreMode {
            score_seen,
            score_unseen,
            damp,
            normalizer,
        }
    }

    pub fn score_seen(&self, depth: usize, mass: usize) -> f64 {
        (self.score_seen)(depth, mass)
    }

    pub fn score_unseen(&self, depth: usize, mass: usize) -> f64 {
        (self.score_unseen)(depth, mass)
    }

    pub fn damp(&self, mass: usize, tree_mass: usize) -> f64 {
        (self.damp)(mass, tree_mass)
    }

    pub fn normalize(&self, raw: f64, tree_mass: usize) -> f64 {
        (self.normalizer)(raw, tree_mass)
    }
}

// ---------------------------------------------------------------------------
// Attribution result
// ---------------------------------------------------------------------------

/// Per-dimension anomaly attribution.
///
/// For each dimension `i`:
/// - `below` = contribution from cuts whose threshold is *above* the query
///   value (i.e., the query was isolated because it is too *small* in dim `i`).
/// - `above` = contribution from cuts whose threshold is *below* the query
///   value (isolated because it is too *large* in dim `i`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Attribution {
    pub below: f64,
    pub above: f64,
}

impl Attribution {
    /// Sum of both components: `below + above`.
    pub fn total(self) -> f64 {
        self.below + self.above
    }

    /// Scale both components by `factor`, returning a new `Attribution`.
    pub fn scale(self, factor: f64) -> Self {
        Attribution {
            below: self.below * factor,
            above: self.above * factor,
        }
    }
}

impl Default for Attribution {
    fn default() -> Self {
        Attribution {
            below: 0.0,
            above: 0.0,
        }
    }
}

impl std::ops::Add for Attribution {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Attribution {
            below: self.below + rhs.below,
            above: self.above + rhs.above,
        }
    }
}

impl std::ops::AddAssign for Attribution {
    fn add_assign(&mut self, rhs: Self) {
        self.below += rhs.below;
        self.above += rhs.above;
    }
}

/// Sum of all attribution components equals `score` (up to floating-point error).
pub fn attribution_total(attr: &[Attribution]) -> f64 {
    attr.iter().map(|a| a.total()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_score_unseen_decreases_with_depth() {
        let s0 = score_unseen(0, 1);
        let s1 = score_unseen(1, 1);
        let s5 = score_unseen(5, 1);
        assert!(s0 > s1 && s1 > s5);
    }

    #[test]
    fn normalizer_scales_with_tree_mass() {
        let s = normalizer(1.0, 256);
        assert!(s > 1.0); // log2(257) ≈ 8
    }
}
