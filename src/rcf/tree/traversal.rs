#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;

use super::{
    RcfTree, blend_with_cut_probability, child_for_query, consider_impute_candidate, nn_threshold,
    should_descend_primary, should_descend_secondary, split_children,
};
use crate::rcf::{
    forest::NeighborCandidate,
    node_arena::Node,
    point_store::PointStore,
    score::{Attribution, ScoreMode},
};

impl RcfTree {
    // -----------------------------------------------------------------------
    // Score traversal
    // -----------------------------------------------------------------------

    /// Compute the anomaly score for `query` using the given `mode`.
    ///
    /// Returns the normalized anomaly score as an `f64`.
    pub fn raw_score(&self, query: &[f32], point_store: &PointStore, mode: &ScoreMode) -> f64 {
        if self.is_effectively_empty() {
            return 0.0;
        }
        let raw = self.score_recursive(self.root, query, point_store, 0, mode);
        mode.normalize(raw, self.tree_mass)
    }

    fn score_recursive(
        &self,
        node_id: usize,
        query: &[f32],
        point_store: &PointStore,
        depth: usize,
        mode: &ScoreMode,
    ) -> f64 {
        match self.arena.get(node_id) {
            Node::Leaf { point_idx, mass } => {
                if point_store.is_equal(query, *point_idx) {
                    mode.damp(*mass, self.tree_mass) * mode.score_seen(depth, *mass)
                } else {
                    mode.score_unseen(depth, *mass)
                }
            }
            Node::Internal {
                left,
                right,
                cut_dim,
                cut_val,
                mass,
                bbox,
            } => {
                // Shared branch-choice helper keeps traversal expressions uniform.
                let child = child_for_query(query, *cut_dim, *cut_val, *left, *right);

                let child_score = self.score_recursive(child, query, point_store, depth + 1, mode);

                let prob = bbox.probability_of_cut(query);
                blend_with_cut_probability(prob, child_score, mode.score_unseen(depth, *mass))
            }
        }
    }

    // -----------------------------------------------------------------------
    // Attribution traversal
    // -----------------------------------------------------------------------

    /// Compute per-dimension anomaly attribution.
    ///
    /// Returns a `Vec<Attribution>` of length `dims`.
    pub fn attribution(&self, query: &[f32], mode: &ScoreMode) -> Vec<Attribution> {
        let mut attr = vec![Attribution::default(); self.dims];
        if self.is_effectively_empty() {
            return attr;
        }
        let norm = mode.normalize(1.0, self.tree_mass);
        self.attribution_recursive(self.root, query, 0, mode, 1.0, norm, &mut attr);
        attr
    }

    /// Accumulate attribution contributions weighted by the path probability.
    ///
    /// `weight` = product of `(1 - prob)` for all ancestors, starts at 1.0.
    /// `norm`   = mode normalizer pre-computed as `normalize(1.0, tree_mass)`.
    #[allow(clippy::too_many_arguments)]
    fn attribution_recursive(
        &self,
        node_id: usize,
        query: &[f32],
        depth: usize,
        mode: &ScoreMode,
        weight: f64,
        norm: f64,
        attr: &mut Vec<Attribution>,
    ) {
        match self.arena.get(node_id) {
            Node::Leaf { .. } => {
                // Leaf contribution is not attributed to a specific dimension;
                // it represents the expected score of a point already in the tree.
            }
            Node::Internal {
                left,
                right,
                cut_dim,
                cut_val,
                mass,
                bbox,
            } => {
                let child = child_for_query(query, *cut_dim, *cut_val, *left, *right);

                let prob = bbox.probability_of_cut(query);
                if prob > 0.0 {
                    let base = mode.score_unseen(depth, *mass);
                    // contribution = weight * prob * base * norm
                    let contribution = weight * prob * base * norm;
                    if query[*cut_dim] <= *cut_val {
                        attr[*cut_dim].above += contribution;
                    } else {
                        attr[*cut_dim].below += contribution;
                    }
                    // Recurse with reduced weight.
                    self.attribution_recursive(
                        child,
                        query,
                        depth + 1,
                        mode,
                        weight * (1.0 - prob),
                        norm,
                        attr,
                    );
                } else {
                    // No probability of isolation at this node; continue descent.
                    self.attribution_recursive(child, query, depth + 1, mode, weight, norm, attr);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Density traversal
    // -----------------------------------------------------------------------

    /// Density estimate at `query` (uses the displacement score function).
    pub fn density(&self, query: &[f32], point_store: &PointStore) -> f64 {
        if self.is_effectively_empty() {
            return 0.0;
        }
        self.density_recursive(self.root, query, point_store)
    }

    fn density_recursive(&self, node_id: usize, query: &[f32], point_store: &PointStore) -> f64 {
        // Density uses score_unseen_displacement = y (tree mass), normalizer = identity.
        match self.arena.get(node_id) {
            Node::Leaf { point_idx, mass } => {
                if point_store.is_equal(query, *point_idx) {
                    *mass as f64
                } else {
                    0.0
                }
            }
            Node::Internal {
                left,
                right,
                cut_dim,
                cut_val,
                mass,
                bbox,
            } => {
                let child = child_for_query(query, *cut_dim, *cut_val, *left, *right);

                let child_density = self.density_recursive(child, query, point_store);

                let prob = bbox.probability_of_cut(query);
                // density mode: score_unseen(depth, mass) = mass (weighted by depth-inverse)
                blend_with_cut_probability(prob, child_density, *mass as f64)
            }
        }
    }

    // -----------------------------------------------------------------------
    // Near-neighbor traversal
    // -----------------------------------------------------------------------

    /// Collect candidate neighbours from this tree.
    ///
    /// Candidates are leaf points that would receive a high isolation score
    /// relative to `query`. Callers should deduplicate/merge across trees.
    pub fn near_neighbors(
        &self,
        query: &[f32],
        point_store: &PointStore,
        mode: &ScoreMode,
        percentile: usize,
    ) -> Vec<NeighborCandidate> {
        let mut results = Vec::new();
        self.near_neighbors_into(query, point_store, mode, percentile, &mut results);
        results
    }

    pub(crate) fn near_neighbors_into(
        &self,
        query: &[f32],
        point_store: &PointStore,
        mode: &ScoreMode,
        percentile: usize,
        results: &mut Vec<NeighborCandidate>,
    ) {
        if self.is_effectively_empty() {
            return;
        }
        let threshold = nn_threshold(percentile);
        self.nn_recursive(self.root, query, point_store, 0, mode, threshold, results);
    }

    #[allow(clippy::too_many_arguments)]
    fn nn_recursive(
        &self,
        node_id: usize,
        query: &[f32],
        point_store: &PointStore,
        depth: usize,
        mode: &ScoreMode,
        threshold: f64,
        results: &mut Vec<NeighborCandidate>,
    ) {
        match self.arena.get(node_id) {
            Node::Leaf { point_idx, mass } => {
                let score = mode.normalize(mode.score_unseen(depth, *mass), self.tree_mass);
                let dist = point_store.l1_distance(query, *point_idx);
                results.push(NeighborCandidate {
                    score,
                    point_idx: *point_idx,
                    distance: dist,
                });
            }
            Node::Internal {
                left,
                right,
                cut_dim,
                cut_val,
                mass: _,
                bbox,
            } => {
                let prob = bbox.probability_of_cut(query);
                // Only descend if this subtree is a viable candidate
                if should_descend_primary(prob, depth, threshold) {
                    let (primary, secondary) =
                        split_children(query[*cut_dim], *cut_val, *left, *right);
                    self.nn_recursive(
                        primary,
                        query,
                        point_store,
                        depth + 1,
                        mode,
                        threshold,
                        results,
                    );
                    if should_descend_secondary(prob) {
                        self.nn_recursive(
                            secondary,
                            query,
                            point_store,
                            depth + 1,
                            mode,
                            threshold,
                            results,
                        );
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Conditional field / Imputation traversal
    // -----------------------------------------------------------------------

    /// Find the leaf that best matches `query` with the given `missing_dims`.
    ///
    /// Returns the best matching candidate. Missing dimensions are
    /// treated as marginalized out (both children are explored when the cut
    /// falls on a missing dimension).
    pub fn conditional_field(
        &self,
        query: &[f32],
        missing: &[bool],
        point_store: &PointStore,
        centrality: f64,
        seed: u64,
    ) -> Option<NeighborCandidate> {
        if self.is_effectively_empty() {
            return None;
        }
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        self.impute_recursive(
            self.root,
            query,
            missing,
            point_store,
            centrality,
            &mut rng,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn impute_recursive(
        &self,
        node_id: usize,
        query: &[f32],
        missing: &[bool],
        point_store: &PointStore,
        centrality: f64,
        rng: &mut Xoshiro256PlusPlus,
        best: Option<NeighborCandidate>,
    ) -> Option<NeighborCandidate> {
        match self.arena.get(node_id) {
            Node::Leaf { point_idx, mass } => {
                let dist = point_store.l1_distance_ignore_missing(query, *point_idx, missing);
                let score = *mass as f64;
                let candidate = NeighborCandidate {
                    score,
                    point_idx: *point_idx,
                    distance: dist,
                };
                consider_impute_candidate(best, candidate, centrality, rng)
            }
            Node::Internal {
                left,
                right,
                cut_dim,
                cut_val,
                ..
            } => {
                if missing[*cut_dim] {
                    // Explore both branches while carrying forward the current best.
                    let best = self.impute_recursive(
                        *left,
                        query,
                        missing,
                        point_store,
                        centrality,
                        rng,
                        best,
                    );
                    self.impute_recursive(
                        *right,
                        query,
                        missing,
                        point_store,
                        centrality,
                        rng,
                        best,
                    )
                } else {
                    let child = child_for_query(query, *cut_dim, *cut_val, *left, *right);
                    self.impute_recursive(child, query, missing, point_store, centrality, rng, best)
                }
            }
        }
    }
}
