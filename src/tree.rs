use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    bounding_box::BoundingBox,
    cut::random_cut,
    error::{RcfError, Result},
    forest::NeighborCandidate,
    node_arena::{NULL, Node, NodeArena},
    point_store::PointStore,
    score::{Attribution, ScoreMode},
};

// ---------------------------------------------------------------------------
// RcfTree
// ---------------------------------------------------------------------------

/// A single Random Cut Tree.  All point-data is owned by the shared
/// [`PointStore`]; the tree only stores indices and bounding-box metadata.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RcfTree {
    pub(crate) root: usize,
    pub(crate) tree_mass: usize,
    arena: NodeArena,
    rng: Xoshiro256PlusPlus,
    dims: usize,
}

fn split_children(query_value: f32, cut_val: f32, left: usize, right: usize) -> (usize, usize) {
    if query_value <= cut_val {
        (left, right)
    } else {
        (right, left)
    }
}

fn split_child_only(query_value: f32, cut_val: f32, left: usize, right: usize) -> usize {
    if query_value <= cut_val { left } else { right }
}

fn nn_threshold(percentile: usize) -> f64 {
    percentile as f64 / 100.0
}

fn should_descend_primary(probability_of_cut: f64, depth: usize, threshold: f64) -> bool {
    probability_of_cut > threshold || depth == 0
}

fn should_descend_secondary(probability_of_cut: f64) -> bool {
    probability_of_cut > 0.5
}

fn consider_impute_candidate(
    best: Option<NeighborCandidate>,
    candidate: NeighborCandidate,
    centrality: f64,
    rng: &mut Xoshiro256PlusPlus,
) -> Option<NeighborCandidate> {
    match best {
        None => Some(candidate),
        Some(best_candidate) => {
            let best_dist = best_candidate.distance;
            let dist = candidate.distance;
            let accept = if centrality >= 1.0 {
                dist < best_dist
            } else {
                let r: f64 = rng.random::<f64>();
                dist < best_dist || r < 1.0 - centrality
            };
            if accept {
                Some(candidate)
            } else {
                Some(best_candidate)
            }
        }
    }
}

impl RcfTree {
    pub fn new(dims: usize, capacity: usize, seed: u64) -> Self {
        RcfTree {
            root: NULL,
            tree_mass: 0,
            arena: NodeArena::new(2 * capacity + 4),
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
            dims,
        }
    }

    /// Descend to the leaf node whose point equals `point`, returning the path
    /// as a Vec of `(node_id, sibling_id)` pairs in root-to-parent order.
    fn path_to_leaf(&self, point: &[f32]) -> (Vec<(usize, usize)>, usize) {
        let mut path: Vec<(usize, usize)> = Vec::new();
        let mut cur = self.root;
        loop {
            match self.arena.get(cur) {
                Node::Leaf { .. } => return (path, cur),
                Node::Internal {
                    left,
                    right,
                    cut_dim,
                    cut_val,
                    ..
                } => {
                    let (child, sibling) = split_children(point[*cut_dim], *cut_val, *left, *right);
                    path.push((cur, sibling));
                    cur = child;
                }
            }
        }
    }

    /// Parent of `node` given `path` (last element of path is the parent).
    fn parent_of(path: &[(usize, usize)]) -> usize {
        path.last().map(|(p, _)| *p).unwrap_or(NULL)
    }

    /// Recompute the bounding box for an internal node from its two children.
    fn recompute_bbox(arena: &mut NodeArena, node_id: usize, point_store: &PointStore) {
        let (left, right) = match arena.get(node_id) {
            Node::Internal { left, right, .. } => (*left, *right),
            Node::Leaf { .. } => return,
        };

        let bbox = subtree_bbox(arena, left, point_store).merge_with(&subtree_bbox(
            arena,
            right,
            point_store,
        ));
        let mass = arena.get(left).mass() + arena.get(right).mass();
        if let Node::Internal {
            bbox: b, mass: m, ..
        } = arena.get_mut(node_id)
        {
            *b = bbox;
            *m = mass;
        }
    }

    /// Walk from `node` up to the root (using `path`), recomputing bboxes.
    fn update_ancestors_after_change(
        arena: &mut NodeArena,
        path: &[(usize, usize)],
        point_store: &PointStore,
    ) {
        for &(ancestor, _) in path.iter().rev() {
            Self::recompute_bbox(arena, ancestor, point_store);
        }
    }

    // -----------------------------------------------------------------------
    // Insert
    // -----------------------------------------------------------------------

    /// Insert `point_idx` into the tree.  `point` is the actual coordinate
    /// vector (borrowed from the point store for computations here).
    pub fn insert(&mut self, point_idx: usize, point_store: &PointStore) -> Result<()> {
        let point = point_store.get(point_idx);

        if self.root == NULL {
            // First point in this tree.
            self.root = self.arena.alloc(Node::Leaf { point_idx, mass: 1 });
            self.tree_mass = 1;
            return Ok(());
        }

        let (path, leaf_id) = self.path_to_leaf(point);
        let leaf_point_idx = match self.arena.get(leaf_id) {
            Node::Leaf { point_idx, .. } => *point_idx,
            _ => unreachable!(),
        };

        self.tree_mass += 1;

        if point_store.is_equal(point, leaf_point_idx) {
            // Duplicate: just increment the leaf mass and update ancestors.
            if let Node::Leaf { mass, .. } = self.arena.get_mut(leaf_id) {
                *mass += 1;
            }
            Self::update_ancestors_after_change(&mut self.arena, &path, point_store);
            return Ok(());
        }

        // Different point: need a new internal node with a random cut.
        let leaf_point = point_store.get(leaf_point_idx);
        let mut current_bbox = BoundingBox::from_point(leaf_point);

        // Find the highest-in-tree cut that separates the new point.
        let mut saved_cut_dim = NULL;
        let mut saved_cut_val = 0.0f32;
        let mut insert_above: usize = leaf_id; // node below which to insert the new split
        let mut parent_above = Self::parent_of(&path); // parent of insert_above
        let mut path_below: Vec<(usize, usize)> = Vec::new(); // path from insert_above down

        // Scan from leaf upward, expanding the bounding box as we go.
        for step in 0..=path.len() {
            let factor: f64 = self.rng.random::<f64>();
            if let Some((cut, sep)) = random_cut(&current_bbox, point, factor)
                && sep
            {
                saved_cut_dim = cut.dim;
                saved_cut_val = cut.val;
                insert_above = if step == 0 {
                    leaf_id
                } else {
                    path[path.len() - step].0
                };
                parent_above = if step == path.len() {
                    NULL
                } else if step == 0 {
                    Self::parent_of(&path)
                } else {
                    path[path.len() - step - 1].0
                };
                // Everything below insert_above is in path_below.
                path_below = path[path.len() - step..].to_vec();
                break;
            }
            // Expand box upward by including the sibling subtree.
            if step < path.len() {
                let sibling = path[path.len() - 1 - step].1;
                let sib_bbox = subtree_bbox(&self.arena, sibling, point_store);
                current_bbox.merge(&sib_bbox);
            }
        }

        // If we never found a separating cut, fall back to a cut at the first
        // dimension where there is variance.
        if saved_cut_dim == NULL {
            // This happens only when the entire tree is a single repeated point.
            // Use a trivial cut between that point and the new one.
            for d in 0..self.dims {
                if (point[d] - leaf_point[d]).abs() > f32::EPSILON {
                    saved_cut_dim = d;
                    saved_cut_val = leaf_point[d].min(point[d]);
                    insert_above = leaf_id;
                    parent_above = Self::parent_of(&path);
                    path_below = Vec::new();
                    break;
                }
            }
            if saved_cut_dim == NULL {
                // Truly identical — treat as duplicate (shouldn't reach here because
                // `is_equal` would have caught it, but be defensive).
                if let Node::Leaf { mass, .. } = self.arena.get_mut(leaf_id) {
                    *mass += 1;
                }
                Self::update_ancestors_after_change(&mut self.arena, &path, point_store);
                return Ok(());
            }
        }

        // Determine which side new point and existing subtree go on.
        let new_leaf_id = self.arena.alloc(Node::Leaf { point_idx, mass: 1 });
        let (new_left, new_right) = if point[saved_cut_dim] <= saved_cut_val {
            (new_leaf_id, insert_above)
        } else {
            (insert_above, new_leaf_id)
        };

        let child_mass = self.arena.get(insert_above).mass() + 1;
        let new_bbox = subtree_bbox(&self.arena, insert_above, point_store)
            .merge_with(&BoundingBox::from_point(point));

        let new_internal = self.arena.alloc(Node::Internal {
            left: new_left,
            right: new_right,
            cut_dim: saved_cut_dim,
            cut_val: saved_cut_val,
            mass: child_mass,
            bbox: new_bbox,
        });

        // Attach new_internal in place of insert_above.
        if parent_above == NULL {
            self.root = new_internal;
        } else {
            match self.arena.get_mut(parent_above) {
                Node::Internal { left, right, .. } => {
                    if *left == insert_above {
                        *left = new_internal;
                    } else {
                        *right = new_internal;
                    }
                }
                _ => unreachable!(),
            }
        }

        // Recompute ancestors above insert_above (path_below gives the path
        // from the original root down to insert_above; we need the portion
        // above parent_above).
        let ancestor_path = if path_below.is_empty() {
            path.as_slice()
        } else {
            // path_below[0].0 == insert_above, so path ancestors are above that.
            let n = path.len() - path_below.len();
            &path[..n]
        };
        // Recompute new_internal itself first (its mass is already set; bbox is set above).
        // Then walk up.
        Self::update_ancestors_after_change(&mut self.arena, ancestor_path, point_store);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Delete
    // -----------------------------------------------------------------------

    /// Remove `point_idx` from the tree.
    pub fn delete(&mut self, point_idx: usize, point_store: &PointStore) -> Result<()> {
        if self.root == NULL {
            return Err(RcfError::EmptyTree);
        }

        let point = point_store.get(point_idx);
        let (path, leaf_id) = self.path_to_leaf(point);

        let leaf_mass = match self.arena.get(leaf_id) {
            Node::Leaf { mass, .. } => *mass,
            _ => unreachable!(),
        };

        if leaf_mass == 0 {
            return Err(RcfError::InvalidArgument(
                "leaf mass is already 0".to_string(),
            ));
        }

        self.tree_mass -= 1;

        if leaf_mass > 1 {
            // Just decrement duplicates.
            if let Node::Leaf { mass, .. } = self.arena.get_mut(leaf_id) {
                *mass -= 1;
            }
            Self::update_ancestors_after_change(&mut self.arena, &path, point_store);
            return Ok(());
        }

        // Last copy: remove the leaf and its parent internal node.
        if path.is_empty() {
            // Tree had only one point.
            self.arena.free(leaf_id);
            self.root = NULL;
            return Ok(());
        }

        let (parent_id, sibling_id) = *path.last().unwrap();
        let grandparent = if path.len() >= 2 {
            path[path.len() - 2].0
        } else {
            NULL
        };

        // Promote sibling to take parent's place.
        if grandparent == NULL {
            self.root = sibling_id;
        } else {
            match self.arena.get_mut(grandparent) {
                Node::Internal { left, right, .. } => {
                    if *left == parent_id {
                        *left = sibling_id;
                    } else {
                        *right = sibling_id;
                    }
                }
                _ => unreachable!(),
            }
        }

        self.arena.free(leaf_id);
        self.arena.free(parent_id);

        // Recompute grandparent and upward.
        let ancestor_path = &path[..path.len() - 1];
        Self::update_ancestors_after_change(&mut self.arena, ancestor_path, point_store);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Score traversal
    // -----------------------------------------------------------------------

    /// Compute the anomaly score for `query` using the given `mode`.
    ///
    /// Returns `(raw_score, tree_mass)` so the caller can apply the normalizer.
    pub fn raw_score(&self, query: &[f32], point_store: &PointStore, mode: &ScoreMode) -> f64 {
        if self.root == NULL || self.tree_mass == 0 {
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
                let (child, _sibling) = split_children(query[*cut_dim], *cut_val, *left, *right);

                let child_score = self.score_recursive(child, query, point_store, depth + 1, mode);

                let prob = bbox.probability_of_cut(query);
                if prob == 0.0 {
                    child_score
                } else {
                    (1.0 - prob) * child_score + prob * mode.score_unseen(depth, *mass)
                }
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
        if self.root == NULL || self.tree_mass == 0 {
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
                let (child, _) = split_children(query[*cut_dim], *cut_val, *left, *right);

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
        if self.root == NULL || self.tree_mass == 0 {
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
                let (child, _) = split_children(query[*cut_dim], *cut_val, *left, *right);

                let child_density = self.density_recursive(child, query, point_store);

                let prob = bbox.probability_of_cut(query);
                if prob == 0.0 {
                    child_density
                } else {
                    // density mode: score_unseen(depth, mass) = mass (weighted by depth-inverse)
                    (1.0 - prob) * child_density + prob * (*mass as f64)
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Near-neighbor traversal
    // -----------------------------------------------------------------------

    /// Collect at most `max_results` candidate neighbours.
    ///
    /// Candidates are leaf points that would receive a high isolation score
    /// relative to `query`.  Callers should deduplicate across trees.
    pub fn near_neighbors(
        &self,
        query: &[f32],
        point_store: &PointStore,
        mode: &ScoreMode,
        percentile: usize,
    ) -> Vec<NeighborCandidate> {
        if self.root == NULL || self.tree_mass == 0 {
            return Vec::new();
        }
        let mut results = Vec::new();
        self.nn_recursive(
            self.root,
            query,
            point_store,
            0,
            mode,
            percentile,
            &mut results,
        );
        results
    }

    #[allow(clippy::too_many_arguments)]
    fn nn_recursive(
        &self,
        node_id: usize,
        query: &[f32],
        point_store: &PointStore,
        depth: usize,
        mode: &ScoreMode,
        percentile: usize,
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
                let threshold = nn_threshold(percentile);
                if should_descend_primary(prob, depth, threshold) {
                    let (primary, secondary) =
                        split_children(query[*cut_dim], *cut_val, *left, *right);
                    self.nn_recursive(
                        primary,
                        query,
                        point_store,
                        depth + 1,
                        mode,
                        percentile,
                        results,
                    );
                    if should_descend_secondary(prob) {
                        self.nn_recursive(
                            secondary,
                            query,
                            point_store,
                            depth + 1,
                            mode,
                            percentile,
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
        if self.root == NULL || self.tree_mass == 0 {
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
                    let child = split_child_only(query[*cut_dim], *cut_val, *left, *right);
                    self.impute_recursive(child, query, missing, point_store, centrality, rng, best)
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    pub fn is_empty(&self) -> bool {
        self.root == NULL
    }

    /// Approximate heap size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.arena.slot_count() * std::mem::size_of::<Option<Node>>()
            + std::mem::size_of::<RcfTree>()
    }
}

// ---------------------------------------------------------------------------
// Utility: compute bounding box of a sub-tree
// ---------------------------------------------------------------------------

/// Build the bounding box for the entire sub-tree rooted at `node_id`.
pub(crate) fn subtree_bbox(
    arena: &NodeArena,
    node_id: usize,
    point_store: &PointStore,
) -> BoundingBox {
    match arena.get(node_id) {
        Node::Leaf { point_idx, .. } => BoundingBox::from_point(point_store.get(*point_idx)),
        Node::Internal { bbox, .. } => bbox.clone(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::{point_store::PointStore, score::ScoreMode};
    use rstest::*;

    fn make_store_and_tree(points: &[Vec<f32>]) -> (PointStore, RcfTree) {
        let dim = points[0].len();
        let mut store = PointStore::new(dim, 1, 64, false);
        let mut tree = RcfTree::new(dim, 32, 42);

        for p in points {
            let idx = store.add(p).unwrap();
            tree.insert(idx, &store).unwrap();
        }
        (store, tree)
    }

    #[test]
    fn insert_and_tree_mass() {
        let pts: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, 0.0]).collect();
        let (_, tree) = make_store_and_tree(&pts);
        assert_eq!(tree.tree_mass, 10);
    }

    #[test]
    fn insert_delete_restores_mass() {
        let pts: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32]).collect();
        let dim = 1;
        let mut store = PointStore::new(dim, 1, 16, false);
        let mut tree = RcfTree::new(dim, 16, 99);
        let mut idxs = Vec::new();
        for p in &pts {
            let idx = store.add(p).unwrap();
            idxs.push(idx);
            tree.insert(idx, &store).unwrap();
        }
        assert_eq!(tree.tree_mass, 5);
        tree.delete(idxs[2], &store).unwrap();
        assert_eq!(tree.tree_mass, 4);
    }

    #[rstest]
    #[case(vec![100.0f32, 0.0])]
    #[case(vec![-50.0f32, -50.0])]
    #[case(vec![0.0f32, 200.0])]
    fn score_outlier_higher_than_inlier(#[case] outlier: Vec<f32>) {
        // Build a dense cluster, then score a far outlier vs a center point.
        let mut pts: Vec<Vec<f32>> = (0..50).map(|_| vec![0.5f32, 0.5]).collect();
        pts.push(vec![0.4, 0.6]); // slight inlier
        let (store, tree) = make_store_and_tree(&pts);

        let mode = ScoreMode::standard();
        let inlier_score = tree.raw_score(&[0.5, 0.5], &store, &mode);
        let outlier_score = tree.raw_score(&outlier, &store, &mode);
        assert!(
            outlier_score > inlier_score,
            "outlier={outlier_score} inlier={inlier_score}"
        );
    }

    #[test]
    fn empty_tree_returns_zero_score() {
        let store = PointStore::new(2, 1, 4, false);
        let tree = RcfTree::new(2, 4, 0);
        assert_abs_diff_eq!(
            tree.raw_score(&[1.0, 1.0], &store, &ScoreMode::standard()),
            0.0,
            epsilon = 1e-12
        );
    }

    #[rstest]
    #[case::val_below_cut(0.2f32, 0.3f32, 10usize, 20usize, 10usize, 20usize)]
    #[case::val_above_cut(0.4f32, 0.3f32, 10usize, 20usize, 20usize, 10usize)]
    #[case::val_equal_cut(0.3f32, 0.3f32, 10usize, 20usize, 10usize, 20usize)]
    fn split_children_respects_cut_direction(
        #[case] val: f32,
        #[case] cut: f32,
        #[case] a: usize,
        #[case] b: usize,
        #[case] expected_primary: usize,
        #[case] expected_secondary: usize,
    ) {
        let (primary, secondary) = split_children(val, cut, a, b);
        assert_eq!(primary, expected_primary);
        assert_eq!(secondary, expected_secondary);
        assert_eq!(split_child_only(val, cut, a, b), expected_primary);
    }

    #[rstest]
    #[case::mass_10(10, 0.1)]
    #[case::mass_40(40, 0.4)]
    #[case::mass_100(100, 1.0)]
    fn nn_threshold_scales_with_tree_mass(#[case] mass: usize, #[case] expected: f64) {
        assert_abs_diff_eq!(nn_threshold(mass), expected, epsilon = 1e-12);
    }

    #[test]
    fn near_neighbor_threshold_helpers_behave_as_expected() {
        let th = nn_threshold(40);

        assert!(should_descend_primary(0.41, 3, th));
        assert!(should_descend_primary(0.0, 0, th));
        assert!(!should_descend_primary(0.39, 2, th));

        assert!(should_descend_secondary(0.51));
        assert!(!should_descend_secondary(0.5));
    }
}
