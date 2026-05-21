use rand::prelude::*;

use super::{RcfTree, merge_subtree_bbox_into, split_children, subtree_bbox_owned};
use crate::error::{RcfError, Result};
use crate::rcf::{
    bounding_box::BoundingBox,
    cut::random_cut,
    node_arena::{NULL, Node, NodeArena},
    point_store::PointStore,
};

impl RcfTree {
    /// Descend to the leaf node whose point equals `point`, returning the path
    /// as a Vec of `(node_id, sibling_id)` pairs in root-to-leaf traversal
    /// order. The last pair's first component is the parent of the returned
    /// leaf.
    fn path_to_leaf(&mut self, point: &[f32]) -> usize {
        self.path_scratch.clear();
        let mut cur = self.root;
        loop {
            match self.arena.get(cur) {
                Node::Leaf { .. } => return cur,
                Node::Internal {
                    left,
                    right,
                    cut_dim,
                    cut_val,
                    ..
                } => {
                    let (child, sibling) = split_children(point[*cut_dim], *cut_val, *left, *right);
                    self.path_scratch.push((cur, sibling));
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

        let mut bbox = subtree_bbox_owned(arena, left, point_store);
        merge_subtree_bbox_into(&mut bbox, arena, right, point_store);
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
    pub(in crate::rcf) fn insert(
        &mut self,
        point_idx: usize,
        point_store: &PointStore,
    ) -> Result<()> {
        let point = point_store.get(point_idx);

        if self.root == NULL {
            // First point in this tree.
            self.root = self.arena.alloc(Node::Leaf { point_idx, mass: 1 });
            self.tree_mass = 1;
            return Ok(());
        }

        let leaf_id = self.path_to_leaf(point);
        let path_len = self.path_scratch.len();
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
            Self::update_ancestors_after_change(&mut self.arena, &self.path_scratch, point_store);
            return Ok(());
        }

        // Different point: need a new internal node with a random cut.
        let leaf_point = point_store.get(leaf_point_idx);
        let mut current_bbox = BoundingBox::from_point(leaf_point);

        // Find the highest-in-tree cut that separates the new point.
        let mut saved_cut_dim = NULL;
        let mut saved_cut_val = 0.0f32;
        let mut insert_above: usize = leaf_id; // node below which to insert the new split
        let mut parent_above = Self::parent_of(&self.path_scratch); // parent of insert_above
        let mut ancestor_prefix_len = path_len;

        // Scan from leaf upward, expanding the bounding box as we go.
        for step in 0..=path_len {
            let factor: f64 = self.rng.random::<f64>();
            if let Some((cut, sep)) = random_cut(&current_bbox, point, factor)
                && sep
            {
                saved_cut_dim = cut.dim;
                saved_cut_val = cut.val;
                insert_above = if step == 0 {
                    leaf_id
                } else {
                    self.path_scratch[path_len - step].0
                };
                parent_above = if step == path_len {
                    NULL
                } else if step == 0 {
                    Self::parent_of(&self.path_scratch)
                } else {
                    self.path_scratch[path_len - step - 1].0
                };
                ancestor_prefix_len = if step == 0 { path_len } else { path_len - step };
                break;
            }
            // Expand box upward by including the sibling subtree.
            if step < path_len {
                let sibling = self.path_scratch[path_len - 1 - step].1;
                merge_subtree_bbox_into(&mut current_bbox, &self.arena, sibling, point_store);
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
                    parent_above = Self::parent_of(&self.path_scratch);
                    ancestor_prefix_len = path_len;
                    break;
                }
            }
            if saved_cut_dim == NULL {
                // Truly identical — treat as duplicate (shouldn't reach here because
                // `is_equal` would have caught it, but be defensive).
                if let Node::Leaf { mass, .. } = self.arena.get_mut(leaf_id) {
                    *mass += 1;
                }
                Self::update_ancestors_after_change(
                    &mut self.arena,
                    &self.path_scratch,
                    point_store,
                );
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
        let mut new_bbox = subtree_bbox_owned(&self.arena, insert_above, point_store);
        new_bbox.merge_point(point);

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

        // Recompute only the ancestors that remain above the insertion point.
        // The new internal node already has its mass and bbox initialized.
        Self::update_ancestors_after_change(
            &mut self.arena,
            &self.path_scratch[..ancestor_prefix_len],
            point_store,
        );

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Delete
    // -----------------------------------------------------------------------

    /// Remove `point_idx` from the tree.
    pub(in crate::rcf) fn delete(
        &mut self,
        point_idx: usize,
        point_store: &PointStore,
    ) -> Result<()> {
        if self.root == NULL {
            return Err(RcfError::EmptyTree);
        }

        let point = point_store.get(point_idx);
        let leaf_id = self.path_to_leaf(point);
        let path_len = self.path_scratch.len();

        let leaf_mass = match self.arena.get(leaf_id) {
            Node::Leaf { mass, .. } => *mass,
            _ => unreachable!(),
        };

        if leaf_mass == 0 {
            return Err(RcfError::InvalidArgument("leaf mass is already 0".into()));
        }

        self.tree_mass -= 1;

        if leaf_mass > 1 {
            // Just decrement duplicates.
            if let Node::Leaf { mass, .. } = self.arena.get_mut(leaf_id) {
                *mass -= 1;
            }
            Self::update_ancestors_after_change(&mut self.arena, &self.path_scratch, point_store);
            return Ok(());
        }

        // Last copy: remove the leaf and its parent internal node.
        if path_len == 0 {
            // Tree had only one point.
            self.arena.free(leaf_id);
            self.root = NULL;
            return Ok(());
        }

        let (parent_id, sibling_id) = self.path_scratch[path_len - 1];
        let grandparent = if path_len >= 2 {
            self.path_scratch[path_len - 2].0
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
        Self::update_ancestors_after_change(
            &mut self.arena,
            &self.path_scratch[..path_len - 1],
            point_store,
        );

        Ok(())
    }
}
