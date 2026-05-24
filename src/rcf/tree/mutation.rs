use rand::prelude::*;

use super::{PathEntry, RcfTree, merge_subtree_bbox_into, owned_subtree_bbox, split_children};
use crate::error::{RcfError, Result};
use crate::rcf::{
    bounding_box::BoundingBox,
    cut::random_cut,
    node_arena::{NULL, Node, NodeArena},
    point_store::PointStore,
};

#[derive(Clone, Copy, Debug)]
struct InsertionPoint {
    cut_dim: usize,
    cut_val: f32,
    insert_above: usize,
    parent_above: usize,
    ancestor_prefix_len: usize,
}

impl RcfTree {
    /// Descend to the leaf node whose point equals `point`, returning the path
    /// in root-to-leaf traversal order. The last entry is the parent of the
    /// returned leaf and the sibling that was not followed at that branch.
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
                    self.path_scratch.push(PathEntry {
                        parent: cur,
                        sibling,
                    });
                    cur = child;
                }
            }
        }
    }

    fn leaf_for_point(&self, point: &[f32]) -> usize {
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
                    cur = split_children(point[*cut_dim], *cut_val, *left, *right).0;
                }
            }
        }
    }

    /// Parent of `node` given `path` (last element of path is the parent).
    fn parent_of(path: &[PathEntry]) -> usize {
        path.last().map(|entry| entry.parent).unwrap_or(NULL)
    }

    /// Recompute the bounding box for an internal node from its two children.
    fn recompute_bbox(arena: &mut NodeArena, node_id: usize, point_store: &PointStore) {
        let (left, right) = match arena.get(node_id) {
            Node::Internal { left, right, .. } => (*left, *right),
            Node::Leaf { .. } => return,
        };

        let mut bbox = owned_subtree_bbox(arena, left, point_store);
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
        path: &[PathEntry],
        point_store: &PointStore,
    ) {
        for entry in path.iter().rev() {
            Self::recompute_bbox(arena, entry.parent, point_store);
        }
    }

    fn insertion_target_for_step(&self, leaf_id: usize, step_from_leaf: usize) -> (usize, usize) {
        let path_len = self.path_scratch.len();
        let insert_above = if step_from_leaf == 0 {
            leaf_id
        } else {
            self.path_scratch[path_len - step_from_leaf].parent
        };

        let parent_above = if step_from_leaf == path_len {
            NULL
        } else if step_from_leaf == 0 {
            Self::parent_of(&self.path_scratch)
        } else {
            self.path_scratch[path_len - step_from_leaf - 1].parent
        };

        (insert_above, parent_above)
    }

    fn fallback_insertion_point(
        &self,
        point: &[f32],
        leaf_point: &[f32],
        leaf_id: usize,
    ) -> Option<InsertionPoint> {
        for d in 0..self.dims {
            if (point[d] - leaf_point[d]).abs() > f32::EPSILON {
                return Some(InsertionPoint {
                    cut_dim: d,
                    cut_val: leaf_point[d].min(point[d]),
                    insert_above: leaf_id,
                    parent_above: Self::parent_of(&self.path_scratch),
                    ancestor_prefix_len: self.path_scratch.len(),
                });
            }
        }
        None
    }

    fn find_insertion_point(
        &mut self,
        point: &[f32],
        leaf_point: &[f32],
        leaf_id: usize,
        point_store: &PointStore,
    ) -> Option<InsertionPoint> {
        let path_len = self.path_scratch.len();
        let mut current_bbox = BoundingBox::from_point(leaf_point);

        // Scan from leaf upward, expanding the bounding box as we go.
        for step_from_leaf in 0..=path_len {
            let factor: f64 = self.rng.random::<f64>();
            if let Some((cut, sep)) = random_cut(&current_bbox, point, factor)
                && sep
            {
                let (insert_above, parent_above) =
                    self.insertion_target_for_step(leaf_id, step_from_leaf);
                return Some(InsertionPoint {
                    cut_dim: cut.dim,
                    cut_val: cut.val,
                    insert_above,
                    parent_above,
                    ancestor_prefix_len: path_len.saturating_sub(step_from_leaf),
                });
            }

            // Expand box upward by including the sibling subtree.
            if step_from_leaf < path_len {
                let sibling = self.path_scratch[path_len - 1 - step_from_leaf].sibling;
                merge_subtree_bbox_into(&mut current_bbox, &self.arena, sibling, point_store);
            }
        }

        // This fallback is only expected when random cuts fail to separate
        // distinct points in a highly degenerate subtree.
        self.fallback_insertion_point(point, leaf_point, leaf_id)
    }

    // -----------------------------------------------------------------------
    // Insert
    // -----------------------------------------------------------------------

    /// Insert `point_idx` into the tree and return the point-store index that
    /// the tree actually references after insertion.
    pub(in crate::rcf) fn insert(
        &mut self,
        point_idx: usize,
        point_store: &PointStore,
    ) -> Result<usize> {
        let point = point_store.get(point_idx);

        if self.root == NULL {
            // First point in this tree.
            self.root = self.arena.alloc(Node::Leaf { point_idx, mass: 1 });
            self.tree_mass = 1;
            return Ok(point_idx);
        }

        let leaf_id = self.path_to_leaf(point);
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
            return Ok(leaf_point_idx);
        }

        // Different point: need a new internal node with a random cut.
        let leaf_point = point_store.get(leaf_point_idx);
        let Some(insertion) = self.find_insertion_point(point, leaf_point, leaf_id, point_store)
        else {
            // Truly identical — treat as duplicate (shouldn't reach here because
            // `is_equal` would have caught it, but be defensive).
            if let Node::Leaf { mass, .. } = self.arena.get_mut(leaf_id) {
                *mass += 1;
            }
            Self::update_ancestors_after_change(&mut self.arena, &self.path_scratch, point_store);
            return Ok(leaf_point_idx);
        };

        // Determine which side new point and existing subtree go on.
        let new_leaf_id = self.arena.alloc(Node::Leaf { point_idx, mass: 1 });
        let (new_left, new_right) = if point[insertion.cut_dim] <= insertion.cut_val {
            (new_leaf_id, insertion.insert_above)
        } else {
            (insertion.insert_above, new_leaf_id)
        };

        let child_mass = self.arena.get(insertion.insert_above).mass() + 1;
        let mut new_bbox = owned_subtree_bbox(&self.arena, insertion.insert_above, point_store);
        new_bbox.merge_point(point);

        let new_internal = self.arena.alloc(Node::Internal {
            left: new_left,
            right: new_right,
            cut_dim: insertion.cut_dim,
            cut_val: insertion.cut_val,
            mass: child_mass,
            bbox: new_bbox,
        });

        // Attach new_internal in place of insert_above.
        if insertion.parent_above == NULL {
            self.root = new_internal;
        } else {
            match self.arena.get_mut(insertion.parent_above) {
                Node::Internal { left, right, .. } => {
                    if *left == insertion.insert_above {
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
            &self.path_scratch[..insertion.ancestor_prefix_len],
            point_store,
        );

        Ok(point_idx)
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
        self.validate_delete(point_idx, point_store)?;

        let point = point_store.get(point_idx);
        let leaf_id = self.path_to_leaf(point);
        let path_len = self.path_scratch.len();

        let leaf_mass = match self.arena.get(leaf_id) {
            Node::Leaf { mass, .. } => *mass,
            _ => unreachable!(),
        };

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

        let leaf_parent = self.path_scratch[path_len - 1];
        let parent_id = leaf_parent.parent;
        let sibling_id = leaf_parent.sibling;
        let grandparent = if path_len >= 2 {
            self.path_scratch[path_len - 2].parent
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

    pub(in crate::rcf) fn validate_delete(
        &self,
        point_idx: usize,
        point_store: &PointStore,
    ) -> Result<()> {
        if self.root == NULL {
            return Err(RcfError::EmptyTree);
        }

        let point = point_store.get(point_idx);
        let leaf_id = self.leaf_for_point(point);

        let leaf_mass = match self.arena.get(leaf_id) {
            Node::Leaf { mass, .. } => *mass,
            _ => unreachable!(),
        };

        if leaf_mass == 0 {
            return Err(RcfError::Runtime("leaf mass is already 0".into()));
        }

        Ok(())
    }
}
