#![cfg(feature = "python")]

use std::path::PathBuf;

use pyo3::prelude::*;

use super::forest::{Forest, NeighborResult};
use super::score::Attribution;
use crate::pyutil::{StrOrBytes, to_py_err};

// ---------------------------------------------------------------------------
// PyForest
// ---------------------------------------------------------------------------
#[derive(IntoPyObject)]
struct PyNeighborResult {
    score: f64,
    point: Vec<f32>,
    distance: f64,
}

impl From<NeighborResult> for PyNeighborResult {
    fn from(value: NeighborResult) -> Self {
        Self {
            score: value.score,
            point: value.point,
            distance: value.distance,
        }
    }
}

#[derive(IntoPyObject)]
struct PyAttribution {
    below: f64,
    above: f64,
}

impl From<Attribution> for PyAttribution {
    fn from(value: Attribution) -> Self {
        Self {
            below: value.below,
            above: value.above,
        }
    }
}

/// A Random Cut Forest: an ensemble of Random Cut Trees sharing point storage.
///
/// Parameters
/// ----------
/// input_dim : int
///     Number of features per observation (before shingling).
/// shingle_size : int, optional
///     Temporal window size (default 1, no shingling).
/// num_trees : int, optional
///     Number of trees in the ensemble (default 50).
/// capacity : int, optional
///     Maximum samples per tree (default 256).
/// time_decay : float, optional
///     Finite non-negative exponential decay for sample weights
///     (default 0 = auto).
/// output_after : int, optional
///     Minimum observations before scoring starts (default 0 = auto).
/// internal_shingling : bool, optional
///     When True, pass one base observation at a time and the forest
///     maintains the rolling shingle buffer (default True).
/// initial_accept_fraction : float, optional
///     Finite value in [0.0, 1.0] controlling warm-up sampler acceptance
///     before capacity (default 0.125).
/// seed : int, optional
///     Random seed for deterministic forests.
#[pyclass(name = "Forest", module = "rcf3.rcf3", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyForest {
    inner: Forest,
}

#[pymethods]
impl PyForest {
    #[new]
    #[pyo3(signature = (
        input_dim,
        shingle_size = 1,
        num_trees = 50,
        capacity = 256,
        time_decay = 0.0,
        output_after = 0,
        internal_shingling = true,
        initial_accept_fraction = 0.125,
        seed = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        input_dim: usize,
        shingle_size: usize,
        num_trees: usize,
        capacity: usize,
        time_decay: f64,
        output_after: usize,
        internal_shingling: bool,
        initial_accept_fraction: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let mut b = Forest::builder(input_dim)
            .shingle_size(shingle_size)
            .num_trees(num_trees)
            .capacity(capacity)
            .time_decay(time_decay)
            .output_after(output_after)
            .internal_shingling(internal_shingling)
            .initial_accept_fraction(initial_accept_fraction);
        if let Some(s) = seed {
            b = b.seed(s);
        }
        let inner = b.build().map_err(to_py_err)?;
        Ok(PyForest { inner })
    }

    /// Incorporate a new observation into the forest.
    ///
    /// When `internal_shingling` is True, pass one base observation of length
    /// `input_dim`. Otherwise pass the full shingled vector of length
    /// `input_dim * shingle_size`.
    fn update(&mut self, point: Vec<f32>) -> PyResult<()> {
        self.inner.update(&point).map_err(to_py_err)
    }

    /// Anomaly score for `point`. Higher means more anomalous.
    fn score(&self, point: Vec<f32>) -> PyResult<f64> {
        self.inner.score(&point).map_err(to_py_err)
    }

    /// Displacement-based anomaly score.
    fn displacement_score(&self, point: Vec<f32>) -> PyResult<f64> {
        self.inner.displacement_score(&point).map_err(to_py_err)
    }

    /// Per-dimension attribution of the anomaly score.
    ///
    /// Returns a list of length `input_dim * shingle_size`.
    fn attribution(&self, point: Vec<f32>) -> PyResult<Vec<PyAttribution>> {
        self.inner
            .attribution(&point)
            .map(|items| items.into_iter().map(PyAttribution::from).collect())
            .map_err(to_py_err)
    }

    /// Density estimate at `point`. Higher means a denser neighbourhood.
    fn density(&self, point: Vec<f32>) -> PyResult<f64> {
        self.inner.density(&point).map_err(to_py_err)
    }

    /// Find approximate near-neighbours of `point`.
    ///
    /// `percentile` controls per-tree traversal aggressiveness in `[0, 100]`;
    /// lower values visit more branches and usually return more candidates.
    /// Returns a list sorted by distance, with duplicate points across trees
    /// merged by point index. At most `top_k` results are returned.
    #[pyo3(signature = (point, top_k = 10, percentile = 50))]
    fn near_neighbors(
        &self,
        point: Vec<f32>,
        top_k: usize,
        percentile: usize,
    ) -> PyResult<Vec<PyNeighborResult>> {
        self.inner
            .near_neighbors(&point, top_k, percentile)
            .map(|items| items.into_iter().map(PyNeighborResult::from).collect())
            .map_err(to_py_err)
    }

    /// Impute the `missing` positions of `point`.
    ///
    /// `point` must have the full dimensionality (`input_dim * shingle_size`).
    /// Values at `missing` indices are ignored; the returned list fills them
    /// with the median of the nearest-neighbour estimates across all trees.
    /// When `centrality = 1.0`, the nearest neighbour in each tree is selected
    /// deterministically; lower values introduce randomness.
    #[pyo3(signature = (point, missing, centrality = 1.0))]
    fn impute(&self, point: Vec<f32>, missing: Vec<usize>, centrality: f64) -> PyResult<Vec<f32>> {
        self.inner
            .impute(&point, &missing, centrality)
            .map_err(to_py_err)
    }

    /// Predict the next `look_ahead` base observations beyond the current shingle buffer.
    ///
    /// Requires `internal_shingling = True`, `shingle_size > 1`, and
    /// `look_ahead <= shingle_size`. Returns a list of length
    /// `look_ahead * input_dim`.
    fn extrapolate(&self, look_ahead: usize) -> PyResult<Vec<f32>> {
        self.inner.extrapolate(look_ahead).map_err(to_py_err)
    }

    /// Return True once scoring functions return meaningful values.
    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    /// Number of observations processed so far.
    fn entries_seen(&self) -> u64 {
        self.inner.entries_seen()
    }

    /// Number of trees in the ensemble.
    fn num_trees(&self) -> usize {
        self.inner.num_trees()
    }

    /// Serialize the entire forest state to a JSON string.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(to_py_err)
    }

    /// Deserialize a forest from a JSON string previously written by `to_json()`.
    #[staticmethod]
    fn from_json(json: StrOrBytes) -> PyResult<Self> {
        let inner = Forest::from_json(json).map_err(to_py_err)?;
        Ok(PyForest { inner })
    }

    /// Serialize the entire forest state to a JSON file.
    fn save_json(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save_json(path).map_err(to_py_err)
    }

    /// Deserialize a forest from a JSON file previously written by `save_json()`.
    #[staticmethod]
    fn load_json(path: PathBuf) -> PyResult<Self> {
        let inner = Forest::load_json(path).map_err(to_py_err)?;
        Ok(PyForest { inner })
    }

    // ---------------------------------------------------------------------------
    // Python Magic Methods
    // ---------------------------------------------------------------------------

    fn __repr__(&self) -> String {
        let c = self.inner.config();
        format!(
            "Forest(input_dim={}, shingle_size={}, num_trees={}, capacity={}, entries_seen={})",
            c.input_dim(),
            c.shingle_size(),
            c.num_trees(),
            c.capacity(),
            self.inner.entries_seen(),
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    #[allow(unused_variables)]
    fn __deepcopy__<'py>(&self, memo: Bound<'py, PyAny>) -> Self {
        self.clone()
    }

    fn __getstate__(&self) -> PyResult<String> {
        self.to_json()
    }

    fn __setstate__(&mut self, state: String) -> PyResult<()> {
        let new = Self::from_json(state.into())?;
        *self = new;
        Ok(())
    }

    fn __getnewargs__(&self) -> (usize,) {
        let c = self.inner.config();
        (c.input_dim(),)
    }
}
