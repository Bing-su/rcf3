#![cfg(feature = "python")]

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::error;
use crate::forest::Forest;

/// Convert an [`RcfError`] to a Python exception.
fn to_py_err(e: error::RcfError) -> PyErr {
    match e {
        error::RcfError::InvalidArgument(msg) => PyValueError::new_err(msg),
        error::RcfError::DimensionMismatch { expected, got } => PyValueError::new_err(format!(
            "dimension mismatch: expected {expected}, got {got}"
        )),
        other => PyRuntimeError::new_err(format!("{other:?}")),
    }
}

// ---------------------------------------------------------------------------
// PyForest
// ---------------------------------------------------------------------------

/// A Random Cut Forest anomaly detector.
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
///     Exponential decay for sample weights (default 0 = auto).
/// output_after : int, optional
///     Minimum observations before scoring starts (default 0 = auto).
/// internal_shingling : bool, optional
///     When True, pass one base observation at a time and the forest
///     maintains the rolling shingle buffer (default True).
/// seed : int, optional
///     Random seed for deterministic forests.
#[pyclass(name = "Forest", module = "arcf.arcf", skip_from_py_object)]
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
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let mut b = Forest::builder(input_dim, shingle_size)
            .num_trees(num_trees)
            .capacity(capacity)
            .time_decay(time_decay)
            .output_after(output_after)
            .internal_shingling(internal_shingling);
        if let Some(s) = seed {
            b = b.seed(s);
        }
        let inner = b.build().map_err(to_py_err)?;
        Ok(PyForest { inner })
    }

    /// Update the forest with a new observation.
    fn update(&mut self, point: Vec<f32>) -> PyResult<()> {
        self.inner.update(&point).map_err(to_py_err)
    }

    /// Anomaly score for `point`.  Returns 0.0 before the forest is ready.
    fn score(&self, point: Vec<f32>) -> PyResult<f64> {
        self.inner.score(&point).map_err(to_py_err)
    }

    /// Displacement-based anomaly score for `point`.
    fn displacement_score(&self, point: Vec<f32>) -> PyResult<f64> {
        self.inner.displacement_score(&point).map_err(to_py_err)
    }

    /// Per-dimension attribution of the anomaly score.
    ///
    /// Returns a list of [below, above] pairs for each dimension.
    fn attribution(&self, point: Vec<f32>) -> PyResult<Vec<[f64; 2]>> {
        self.inner.attribution(&point).map_err(to_py_err)
    }

    /// Density estimate at `point`.
    fn density(&self, point: Vec<f32>) -> PyResult<f64> {
        self.inner.density(&point).map_err(to_py_err)
    }

    /// Find approximate near-neighbours of `point`.
    ///
    /// Returns a list of (score, point, distance) tuples.
    #[pyo3(signature = (point, top_k = 10, percentile = 50))]
    fn near_neighbors(
        &self,
        point: Vec<f32>,
        top_k: usize,
        percentile: usize,
    ) -> PyResult<Vec<(f64, Vec<f32>, f64)>> {
        self.inner
            .near_neighbors(&point, top_k, percentile)
            .map_err(to_py_err)
    }

    /// Impute the missing dimensions of `point`.
    ///
    /// Parameters
    /// ----------
    /// point : list[float]
    ///     Full-dimensional query (missing values will be ignored).
    /// missing : list[int]
    ///     Indices of dimensions to impute.
    /// centrality : float, optional
    ///     1.0 = always pick the nearest candidate (deterministic).
    #[pyo3(signature = (point, missing, centrality = 1.0))]
    fn impute(&self, point: Vec<f32>, missing: Vec<usize>, centrality: f64) -> PyResult<Vec<f32>> {
        self.inner
            .impute(&point, &missing, centrality)
            .map_err(to_py_err)
    }

    /// Predict the next `look_ahead` base observations.
    ///
    /// Requires `internal_shingling = True` and `shingle_size > 1`.
    /// Returns a flat list of length `look_ahead * input_dim`.
    fn extrapolate(&self, look_ahead: usize) -> PyResult<Vec<f32>> {
        self.inner.extrapolate(look_ahead).map_err(to_py_err)
    }

    /// Whether the forest has seen enough observations to return scores.
    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    /// Number of observations processed so far.
    fn entries_seen(&self) -> u64 {
        self.inner.entries_seen()
    }

    /// Number of trees.
    fn num_trees(&self) -> usize {
        self.inner.num_trees()
    }

    /// Serialise the forest state to a JSON string.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(to_py_err)
    }

    /// Load a forest from a JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner = Forest::from_json(json).map_err(to_py_err)?;
        Ok(PyForest { inner })
    }

    /// Serialise the forest state to a JSON file.
    fn save_json(&self, path: &str) -> PyResult<()> {
        self.inner.save_json(path).map_err(to_py_err)
    }

    /// Load a forest from a JSON file.
    #[staticmethod]
    fn load_json(path: &str) -> PyResult<Self> {
        let inner = Forest::load_json(path).map_err(to_py_err)?;
        Ok(PyForest { inner })
    }

    // Python Magic Methods

    fn __repr__(&self) -> String {
        let c = self.inner.config();
        format!(
            "Forest(input_dim={}, shingle_size={}, num_trees={}, capacity={}, entries_seen={})",
            c.input_dim,
            c.shingle_size,
            c.num_trees,
            c.capacity,
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
        let new = Self::from_json(&state)?;
        *self = new;
        Ok(())
    }
}
