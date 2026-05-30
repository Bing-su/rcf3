#![cfg(feature = "python")]

use std::path::PathBuf;

use pyo3::prelude::*;

use super::OnlineIForest;
use crate::pyutil::{StrOrBytes, to_py_err};

/// Online Isolation Forest detector for numerical streams.
///
/// Use `update()` or `update_and_score()` to ingest observations. Use
/// `score()` to preview the current anomaly score for a point without mutating
/// detector state.
///
/// Parameters
/// ----------
/// input_dim : int
///     Number of numerical features in each point.
/// num_trees : int, optional
///     Number of trees in the ensemble (default 32).
/// window_size : int, optional
///     Number of recent points retained by the sliding window (default 2048).
/// max_leaf_samples : int, optional
///     Base leaf-splitting threshold (default 32).
/// seed : int, optional
///     Random seed for deterministic trees.
#[pyclass(name = "OnlineIForest", module = "rcf3.rcf3", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyOnlineIForest {
    inner: OnlineIForest,
}

#[pymethods]
impl PyOnlineIForest {
    #[new]
    #[pyo3(signature = (
        input_dim,
        num_trees = 32,
        window_size = 2048,
        max_leaf_samples = 32,
        seed = None
    ))]
    fn py_new(
        input_dim: usize,
        num_trees: usize,
        window_size: usize,
        max_leaf_samples: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let mut builder = OnlineIForest::builder(input_dim)
            .num_trees(num_trees)
            .window_size(window_size)
            .max_leaf_samples(max_leaf_samples);
        if let Some(seed) = seed {
            builder = builder.seed(seed);
        }
        let inner = builder.build().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Ingest a point without returning its score.
    fn update(&mut self, py: Python<'_>, point: Vec<f32>) -> PyResult<()> {
        py.detach(|| self.inner.update(&point).map_err(to_py_err))
    }

    /// Preview the current anomaly score for `point` without mutating state.
    ///
    /// This can differ from `update_and_score()` because the preview is
    /// computed before `point` is learned by the forest, unlike the
    /// preview-style scoring semantics used by the other detectors. By
    /// contrast, `update_and_score(point)` returns the same value as calling
    /// `update(point)` and then `score(point)`.
    ///
    /// Calling this before `is_ready()` is allowed, but the value is not a
    /// stable anomaly estimate yet.
    fn score(&self, py: Python<'_>, point: Vec<f32>) -> PyResult<f64> {
        py.detach(|| self.inner.score(&point).map_err(to_py_err))
    }

    /// Ingest a point and return its anomaly score under the updated forest.
    ///
    /// This has the same behavior as calling `update(point)` first and then
    /// `score(point)` with the same point. This update-then-score order is
    /// specific to Online Isolation Forest.
    fn update_and_score(&mut self, py: Python<'_>, point: Vec<f32>) -> PyResult<f64> {
        py.detach(|| self.inner.update_and_score(&point).map_err(to_py_err))
    }

    /// Return True once at least one point has been processed.
    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    /// Number of points processed so far.
    fn entries_seen(&self) -> u64 {
        self.inner.entries_seen()
    }

    /// Number of trees in the ensemble.
    fn num_trees(&self) -> usize {
        self.inner.num_trees()
    }

    /// Serialize detector state to JSON.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(to_py_err)
    }

    /// Deserialize detector state from JSON previously written by `to_json()`.
    #[staticmethod]
    fn from_json(json: StrOrBytes) -> PyResult<Self> {
        let inner = OnlineIForest::from_json(json).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Serialize detector state to a JSON file.
    fn save_json(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save_json(path).map_err(to_py_err)
    }

    /// Deserialize detector state from a JSON file previously written by `save_json()`.
    #[staticmethod]
    fn load_json(path: PathBuf) -> PyResult<Self> {
        let inner = OnlineIForest::load_json(path).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    // ---------------------------------------------------------------------------
    // Python Magic Methods
    // ---------------------------------------------------------------------------

    fn __repr__(&self) -> String {
        let c = self.inner.config();
        format!(
            "OnlineIForest(input_dim={}, num_trees={}, window_size={}, max_leaf_samples={}, entries_seen={})",
            c.input_dim(),
            c.num_trees(),
            c.window_size(),
            c.max_leaf_samples(),
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
