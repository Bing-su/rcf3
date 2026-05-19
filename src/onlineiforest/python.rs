#![cfg(feature = "python")]

use std::path::PathBuf;

use pyo3::prelude::*;

use super::OnlineIForest;
use crate::pyutil::{StrOrBytes, to_py_err};

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

    fn update(&mut self, point: Vec<f32>) -> PyResult<()> {
        self.inner.update(&point).map_err(to_py_err)
    }

    fn score(&self, point: Vec<f32>) -> PyResult<f64> {
        self.inner.score(&point).map_err(to_py_err)
    }

    fn update_and_score(&mut self, point: Vec<f32>) -> PyResult<f64> {
        self.inner.update_and_score(&point).map_err(to_py_err)
    }

    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    fn entries_seen(&self) -> u64 {
        self.inner.entries_seen()
    }

    fn num_trees(&self) -> usize {
        self.inner.num_trees()
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(to_py_err)
    }

    #[staticmethod]
    fn from_json(json: StrOrBytes) -> PyResult<Self> {
        let inner = OnlineIForest::from_json(json).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn save_json(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save_json(path).map_err(to_py_err)
    }

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
        let new = Self::from_json(StrOrBytes::Str(state))?;
        *self = new;
        Ok(())
    }

    fn __getnewargs__(&self) -> (usize,) {
        let c = self.inner.config();
        (c.input_dim(),)
    }
}
