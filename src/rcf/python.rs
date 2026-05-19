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
        let mut b = Forest::builder(input_dim)
            .shingle_size(shingle_size)
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

    fn update(&mut self, point: Vec<f32>) -> PyResult<()> {
        self.inner.update(&point).map_err(to_py_err)
    }

    fn score(&self, point: Vec<f32>) -> PyResult<f64> {
        self.inner.score(&point).map_err(to_py_err)
    }

    fn displacement_score(&self, point: Vec<f32>) -> PyResult<f64> {
        self.inner.displacement_score(&point).map_err(to_py_err)
    }

    fn attribution(&self, point: Vec<f32>) -> PyResult<Vec<PyAttribution>> {
        self.inner
            .attribution(&point)
            .map(|items| items.into_iter().map(PyAttribution::from).collect())
            .map_err(to_py_err)
    }

    fn density(&self, point: Vec<f32>) -> PyResult<f64> {
        self.inner.density(&point).map_err(to_py_err)
    }

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

    #[pyo3(signature = (point, missing, centrality = 1.0))]
    fn impute(&self, point: Vec<f32>, missing: Vec<usize>, centrality: f64) -> PyResult<Vec<f32>> {
        self.inner
            .impute(&point, &missing, centrality)
            .map_err(to_py_err)
    }

    fn extrapolate(&self, look_ahead: usize) -> PyResult<Vec<f32>> {
        self.inner.extrapolate(look_ahead).map_err(to_py_err)
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
        let inner = Forest::from_json(json).map_err(to_py_err)?;
        Ok(PyForest { inner })
    }

    fn save_json(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save_json(path).map_err(to_py_err)
    }

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
        let new = Self::from_json(StrOrBytes::Str(state))?;
        *self = new;
        Ok(())
    }

    fn __getnewargs__(&self) -> (usize,) {
        let c = self.inner.config();
        (c.input_dim,)
    }
}
