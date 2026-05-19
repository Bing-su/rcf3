#![cfg(feature = "python")]

use std::path::PathBuf;

use pyo3::prelude::*;

use super::{MStream, MStreamScore};
use crate::pyutil::{StrOrBytes, to_py_err};

#[derive(IntoPyObject)]
struct PyMStreamScore {
    total: f64,
    record: f64,
    numeric_features: Vec<f64>,
    categorical_features: Vec<f64>,
}

impl From<MStreamScore> for PyMStreamScore {
    fn from(value: MStreamScore) -> Self {
        Self {
            total: value.total,
            record: value.record,
            numeric_features: value.numeric_features,
            categorical_features: value.categorical_features,
        }
    }
}

#[pyclass(name = "MStream", module = "rcf3.rcf3", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMStream {
    inner: MStream,
}

#[pymethods]
impl PyMStream {
    #[new]
    #[pyo3(signature = (
        numeric_dim,
        categorical_dim,
        num_rows = 2,
        num_buckets = 1024,
        alpha = 0.8,
        seed = None
    ))]
    fn py_new(
        numeric_dim: usize,
        categorical_dim: usize,
        num_rows: usize,
        num_buckets: usize,
        alpha: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let mut builder = MStream::builder(numeric_dim, categorical_dim)
            .num_rows(num_rows)
            .num_buckets(num_buckets)
            .alpha(alpha);
        if let Some(seed) = seed {
            builder = builder.seed(seed);
        }
        let inner = builder.build().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn update(&mut self, numeric: Vec<f32>, categorical: Vec<i64>, timestamp: u64) -> PyResult<()> {
        self.inner
            .update(&numeric, &categorical, timestamp)
            .map_err(to_py_err)
    }

    fn score(&self, numeric: Vec<f32>, categorical: Vec<i64>, timestamp: u64) -> PyResult<f64> {
        self.inner
            .score(&numeric, &categorical, timestamp)
            .map_err(to_py_err)
    }

    fn update_and_score(
        &mut self,
        numeric: Vec<f32>,
        categorical: Vec<i64>,
        timestamp: u64,
    ) -> PyResult<f64> {
        self.inner
            .update_and_score(&numeric, &categorical, timestamp)
            .map_err(to_py_err)
    }

    fn score_detailed(
        &self,
        numeric: Vec<f32>,
        categorical: Vec<i64>,
        timestamp: u64,
    ) -> PyResult<PyMStreamScore> {
        self.inner
            .score_detailed(&numeric, &categorical, timestamp)
            .map(PyMStreamScore::from)
            .map_err(to_py_err)
    }

    fn update_and_score_detailed(
        &mut self,
        numeric: Vec<f32>,
        categorical: Vec<i64>,
        timestamp: u64,
    ) -> PyResult<PyMStreamScore> {
        self.inner
            .update_and_score_detailed(&numeric, &categorical, timestamp)
            .map(PyMStreamScore::from)
            .map_err(to_py_err)
    }

    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    fn entries_seen(&self) -> u64 {
        self.inner.entries_seen()
    }

    fn current_time(&self) -> Option<u64> {
        self.inner.current_time()
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(to_py_err)
    }

    #[staticmethod]
    fn from_json(json: StrOrBytes) -> PyResult<Self> {
        let inner = MStream::from_json(json).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn save_json(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save_json(path).map_err(to_py_err)
    }

    #[staticmethod]
    fn load_json(path: PathBuf) -> PyResult<Self> {
        let inner = MStream::load_json(path).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    // ---------------------------------------------------------------------------
    // Python Magic Methods
    // ---------------------------------------------------------------------------

    fn __repr__(&self) -> String {
        let c = self.inner.config();
        format!(
            "MStream(numeric_dim={}, categorical_dim={}, num_rows={}, num_buckets={}, alpha={}, entries_seen={})",
            c.numeric_dim(),
            c.categorical_dim(),
            c.num_rows(),
            c.num_buckets(),
            c.alpha(),
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

    fn __getnewargs__(&self) -> (usize, usize) {
        let c = self.inner.config();
        (c.numeric_dim(), c.categorical_dim())
    }
}
