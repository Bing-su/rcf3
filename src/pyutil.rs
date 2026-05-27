#![cfg(feature = "python")]

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::BTreeMap;

use crate::error::RcfError;

/// Convert an [`RcfError`] to a Python exception.
pub(crate) fn to_py_err(e: RcfError) -> PyErr {
    match e {
        RcfError::InvalidArgument(msg) => PyValueError::new_err(msg),
        RcfError::DimensionMismatch { expected, got } => PyValueError::new_err(format!(
            "dimension mismatch: expected {expected}, got {got}"
        )),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

#[derive(FromPyObject)]
pub(crate) enum StrOrBytes {
    Str(String),
    Bytes(Vec<u8>),
}

impl AsRef<[u8]> for StrOrBytes {
    fn as_ref(&self) -> &[u8] {
        match self {
            StrOrBytes::Str(s) => s.as_bytes(),
            StrOrBytes::Bytes(b) => b,
        }
    }
}

impl From<String> for StrOrBytes {
    fn from(value: String) -> Self {
        Self::Str(value)
    }
}

impl From<Vec<u8>> for StrOrBytes {
    fn from(value: Vec<u8>) -> Self {
        Self::Bytes(value)
    }
}

#[derive(FromPyObject)]
pub(crate) enum KeyValueLike {
    Pairs(Vec<(String, f64)>),
    Dict(BTreeMap<String, f64>),
}

impl IntoIterator for KeyValueLike {
    type Item = (String, f64);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            KeyValueLike::Pairs(vec) => vec.into_iter(),
            KeyValueLike::Dict(map) => map.into_iter().collect::<Vec<_>>().into_iter(),
        }
    }
}
