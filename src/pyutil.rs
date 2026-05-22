#![cfg(feature = "python")]

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::error::RcfError;

/// Convert an [`RcfError`] to a Python exception.
pub(crate) fn to_py_err(e: RcfError) -> PyErr {
    match e {
        RcfError::InvalidArgument(msg) | RcfError::InvalidSerializedConfig(msg) => {
            PyValueError::new_err(msg)
        }
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
