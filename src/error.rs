#[cfg(not(feature = "std"))]
use alloc::string::String;

use thiserror::Error;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum RcfError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("forest not ready: insufficient data")]
    NotReady,
    #[error("index out of bounds: {0}")]
    IndexOutOfBounds(usize),
    #[error("overflow: {0}")]
    Overflow(String),
    #[error("runtime error: {0}")]
    Runtime(String),
    #[error("I/O error: {0}")]
    Io(String),
}

pub type Result<T> = core::result::Result<T, RcfError>;
