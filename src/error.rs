#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(all(not(feature = "std"), feature = "serde"))]
use alloc::string::ToString;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum RcfError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("invalid serialized config: {0}")]
    InvalidSerializedConfig(String),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("forest not ready: insufficient data")]
    NotReady,
    #[error("index out of bounds: {0}")]
    IndexOutOfBounds(usize),
    #[error("overflow: {0}")]
    Overflow(String),
    #[error("operation on empty tree")]
    EmptyTree,
    #[error("I/O error: {0}")]
    Io(String),
}

#[cfg(feature = "serde")]
impl RcfError {
    pub(crate) fn invalid_serialized_config(error: Self) -> Self {
        match error {
            Self::InvalidArgument(msg) => Self::InvalidSerializedConfig(msg),
            other => Self::InvalidSerializedConfig(other.to_string()),
        }
    }
}

pub type Result<T> = core::result::Result<T, RcfError>;
