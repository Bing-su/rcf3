use thiserror::Error;

#[derive(Error, Debug)]
pub enum RcfError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("forest not ready: insufficient data")]
    NotReady,
    #[error("index out of bounds: {0}")]
    IndexOutOfBounds(usize),
    #[error("operation on empty tree")]
    EmptyTree,
    #[error("I/O error: {0}")]
    Io(String),
}

pub type Result<T> = std::result::Result<T, RcfError>;
