use std::fmt;

#[derive(Debug)]
pub enum RcfError {
    InvalidArgument(String),
    DimensionMismatch { expected: usize, got: usize },
    NotReady,
    IndexOutOfBounds(usize),
    EmptyTree,
    Io(String),
}

impl fmt::Display for RcfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RcfError::InvalidArgument(msg) => write!(f, "Invalid argument: {msg}"),
            RcfError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {expected}, got {got}")
            }
            RcfError::NotReady => write!(f, "Forest not ready: insufficient data"),
            RcfError::IndexOutOfBounds(i) => write!(f, "Index out of bounds: {i}"),
            RcfError::EmptyTree => write!(f, "Operation on empty tree"),
            RcfError::Io(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for RcfError {}

pub type Result<T> = std::result::Result<T, RcfError>;
