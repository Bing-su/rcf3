#![deny(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(test)]
extern crate std;
// Public modules.
pub mod bounding_box;
pub mod config;
pub mod cut;
pub mod error;
pub mod forest;
mod math_utils;
pub mod mstream;
pub mod node_arena;
pub mod point_store;
pub mod sampler;
pub mod score;
pub mod tree;

// Re-exports for ergonomic use as a library crate.
pub use config::RcfConfig;
pub use error::{RcfError, Result};
pub use forest::{Forest, ForestBuilder};
pub use mstream::{MStream, MStreamBuilder, MStreamConfig};
pub use score::{Attribution, ScoreMode};

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
mod python;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
mod rcf3 {
    #[pymodule_export]
    #[allow(non_upper_case_globals)]
    const __version__: &str = env!("CARGO_PKG_VERSION");

    #[pymodule_export]
    use crate::python::PyForest;
}
