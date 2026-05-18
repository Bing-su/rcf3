#![deny(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod error;
pub mod mstream;
#[cfg(feature = "python")]
mod pyutil;
pub mod rcf;

pub use error::{RcfError, Result};
pub use mstream::{MStream, MStreamBuilder, MStreamConfig, MStreamScore};
pub use rcf::{Attribution, Forest, ForestBuilder, RcfConfig, ScoreMode};
pub use rcf::{bounding_box, config, cut, forest, node_arena, point_store, sampler, score, tree};

// ---------------------------------------------------------------------------
// Python module registration
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
mod rcf3 {
    #[pymodule_export]
    #[allow(non_upper_case_globals)]
    const __version__: &str = env!("CARGO_PKG_VERSION");

    #[pymodule_export]
    use crate::mstream::python::PyMStream;
    #[pymodule_export]
    use crate::rcf::python::PyForest;
}
