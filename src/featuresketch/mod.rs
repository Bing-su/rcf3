//! FeatureSketch detector for sparse, schema-evolving feature streams.

mod chain;
mod config;
mod detector;
mod input;
mod projection;
#[cfg(feature = "python")]
pub(crate) mod python;
mod sketch;

pub use config::FeatureSketchConfig;
pub use detector::{FeatureSketch, FeatureSketchBuilder};
