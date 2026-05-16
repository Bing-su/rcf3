//! Mixed numerical/categorical streaming anomaly detection with mStream.
//!
//! Use [`MStream`] to ingest records online, [`MStreamScore`] when you need
//! per-feature score decomposition, and the detector's JSON helpers when
//! persisting or restoring state.

mod clock;
mod config;
mod detector;
mod normalization;
mod scoring;
mod sketch;

pub use config::MStreamConfig;
pub use detector::{MStream, MStreamBuilder, MStreamScore};
