//! Mixed numerical/categorical streaming anomaly detection with mStream.
//!
//! Use [`MStream`] to ingest records online, [`MStreamScore`] when you need
//! per-feature score decomposition, and [`MStreamSnapshot`] when persisting or
//! restoring detector state.

mod clock;
mod config;
mod detector;
mod normalization;
mod scoring;
mod sketch;

pub use config::MStreamConfig;
pub use detector::{MStream, MStreamBuilder, MStreamScore, MStreamSnapshot};
