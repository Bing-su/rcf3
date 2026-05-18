//! Online Isolation Forest for numerical streaming anomaly detection.
//!
//! [`OnlineIForest`] maintains an ensemble of online isolation trees over a
//! sliding window. Each tree expands dense regions by splitting bins and
//! contracts stale regions by forgetting old points as the stream advances.

mod config;
mod detector;
mod node;
mod tree;

pub use config::OnlineIForestConfig;
pub use detector::{OnlineIForest, OnlineIForestBuilder};
