#[cfg(not(feature = "std"))]
use alloc::format;

use crate::error::{RcfError, Result};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Advances mStream's logical clock from caller-provided timestamps.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(crate) struct StreamClock {
    pub(crate) current_time: Option<u64>,
    pub(crate) current_tick: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ClockStep {
    pub(crate) tick_gap: u64,
    pub(crate) current_tick: u64,
}

impl StreamClock {
    pub(crate) fn current_time(&self) -> Option<u64> {
        self.current_time
    }

    pub(crate) fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Describe the logical clock step that `timestamp` would produce.
    pub(crate) fn preview(&self, timestamp: u64) -> Result<ClockStep> {
        if timestamp == 0 {
            return Err(RcfError::InvalidArgument("timestamp must be > 0".into()));
        }

        match self.current_time {
            None => Ok(ClockStep {
                tick_gap: 0,
                current_tick: 1,
            }),
            Some(previous) if timestamp > previous => {
                let tick_gap = timestamp - previous;
                Ok(ClockStep {
                    tick_gap,
                    current_tick: self.current_tick + tick_gap,
                })
            }
            Some(previous) if timestamp < previous => Err(RcfError::InvalidArgument(format!(
                "timestamps must be non-decreasing: previous={previous}, got={timestamp}"
            ))),
            Some(_) => Ok(ClockStep {
                tick_gap: 0,
                current_tick: self.current_tick,
            }),
        }
    }

    /// Advance time and return how many decay ticks elapsed since the prior record.
    pub(crate) fn advance(&mut self, timestamp: u64) -> Result<u64> {
        let step = self.preview(timestamp)?;
        self.current_time = Some(timestamp);
        self.current_tick = step.current_tick;
        Ok(step.tick_gap)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(10, Some(10), 1)]
    fn first_timestamp_starts_clock(
        #[case] timestamp: u64,
        #[case] expected_time: Option<u64>,
        #[case] expected_tick: u64,
    ) {
        let mut tracker = StreamClock::default();

        let gap = tracker.advance(timestamp).unwrap();

        assert_eq!(gap, 0);
        assert_eq!(tracker.current_time(), expected_time);
        assert_eq!(tracker.current_tick(), expected_tick);
    }

    #[rstest]
    #[case(10, 0, Some(10), 1)]
    #[case(13, 3, Some(13), 4)]
    fn advances_after_clock_started(
        #[case] timestamp: u64,
        #[case] expected_gap: u64,
        #[case] expected_time: Option<u64>,
        #[case] expected_tick: u64,
    ) {
        let mut tracker = StreamClock::default();
        tracker.advance(10).unwrap();

        let gap = tracker.advance(timestamp).unwrap();

        assert_eq!(gap, expected_gap);
        assert_eq!(tracker.current_time(), expected_time);
        assert_eq!(tracker.current_tick(), expected_tick);
    }

    #[test]
    fn rejects_zero_timestamp() {
        let err = StreamClock::default().advance(0).unwrap_err();
        assert!(matches!(err, RcfError::InvalidArgument(_)));
    }

    #[test]
    fn rejects_decreasing_timestamp() {
        let mut tracker = StreamClock::default();
        tracker.advance(10).unwrap();

        let err = tracker.advance(9).unwrap_err();

        assert!(matches!(err, RcfError::InvalidArgument(_)));
    }

    proptest::proptest! {
        #[test]
        fn accepts_non_decreasing_sequences(
            gaps in proptest::collection::vec(0u64..=8, 1..=32),
        ) {
            let mut tracker = StreamClock::default();
            let mut timestamp = 1;

            for gap in gaps {
                timestamp += gap;
                prop_assert!(tracker.advance(timestamp).is_ok());
            }
        }

        #[test]
        fn rejects_a_decrease_after_progress(
            start in 2u64..=1_000,
            advance_by in 1u64..=32,
            drop_by in 1u64..=32,
        ) {
            let mut tracker = StreamClock::default();
            tracker.advance(start).unwrap();
            tracker.advance(start + advance_by).unwrap();

            let decreased = start + advance_by - drop_by.min(start + advance_by - 1);
            let result = tracker.advance(decreased);

            if decreased < start + advance_by {
                prop_assert!(result.is_err());
            }
        }
    }
}
