/// Convert one sketch count pair into the paper's anomaly contribution.
pub(crate) fn counts_to_anom(total: f64, current: f64, current_tick: u64) -> f64 {
    let cur_t = (current_tick as f64).max(1.0);
    let cur_mean = total / cur_t;
    if cur_mean <= f64::EPSILON {
        return 0.0;
    }

    let sqerr = current - cur_mean;
    let sqerr = sqerr * sqerr;

    sqerr / cur_mean + sqerr / (cur_mean * (cur_t - 1.0).max(1.0))
}

/// Preview the score after one record is inserted into a sketch pair.
pub(crate) fn preview_insert_score(
    total_before: f64,
    current_before: f64,
    decay_factor: f64,
    current_tick: u64,
) -> f64 {
    counts_to_anom(
        total_before + 1.0,
        current_before * decay_factor + 1.0,
        current_tick,
    )
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(10.0, 5.0, 2, 0.0)]
    #[case(10.0, 0.0, 2, 10.0)]
    #[case(10.0, 10.0, 2, 10.0)]
    #[case(0.0, 0.0, 1, 0.0)]
    fn computes_expected_score(
        #[case] total: f64,
        #[case] current: f64,
        #[case] current_tick: u64,
        #[case] expected: f64,
    ) {
        let actual = counts_to_anom(total, current, current_tick);
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
    }

    #[test]
    fn initial_tick_is_stable() {
        assert_eq!(counts_to_anom(1.0, 1.0, 1), 0.0);
    }

    #[test]
    fn preview_insert_applies_decay_before_current_insert() {
        let score = preview_insert_score(3.0, 2.0, 0.5, 4);
        let expected = counts_to_anom(4.0, 2.0, 4);

        assert_abs_diff_eq!(score, expected, epsilon = 1e-12);
    }
}
