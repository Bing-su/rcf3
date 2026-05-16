use crate::error::{RcfError, Result};

pub(crate) fn counts_to_anom(total: f64, current: f64, current_time: u64) -> f64 {
    let cur_t = (current_time as f64).max(1.0);
    let cur_mean = total / cur_t;
    if cur_mean <= f64::EPSILON {
        return 0.0;
    }

    let sqerr = current - cur_mean;
    let sqerr = sqerr * sqerr;

    sqerr / cur_mean + sqerr / (cur_mean * (cur_t - 1.0).max(1.0))
}

pub(crate) fn ceil_log2(value: usize) -> Result<usize> {
    if value < 2 {
        return Err(RcfError::InvalidArgument("num_buckets must be >= 2".into()));
    }

    let bits = ((value - 1).ilog2() + 1) as usize;
    Ok(bits)
}

#[cfg(test)]
mod tests {
    use super::counts_to_anom;

    #[test]
    fn counts_to_anom_penalizes_negative_deviation() {
        let score = counts_to_anom(10.0, 0.0, 2);
        assert!(score > 0.0);
        assert!((score - 10.0).abs() < 1e-12);
    }
}
