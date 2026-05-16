use crate::error::{RcfError, Result};
pub(crate) use crate::math_utils::{floor_f64, ln_f64, log10_f64};

pub(crate) fn counts_to_anom(total: f64, current: f64, current_time: u64) -> f64 {
    let cur_t = (current_time as f64).max(1.0);
    let cur_mean = total / cur_t;
    if cur_mean <= f64::EPSILON {
        return 0.0;
    }

    let sqerr = (current - cur_mean).max(0.0);
    let sqerr = sqerr * sqerr;

    sqerr / cur_mean + sqerr / (cur_mean * (cur_t - 1.0).max(1.0))
}

pub(crate) fn ceil_log2(value: usize) -> Result<usize> {
    if value < 2 {
        return Err(RcfError::InvalidArgument("num_buckets must be >= 2".into()));
    }

    let mut p = 1usize;
    let mut bits = 0usize;
    while p < value {
        p <<= 1;
        bits += 1;
    }
    Ok(bits)
}

pub(crate) fn uniform_symmetric(x: u64) -> f64 {
    let u = (x as f64) / (u64::MAX as f64);
    u * 2.0 - 1.0
}
