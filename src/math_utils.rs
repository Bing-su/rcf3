#[cfg(feature = "std")]
pub(crate) fn floor_f64(x: f64) -> f64 {
    x.floor()
}

#[cfg(not(feature = "std"))]
pub(crate) fn floor_f64(x: f64) -> f64 {
    libm::floor(x)
}

#[cfg(feature = "std")]
pub(crate) fn log10_f64(x: f64) -> f64 {
    x.log10()
}

#[cfg(not(feature = "std"))]
pub(crate) fn log10_f64(x: f64) -> f64 {
    libm::log10(x)
}

#[cfg(feature = "std")]
pub(crate) fn ln_f64(x: f64) -> f64 {
    x.ln()
}

#[cfg(not(feature = "std"))]
pub(crate) fn ln_f64(x: f64) -> f64 {
    libm::log(x)
}
