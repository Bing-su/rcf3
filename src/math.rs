#[cfg(feature = "std")]
pub(crate) fn ln(x: f64) -> f64 {
    x.ln()
}

#[cfg(not(feature = "std"))]
pub(crate) fn ln(x: f64) -> f64 {
    libm::log(x)
}

#[cfg(feature = "std")]
pub(crate) fn log2(x: f64) -> f64 {
    x.log2()
}

#[cfg(not(feature = "std"))]
pub(crate) fn log2(x: f64) -> f64 {
    libm::log2(x)
}

#[cfg(feature = "std")]
pub(crate) fn powf(x: f64, n: f64) -> f64 {
    x.powf(n)
}

#[cfg(not(feature = "std"))]
pub(crate) fn powf(x: f64, n: f64) -> f64 {
    libm::pow(x, n)
}

#[cfg(feature = "std")]
pub(crate) fn floor(x: f64) -> f64 {
    x.floor()
}

#[cfg(not(feature = "std"))]
pub(crate) fn floor(x: f64) -> f64 {
    libm::floor(x)
}

#[cfg(feature = "std")]
pub(crate) fn asinh(x: f64) -> f64 {
    x.asinh()
}

#[cfg(not(feature = "std"))]
pub(crate) fn asinh(x: f64) -> f64 {
    libm::asinh(x)
}
