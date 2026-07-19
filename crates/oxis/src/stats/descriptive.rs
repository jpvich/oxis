//! Descriptive moments of a sample.
//!
//! All estimators are **population / biased (÷n)**, matching the defaults of the
//! validation oracle (`numpy.mean`, `numpy.var(ddof=0)`, `numpy.std(ddof=0)`,
//! `scipy.stats.skew(bias=True)`, `scipy.stats.kurtosis(fisher=True, bias=True)`).
//! Skewness and excess kurtosis are the standardized third and fourth central
//! moments: `m3 / m2^{3/2}` and `m4 / m2^2 − 3`.

use crate::core::OxisError;

/// Arithmetic mean. Errors on an empty sample.
pub fn mean(xs: &[f64]) -> Result<f64, OxisError> {
    if xs.is_empty() {
        return Err(OxisError::invalid_input("mean: empty sample"));
    }
    Ok(xs.iter().sum::<f64>() / xs.len() as f64)
}

/// The `k`-th central moment `m_k = (1/n) Σ (xᵢ − x̄)ᵏ` (population).
fn central_moment(xs: &[f64], mean: f64, k: i32) -> f64 {
    let n = xs.len() as f64;
    xs.iter().map(|x| (x - mean).powi(k)).sum::<f64>() / n
}

/// Population variance `(1/n) Σ (xᵢ − x̄)²`. Errors on an empty sample.
pub fn variance(xs: &[f64]) -> Result<f64, OxisError> {
    let m = mean(xs)?;
    Ok(central_moment(xs, m, 2))
}

/// Population standard deviation `√variance`. Errors on an empty sample.
pub fn std_dev(xs: &[f64]) -> Result<f64, OxisError> {
    Ok(variance(xs)?.sqrt())
}

/// Sample skewness (biased, population) `m3 / m2^{3/2}`.
///
/// # Errors
/// [`OxisError::InvalidInput`] if `n < 2` or the variance is zero (skewness is
/// undefined for a degenerate sample).
pub fn skewness(xs: &[f64]) -> Result<f64, OxisError> {
    if xs.len() < 2 {
        return Err(OxisError::invalid_input("skewness: need at least 2 points"));
    }
    let m = mean(xs)?;
    let m2 = central_moment(xs, m, 2);
    if m2 <= 0.0 {
        return Err(OxisError::invalid_input("skewness: zero variance"));
    }
    let m3 = central_moment(xs, m, 3);
    Ok(m3 / m2.powf(1.5))
}

/// Excess kurtosis (Fisher, biased) `m4 / m2² − 3`.
///
/// # Errors
/// [`OxisError::InvalidInput`] if `n < 2` or the variance is zero.
pub fn excess_kurtosis(xs: &[f64]) -> Result<f64, OxisError> {
    if xs.len() < 2 {
        return Err(OxisError::invalid_input(
            "excess_kurtosis: need at least 2 points",
        ));
    }
    let m = mean(xs)?;
    let m2 = central_moment(xs, m, 2);
    if m2 <= 0.0 {
        return Err(OxisError::invalid_input("excess_kurtosis: zero variance"));
    }
    let m4 = central_moment(xs, m, 4);
    Ok(m4 / (m2 * m2) - 3.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn mean_variance_textbook() {
        let xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        // Classic example: mean 5, population variance 4, std 2.
        assert!((mean(&xs).unwrap() - 5.0).abs() < TOL);
        assert!((variance(&xs).unwrap() - 4.0).abs() < TOL);
        assert!((std_dev(&xs).unwrap() - 2.0).abs() < TOL);
    }

    #[test]
    fn symmetric_sample_has_zero_skew() {
        let xs = [-2.0, -1.0, 0.0, 1.0, 2.0];
        assert!(skewness(&xs).unwrap().abs() < TOL);
    }

    #[test]
    fn empty_and_degenerate_error_not_panic() {
        assert!(mean(&[]).is_err());
        assert!(variance(&[]).is_err());
        assert!(skewness(&[1.0]).is_err());
        assert!(skewness(&[3.0, 3.0, 3.0]).is_err());
        assert!(excess_kurtosis(&[3.0, 3.0]).is_err());
    }
}
