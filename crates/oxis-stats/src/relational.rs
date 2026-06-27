//! Pairwise relational statistics and autocorrelation.
//!
//! Covariance, correlation, and beta are **population (÷n)** estimators, matching
//! `numpy.cov(bias=True)` / `numpy.corrcoef` and `beta = cov(a,b)/var(b)`.
//! Autocorrelation uses the numpy-style biased estimator: mean-centered over the
//! whole series with the full sum-of-squares as the denominator.

use crate::descriptive::mean;
use oxis_core::OxisError;

fn check_pair(a: &[f64], b: &[f64], who: &str) -> Result<(), OxisError> {
    if a.len() != b.len() {
        return Err(OxisError::invalid_input(format!(
            "{who}: series length mismatch ({} vs {})",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Err(OxisError::invalid_input(format!("{who}: empty series")));
    }
    Ok(())
}

/// Population covariance `(1/n) Σ (aᵢ − ā)(bᵢ − b̄)`.
///
/// # Errors
/// [`OxisError::InvalidInput`] on a length mismatch or an empty series.
pub fn covariance(a: &[f64], b: &[f64]) -> Result<f64, OxisError> {
    check_pair(a, b, "covariance")?;
    let (ma, mb) = (mean(a)?, mean(b)?);
    let n = a.len() as f64;
    let s: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - ma) * (y - mb))
        .sum();
    Ok(s / n)
}

/// Pearson correlation coefficient.
///
/// # Errors
/// [`OxisError::InvalidInput`] on a length mismatch, an empty series, or if either
/// series has zero variance (correlation is undefined).
pub fn correlation(a: &[f64], b: &[f64]) -> Result<f64, OxisError> {
    let cov = covariance(a, b)?;
    let ma = mean(a)?;
    let mb = mean(b)?;
    let n = a.len() as f64;
    let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>() / n;
    let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum::<f64>() / n;
    if va <= 0.0 || vb <= 0.0 {
        return Err(OxisError::invalid_input("correlation: zero variance"));
    }
    Ok(cov / (va.sqrt() * vb.sqrt()))
}

/// Beta of `asset` against `benchmark`: `cov(asset, bench) / var(bench)`.
///
/// # Errors
/// [`OxisError::InvalidInput`] on a length mismatch, an empty series, or zero
/// benchmark variance.
pub fn beta(asset: &[f64], benchmark: &[f64]) -> Result<f64, OxisError> {
    let cov = covariance(asset, benchmark)?;
    let mb = mean(benchmark)?;
    let n = benchmark.len() as f64;
    let vb: f64 = benchmark.iter().map(|y| (y - mb).powi(2)).sum::<f64>() / n;
    if vb <= 0.0 {
        return Err(OxisError::invalid_input("beta: zero benchmark variance"));
    }
    Ok(cov / vb)
}

/// Autocorrelation at `lag` (numpy-style biased estimator).
///
/// `r_k = Σₜ (xₜ − x̄)(xₜ₊ₖ − x̄) / Σₜ (xₜ − x̄)²`, mean-centered over the whole
/// series. Lag 0 is `1.0`.
///
/// # Errors
/// [`OxisError::InvalidInput`] if `lag ≥ n`, the series is empty, or it has zero
/// variance.
pub fn autocorrelation(xs: &[f64], lag: usize) -> Result<f64, OxisError> {
    if xs.is_empty() {
        return Err(OxisError::invalid_input("autocorrelation: empty series"));
    }
    if lag >= xs.len() {
        return Err(OxisError::invalid_input(
            "autocorrelation: lag must be < series length",
        ));
    }
    let m = mean(xs)?;
    let denom: f64 = xs.iter().map(|x| (x - m).powi(2)).sum();
    if denom <= 0.0 {
        return Err(OxisError::invalid_input("autocorrelation: zero variance"));
    }
    if lag == 0 {
        return Ok(1.0);
    }
    let num: f64 = (0..xs.len() - lag)
        .map(|t| (xs[t] - m) * (xs[t + lag] - m))
        .sum();
    Ok(num / denom)
}

/// Autocorrelation function for lags `0..=max_lag`.
///
/// # Errors
/// Propagates [`autocorrelation`] errors (e.g. `max_lag ≥ n`).
pub fn acf(xs: &[f64], max_lag: usize) -> Result<Vec<f64>, OxisError> {
    (0..=max_lag).map(|k| autocorrelation(xs, k)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn perfectly_correlated() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [2.0, 4.0, 6.0, 8.0];
        assert!((correlation(&a, &b).unwrap() - 1.0).abs() < TOL);
        // b = 2a → beta of b on a is 2.
        assert!((beta(&b, &a).unwrap() - 2.0).abs() < TOL);
    }

    #[test]
    fn acf_lag0_is_one() {
        let xs = [1.0, -1.0, 1.0, -1.0, 1.0];
        let f = acf(&xs, 2).unwrap();
        assert!((f[0] - 1.0).abs() < TOL);
        // Perfect alternation → lag-1 autocorrelation is strongly negative.
        assert!(f[1] < 0.0);
    }

    #[test]
    fn invalid_inputs_error_not_panic() {
        assert!(covariance(&[1.0, 2.0], &[1.0]).is_err());
        assert!(correlation(&[1.0, 1.0], &[1.0, 1.0]).is_err());
        assert!(beta(&[1.0, 2.0], &[3.0, 3.0]).is_err());
        assert!(autocorrelation(&[1.0, 2.0], 2).is_err());
    }
}
