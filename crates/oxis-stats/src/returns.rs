//! Return transforms and annualization.
//!
//! Conventions: simple return `rₜ = pₜ/pₜ₋₁ − 1`; log return `ln(pₜ/pₜ₋₁)`;
//! cumulative return `∏(1+rₜ) − 1`; annualized (geometric) return
//! `(∏(1+rₜ))^{ppy/n} − 1`; annualized volatility `σ·√ppy` with `σ` the
//! population standard deviation of the per-period returns.

use crate::descriptive::std_dev;
use oxis_core::OxisError;

/// Simple (arithmetic) returns from a price series.
///
/// # Errors
/// [`OxisError::InvalidInput`] if fewer than 2 prices, or any price is `≤ 0`.
pub fn simple_returns(prices: &[f64]) -> Result<Vec<f64>, OxisError> {
    if prices.len() < 2 {
        return Err(OxisError::invalid_input(
            "simple_returns: need at least 2 prices",
        ));
    }
    if prices.iter().any(|&p| p <= 0.0) {
        return Err(OxisError::invalid_input(
            "simple_returns: prices must be positive",
        ));
    }
    Ok(prices.windows(2).map(|w| w[1] / w[0] - 1.0).collect())
}

/// Log (continuously-compounded) returns from a price series.
///
/// # Errors
/// [`OxisError::InvalidInput`] if fewer than 2 prices, or any price is `≤ 0`.
pub fn log_returns(prices: &[f64]) -> Result<Vec<f64>, OxisError> {
    if prices.len() < 2 {
        return Err(OxisError::invalid_input(
            "log_returns: need at least 2 prices",
        ));
    }
    if prices.iter().any(|&p| p <= 0.0) {
        return Err(OxisError::invalid_input(
            "log_returns: prices must be positive",
        ));
    }
    Ok(prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect())
}

/// Cumulative (total) return over the series, `∏(1+rₜ) − 1`.
///
/// # Errors
/// [`OxisError::InvalidInput`] on an empty series.
pub fn cumulative_return(rets: &[f64]) -> Result<f64, OxisError> {
    if rets.is_empty() {
        return Err(OxisError::invalid_input("cumulative_return: empty series"));
    }
    let growth: f64 = rets.iter().map(|r| 1.0 + r).product();
    Ok(growth - 1.0)
}

/// Geometric annualized return `(∏(1+rₜ))^{ppy/n} − 1`.
///
/// # Errors
/// [`OxisError::InvalidInput`] on an empty series, `ppy ≤ 0`, or if total growth
/// is non-positive (a cumulative loss of 100% or worse, where the geometric mean
/// is undefined).
pub fn annualized_return(rets: &[f64], periods_per_year: f64) -> Result<f64, OxisError> {
    if rets.is_empty() {
        return Err(OxisError::invalid_input("annualized_return: empty series"));
    }
    if periods_per_year <= 0.0 {
        return Err(OxisError::invalid_input(
            "annualized_return: periods_per_year must be positive",
        ));
    }
    let growth: f64 = rets.iter().map(|r| 1.0 + r).product();
    if growth <= 0.0 {
        return Err(OxisError::invalid_input(
            "annualized_return: total growth non-positive",
        ));
    }
    let exponent = periods_per_year / rets.len() as f64;
    Ok(growth.powf(exponent) - 1.0)
}

/// Annualized volatility `σ·√ppy` (population std of per-period returns).
///
/// # Errors
/// [`OxisError::InvalidInput`] on an empty series or `ppy ≤ 0`.
pub fn annualized_volatility(rets: &[f64], periods_per_year: f64) -> Result<f64, OxisError> {
    if periods_per_year <= 0.0 {
        return Err(OxisError::invalid_input(
            "annualized_volatility: periods_per_year must be positive",
        ));
    }
    Ok(std_dev(rets)? * periods_per_year.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn simple_returns_basic() {
        let r = simple_returns(&[100.0, 110.0, 99.0]).unwrap();
        assert!((r[0] - 0.10).abs() < TOL);
        assert!((r[1] - (-0.10)).abs() < TOL);
    }

    #[test]
    fn cumulative_matches_compounding() {
        let r = [0.10, -0.10];
        // (1.1)(0.9) - 1 = -0.01
        assert!((cumulative_return(&r).unwrap() - (-0.01)).abs() < TOL);
    }

    #[test]
    fn annualized_return_doubling_in_half_a_year() {
        // One period that doubles, ppy=2 → annualized (geometric) = 2^2 - 1 = 3.
        assert!((annualized_return(&[1.0], 2.0).unwrap() - 3.0).abs() < TOL);
    }

    #[test]
    fn invalid_inputs_error_not_panic() {
        assert!(simple_returns(&[100.0]).is_err());
        assert!(log_returns(&[100.0, -5.0]).is_err());
        assert!(cumulative_return(&[]).is_err());
        assert!(annualized_return(&[-1.5], 252.0).is_err());
        assert!(annualized_volatility(&[0.01, 0.02], 0.0).is_err());
    }
}
