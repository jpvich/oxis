//! Portfolio risk aggregation, reusing `oxis::stats` for the single-series metrics.
//!
//! Builds the N×N **population** covariance matrix (÷n, matching
//! `numpy.cov(bias=True)`) from N aligned asset-return series, the weighted
//! portfolio return series, and the portfolio variance / volatility `wᵀΣw`.
//! Value-at-Risk is delegated to `oxis::stats` on the portfolio return series.

use crate::core::OxisError;
use crate::stats::covariance;

fn check_matrix(returns: &[Vec<f64>]) -> Result<usize, OxisError> {
    if returns.is_empty() {
        return Err(OxisError::invalid_input("risk: no asset return series"));
    }
    let t = returns[0].len();
    if t == 0 {
        return Err(OxisError::invalid_input("risk: empty return series"));
    }
    if returns.iter().any(|r| r.len() != t) {
        return Err(OxisError::invalid_input(
            "risk: asset return series must be equal length",
        ));
    }
    Ok(t)
}

/// The N×N population covariance matrix of N aligned asset-return series.
///
/// # Errors
/// [`OxisError::InvalidInput`] on empty / ragged input.
pub fn covariance_matrix(returns: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, OxisError> {
    check_matrix(returns)?;
    let n = returns.len();
    let mut cov = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let c = covariance(&returns[i], &returns[j])?;
            cov[i][j] = c;
            cov[j][i] = c;
        }
    }
    Ok(cov)
}

/// The weighted portfolio return series `rₜ = Σᵢ wᵢ·rᵢ,ₜ`.
///
/// # Errors
/// [`OxisError::InvalidInput`] on empty / ragged input or a weight/series count
/// mismatch.
pub fn portfolio_returns(
    asset_returns: &[Vec<f64>],
    weights: &[f64],
) -> Result<Vec<f64>, OxisError> {
    let t = check_matrix(asset_returns)?;
    if weights.len() != asset_returns.len() {
        return Err(OxisError::invalid_input(
            "portfolio_returns: weights must match the number of assets",
        ));
    }
    let mut out = vec![0.0_f64; t];
    for (w, series) in weights.iter().zip(asset_returns.iter()) {
        for (acc, &r) in out.iter_mut().zip(series.iter()) {
            *acc += w * r;
        }
    }
    Ok(out)
}

fn check_cov_weights(cov: &[Vec<f64>], w: &[f64]) -> Result<(), OxisError> {
    let n = cov.len();
    if n == 0 || w.len() != n || cov.iter().any(|r| r.len() != n) {
        return Err(OxisError::invalid_input(
            "portfolio risk: covariance must be square and match weights",
        ));
    }
    Ok(())
}

/// Portfolio variance `wᵀΣw`. Tiny negative values from rounding are clamped to 0.
///
/// # Errors
/// [`OxisError::InvalidInput`] if `cov` is not square or does not match `w`.
pub fn portfolio_variance(cov: &[Vec<f64>], w: &[f64]) -> Result<f64, OxisError> {
    check_cov_weights(cov, w)?;
    let n = w.len();
    let mut acc = 0.0;
    for i in 0..n {
        for j in 0..n {
            acc += w[i] * cov[i][j] * w[j];
        }
    }
    Ok(acc.max(0.0))
}

/// Portfolio volatility `√(wᵀΣw)`.
///
/// # Errors
/// As [`portfolio_variance`].
pub fn portfolio_volatility(cov: &[Vec<f64>], w: &[f64]) -> Result<f64, OxisError> {
    Ok(portfolio_variance(cov, w)?.sqrt())
}

/// Annualized portfolio volatility `vol·√ppy`.
///
/// # Errors
/// [`OxisError::InvalidInput`] as [`portfolio_variance`], or `ppy ≤ 0`.
pub fn annualized_volatility(
    cov: &[Vec<f64>],
    w: &[f64],
    periods_per_year: f64,
) -> Result<f64, OxisError> {
    if periods_per_year <= 0.0 {
        return Err(OxisError::invalid_input(
            "annualized_volatility: periods_per_year must be positive",
        ));
    }
    Ok(portfolio_volatility(cov, w)? * periods_per_year.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn diagonal_covariance_and_variance() {
        // Two uncorrelated-ish series; check symmetry + wᵀΣw.
        let returns = vec![
            vec![0.01, -0.02, 0.03, -0.01],
            vec![0.02, 0.00, -0.01, 0.015],
        ];
        let cov = covariance_matrix(&returns).unwrap();
        assert_eq!(cov.len(), 2);
        assert!((cov[0][1] - cov[1][0]).abs() < TOL);
        let var = portfolio_variance(&cov, &[0.6, 0.4]).unwrap();
        assert!(var >= 0.0);
        assert!((portfolio_volatility(&cov, &[0.6, 0.4]).unwrap() - var.sqrt()).abs() < TOL);
    }

    #[test]
    fn portfolio_returns_weighted_sum() {
        let r = portfolio_returns(&[vec![0.10, 0.20], vec![0.00, -0.10]], &[0.5, 0.5]).unwrap();
        assert!((r[0] - 0.05).abs() < TOL);
        assert!((r[1] - 0.05).abs() < TOL);
    }

    #[test]
    fn invalid_inputs_error_not_panic() {
        assert!(covariance_matrix(&[]).is_err());
        assert!(covariance_matrix(&[vec![0.1, 0.2], vec![0.1]]).is_err());
        assert!(portfolio_variance(&[vec![1.0, 0.0]], &[1.0]).is_err());
        assert!(annualized_volatility(&[vec![1.0]], &[1.0], 0.0).is_err());
    }
}
