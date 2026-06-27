//! Markowitz mean-variance optimization (unconstrained / closed-form).
//!
//! With only the budget constraint `Σw = 1` (no long-only or other inequality
//! constraints, so shorting is allowed), the efficient-frontier weights have a
//! closed form in terms of `Σ⁻¹`. We never form the inverse explicitly: the two
//! solves `x₁ = Σ⁻¹·1` and `x_μ = Σ⁻¹·μ` (via [`solve_linear_system`]) give the
//! scalars `A = 1ᵀx₁`, `B = 1ᵀx_μ`, `C = μᵀx_μ`, `D = AC − B²`, from which every
//! frontier portfolio follows. This matches `numpy.linalg.solve` bit-for-bit.
//!
//! Constrained (long-only / QP) optimization is intentionally out of scope.

use crate::risk::portfolio_variance;
use oxis_core::{OxisError, solve_linear_system};

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn check(cov: &[Vec<f64>], mean: Option<&[f64]>) -> Result<usize, OxisError> {
    let n = cov.len();
    if n == 0 || cov.iter().any(|r| r.len() != n) {
        return Err(OxisError::invalid_input(
            "optimize: covariance must be a non-empty square matrix",
        ));
    }
    if let Some(m) = mean {
        if m.len() != n {
            return Err(OxisError::invalid_input(
                "optimize: mean length must match covariance order",
            ));
        }
    }
    Ok(n)
}

/// Global minimum-variance weights `w = Σ⁻¹1 / (1ᵀΣ⁻¹1)`.
///
/// # Errors
/// [`OxisError::InvalidInput`] on a malformed matrix; [`OxisError::Numerical`] if
/// the covariance is singular or `A = 1ᵀΣ⁻¹1` is zero.
pub fn min_variance_weights(cov: &[Vec<f64>]) -> Result<Vec<f64>, OxisError> {
    let n = check(cov, None)?;
    let ones = vec![1.0; n];
    let x1 = solve_linear_system(cov, &ones)?;
    let a = x1.iter().sum::<f64>();
    if a == 0.0 {
        return Err(OxisError::numerical("min_variance_weights: 1ᵀΣ⁻¹1 is zero"));
    }
    Ok(x1.iter().map(|v| v / a).collect())
}

/// Tangency (max-Sharpe) weights `w ∝ Σ⁻¹(μ − rf·1)`, normalized to sum to 1.
///
/// # Errors
/// [`OxisError::InvalidInput`] on a malformed matrix / mean; [`OxisError::Numerical`]
/// if the covariance is singular or the excess-return weights sum to zero.
pub fn tangency_weights(cov: &[Vec<f64>], mean: &[f64], rf: f64) -> Result<Vec<f64>, OxisError> {
    let n = check(cov, Some(mean))?;
    let excess: Vec<f64> = (0..n).map(|i| mean[i] - rf).collect();
    let z = solve_linear_system(cov, &excess)?;
    let s = z.iter().sum::<f64>();
    if s == 0.0 {
        return Err(OxisError::numerical(
            "tangency_weights: excess-return weights sum to zero",
        ));
    }
    Ok(z.iter().map(|v| v / s).collect())
}

/// Efficient-frontier weights achieving a target expected return `target`.
///
/// `w = x₁·(C − B·target)/D + x_μ·(A·target − B)/D`.
///
/// # Errors
/// [`OxisError::InvalidInput`] on a malformed matrix / mean; [`OxisError::Numerical`]
/// if the covariance is singular or `D = AC − B² = 0` (degenerate frontier).
pub fn efficient_frontier_point(
    cov: &[Vec<f64>],
    mean: &[f64],
    target: f64,
) -> Result<Vec<f64>, OxisError> {
    let n = check(cov, Some(mean))?;
    let ones = vec![1.0; n];
    let x1 = solve_linear_system(cov, &ones)?;
    let xmu = solve_linear_system(cov, mean)?;
    let a = x1.iter().sum::<f64>();
    let b = xmu.iter().sum::<f64>();
    let c = dot(mean, &xmu);
    let d = a * c - b * b;
    if d == 0.0 {
        return Err(OxisError::numerical(
            "efficient_frontier_point: degenerate frontier (AC − B² = 0)",
        ));
    }
    let g = (c - b * target) / d;
    let h = (a * target - b) / d;
    Ok((0..n).map(|i| x1[i] * g + xmu[i] * h).collect())
}

/// Expected return, variance, and volatility of a weighted portfolio.
///
/// # Errors
/// [`OxisError::InvalidInput`] on a dimension mismatch.
pub fn frontier_stats(
    weights: &[f64],
    mean: &[f64],
    cov: &[Vec<f64>],
) -> Result<(f64, f64, f64), OxisError> {
    if weights.len() != mean.len() {
        return Err(OxisError::invalid_input(
            "frontier_stats: weights and mean length mismatch",
        ));
    }
    let ret = dot(weights, mean);
    let var = portfolio_variance(cov, weights)?;
    Ok((ret, var, var.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn sample_cov() -> Vec<Vec<f64>> {
        vec![
            vec![0.0100, 0.0018, 0.0011],
            vec![0.0018, 0.0109, 0.0026],
            vec![0.0011, 0.0026, 0.0199],
        ]
    }

    #[test]
    fn min_variance_weights_sum_to_one() {
        let w = min_variance_weights(&sample_cov()).unwrap();
        assert!((w.iter().sum::<f64>() - 1.0).abs() < TOL);
    }

    #[test]
    fn frontier_point_hits_target_return() {
        let cov = sample_cov();
        let mean = [0.08, 0.10, 0.13];
        let w = efficient_frontier_point(&cov, &mean, 0.11).unwrap();
        assert!((w.iter().sum::<f64>() - 1.0).abs() < TOL);
        let (ret, _, _) = frontier_stats(&w, &mean, &cov).unwrap();
        assert!((ret - 0.11).abs() < TOL);
    }

    #[test]
    fn tangency_weights_sum_to_one() {
        let w = tangency_weights(&sample_cov(), &[0.08, 0.10, 0.13], 0.02).unwrap();
        assert!((w.iter().sum::<f64>() - 1.0).abs() < TOL);
    }

    #[test]
    fn singular_covariance_errors_not_panics() {
        let singular = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        assert!(min_variance_weights(&singular).is_err());
    }
}
