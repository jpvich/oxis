//! Risk and risk-adjusted-performance metrics: Sharpe / Sortino / Calmar,
//! Value-at-Risk and Expected Shortfall (historical, parametric Gaussian, and
//! Cornish-Fisher), tracking error and information ratio.
//!
//! **Conventions.** VaR and ES are returned as **positive loss magnitudes**.
//! Annualization scales per-period inputs: Sharpe and the information ratio are
//! multiplied by `√ppy`, volatilities by `√ppy`. The Sortino downside deviation
//! squares only sub-MAR returns over the **full-period denominator** `n`. The
//! historical VaR/ES use numpy's linear-interpolation quantile so the empirical
//! tail matches `numpy.quantile(r, 1−c)` exactly.

use crate::descriptive::{excess_kurtosis, mean, skewness, std_dev};
use crate::drawdown::max_drawdown;
use crate::returns::{annualized_return, simple_returns};
use oxis_core::{OxisError, brent, normal_cdf, normal_pdf};

/// The standard-normal quantile `Φ⁻¹(p)`, found by inverting [`normal_cdf`] with
/// Brent's method on a wide bracket. `p` must lie strictly in `(0, 1)`.
fn normal_quantile(p: f64) -> Result<f64, OxisError> {
    if !(0.0..=1.0).contains(&p) || p <= 0.0 || p >= 1.0 {
        return Err(OxisError::invalid_input(
            "normal_quantile: p must be in (0, 1)",
        ));
    }
    brent(-40.0, 40.0, 1e-14, 200, |x| normal_cdf(x) - p)
}

/// numpy-style linear-interpolation quantile of an ascending-sorted slice.
fn quantile_linear(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = (n as f64 - 1.0) * q;
    let lo = h.floor() as usize;
    if lo + 1 >= n {
        return sorted[n - 1];
    }
    let frac = h - lo as f64;
    sorted[lo] + frac * (sorted[lo + 1] - sorted[lo])
}

fn check_confidence(c: f64) -> Result<(), OxisError> {
    if c <= 0.0 || c >= 1.0 {
        return Err(OxisError::invalid_input(
            "confidence level must be in (0, 1)",
        ));
    }
    Ok(())
}

fn check_ppy(ppy: f64) -> Result<(), OxisError> {
    if ppy <= 0.0 {
        return Err(OxisError::invalid_input(
            "periods_per_year must be positive",
        ));
    }
    Ok(())
}

/// Annualized Sharpe ratio `(mean(r) − rf)/σ · √ppy` (population σ).
///
/// # Errors
/// [`OxisError::InvalidInput`] on an empty series, non-positive `ppy`, or zero
/// volatility.
pub fn sharpe_ratio(
    rets: &[f64],
    risk_free_per_period: f64,
    periods_per_year: f64,
) -> Result<f64, OxisError> {
    check_ppy(periods_per_year)?;
    let sd = std_dev(rets)?;
    if sd <= 0.0 {
        return Err(OxisError::invalid_input("sharpe_ratio: zero volatility"));
    }
    let excess = mean(rets)? - risk_free_per_period;
    Ok(excess / sd * periods_per_year.sqrt())
}

/// Annualized Sortino ratio `(mean(r) − MAR)/dd · √ppy`, with downside deviation
/// `dd = √( (1/n) Σ min(rₜ − MAR, 0)² )`.
///
/// # Errors
/// [`OxisError::InvalidInput`] on an empty series, non-positive `ppy`, or zero
/// downside deviation (no returns below MAR).
pub fn sortino_ratio(
    rets: &[f64],
    mar_per_period: f64,
    periods_per_year: f64,
) -> Result<f64, OxisError> {
    check_ppy(periods_per_year)?;
    if rets.is_empty() {
        return Err(OxisError::invalid_input("sortino_ratio: empty series"));
    }
    let n = rets.len() as f64;
    let downside: f64 = rets
        .iter()
        .map(|r| (r - mar_per_period).min(0.0).powi(2))
        .sum::<f64>()
        / n;
    let dd = downside.sqrt();
    if dd <= 0.0 {
        return Err(OxisError::invalid_input(
            "sortino_ratio: zero downside deviation (no returns below MAR)",
        ));
    }
    Ok((mean(rets)? - mar_per_period) / dd * periods_per_year.sqrt())
}

/// Calmar ratio: annualized return ÷ maximum drawdown, from a price series.
///
/// # Errors
/// [`OxisError::InvalidInput`] on bad prices, non-positive `ppy`, or zero drawdown.
pub fn calmar_ratio(prices: &[f64], periods_per_year: f64) -> Result<f64, OxisError> {
    check_ppy(periods_per_year)?;
    let rets = simple_returns(prices)?;
    let ann = annualized_return(&rets, periods_per_year)?;
    let mdd = max_drawdown(prices)?.max_drawdown;
    if mdd <= 0.0 {
        return Err(OxisError::invalid_input("calmar_ratio: zero drawdown"));
    }
    Ok(ann / mdd)
}

/// Historical (empirical) Value-at-Risk at confidence `c`, as a positive loss.
///
/// # Errors
/// [`OxisError::InvalidInput`] on an empty series or `c ∉ (0, 1)`.
pub fn historical_var(rets: &[f64], confidence: f64) -> Result<f64, OxisError> {
    check_confidence(confidence)?;
    if rets.is_empty() {
        return Err(OxisError::invalid_input("historical_var: empty series"));
    }
    let mut sorted = rets.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = quantile_linear(&sorted, 1.0 - confidence);
    Ok(-q)
}

/// Historical Expected Shortfall at confidence `c`: the mean of the left tail at
/// or below the VaR quantile, as a positive loss.
///
/// # Errors
/// [`OxisError::InvalidInput`] on an empty series or `c ∉ (0, 1)`.
pub fn historical_es(rets: &[f64], confidence: f64) -> Result<f64, OxisError> {
    check_confidence(confidence)?;
    if rets.is_empty() {
        return Err(OxisError::invalid_input("historical_es: empty series"));
    }
    let mut sorted = rets.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = quantile_linear(&sorted, 1.0 - confidence);
    let tail: Vec<f64> = rets.iter().copied().filter(|&r| r <= threshold).collect();
    let tail = if tail.is_empty() {
        vec![sorted[0]]
    } else {
        tail
    };
    let m = tail.iter().sum::<f64>() / tail.len() as f64;
    Ok(-m)
}

/// Parametric (Gaussian) VaR at confidence `c`, as a positive loss.
///
/// # Errors
/// [`OxisError::InvalidInput`] on an empty series or `c ∉ (0, 1)`.
pub fn parametric_var(rets: &[f64], confidence: f64) -> Result<f64, OxisError> {
    check_confidence(confidence)?;
    let mu = mean(rets)?;
    let sigma = std_dev(rets)?;
    let z = normal_quantile(1.0 - confidence)?;
    Ok(-(mu + z * sigma))
}

/// Parametric (Gaussian) Expected Shortfall at confidence `c`, as a positive loss.
///
/// `ES = −μ + σ·φ(z_α)/α` with `α = 1−c` and `z_α = Φ⁻¹(α)`.
///
/// # Errors
/// [`OxisError::InvalidInput`] on an empty series or `c ∉ (0, 1)`.
pub fn parametric_es(rets: &[f64], confidence: f64) -> Result<f64, OxisError> {
    check_confidence(confidence)?;
    let mu = mean(rets)?;
    let sigma = std_dev(rets)?;
    let alpha = 1.0 - confidence;
    let z = normal_quantile(alpha)?;
    Ok(-mu + sigma * normal_pdf(z) / alpha)
}

/// Cornish-Fisher (modified) VaR at confidence `c`, adjusting the Gaussian
/// quantile for sample skewness and excess kurtosis. Positive loss.
///
/// # Errors
/// [`OxisError::InvalidInput`] on `n < 2`, zero variance, or `c ∉ (0, 1)`.
pub fn cornish_fisher_var(rets: &[f64], confidence: f64) -> Result<f64, OxisError> {
    check_confidence(confidence)?;
    let mu = mean(rets)?;
    let sigma = std_dev(rets)?;
    let s = skewness(rets)?;
    let k = excess_kurtosis(rets)?;
    let z = normal_quantile(1.0 - confidence)?;
    let z_cf = z + (z * z - 1.0) * s / 6.0 + (z.powi(3) - 3.0 * z) * k / 24.0
        - (2.0 * z.powi(3) - 5.0 * z) * s * s / 36.0;
    Ok(-(mu + z_cf * sigma))
}

fn active_returns(port: &[f64], bench: &[f64], who: &str) -> Result<Vec<f64>, OxisError> {
    if port.len() != bench.len() {
        return Err(OxisError::invalid_input(format!(
            "{who}: series length mismatch ({} vs {})",
            port.len(),
            bench.len()
        )));
    }
    if port.is_empty() {
        return Err(OxisError::invalid_input(format!("{who}: empty series")));
    }
    Ok(port.iter().zip(bench.iter()).map(|(p, b)| p - b).collect())
}

/// Annualized tracking error: `σ(port − bench) · √ppy` (population σ).
///
/// # Errors
/// [`OxisError::InvalidInput`] on a length mismatch, empty series, or `ppy ≤ 0`.
pub fn tracking_error(
    port: &[f64],
    bench: &[f64],
    periods_per_year: f64,
) -> Result<f64, OxisError> {
    check_ppy(periods_per_year)?;
    let active = active_returns(port, bench, "tracking_error")?;
    Ok(std_dev(&active)? * periods_per_year.sqrt())
}

/// Annualized information ratio: `mean(active)/σ(active) · √ppy`.
///
/// # Errors
/// [`OxisError::InvalidInput`] on a length mismatch, empty series, `ppy ≤ 0`, or
/// zero active-return volatility.
pub fn information_ratio(
    port: &[f64],
    bench: &[f64],
    periods_per_year: f64,
) -> Result<f64, OxisError> {
    check_ppy(periods_per_year)?;
    let active = active_returns(port, bench, "information_ratio")?;
    let sd = std_dev(&active)?;
    if sd <= 0.0 {
        return Err(OxisError::invalid_input(
            "information_ratio: zero active-return volatility",
        ));
    }
    Ok(mean(&active)? / sd * periods_per_year.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    #[test]
    fn normal_quantile_known_points() {
        assert!(normal_quantile(0.5).unwrap().abs() < 1e-10);
        assert!((normal_quantile(0.975).unwrap() - 1.959963984540054).abs() < 1e-8);
        assert!((normal_quantile(0.05).unwrap() + 1.6448536269514722).abs() < 1e-8);
    }

    #[test]
    fn quantile_linear_matches_numpy_example() {
        // numpy.quantile([1,2,3,4], 0.5, method="linear") == 2.5
        let sorted = [1.0, 2.0, 3.0, 4.0];
        assert!((quantile_linear(&sorted, 0.5) - 2.5).abs() < TOL);
        assert!((quantile_linear(&sorted, 0.25) - 1.75).abs() < TOL);
    }

    #[test]
    fn sortino_zero_downside_errors() {
        // All returns above MAR=0 → no downside → error, not NaN.
        assert!(sortino_ratio(&[0.01, 0.02, 0.03], 0.0, 252.0).is_err());
    }

    #[test]
    fn var_ordering_param_vs_historical_finite() {
        let r = [-0.03, 0.01, -0.02, 0.015, -0.01, 0.02, -0.025, 0.005];
        assert!(historical_var(&r, 0.95).unwrap().is_finite());
        assert!(parametric_var(&r, 0.95).unwrap().is_finite());
        assert!(parametric_es(&r, 0.95).unwrap() >= parametric_var(&r, 0.95).unwrap());
    }

    #[test]
    fn invalid_inputs_error_not_panic() {
        assert!(sharpe_ratio(&[0.01, 0.01], 0.0, 252.0).is_err()); // zero vol
        assert!(historical_var(&[], 0.95).is_err());
        assert!(parametric_var(&[0.01, 0.02], 1.5).is_err()); // bad confidence
        assert!(tracking_error(&[0.01], &[0.01, 0.02], 252.0).is_err());
    }
}
