//! # oxis-stats — statistics & time-series analytics (Ring 3)
//!
//! A **Kind A** (pure compute) module: descriptive statistics, return transforms,
//! risk and risk-adjusted-performance metrics (Sharpe / Sortino / Calmar),
//! Value-at-Risk and Expected Shortfall (historical, parametric Gaussian, and
//! Cornish-Fisher), drawdown, pairwise relational statistics (covariance,
//! correlation, beta), active-return metrics (tracking error, information ratio),
//! autocorrelation, and the Jarque-Bera normality test. Every function is a pure
//! operation over `&[f64]` and returns a `Result` — never `NaN`, `Inf`, or a
//! panic. Other modules (portfolio, ML) build on this one.
//!
//! ## Conventions (mirrored exactly by the numpy/scipy validation oracle)
//!
//! - **Moments are population / biased (÷n)** to match `numpy.var(ddof=0)`,
//!   `scipy.stats.skew(bias=True)`, and `scipy.stats.kurtosis(fisher=True,
//!   bias=True)` (excess kurtosis).
//! - **VaR and ES are positive loss magnitudes.** Historical VaR uses numpy's
//!   linear-interpolation quantile of the returns at probability `1 − c`.
//! - **Sortino** squares only sub-MAR returns over the full denominator `n`.
//! - **Annualization** scales per-period inputs: volatility and Sharpe / IR /
//!   tracking error by `√ppy`; annualized return is geometric.
//! - **Jarque-Bera** `= n/6·(S² + K²/4)` with `K` the excess kurtosis; the p-value
//!   is the χ²₂ survival `exp(−JB/2)`.
//! - **Autocorrelation** is the numpy-style biased estimator (mean-centered, full
//!   sum-of-squares denominator).

#![forbid(unsafe_code)]

mod descriptive;
mod drawdown;
mod normality;
mod relational;
mod result;
mod returns;
mod risk;

pub use descriptive::{excess_kurtosis, mean, skewness, std_dev, variance};
pub use drawdown::{Drawdown, max_drawdown};
pub use normality::jarque_bera;
pub use relational::{acf, autocorrelation, beta, correlation, covariance};
pub use result::{SampleKind, StatsReport, StatsRequest, assemble};
pub use returns::{
    annualized_return, annualized_volatility, cumulative_return, log_returns, simple_returns,
};
pub use risk::{
    calmar_ratio, cornish_fisher_var, historical_es, historical_var, information_ratio,
    parametric_es, parametric_var, sharpe_ratio, sortino_ratio, tracking_error,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The `module → core` dependency direction compiles and a metric runs.
    #[test]
    fn end_to_end_smoke() {
        let r = [0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.005, 0.012];
        assert!(mean(&r).is_ok());
        assert!(sharpe_ratio(&r, 0.0, 252.0).unwrap().is_finite());
        assert!(historical_var(&r, 0.95).unwrap() > 0.0);
    }
}
