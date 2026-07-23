//! [`StatsReport`] — the renderable result that gathers every metric for one
//! sample (plus optional price, benchmark, and lag inputs) into a single wide
//! record, and [`assemble`] which computes it.
//!
//! Metrics that don't apply to the given inputs render as `Cell::Null` / JSON
//! `null`: financial metrics when the sample is generic `Values`, drawdown and
//! Calmar when no price series is supplied, and the relational metrics when no
//! benchmark is supplied — the same optional-field pattern as `CurveQuery`.

use crate::core::{Cell, Column, OxisError, Tabular};
use crate::stats::descriptive::{excess_kurtosis, mean, skewness, std_dev, variance};
use crate::stats::drawdown::max_drawdown;
use crate::stats::normality::jarque_bera;
use crate::stats::relational::{autocorrelation, beta, correlation, covariance};
use crate::stats::returns::{annualized_return, annualized_volatility, cumulative_return};
use crate::stats::risk::{
    cornish_fisher_var, historical_es, historical_var, information_ratio, parametric_es,
    parametric_var, sharpe_ratio, sortino_ratio, tracking_error,
};
use serde::Serialize;

/// Whether the primary sample is a series of periodic returns (financial metrics
/// apply) or a generic numeric sample (descriptive metrics only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleKind {
    /// A periodic-returns series — enables returns / risk / VaR metrics.
    Returns,
    /// A generic numeric sample — descriptive statistics only.
    Values,
}

/// Inputs for [`assemble`]: the sample plus optional price / benchmark series and
/// the scalar parameters.
#[derive(Debug, Clone, Copy)]
pub struct StatsRequest<'a> {
    /// The primary sample (returns or generic values).
    pub sample: &'a [f64],
    /// How to interpret `sample`.
    pub kind: SampleKind,
    /// Price/equity series for drawdown & Calmar (optional).
    pub prices: Option<&'a [f64]>,
    /// Benchmark returns for beta / TE / IR / covariance / correlation (optional).
    pub benchmark: Option<&'a [f64]>,
    /// Per-period risk-free rate / MAR (for Sharpe & Sortino).
    pub risk_free: f64,
    /// Periods per year (annualization factor).
    pub periods_per_year: f64,
    /// Confidence level for VaR / ES (e.g. `0.95`).
    pub confidence: f64,
    /// Extra autocorrelation lag to report (optional).
    pub lag: Option<usize>,
}

/// A wide record of every computed statistic for one sample.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StatsReport {
    /// Number of observations in the sample.
    pub count: usize,
    /// Echoed periods-per-year annualization factor.
    pub periods_per_year: f64,
    /// Echoed VaR/ES confidence level.
    pub confidence: f64,
    /// Mean.
    pub mean: f64,
    /// Population variance.
    pub variance: f64,
    /// Population standard deviation.
    pub std_dev: f64,
    /// Skewness (biased), if defined.
    pub skewness: Option<f64>,
    /// Excess kurtosis (Fisher, biased), if defined.
    pub excess_kurtosis: Option<f64>,
    /// Jarque-Bera statistic, if defined.
    pub jarque_bera: Option<f64>,
    /// Jarque-Bera p-value, if defined.
    pub jarque_bera_pvalue: Option<f64>,
    /// Lag-1 autocorrelation, if defined.
    pub autocorr_lag1: Option<f64>,
    /// Autocorrelation at the requested `lag`, if any.
    pub autocorr_at_lag: Option<f64>,
    /// Cumulative return (returns samples only).
    pub cumulative_return: Option<f64>,
    /// Annualized (geometric) return.
    pub annualized_return: Option<f64>,
    /// Annualized volatility.
    pub annualized_volatility: Option<f64>,
    /// Annualized Sharpe ratio.
    pub sharpe: Option<f64>,
    /// Annualized Sortino ratio.
    pub sortino: Option<f64>,
    /// Historical VaR (positive loss).
    pub historical_var: Option<f64>,
    /// Historical Expected Shortfall (positive loss).
    pub historical_es: Option<f64>,
    /// Parametric Gaussian VaR (positive loss).
    pub parametric_var: Option<f64>,
    /// Parametric Gaussian Expected Shortfall (positive loss).
    pub parametric_es: Option<f64>,
    /// Cornish-Fisher VaR (positive loss).
    pub cornish_fisher_var: Option<f64>,
    /// Maximum drawdown (positive fraction; needs a price series).
    pub max_drawdown: Option<f64>,
    /// Periods from peak to the worst trough.
    pub max_drawdown_duration: Option<usize>,
    /// Calmar ratio (needs a price series).
    pub calmar: Option<f64>,
    /// Covariance with the benchmark.
    pub covariance: Option<f64>,
    /// Correlation with the benchmark.
    pub correlation: Option<f64>,
    /// Beta vs the benchmark.
    pub beta: Option<f64>,
    /// Annualized tracking error vs the benchmark.
    pub tracking_error: Option<f64>,
    /// Annualized information ratio vs the benchmark.
    pub information_ratio: Option<f64>,
}

/// Compute every applicable statistic for a [`StatsRequest`].
///
/// The descriptive moments (`mean`, `variance`, `std_dev`) are required and
/// propagate errors (e.g. an empty sample). Every other metric is best-effort:
/// it is reported when its inputs make it well-defined and left `None` otherwise,
/// so a single call describes whatever the caller supplied without panicking.
///
/// # Errors
/// [`OxisError::InvalidInput`] if the sample is empty.
pub fn assemble(req: &StatsRequest) -> Result<StatsReport, OxisError> {
    let s = req.sample;
    let ppy = req.periods_per_year;
    let conf = req.confidence;

    // Required descriptive core.
    let mean_v = mean(s)?;
    let variance_v = variance(s)?;
    let std_v = std_dev(s)?;

    // Best-effort descriptive extras (undefined for degenerate samples → None).
    let skew = skewness(s).ok();
    let kurt = excess_kurtosis(s).ok();
    let (jb, jbp) = match jarque_bera(s) {
        Ok((a, b)) => (Some(a), Some(b)),
        Err(_) => (None, None),
    };
    let autocorr_lag1 = autocorrelation(s, 1).ok();
    let autocorr_at_lag = req.lag.and_then(|l| autocorrelation(s, l).ok());

    // Financial metrics apply only to a returns series.
    let financial = matches!(req.kind, SampleKind::Returns);
    let cumulative_return_v = financial.then(|| cumulative_return(s).ok()).flatten();
    let annualized_return_v = financial.then(|| annualized_return(s, ppy).ok()).flatten();
    let annualized_volatility_v = financial
        .then(|| annualized_volatility(s, ppy).ok())
        .flatten();
    let sharpe = financial
        .then(|| sharpe_ratio(s, req.risk_free, ppy).ok())
        .flatten();
    let sortino = financial
        .then(|| sortino_ratio(s, req.risk_free, ppy).ok())
        .flatten();
    let historical_var_v = financial.then(|| historical_var(s, conf).ok()).flatten();
    let historical_es_v = financial.then(|| historical_es(s, conf).ok()).flatten();
    let parametric_var_v = financial.then(|| parametric_var(s, conf).ok()).flatten();
    let parametric_es_v = financial.then(|| parametric_es(s, conf).ok()).flatten();
    let cornish_fisher_var_v = financial
        .then(|| cornish_fisher_var(s, conf).ok())
        .flatten();

    // Drawdown / Calmar need a price series.
    let (max_drawdown_v, max_drawdown_duration, calmar) = match req.prices {
        Some(p) => {
            let dd = max_drawdown(p).ok();
            (
                dd.map(|d| d.max_drawdown),
                dd.map(|d| d.duration),
                crate::stats::risk::calmar_ratio(p, ppy).ok(),
            )
        }
        None => (None, None, None),
    };

    // Relational metrics need a benchmark.
    let (covariance_v, correlation_v, beta_v, tracking_error_v, information_ratio_v) =
        match req.benchmark {
            Some(b) => (
                covariance(s, b).ok(),
                correlation(s, b).ok(),
                beta(s, b).ok(),
                tracking_error(s, b, ppy).ok(),
                information_ratio(s, b, ppy).ok(),
            ),
            None => (None, None, None, None, None),
        };

    Ok(StatsReport {
        count: s.len(),
        periods_per_year: ppy,
        confidence: conf,
        mean: mean_v,
        variance: variance_v,
        std_dev: std_v,
        skewness: skew,
        excess_kurtosis: kurt,
        jarque_bera: jb,
        jarque_bera_pvalue: jbp,
        autocorr_lag1,
        autocorr_at_lag,
        cumulative_return: cumulative_return_v,
        annualized_return: annualized_return_v,
        annualized_volatility: annualized_volatility_v,
        sharpe,
        sortino,
        historical_var: historical_var_v,
        historical_es: historical_es_v,
        parametric_var: parametric_var_v,
        parametric_es: parametric_es_v,
        cornish_fisher_var: cornish_fisher_var_v,
        max_drawdown: max_drawdown_v,
        max_drawdown_duration,
        calmar,
        covariance: covariance_v,
        correlation: correlation_v,
        beta: beta_v,
        tracking_error: tracking_error_v,
        information_ratio: information_ratio_v,
    })
}

impl Tabular for StatsReport {
    fn columns(&self) -> Vec<Column> {
        vec![
            Column::new("count"),
            Column::new("periods_per_year"),
            Column::new("confidence"),
            Column::new("mean"),
            Column::new("variance"),
            Column::new("std_dev"),
            Column::new("skewness"),
            Column::new("excess_kurtosis"),
            Column::new("jarque_bera"),
            Column::new("jarque_bera_pvalue"),
            Column::new("autocorr_lag1"),
            Column::new("autocorr_at_lag"),
            Column::new("cumulative_return"),
            Column::new("annualized_return"),
            Column::new("annualized_volatility"),
            Column::new("sharpe"),
            Column::new("sortino"),
            Column::new("historical_var"),
            Column::new("historical_es"),
            Column::new("parametric_var"),
            Column::new("parametric_es"),
            Column::new("cornish_fisher_var"),
            Column::new("max_drawdown"),
            Column::new("max_drawdown_duration"),
            Column::new("calmar"),
            Column::new("covariance"),
            Column::new("correlation"),
            Column::new("beta"),
            Column::new("tracking_error"),
            Column::new("information_ratio"),
        ]
    }

    fn cells(&self) -> Vec<Cell> {
        vec![
            Cell::Int(self.count as i64),
            Cell::F64(self.periods_per_year),
            Cell::F64(self.confidence),
            Cell::F64(self.mean),
            Cell::F64(self.variance),
            Cell::F64(self.std_dev),
            self.skewness.into(),
            self.excess_kurtosis.into(),
            self.jarque_bera.into(),
            self.jarque_bera_pvalue.into(),
            self.autocorr_lag1.into(),
            self.autocorr_at_lag.into(),
            self.cumulative_return.into(),
            self.annualized_return.into(),
            self.annualized_volatility.into(),
            self.sharpe.into(),
            self.sortino.into(),
            self.historical_var.into(),
            self.historical_es.into(),
            self.parametric_var.into(),
            self.parametric_es.into(),
            self.cornish_fisher_var.into(),
            self.max_drawdown.into(),
            self.max_drawdown_duration.map(|d| d as i64).into(),
            self.calmar.into(),
            self.covariance.into(),
            self.correlation.into(),
            self.beta.into(),
            self.tracking_error.into(),
            self.information_ratio.into(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tabular_columns_and_cells_align() {
        let s = [0.01, -0.02, 0.015, 0.005, -0.01, 0.02];
        let req = StatsRequest {
            sample: &s,
            kind: SampleKind::Returns,
            prices: None,
            benchmark: None,
            risk_free: 0.0,
            periods_per_year: 252.0,
            confidence: 0.95,
            lag: None,
        };
        let r = assemble(&req).unwrap();
        assert_eq!(r.columns().len(), r.cells().len());
        // Returns sample → financial metrics present, benchmark ones Null.
        assert!(r.sharpe.is_some());
        assert!(r.beta.is_none());
        assert!(r.max_drawdown.is_none());
    }

    #[test]
    fn values_kind_suppresses_financial_metrics() {
        let s = [1.0, 2.0, 3.0, 4.0, 5.0];
        let req = StatsRequest {
            sample: &s,
            kind: SampleKind::Values,
            prices: None,
            benchmark: None,
            risk_free: 0.0,
            periods_per_year: 252.0,
            confidence: 0.95,
            lag: Some(2),
        };
        let r = assemble(&req).unwrap();
        assert!(r.sharpe.is_none());
        assert!(r.historical_var.is_none());
        assert!(r.autocorr_at_lag.is_some());
        assert!(r.mean > 0.0);
    }

    #[test]
    fn empty_sample_errors() {
        let req = StatsRequest {
            sample: &[],
            kind: SampleKind::Returns,
            prices: None,
            benchmark: None,
            risk_free: 0.0,
            periods_per_year: 252.0,
            confidence: 0.95,
            lag: None,
        };
        assert!(assemble(&req).is_err());
    }
}
