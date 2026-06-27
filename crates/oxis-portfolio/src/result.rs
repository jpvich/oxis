//! Renderable result types for the portfolio surfaces.
//!
//! Each derives `Serialize` and implements [`Tabular`] as a **single record**
//! (equal-length `columns`/`cells`, `Option` → `Cell::Null`). Weight vectors —
//! which have no list cell in the output layer — are carried in serde for the
//! JSON / Python surfaces and rendered as a comma-joined `weights` string for the
//! human / TSV table. [`HoldingValuation`] also implements `Tabular` so the CLI
//! can loop-render the per-holding table beneath the aggregate summary.

use crate::risk::{
    annualized_volatility, covariance_matrix, portfolio_returns, portfolio_variance,
};
use crate::valuation::{HoldingValuation, HoldingsValuation};
use oxis_core::{Cell, Column, OxisError, Tabular};
use oxis_stats::{historical_var, parametric_var};
use serde::Serialize;

fn weights_csv(weights: &[f64]) -> String {
    weights
        .iter()
        .map(|w| w.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

impl Tabular for HoldingValuation {
    fn columns(&self) -> Vec<Column> {
        vec![
            Column::new("symbol"),
            Column::new("quantity"),
            Column::new("average_cost"),
            Column::new("price"),
            Column::new("cost_basis"),
            Column::new("market_value"),
            Column::new("unrealized_pnl"),
            Column::new("unrealized_pnl_pct"),
            Column::new("weight"),
        ]
    }

    fn cells(&self) -> Vec<Cell> {
        vec![
            Cell::str(self.symbol.clone()),
            Cell::F64(self.quantity),
            self.average_cost.into(),
            Cell::F64(self.price),
            Cell::F64(self.cost_basis),
            Cell::F64(self.market_value),
            Cell::F64(self.unrealized_pnl),
            self.unrealized_pnl_pct.into(),
            Cell::F64(self.weight),
        ]
    }
}

impl Tabular for HoldingsValuation {
    fn columns(&self) -> Vec<Column> {
        vec![
            Column::new("n_holdings"),
            Column::new("total_cost_basis"),
            Column::new("total_market_value"),
            Column::new("total_unrealized_pnl"),
            Column::new("total_unrealized_pnl_pct"),
        ]
    }

    fn cells(&self) -> Vec<Cell> {
        vec![
            Cell::Int(self.n_holdings as i64),
            Cell::F64(self.total_cost_basis),
            Cell::F64(self.total_market_value),
            Cell::F64(self.total_unrealized_pnl),
            self.total_unrealized_pnl_pct.into(),
        ]
    }
}

/// Performance returns (TWR and/or MWR) over a number of periods.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerformanceReport {
    /// Time-weighted return, if requested.
    pub twr: Option<f64>,
    /// Money-weighted return (IRR), if requested.
    pub mwr: Option<f64>,
}

impl Tabular for PerformanceReport {
    fn columns(&self) -> Vec<Column> {
        vec![Column::new("twr"), Column::new("mwr")]
    }
    fn cells(&self) -> Vec<Cell> {
        vec![self.twr.into(), self.mwr.into()]
    }
}

/// Allocation weights for a set of positions.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AllocationReport {
    /// Number of positions.
    pub n_assets: usize,
    /// Weights, one per position.
    pub weights: Vec<f64>,
}

impl Tabular for AllocationReport {
    fn columns(&self) -> Vec<Column> {
        vec![Column::new("n_assets"), Column::new("weights")]
    }
    fn cells(&self) -> Vec<Cell> {
        vec![
            Cell::Int(self.n_assets as i64),
            Cell::str(weights_csv(&self.weights)),
        ]
    }
}

/// Portfolio risk aggregation result.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RiskReport {
    /// Portfolio variance `wᵀΣw`.
    pub variance: f64,
    /// Portfolio volatility (per period).
    pub volatility: f64,
    /// Annualized volatility `vol·√ppy`.
    pub annualized_volatility: f64,
    /// Historical VaR of the portfolio return series (positive loss).
    pub historical_var: f64,
    /// Parametric Gaussian VaR of the portfolio return series (positive loss).
    pub parametric_var: f64,
    /// Confidence level used for VaR.
    pub confidence: f64,
    /// Periods-per-year annualization factor.
    pub periods_per_year: f64,
}

impl Tabular for RiskReport {
    fn columns(&self) -> Vec<Column> {
        vec![
            Column::new("variance"),
            Column::new("volatility"),
            Column::new("annualized_volatility"),
            Column::new("historical_var"),
            Column::new("parametric_var"),
            Column::new("confidence"),
            Column::new("periods_per_year"),
        ]
    }
    fn cells(&self) -> Vec<Cell> {
        vec![
            Cell::F64(self.variance),
            Cell::F64(self.volatility),
            Cell::F64(self.annualized_volatility),
            Cell::F64(self.historical_var),
            Cell::F64(self.parametric_var),
            Cell::F64(self.confidence),
            Cell::F64(self.periods_per_year),
        ]
    }
}

/// A Markowitz optimization result (weights + frontier statistics).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OptimizationReport {
    /// Which portfolio (`min-variance`, `tangency`, `frontier`).
    pub flavor: &'static str,
    /// Expected return `wᵀμ`.
    pub expected_return: f64,
    /// Variance `wᵀΣw`.
    pub variance: f64,
    /// Volatility `√(wᵀΣw)`.
    pub volatility: f64,
    /// Optimal weights.
    pub weights: Vec<f64>,
}

impl Tabular for OptimizationReport {
    fn columns(&self) -> Vec<Column> {
        vec![
            Column::new("flavor"),
            Column::new("expected_return"),
            Column::new("variance"),
            Column::new("volatility"),
            Column::new("weights"),
        ]
    }
    fn cells(&self) -> Vec<Cell> {
        vec![
            Cell::str(self.flavor),
            Cell::F64(self.expected_return),
            Cell::F64(self.variance),
            Cell::F64(self.volatility),
            Cell::str(weights_csv(&self.weights)),
        ]
    }
}

/// Build a [`RiskReport`] from N aligned asset-return series and weights,
/// reusing `oxis-stats` for the single-series VaR metrics.
///
/// # Errors
/// [`OxisError::InvalidInput`] on empty / ragged input, a weight mismatch, bad
/// `ppy`/`confidence`.
pub fn portfolio_risk(
    asset_returns: &[Vec<f64>],
    weights: &[f64],
    periods_per_year: f64,
    confidence: f64,
) -> Result<RiskReport, OxisError> {
    let cov = covariance_matrix(asset_returns)?;
    let variance = portfolio_variance(&cov, weights)?;
    let port = portfolio_returns(asset_returns, weights)?;
    Ok(RiskReport {
        variance,
        volatility: variance.sqrt(),
        annualized_volatility: annualized_volatility(&cov, weights, periods_per_year)?,
        historical_var: historical_var(&port, confidence)?,
        parametric_var: parametric_var(&port, confidence)?,
        confidence,
        periods_per_year,
    })
}

/// Assemble an [`OptimizationReport`] from chosen weights, the mean vector, and
/// the covariance matrix.
///
/// # Errors
/// [`OxisError::InvalidInput`] on a dimension mismatch.
pub fn optimization_report(
    flavor: &'static str,
    weights: Vec<f64>,
    mean: &[f64],
    cov: &[Vec<f64>],
) -> Result<OptimizationReport, OxisError> {
    let (expected_return, variance, volatility) =
        crate::optimize::frontier_stats(&weights, mean, cov)?;
    Ok(OptimizationReport {
        flavor,
        expected_return,
        variance,
        volatility,
        weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holdings::Holding;
    use crate::valuation::value_holdings;

    #[test]
    fn result_structs_align_columns_and_cells() {
        let v = value_holdings(
            &[
                Holding::single("AAPL", 10.0, 150.0),
                Holding::single("MSFT", 5.0, 300.0),
            ],
            &[175.0, 320.0],
        )
        .unwrap();
        assert_eq!(v.columns().len(), v.cells().len());
        assert_eq!(v.holdings[0].columns().len(), v.holdings[0].cells().len());

        let perf = PerformanceReport {
            twr: Some(0.1),
            mwr: None,
        };
        assert_eq!(perf.columns().len(), perf.cells().len());

        let alloc = AllocationReport {
            n_assets: 2,
            weights: vec![0.6, 0.4],
        };
        assert_eq!(alloc.columns().len(), alloc.cells().len());
    }

    #[test]
    fn portfolio_risk_assembles() {
        let returns = vec![
            vec![0.01, -0.02, 0.03, -0.01, 0.02],
            vec![0.02, 0.00, -0.01, 0.015, -0.005],
        ];
        let r = portfolio_risk(&returns, &[0.5, 0.5], 252.0, 0.95).unwrap();
        assert_eq!(r.columns().len(), r.cells().len());
        assert!(r.volatility >= 0.0);
        assert!(r.historical_var.is_finite());
    }
}
