//! # oxis::portfolio — portfolio & risk analytics (Ring 3)
//!
//! The first OXIS module designed to **interact with other modules**: it consumes
//! `oxis::stats` (covariance, VaR) and the core's typed records to compute
//! holdings valuation, performance (TWR / MWR), allocation, risk aggregation, and
//! Markowitz mean-variance optimization — without importing any module's
//! internals. It is a pure, sync **Kind A** module: it operates on already-fetched
//! price / return records passed in, not a live data source (the async
//! [`DataSource`](crate::core::DataSource) wiring lands with `oxis::data`).
//!
//! ## Money is `f64` (a deliberate, documented deviation)
//!
//! The project spec earmarks "decimal-precise money" for the portfolio ring. This
//! milestone uses **`f64` throughout** on purpose: portfolio analytics are ratios
//! and linear algebra validated against a numpy (float) oracle, and a decimal type
//! cannot do `sqrt` / `exp` / matrix solves cleanly — it would fight the oracle
//! without improving any analytic result. Decimal-precise accounting belongs to a
//! future transaction-ledger module (exact cash/cost reconciliation), which can
//! add it locally without touching these analytics.
//!
//! ## Conventions
//!
//! - **TWR** chains sub-period returns `rᵢ = Vᵢ/(Vᵢ₋₁+flowᵢ)−1` geometrically.
//! - **MWR/IRR** is the rate zeroing `Σ cf/(1+r)^t`, `t` = Act/365 from the first
//!   cash-flow date; invested amounts negative, received positive.
//! - **Covariance** is population (÷n), matching `numpy.cov(bias=True)`.
//! - **Markowitz** weights use the budget-constrained closed form via `Σ⁻¹1` and
//!   `Σ⁻¹μ` (solved, not inverted; shorting allowed — no long-only constraint).
//! - **Portfolio VaR** delegates to `oxis::stats` on the portfolio return series.

mod allocation;
mod holdings;
mod optimize;
mod performance;
mod result;
mod risk;
mod valuation;

pub use allocation::weights;
pub use holdings::{Holding, Lot, average_unit_cost, cost_basis, total_quantity};
pub use optimize::{
    efficient_frontier_point, frontier_stats, min_variance_weights, tangency_weights,
};
pub use performance::{mwr, twr};
pub use result::{
    AllocationReport, OptimizationReport, PerformanceReport, RiskReport, optimization_report,
    portfolio_risk,
};
pub use risk::{
    annualized_volatility, covariance_matrix, portfolio_returns, portfolio_variance,
    portfolio_volatility,
};
pub use valuation::{HoldingValuation, HoldingsValuation, value_holdings};

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end: value a portfolio, run TWR, and optimize — the module→module
    /// (`oxis::stats`) and module→core (`oxis::core` linalg) wiring all compile.
    #[test]
    fn end_to_end_smoke() {
        let holdings = vec![
            Holding::single("A", 10.0, 100.0),
            Holding::single("B", 5.0, 200.0),
        ];
        let v = value_holdings(&holdings, &[120.0, 210.0]).unwrap();
        assert_eq!(v.n_holdings, 2);

        assert!(twr(&[100.0, 110.0, 121.0], &[0.0, 0.0]).unwrap() > 0.0);

        let cov = covariance_matrix(&[
            vec![0.01, -0.02, 0.03, -0.01],
            vec![0.02, 0.00, -0.01, 0.015],
        ])
        .unwrap();
        let w = min_variance_weights(&cov).unwrap();
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }
}
