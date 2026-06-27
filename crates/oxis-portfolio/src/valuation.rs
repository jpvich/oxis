//! Mark-to-market valuation of holdings.
//!
//! Given holdings and a price per holding (positionally aligned — the app edge
//! has already joined symbol → price), compute per-holding market value,
//! unrealized P&L, and portfolio weight, plus the aggregate totals.

use crate::holdings::{Holding, cost_basis, total_quantity};
use oxis_core::OxisError;
use serde::Serialize;

/// The valuation of one holding at a given price.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HoldingValuation {
    /// Instrument identifier.
    pub symbol: String,
    /// Total quantity held.
    pub quantity: f64,
    /// Average unit cost (`cost_basis / quantity`); `None` if quantity is zero.
    pub average_cost: Option<f64>,
    /// Mark price used for valuation.
    pub price: f64,
    /// Total cost basis.
    pub cost_basis: f64,
    /// Market value `quantity · price`.
    pub market_value: f64,
    /// Unrealized P&L `market_value − cost_basis`.
    pub unrealized_pnl: f64,
    /// Unrealized P&L as a fraction of cost basis; `None` if cost basis is zero.
    pub unrealized_pnl_pct: Option<f64>,
    /// Portfolio weight `market_value / total_market_value`.
    pub weight: f64,
}

/// Aggregate valuation across all holdings, carrying the per-holding breakdown.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HoldingsValuation {
    /// Number of holdings.
    pub n_holdings: usize,
    /// Sum of cost bases.
    pub total_cost_basis: f64,
    /// Sum of market values.
    pub total_market_value: f64,
    /// Sum of unrealized P&L.
    pub total_unrealized_pnl: f64,
    /// Total unrealized P&L as a fraction of total cost basis; `None` if zero.
    pub total_unrealized_pnl_pct: Option<f64>,
    /// Per-holding valuations.
    pub holdings: Vec<HoldingValuation>,
}

/// Value `holdings` at `prices` (aligned positionally).
///
/// # Errors
/// [`OxisError::InvalidInput`] if `holdings` is empty or `prices` has a different
/// length. When the total market value is zero, weights are `0.0` (not `NaN`).
pub fn value_holdings(
    holdings: &[Holding],
    prices: &[f64],
) -> Result<HoldingsValuation, OxisError> {
    if holdings.is_empty() {
        return Err(OxisError::invalid_input("value_holdings: no holdings"));
    }
    if holdings.len() != prices.len() {
        return Err(OxisError::invalid_input(
            "value_holdings: prices must align with holdings (one price each)",
        ));
    }

    // First pass: market values + total (needed for weights).
    let mut market_values = Vec::with_capacity(holdings.len());
    for (h, &price) in holdings.iter().zip(prices.iter()) {
        market_values.push(total_quantity(h) * price);
    }
    let total_mv: f64 = market_values.iter().sum();

    let mut out = Vec::with_capacity(holdings.len());
    let mut total_cost = 0.0;
    for ((h, &price), &mv) in holdings.iter().zip(prices.iter()).zip(market_values.iter()) {
        let qty = total_quantity(h);
        let basis = cost_basis(h);
        total_cost += basis;
        let pnl = mv - basis;
        out.push(HoldingValuation {
            symbol: h.symbol.clone(),
            quantity: qty,
            average_cost: if qty == 0.0 { None } else { Some(basis / qty) },
            price,
            cost_basis: basis,
            market_value: mv,
            unrealized_pnl: pnl,
            unrealized_pnl_pct: if basis == 0.0 {
                None
            } else {
                Some(pnl / basis)
            },
            weight: if total_mv == 0.0 { 0.0 } else { mv / total_mv },
        });
    }

    let total_pnl = total_mv - total_cost;
    Ok(HoldingsValuation {
        n_holdings: holdings.len(),
        total_cost_basis: total_cost,
        total_market_value: total_mv,
        total_unrealized_pnl: total_pnl,
        total_unrealized_pnl_pct: if total_cost == 0.0 {
            None
        } else {
            Some(total_pnl / total_cost)
        },
        holdings: out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holdings::Holding;

    const TOL: f64 = 1e-12;

    #[test]
    fn values_and_weights() {
        let holdings = vec![
            Holding::single("AAPL", 10.0, 150.0),
            Holding::single("MSFT", 5.0, 300.0),
        ];
        let v = value_holdings(&holdings, &[175.0, 320.0]).unwrap();
        // MV: 1750 + 1600 = 3350. PnL: (1750-1500)+(1600-1500)=350.
        assert!((v.total_market_value - 3350.0).abs() < TOL);
        assert!((v.total_unrealized_pnl - 350.0).abs() < TOL);
        assert!((v.holdings[0].weight - 1750.0 / 3350.0).abs() < TOL);
        let wsum: f64 = v.holdings.iter().map(|h| h.weight).sum();
        assert!((wsum - 1.0).abs() < TOL);
    }

    #[test]
    fn invalid_inputs_error_not_panic() {
        assert!(value_holdings(&[], &[]).is_err());
        assert!(value_holdings(&[Holding::single("X", 1.0, 1.0)], &[1.0, 2.0]).is_err());
    }
}
