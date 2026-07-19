//! [`GreeksResult`] — the renderable result of a Greeks computation.
//!
//! Carries the inputs (so output is self-describing) and the five sensitivities,
//! derives `Serialize`, and implements [`Tabular`] so the core output layer
//! renders it as human / JSON / TSV — the module never formats by hand. Mirrors
//! `crate::pricing::PriceResult`'s shape.

use crate::core::{Cell, Column, EuropeanOption, MarketData, OptionType, Tabular};
use serde::Serialize;

use crate::greeks::analytic::Greeks;

/// The Greeks of a single option, with the inputs that produced them.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct GreeksResult {
    /// How the Greeks were computed (e.g. `"analytic"`).
    pub method: &'static str,
    /// Call or put.
    pub option_type: OptionType,
    /// Spot price input.
    pub spot: f64,
    /// Strike price input.
    pub strike: f64,
    /// Risk-free rate input.
    pub rate: f64,
    /// Volatility input.
    pub volatility: f64,
    /// Time to expiry (years) input.
    pub time: f64,
    /// Dividend yield input.
    pub dividend_yield: f64,
    /// `∂V/∂S`.
    pub delta: f64,
    /// `∂²V/∂S²`.
    pub gamma: f64,
    /// `∂V/∂σ` (per unit vol).
    pub vega: f64,
    /// `∂V/∂t` (per year).
    pub theta: f64,
    /// `∂V/∂r` (per unit rate).
    pub rho: f64,
}

impl GreeksResult {
    /// Assemble a result from the inputs and computed [`Greeks`].
    pub fn new(
        method: &'static str,
        option: &EuropeanOption,
        market: &MarketData,
        greeks: &Greeks,
    ) -> Self {
        Self {
            method,
            option_type: option.option_type,
            spot: market.spot,
            strike: option.strike,
            rate: market.rate,
            volatility: market.volatility,
            time: option.expiry_years,
            dividend_yield: market.dividend_yield,
            delta: greeks.delta,
            gamma: greeks.gamma,
            vega: greeks.vega,
            theta: greeks.theta,
            rho: greeks.rho,
        }
    }
}

impl Tabular for GreeksResult {
    fn columns(&self) -> Vec<Column> {
        vec![
            Column::new("method"),
            Column::new("option_type"),
            Column::new("spot"),
            Column::new("strike"),
            Column::new("rate"),
            Column::new("volatility"),
            Column::new("time"),
            Column::new("dividend_yield"),
            Column::new("delta"),
            Column::new("gamma"),
            Column::new("vega"),
            Column::new("theta"),
            Column::new("rho"),
        ]
    }

    fn cells(&self) -> Vec<Cell> {
        vec![
            Cell::str(self.method),
            Cell::str(self.option_type.as_str()),
            Cell::F64(self.spot),
            Cell::F64(self.strike),
            Cell::F64(self.rate),
            Cell::F64(self.volatility),
            Cell::F64(self.time),
            Cell::F64(self.dividend_yield),
            Cell::F64(self.delta),
            Cell::F64(self.gamma),
            Cell::F64(self.vega),
            Cell::F64(self.theta),
            Cell::F64(self.rho),
        ]
    }
}
