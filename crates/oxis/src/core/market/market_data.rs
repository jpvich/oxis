//! [`MarketData`] — the flat market snapshot the option engines price against.

use serde::{Deserialize, Serialize};

/// A market-data snapshot for pricing a single underlying.
///
/// `rate` and `dividend_yield` are **continuously compounded**, matching the
/// Black-Scholes-Merton convention the engines assume.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MarketData {
    /// Current spot price of the underlying.
    pub spot: f64,
    /// Continuously compounded risk-free rate.
    pub rate: f64,
    /// Volatility (annualized standard deviation of returns).
    pub volatility: f64,
    /// Continuously compounded dividend yield.
    pub dividend_yield: f64,
}

impl MarketData {
    /// Construct a snapshot.
    pub fn new(spot: f64, rate: f64, volatility: f64, dividend_yield: f64) -> Self {
        Self {
            spot,
            rate,
            volatility,
            dividend_yield,
        }
    }
}
