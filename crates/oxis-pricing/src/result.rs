//! [`PriceResult`] — the renderable result of a pricing call.
//!
//! Carries the model, the inputs (so output is self-describing), the price, and
//! an optional standard error (populated by Monte Carlo, `None` for closed-form /
//! tree methods). Derives `Serialize` and implements [`Tabular`] so the core's
//! output layer renders it as human / JSON / TSV — the module never formats by
//! hand.

use oxis_core::{Cell, Column, ExerciseStyle, MarketData, OptionType, Tabular};
use serde::Serialize;

/// The outcome of pricing a single option.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PriceResult {
    /// Model name (e.g. `"black-scholes"`).
    pub model: &'static str,
    /// Call or put.
    pub option_type: OptionType,
    /// European or American.
    pub exercise: ExerciseStyle,
    /// Spot price input.
    pub spot: f64,
    /// Strike price input.
    pub strike: f64,
    /// Continuously compounded risk-free rate input.
    pub rate: f64,
    /// Volatility input.
    pub volatility: f64,
    /// Time to expiry (years) input.
    pub time: f64,
    /// Continuously compounded dividend yield input.
    pub dividend_yield: f64,
    /// The computed present value.
    pub price: f64,
    /// Monte Carlo standard error, if applicable.
    pub standard_error: Option<f64>,
}

impl PriceResult {
    /// Build a result from inputs + a computed price (no standard error).
    pub fn new(
        model: &'static str,
        option_type: OptionType,
        exercise: ExerciseStyle,
        option: &oxis_core::EuropeanOption,
        market: &MarketData,
        price: f64,
    ) -> Self {
        Self {
            model,
            option_type,
            exercise,
            spot: market.spot,
            strike: option.strike,
            rate: market.rate,
            volatility: market.volatility,
            time: option.expiry_years,
            dividend_yield: market.dividend_yield,
            price,
            standard_error: None,
        }
    }

    /// Attach a Monte Carlo standard error to the result (builder style).
    ///
    /// Use for the stochastic engines (`monte_carlo_european`, `lsm_american`);
    /// closed-form and tree methods leave `standard_error` as `None`.
    pub fn with_standard_error(mut self, standard_error: f64) -> Self {
        self.standard_error = Some(standard_error);
        self
    }
}

impl Tabular for PriceResult {
    fn columns(&self) -> Vec<Column> {
        vec![
            Column::new("model"),
            Column::new("option_type"),
            Column::new("exercise"),
            Column::new("spot"),
            Column::new("strike"),
            Column::new("rate"),
            Column::new("volatility"),
            Column::new("time"),
            Column::new("dividend_yield"),
            Column::new("price"),
            Column::new("standard_error"),
        ]
    }

    fn cells(&self) -> Vec<Cell> {
        vec![
            Cell::str(self.model),
            Cell::str(self.option_type.as_str()),
            Cell::str(self.exercise.as_str()),
            Cell::F64(self.spot),
            Cell::F64(self.strike),
            Cell::F64(self.rate),
            Cell::F64(self.volatility),
            Cell::F64(self.time),
            Cell::F64(self.dividend_yield),
            Cell::F64(self.price),
            self.standard_error.into(),
        ]
    }
}
