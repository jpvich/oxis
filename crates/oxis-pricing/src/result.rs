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

/// The outcome of pricing an exotic option (barrier / lookback / Asian).
///
/// One self-describing record across the three families: dimensions that a given
/// exotic doesn't use (barrier level, average/strike type) and the Monte Carlo
/// standard error (closed-form prices have none) render as JSON `null` — the same
/// `Option` → `Cell::Null` convention `PriceResult.standard_error` uses.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExoticResult {
    /// Exotic family + variant (e.g. `"barrier"`, `"asian"`, `"lookback"`).
    pub model: &'static str,
    /// Call or put.
    pub option_type: OptionType,
    /// Spot price input.
    pub spot: f64,
    /// Strike price input (ignored by floating-strike lookbacks).
    pub strike: f64,
    /// Continuously compounded risk-free rate input.
    pub rate: f64,
    /// Volatility input.
    pub volatility: f64,
    /// Time to expiry (years) input.
    pub time: f64,
    /// Continuously compounded dividend yield input.
    pub dividend_yield: f64,
    /// Barrier level (barrier options only).
    pub barrier: Option<f64>,
    /// Barrier / averaging / strike variant label (e.g. `"down-out"`,
    /// `"arithmetic"`, `"floating"`).
    pub variant: Option<&'static str>,
    /// The computed present value.
    pub price: f64,
    /// Monte Carlo standard error, if applicable (arithmetic-average Asian).
    pub standard_error: Option<f64>,
}

impl Tabular for ExoticResult {
    fn columns(&self) -> Vec<Column> {
        vec![
            Column::new("model"),
            Column::new("option_type"),
            Column::new("spot"),
            Column::new("strike"),
            Column::new("rate"),
            Column::new("volatility"),
            Column::new("time"),
            Column::new("dividend_yield"),
            Column::new("barrier"),
            Column::new("variant"),
            Column::new("price"),
            Column::new("standard_error"),
        ]
    }

    fn cells(&self) -> Vec<Cell> {
        vec![
            Cell::str(self.model),
            Cell::str(self.option_type.as_str()),
            Cell::F64(self.spot),
            Cell::F64(self.strike),
            Cell::F64(self.rate),
            Cell::F64(self.volatility),
            Cell::F64(self.time),
            Cell::F64(self.dividend_yield),
            self.barrier.into(),
            match self.variant {
                Some(v) => Cell::str(v),
                None => Cell::Null,
            },
            Cell::F64(self.price),
            self.standard_error.into(),
        ]
    }
}
