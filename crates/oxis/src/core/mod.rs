//! # oxis::core
//!
//! The stable core of OXIS — the platform every OXIS module builds against.
//!
//! OXIS is a **stable core + a growing catalog of modules**. This crate is the
//! core: the fixed contracts shared across modules. The dependency direction is
//! one-way — **module → core only**; the core never depends on a module.
//!
//! ## What lives here
//!
//! - [`types`] — financial types (options, money, dates/day-count).
//! - [`market`] — market-data inputs ([`MarketData`](market::MarketData)).
//! - [`series`] — the typed time-series interchange records every module speaks
//!   ([`Ohlcv`](series::Ohlcv), [`TimeSeries`](series::TimeSeries)).
//! - [`output`] — the [`Tabular`](output::Tabular) trait + human/JSON/TSV renderers.
//!   Modules never format output by hand.
//! - [`error`] — [`OxisError`](error::OxisError), the library error type.
//! - [`context`] — [`RunContext`](context::RunContext), shared run-time config.
//! - [`source`] — the [`DataSource`](source::DataSource) trait: the contract a
//!   *service* module (e.g. market-data) implements. The trait lives here so
//!   modules can depend on the *capability* without depending on a concrete
//!   provider crate.
//! - [`contract`] — documentation of the **two module kinds** every contributor
//!   builds against.
//!
//! ## The lean-core guardrail
//!
//! The core is deliberately small and runtime-agnostic: **no Polars/Arrow, no
//! async runtime, no HTTP**. Heavy columnar machinery (Polars) is adopted only
//! feature-gated *inside* the stats/data modules; I/O lives only inside *service*
//! modules. See [`contract`] and `docs/architecture.md`.
//!
//! This crate is under active development. See <https://github.com/jpvich/oxis>.

pub mod context;
pub mod contract;
pub mod error;
pub mod market;
pub mod math;
pub mod output;
pub mod series;
pub mod source;
pub mod types;

// Re-export the most-used items at the crate root for ergonomic `use crate::core::*`.
pub use context::{OutputFormat, RunContext};
pub use error::OxisError;
pub use market::MarketData;
pub use math::{
    NaturalCubicSpline, brent, invert, linear_interpolate, mean_and_se, newton, normal_cdf,
    normal_pdf, path_seed, poly_least_squares, sample_mean_var, solve_linear_system, splitmix64,
};
pub use output::{Cell, Column, Tabular};
pub use series::{DateRange, Ohlcv, TimeSeries};
pub use source::DataSource;
pub use types::{
    AmericanOption, Currency, Date, DayCount, EuropeanOption, ExerciseStyle, Money, OptionType,
};

/// A convenient module result alias — `Result<T, OxisError>`.
pub type Result<T> = core::result::Result<T, OxisError>;
