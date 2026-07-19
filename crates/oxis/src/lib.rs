//! # OXIS — Open eXtensible Instruments & Statistics
//!
//! A validated quantitative-finance library, CLI, and REPL, shipped as a
//! **single crate**. A Rust user adds one dependency and reaches every domain
//! through a short module path:
//!
//! ```toml
//! # Cargo.toml — the whole library:
//! oxis = "0.1"
//! # …or only the modules you need:
//! oxis = { version = "0.1", default-features = false, features = ["pricing", "ml"] }
//! ```
//!
//! ```no_run
//! use oxis::core::{EuropeanOption, MarketData, OptionType};
//! use oxis::pricing::black_scholes;
//!
//! let market = MarketData::new(100.0, 0.05, 0.2, 0.0);
//! let option = EuropeanOption { strike: 100.0, expiry_years: 1.0, option_type: OptionType::Call };
//! let price = black_scholes(&option, &market).unwrap();
//! ```
//!
//! ## Module map
//!
//! Each domain is an internal module, gated behind a feature of the same name so
//! a library user compiles only what they enable (the `core` module is always
//! present; `full` turns them all on; the default `cli` feature implies `full`).
//!
//! | Path | Feature | Domain |
//! |------|---------|--------|
//! | [`core`] | always | shared types, output, errors |
//! | [`pricing`] | `pricing` | Black-Scholes, binomial, Monte Carlo, exotics |
//! | [`greeks`] | `greeks` | analytic and finite-difference sensitivities |
//! | [`curves`] | `curves` | yield curves and term structures |
//! | [`bonds`] | `bonds` | fixed-rate bond pricing and risk |
//! | [`stochastic`] | `stochastic` | stochastic-process simulation |
//! | [`stats`] | `stats` | descriptive, risk, and performance statistics |
//! | [`portfolio`] | `portfolio` | valuation, allocation, optimization |
//! | [`ml`] | `ml` | neural pricing (differential ML, Deep LSM, DOS) |

#![forbid(unsafe_code)]

/// Shared core: market types, option/exercise enums, output, and errors.
///
/// Always available — every other module is built on it.
pub mod core;

#[cfg(feature = "pricing")]
pub mod pricing;

#[cfg(feature = "greeks")]
pub mod greeks;

#[cfg(feature = "curves")]
pub mod curves;

#[cfg(feature = "bonds")]
pub mod bonds;

#[cfg(feature = "stochastic")]
pub mod stochastic;

#[cfg(feature = "stats")]
pub mod stats;

#[cfg(feature = "portfolio")]
pub mod portfolio;

#[cfg(feature = "ml")]
pub mod ml;

#[cfg(feature = "data")]
pub mod data;
