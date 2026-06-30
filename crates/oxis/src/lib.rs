//! # OXIS — Open eXtensible Instruments & Statistics
//!
//! The single front door to the library. OXIS is built as a modular workspace
//! — a stable `oxis-core` plus one crate per domain (pricing, greeks, curves,
//! bonds, stochastic processes, statistics, portfolio, ML). This crate
//! re-exports each of those modules under a short path so a Rust user depends
//! on **one** crate and reaches everything through it:
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
//! The internal `oxis-*` crates remain as dependencies but are an
//! implementation detail; application code should name only `oxis`.
//!
//! ## Module map
//!
//! | Path | Crate | Domain |
//! |------|-------|--------|
//! | [`core`] | `oxis-core` | shared types, output, errors (always available) |
//! | [`pricing`] | `oxis-pricing` | Black-Scholes, binomial, Monte Carlo, exotics |
//! | [`greeks`] | `oxis-greeks` | analytic and finite-difference sensitivities |
//! | [`curves`] | `oxis-curves` | yield curves and term structures |
//! | [`bonds`] | `oxis-bonds` | fixed-rate bond pricing and risk |
//! | [`stochastic`] | `oxis-stochastic` | stochastic-process simulation |
//! | [`stats`] | `oxis-stats` | descriptive, risk, and performance statistics |
//! | [`portfolio`] | `oxis-portfolio` | valuation, allocation, optimization |
//! | [`ml`] | `oxis-ml` | neural pricing (differential ML, Deep LSM, DOS) |

#![forbid(unsafe_code)]

/// Shared core: market types, option/exercise enums, output, and errors.
///
/// Always available — every other module is built on it.
pub use oxis_core as core;

#[cfg(feature = "pricing")]
pub use oxis_pricing as pricing;

#[cfg(feature = "greeks")]
pub use oxis_greeks as greeks;

#[cfg(feature = "curves")]
pub use oxis_curves as curves;

#[cfg(feature = "bonds")]
pub use oxis_bonds as bonds;

#[cfg(feature = "stochastic")]
pub use oxis_stochastic as stochastic;

#[cfg(feature = "stats")]
pub use oxis_stats as stats;

#[cfg(feature = "portfolio")]
pub use oxis_portfolio as portfolio;

#[cfg(feature = "ml")]
pub use oxis_ml as ml;
