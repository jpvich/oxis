//! # oxis::data — market-data providers (long-term ring, reserved)
//!
//! The reference **Kind B** (service) module. It will implement
//! [`crate::core::source::DataSource`] for concrete providers (Yahoo, etc.) with a
//! resolver chain, caching, rate-limiting, and circuit-breaking — all I/O
//! confined to this crate. Consumers (e.g. `oxis::portfolio`) depend on the
//! `DataSource` *trait* in the core, not on this crate, so the market-data ring
//! can land last without anything else waiting on it.
//!
//! Lessons applied (akshare / wealthfolio): typed records instead of loose
//! frames; a provider-agnostic trait instead of fragile per-source scrapers;
//! caching + rate-limiting from the start.
//!
//! **Status: reserved skeleton.** [`StubSource`] implements the boundary so the
//! contract compiles; it fetches nothing yet.

use crate::core::OxisError;
use crate::core::series::{DateRange, Ohlcv};
use crate::core::source::DataSource;

/// A placeholder [`DataSource`] proving the service boundary compiles. Every call
/// reports that the market-data ring is not implemented yet.
#[derive(Debug, Default, Clone, Copy)]
pub struct StubSource;

impl DataSource for StubSource {
    fn name(&self) -> &str {
        "stub"
    }

    async fn ohlcv(&self, _symbol: &str, _range: DateRange) -> Result<Vec<Ohlcv>, OxisError> {
        Err(OxisError::data_source(
            "oxis::data is not implemented yet (market-data ring is long-term)",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Proves the `DataSource` boundary contract is satisfiable from a module.
    #[test]
    fn stub_source_implements_the_boundary() {
        assert_eq!(StubSource.name(), "stub");
    }
}
