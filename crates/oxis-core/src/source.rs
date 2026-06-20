//! The [`DataSource`] trait — the contract a *service* module implements.
//!
//! This is the boundary for the **second module kind**: stateful, I/O-bound
//! providers (a market-data fetcher, a database). The trait lives in the core so
//! a consumer (e.g. the future portfolio module) can depend on *"something that
//! provides prices"* without depending on a concrete provider crate — classic
//! dependency inversion, the same pattern wealthfolio uses.
//!
//! ## Why `async`, and why the core stays runtime-free
//!
//! Fetching data is inherently asynchronous, so the trait uses a native
//! `async fn`. Defining an `async fn` in a trait does **not** pull in an async
//! runtime — a runtime (e.g. `tokio`) is only needed by the crate that *drives*
//! the future to completion (the data module and the app edge). The core
//! therefore declares the capability without depending on `tokio` or any HTTP
//! crate, preserving the lean-core guardrail.

use crate::error::OxisError;
use crate::series::{DateRange, Ohlcv};

/// A provider of market data, implemented by *service* modules such as
/// `oxis-data`.
///
/// Implementations own their own HTTP client, cache, and rate-limiting; all of
/// that I/O is confined to the implementing crate. Results come back as the
/// typed interchange records from [`crate::series`], never a provider-specific
/// shape.
// We intentionally do not require `Send` on the returned future: the core does
// not impose a threading model on implementors. Callers needing `Send` can add
// the bound at the call site or via a wrapper.
#[allow(async_fn_in_trait)]
pub trait DataSource {
    /// A stable identifier for this source (e.g. `"yahoo"`), for logging/selection.
    fn name(&self) -> &str;

    /// Fetch OHLCV bars for `symbol` over the inclusive `range`.
    async fn ohlcv(&self, symbol: &str, range: DateRange) -> Result<Vec<Ohlcv>, OxisError>;
}
