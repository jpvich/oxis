//! # oxis-portfolio — portfolio & risk analytics (Ring 3, reserved)
//!
//! The first module designed to **interact with other modules**: it consumes the
//! [`DataSource`](oxis_core::source::DataSource) contract for prices and the
//! pricing/stats modules' public result types to compute holdings, valuation,
//! performance (TWR / MWR), and allocation — without importing any module's
//! internals.
//!
//! Design notes carried from the reference study (wealthfolio): trait-first
//! services, daily snapshot caching, lot-tracked cost basis, and decimal-precise
//! money for accounting (distinct from pricing's `f64`).
//!
//! **Status: reserved skeleton.** No analytics implemented yet.

#![forbid(unsafe_code)]

#[cfg(test)]
mod tests {
    use oxis_core::{Currency, Money};

    /// Proves the `module → core` dependency direction compiles.
    #[test]
    fn builds_against_core_money() {
        let m = Money {
            amount: 1_000.0,
            currency: Currency::new("usd").unwrap(),
        };
        assert_eq!(m.currency.code(), "USD");
    }
}
