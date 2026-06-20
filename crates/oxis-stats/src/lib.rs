//! # oxis-stats — statistics & time-series analytics (Ring 3, reserved)
//!
//! A **Kind A** (pure compute) module planned to provide descriptive statistics,
//! distributions, regression, returns, and risk metrics (Sharpe/Sortino, VaR/ES),
//! operating on the typed interchange records in [`oxis_core::series`].
//!
//! Other modules build on this one (e.g. portfolio risk consumes it). When it is
//! implemented, heavy columnar work may use Polars **behind a feature flag,
//! locally** — never leaking Polars types across the crate boundary.
//!
//! **Status: reserved skeleton.** The boundary exists so Ring 3 drops into a
//! pre-cut slot; no analytics are implemented yet.

#![forbid(unsafe_code)]

#[cfg(test)]
mod tests {
    use oxis_core::{Date, TimeSeries};

    /// Proves the `module → core` dependency direction compiles.
    #[test]
    fn builds_against_core_series() {
        let d = Date::new(2024, 1, 1).unwrap();
        let ts = TimeSeries::new(vec![d], vec![1.0]).unwrap();
        assert_eq!(ts.len(), 1);
    }
}
