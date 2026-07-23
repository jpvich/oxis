//! # oxis::bonds — fixed-income bonds (Ring 2)
//!
//! A **Kind A** module (see [`crate::core::contract`]): a pure, I/O-free compute
//! core whose every result is validated against QuantLib. The central type is
//! [`FixedRateBond`] — built from a regular schedule or explicit cashflows, then
//! priced from a flat yield or an [`crate::curves::YieldCurve`], with
//! yield-to-maturity, duration (Macaulay & modified), and convexity.
//!
//! This is the first OXIS module that depends on another module
//! (`oxis::curves`, for curve discounting), composing through its public API only.
//! Yield-based analytics compound at the coupon frequency (the market / QuantLib
//! convention); curve discounting is continuous. Prices are quoted per face, with
//! `clean = dirty − accrued`.
//!
//! Instrument bootstrapping (building a curve from bond/swap quotes) is a separate
//! concern and lands in a later milestone.

mod bond;
mod result;

pub use bond::{
    Cashflow, FixedRateBond, convexity, dirty_price_with_curve, dirty_price_with_yield,
    macaulay_duration, modified_duration, yield_from_dirty,
};
pub use result::BondResult;
