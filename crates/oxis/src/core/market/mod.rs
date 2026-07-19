//! Market-data inputs to pricing.
//!
//! For Phase 1 this is the flat [`MarketData`] snapshot the option engines need.
//! Term structures / yield curves (Ring 2) will join this module without changing
//! the pricing contract.

mod market_data;

pub use market_data::MarketData;
