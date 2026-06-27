//! # oxis-curves — yield curves & term structures (Ring 2)
//!
//! A **Kind A** module (see [`oxis_core::contract`]): a pure, I/O-free compute
//! core whose every query is validated against QuantLib. The central type is
//! [`YieldCurve`] — built once from a flat rate, zero rates, or discount factors,
//! then queried for discount factors, zero rates, and forward rates under three
//! interpolation schemes ([`Interpolation`]).
//!
//! All rates are **continuously compounded** and time is in years (`Act/365`),
//! matching the rest of OXIS and the QuantLib oracle. Curves do not extrapolate
//! beyond their pillars.
//!
//! This is the foundational Ring 2 module: bonds / fixed income and discounted
//! exotics build on it. Instrument bootstrapping (deposits/FRAs/swaps) lands with
//! the fixed-income module.

#![forbid(unsafe_code)]

mod curve;
mod result;

pub use curve::{Interpolation, YieldCurve};
pub use result::CurveQuery;
