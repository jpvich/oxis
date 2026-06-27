//! [`BondResult`] — the renderable result of pricing/analysing a bond.
//!
//! Carries the bond's identity (face, coupon, frequency), the clean/dirty price
//! and accrued interest, and — when priced from a yield — the yield, durations,
//! and convexity. Derives `Serialize` and implements [`Tabular`] so the core's
//! output layer renders it as human / JSON / TSV. Fields not produced by the
//! chosen mode render as empty / JSON `null` (the same `Option` → `Cell::Null`
//! convention `PriceResult.standard_error` uses).

use oxis_core::{Cell, Column, Tabular};
use serde::Serialize;

/// The outcome of pricing / analysing a single bond.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BondResult {
    /// Face / notional.
    pub face: f64,
    /// Annual coupon rate.
    pub coupon_rate: f64,
    /// Coupon payments per year.
    pub frequency: u32,
    /// Clean price (`dirty − accrued`).
    pub clean_price: f64,
    /// Dirty (full) price.
    pub dirty_price: f64,
    /// Accrued interest at settlement.
    pub accrued: f64,
    /// Yield used / solved (compounded at `frequency`); `None` for curve pricing.
    pub bond_yield: Option<f64>,
    /// Macaulay duration, if a yield is available.
    pub macaulay_duration: Option<f64>,
    /// Modified duration, if a yield is available.
    pub modified_duration: Option<f64>,
    /// Convexity, if a yield is available.
    pub convexity: Option<f64>,
}

impl Tabular for BondResult {
    fn columns(&self) -> Vec<Column> {
        vec![
            Column::new("face"),
            Column::new("coupon_rate"),
            Column::new("frequency"),
            Column::new("clean_price"),
            Column::new("dirty_price"),
            Column::new("accrued"),
            Column::new("yield"),
            Column::new("macaulay_duration"),
            Column::new("modified_duration"),
            Column::new("convexity"),
        ]
    }

    fn cells(&self) -> Vec<Cell> {
        vec![
            Cell::F64(self.face),
            Cell::F64(self.coupon_rate),
            Cell::Int(self.frequency as i64),
            Cell::F64(self.clean_price),
            Cell::F64(self.dirty_price),
            Cell::F64(self.accrued),
            self.bond_yield.into(),
            self.macaulay_duration.into(),
            self.modified_duration.into(),
            self.convexity.into(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_through_output_layer_with_optional_nulls() {
        let r = BondResult {
            face: 100.0,
            coupon_rate: 0.05,
            frequency: 2,
            clean_price: 100.0,
            dirty_price: 100.0,
            accrued: 0.0,
            bond_yield: Some(0.05),
            macaulay_duration: Some(4.4),
            modified_duration: Some(4.3),
            convexity: None,
        };
        assert_eq!(r.columns().len(), r.cells().len());
        let json = oxis_core::output::render(&r, oxis_core::OutputFormat::Json);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["face"], 100.0);
        assert_eq!(parsed["frequency"], 2);
        assert!(parsed["convexity"].is_null());
        assert!((parsed["yield"].as_f64().unwrap() - 0.05).abs() < 1e-12);
    }
}
