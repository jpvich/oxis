//! [`CurveQuery`] — the renderable result of querying a [`YieldCurve`].
//!
//! Carries the curve's construction label, the query time, the discount factor
//! and zero rate, and — when a forward leg was requested — the forward end and
//! the forward rate. Derives `Serialize` and implements [`Tabular`] so the core's
//! output layer renders it as human / JSON / TSV; the module never formats by
//! hand. Absent forward fields render as empty text / JSON `null`, the same way
//! `PriceResult`'s optional standard error does.

use crate::curve::YieldCurve;
use oxis_core::{Cell, Column, OxisError, Tabular};
use serde::Serialize;

/// The outcome of querying a curve at one time (optionally with a forward leg).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CurveQuery {
    /// How the curve was built (`"flat"`, `"linear"`, `"log-linear"`,
    /// `"natural-cubic"`).
    pub interpolation: &'static str,
    /// Query time in years.
    pub t: f64,
    /// Discount factor `P(t)`.
    pub discount: f64,
    /// Continuously-compounded zero rate `z(t)`.
    pub zero_rate: f64,
    /// Forward leg end time, if a forward rate was requested.
    pub forward_to: Option<f64>,
    /// Continuously-compounded forward rate over `[t, forward_to]`, if requested.
    pub forward_rate: Option<f64>,
}

impl YieldCurve {
    /// Query the curve at `t`, optionally also computing the forward rate from
    /// `t` to `forward_to`, and package it as a renderable [`CurveQuery`].
    ///
    /// # Errors
    /// Propagates [`OxisError`] from the underlying queries (e.g. `t` out of
    /// range, or `forward_to <= t`).
    pub fn query(&self, t: f64, forward_to: Option<f64>) -> Result<CurveQuery, OxisError> {
        let discount = self.discount(t)?;
        let zero_rate = self.zero_rate(t)?;
        let forward_rate = match forward_to {
            Some(t2) => Some(self.forward_rate(t, t2)?),
            None => None,
        };
        Ok(CurveQuery {
            interpolation: self.interpolation_label(),
            t,
            discount,
            zero_rate,
            forward_to,
            forward_rate,
        })
    }
}

impl Tabular for CurveQuery {
    fn columns(&self) -> Vec<Column> {
        vec![
            Column::new("interpolation"),
            Column::new("t"),
            Column::new("discount"),
            Column::new("zero_rate"),
            Column::new("forward_to"),
            Column::new("forward_rate"),
        ]
    }

    fn cells(&self) -> Vec<Cell> {
        vec![
            Cell::str(self.interpolation),
            Cell::F64(self.t),
            Cell::F64(self.discount),
            Cell::F64(self.zero_rate),
            self.forward_to.into(),
            self.forward_rate.into(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Interpolation;

    #[test]
    fn query_without_forward_leaves_forward_fields_null() {
        let c = YieldCurve::flat(0.03).unwrap();
        let q = c.query(2.0, None).unwrap();
        assert_eq!(q.interpolation, "flat");
        assert!(q.forward_to.is_none());
        assert!(q.forward_rate.is_none());
        // Tabular contract: equal-length columns and cells.
        assert_eq!(q.columns().len(), q.cells().len());
    }

    #[test]
    fn query_with_forward_populates_and_renders() {
        let times = [1.0, 2.0, 3.0];
        let rates = [0.02, 0.025, 0.03];
        let c = YieldCurve::from_zero_rates(&times, &rates, Interpolation::Linear).unwrap();
        let q = c.query(1.5, Some(2.5)).unwrap();
        assert_eq!(q.interpolation, "linear");
        assert!(q.forward_rate.is_some());

        let json = oxis_core::output::render(&q, oxis_core::OutputFormat::Json);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["interpolation"], "linear");
        assert!(parsed["discount"].as_f64().unwrap() < 1.0);
        assert!(parsed["forward_rate"].as_f64().is_some());
    }
}
