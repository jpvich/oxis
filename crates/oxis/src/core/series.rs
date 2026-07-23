//! Typed time-series interchange records — the lingua franca every module speaks.
//!
//! These plain `serde` structs are the **boundary type** that crosses crate
//! lines: a stats or portfolio module asks a data source for `Vec<Ohlcv>` or a
//! [`TimeSeries`], never for a Polars `DataFrame`. Heavy columnar machinery
//! (Polars) is adopted only *inside* the stats/data modules, behind a feature
//! flag, with local conversions to/from these types — so the core stays lean and
//! modules stay decoupled (see `docs/architecture.md`).

use crate::core::error::OxisError;
use crate::core::types::Date;
use serde::{Deserialize, Serialize};

/// A single OHLCV bar (one period of price/volume data).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ohlcv {
    /// The bar's timestamp (period start or close, per the source's convention).
    pub ts: Date,
    /// Opening price.
    pub open: f64,
    /// Highest price.
    pub high: f64,
    /// Lowest price.
    pub low: f64,
    /// Closing price.
    pub close: f64,
    /// Traded volume.
    pub volume: f64,
}

/// A date-indexed series of values of type `T` (e.g. closing prices, returns).
///
/// The `index` and `values` are parallel and must be the same length;
/// [`new`](TimeSeries::new) enforces this.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeSeries<T> {
    /// The dates, one per value, in ascending order by convention.
    pub index: Vec<Date>,
    /// The values, aligned to `index`.
    pub values: Vec<T>,
}

impl<T> TimeSeries<T> {
    /// Construct a series, validating that `index` and `values` align.
    pub fn new(index: Vec<Date>, values: Vec<T>) -> Result<Self, OxisError> {
        if index.len() != values.len() {
            return Err(OxisError::invalid_input(format!(
                "time series index/values length mismatch: {} vs {}",
                index.len(),
                values.len()
            )));
        }
        Ok(Self { index, values })
    }

    /// Number of observations.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the series has no observations.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// An inclusive date range `[start, end]`, used by data-source queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DateRange {
    /// First date (inclusive).
    pub start: Date,
    /// Last date (inclusive).
    pub end: Date,
}

impl DateRange {
    /// Construct a range, requiring `start <= end`.
    pub fn new(start: Date, end: Date) -> Result<Self, OxisError> {
        if start > end {
            return Err(OxisError::invalid_input("date range start after end"));
        }
        Ok(Self { start, end })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_series_rejects_mismatched_lengths() {
        let d = Date::new(2024, 1, 1).unwrap();
        assert!(TimeSeries::new(vec![d], vec![1.0, 2.0]).is_err());
        assert!(TimeSeries::new(vec![d, d], vec![1.0, 2.0]).is_ok());
    }

    #[test]
    fn date_range_requires_order() {
        let a = Date::new(2024, 1, 1).unwrap();
        let b = Date::new(2024, 6, 1).unwrap();
        assert!(DateRange::new(a, b).is_ok());
        assert!(DateRange::new(b, a).is_err());
    }
}
