//! Calendar dates and day-count conventions.
//!
//! [`Date`] is a lightweight proleptic-Gregorian calendar date. Day differences
//! are computed via a serial-day conversion (Howard Hinnant's `days_from_civil`),
//! which is exact and branch-cheap. [`DayCount`] turns a pair of dates into a
//! year fraction under a named market convention.

use crate::error::OxisError;
use serde::{Deserialize, Serialize};

/// A calendar date in the proleptic Gregorian calendar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Date {
    year: i32,
    month: u8,
    day: u8,
}

impl Date {
    /// Construct a date, validating the month (1–12) and day (1–last-of-month,
    /// leap-year aware).
    pub fn new(year: i32, month: u8, day: u8) -> Result<Self, OxisError> {
        if !(1..=12).contains(&month) {
            return Err(OxisError::invalid_input(format!(
                "month {month} not in 1..=12"
            )));
        }
        let last = days_in_month(year, month);
        if !(1..=last).contains(&day) {
            return Err(OxisError::invalid_input(format!(
                "day {day} not in 1..={last} for {year}-{month:02}"
            )));
        }
        Ok(Self { year, month, day })
    }

    /// The year component.
    pub fn year(&self) -> i32 {
        self.year
    }
    /// The month component (1–12).
    pub fn month(&self) -> u8 {
        self.month
    }
    /// The day component (1–31).
    pub fn day(&self) -> u8 {
        self.day
    }

    /// Days since the Unix epoch (1970-01-01); negative before it. Used for exact
    /// day differences.
    pub fn serial(&self) -> i64 {
        days_from_civil(self.year as i64, self.month as i64, self.day as i64)
    }

    /// The number of calendar days from `self` to `other` (negative if `other`
    /// precedes `self`).
    pub fn days_until(&self, other: Date) -> i64 {
        other.serial() - self.serial()
    }
}

/// A day-count convention: how to turn a date interval into a year fraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DayCount {
    /// Actual/365 Fixed: actual days divided by 365.
    #[default]
    Act365,
    /// Actual/360: actual days divided by 360.
    Act360,
    /// 30/360 (US/NASD): 360-day year with month-end adjustments.
    Thirty360,
}

impl DayCount {
    /// The year fraction between `start` and `end` under this convention.
    pub fn year_fraction(&self, start: Date, end: Date) -> f64 {
        match self {
            DayCount::Act365 => start.days_until(end) as f64 / 365.0,
            DayCount::Act360 => start.days_until(end) as f64 / 360.0,
            DayCount::Thirty360 => days_30_360(start, end) as f64 / 360.0,
        }
    }
}

fn is_leap(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// Days since 1970-01-01 (Howard Hinnant, *chrono-compatible*; public-domain).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

/// 30/360 US (NASD) day count between two dates.
fn days_30_360(start: Date, end: Date) -> i64 {
    let mut d1 = start.day as i64;
    let mut d2 = end.day as i64;
    if d1 == 31 {
        d1 = 30;
    }
    if d2 == 31 && d1 == 30 {
        d2 = 30;
    }
    360 * (end.year as i64 - start.year as i64)
        + 30 * (end.month as i64 - start.month as i64)
        + (d2 - d1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }

    #[test]
    fn rejects_invalid_dates() {
        assert!(Date::new(2024, 13, 1).is_err());
        assert!(Date::new(2023, 2, 29).is_err()); // not a leap year
        assert!(Date::new(2024, 2, 29).is_ok()); // leap year
    }

    #[test]
    fn day_differences_are_exact() {
        let a = Date::new(2024, 1, 1).unwrap();
        let b = Date::new(2025, 1, 1).unwrap();
        assert_eq!(a.days_until(b), 366); // 2024 is a leap year
        assert_eq!(b.days_until(a), -366);
    }

    #[test]
    fn act365_and_act360() {
        let a = Date::new(2024, 1, 1).unwrap();
        let b = Date::new(2024, 7, 1).unwrap(); // 182 days
        assert_close(DayCount::Act365.year_fraction(a, b), 182.0 / 365.0);
        assert_close(DayCount::Act360.year_fraction(a, b), 182.0 / 360.0);
    }

    #[test]
    fn thirty_360_half_year() {
        let a = Date::new(2024, 1, 1).unwrap();
        let b = Date::new(2024, 7, 1).unwrap();
        // 30/360: exactly half a year between matching days six months apart.
        assert_close(DayCount::Thirty360.year_fraction(a, b), 0.5);
    }
}
