//! Option contract types — the plain inputs the pricing modules operate on.
//!
//! Time-to-expiry is expressed in **years** (`f64`), the unit the closed-form and
//! tree/MC engines use directly. A `DayCount` (see [`super::date`]) converts
//! calendar dates to this fraction at the app edge.

use serde::{Deserialize, Serialize};

/// Whether an option is a call or a put.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptionType {
    /// Right to buy.
    Call,
    /// Right to sell.
    Put,
}

/// When an option may be exercised.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExerciseStyle {
    /// Exercisable only at expiry.
    European,
    /// Exercisable at any time up to expiry.
    American,
}

/// A European (exercise-at-expiry) vanilla option.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EuropeanOption {
    /// Strike price.
    pub strike: f64,
    /// Time to expiry, in years.
    pub expiry_years: f64,
    /// Call or put.
    pub option_type: OptionType,
}

/// An American (exercise-any-time) vanilla option.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AmericanOption {
    /// Strike price.
    pub strike: f64,
    /// Time to expiry, in years.
    pub expiry_years: f64,
    /// Call or put.
    pub option_type: OptionType,
}

impl OptionType {
    /// The intrinsic payoff `max(S-K, 0)` (call) or `max(K-S, 0)` (put).
    pub fn intrinsic(&self, spot: f64, strike: f64) -> f64 {
        match self {
            OptionType::Call => (spot - strike).max(0.0),
            OptionType::Put => (strike - spot).max(0.0),
        }
    }

    /// The lowercase label (`"call"` / `"put"`), for output and parsing.
    pub fn as_str(&self) -> &'static str {
        match self {
            OptionType::Call => "call",
            OptionType::Put => "put",
        }
    }
}

impl ExerciseStyle {
    /// The lowercase label (`"european"` / `"american"`), for output and parsing.
    pub fn as_str(&self) -> &'static str {
        match self {
            ExerciseStyle::European => "european",
            ExerciseStyle::American => "american",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intrinsic_payoffs() {
        assert_eq!(OptionType::Call.intrinsic(120.0, 100.0), 20.0);
        assert_eq!(OptionType::Call.intrinsic(80.0, 100.0), 0.0);
        assert_eq!(OptionType::Put.intrinsic(80.0, 100.0), 20.0);
        assert_eq!(OptionType::Put.intrinsic(120.0, 100.0), 0.0);
    }
}
