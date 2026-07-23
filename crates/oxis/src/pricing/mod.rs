//! # oxis::pricing — option pricing models (Ring 1)
//!
//! The reference **Kind A** module: a pure, I/O-free compute core whose every
//! pricing model is validated against QuantLib (see [`crate::core::contract`]).
//!
//! Implemented: Black-Scholes European (closed-form). Planned (Milestone 2):
//! CRR binomial (European + American), Monte Carlo + Longstaff-Schwartz, and
//! implied volatility.

mod asian;
mod barrier;
mod binomial;
mod black_scholes;
mod implied_vol;
mod lookback;
mod lsm;
mod monte_carlo;
mod result;

pub use asian::{arithmetic_asian_price, geometric_asian_price};
pub use barrier::{BarrierType, barrier_price};
pub use binomial::{DEFAULT_STEPS, binomial};
pub use black_scholes::black_scholes;
pub use implied_vol::{ImpliedVolResult, implied_volatility};
pub use lookback::{LookbackStrike, lookback_price};
pub use lsm::lsm_american;
pub use monte_carlo::{McConfig, McEstimate, monte_carlo_european};
pub use result::{ExoticResult, PriceResult};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{EuropeanOption, ExerciseStyle, MarketData, OptionType, Tabular};

    /// The full Kind-A seam: pure core → result type → output layer.
    #[test]
    fn price_result_renders_through_output_layer() {
        let option = EuropeanOption {
            strike: 105.0,
            expiry_years: 1.0,
            option_type: OptionType::Call,
        };
        let market = MarketData::new(100.0, 0.05, 0.2, 0.0);
        let price = black_scholes(&option, &market).unwrap();

        let result = PriceResult::new(
            "black-scholes",
            OptionType::Call,
            ExerciseStyle::European,
            &option,
            &market,
            price,
        );

        // Tabular contract: equal-length columns and cells.
        assert_eq!(result.columns().len(), result.cells().len());
        let json = crate::core::output::render(&result, crate::core::OutputFormat::Json);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["model"], "black-scholes");
        assert_eq!(parsed["option_type"], "call");
        assert!(parsed["price"].as_f64().unwrap() > 0.0);
        assert!(parsed["standard_error"].is_null());
    }
}
