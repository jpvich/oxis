//! # oxis::greeks — option sensitivities (Ring 1)
//!
//! A **Kind A** (pure compute) module: Delta, Gamma, Vega, Theta, Rho.
//! [`analytic_greeks`] gives exact closed-form Black-Scholes Greeks;
//! [`finite_diff_greeks`] is a general central-difference fallback, generic over
//! any European pricer, so this crate depends only on `oxis::core`. Validated
//! against QuantLib's `AnalyticEuropeanEngine` Greeks.

mod analytic;
mod finite_diff;
mod result;

pub use analytic::{Greeks, analytic_greeks};
pub use finite_diff::{Bumps, finite_diff_greeks, finite_diff_greeks_with};
pub use result::GreeksResult;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::output::render;
    use crate::core::{EuropeanOption, MarketData, OptionType, OutputFormat, Tabular};
    use crate::pricing::black_scholes;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b} (tol {tol})");
    }

    /// Analytic Greeks must agree with finite differences of the closed-form
    /// price — the cheapest sanity check that the analytic formulas are right.
    #[test]
    fn analytic_matches_finite_difference() {
        let market = MarketData::new(100.0, 0.05, 0.2, 0.03);
        for option_type in [OptionType::Call, OptionType::Put] {
            for &(strike, t) in &[(90.0, 1.0), (100.0, 0.5), (110.0, 2.0)] {
                let option = EuropeanOption {
                    strike,
                    expiry_years: t,
                    option_type,
                };
                let a = analytic_greeks(&option, &market).unwrap();
                let f = finite_diff_greeks(&option, &market, black_scholes).unwrap();
                close(a.delta, f.delta, 1e-5);
                close(a.gamma, f.gamma, 1e-4);
                close(a.vega, f.vega, 1e-4);
                close(a.theta, f.theta, 1e-4);
                close(a.rho, f.rho, 1e-4);
            }
        }
    }

    /// The full Kind-A seam: pure core → result type → output layer.
    #[test]
    fn greeks_result_renders_through_output_layer() {
        let option = EuropeanOption {
            strike: 100.0,
            expiry_years: 1.0,
            option_type: OptionType::Call,
        };
        let market = MarketData::new(100.0, 0.05, 0.2, 0.0);
        let greeks = analytic_greeks(&option, &market).unwrap();
        let result = GreeksResult::new("analytic", &option, &market, &greeks);

        assert_eq!(result.columns().len(), result.cells().len());
        let json = render(&result, OutputFormat::Json);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["method"], "analytic");
        assert_eq!(parsed["option_type"], "call");
        assert!(parsed["delta"].as_f64().unwrap() > 0.0);
    }
}
