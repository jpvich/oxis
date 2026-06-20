//! QuantLib validation suite for the analytic Greeks.
//!
//! Cross-checks [`analytic_greeks`] against QuantLib's `AnalyticEuropeanEngine`
//! delta/gamma/vega/theta/rho. The reference data is generated offline by
//! `validation/generate_reference.py` (QuantLib is NOT a runtime dependency).
//! QuantLib's conventions — vega and rho per unit, theta per year — match the
//! OXIS conventions, so the comparison is direct.

use oxis_core::{EuropeanOption, MarketData, OptionType};
use oxis_greeks::analytic_greeks;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GreeksFile {
    oracle: String,
    oracle_version: String,
    tolerance: f64,
    cases: Vec<GreeksCase>,
}

#[derive(Debug, Deserialize)]
struct GreeksCase {
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    dividend_yield: f64,
    time: f64,
    option_type: String,
    delta: f64,
    gamma: f64,
    vega: f64,
    theta: f64,
    rho: f64,
}

fn load() -> GreeksFile {
    let path = format!(
        "{}/../../validation/reference/greeks.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read reference data at {path}: {e}"));
    serde_json::from_str(&json).expect("reference data is valid JSON")
}

#[test]
fn greeks_match_quantlib() {
    let reference = load();
    assert_eq!(reference.oracle, "QuantLib");
    assert!(!reference.cases.is_empty(), "reference data is empty");

    let mut max_abs_err = 0.0_f64;
    for (i, case) in reference.cases.iter().enumerate() {
        let option_type = match case.option_type.as_str() {
            "call" => OptionType::Call,
            "put" => OptionType::Put,
            other => panic!("case {i}: unknown option type {other:?}"),
        };
        let option = EuropeanOption {
            strike: case.strike,
            expiry_years: case.time,
            option_type,
        };
        let market = MarketData::new(case.spot, case.rate, case.volatility, case.dividend_yield);
        let g =
            analytic_greeks(&option, &market).unwrap_or_else(|e| panic!("case {i} failed: {e}"));

        for (name, ours, theirs) in [
            ("delta", g.delta, case.delta),
            ("gamma", g.gamma, case.gamma),
            ("vega", g.vega, case.vega),
            ("theta", g.theta, case.theta),
            ("rho", g.rho, case.rho),
        ] {
            let abs_err = (ours - theirs).abs();
            max_abs_err = max_abs_err.max(abs_err);
            assert!(
                abs_err <= reference.tolerance,
                "case {i} {name} ({} S={} K={} vol={} t={}): \
                 oxis={ours:.10} quantlib={theirs:.10} abs_err={abs_err:.3e} > tol={:.3e}",
                case.option_type,
                case.spot,
                case.strike,
                case.volatility,
                case.time,
                reference.tolerance,
            );
        }
    }

    eprintln!(
        "validated {} Greeks cases (×5) vs {} {} — max abs error {:.3e} (tol {:.3e})",
        reference.cases.len(),
        reference.oracle,
        reference.oracle_version,
        max_abs_err,
        reference.tolerance,
    );
}
