//! QuantLib validation suite for the pricing models.
//!
//! This is the test that earns OXIS its core promise: every pricing model is
//! cross-checked against an independent oracle within a documented tolerance.
//! The reference data in `validation/reference/*.json` is generated offline by
//! `validation/generate_reference.py` against QuantLib (NOT a runtime
//! dependency). This test deserializes it and asserts the OXIS price matches
//! QuantLib at the tolerance recorded in the file.
//!
//! Regenerate the reference data with:
//! ```text
//! cd validation && pip install -r requirements.txt && python generate_reference.py
//! ```

use oxis_core::{EuropeanOption, ExerciseStyle, MarketData, OptionType};
use oxis_pricing::{binomial, black_scholes, implied_volatility};
use serde::Deserialize;
use serde::de::DeserializeOwned;

#[derive(Debug, Deserialize)]
struct ReferenceFile {
    oracle: String,
    oracle_version: String,
    tolerance: f64,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    dividend_yield: f64,
    time: f64,
    option_type: String,
    price: f64,
}

/// Load and deserialize a reference file from `validation/reference/`.
fn load<T: DeserializeOwned>(name: &str) -> T {
    let path = format!(
        "{}/../../validation/reference/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read reference data at {path}: {e}"));
    serde_json::from_str(&json).expect("reference data is valid JSON")
}

fn parse_option_type(s: &str, i: usize) -> OptionType {
    match s {
        "call" => OptionType::Call,
        "put" => OptionType::Put,
        other => panic!("case {i}: unknown option type {other:?}"),
    }
}

#[test]
fn black_scholes_matches_quantlib() {
    let reference: ReferenceFile = load("black_scholes.json");
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

        let ours = black_scholes(&option, &market)
            .unwrap_or_else(|e| panic!("case {i} failed to price: {e}"));
        let abs_err = (ours - case.price).abs();
        max_abs_err = max_abs_err.max(abs_err);

        assert!(
            abs_err <= reference.tolerance,
            "case {i} ({} S={} K={} r={} vol={} q={} t={}): \
             oxis={ours:.12} quantlib={:.12} abs_err={abs_err:.3e} > tol={:.3e}",
            case.option_type,
            case.spot,
            case.strike,
            case.rate,
            case.volatility,
            case.dividend_yield,
            case.time,
            case.price,
            reference.tolerance,
        );
    }

    eprintln!(
        "validated {} cases vs {} {} — max abs error {:.3e} (tol {:.3e})",
        reference.cases.len(),
        reference.oracle,
        reference.oracle_version,
        max_abs_err,
        reference.tolerance,
    );
}

#[derive(Debug, Deserialize)]
struct BinomialFile {
    oracle: String,
    oracle_version: String,
    tolerance: f64,
    cases: Vec<BinomialCase>,
}

#[derive(Debug, Deserialize)]
struct BinomialCase {
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    dividend_yield: f64,
    time: f64,
    option_type: String,
    style: String,
    steps: usize,
    price: f64,
}

#[test]
fn binomial_matches_quantlib() {
    let reference: BinomialFile = load("binomial.json");
    assert_eq!(reference.oracle, "QuantLib");
    assert!(!reference.cases.is_empty(), "reference data is empty");

    let mut max_abs_err = 0.0_f64;
    for (i, case) in reference.cases.iter().enumerate() {
        let option_type = parse_option_type(&case.option_type, i);
        let style = match case.style.as_str() {
            "european" => ExerciseStyle::European,
            "american" => ExerciseStyle::American,
            other => panic!("case {i}: unknown style {other:?}"),
        };
        let market = MarketData::new(case.spot, case.rate, case.volatility, case.dividend_yield);

        let ours = binomial(
            option_type,
            style,
            &market,
            case.strike,
            case.time,
            case.steps,
        )
        .unwrap_or_else(|e| panic!("case {i} failed to price: {e}"));
        let abs_err = (ours - case.price).abs();
        max_abs_err = max_abs_err.max(abs_err);

        assert!(
            abs_err <= reference.tolerance,
            "case {i} ({} {} S={} K={} vol={} t={} N={}): \
             oxis={ours:.12} quantlib={:.12} abs_err={abs_err:.3e} > tol={:.3e}",
            case.style,
            case.option_type,
            case.spot,
            case.strike,
            case.volatility,
            case.time,
            case.steps,
            case.price,
            reference.tolerance,
        );
    }

    eprintln!(
        "validated {} binomial cases vs {} {} — max abs error {:.3e} (tol {:.3e})",
        reference.cases.len(),
        reference.oracle,
        reference.oracle_version,
        max_abs_err,
        reference.tolerance,
    );
}

#[derive(Debug, Deserialize)]
struct ImpliedVolFile {
    oracle: String,
    oracle_version: String,
    tolerance: f64,
    cases: Vec<ImpliedVolCase>,
}

#[derive(Debug, Deserialize)]
struct ImpliedVolCase {
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    time: f64,
    option_type: String,
    market_price: f64,
    implied_volatility: f64,
}

#[test]
fn implied_vol_matches_quantlib() {
    let reference: ImpliedVolFile = load("implied_vol.json");
    assert_eq!(reference.oracle, "QuantLib");
    assert!(!reference.cases.is_empty(), "reference data is empty");

    let mut max_abs_err = 0.0_f64;
    for (i, case) in reference.cases.iter().enumerate() {
        let option_type = parse_option_type(&case.option_type, i);
        let option = EuropeanOption {
            strike: case.strike,
            expiry_years: case.time,
            option_type,
        };
        // Volatility field is the unknown; pass 0.0.
        let market = MarketData::new(case.spot, case.rate, 0.0, case.dividend_yield);

        let ours = implied_volatility(&option, case.market_price, &market)
            .unwrap_or_else(|e| panic!("case {i} failed to solve: {e}"));
        let abs_err = (ours - case.implied_volatility).abs();
        max_abs_err = max_abs_err.max(abs_err);

        assert!(
            abs_err <= reference.tolerance,
            "case {i} ({} S={} K={} t={} px={}): \
             oxis={ours:.10} quantlib={:.10} abs_err={abs_err:.3e} > tol={:.3e}",
            case.option_type,
            case.spot,
            case.strike,
            case.time,
            case.market_price,
            case.implied_volatility,
            reference.tolerance,
        );
    }

    eprintln!(
        "validated {} implied-vol cases vs {} {} — max abs error {:.3e} (tol {:.3e})",
        reference.cases.len(),
        reference.oracle,
        reference.oracle_version,
        max_abs_err,
        reference.tolerance,
    );
}
