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
use oxis_pricing::{
    McConfig, binomial, black_scholes, implied_volatility, lsm_american, monte_carlo_european,
};
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

/// European Monte Carlo is validated against the Black-Scholes closed form,
/// which is itself QuantLib-validated (see `black_scholes_matches_quantlib`).
/// Because MC is stochastic, the bar is statistical: the price must land within
/// a few standard errors of the exact value. We use a fixed seed and a large
/// path count so the assertion is deterministic across runs.
#[test]
fn monte_carlo_european_matches_black_scholes() {
    let reference: ReferenceFile = load("black_scholes.json");
    assert!(!reference.cases.is_empty(), "reference data is empty");

    // 4 standard errors ≈ a 1-in-16000 per-case false-failure rate; with a fixed
    // seed the outcome is deterministic regardless.
    const K_SIGMA: f64 = 4.0;
    let cfg = McConfig {
        paths: 1_000_000,
        steps: 1,
        seed: 20_240_101,
    };

    let mut max_n_sigma = 0.0_f64;
    for (i, case) in reference.cases.iter().enumerate() {
        let option_type = parse_option_type(&case.option_type, i);
        let option = EuropeanOption {
            strike: case.strike,
            expiry_years: case.time,
            option_type,
        };
        let market = MarketData::new(case.spot, case.rate, case.volatility, case.dividend_yield);

        let est = monte_carlo_european(&option, &market, &cfg)
            .unwrap_or_else(|e| panic!("case {i} failed to price: {e}"));
        let err = (est.price - case.price).abs();
        // Guard against the deterministic zero-SE limits (none here, but safe).
        let tol = (K_SIGMA * est.standard_error).max(1e-9);
        let n_sigma = if est.standard_error > 0.0 {
            err / est.standard_error
        } else {
            0.0
        };
        max_n_sigma = max_n_sigma.max(n_sigma);

        assert!(
            err <= tol,
            "case {i} ({} S={} K={} vol={} t={}): mc={:.6} bs={:.6} err={err:.3e} \
             se={:.3e} = {n_sigma:.2}σ > {K_SIGMA}σ",
            case.option_type,
            case.spot,
            case.strike,
            case.volatility,
            case.time,
            est.price,
            case.price,
            est.standard_error,
        );
    }

    eprintln!(
        "validated {} European MC cases vs Black-Scholes — worst deviation {:.2}σ (bar {K_SIGMA}σ)",
        reference.cases.len(),
        max_n_sigma,
    );
}

#[derive(Debug, Deserialize)]
struct McAmericanFile {
    oracle: String,
    oracle_version: String,
    cases: Vec<McAmericanCase>,
}

#[derive(Debug, Deserialize)]
struct McAmericanCase {
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    dividend_yield: f64,
    time: f64,
    option_type: String,
    steps: usize,
    price: f64,
    error_estimate: f64,
}

/// American Longstaff-Schwartz is validated two ways:
///
/// 1. **vs QuantLib's own LSM engine** (`MCAmericanEngine`) — apples-to-apples,
///    since both are the same method and share its lower-bound bias. Bar:
///    `5·combined_SE + 0.05`. The absolute floor covers deep-in-the-money cases
///    where immediate exercise is optimal: there OXIS returns the exact
///    intrinsic (SE → 0) while QuantLib's regression can dip slightly *below*
///    intrinsic, a fixed small gap rather than a statistical one.
///
/// 2. **vs our QuantLib-validated binomial American price** (the true value).
///    LSM is a *lower-bound* estimator — a regression-based exercise policy is
///    necessarily suboptimal — so it sits below the tree. Bar:
///    `5·SE + 0.025·price`, i.e. statistical error plus a documented ≤2.5%
///    LSM bias band (largest for high-vol / dividend cases where the
///    continuation surface is hardest for a degree-2 basis to fit).
#[test]
fn lsm_american_matches_quantlib_and_binomial() {
    let reference: McAmericanFile = load("monte_carlo_american.json");
    assert_eq!(reference.oracle, "QuantLib");
    assert!(!reference.cases.is_empty(), "reference data is empty");

    let mut worst_vs_ql = 0.0_f64;
    let mut worst_vs_tree = 0.0_f64;
    for (i, case) in reference.cases.iter().enumerate() {
        let option_type = parse_option_type(&case.option_type, i);
        let market = MarketData::new(case.spot, case.rate, case.volatility, case.dividend_yield);

        // Match QuantLib's time-grid size so the two LSM estimators share the
        // same discretization; a fixed seed keeps the assertion deterministic.
        let cfg = McConfig {
            paths: 100_000,
            steps: case.steps,
            seed: 20_240_102,
        };
        let est = lsm_american(option_type, &market, case.strike, case.time, &cfg)
            .unwrap_or_else(|e| panic!("case {i} failed to price: {e}"));

        // (1) vs QuantLib LSM: within 5 combined standard errors (+ a small
        // floor for the near-deterministic immediate-exercise cases where both
        // SEs collapse but the two regressions differ by a fixed small gap).
        let combined_se = (est.standard_error.powi(2) + case.error_estimate.powi(2)).sqrt();
        let ql_tol = 5.0 * combined_se + 0.05;
        let err_ql = (est.price - case.price).abs();
        worst_vs_ql = worst_vs_ql.max(err_ql);
        assert!(
            err_ql <= ql_tol,
            "case {i} ({} S={} K={} vol={} t={}): lsm={:.6} ql={:.6} err={err_ql:.3e} \
             > tol={ql_tol:.3e} (our_se={:.3e} ql_se={:.3e})",
            case.option_type,
            case.spot,
            case.strike,
            case.volatility,
            case.time,
            est.price,
            case.price,
            est.standard_error,
            case.error_estimate,
        );

        // (2) vs our binomial American (QuantLib-validated, lower estimator
        // bias): LSM sits a touch below the tree, so allow a small bias band.
        let tree = binomial(
            option_type,
            ExerciseStyle::American,
            &market,
            case.strike,
            case.time,
            2000,
        )
        .unwrap();
        let tree_tol = 5.0 * est.standard_error + 0.025 * tree;
        let err_tree = (est.price - tree).abs();
        worst_vs_tree = worst_vs_tree.max(err_tree);
        assert!(
            err_tree <= tree_tol,
            "case {i} ({} S={} K={} vol={} t={}): lsm={:.6} binomial={:.6} \
             err={err_tree:.3e} > tol={tree_tol:.3e}",
            case.option_type,
            case.spot,
            case.strike,
            case.volatility,
            case.time,
            est.price,
            tree,
        );
    }

    eprintln!(
        "validated {} American LSM cases vs {} {} — worst |Δ| vs QuantLib LSM {:.3e}, \
         vs binomial (true value) {:.3e}",
        reference.cases.len(),
        reference.oracle,
        reference.oracle_version,
        worst_vs_ql,
        worst_vs_tree,
    );
}
