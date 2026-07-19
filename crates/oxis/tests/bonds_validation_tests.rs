//! QuantLib validation suite for fixed-rate bonds.
//!
//! Earns `oxis-bonds` its place in the validated core: bond price, accrued,
//! yield-to-maturity, duration (Macaulay & modified), convexity, and
//! curve-discounted price are cross-checked against QuantLib's `FixedRateBond` /
//! `BondFunctions`. The reference data in `validation/reference/bonds.json` is
//! generated offline by `validation/generate_reference.py` against QuantLib (NOT a
//! runtime dependency). Bond pricing is deterministic closed-form, so the
//! tolerance is tight (≤ 1e-8).
//!
//! Regenerate the reference data with:
//! ```text
//! cd validation && pip install -r requirements.txt && python generate_reference.py
//! ```

use oxis::bonds::{Cashflow, FixedRateBond};
use oxis::curves::YieldCurve;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct BondFile {
    oracle: String,
    oracle_version: String,
    tolerance: f64,
    cases: Vec<BondCase>,
}

#[derive(Debug, Deserialize)]
struct BondCase {
    coupon_rate: f64,
    frequency: u32,
    maturity: f64,
    face: f64,
    test_yield: f64,
    cashflow_times: Vec<f64>,
    cashflow_amounts: Vec<f64>,
    accrued: f64,
    clean_price: f64,
    dirty_price: f64,
    yield_roundtrip: f64,
    macaulay_duration: f64,
    modified_duration: f64,
    convexity: f64,
    flat_rate: f64,
    curve_dirty_price: f64,
}

fn load(name: &str) -> BondFile {
    let path = format!(
        "{}/../../validation/reference/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read reference data at {path}: {e}"));
    serde_json::from_str(&json).expect("reference data is valid JSON")
}

fn build(case: &BondCase) -> FixedRateBond {
    let cashflows: Vec<Cashflow> = case
        .cashflow_times
        .iter()
        .zip(case.cashflow_amounts.iter())
        .map(|(&time, &amount)| Cashflow { time, amount })
        .collect();
    FixedRateBond::from_cashflows(
        case.face,
        case.coupon_rate,
        case.frequency,
        cashflows,
        case.accrued,
    )
    .expect("valid bond")
}

#[test]
fn bonds_match_quantlib() {
    let reference = load("bonds.json");
    assert_eq!(reference.oracle, "QuantLib");
    assert!(reference.oracle_version.starts_with("1."));
    let tol = reference.tolerance;
    assert!(!reference.cases.is_empty());

    let mut worst = 0.0_f64;
    for (i, case) in reference.cases.iter().enumerate() {
        let bond = build(case);
        let mut check = |label: &str, got: f64, want: f64| {
            let err = (got - want).abs();
            assert!(
                err <= tol,
                "case {i} ({}c {}f {}y): {label} {got} vs QuantLib {want} (|Δ| {err} > {tol})",
                case.coupon_rate,
                case.frequency,
                case.maturity
            );
            worst = worst.max(err);
        };

        check(
            "dirty",
            bond.dirty_price_from_yield(case.test_yield).unwrap(),
            case.dirty_price,
        );
        check(
            "clean",
            bond.clean_price_from_yield(case.test_yield).unwrap(),
            case.clean_price,
        );
        check(
            "ytm",
            bond.yield_to_maturity(case.clean_price).unwrap(),
            case.yield_roundtrip,
        );
        check(
            "macaulay",
            bond.macaulay_duration(case.test_yield).unwrap(),
            case.macaulay_duration,
        );
        check(
            "modified",
            bond.modified_duration(case.test_yield).unwrap(),
            case.modified_duration,
        );
        check(
            "convexity",
            bond.convexity(case.test_yield).unwrap(),
            case.convexity,
        );

        let curve = YieldCurve::flat(case.flat_rate).unwrap();
        check(
            "curve_dirty",
            bond.price_from_curve(&curve).unwrap().0,
            case.curve_dirty_price,
        );
    }

    eprintln!(
        "validated {} bonds vs QuantLib {} — worst |Δ| {worst:.3e} (tol {tol:.0e})",
        reference.cases.len(),
        reference.oracle_version,
    );
}

#[test]
fn regular_schedule_reproduces_quantlib_cashflows() {
    // For a bond settling on a coupon date, the ergonomic `regular` builder must
    // produce the same cashflows QuantLib does. Use the par 5y semiannual case.
    let reference = load("bonds.json");
    let case = reference
        .cases
        .iter()
        .find(|c| c.coupon_rate == 0.05 && c.frequency == 2 && c.maturity == 5.0)
        .expect("the par 5y semiannual case");
    let n = (case.maturity * case.frequency as f64).round() as u32;
    let bond = FixedRateBond::regular(case.face, case.coupon_rate, case.frequency, n).unwrap();
    assert_eq!(bond.cashflows().len(), case.cashflow_times.len());
    for (cf, (&t, &a)) in bond
        .cashflows()
        .iter()
        .zip(case.cashflow_times.iter().zip(case.cashflow_amounts.iter()))
    {
        assert!((cf.time - t).abs() < 1e-12, "time {} vs {t}", cf.time);
        assert!((cf.amount - a).abs() < 1e-8, "amount {} vs {a}", cf.amount);
    }
}
