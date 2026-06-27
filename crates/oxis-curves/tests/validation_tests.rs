//! QuantLib validation suite for the yield-curve term structures.
//!
//! Earns `oxis-curves` its place in the validated core: every interpolation
//! scheme is cross-checked against the matching QuantLib term structure within a
//! tight tolerance. The reference data in `validation/reference/yield_curve.json`
//! is generated offline by `validation/generate_reference.py` against QuantLib
//! (NOT a runtime dependency). Because the curves are deterministic closed-form
//! interpolations, the tolerance is tight (≤ 1e-10), like Black-Scholes — not the
//! statistical band the Monte Carlo models use.
//!
//! Regenerate the reference data with:
//! ```text
//! cd validation && pip install -r requirements.txt && python generate_reference.py
//! ```

use oxis_curves::{Interpolation, YieldCurve};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct YieldCurveFile {
    oracle: String,
    oracle_version: String,
    tolerance: f64,
    cases: Vec<CurveCase>,
}

#[derive(Debug, Deserialize)]
struct CurveCase {
    curve: String,
    interpolation: String,
    pillar_times: Vec<f64>,
    pillar_rates: Vec<f64>,
    t: f64,
    discount: f64,
    zero_rate: f64,
    forward_t2: Option<f64>,
    forward_rate: Option<f64>,
}

fn load(name: &str) -> YieldCurveFile {
    let path = format!(
        "{}/../../validation/reference/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read reference data at {path}: {e}"));
    serde_json::from_str(&json).expect("reference data is valid JSON")
}

fn parse_interp(s: &str, i: usize) -> Interpolation {
    match s {
        "linear" => Interpolation::Linear,
        "log-linear" => Interpolation::LogLinear,
        "natural-cubic" => Interpolation::NaturalCubic,
        other => panic!("case {i}: unknown interpolation {other:?}"),
    }
}

#[test]
fn yield_curve_matches_quantlib() {
    let reference = load("yield_curve.json");
    assert_eq!(reference.oracle, "QuantLib");
    assert!(
        reference.oracle_version.starts_with("1."),
        "unexpected QuantLib version {}",
        reference.oracle_version
    );
    let tol = reference.tolerance;
    assert!(!reference.cases.is_empty(), "no yield-curve cases loaded");

    let mut worst = 0.0_f64;
    for (i, case) in reference.cases.iter().enumerate() {
        let interp = parse_interp(&case.interpolation, i);
        let curve = YieldCurve::from_zero_rates(&case.pillar_times, &case.pillar_rates, interp)
            .unwrap_or_else(|e| panic!("case {i} ({}): build failed: {e}", case.curve));

        let df = curve
            .discount(case.t)
            .unwrap_or_else(|e| panic!("case {i}: discount failed: {e}"));
        let z = curve
            .zero_rate(case.t)
            .unwrap_or_else(|e| panic!("case {i}: zero_rate failed: {e}"));

        let d_err = (df - case.discount).abs();
        let z_err = (z - case.zero_rate).abs();
        assert!(
            d_err <= tol,
            "case {i} ({} {}): discount {df} vs QuantLib {} (|Δ| {d_err} > {tol})",
            case.curve,
            case.interpolation,
            case.discount
        );
        assert!(
            z_err <= tol,
            "case {i} ({} {}): zero_rate {z} vs QuantLib {} (|Δ| {z_err} > {tol})",
            case.curve,
            case.interpolation,
            case.zero_rate
        );
        worst = worst.max(d_err).max(z_err);

        if let (Some(t2), Some(expected_f)) = (case.forward_t2, case.forward_rate) {
            let f = curve
                .forward_rate(case.t, t2)
                .unwrap_or_else(|e| panic!("case {i}: forward_rate failed: {e}"));
            let f_err = (f - expected_f).abs();
            assert!(
                f_err <= tol,
                "case {i} ({} {}): forward {f} vs QuantLib {expected_f} (|Δ| {f_err} > {tol})",
                case.curve,
                case.interpolation
            );
            worst = worst.max(f_err);
        }
    }

    eprintln!(
        "validated {} yield-curve queries vs QuantLib {} — worst |Δ| {worst:.3e} (tol {tol:.0e})",
        reference.cases.len(),
        reference.oracle_version,
    );
}
