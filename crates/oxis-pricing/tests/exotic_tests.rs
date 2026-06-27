//! QuantLib validation suite for the exotic options (Ring 2).
//!
//! Cross-checks the barrier, lookback, and Asian pricers — plus the Heston
//! path-MC tie-in — against QuantLib 1.42.1. Closed-form exotics (barrier,
//! lookback, geometric Asian) hit a tight tolerance (≤ 1e-8); the Monte Carlo
//! ones (arithmetic Asian, Heston European) are compared within a combined
//! standard-error band, the same statistical bar the Longstaff-Schwartz suite
//! uses. Reference data lives in `validation/reference/{barrier,lookback,asian,
//! heston_european}.json`, generated offline by `validation/generate_reference.py`.
//!
//! Regenerate with:
//! ```text
//! cd validation && pip install -r requirements.txt && python generate_reference.py
//! ```

use oxis_core::{EuropeanOption, MarketData, OptionType, mean_and_se};
use oxis_pricing::{
    BarrierType, LookbackStrike, arithmetic_asian_price, barrier_price, geometric_asian_price,
    lookback_price,
};
use oxis_stochastic::{Process, SimConfig, simulate_terminal};
use serde::Deserialize;

fn load<T: serde::de::DeserializeOwned>(name: &str) -> T {
    let path = format!(
        "{}/../../validation/reference/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read reference data at {path}: {e}"));
    serde_json::from_str(&json).expect("reference data is valid JSON")
}

fn option_type(s: &str) -> OptionType {
    match s {
        "call" => OptionType::Call,
        "put" => OptionType::Put,
        other => panic!("unknown option type {other:?}"),
    }
}

fn make(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    div: f64,
    t: f64,
    kind: &str,
) -> (EuropeanOption, MarketData) {
    (
        EuropeanOption {
            strike,
            expiry_years: t,
            option_type: option_type(kind),
        },
        MarketData::new(spot, rate, vol, div),
    )
}

// ----------------------------------------------------------------------------

#[derive(Deserialize)]
struct BarrierFile {
    oracle: String,
    tolerance: f64,
    cases: Vec<BarrierCase>,
}
#[derive(Deserialize)]
struct BarrierCase {
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    div: f64,
    t: f64,
    #[serde(rename = "type")]
    kind: String,
    barrier_type: String,
    barrier: f64,
    price: f64,
}

#[test]
fn barriers_match_quantlib() {
    let reference: BarrierFile = load("barrier.json");
    assert_eq!(reference.oracle, "QuantLib");
    let tol = reference.tolerance;
    let mut worst = 0.0_f64;
    for (i, c) in reference.cases.iter().enumerate() {
        let (opt, mkt) = make(c.spot, c.strike, c.rate, c.vol, c.div, c.t, &c.kind);
        let bt = match c.barrier_type.as_str() {
            "down-in" => BarrierType::DownIn,
            "down-out" => BarrierType::DownOut,
            "up-in" => BarrierType::UpIn,
            "up-out" => BarrierType::UpOut,
            other => panic!("unknown barrier type {other:?}"),
        };
        let got = barrier_price(&opt, &mkt, bt, c.barrier).unwrap();
        let err = (got - c.price).abs();
        assert!(
            err <= tol,
            "case {i} ({} {}): {got} vs QuantLib {} (|Δ| {err} > {tol})",
            c.kind,
            c.barrier_type,
            c.price
        );
        worst = worst.max(err);
    }
    eprintln!(
        "validated {} barriers — worst |Δ| {worst:.3e}",
        reference.cases.len()
    );
}

// ----------------------------------------------------------------------------

#[derive(Deserialize)]
struct LookbackFile {
    tolerance: f64,
    cases: Vec<LookbackCase>,
}
#[derive(Deserialize)]
struct LookbackCase {
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    div: f64,
    t: f64,
    #[serde(rename = "type")]
    kind: String,
    strike_type: String,
    price: f64,
}

#[test]
fn lookbacks_match_quantlib() {
    let reference: LookbackFile = load("lookback.json");
    let tol = reference.tolerance;
    let mut worst = 0.0_f64;
    for (i, c) in reference.cases.iter().enumerate() {
        let (opt, mkt) = make(c.spot, c.strike, c.rate, c.vol, c.div, c.t, &c.kind);
        let st = match c.strike_type.as_str() {
            "floating" => LookbackStrike::Floating,
            "fixed" => LookbackStrike::Fixed,
            other => panic!("unknown strike type {other:?}"),
        };
        let got = lookback_price(&opt, &mkt, st).unwrap();
        let err = (got - c.price).abs();
        assert!(
            err <= tol,
            "case {i} ({} {}): {got} vs QuantLib {} (|Δ| {err} > {tol})",
            c.kind,
            c.strike_type,
            c.price
        );
        worst = worst.max(err);
    }
    eprintln!(
        "validated {} lookbacks — worst |Δ| {worst:.3e}",
        reference.cases.len()
    );
}

// ----------------------------------------------------------------------------

#[derive(Deserialize)]
struct AsianFile {
    tolerance: f64,
    cases: Vec<AsianCase>,
}
#[derive(Deserialize)]
struct AsianCase {
    average: String,
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    div: f64,
    t: f64,
    #[serde(rename = "type")]
    kind: String,
    price: f64,
    ql_error: Option<f64>,
    n_fixings: Option<usize>,
    paths: Option<usize>,
    seed: Option<u64>,
}

#[test]
fn asians_match_quantlib() {
    let reference: AsianFile = load("asian.json");
    let tol = reference.tolerance;
    for (i, c) in reference.cases.iter().enumerate() {
        let (opt, mkt) = make(c.spot, c.strike, c.rate, c.vol, c.div, c.t, &c.kind);
        match c.average.as_str() {
            "geometric" => {
                let got = geometric_asian_price(&opt, &mkt).unwrap();
                let err = (got - c.price).abs();
                assert!(
                    err <= tol,
                    "case {i} (geometric {}): {got} vs QuantLib {} (|Δ| {err} > {tol})",
                    c.kind,
                    c.price
                );
            }
            "arithmetic" => {
                let cfg = SimConfig {
                    paths: c.paths.unwrap(),
                    steps: 0,
                    seed: c.seed.unwrap(),
                };
                let est = arithmetic_asian_price(&opt, &mkt, c.n_fixings.unwrap(), &cfg).unwrap();
                let se = est.standard_error;
                let combined = (se * se + c.ql_error.unwrap().powi(2)).sqrt();
                let err = (est.price - c.price).abs();
                assert!(
                    err <= 4.0 * combined + 1e-9,
                    "case {i} (arithmetic {}): {} (se {se}) vs QuantLib {} ± {} (|Δ| {err} > {})",
                    c.kind,
                    est.price,
                    c.price,
                    c.ql_error.unwrap(),
                    4.0 * combined
                );
            }
            other => panic!("unknown average {other:?}"),
        }
    }
    eprintln!(
        "validated {} Asian options vs QuantLib",
        reference.cases.len()
    );
}

// ----------------------------------------------------------------------------

#[derive(Deserialize)]
struct HestonFile {
    cases: Vec<HestonCase>,
}
#[derive(Deserialize)]
struct HestonCase {
    spot: f64,
    strike: f64,
    rate: f64,
    div: f64,
    t: f64,
    #[serde(rename = "type")]
    kind: String,
    v0: f64,
    kappa: f64,
    theta: f64,
    xi: f64,
    rho: f64,
    price: f64,
    paths: usize,
    steps: usize,
    seed: u64,
}

#[test]
fn heston_european_matches_quantlib() {
    // The end-to-end Heston check: price a European option by Monte Carlo over
    // Heston paths from oxis-stochastic and compare to QuantLib's semi-analytic
    // AnalyticHestonEngine. Full-truncation Euler carries a small discretization
    // bias, so the band is a few standard errors plus a small absolute allowance.
    let reference: HestonFile = load("heston_european.json");
    let mut worst = 0.0_f64;
    for (i, c) in reference.cases.iter().enumerate() {
        let ot = option_type(&c.kind);
        let process = Process::Heston {
            mu: c.rate - c.div, // risk-neutral asset drift
            v0: c.v0,
            kappa: c.kappa,
            theta: c.theta,
            xi: c.xi,
            rho: c.rho,
        };
        let cfg = SimConfig {
            paths: c.paths,
            steps: c.steps,
            seed: c.seed,
        };
        let sample = simulate_terminal(&process, c.spot, c.t, &cfg).unwrap();
        let disc = (-c.rate * c.t).exp();
        // Antithetic pairing on the payoff (terminals laid out [up, dn, …]).
        let pair_payoffs: Vec<f64> = sample
            .terminals
            .chunks_exact(2)
            .map(|p| 0.5 * (ot.intrinsic(p[0], c.strike) + ot.intrinsic(p[1], c.strike)))
            .collect();
        let (mean, se) = mean_and_se(&pair_payoffs);
        let price = disc * mean;
        let se = disc * se;
        let err = (price - c.price).abs();
        assert!(
            err <= 5.0 * se + 0.15,
            "case {i} ({}): MC {price} (se {se}) vs QuantLib {} (|Δ| {err} > {})",
            c.kind,
            c.price,
            5.0 * se + 0.15
        );
        worst = worst.max(err);
    }
    eprintln!(
        "validated {} Heston Europeans vs QuantLib — worst |Δ| {worst:.3e}",
        reference.cases.len()
    );
}
