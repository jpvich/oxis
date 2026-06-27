//! Closed-form validation suite for the stochastic process generators.
//!
//! A path simulator has no "price" to check, so the oracle is the **closed-form
//! moment** of the law it samples. `validation/reference/processes.json` holds the
//! exact terminal mean/variance of each process, computed independently in Python
//! by `validation/generate_reference.py` (pure analytic formulas — no QuantLib
//! needed for these). Two things are validated:
//!
//! 1. **The Rust moment formulas** ([`Process::analytic_moments`]) reproduce the
//!    independent Python closed form to ~1e-9 (catches formula typos).
//! 2. **The simulator** reproduces those moments: the simulated terminal mean and
//!    standard deviation fall within a documented band of the reference.
//!
//! The exact-in-distribution schemes (GBM, OU/Vasicek, Merton) match up to sampling
//! error; the full-truncation square-root schemes (CIR, Heston variance) carry a
//! small `O(dt)` discretization bias, which the relative bands absorb. Heston has no
//! closed-form variance here — its dynamics are validated by the European-price
//! cross-check against QuantLib's `AnalyticHestonEngine` in `oxis-pricing`.
//!
//! Regenerate the reference data with:
//! ```text
//! cd validation && pip install -r requirements.txt && python generate_reference.py
//! ```

use oxis_core::{mean_and_se, sample_mean_var};
use oxis_stochastic::{Process, SimConfig, simulate_terminal};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ProcessFile {
    oracle: String,
    cases: Vec<ProcessCase>,
}

#[derive(Debug, Deserialize)]
struct ProcessCase {
    process: String,
    x0: f64,
    t: f64,
    steps: usize,
    paths: usize,
    seed: u64,
    mu: Option<f64>,
    sigma: Option<f64>,
    kappa: Option<f64>,
    theta: Option<f64>,
    lambda: Option<f64>,
    jump_mean: Option<f64>,
    jump_std: Option<f64>,
    v0: Option<f64>,
    xi: Option<f64>,
    rho: Option<f64>,
    mean: f64,
    var: Option<f64>,
    /// Relative tolerance on the terminal std (absorbs estimator noise +
    /// discretization for the square-root schemes).
    std_rel_tol: f64,
}

fn load(name: &str) -> ProcessFile {
    let path = format!(
        "{}/../../validation/reference/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read reference data at {path}: {e}"));
    serde_json::from_str(&json).expect("reference data is valid JSON")
}

fn build(c: &ProcessCase) -> Process {
    let req =
        |o: Option<f64>, name: &str| o.unwrap_or_else(|| panic!("{} missing {name}", c.process));
    match c.process.as_str() {
        "gbm" => Process::Gbm {
            mu: req(c.mu, "mu"),
            sigma: req(c.sigma, "sigma"),
        },
        "ornstein-uhlenbeck" => Process::OrnsteinUhlenbeck {
            kappa: req(c.kappa, "kappa"),
            theta: req(c.theta, "theta"),
            sigma: req(c.sigma, "sigma"),
        },
        "vasicek" => Process::Vasicek {
            kappa: req(c.kappa, "kappa"),
            theta: req(c.theta, "theta"),
            sigma: req(c.sigma, "sigma"),
        },
        "cir" => Process::Cir {
            kappa: req(c.kappa, "kappa"),
            theta: req(c.theta, "theta"),
            sigma: req(c.sigma, "sigma"),
        },
        "merton-jump" => Process::MertonJump {
            mu: req(c.mu, "mu"),
            sigma: req(c.sigma, "sigma"),
            lambda: req(c.lambda, "lambda"),
            jump_mean: req(c.jump_mean, "jump_mean"),
            jump_std: req(c.jump_std, "jump_std"),
        },
        "heston" => Process::Heston {
            mu: req(c.mu, "mu"),
            v0: req(c.v0, "v0"),
            kappa: req(c.kappa, "kappa"),
            theta: req(c.theta, "theta"),
            xi: req(c.xi, "xi"),
            rho: req(c.rho, "rho"),
        },
        other => panic!("unknown process {other:?}"),
    }
}

#[test]
fn analytic_moments_match_independent_closed_form() {
    let reference = load("processes.json");
    assert!(!reference.cases.is_empty());
    for (i, c) in reference.cases.iter().enumerate() {
        let process = build(c);
        let (mean, var) = process.analytic_moments(c.x0, c.t);
        assert!(
            (mean - c.mean).abs() <= 1e-9 + 1e-9 * c.mean.abs(),
            "case {i} ({}): analytic mean {mean} vs reference {}",
            c.process,
            c.mean
        );
        match (var, c.var) {
            (Some(v), Some(rv)) => assert!(
                (v - rv).abs() <= 1e-9 + 1e-9 * rv.abs(),
                "case {i} ({}): analytic var {v} vs reference {rv}",
                c.process
            ),
            (None, None) => {}
            _ => panic!("case {i} ({}): variance presence mismatch", c.process),
        }
    }
}

#[test]
fn simulated_moments_match_reference() {
    let reference = load("processes.json");
    assert_eq!(reference.oracle, "closed-form");

    let mut worst_mean = 0.0_f64;
    for (i, c) in reference.cases.iter().enumerate() {
        let process = build(c);
        let cfg = SimConfig {
            paths: c.paths,
            steps: c.steps,
            seed: c.seed,
        };
        let sample = simulate_terminal(&process, c.x0, c.t, &cfg).unwrap();
        let (sample_mean, mean_se) = mean_and_se(&sample.pair_means);
        let (_, sample_var) = sample_mean_var(&sample.terminals);
        let sample_std = sample_var.sqrt();

        // Mean: within 5 standard errors plus a 1% relative allowance for the
        // square-root schemes' discretization bias.
        let mean_band = 5.0 * mean_se + 0.01 * c.mean.abs() + 1e-9;
        let mean_err = (sample_mean - c.mean).abs();
        assert!(
            mean_err <= mean_band,
            "case {i} ({}): sample mean {sample_mean} vs {} (|Δ| {mean_err} > {mean_band})",
            c.process,
            c.mean
        );
        worst_mean = worst_mean.max(mean_err);

        // Std: relative band per case (looser for the discretized schemes).
        if let Some(rv) = c.var {
            let ref_std = rv.sqrt();
            let std_err = (sample_std - ref_std).abs();
            let std_band = c.std_rel_tol * ref_std + 1e-9;
            assert!(
                std_err <= std_band,
                "case {i} ({}): sample std {sample_std} vs {ref_std} (|Δ| {std_err} > {std_band})",
                c.process
            );
        }
    }
    eprintln!(
        "validated {} processes vs closed-form moments — worst mean |Δ| {worst_mean:.3e}",
        reference.cases.len()
    );
}
