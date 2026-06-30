//! Validation of the Deep Optimal Stopping engine, in two layers.
//!
//! 1. **Inference exactness** — a fixed-weight 1-input net: the Rust *stop
//!    probability* `sigmoid(net(x))` must match numpy to `<= 1e-12`, proving the
//!    sigmoid-output forward math is correct.
//! 2. **Model accuracy** — a Rust-trained DOS estimate must price an American put
//!    within the documented band of a QuantLib CRR tree over a spot grid
//!    (`|price - binomial| <= se_mult*SE + abs`). The learned policy gives a valid
//!    low-biased estimate, so a band — not exactness — is the contract.
//!
//! Reference data: `validation/reference/dos.json`.

use oxis_core::{MarketData, OptionType};
use oxis_ml::{AmericanMlConfig, Layer, Mlp, dos_american, sigmoid};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct DosFile {
    oracle: String,
    oracle_version: String,
    model: String,
    inference_tolerance: f64,
    input_dim: usize,
    layers: Vec<LayerJson>,
    cases: Vec<InferCase>,
    accuracy: Accuracy,
}

#[derive(Debug, Deserialize)]
struct LayerJson {
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct InferCase {
    x: Vec<f64>,
    p: f64,
}

#[derive(Debug, Deserialize)]
struct Accuracy {
    spec: SpecJson,
    train: TrainJson,
    grid: Vec<f64>,
    binomial_price: Vec<f64>,
    bands: Bands,
}

#[derive(Debug, Deserialize)]
struct SpecJson {
    strike: f64,
    rate: f64,
    vol: f64,
    maturity: f64,
    option_type: String,
}

#[derive(Debug, Deserialize)]
struct TrainJson {
    paths: usize,
    steps: usize,
    hidden: Vec<usize>,
    epochs: usize,
    seed: u64,
}

#[derive(Debug, Deserialize)]
struct Bands {
    se_mult: f64,
    abs: f64,
}

fn load() -> DosFile {
    let path = format!(
        "{}/../../validation/reference/dos.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read reference data at {path}: {e}"));
    serde_json::from_str(&json).expect("reference data is valid JSON")
}

fn parse_type(s: &str) -> OptionType {
    match s {
        "call" => OptionType::Call,
        "put" => OptionType::Put,
        other => panic!("unknown option type {other:?}"),
    }
}

/// Layer 1 — the stop probability of a fixed-weight net matches numpy to machine
/// precision.
#[test]
fn inference_sigmoid_matches_numpy() {
    let reference = load();
    assert_eq!(reference.oracle, "QuantLib");
    assert_eq!(reference.model, "dos");
    let tol = reference.inference_tolerance;

    let layers: Vec<Layer> = reference
        .layers
        .iter()
        .map(|l| Layer {
            w: l.w.clone(),
            b: l.b.clone(),
        })
        .collect();
    let mlp = Mlp {
        input_dim: reference.input_dim,
        layers,
    };

    let mut worst = 0.0_f64;
    for (i, case) in reference.cases.iter().enumerate() {
        let fwd = mlp.forward(&case.x);
        let p = sigmoid(mlp.value(&fwd));
        let ep = (p - case.p).abs();
        assert!(ep <= tol, "case {i}: stop-prob error {ep:.3e} > {tol:.1e}");
        worst = worst.max(ep);
    }
    eprintln!(
        "dos inference: {} cases vs QuantLib {} — worst |Δ| {worst:.3e} (tol {tol:.1e})",
        reference.cases.len(),
        reference.oracle_version,
    );
}

/// Layer 2 — a trained DOS policy prices the American put within the documented
/// band of the QuantLib CRR tree over the spot grid.
#[test]
fn accuracy_within_band_vs_binomial() {
    let reference = load();
    let acc = &reference.accuracy;
    let option_type = parse_type(&acc.spec.option_type);

    let mut worst = 0.0_f64;
    for (&spot, &binomial) in acc.grid.iter().zip(acc.binomial_price.iter()) {
        let cfg = AmericanMlConfig {
            market: MarketData::new(spot, acc.spec.rate, acc.spec.vol, 0.0),
            strike: acc.spec.strike,
            expiry: acc.spec.maturity,
            paths: acc.train.paths,
            steps: acc.train.steps,
            seed: acc.train.seed,
            hidden: acc.train.hidden.clone(),
            epochs: acc.train.epochs,
        };
        let est = dos_american(option_type, &cfg).unwrap();
        let gap = (est.price - binomial).abs();
        let budget = acc.bands.se_mult * est.standard_error + acc.bands.abs;
        eprintln!(
            "dos accuracy: S={spot:.0} dos={:.4} (se {:.4}) binomial={binomial:.4} \
             |Δ|={gap:.4} budget={budget:.4}",
            est.price, est.standard_error,
        );
        worst = worst.max(gap);
        assert!(
            gap <= budget,
            "S={spot:.0}: |dos − binomial| = {gap:.4} exceeds {budget:.4}"
        );
    }
    eprintln!(
        "dos accuracy: worst |Δ| {worst:.4} across {} spots",
        acc.grid.len()
    );
}
