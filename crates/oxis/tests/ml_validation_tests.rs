//! Validation of `oxis-ml` against a numpy/scipy oracle, in two layers.
//!
//! 1. **Inference exactness** — a fixed-weight softplus MLP: the Rust forward value
//!    and twin input-gradient must match numpy to `<= 1e-12`. This proves the
//!    network math (forward + backprop) is correct, honouring OXIS's exactness DNA.
//! 2. **Model accuracy** — a Rust-trained differential-ML surrogate must price an
//!    option within documented error bands of Black-Scholes over a held-out spot
//!    grid. This proves the *model* is accurate (not exact), which is all an ML
//!    approximation can be. Bands live in the reference file.
//!
//! Reference data: `validation/reference/ml.json`.

use oxis::core::OptionType;
use oxis::ml::{BsSpec, Layer, Mlp, TrainConfig, train_differential};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct MlFile {
    oracle: String,
    oracle_version: String,
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
    y: f64,
    dydx: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct Accuracy {
    spec: SpecJson,
    train: TrainJson,
    grid: Vec<f64>,
    bs_price: Vec<f64>,
    bs_delta: Vec<f64>,
    bands: Bands,
}

#[derive(Debug, Deserialize)]
struct SpecJson {
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    maturity: f64,
    option_type: String,
}

#[derive(Debug, Deserialize)]
struct TrainJson {
    n_samples: usize,
    hidden: Vec<usize>,
    epochs: usize,
    spread: f64,
    seed: u64,
}

#[derive(Debug, Deserialize)]
struct Bands {
    price_max_abs: f64,
    price_rmse: f64,
    delta_max_abs: f64,
    delta_rmse: f64,
}

fn load() -> MlFile {
    let path = format!(
        "{}/../../validation/reference/ml.json",
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

/// Layer 1 — the forward value and twin input-gradient of a fixed-weight net match
/// numpy to machine precision.
#[test]
fn inference_matches_numpy() {
    let reference = load();
    assert_eq!(reference.oracle, "numpy/scipy");
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
        let (y, grad) = mlp.predict_with_grad(&case.x);
        let ey = (y - case.y).abs();
        assert!(ey <= tol, "case {i}: value error {ey:.3e} > {tol:.1e}");
        worst = worst.max(ey);
        assert_eq!(grad.len(), case.dydx.len());
        for (g, e) in grad.iter().zip(case.dydx.iter()) {
            let eg = (g - e).abs();
            assert!(eg <= tol, "case {i}: grad error {eg:.3e} > {tol:.1e}");
            worst = worst.max(eg);
        }
    }
    eprintln!(
        "ml inference: {} cases vs {} — worst |Δ| {worst:.3e} (tol {tol:.1e})",
        reference.cases.len(),
        reference.oracle_version,
    );
}

/// Layer 2 — a trained surrogate prices within the documented bands of
/// Black-Scholes over the spot grid.
#[test]
fn accuracy_within_bands_vs_bs() {
    let reference = load();
    let acc = &reference.accuracy;
    let cfg = TrainConfig {
        spec: BsSpec {
            spot: acc.spec.spot,
            strike: acc.spec.strike,
            rate: acc.spec.rate,
            vol: acc.spec.vol,
            maturity: acc.spec.maturity,
            option_type: parse_type(&acc.spec.option_type),
        },
        n_samples: acc.train.n_samples,
        hidden: acc.train.hidden.clone(),
        epochs: acc.train.epochs,
        spread: acc.train.spread,
        seed: acc.train.seed,
    };
    let model = train_differential(&cfg).unwrap();

    let n = acc.grid.len() as f64;
    let (mut p_max, mut d_max, mut p_sse, mut d_sse) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for ((&s, &bp), &bd) in acc
        .grid
        .iter()
        .zip(acc.bs_price.iter())
        .zip(acc.bs_delta.iter())
    {
        let (mp, md) = model.price_and_delta(s);
        p_max = p_max.max((mp - bp).abs());
        d_max = d_max.max((md - bd).abs());
        p_sse += (mp - bp).powi(2);
        d_sse += (md - bd).powi(2);
    }
    let p_rmse = (p_sse / n).sqrt();
    let d_rmse = (d_sse / n).sqrt();

    eprintln!(
        "ml accuracy: price max={p_max:.4} rmse={p_rmse:.4} | delta max={d_max:.4} rmse={d_rmse:.4} \
         (bands price {:.2}/{:.2}, delta {:.3}/{:.3})",
        acc.bands.price_max_abs,
        acc.bands.price_rmse,
        acc.bands.delta_max_abs,
        acc.bands.delta_rmse,
    );
    assert!(p_max <= acc.bands.price_max_abs, "price max {p_max:.4}");
    assert!(p_rmse <= acc.bands.price_rmse, "price rmse {p_rmse:.4}");
    assert!(d_max <= acc.bands.delta_max_abs, "delta max {d_max:.4}");
    assert!(d_rmse <= acc.bands.delta_rmse, "delta rmse {d_rmse:.4}");
}
