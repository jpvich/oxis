//! # oxis-ml — ML-based pricing (Ring 4) — the OXIS differentiator
//!
//! Validated machine-learning pricing, **hand-rolled with no ML framework**. The
//! first model is **Differential Machine Learning** (Huge & Savine, 2020): a
//! *twin network* — a softplus MLP whose forward pass predicts an option's price
//! and whose backprop pass predicts its delta — trained on simulated payoffs *and*
//! their pathwise differentials. The network, its twin (input-gradient) pass, and
//! the doubled-network training gradient are plain linear algebra over
//! [`oxis_core`]'s math layer; nothing here depends on `candle`/`burn`/`tch`, so
//! the binary stays portable and every number is auditable.
//!
//! Model *inference* is a **Kind A** (pure compute) concern. It lands on top of the
//! validated classical engines (`oxis-pricing`, `oxis-greeks`) so its accuracy is
//! measured against a trusted baseline.
//!
//! ## Method (European Black-Scholes, 1-D in spot)
//!
//! For a single exact GBM step `t → T`, each training sample carries the discounted
//! payoff `y` and its pathwise differential `q = ∂y/∂S_t` (indicator × elasticity).
//! The loss mixes a value term and a differential term,
//! `L = α·mean(ŷ − y)² + β·mean λ²(ĝ − q)²`, on standardized data. See
//! [`train`] for the gradient derivation and [`data`] for the labels.
//!
//! ## Validation contract — two layers
//!
//! An ML model is an *approximation*: it will not match Black-Scholes to `1e-10`.
//! OXIS's non-negotiable ("no model is done without a validation test vs a trusted
//! baseline") is preserved by splitting the test in two:
//!
//! 1. **Inference exactness** — the forward value and input-gradient of a
//!    *fixed-weight* net match an independent numpy reference to **≤ 1e-12**. This
//!    proves the math (forward + twin backprop) is correct.
//! 2. **Model accuracy** — the *trained* net's price and delta lie within a
//!    **documented error band** vs Black-Scholes over a held-out spot grid. This
//!    proves the model is accurate, not exact. Bands are recorded in
//!    `docs/models.md`.
//!
//! Both run in `crates/oxis-ml/tests/validation_tests.rs` against
//! `validation/reference/ml.json`.

#![forbid(unsafe_code)]

mod activation;
mod american;
mod data;
mod deep_lsm;
mod dos;
mod mlp;
mod optim;
mod result;
mod train;

pub use activation::{sigmoid, softplus, softplus_prime, softplus_second};
pub use data::{BsSpec, DiffSample, generate_european};
pub use deep_lsm::{AmericanMlConfig, deep_lsm_american};
pub use dos::dos_american;
pub use mlp::{Forward, Layer, Mlp, Twin};
pub use result::{
    AmericanMlReport, MlPricingReport, deep_lsm_price, differential_ml_price, dos_price,
};
pub use train::{TrainConfig, TrainedModel, train_differential};

#[cfg(test)]
mod tests {
    use super::*;
    use oxis_core::OptionType;

    /// End-to-end: train a tiny surrogate and check it prices an ATM call in the
    /// right ballpark, exercising `module → core` and `module → pricing/greeks`.
    #[test]
    fn end_to_end_smoke() {
        let cfg = TrainConfig {
            spec: BsSpec {
                spot: 100.0,
                strike: 100.0,
                rate: 0.05,
                vol: 0.2,
                maturity: 1.0,
                option_type: OptionType::Call,
            },
            n_samples: 2048,
            hidden: vec![16, 16],
            epochs: 40,
            spread: 2.0,
            seed: 1,
        };
        let report = differential_ml_price(&cfg).unwrap();
        assert!(report.ml_price.is_finite());
        assert!(
            report.price_abs_err < 2.0,
            "abs err {}",
            report.price_abs_err
        );
    }
}
