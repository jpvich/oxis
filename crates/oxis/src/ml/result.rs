//! Renderable result for the ML pricing surfaces.
//!
//! [`MlPricingReport`] places the differential-ML estimate next to the classical
//! Black-Scholes value — the whole point of OXIS's ML ring is that the learned
//! price is *measured against* a trusted engine. It derives `Serialize` and
//! implements [`Tabular`] as a single record.

use crate::core::{
    Cell, Column, EuropeanOption, ExerciseStyle, MarketData, OptionType, OxisError, Tabular,
};
use crate::greeks::analytic_greeks;
use crate::ml::data::BsSpec;
use crate::ml::deep_lsm::{AmericanMlConfig, deep_lsm_american};
use crate::ml::dos::dos_american;
use crate::ml::train::{TrainConfig, train_differential};
use crate::pricing::McEstimate;
use crate::pricing::{binomial, black_scholes};
use serde::Serialize;

/// A differential-ML pricing result with its classical baseline.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MlPricingReport {
    /// Spot the price/delta were evaluated at.
    pub spot: f64,
    /// `"call"` or `"put"`.
    pub option_type: &'static str,
    /// ML-predicted price.
    pub ml_price: f64,
    /// ML-predicted delta (the twin network's input-gradient).
    pub ml_delta: f64,
    /// Black-Scholes price (oracle).
    pub bs_price: f64,
    /// Black-Scholes delta (oracle).
    pub bs_delta: f64,
    /// `|ml_price − bs_price|`.
    pub price_abs_err: f64,
    /// `|ml_delta − bs_delta|`.
    pub delta_abs_err: f64,
    /// Training samples used.
    pub n_samples: usize,
    /// Training epochs.
    pub epochs: usize,
    /// Final training loss (standardized scale).
    pub final_loss: f64,
}

impl Tabular for MlPricingReport {
    fn columns(&self) -> Vec<Column> {
        vec![
            Column::new("spot"),
            Column::new("option_type"),
            Column::new("ml_price"),
            Column::new("ml_delta"),
            Column::new("bs_price"),
            Column::new("bs_delta"),
            Column::new("price_abs_err"),
            Column::new("delta_abs_err"),
            Column::new("n_samples"),
            Column::new("epochs"),
            Column::new("final_loss"),
        ]
    }
    fn cells(&self) -> Vec<Cell> {
        vec![
            Cell::F64(self.spot),
            Cell::str(self.option_type),
            Cell::F64(self.ml_price),
            Cell::F64(self.ml_delta),
            Cell::F64(self.bs_price),
            Cell::F64(self.bs_delta),
            Cell::F64(self.price_abs_err),
            Cell::F64(self.delta_abs_err),
            Cell::Int(self.n_samples as i64),
            Cell::Int(self.epochs as i64),
            Cell::F64(self.final_loss),
        ]
    }
}

/// Train a differential-ML surrogate for `cfg` and report its price/delta at the
/// spec's reference spot, alongside the Black-Scholes baseline.
///
/// # Errors
/// Propagates training or pricing errors ([`OxisError::InvalidInput`]).
pub fn differential_ml_price(cfg: &TrainConfig) -> Result<MlPricingReport, OxisError> {
    let spec = cfg.spec;
    let model = train_differential(cfg)?;
    let (ml_price, ml_delta) = model.price_and_delta(spec.spot);
    let (bs_price, bs_delta) = black_scholes_price_delta(&spec)?;

    Ok(MlPricingReport {
        spot: spec.spot,
        option_type: spec.option_type.as_str(),
        ml_price,
        ml_delta,
        bs_price,
        bs_delta,
        price_abs_err: (ml_price - bs_price).abs(),
        delta_abs_err: (ml_delta - bs_delta).abs(),
        n_samples: model.n_samples,
        epochs: model.epochs,
        final_loss: model.final_loss,
    })
}

/// A neural American pricing result with its trusted binomial baseline.
///
/// One report type serves both engines; `method` distinguishes `"deep-lsm"` from
/// `"dos"`. The estimate is low-biased, so `binomial_price` (a 2000-step CRR tree,
/// the QuantLib-validated baseline) is the oracle and `abs_err` the headline gap.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AmericanMlReport {
    /// Pricing method: `"deep-lsm"` or `"dos"`.
    pub method: &'static str,
    /// `"call"` or `"put"`.
    pub option_type: &'static str,
    /// Spot.
    pub spot: f64,
    /// Strike.
    pub strike: f64,
    /// Risk-free rate (continuously compounded).
    pub rate: f64,
    /// Volatility.
    pub vol: f64,
    /// Time to expiry in years.
    pub maturity: f64,
    /// Neural Monte-Carlo price estimate.
    pub ml_price: f64,
    /// Antithetic standard error of `ml_price`.
    pub standard_error: f64,
    /// Binomial (CRR, 2000-step American) baseline price.
    pub binomial_price: f64,
    /// `|ml_price − binomial_price|`.
    pub abs_err: f64,
    /// Simulated paths.
    pub paths: usize,
    /// Exercise dates.
    pub steps: usize,
    /// Training epochs per exercise date.
    pub epochs: usize,
}

impl Tabular for AmericanMlReport {
    fn columns(&self) -> Vec<Column> {
        vec![
            Column::new("method"),
            Column::new("option_type"),
            Column::new("spot"),
            Column::new("strike"),
            Column::new("rate"),
            Column::new("vol"),
            Column::new("maturity"),
            Column::new("ml_price"),
            Column::new("standard_error"),
            Column::new("binomial_price"),
            Column::new("abs_err"),
            Column::new("paths"),
            Column::new("steps"),
            Column::new("epochs"),
        ]
    }
    fn cells(&self) -> Vec<Cell> {
        vec![
            Cell::str(self.method),
            Cell::str(self.option_type),
            Cell::F64(self.spot),
            Cell::F64(self.strike),
            Cell::F64(self.rate),
            Cell::F64(self.vol),
            Cell::F64(self.maturity),
            Cell::F64(self.ml_price),
            Cell::F64(self.standard_error),
            Cell::F64(self.binomial_price),
            Cell::F64(self.abs_err),
            Cell::Int(self.paths as i64),
            Cell::Int(self.steps as i64),
            Cell::Int(self.epochs as i64),
        ]
    }
}

/// Price an American option by Deep LSM and report it against the binomial baseline.
///
/// # Errors
/// Propagates pricing errors ([`OxisError::InvalidInput`]).
pub fn deep_lsm_price(
    option_type: OptionType,
    cfg: &AmericanMlConfig,
) -> Result<AmericanMlReport, OxisError> {
    american_report(
        "deep-lsm",
        option_type,
        cfg,
        deep_lsm_american(option_type, cfg)?,
    )
}

/// Price an American option by Deep Optimal Stopping and report it against the
/// binomial baseline.
///
/// # Errors
/// Propagates pricing errors ([`OxisError::InvalidInput`]).
pub fn dos_price(
    option_type: OptionType,
    cfg: &AmericanMlConfig,
) -> Result<AmericanMlReport, OxisError> {
    american_report("dos", option_type, cfg, dos_american(option_type, cfg)?)
}

/// Assemble an [`AmericanMlReport`]: pair a neural estimate with the 2000-step CRR
/// American tree (the QuantLib-validated baseline) for the same contract.
fn american_report(
    method: &'static str,
    option_type: OptionType,
    cfg: &AmericanMlConfig,
    est: McEstimate,
) -> Result<AmericanMlReport, OxisError> {
    let tree = binomial(
        option_type,
        ExerciseStyle::American,
        &cfg.market,
        cfg.strike,
        cfg.expiry,
        2000,
    )?;
    Ok(AmericanMlReport {
        method,
        option_type: option_type.as_str(),
        spot: cfg.market.spot,
        strike: cfg.strike,
        rate: cfg.market.rate,
        vol: cfg.market.volatility,
        maturity: cfg.expiry,
        ml_price: est.price,
        standard_error: est.standard_error,
        binomial_price: tree,
        abs_err: (est.price - tree).abs(),
        paths: cfg.paths,
        steps: cfg.steps,
        epochs: cfg.epochs,
    })
}

/// Black-Scholes price and delta for a spec (zero dividend yield).
pub(crate) fn black_scholes_price_delta(spec: &BsSpec) -> Result<(f64, f64), OxisError> {
    let opt = EuropeanOption {
        strike: spec.strike,
        expiry_years: spec.maturity,
        option_type: spec.option_type,
    };
    let mkt = MarketData::new(spec.spot, spec.rate, spec.vol, 0.0);
    let price = black_scholes(&opt, &mkt)?;
    let delta = analytic_greeks(&opt, &mkt)?.delta;
    Ok((price, delta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::OptionType;

    #[test]
    fn report_columns_and_cells_align() {
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
            seed: 9,
        };
        let report = differential_ml_price(&cfg).unwrap();
        assert_eq!(report.columns().len(), report.cells().len());
        assert!(report.bs_price > 0.0);
        assert!(report.ml_price.is_finite() && report.ml_delta.is_finite());
    }
}
