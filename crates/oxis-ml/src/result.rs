//! Renderable result for the ML pricing surfaces.
//!
//! [`MlPricingReport`] places the differential-ML estimate next to the classical
//! Black-Scholes value — the whole point of OXIS's ML ring is that the learned
//! price is *measured against* a trusted engine. It derives `Serialize` and
//! implements [`Tabular`] as a single record.

use crate::data::BsSpec;
use crate::train::{TrainConfig, train_differential};
use oxis_core::{Cell, Column, EuropeanOption, MarketData, OxisError, Tabular};
use oxis_greeks::analytic_greeks;
use oxis_pricing::black_scholes;
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
    use oxis_core::OptionType;

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
