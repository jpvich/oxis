//! Implied volatility: given a market price, solve for the σ that reproduces it
//! under Black-Scholes.
//!
//! Newton-Raphson on vega (quadratic convergence near the solution), with a
//! Brent bracketing fallback for robustness when Newton stalls (deep
//! ITM/OTM, tiny vega). Both root finders live in `oxis::core::math`.

use crate::core::{Cell, Column, EuropeanOption, MarketData, OptionType, OxisError, Tabular};
use crate::core::{brent, newton, normal_pdf};
use serde::Serialize;

use crate::pricing::black_scholes;

/// The largest volatility the solver will search up to (1000%).
const MAX_VOL: f64 = 10.0;
// Tight residual so that even low-vega (deep ITM/OTM) cases resolve σ well: the
// recovered σ error scales like TOL/vega, so 1e-12 keeps σ accurate to ~1e-6
// even when vega is ~1e-5.
const TOL: f64 = 1e-12;
const MAX_ITER: usize = 100;

/// Solve for the Black-Scholes implied volatility matching `market_price`.
///
/// The `market.volatility` field is ignored (it is the unknown); all other
/// fields (`spot`, `rate`, `dividend_yield`) and the option terms are used.
///
/// # Errors
/// [`OxisError::InvalidInput`] if `market_price` is outside the no-arbitrage
/// bounds for these inputs (no volatility can reproduce it), or
/// [`OxisError::Numerical`] if neither solver converges.
pub fn implied_volatility(
    option: &EuropeanOption,
    market_price: f64,
    market: &MarketData,
) -> Result<f64, OxisError> {
    let EuropeanOption {
        strike: k,
        expiry_years: t,
        option_type,
    } = *option;
    let MarketData {
        spot: s,
        rate: r,
        dividend_yield: q,
        ..
    } = *market;

    if k <= 0.0 {
        return Err(OxisError::invalid_input("strike must be > 0"));
    }
    if s <= 0.0 {
        return Err(OxisError::invalid_input("spot must be > 0"));
    }
    if t <= 0.0 {
        return Err(OxisError::invalid_input(
            "time to expiry must be > 0 to imply volatility",
        ));
    }

    // No-arbitrage bounds: price is monotone in σ, from the σ→0 limit
    // (discounted forward intrinsic) up to the σ→∞ limit.
    let disc_r = (-r * t).exp();
    let disc_q = (-q * t).exp();
    let forward = s * ((r - q) * t).exp();
    let lower = disc_r * option_type.intrinsic(forward, k);
    let upper = match option_type {
        OptionType::Call => s * disc_q,
        OptionType::Put => k * disc_r,
    };
    if market_price < lower - 1e-12 || market_price > upper + 1e-12 {
        return Err(OxisError::invalid_input(format!(
            "price {market_price} outside no-arbitrage bounds [{lower}, {upper}]"
        )));
    }

    let price_at = |sigma: f64| -> Result<f64, OxisError> {
        let opt = EuropeanOption {
            strike: k,
            expiry_years: t,
            option_type,
        };
        let mkt = MarketData::new(s, r, sigma, q);
        black_scholes(&opt, &mkt)
    };

    // Closed-form vega (per unit vol) at sigma — Newton's derivative.
    let vega_at = |sigma: f64| -> f64 {
        let sqrt_t = t.sqrt();
        let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
        s * disc_q * normal_pdf(d1) * sqrt_t
    };

    // Brenner-Subrahmanyam ATM seed, clamped to a sane range.
    let seed = ((2.0 * std::f64::consts::PI / t).sqrt() * market_price / s).clamp(1e-3, 3.0);

    // Newton first.
    if let Ok(sigma) = newton(seed, TOL, MAX_ITER, |sigma| {
        let g = price_at(sigma)
            .map(|p| p - market_price)
            .unwrap_or(f64::NAN);
        (g, vega_at(sigma))
    }) && sigma > 0.0
        && sigma.is_finite()
    {
        return Ok(sigma);
    }

    // Brent fallback over the full search range.
    brent(1e-9, MAX_VOL, TOL, MAX_ITER, |sigma| {
        price_at(sigma)
            .map(|p| p - market_price)
            .unwrap_or(f64::NAN)
    })
}

/// Renderable result of an implied-volatility solve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ImpliedVolResult {
    /// Call or put.
    pub option_type: OptionType,
    /// Spot price input.
    pub spot: f64,
    /// Strike price input.
    pub strike: f64,
    /// Risk-free rate input.
    pub rate: f64,
    /// Time to expiry (years) input.
    pub time: f64,
    /// Dividend yield input.
    pub dividend_yield: f64,
    /// The observed market price the solve targeted.
    pub market_price: f64,
    /// The implied volatility.
    pub implied_volatility: f64,
}

impl ImpliedVolResult {
    /// Assemble a result from inputs and the solved volatility.
    pub fn new(
        option: &EuropeanOption,
        market: &MarketData,
        market_price: f64,
        implied_volatility: f64,
    ) -> Self {
        Self {
            option_type: option.option_type,
            spot: market.spot,
            strike: option.strike,
            rate: market.rate,
            time: option.expiry_years,
            dividend_yield: market.dividend_yield,
            market_price,
            implied_volatility,
        }
    }
}

impl Tabular for ImpliedVolResult {
    fn columns(&self) -> Vec<Column> {
        vec![
            Column::new("option_type"),
            Column::new("spot"),
            Column::new("strike"),
            Column::new("rate"),
            Column::new("time"),
            Column::new("dividend_yield"),
            Column::new("market_price"),
            Column::new("implied_volatility"),
        ]
    }

    fn cells(&self) -> Vec<Cell> {
        vec![
            Cell::str(self.option_type.as_str()),
            Cell::F64(self.spot),
            Cell::F64(self.strike),
            Cell::F64(self.rate),
            Cell::F64(self.time),
            Cell::F64(self.dividend_yield),
            Cell::F64(self.market_price),
            Cell::F64(self.implied_volatility),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b} (tol {tol})");
    }

    fn round_trip(option_type: OptionType, k: f64, t: f64, sigma: f64, q: f64, tol: f64) {
        let option = EuropeanOption {
            strike: k,
            expiry_years: t,
            option_type,
        };
        let market = MarketData::new(100.0, 0.05, sigma, q);
        let price = black_scholes(&option, &market).unwrap();
        let recovered = implied_volatility(&option, price, &market).unwrap();
        close(recovered, sigma, tol);
    }

    #[test]
    fn round_trips_across_inputs() {
        // Broad grid at the spec's ≤1e-6 tolerance. Deep-OTM low-vol cases are
        // inherently conditioning-limited (vega → 0), so 1e-6 is the honest bar.
        for &ot in &[OptionType::Call, OptionType::Put] {
            for &k in &[80.0, 100.0, 125.0] {
                for &sigma in &[0.05, 0.2, 0.6, 1.2] {
                    round_trip(ot, k, 0.75, sigma, 0.0, 1e-6);
                }
            }
        }
    }

    #[test]
    fn round_trips_atm_tightly() {
        // Well-conditioned ATM cases recover σ to near machine precision.
        round_trip(OptionType::Call, 100.0, 1.0, 0.2, 0.0, 1e-9);
        round_trip(OptionType::Put, 100.0, 0.5, 0.35, 0.0, 1e-9);
    }

    #[test]
    fn round_trips_with_dividends() {
        round_trip(OptionType::Call, 105.0, 1.5, 0.3, 0.04, 1e-6);
        round_trip(OptionType::Put, 95.0, 0.25, 0.45, 0.02, 1e-6);
    }

    #[test]
    fn rejects_price_outside_bounds() {
        let option = EuropeanOption {
            strike: 100.0,
            expiry_years: 1.0,
            option_type: OptionType::Call,
        };
        let market = MarketData::new(100.0, 0.05, 0.2, 0.0);
        // Above the upper bound S·e^{-qT} = 100.
        assert!(implied_volatility(&option, 150.0, &market).is_err());
        // Below intrinsic.
        assert!(implied_volatility(&option, -1.0, &market).is_err());
    }

    #[test]
    fn rejects_zero_time() {
        let option = EuropeanOption {
            strike: 100.0,
            expiry_years: 0.0,
            option_type: OptionType::Call,
        };
        let market = MarketData::new(100.0, 0.05, 0.2, 0.0);
        assert!(implied_volatility(&option, 1.0, &market).is_err());
    }
}
