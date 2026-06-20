//! Closed-form Black-Scholes-Merton Greeks for European options.
//!
//! Conventions (chosen to match QuantLib's `AnalyticEuropeanEngine`, so the
//! validation cross-check is apples-to-apples):
//!
//! - **delta** `∂V/∂S` — per unit spot.
//! - **gamma** `∂²V/∂S²` — per unit spot, squared.
//! - **vega**  `∂V/∂σ` — per **unit** volatility (i.e. a move of `1.00` = 100
//!   vol points), *not* per 1%.
//! - **theta** `∂V/∂t` — per **year** (calendar), i.e. `-∂V/∂T`. Divide by 365
//!   for per-day theta.
//! - **rho**   `∂V/∂r` — per **unit** rate (a move of `1.00` = 100bp·100), not
//!   per 1%.
//!
//! Edge cases (`T=0`, `σ=0`, `S=0`) return finite limiting values rather than
//! `NaN`/`Inf`.

use oxis_core::{EuropeanOption, MarketData, OptionType, OxisError, normal_cdf, normal_pdf};
use serde::Serialize;

/// The five first/second-order sensitivities of an option price.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Greeks {
    /// `∂V/∂S` (per unit spot).
    pub delta: f64,
    /// `∂²V/∂S²` (per unit spot squared).
    pub gamma: f64,
    /// `∂V/∂σ` (per unit volatility).
    pub vega: f64,
    /// `∂V/∂t` (per year; `-∂V/∂T`).
    pub theta: f64,
    /// `∂V/∂r` (per unit rate).
    pub rho: f64,
}

/// Closed-form Greeks for a European option under Black-Scholes-Merton.
///
/// # Errors
/// [`OxisError::InvalidInput`] for inputs outside the model's domain
/// (non-positive strike, negative spot/vol/time).
pub fn analytic_greeks(option: &EuropeanOption, market: &MarketData) -> Result<Greeks, OxisError> {
    let EuropeanOption {
        strike: k,
        expiry_years: t,
        option_type,
    } = *option;
    let MarketData {
        spot: s,
        rate: r,
        volatility: sigma,
        dividend_yield: q,
    } = *market;

    if k <= 0.0 {
        return Err(OxisError::invalid_input("strike must be > 0"));
    }
    if s < 0.0 {
        return Err(OxisError::invalid_input("spot must be >= 0"));
    }
    if sigma < 0.0 {
        return Err(OxisError::invalid_input("volatility must be >= 0"));
    }
    if t < 0.0 {
        return Err(OxisError::invalid_input("time to expiry must be >= 0"));
    }

    // Degenerate limits: with T=0, σ=0, or S=0 the density collapses and the
    // smooth Greeks are zero except where a kink makes them undefined; we report
    // the well-defined limit (0) rather than NaN. Price-level edge behavior is
    // covered by the pricer itself.
    if t == 0.0 || sigma == 0.0 || s == 0.0 {
        let disc_q = (-q * t).exp();
        // Delta still has a meaningful limit for an in-the-money forward.
        let delta = match (t, s) {
            (0.0, _) => directional_step_delta(option_type, s, k),
            (_, 0.0) => 0.0,
            _ => {
                // σ = 0, T > 0, S > 0: deterministic; delta is the discounted
                // indicator of finishing in the money.
                let forward = s * ((r - q) * t).exp();
                let itm = match option_type {
                    OptionType::Call => forward > k,
                    OptionType::Put => forward < k,
                };
                if itm {
                    match option_type {
                        OptionType::Call => disc_q,
                        OptionType::Put => -disc_q,
                    }
                } else {
                    0.0
                }
            }
        };
        return Ok(Greeks {
            delta,
            gamma: 0.0,
            vega: 0.0,
            theta: 0.0,
            rho: 0.0,
        });
    }

    let sqrt_t = t.sqrt();
    let disc_r = (-r * t).exp();
    let disc_q = (-q * t).exp();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;
    let nd1 = normal_cdf(d1);
    let nd2 = normal_cdf(d2);
    let pdf_d1 = normal_pdf(d1);

    let gamma = disc_q * pdf_d1 / (s * sigma * sqrt_t);
    let vega = s * disc_q * pdf_d1 * sqrt_t;

    let (delta, theta, rho) = match option_type {
        OptionType::Call => {
            let delta = disc_q * nd1;
            let theta = -(s * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_t) - r * k * disc_r * nd2
                + q * s * disc_q * nd1;
            let rho = k * t * disc_r * nd2;
            (delta, theta, rho)
        }
        OptionType::Put => {
            let delta = -disc_q * normal_cdf(-d1);
            let theta = -(s * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_t)
                + r * k * disc_r * normal_cdf(-d2)
                - q * s * disc_q * normal_cdf(-d1);
            let rho = -k * t * disc_r * normal_cdf(-d2);
            (delta, theta, rho)
        }
    };

    Ok(Greeks {
        delta,
        gamma,
        vega,
        theta,
        rho,
    })
}

/// Delta in the `T = 0` limit: the option is at its kink, so we report the
/// one-sided step (0 or ±1) consistent with the intrinsic payoff's slope.
fn directional_step_delta(option_type: OptionType, s: f64, k: f64) -> f64 {
    match option_type {
        OptionType::Call => {
            if s > k {
                1.0
            } else {
                0.0
            }
        }
        OptionType::Put => {
            if s < k {
                -1.0
            } else {
                0.0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b} (tol {tol})");
    }

    #[test]
    fn atm_call_delta_near_half() {
        // ATM-forward call delta is a little above 0.5.
        let g = analytic_greeks(
            &EuropeanOption {
                strike: 100.0,
                expiry_years: 1.0,
                option_type: OptionType::Call,
            },
            &MarketData::new(100.0, 0.05, 0.2, 0.0),
        )
        .unwrap();
        assert!(g.delta > 0.5 && g.delta < 0.8, "delta {}", g.delta);
        assert!(g.gamma > 0.0 && g.vega > 0.0);
    }

    #[test]
    fn call_put_delta_relation() {
        // Call delta - put delta = e^{-qT}.
        let m = MarketData::new(100.0, 0.05, 0.2, 0.03);
        let call = analytic_greeks(
            &EuropeanOption {
                strike: 105.0,
                expiry_years: 0.5,
                option_type: OptionType::Call,
            },
            &m,
        )
        .unwrap();
        let put = analytic_greeks(
            &EuropeanOption {
                strike: 105.0,
                expiry_years: 0.5,
                option_type: OptionType::Put,
            },
            &m,
        )
        .unwrap();
        close(call.delta - put.delta, (-0.03_f64 * 0.5).exp(), 1e-12);
        // Gamma and vega are identical for call and put.
        close(call.gamma, put.gamma, 1e-12);
        close(call.vega, put.vega, 1e-12);
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert!(
            analytic_greeks(
                &EuropeanOption {
                    strike: -1.0,
                    expiry_years: 1.0,
                    option_type: OptionType::Call,
                },
                &MarketData::new(100.0, 0.05, 0.2, 0.0),
            )
            .is_err()
        );
    }
}
