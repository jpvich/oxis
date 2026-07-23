//! Black-Scholes-Merton closed-form pricing for European options.
//!
//! Method (with continuous dividend yield `q`):
//! ```text
//! d1 = (ln(S/K) + (r - q + σ²/2)·T) / (σ·√T)
//! d2 = d1 - σ·√T
//! Call = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
//! Put  = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)
//! ```
//! Edge cases (`T=0`, `σ=0`, `S=0`) are handled as exact mathematical limits so
//! the function never returns `NaN`/`Inf` or panics.

use crate::core::math::normal_cdf;
use crate::core::{EuropeanOption, MarketData, OptionType, OxisError};

/// Price a European option with the Black-Scholes-Merton formula.
///
/// Returns the option's present value. Errors on inputs outside the model's
/// domain (negative volatility/time, non-positive strike, negative spot).
pub fn black_scholes(option: &EuropeanOption, market: &MarketData) -> Result<f64, OxisError> {
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

    // T = 0: value is the intrinsic payoff.
    if t == 0.0 {
        return Ok(option_type.intrinsic(s, k));
    }

    let disc_r = (-r * t).exp(); // discount factor on the strike
    let disc_q = (-q * t).exp(); // discount factor on the spot (carry)

    // S = 0: a call is worthless; a put is the discounted strike.
    if s == 0.0 {
        return Ok(match option_type {
            OptionType::Call => 0.0,
            OptionType::Put => k * disc_r,
        });
    }

    // σ = 0: the underlying grows deterministically at (r - q); the payoff is the
    // discounted intrinsic value taken on the forward.
    if sigma == 0.0 {
        let forward = s * ((r - q) * t).exp();
        let intrinsic_fwd = option_type.intrinsic(forward, k);
        return Ok(disc_r * intrinsic_fwd);
    }

    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;

    let price = match option_type {
        OptionType::Call => s * disc_q * normal_cdf(d1) - k * disc_r * normal_cdf(d2),
        OptionType::Put => k * disc_r * normal_cdf(-d2) - s * disc_q * normal_cdf(-d1),
    };
    Ok(price)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opt(option_type: OptionType) -> EuropeanOption {
        EuropeanOption {
            strike: 100.0,
            expiry_years: 1.0,
            option_type,
        }
    }
    fn mkt() -> MarketData {
        MarketData::new(100.0, 0.05, 0.2, 0.0)
    }

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b} (tol {tol})");
    }

    #[test]
    fn matches_textbook_values() {
        // S=K=100, r=5%, σ=20%, T=1, q=0: standard reference values.
        close(
            black_scholes(&opt(OptionType::Call), &mkt()).unwrap(),
            10.450_583_572,
            1e-6,
        );
        close(
            black_scholes(&opt(OptionType::Put), &mkt()).unwrap(),
            5.573_526_022,
            1e-6,
        );
    }

    #[test]
    fn put_call_parity_holds() {
        // C - P = S·e^(-qT) - K·e^(-rT)
        let m = MarketData::new(120.0, 0.03, 0.35, 0.01);
        let call = black_scholes(
            &EuropeanOption {
                strike: 110.0,
                expiry_years: 0.75,
                option_type: OptionType::Call,
            },
            &m,
        )
        .unwrap();
        let put = black_scholes(
            &EuropeanOption {
                strike: 110.0,
                expiry_years: 0.75,
                option_type: OptionType::Put,
            },
            &m,
        )
        .unwrap();
        let lhs = call - put;
        let rhs = m.spot * (-m.dividend_yield * 0.75).exp() - 110.0 * (-m.rate * 0.75).exp();
        close(lhs, rhs, 1e-10);
    }

    #[test]
    fn edge_case_zero_time_is_intrinsic() {
        let m = MarketData::new(120.0, 0.05, 0.2, 0.0);
        let c = EuropeanOption {
            strike: 100.0,
            expiry_years: 0.0,
            option_type: OptionType::Call,
        };
        assert_eq!(black_scholes(&c, &m).unwrap(), 20.0);
        let p = EuropeanOption {
            strike: 100.0,
            expiry_years: 0.0,
            option_type: OptionType::Put,
        };
        assert_eq!(black_scholes(&p, &m).unwrap(), 0.0);
    }

    #[test]
    fn edge_case_zero_vol_is_discounted_forward_intrinsic() {
        let m = MarketData::new(100.0, 0.05, 0.0, 0.0);
        let c = EuropeanOption {
            strike: 100.0,
            expiry_years: 1.0,
            option_type: OptionType::Call,
        };
        // forward = 100·e^0.05 = 105.127..., discounted intrinsic = e^-0.05·5.127...
        let expected = (-0.05_f64).exp() * (100.0 * 0.05_f64.exp() - 100.0);
        close(black_scholes(&c, &m).unwrap(), expected, 1e-12);
    }

    #[test]
    fn edge_case_zero_spot() {
        let m = MarketData::new(0.0, 0.05, 0.2, 0.0);
        let c = EuropeanOption {
            strike: 100.0,
            expiry_years: 1.0,
            option_type: OptionType::Call,
        };
        assert_eq!(black_scholes(&c, &m).unwrap(), 0.0);
        let p = EuropeanOption {
            strike: 100.0,
            expiry_years: 1.0,
            option_type: OptionType::Put,
        };
        close(
            black_scholes(&p, &m).unwrap(),
            100.0 * (-0.05_f64).exp(),
            1e-12,
        );
    }

    #[test]
    fn call_increases_with_spot() {
        let o = opt(OptionType::Call);
        let lo = black_scholes(&o, &MarketData::new(90.0, 0.05, 0.2, 0.0)).unwrap();
        let hi = black_scholes(&o, &MarketData::new(110.0, 0.05, 0.2, 0.0)).unwrap();
        assert!(hi > lo);
    }

    #[test]
    fn rejects_invalid_inputs() {
        let m = mkt();
        assert!(
            black_scholes(
                &EuropeanOption {
                    strike: 0.0,
                    expiry_years: 1.0,
                    option_type: OptionType::Call
                },
                &m
            )
            .is_err()
        );
        let bad = MarketData::new(100.0, 0.05, -0.2, 0.0);
        assert!(black_scholes(&opt(OptionType::Call), &bad).is_err());
    }
}
