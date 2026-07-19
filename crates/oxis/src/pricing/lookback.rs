//! Continuous lookback options under Black-Scholes — closed form.
//!
//! Prices the **floating-strike** lookback (Goldman-Sosin-Gatto) and the
//! **fixed-strike** lookback (Conze-Viswanathan) for a freshly issued option, i.e.
//! the realized running extremum equals the current spot at inception. Monitoring
//! is continuous. Validated against QuantLib's
//! `AnalyticContinuousFloatingLookbackEngine` / `…FixedLookback…`.
//!
//! - Floating-strike **call** pays `S_T − min S`; **put** pays `max S − S_T`.
//! - Fixed-strike **call** pays `max(max S − K, 0)`; **put** pays `max(K − min S, 0)`.

use crate::core::{EuropeanOption, MarketData, OptionType, OxisError, normal_cdf};

/// Whether the strike floats with the realized extremum or is fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LookbackStrike {
    /// Floating strike: payoff measured against the realized min/max.
    Floating,
    /// Fixed strike `K`: payoff is `max(extremum − K, 0)` style.
    Fixed,
}

impl LookbackStrike {
    /// A short, stable identifier for output / the CLI.
    pub fn as_str(self) -> &'static str {
        match self {
            LookbackStrike::Floating => "floating",
            LookbackStrike::Fixed => "fixed",
        }
    }
}

/// Price a continuous lookback option for a freshly issued contract (realized
/// extremum = spot).
///
/// For [`LookbackStrike::Floating`] the `option.strike` is ignored (the payoff
/// references the realized extremum).
///
/// # Errors
/// [`OxisError::InvalidInput`] for non-positive spot/strike or negative
/// volatility/expiry.
pub fn lookback_price(
    option: &EuropeanOption,
    market: &MarketData,
    strike_type: LookbackStrike,
) -> Result<f64, OxisError> {
    let (s, k) = (market.spot, option.strike);
    let (r, q, sigma, t) = (
        market.rate,
        market.dividend_yield,
        market.volatility,
        option.expiry_years,
    );
    if s <= 0.0 || (strike_type == LookbackStrike::Fixed && k <= 0.0) {
        return Err(OxisError::invalid_input("spot and strike must be > 0"));
    }
    if sigma < 0.0 || t < 0.0 {
        return Err(OxisError::invalid_input(
            "volatility and expiry must be >= 0",
        ));
    }
    // Deterministic limit: no diffusion, so the realized extremum is the
    // deterministic terminal/initial value and the payoff collapses to intrinsic.
    if t == 0.0 || sigma == 0.0 {
        return Ok(deterministic_lookback(
            option.option_type,
            strike_type,
            s,
            k,
            r,
            q,
            t,
        ));
    }

    // Avoid the removable singularity at cost-of-carry b = 0 (the σ²/(2b) terms).
    let b_raw = r - q;
    let b = if b_raw.abs() < 1e-8 {
        if b_raw < 0.0 { -1e-8 } else { 1e-8 }
    } else {
        b_raw
    };

    let price = match strike_type {
        LookbackStrike::Floating => floating(option.option_type, s, b, r, sigma, t),
        LookbackStrike::Fixed => fixed(option.option_type, s, k, b, r, sigma, t),
    };
    Ok(price.max(0.0))
}

/// Goldman-Sosin-Gatto floating-strike lookback (extremum = spot).
///
/// `b = r − q` is the cost of carry. `drift_adj = (2b/σ²)·σ√T` is the term Haug's
/// formulas subtract from / add to the normal arguments; `power = −2b/σ²` is the
/// reflection exponent (`(S/extremum)^power`, which is `1` for a fresh option).
fn floating(option_type: OptionType, s: f64, b: f64, r: f64, sigma: f64, t: f64) -> f64 {
    let v = sigma * t.sqrt();
    let carry = ((b - r) * t).exp();
    let disc = (-r * t).exp();
    let coef = sigma * sigma / (2.0 * b);
    let drift_adj = (2.0 * b / (sigma * sigma)) * v;
    let bt = (b * t).exp();

    match option_type {
        // Call pays S_T − min; fresh ⇒ min = S, so ln(S/min) = 0.
        OptionType::Call => {
            let a1 = (b + 0.5 * sigma * sigma) * t / v;
            let a2 = a1 - v;
            s * carry * normal_cdf(a1) - s * disc * normal_cdf(a2)
                + s * disc * coef * (normal_cdf(-a1 + drift_adj) - bt * normal_cdf(-a1))
        }
        // Put pays max − S_T; fresh ⇒ max = S, so ln(S/max) = 0.
        OptionType::Put => {
            let b1 = (b + 0.5 * sigma * sigma) * t / v;
            let b2 = b1 - v;
            s * disc * normal_cdf(-b2) - s * carry * normal_cdf(-b1)
                + s * disc * coef * (-normal_cdf(b1 - drift_adj) + bt * normal_cdf(b1))
        }
    }
}

/// Conze-Viswanathan fixed-strike lookback (extremum = spot).
fn fixed(option_type: OptionType, s: f64, k: f64, b: f64, r: f64, sigma: f64, t: f64) -> f64 {
    let v = sigma * t.sqrt();
    let carry = ((b - r) * t).exp();
    let disc = (-r * t).exp();
    let coef = sigma * sigma / (2.0 * b);
    let drift_adj = (2.0 * b / (sigma * sigma)) * v;
    let power = -2.0 * b / (sigma * sigma);
    let bt = (b * t).exp();
    // Fresh option: realized max (call) and min (put) both equal the spot S.
    let m = s;

    match option_type {
        OptionType::Call => {
            if k >= m {
                let d1 = ((s / k).ln() + (b + 0.5 * sigma * sigma) * t) / v;
                let d2 = d1 - v;
                s * carry * normal_cdf(d1) - k * disc * normal_cdf(d2)
                    + s * disc
                        * coef
                        * (-((s / k).powf(power)) * normal_cdf(d1 - drift_adj)
                            + bt * normal_cdf(d1))
            } else {
                let e1 = ((s / m).ln() + (b + 0.5 * sigma * sigma) * t) / v;
                let e2 = e1 - v;
                disc * (m - k) + s * carry * normal_cdf(e1) - m * disc * normal_cdf(e2)
                    + s * disc
                        * coef
                        * (-((s / m).powf(power)) * normal_cdf(e1 - drift_adj)
                            + bt * normal_cdf(e1))
            }
        }
        OptionType::Put => {
            if k <= m {
                let d1 = ((s / k).ln() + (b + 0.5 * sigma * sigma) * t) / v;
                let d2 = d1 - v;
                k * disc * normal_cdf(-d2) - s * carry * normal_cdf(-d1)
                    + s * disc
                        * coef
                        * ((s / k).powf(power) * normal_cdf(-d1 + drift_adj) - bt * normal_cdf(-d1))
            } else {
                let f1 = ((s / m).ln() + (b + 0.5 * sigma * sigma) * t) / v;
                let f2 = f1 - v;
                disc * (k - m) - s * carry * normal_cdf(-f1)
                    + m * disc * normal_cdf(-f2)
                    + s * disc
                        * coef
                        * ((s / m).powf(power) * normal_cdf(-f1 + drift_adj) - bt * normal_cdf(-f1))
            }
        }
    }
}

/// Zero-vol / zero-time limit: extremum equals the (deterministic) spot path.
fn deterministic_lookback(
    option_type: OptionType,
    strike_type: LookbackStrike,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
) -> f64 {
    let s_t = s * ((r - q) * t).exp();
    let disc = (-r * t).exp();
    let (lo, hi) = (s.min(s_t), s.max(s_t));
    match (strike_type, option_type) {
        (LookbackStrike::Floating, OptionType::Call) => disc * (s_t - lo),
        (LookbackStrike::Floating, OptionType::Put) => disc * (hi - s_t),
        (LookbackStrike::Fixed, OptionType::Call) => disc * (hi - k).max(0.0),
        (LookbackStrike::Fixed, OptionType::Put) => disc * (k - lo).max(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mkt() -> MarketData {
        MarketData::new(100.0, 0.06, 0.3, 0.02)
    }

    fn opt(option_type: OptionType, k: f64) -> EuropeanOption {
        EuropeanOption {
            strike: k,
            expiry_years: 1.0,
            option_type,
        }
    }

    #[test]
    fn floating_lookback_is_worth_more_than_vanilla() {
        // A floating-strike lookback call (buy at the lowest) dominates the
        // at-the-money vanilla call, so it must be worth strictly more.
        let m = mkt();
        let lb =
            lookback_price(&opt(OptionType::Call, 100.0), &m, LookbackStrike::Floating).unwrap();
        let vanilla = crate::pricing::black_scholes(&opt(OptionType::Call, 100.0), &m).unwrap();
        assert!(
            lb > vanilla,
            "lookback {lb} should exceed vanilla {vanilla}"
        );
        assert!(lb > 0.0);
    }

    #[test]
    fn fixed_lookback_dominates_vanilla() {
        let m = mkt();
        let lb = lookback_price(&opt(OptionType::Call, 105.0), &m, LookbackStrike::Fixed).unwrap();
        let vanilla = crate::pricing::black_scholes(&opt(OptionType::Call, 105.0), &m).unwrap();
        assert!(
            lb > vanilla,
            "fixed lookback {lb} should exceed vanilla {vanilla}"
        );
    }

    #[test]
    fn rejects_bad_inputs() {
        let m = MarketData::new(-1.0, 0.05, 0.2, 0.0);
        assert!(
            lookback_price(&opt(OptionType::Call, 100.0), &m, LookbackStrike::Floating).is_err()
        );
    }
}
