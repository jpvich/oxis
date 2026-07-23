//! Single-barrier options under Black-Scholes — closed form (Reiner-Rubinstein).
//!
//! Prices the eight continuously monitored single-barrier types (down/up ×
//! in/out × call/put) with **zero rebate**, using the standard Reiner-Rubinstein
//! building blocks `A, B, C, D` (Haug, *The Complete Guide to Option Pricing
//! Formulas*). Only the four *knock-in* prices are coded directly; each *knock-out*
//! is recovered from the exact European parity `in + out = vanilla` (valid for a
//! zero-rebate continuously monitored barrier), reusing the QuantLib-validated
//! [`crate::pricing::black_scholes`] for the vanilla leg. Validated against QuantLib's
//! `AnalyticBarrierEngine`.

use crate::core::{EuropeanOption, MarketData, OptionType, OxisError, normal_cdf};

use crate::pricing::black_scholes;

/// Which barrier, by direction and knock sense.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierType {
    /// Knocked **in** if the spot falls to the (lower) barrier.
    DownIn,
    /// Knocked **out** if the spot falls to the (lower) barrier.
    DownOut,
    /// Knocked **in** if the spot rises to the (upper) barrier.
    UpIn,
    /// Knocked **out** if the spot rises to the (upper) barrier.
    UpOut,
}

impl BarrierType {
    /// A short, stable identifier for output / the CLI.
    pub fn as_str(self) -> &'static str {
        match self {
            BarrierType::DownIn => "down-in",
            BarrierType::DownOut => "down-out",
            BarrierType::UpIn => "up-in",
            BarrierType::UpOut => "up-out",
        }
    }

    fn is_down(self) -> bool {
        matches!(self, BarrierType::DownIn | BarrierType::DownOut)
    }

    fn is_in(self) -> bool {
        matches!(self, BarrierType::DownIn | BarrierType::UpIn)
    }
}

/// Price a continuously monitored single-barrier option (zero rebate).
///
/// `barrier` is the barrier level `H`; the remaining inputs come from `market`
/// (spot, rate, volatility, dividend yield) and `option` (strike, expiry, call/put).
///
/// # Errors
/// [`OxisError::InvalidInput`] for non-positive spot/strike/barrier or negative
/// volatility/expiry.
pub fn barrier_price(
    option: &EuropeanOption,
    market: &MarketData,
    barrier_type: BarrierType,
    barrier: f64,
) -> Result<f64, OxisError> {
    let (s, k, h) = (market.spot, option.strike, barrier);
    let (r, q, sigma, t) = (
        market.rate,
        market.dividend_yield,
        market.volatility,
        option.expiry_years,
    );
    if !(s > 0.0 && k > 0.0 && h > 0.0) {
        return Err(OxisError::invalid_input(
            "spot, strike, and barrier must be > 0",
        ));
    }
    if sigma < 0.0 || t < 0.0 {
        return Err(OxisError::invalid_input(
            "volatility and expiry must be >= 0",
        ));
    }

    // Deterministic limits (no diffusion to interact with the barrier).
    if t == 0.0 || sigma == 0.0 {
        return Ok(deterministic_barrier(
            option.option_type,
            barrier_type,
            s,
            k,
            h,
            r,
            q,
            sigma,
            t,
        ));
    }

    // Spot already on the barrier's far side → the barrier is touched at t = 0, so
    // the Reiner-Rubinstein formula (which assumes the live side) does not apply.
    let already_touched = if barrier_type.is_down() {
        s <= h
    } else {
        s >= h
    };
    if already_touched {
        return Ok(if barrier_type.is_in() {
            black_scholes(option, market)?
        } else {
            0.0
        });
    }

    let knock_in = knock_in_price(option.option_type, barrier_type, s, k, h, r, q, sigma, t);
    if barrier_type.is_in() {
        Ok(knock_in)
    } else {
        // out = vanilla − in (exact parity for a zero-rebate European barrier).
        let vanilla = black_scholes(option, market)?;
        Ok((vanilla - knock_in).max(0.0))
    }
}

/// The four knock-in prices via Reiner-Rubinstein `A, B, C, D`.
#[allow(clippy::too_many_arguments)]
fn knock_in_price(
    option_type: OptionType,
    barrier_type: BarrierType,
    s: f64,
    k: f64,
    h: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
) -> f64 {
    let b = r - q;
    let sig_sqrt_t = sigma * t.sqrt();
    let mu = (b - 0.5 * sigma * sigma) / (sigma * sigma);
    let phi = match option_type {
        OptionType::Call => 1.0,
        OptionType::Put => -1.0,
    };
    // η: +1 for a down barrier, −1 for an up barrier.
    let eta = if barrier_type.is_down() { 1.0 } else { -1.0 };

    let x1 = (s / k).ln() / sig_sqrt_t + (1.0 + mu) * sig_sqrt_t;
    let x2 = (s / h).ln() / sig_sqrt_t + (1.0 + mu) * sig_sqrt_t;
    let y1 = (h * h / (s * k)).ln() / sig_sqrt_t + (1.0 + mu) * sig_sqrt_t;
    let y2 = (h / s).ln() / sig_sqrt_t + (1.0 + mu) * sig_sqrt_t;

    let carry = ((b - r) * t).exp();
    let disc = (-r * t).exp();
    let pow_a = (h / s).powf(2.0 * (mu + 1.0));
    let pow_b = (h / s).powf(2.0 * mu);

    let term_a = |p: f64| {
        p * s * carry * normal_cdf(p * x1) - p * k * disc * normal_cdf(p * x1 - p * sig_sqrt_t)
    };
    let term_b = |p: f64| {
        p * s * carry * normal_cdf(p * x2) - p * k * disc * normal_cdf(p * x2 - p * sig_sqrt_t)
    };
    let term_c = |p: f64, e: f64| {
        p * s * carry * pow_a * normal_cdf(e * y1)
            - p * k * disc * pow_b * normal_cdf(e * y1 - e * sig_sqrt_t)
    };
    let term_d = |p: f64, e: f64| {
        p * s * carry * pow_a * normal_cdf(e * y2)
            - p * k * disc * pow_b * normal_cdf(e * y2 - e * sig_sqrt_t)
    };

    let strike_above_barrier = k >= h;
    match (option_type, barrier_type.is_down()) {
        // Down-and-in call.
        (OptionType::Call, true) => {
            if strike_above_barrier {
                term_c(phi, eta)
            } else {
                term_a(phi) - term_b(phi) + term_d(phi, eta)
            }
        }
        // Up-and-in call.
        (OptionType::Call, false) => {
            if strike_above_barrier {
                term_a(phi)
            } else {
                term_b(phi) - term_c(phi, eta) + term_d(phi, eta)
            }
        }
        // Down-and-in put.
        (OptionType::Put, true) => {
            if strike_above_barrier {
                term_b(phi) - term_c(phi, eta) + term_d(phi, eta)
            } else {
                term_a(phi)
            }
        }
        // Up-and-in put.
        (OptionType::Put, false) => {
            if strike_above_barrier {
                term_a(phi) - term_b(phi) + term_d(phi, eta)
            } else {
                term_c(phi, eta)
            }
        }
    }
}

/// Zero-vol / zero-time limit: the path is deterministic, so the barrier is
/// either touched along `S → S·e^{(r−q)t}` or not, and the option pays its
/// discounted intrinsic accordingly.
#[allow(clippy::too_many_arguments)]
fn deterministic_barrier(
    option_type: OptionType,
    barrier_type: BarrierType,
    s: f64,
    k: f64,
    h: f64,
    r: f64,
    q: f64,
    _sigma: f64,
    t: f64,
) -> f64 {
    let s_t = s * ((r - q) * t).exp();
    let (lo, hi) = (s.min(s_t), s.max(s_t));
    let touched = if barrier_type.is_down() {
        lo <= h
    } else {
        hi >= h
    };
    let intrinsic_pv = (-r * t).exp() * option_type.intrinsic(s_t, k);
    let knocked_in = touched;
    if barrier_type.is_in() {
        if knocked_in { intrinsic_pv } else { 0.0 }
    } else if knocked_in {
        0.0
    } else {
        intrinsic_pv
    }
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
        MarketData::new(100.0, 0.05, 0.25, 0.0)
    }

    #[test]
    fn in_plus_out_equals_vanilla() {
        // Parity is built in for knock-outs, but verify it holds numerically for
        // every direction/type against the Black-Scholes vanilla.
        for ot in [OptionType::Call, OptionType::Put] {
            for (din, dout) in [
                (BarrierType::DownIn, BarrierType::DownOut),
                (BarrierType::UpIn, BarrierType::UpOut),
            ] {
                let o = opt(ot);
                let m = mkt();
                let h = 90.0;
                let cin = barrier_price(&o, &m, din, h).unwrap();
                let cout = barrier_price(&o, &m, dout, h).unwrap();
                let vanilla = black_scholes(&o, &m).unwrap();
                assert!(
                    (cin + cout - vanilla).abs() < 1e-10,
                    "{ot:?} {din:?}/{dout:?}: in {cin} + out {cout} != vanilla {vanilla}"
                );
            }
        }
    }

    #[test]
    fn far_barrier_limits() {
        let o = opt(OptionType::Call);
        let m = mkt();
        let vanilla = black_scholes(&o, &m).unwrap();
        // A down-and-out call with a barrier far below spot ≈ vanilla.
        let dao = barrier_price(&o, &m, BarrierType::DownOut, 1.0).unwrap();
        assert!((dao - vanilla).abs() < 1e-6, "dao {dao} vanilla {vanilla}");
        // A down-and-in call with that same far barrier ≈ 0.
        let dai = barrier_price(&o, &m, BarrierType::DownIn, 1.0).unwrap();
        assert!(dai < 1e-6, "dai {dai}");
    }

    #[test]
    fn already_knocked_out_is_zero() {
        let o = opt(OptionType::Call);
        let m = mkt();
        // Spot already at/below a down-out barrier → worthless.
        let p = barrier_price(&o, &m, BarrierType::DownOut, 100.0).unwrap();
        assert!(p < 1e-9, "expected ~0, got {p}");
    }

    #[test]
    fn rejects_bad_inputs() {
        let o = opt(OptionType::Call);
        let m = mkt();
        assert!(barrier_price(&o, &m, BarrierType::DownIn, -1.0).is_err());
    }
}
