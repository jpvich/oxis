//! Finite-difference Greeks — a general fallback for any European pricer.
//!
//! Generic over a pricing function `Fn(&EuropeanOption, &MarketData) ->
//! Result<f64>`, so this module depends only on `oxis::core` (never on
//! `oxis::pricing`): the caller passes in whichever pricer it wants (closed-form,
//! binomial-European, ...), preserving the one-way module→core rule. Used to
//! check the analytic Greeks and to provide Greeks for engines without
//! closed-form derivatives.
//!
//! Central differences throughout. Bump sizes (documented, overridable via
//! [`Bumps`]): spot is bumped **relatively** (`h_S = S·1e-4`) since its scale
//! varies; volatility, rate, and time are bumped **absolutely** (`1e-4`). Theta
//! is reported per year as `-∂V/∂T` to match the analytic convention.

use crate::core::{EuropeanOption, MarketData, OxisError};

use crate::greeks::analytic::Greeks;

/// Finite-difference bump sizes. [`Bumps::default`] matches the documented
/// defaults; override for accuracy/stability studies.
#[derive(Debug, Clone, Copy)]
pub struct Bumps {
    /// Relative bump applied to spot (`h_S = S · spot_rel`).
    pub spot_rel: f64,
    /// Absolute bump applied to volatility.
    pub vol: f64,
    /// Absolute bump applied to the rate.
    pub rate: f64,
    /// Absolute bump applied to time to expiry.
    pub time: f64,
}

impl Default for Bumps {
    fn default() -> Self {
        Self {
            spot_rel: 1e-4,
            vol: 1e-4,
            rate: 1e-4,
            time: 1e-4,
        }
    }
}

/// Compute Greeks by central finite differences around `(option, market)`,
/// re-pricing with `price_fn`.
///
/// # Errors
/// Propagates any [`OxisError`] from `price_fn` at a bumped point.
pub fn finite_diff_greeks<F>(
    option: &EuropeanOption,
    market: &MarketData,
    price_fn: F,
) -> Result<Greeks, OxisError>
where
    F: Fn(&EuropeanOption, &MarketData) -> Result<f64, OxisError>,
{
    finite_diff_greeks_with(option, market, &Bumps::default(), price_fn)
}

/// As [`finite_diff_greeks`], with explicit bump sizes.
///
/// # Errors
/// Propagates any [`OxisError`] from `price_fn` at a bumped point.
pub fn finite_diff_greeks_with<F>(
    option: &EuropeanOption,
    market: &MarketData,
    bumps: &Bumps,
    price_fn: F,
) -> Result<Greeks, OxisError>
where
    F: Fn(&EuropeanOption, &MarketData) -> Result<f64, OxisError>,
{
    let with_market = |f: &dyn Fn(&mut MarketData)| -> Result<f64, OxisError> {
        let mut m = *market;
        f(&mut m);
        price_fn(option, &m)
    };
    let with_option = |f: &dyn Fn(&mut EuropeanOption)| -> Result<f64, OxisError> {
        let mut o = *option;
        f(&mut o);
        price_fn(&o, market)
    };

    let h_s = (market.spot.abs() * bumps.spot_rel).max(bumps.spot_rel);
    let base = price_fn(option, market)?;

    let v_s_up = with_market(&|m| m.spot += h_s)?;
    let v_s_dn = with_market(&|m| m.spot -= h_s)?;
    let delta = (v_s_up - v_s_dn) / (2.0 * h_s);
    let gamma = (v_s_up - 2.0 * base + v_s_dn) / (h_s * h_s);

    let hv = bumps.vol;
    let v_v_up = with_market(&|m| m.volatility += hv)?;
    let v_v_dn = with_market(&|m| m.volatility -= hv)?;
    let vega = (v_v_up - v_v_dn) / (2.0 * hv);

    let hr = bumps.rate;
    let v_r_up = with_market(&|m| m.rate += hr)?;
    let v_r_dn = with_market(&|m| m.rate -= hr)?;
    let rho = (v_r_up - v_r_dn) / (2.0 * hr);

    // theta = ∂V/∂t = -∂V/∂T. Bump time to expiry, shrink near zero so we never
    // step to negative time.
    let ht = bumps.time.min(option.expiry_years / 2.0).max(0.0);
    let theta = if ht > 0.0 {
        let v_t_up = with_option(&|o| o.expiry_years += ht)?;
        let v_t_dn = with_option(&|o| o.expiry_years -= ht)?;
        -(v_t_up - v_t_dn) / (2.0 * ht)
    } else {
        0.0
    };

    Ok(Greeks {
        delta,
        gamma,
        vega,
        theta,
        rho,
    })
}

#[cfg(test)]
mod tests {
    // Cross-checks of finite-diff vs analytic live in lib.rs (they need a pricer
    // from oxis::pricing as a dev-dependency).
}
