//! Average-price Asian options under Black-Scholes.
//!
//! Two averaging conventions, both fixed-strike "average price" (payoff
//! `max(φ(A − K), 0)` with `A` the average underlying):
//!
//! - **Geometric** average — closed form (Kemna-Vorst): the geometric average of
//!   GBM is itself log-normal, so the price is Black-Scholes with an adjusted
//!   volatility `σ/√3` and cost of carry `½(b − σ²/6)`. Continuous averaging.
//!   Validated against QuantLib's `AnalyticContinuousGeometricAveragePriceAsianEngine`.
//! - **Arithmetic** average — no closed form; priced by Monte Carlo over GBM paths
//!   from [`oxis_stochastic`], averaging the underlying on `n` equally spaced
//!   fixing dates and reporting a standard error. Validated against QuantLib's
//!   `MCDiscreteArithmeticAPEngine` within a combined standard-error band.

use crate::core::{EuropeanOption, MarketData, OptionType, OxisError, mean_and_se, normal_cdf};
use crate::stochastic::{Process, SimConfig, simulate_paths};

use crate::pricing::McEstimate;

/// Closed-form price of a continuous **geometric** average-price Asian option.
///
/// # Errors
/// [`OxisError::InvalidInput`] for non-positive spot/strike or negative
/// volatility/expiry.
pub fn geometric_asian_price(
    option: &EuropeanOption,
    market: &MarketData,
) -> Result<f64, OxisError> {
    let (s, k) = (market.spot, option.strike);
    let (r, q, sigma, t) = (
        market.rate,
        market.dividend_yield,
        market.volatility,
        option.expiry_years,
    );
    if s <= 0.0 || k <= 0.0 {
        return Err(OxisError::invalid_input("spot and strike must be > 0"));
    }
    if sigma < 0.0 || t < 0.0 {
        return Err(OxisError::invalid_input(
            "volatility and expiry must be >= 0",
        ));
    }

    let b = r - q;
    // Geometric-average adjustments: vol shrinks by √3, carry halves with a
    // convexity correction.
    let sigma_g = sigma / 3.0_f64.sqrt();
    let b_g = 0.5 * (b - sigma * sigma / 6.0);

    if t == 0.0 || sigma == 0.0 {
        // Deterministic geometric average of S·e^{b·u} over [0,t] = S·e^{b_g'·t}.
        let avg = s * (0.5 * b * t).exp();
        return Ok((-r * t).exp() * option.option_type.intrinsic(avg, k));
    }

    let v = sigma_g * t.sqrt();
    let d1 = ((s / k).ln() + (b_g + 0.5 * sigma_g * sigma_g) * t) / v;
    let d2 = d1 - v;
    let carry = ((b_g - r) * t).exp();
    let disc = (-r * t).exp();
    let price = match option.option_type {
        OptionType::Call => s * carry * normal_cdf(d1) - k * disc * normal_cdf(d2),
        OptionType::Put => k * disc * normal_cdf(-d2) - s * carry * normal_cdf(-d1),
    };
    Ok(price.max(0.0))
}

/// Monte Carlo price of a discrete **arithmetic** average-price Asian option,
/// averaging on `n_fixings` equally spaced dates `t_i = i·T/n` (`i = 1..n`).
///
/// Paths are simulated under the risk-neutral GBM (`drift = r − q`) by
/// [`oxis_stochastic`]; the antithetic pairing is preserved so the standard error
/// captures the variance reduction.
///
/// # Errors
/// [`OxisError::InvalidInput`] for non-positive spot/strike, negative
/// volatility/expiry, or zero fixings/paths.
pub fn arithmetic_asian_price(
    option: &EuropeanOption,
    market: &MarketData,
    n_fixings: usize,
    cfg: &SimConfig,
) -> Result<McEstimate, OxisError> {
    let (s, k) = (market.spot, option.strike);
    let (r, q, sigma, t) = (
        market.rate,
        market.dividend_yield,
        market.volatility,
        option.expiry_years,
    );
    if s <= 0.0 || k <= 0.0 {
        return Err(OxisError::invalid_input("spot and strike must be > 0"));
    }
    if sigma < 0.0 || t < 0.0 {
        return Err(OxisError::invalid_input(
            "volatility and expiry must be >= 0",
        ));
    }
    if n_fixings == 0 {
        return Err(OxisError::invalid_input("n_fixings must be >= 1"));
    }

    let disc = (-r * t).exp();
    let option_type = option.option_type;

    // Deterministic limit: a single risk-neutral path.
    if t == 0.0 || sigma == 0.0 {
        let mut avg = 0.0;
        for i in 1..=n_fixings {
            let ti = i as f64 * t / n_fixings as f64;
            avg += s * ((r - q) * ti).exp();
        }
        avg /= n_fixings as f64;
        return Ok(McEstimate {
            price: disc * option_type.intrinsic(avg, k),
            standard_error: 0.0,
        });
    }

    // Simulate GBM paths with the grid aligned to the fixing dates.
    let process = Process::Gbm { mu: r - q, sigma };
    let sim_cfg = SimConfig {
        paths: cfg.paths,
        steps: n_fixings,
        seed: cfg.seed,
    };
    let paths = simulate_paths(&process, s, t, &sim_cfg)?;

    // Per-path discounted payoff on the arithmetic average of the fixing dates
    // (path indices 1..=n_fixings; index 0 is S₀).
    let payoff = |path: &[f64]| {
        let avg = path[1..=n_fixings].iter().sum::<f64>() / n_fixings as f64;
        option_type.intrinsic(avg, k)
    };

    // Average within antithetic pairs (paths laid out [up, dn, up, dn, …]).
    let pair_means: Vec<f64> = paths
        .chunks_exact(2)
        .map(|pair| 0.5 * (payoff(&pair[0]) + payoff(&pair[1])))
        .collect();
    let (mean, se) = mean_and_se(&pair_means);

    Ok(McEstimate {
        price: disc * mean,
        standard_error: disc * se,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opt(option_type: OptionType, k: f64) -> EuropeanOption {
        EuropeanOption {
            strike: k,
            expiry_years: 1.0,
            option_type,
        }
    }

    fn mkt() -> MarketData {
        MarketData::new(100.0, 0.05, 0.2, 0.0)
    }

    #[test]
    fn geometric_asian_is_cheaper_than_vanilla() {
        // Averaging dampens volatility, so an average-price call is worth less
        // than the corresponding vanilla European call.
        let o = opt(OptionType::Call, 100.0);
        let m = mkt();
        let asian = geometric_asian_price(&o, &m).unwrap();
        let vanilla = crate::pricing::black_scholes(&o, &m).unwrap();
        assert!(
            asian > 0.0 && asian < vanilla,
            "asian {asian} vanilla {vanilla}"
        );
    }

    #[test]
    fn arithmetic_exceeds_geometric() {
        // Arithmetic mean ≥ geometric mean ⇒ the arithmetic-average call is worth
        // at least the geometric one (within Monte Carlo error).
        let o = opt(OptionType::Call, 100.0);
        let m = mkt();
        let cfg = SimConfig {
            paths: 200_000,
            steps: 0,
            seed: 7,
        };
        let geo = geometric_asian_price(&o, &m).unwrap();
        let arith = arithmetic_asian_price(&o, &m, 50, &cfg).unwrap();
        assert!(
            arith.price + 4.0 * arith.standard_error > geo,
            "arith {} (se {}) vs geo {geo}",
            arith.price,
            arith.standard_error
        );
    }

    #[test]
    fn deterministic_limits_have_zero_error() {
        let o = opt(OptionType::Call, 90.0);
        let m = MarketData::new(100.0, 0.05, 0.0, 0.0);
        let cfg = SimConfig {
            paths: 1000,
            steps: 0,
            seed: 1,
        };
        let est = arithmetic_asian_price(&o, &m, 12, &cfg).unwrap();
        assert_eq!(est.standard_error, 0.0);
        assert!(est.price > 0.0);
    }

    #[test]
    fn rejects_bad_inputs() {
        let m = mkt();
        assert!(geometric_asian_price(&opt(OptionType::Call, -1.0), &m).is_err());
        let cfg = SimConfig::default();
        assert!(arithmetic_asian_price(&opt(OptionType::Call, 100.0), &m, 0, &cfg).is_err());
    }
}
