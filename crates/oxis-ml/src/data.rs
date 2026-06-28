//! Simulated training data for differential ML pricing of a European option.
//!
//! Each sample is one Monte Carlo draw of the discounted payoff **and its pathwise
//! differential** w.r.t. the spot — the extra label that makes differential ML
//! data-efficient. For a single exact GBM step `t → T`,
//! `S_T = S_t·exp((r − σ²/2)τ + σ√τ z)`:
//!
//! - value label `y = e^{−rτ}·payoff(S_T)`,
//! - call differential `q = ∂y/∂S_t = e^{−rτ}·1{S_T ≥ K}·(S_T / S_t)`
//!   (indicator × elasticity), put `q = −e^{−rτ}·1{S_T < K}·(S_T / S_t)`.
//!
//! Input spots `S_t` are spread around `S0` (log-normally, with a widened spread)
//! so the trained surface covers the validation grid. Sampling is seeded per index
//! via [`oxis_core::path_seed`], so a `(seed, n)` fixes the whole training set —
//! the same bit-reproducibility discipline as the Monte Carlo pricers.

use oxis_core::{OptionType, OxisError, path_seed};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};

/// The Black-Scholes contract + market a surrogate is trained for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BsSpec {
    /// Reference spot (centre of the training spread and the price query point).
    pub spot: f64,
    /// Strike.
    pub strike: f64,
    /// Continuously compounded risk-free rate.
    pub rate: f64,
    /// Volatility.
    pub vol: f64,
    /// Time to maturity, in years.
    pub maturity: f64,
    /// Call or put.
    pub option_type: OptionType,
}

/// One differential-ML training sample: input spot, value label, and the pathwise
/// differential of the value w.r.t. the input spot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiffSample {
    /// Input spot `S_t`.
    pub x: f64,
    /// Discounted payoff label `y`.
    pub y: f64,
    /// Pathwise differential label `q = ∂y/∂S_t`.
    pub dydx: f64,
}

/// Generate `n` differential samples for `spec`. `spread` widens the log-normal
/// distribution of input spots (a multiple of the model's own `σ√τ`) so the
/// trained surface covers a band around `spot`.
///
/// # Errors
/// [`OxisError::InvalidInput`] for a non-positive `n`, strike, or maturity, or a
/// negative volatility.
pub fn generate_european(
    spec: &BsSpec,
    n: usize,
    spread: f64,
    seed: u64,
) -> Result<Vec<DiffSample>, OxisError> {
    if n == 0 {
        return Err(OxisError::invalid_input("n_samples must be >= 1"));
    }
    if spec.strike <= 0.0 || spec.spot <= 0.0 {
        return Err(OxisError::invalid_input("spot and strike must be > 0"));
    }
    if spec.maturity <= 0.0 {
        return Err(OxisError::invalid_input("maturity must be > 0"));
    }
    if spec.vol < 0.0 {
        return Err(OxisError::invalid_input("volatility must be >= 0"));
    }

    let tau = spec.maturity;
    let sqrt_tau = tau.sqrt();
    let disc = (-spec.rate * tau).exp();
    let drift = (spec.rate - 0.5 * spec.vol * spec.vol) * tau;
    let vol_step = spec.vol * sqrt_tau;
    // Log-normal spread of input spots around `spot`, centred (median = spot).
    let input_sd = (spread * spec.vol * sqrt_tau).max(1e-6);

    let samples = (0..n)
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(path_seed(seed, i));
            let z_spot: f64 = StandardNormal.sample(&mut rng);
            let z_step: f64 = StandardNormal.sample(&mut rng);

            let s_t = spec.spot * (input_sd * z_spot - 0.5 * input_sd * input_sd).exp();
            let s_t_final = (drift + vol_step * z_step).exp();
            let s_big_t = s_t * s_t_final;

            let payoff = spec.option_type.intrinsic(s_big_t, spec.strike);
            let y = disc * payoff;
            // Pathwise differential: indicator that the option finished in the money
            // times the elasticity S_T / S_t (the derivative of S_T w.r.t. S_t).
            let in_money = match spec.option_type {
                OptionType::Call => s_big_t >= spec.strike,
                OptionType::Put => s_big_t < spec.strike,
            };
            let sign = match spec.option_type {
                OptionType::Call => 1.0,
                OptionType::Put => -1.0,
            };
            let dydx = if in_money {
                sign * disc * (s_big_t / s_t)
            } else {
                0.0
            };
            DiffSample { x: s_t, y, dydx }
        })
        .collect();

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxis_core::{EuropeanOption, MarketData};
    use oxis_greeks::analytic_greeks;
    use oxis_pricing::black_scholes;

    fn spec(option_type: OptionType) -> BsSpec {
        BsSpec {
            spot: 100.0,
            strike: 100.0,
            rate: 0.05,
            vol: 0.2,
            maturity: 1.0,
            option_type,
        }
    }

    #[test]
    fn rejects_bad_inputs() {
        assert!(generate_european(&spec(OptionType::Call), 0, 1.0, 1).is_err());
        let mut bad = spec(OptionType::Call);
        bad.maturity = 0.0;
        assert!(generate_european(&bad, 10, 1.0, 1).is_err());
    }

    #[test]
    fn deterministic() {
        let a = generate_european(&spec(OptionType::Call), 1000, 1.5, 42).unwrap();
        let b = generate_european(&spec(OptionType::Call), 1000, 1.5, 42).unwrap();
        assert_eq!(a, b);
    }

    /// Samples drawn at (approximately) the reference spot must average to the BS
    /// price and delta — the differential labels are an unbiased delta estimator.
    #[test]
    fn labels_unbiased_near_spot() {
        let s = spec(OptionType::Call);
        // Narrow spread so most input spots sit near `spot`; large N for a tight SE.
        let data = generate_european(&s, 400_000, 0.02, 7).unwrap();
        let n = data.len() as f64;
        let y_bar: f64 = data.iter().map(|d| d.y).sum::<f64>() / n;
        let q_bar: f64 = data.iter().map(|d| d.dydx).sum::<f64>() / n;

        let opt = EuropeanOption {
            strike: s.strike,
            expiry_years: s.maturity,
            option_type: s.option_type,
        };
        let mkt = MarketData::new(s.spot, s.rate, s.vol, 0.0);
        let bs = black_scholes(&opt, &mkt).unwrap();
        let delta = analytic_greeks(&opt, &mkt).unwrap().delta;

        // Loose bands: a narrow but non-zero spread biases these slightly.
        assert!((y_bar - bs).abs() < 0.4, "y_bar {y_bar} vs bs {bs}");
        assert!(
            (q_bar - delta).abs() < 0.03,
            "q_bar {q_bar} vs delta {delta}"
        );
    }
}
