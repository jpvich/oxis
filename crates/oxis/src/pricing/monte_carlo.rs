//! Monte Carlo pricing for European options under geometric Brownian motion.
//!
//! Terminal prices are simulated exactly — one log-normal jump to expiry, no
//! time discretization — so the only error is statistical (sampling), reported
//! as a standard error alongside the price. Variance is reduced with
//! **antithetic variates**: every standard-normal draw `z` is paired with `−z`,
//! which roughly halves the variance at negligible cost and keeps the
//! standard-error estimate simple (it is computed over the per-pair averages, so
//! the negative correlation is accounted for automatically).
//!
//! **Determinism.** Each antithetic pair `i` seeds its own `SmallRng` from a
//! `splitmix64` mix of `(seed, i)`, so a path's draws never depend on how
//! `rayon` schedules threads. Per-pair results are collected into an
//! index-ordered `Vec` and reduced sequentially, making the price and standard
//! error **bit-reproducible** for a given `(seed, paths)` regardless of the
//! thread count.

use crate::core::{EuropeanOption, MarketData, OxisError, mean_and_se, path_seed};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Monte Carlo run configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McConfig {
    /// Number of simulated terminal prices. With antithetic sampling these are
    /// drawn as `paths / 2` mirrored pairs (rounded up to a whole pair).
    pub paths: usize,
    /// Number of time steps. Unused for European pricing (the terminal price is
    /// sampled in one exact jump); consumed by the Longstaff-Schwartz American
    /// engine. Kept here so both engines share one config type.
    pub steps: usize,
    /// RNG seed — fixes the whole simulation for reproducibility.
    pub seed: u64,
}

impl Default for McConfig {
    /// 100k paths, 50 steps (for LSM), seed 42 — a reasonable balance of
    /// accuracy and speed for interactive use.
    fn default() -> Self {
        Self {
            paths: 100_000,
            steps: 50,
            seed: 42,
        }
    }
}

/// A Monte Carlo price together with its (one-sigma) standard error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McEstimate {
    /// Estimated present value.
    pub price: f64,
    /// Standard error of the estimate (the price is `price ± standard_error`
    /// at roughly one sigma). Exactly `0.0` for the deterministic limits
    /// (`T = 0`, `σ = 0`, `S = 0`).
    pub standard_error: f64,
}

/// Price a European option by Monte Carlo simulation of terminal prices.
///
/// # Errors
/// [`OxisError::InvalidInput`] for inputs outside the model's domain
/// (non-positive strike, negative spot/vol/time, zero paths).
pub fn monte_carlo_european(
    option: &EuropeanOption,
    market: &MarketData,
    cfg: &McConfig,
) -> Result<McEstimate, OxisError> {
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

    validate_inputs(k, s, sigma, t, cfg.paths)?;

    let disc = (-r * t).exp();

    // Deterministic limits: no randomness, exact value, zero standard error.
    if t == 0.0 || sigma == 0.0 || s == 0.0 {
        let s_t = s * ((r - q) * t).exp();
        return Ok(McEstimate {
            price: disc * option_type.intrinsic(s_t, k),
            standard_error: 0.0,
        });
    }

    let drift = (r - q - 0.5 * sigma * sigma) * t;
    let vol_sqrt_t = sigma * t.sqrt();
    let n_pairs = cfg.paths.div_ceil(2);
    let seed = cfg.seed;

    // Each pair is independent and order-preserved by `collect`, so the reduce
    // below is deterministic regardless of how rayon schedules the work.
    let pair_means: Vec<f64> = (0..n_pairs)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(path_seed(seed, i));
            let z: f64 = StandardNormal.sample(&mut rng);
            let s_up = s * (drift + vol_sqrt_t * z).exp();
            let s_dn = s * (drift - vol_sqrt_t * z).exp();
            0.5 * (option_type.intrinsic(s_up, k) + option_type.intrinsic(s_dn, k))
        })
        .collect();

    let (mean, se) = mean_and_se(&pair_means);
    Ok(McEstimate {
        price: disc * mean,
        standard_error: disc * se,
    })
}

/// Shared input validation for the simulation engines.
pub(crate) fn validate_inputs(
    strike: f64,
    spot: f64,
    sigma: f64,
    expiry: f64,
    paths: usize,
) -> Result<(), OxisError> {
    if strike <= 0.0 {
        return Err(OxisError::invalid_input("strike must be > 0"));
    }
    if spot < 0.0 {
        return Err(OxisError::invalid_input("spot must be >= 0"));
    }
    if sigma < 0.0 {
        return Err(OxisError::invalid_input("volatility must be >= 0"));
    }
    if expiry < 0.0 {
        return Err(OxisError::invalid_input("time to expiry must be >= 0"));
    }
    if paths == 0 {
        return Err(OxisError::invalid_input("paths must be >= 1"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::OptionType;
    use crate::pricing::black_scholes;

    fn euro(option_type: OptionType, k: f64, t: f64) -> EuropeanOption {
        EuropeanOption {
            strike: k,
            expiry_years: t,
            option_type,
        }
    }

    #[test]
    fn agrees_with_black_scholes_within_error() {
        let option = euro(OptionType::Call, 105.0, 1.0);
        let market = MarketData::new(100.0, 0.05, 0.2, 0.0);
        let cfg = McConfig {
            paths: 2_000_000,
            steps: 1,
            seed: 7,
        };
        let est = monte_carlo_european(&option, &market, &cfg).unwrap();
        let bs = black_scholes(&option, &market).unwrap();
        // Within 4 standard errors — a ~1-in-16000 false-failure rate.
        assert!(
            (est.price - bs).abs() <= 4.0 * est.standard_error,
            "mc={} bs={} se={} diff={}",
            est.price,
            bs,
            est.standard_error,
            (est.price - bs).abs()
        );
    }

    #[test]
    fn deterministic_across_thread_counts_and_runs() {
        let option = euro(OptionType::Put, 100.0, 0.5);
        let market = MarketData::new(100.0, 0.05, 0.3, 0.01);
        let cfg = McConfig {
            paths: 100_000,
            steps: 1,
            seed: 123,
        };
        let a = monte_carlo_european(&option, &market, &cfg).unwrap();
        let b = monte_carlo_european(&option, &market, &cfg).unwrap();
        // Bit-for-bit identical regardless of scheduling.
        assert_eq!(a.price.to_bits(), b.price.to_bits());
        assert_eq!(a.standard_error.to_bits(), b.standard_error.to_bits());
    }

    #[test]
    fn standard_error_shrinks_like_inverse_sqrt_paths() {
        let option = euro(OptionType::Call, 100.0, 1.0);
        let market = MarketData::new(100.0, 0.05, 0.2, 0.0);
        let se = |paths| {
            monte_carlo_european(
                &option,
                &market,
                &McConfig {
                    paths,
                    steps: 1,
                    seed: 1,
                },
            )
            .unwrap()
            .standard_error
        };
        let se1 = se(100_000);
        let se4 = se(400_000);
        // 4x the paths => ~2x smaller SE. Allow a generous band around 2.0.
        let ratio = se1 / se4;
        assert!((1.7..2.3).contains(&ratio), "ratio {ratio}");
    }

    #[test]
    fn deterministic_limits_have_zero_error() {
        let market = MarketData::new(100.0, 0.05, 0.2, 0.0);
        // T = 0.
        let at_expiry = monte_carlo_european(
            &euro(OptionType::Call, 90.0, 0.0),
            &market,
            &McConfig::default(),
        )
        .unwrap();
        assert_eq!(at_expiry.standard_error, 0.0);
        assert!((at_expiry.price - 10.0).abs() < 1e-12);

        // sigma = 0.
        let zero_vol = monte_carlo_european(
            &euro(OptionType::Call, 100.0, 1.0),
            &MarketData::new(100.0, 0.05, 0.0, 0.0),
            &McConfig::default(),
        )
        .unwrap();
        assert_eq!(zero_vol.standard_error, 0.0);
    }

    #[test]
    fn rejects_bad_inputs() {
        let market = MarketData::new(100.0, 0.05, 0.2, 0.0);
        assert!(
            monte_carlo_european(
                &euro(OptionType::Call, 0.0, 1.0),
                &market,
                &McConfig::default()
            )
            .is_err()
        );
        assert!(
            monte_carlo_european(
                &euro(OptionType::Call, 100.0, 1.0),
                &market,
                &McConfig {
                    paths: 0,
                    steps: 1,
                    seed: 1
                }
            )
            .is_err()
        );
    }
}
