//! Shared American-option scaffolding for the neural engines: GBM path
//! simulation and the deterministic-limit detector.
//!
//! Both neural engines ([`crate::deep_lsm`], [`crate::dos`]) price a 1-D American
//! option by simulating full GBM price paths over a time grid and deciding, at
//! each exercise date, whether to exercise or continue. The simulation here
//! mirrors [`oxis_pricing`]'s Longstaff-Schwartz engine exactly — antithetic
//! pairs, per-pair counter-based seeding, and an ordered/sequential reduction —
//! so prices are bit-reproducible for a given `(seed, paths, steps)` and directly
//! comparable to the classical baseline.
//!
//! `oxis_pricing`'s input validator is private to that crate, so the equivalent
//! checks are replicated in [`validate_inputs`].

use oxis_core::{MarketData, OptionType, OxisError, path_seed};
use oxis_pricing::{McConfig, McEstimate};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// A simulated set of antithetic GBM price paths.
///
/// `paths` is laid out as `[up0, dn0, up1, dn1, …]`; each entry is one full path
/// of length `n_steps + 1` (including `S₀`).
pub(crate) struct PathSet {
    pub(crate) paths: Vec<Vec<f64>>,
    pub(crate) n_steps: usize,
    pub(crate) dt: f64,
}

/// Validate the market/contract/MC inputs shared by the American engines.
///
/// Mirrors `oxis_pricing::monte_carlo::validate_inputs` (which is crate-private)
/// plus the `steps >= 1` check that the path grid requires.
///
/// # Errors
/// [`OxisError::InvalidInput`] for non-positive strike, negative spot/vol/time,
/// or zero paths/steps.
pub(crate) fn validate_inputs(
    strike: f64,
    market: &MarketData,
    expiry: f64,
    cfg: &McConfig,
) -> Result<(), OxisError> {
    if strike <= 0.0 {
        return Err(OxisError::invalid_input("strike must be > 0"));
    }
    if market.spot < 0.0 {
        return Err(OxisError::invalid_input("spot must be >= 0"));
    }
    if market.volatility < 0.0 {
        return Err(OxisError::invalid_input("volatility must be >= 0"));
    }
    if expiry < 0.0 {
        return Err(OxisError::invalid_input("time to expiry must be >= 0"));
    }
    if cfg.paths == 0 {
        return Err(OxisError::invalid_input("paths must be >= 1"));
    }
    if cfg.steps == 0 {
        return Err(OxisError::invalid_input("steps must be >= 1"));
    }
    Ok(())
}

/// The deterministic American value when the path is fixed (`T = 0`, `σ = 0`, or
/// `S = 0`): the best discounted exercise over the time grid, with `SE = 0`.
///
/// Returns `None` in the stochastic case (the engines then simulate).
pub(crate) fn deterministic_american(
    option_type: OptionType,
    market: &MarketData,
    strike: f64,
    expiry: f64,
    cfg: &McConfig,
) -> Option<McEstimate> {
    let MarketData {
        spot: s,
        rate: r,
        volatility: sigma,
        dividend_yield: q,
    } = *market;
    if !(expiry == 0.0 || sigma == 0.0 || s == 0.0) {
        return None;
    }
    let n_steps = cfg.steps.max(1);
    let dt = expiry / n_steps as f64;
    let mut best = option_type.intrinsic(s, strike); // exercise now (t = 0)
    for j in 1..=n_steps {
        let tj = j as f64 * dt;
        let s_tj = s * ((r - q) * tj).exp();
        best = best.max((-r * tj).exp() * option_type.intrinsic(s_tj, strike));
    }
    Some(McEstimate {
        price: best,
        standard_error: 0.0,
    })
}

/// Simulate antithetic GBM price paths under Black-Scholes.
///
/// Each pair is seeded by `path_seed(cfg.seed, i)` and generated in parallel but
/// collected in index order, so the downstream backward induction is
/// deterministic. Assumes the stochastic case (`σ > 0`, `S > 0`, `T > 0`); the
/// engines call [`deterministic_american`] first for the degenerate limits.
///
/// # Errors
/// [`OxisError::InvalidInput`] for zero paths or steps.
pub(crate) fn simulate_paths(
    market: &MarketData,
    expiry: f64,
    cfg: &McConfig,
) -> Result<PathSet, OxisError> {
    if cfg.paths == 0 {
        return Err(OxisError::invalid_input("paths must be >= 1"));
    }
    if cfg.steps == 0 {
        return Err(OxisError::invalid_input("steps must be >= 1"));
    }
    let MarketData {
        spot: s,
        rate: r,
        volatility: sigma,
        dividend_yield: q,
    } = *market;

    let n_steps = cfg.steps;
    let dt = expiry / n_steps as f64;
    let drift_dt = (r - q - 0.5 * sigma * sigma) * dt;
    let vol_sqrt_dt = sigma * dt.sqrt();
    let ln_s0 = s.ln();
    let n_pairs = cfg.paths.div_ceil(2);
    let seed = cfg.seed;

    let paths: Vec<Vec<f64>> = (0..n_pairs)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(path_seed(seed, i));
            let mut up = Vec::with_capacity(n_steps + 1);
            let mut dn = Vec::with_capacity(n_steps + 1);
            up.push(s);
            dn.push(s);
            let (mut ln_up, mut ln_dn) = (ln_s0, ln_s0);
            for _ in 0..n_steps {
                let z: f64 = StandardNormal.sample(&mut rng);
                ln_up += drift_dt + vol_sqrt_dt * z;
                ln_dn += drift_dt - vol_sqrt_dt * z;
                up.push(ln_up.exp());
                dn.push(ln_dn.exp());
            }
            [up, dn]
        })
        .flatten_iter()
        .collect();

    Ok(PathSet { paths, n_steps, dt })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn market() -> MarketData {
        MarketData::new(100.0, 0.05, 0.2, 0.0)
    }

    #[test]
    fn path_shape_and_layout() {
        let cfg = McConfig {
            paths: 8,
            steps: 5,
            seed: 1,
        };
        let ps = simulate_paths(&market(), 1.0, &cfg).unwrap();
        assert_eq!(ps.paths.len(), 8);
        for p in &ps.paths {
            assert_eq!(p.len(), 6); // n_steps + 1
            assert_eq!(p[0], 100.0); // starts at S₀
        }
    }

    #[test]
    fn antithetic_pairs_are_mirror_images() {
        // ln(up) and ln(dn) are symmetric about the deterministic drift, so the
        // geometric mean of each pair's terminal value is the same draw-free path.
        let cfg = McConfig {
            paths: 2,
            steps: 10,
            seed: 7,
        };
        let ps = simulate_paths(&market(), 1.0, &cfg).unwrap();
        let (up, dn) = (&ps.paths[0], &ps.paths[1]);
        let drift = (0.05 - 0.5 * 0.2 * 0.2) * (1.0 / 10.0);
        for j in 1..up.len() {
            let mid = 0.5 * (up[j].ln() + dn[j].ln());
            assert!((mid - (100.0_f64.ln() + j as f64 * drift)).abs() < 1e-9);
        }
    }

    #[test]
    fn deterministic_when_sigma_zero() {
        let mkt = MarketData::new(100.0, 0.05, 0.0, 0.0);
        let cfg = McConfig {
            paths: 100,
            steps: 10,
            seed: 1,
        };
        // Zero-vol put: forward grows at the risk-free rate, so an ATM put can only
        // be worth its immediate intrinsic (0 here) — best discounted exercise.
        let est = deterministic_american(OptionType::Put, &mkt, 100.0, 1.0, &cfg).unwrap();
        assert_eq!(est.standard_error, 0.0);
        assert!(est.price >= 0.0);
        // A deep-ITM zero-vol put exercises immediately at intrinsic.
        let est2 = deterministic_american(OptionType::Put, &mkt, 1000.0, 1.0, &cfg).unwrap();
        assert!((est2.price - 900.0).abs() < 1e-9);
    }

    #[test]
    fn no_deterministic_in_stochastic_case() {
        let cfg = McConfig {
            paths: 10,
            steps: 5,
            seed: 1,
        };
        assert!(deterministic_american(OptionType::Put, &market(), 100.0, 1.0, &cfg).is_none());
    }

    #[test]
    fn rejects_bad_inputs() {
        let bad = McConfig {
            paths: 0,
            steps: 5,
            seed: 1,
        };
        assert!(validate_inputs(100.0, &market(), 1.0, &bad).is_err());
        let bad_steps = McConfig {
            paths: 10,
            steps: 0,
            seed: 1,
        };
        assert!(validate_inputs(100.0, &market(), 1.0, &bad_steps).is_err());
        assert!(validate_inputs(0.0, &market(), 1.0, &McConfig::default()).is_err());
    }
}
