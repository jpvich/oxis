//! Longstaff-Schwartz American Monte Carlo (least-squares method).
//!
//! American options may be exercised early, so a single terminal draw is not
//! enough — we simulate full GBM paths over a time grid and decide, at each
//! exercise date, whether to exercise or continue. The continuation value is
//! unknown, so LSM **estimates it by regression**: at each step, the discounted
//! future cashflows of the in-the-money paths are regressed on a low-degree
//! polynomial of the underlying (`{1, x, x²}` with `x = S/K` for conditioning),
//! and a path exercises where its immediate payoff beats the fitted
//! continuation value. Working backward from maturity yields each path's
//! cashflow; their mean is the price.
//!
//! Variance reduction (antithetic pairs), per-path counter-based seeding, and
//! the ordered/sequential reduction match [`crate::pricing::monte_carlo`], so the price
//! and standard error are bit-reproducible for a given `(seed, paths, steps)`.

use crate::core::{MarketData, OptionType, OxisError, mean_and_se, path_seed, poly_least_squares};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use crate::pricing::monte_carlo::{McConfig, McEstimate, validate_inputs};

/// Polynomial basis degree for the continuation-value regression. Degree 2
/// (`{1, x, x²}`) matches QuantLib's default `MCAmericanEngine` polynomial order.
const BASIS_DEGREE: usize = 2;

/// Price an American option with the Longstaff-Schwartz method.
///
/// `expiry` is the time to expiry in years; `cfg.steps` is the number of
/// exercise dates in the time grid (`>= 1`), `cfg.paths` the number of simulated
/// paths (drawn as antithetic pairs).
///
/// # Errors
/// [`OxisError::InvalidInput`] for inputs outside the model's domain
/// (non-positive strike, negative spot/vol/time, zero paths/steps).
pub fn lsm_american(
    option_type: OptionType,
    market: &MarketData,
    strike: f64,
    expiry: f64,
    cfg: &McConfig,
) -> Result<McEstimate, OxisError> {
    let MarketData {
        spot: s,
        rate: r,
        volatility: sigma,
        dividend_yield: q,
    } = *market;
    let k = strike;
    let t = expiry;

    validate_inputs(k, s, sigma, t, cfg.paths)?;
    if cfg.steps == 0 {
        return Err(OxisError::invalid_input("steps must be >= 1"));
    }

    let n_steps = cfg.steps;
    let dt = t / n_steps as f64;

    // Deterministic limits (T = 0, σ = 0, S = 0): the path is fixed, so the
    // American value is the best discounted exercise over the time grid.
    if t == 0.0 || sigma == 0.0 || s == 0.0 {
        let mut best = option_type.intrinsic(s, k); // exercise now (t = 0)
        for j in 1..=n_steps {
            let tj = j as f64 * dt;
            let s_tj = s * ((r - q) * tj).exp();
            best = best.max((-r * tj).exp() * option_type.intrinsic(s_tj, k));
        }
        return Ok(McEstimate {
            price: best,
            standard_error: 0.0,
        });
    }

    let drift_dt = (r - q - 0.5 * sigma * sigma) * dt;
    let vol_sqrt_dt = sigma * dt.sqrt();
    let ln_s0 = s.ln();
    let n_pairs = cfg.paths.div_ceil(2);
    let seed = cfg.seed;

    // Simulate antithetic path pairs in parallel; collect in index order so the
    // backward induction below is deterministic. Each entry is one full path of
    // length `n_steps + 1` (including S₀).
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

    let n_paths = paths.len();

    // Present-valued cashflow per path. Initialize with exercise at maturity.
    let disc_t = (-r * t).exp();
    let mut cashflow: Vec<f64> = paths
        .iter()
        .map(|p| disc_t * option_type.intrinsic(p[n_steps], k))
        .collect();

    // Backward induction over interior exercise dates j = n_steps-1 .. 1.
    for j in (1..n_steps).rev() {
        let tj = j as f64 * dt;
        let disc_j = (-r * tj).exp();

        // In-the-money paths: regress their PV continuation on x = S/K.
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut itm_idx = Vec::new();
        for (p, path) in paths.iter().enumerate() {
            let spot_j = path[j];
            if option_type.intrinsic(spot_j, k) > 0.0 {
                xs.push(spot_j / k);
                ys.push(cashflow[p]);
                itm_idx.push(p);
            }
        }

        // Need more ITM points than coefficients to fit; otherwise don't
        // exercise at this step (carry continuation forward).
        if itm_idx.len() <= BASIS_DEGREE {
            continue;
        }

        let coeffs = poly_least_squares(&xs, &ys, BASIS_DEGREE)?;
        for (&p, &x) in itm_idx.iter().zip(xs.iter()) {
            let continuation = horner(&coeffs, x);
            let exercise_pv = disc_j * option_type.intrinsic(paths[p][j], k);
            if exercise_pv > continuation {
                cashflow[p] = exercise_pv;
            }
        }
    }

    // Per-pair averaging (antithetic): paths are laid out [up0, dn0, up1, dn1, …].
    let pair_means: Vec<f64> = cashflow
        .chunks_exact(2)
        .map(|c| 0.5 * (c[0] + c[1]))
        .collect();
    let (continuation_price, se) = mean_and_se(&pair_means);

    // Step 0: exercise immediately if that beats continuing (single shared S₀).
    let exercise_now = option_type.intrinsic(s, k);
    if exercise_now > continuation_price {
        return Ok(McEstimate {
            price: exercise_now,
            standard_error: 0.0,
        });
    }

    debug_assert_eq!(n_paths, 2 * n_pairs);
    Ok(McEstimate {
        price: continuation_price,
        standard_error: se,
    })
}

/// Evaluate a polynomial `Σ cᵢ·xⁱ` (lowest order first) by Horner's rule.
fn horner(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{EuropeanOption, ExerciseStyle};
    use crate::pricing::{binomial, black_scholes};

    fn market(vol: f64, q: f64) -> MarketData {
        MarketData::new(100.0, 0.05, vol, q)
    }

    #[test]
    fn american_put_matches_binomial() {
        // The classic early-exercise case: an ITM American put exceeds European.
        let cfg = McConfig {
            paths: 200_000,
            steps: 50,
            seed: 11,
        };
        let est = lsm_american(OptionType::Put, &market(0.3, 0.0), 100.0, 1.0, &cfg).unwrap();
        let tree = binomial(
            OptionType::Put,
            ExerciseStyle::American,
            &market(0.3, 0.0),
            100.0,
            1.0,
            2000,
        )
        .unwrap();
        // LSM is a lower-biased estimator; within a few SE + tree error of the
        // (QuantLib-validated) binomial price.
        assert!(
            (est.price - tree).abs() <= 4.0 * est.standard_error + 0.05,
            "lsm={} binomial={} se={}",
            est.price,
            tree,
            est.standard_error
        );
    }

    #[test]
    fn american_call_no_dividend_matches_european() {
        // Without dividends an American call should not be exercised early, so it
        // equals the European (Black-Scholes) value.
        let cfg = McConfig {
            paths: 200_000,
            steps: 50,
            seed: 5,
        };
        let est = lsm_american(OptionType::Call, &market(0.2, 0.0), 100.0, 1.0, &cfg).unwrap();
        let euro = black_scholes(
            &EuropeanOption {
                strike: 100.0,
                expiry_years: 1.0,
                option_type: OptionType::Call,
            },
            &market(0.2, 0.0),
        )
        .unwrap();
        assert!(
            (est.price - euro).abs() <= 4.0 * est.standard_error + 0.05,
            "lsm={} euro={} se={}",
            est.price,
            euro,
            est.standard_error
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let cfg = McConfig {
            paths: 50_000,
            steps: 25,
            seed: 99,
        };
        let a = lsm_american(OptionType::Put, &market(0.3, 0.0), 110.0, 1.0, &cfg).unwrap();
        let b = lsm_american(OptionType::Put, &market(0.3, 0.0), 110.0, 1.0, &cfg).unwrap();
        assert_eq!(a.price.to_bits(), b.price.to_bits());
        assert_eq!(a.standard_error.to_bits(), b.standard_error.to_bits());
    }

    #[test]
    fn deep_itm_put_exercises_immediately() {
        // Deep ITM American put: immediate exercise dominates; SE collapses to 0.
        let cfg = McConfig::default();
        let est = lsm_american(OptionType::Put, &market(0.2, 0.0), 1000.0, 1.0, &cfg).unwrap();
        // Immediate intrinsic is 1000 - 100 = 900.
        assert!((est.price - 900.0).abs() < 1e-9);
        assert_eq!(est.standard_error, 0.0);
    }

    #[test]
    fn rejects_bad_inputs() {
        assert!(
            lsm_american(
                OptionType::Put,
                &market(0.2, 0.0),
                100.0,
                1.0,
                &McConfig {
                    paths: 0,
                    steps: 10,
                    seed: 1
                }
            )
            .is_err()
        );
        assert!(
            lsm_american(
                OptionType::Put,
                &market(0.2, 0.0),
                100.0,
                1.0,
                &McConfig {
                    paths: 100,
                    steps: 0,
                    seed: 1
                }
            )
            .is_err()
        );
        assert!(
            lsm_american(
                OptionType::Put,
                &market(0.2, 0.0),
                0.0,
                1.0,
                &McConfig::default()
            )
            .is_err()
        );
    }
}
