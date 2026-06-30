//! Deep Longstaff-Schwartz: American pricing where the per-date continuation
//! value is fit by a small neural network instead of a low-degree polynomial.
//!
//! The algorithm is exactly Longstaff-Schwartz (simulate GBM paths, work
//! backward, exercise where immediate payoff beats the estimated continuation)
//! with one substitution: at each interior exercise date the in-the-money paths'
//! discounted continuation is regressed on a fresh softplus MLP of `x = S/K`
//! rather than the classical basis `{1, x, x²}`. Everything else — antithetic
//! pairs, per-pair seeding, the per-pair average and standard error, the step-0
//! immediate-exercise check — matches `oxis_pricing::lsm_american`, so the price
//! is a low-biased estimate directly comparable to the classical engine and the
//! QuantLib-validated binomial tree.

use crate::american::{deterministic_american, simulate_paths, validate_inputs};
use crate::mlp::Mlp;
use crate::optim::{Adam, Grad, backward_value, mean_std, one_cycle_lr};
use oxis_core::{MarketData, OptionType, OxisError, mean_and_se, path_seed};
use oxis_pricing::{McConfig, McEstimate};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

/// Hyperparameters for the neural American engines (Deep LSM and DOS).
#[derive(Debug, Clone, PartialEq)]
pub struct AmericanMlConfig {
    /// Market data (spot, rate, volatility, dividend yield; continuously compounded).
    pub market: MarketData,
    /// Strike.
    pub strike: f64,
    /// Time to expiry in years.
    pub expiry: f64,
    /// Number of simulated paths (drawn as antithetic pairs).
    pub paths: usize,
    /// Number of exercise dates in the time grid (`>= 1`).
    pub steps: usize,
    /// RNG seed — fixes paths, per-date initialisation, and shuffling.
    pub seed: u64,
    /// Hidden-layer widths of the per-date continuation network.
    pub hidden: Vec<usize>,
    /// Training epochs per exercise date.
    pub epochs: usize,
}

impl Default for AmericanMlConfig {
    fn default() -> Self {
        Self {
            market: MarketData::new(100.0, 0.05, 0.3, 0.0),
            strike: 100.0,
            expiry: 1.0,
            paths: 8192,
            steps: 50,
            seed: 11,
            hidden: vec![16],
            epochs: 20,
        }
    }
}

impl AmericanMlConfig {
    fn mc(&self) -> McConfig {
        McConfig {
            paths: self.paths,
            steps: self.steps,
            seed: self.seed,
        }
    }
}

/// Price a 1-D American option (GBM/Black-Scholes) by Deep Longstaff-Schwartz.
///
/// Returns a low-biased Monte-Carlo estimate with its antithetic standard error.
///
/// # Errors
/// [`OxisError::InvalidInput`] for inputs outside the model's domain (non-positive
/// strike, negative spot/vol/time, zero paths/steps, empty hidden spec, zero epochs).
pub fn deep_lsm_american(
    option_type: OptionType,
    cfg: &AmericanMlConfig,
) -> Result<McEstimate, OxisError> {
    if cfg.hidden.is_empty() {
        return Err(OxisError::invalid_input("hidden layers must be non-empty"));
    }
    if cfg.epochs == 0 {
        return Err(OxisError::invalid_input("epochs must be >= 1"));
    }
    let mc = cfg.mc();
    validate_inputs(cfg.strike, &cfg.market, cfg.expiry, &mc)?;

    // Fixed-path limits (T = 0, σ = 0, S = 0): the path is deterministic.
    if let Some(est) = deterministic_american(option_type, &cfg.market, cfg.strike, cfg.expiry, &mc)
    {
        return Ok(est);
    }

    let k = cfg.strike;
    let r = cfg.market.rate;
    let ps = simulate_paths(&cfg.market, cfg.expiry, &mc)?;
    let n_steps = ps.n_steps;
    let dt = ps.dt;

    // Present-valued cashflow per path, initialised with exercise at maturity.
    let disc_t = (-r * cfg.expiry).exp();
    let mut cashflow: Vec<f64> = ps
        .paths
        .iter()
        .map(|p| disc_t * option_type.intrinsic(p[n_steps], k))
        .collect();

    // Minimum in-the-money points before we trust a fit (else carry continuation
    // forward) — the neural analogue of LSM's "more points than coefficients".
    let min_itm = cfg.hidden[0].max(2);

    // Backward induction over interior exercise dates j = n_steps-1 .. 1.
    for j in (1..n_steps).rev() {
        let tj = j as f64 * dt;
        let disc_j = (-r * tj).exp();

        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut itm_idx = Vec::new();
        for (p, path) in ps.paths.iter().enumerate() {
            let spot_j = path[j];
            if option_type.intrinsic(spot_j, k) > 0.0 {
                xs.push(spot_j / k);
                ys.push(cashflow[p]);
                itm_idx.push(p);
            }
        }
        if itm_idx.len() <= min_itm {
            continue;
        }

        // Fit the continuation value with a fresh per-date network; seeds chosen
        // to avoid colliding with the path seeds `path_seed(cfg.seed, i)`.
        let net = fit_continuation(
            &xs,
            &ys,
            &cfg.hidden,
            cfg.epochs,
            path_seed(cfg.seed, usize::MAX - 1 - j),
            path_seed(cfg.seed, 0xABCD ^ j),
        );
        for (&p, &x) in itm_idx.iter().zip(xs.iter()) {
            let continuation = net.predict(x);
            let exercise_pv = disc_j * option_type.intrinsic(ps.paths[p][j], k);
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

    // Step 0: exercise immediately if that beats continuing.
    let exercise_now = option_type.intrinsic(cfg.market.spot, k);
    if exercise_now > continuation_price {
        return Ok(McEstimate {
            price: exercise_now,
            standard_error: 0.0,
        });
    }

    Ok(McEstimate {
        price: continuation_price,
        standard_error: se,
    })
}

/// A fitted continuation-value network plus its standardisation.
struct ContinuationNet {
    mlp: Mlp,
    x_mean: f64,
    x_std: f64,
    y_mean: f64,
    y_std: f64,
}

impl ContinuationNet {
    /// Predicted (raw) continuation value at feature `x = S/K`.
    fn predict(&self, x_raw: f64) -> f64 {
        let xn = (x_raw - self.x_mean) / self.x_std;
        let fwd = self.mlp.forward(&[xn]);
        self.y_mean + self.y_std * self.mlp.value(&fwd)
    }
}

/// Fit a scalar continuation net `ŷ ≈ E[continuation | x]` by minimising MSE on
/// standardized `(x, y)`, with Adam + the one-cycle LR schedule. Deterministic
/// given the two seeds.
fn fit_continuation(
    xs: &[f64],
    ys: &[f64],
    hidden: &[usize],
    epochs: usize,
    init_seed: u64,
    shuffle_seed: u64,
) -> ContinuationNet {
    let m = xs.len();
    let (x_mean, x_std) = mean_std(xs);
    let (y_mean, y_std) = mean_std(ys);
    let xn: Vec<f64> = xs.iter().map(|&v| (v - x_mean) / x_std).collect();
    let yn: Vec<f64> = ys.iter().map(|&v| (v - y_mean) / y_std).collect();

    let mut rng = SmallRng::seed_from_u64(init_seed);
    let mut mlp = Mlp::new(1, hidden, &mut rng);
    let mut adam = Adam::new(&mlp);

    let batch_size = m.min(256.max(m / 16)).max(1);
    let batches = m.div_ceil(batch_size);
    let total_steps = (epochs * batches).max(1);

    let mut idx: Vec<usize> = (0..m).collect();
    let mut shuffler = SmallRng::seed_from_u64(shuffle_seed);
    let mut step = 0usize;

    for _ in 0..epochs {
        idx.shuffle(&mut shuffler);
        for batch in idx.chunks(batch_size) {
            let lr = one_cycle_lr(step as f64 / total_steps as f64);
            let mut grad = Grad::zeros_like(&mlp);
            for &i in batch {
                let fwd = mlp.forward(&[xn[i]]);
                let y_hat = mlp.value(&fwd);
                // MSE adjoint ∂(ŷ−y)²/∂ŷ = 2(ŷ−y); the 1/m batch scaling is below.
                let dy = 2.0 * (y_hat - yn[i]);
                backward_value(&mlp, &fwd, dy, &mut grad);
            }
            let scale = 1.0 / batch.len() as f64;
            for layer in grad.gw.iter_mut() {
                for row in layer.iter_mut() {
                    for g in row.iter_mut() {
                        *g *= scale;
                    }
                }
            }
            for layer in grad.gb.iter_mut() {
                for g in layer.iter_mut() {
                    *g *= scale;
                }
            }
            adam.step(&mut mlp, &grad, lr);
            step += 1;
        }
    }

    ContinuationNet {
        mlp,
        x_mean,
        x_std,
        y_mean,
        y_std,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxis_core::ExerciseStyle;
    use oxis_pricing::{binomial, lsm_american};

    fn cfg(vol: f64, strike: f64) -> AmericanMlConfig {
        AmericanMlConfig {
            market: MarketData::new(100.0, 0.05, vol, 0.0),
            strike,
            expiry: 1.0,
            paths: 4096,
            steps: 10,
            seed: 11,
            hidden: vec![16],
            epochs: 20,
        }
    }

    #[test]
    fn american_put_matches_binomial() {
        let c = cfg(0.3, 100.0);
        let est = deep_lsm_american(OptionType::Put, &c).unwrap();
        let tree = binomial(
            OptionType::Put,
            ExerciseStyle::American,
            &c.market,
            c.strike,
            c.expiry,
            2000,
        )
        .unwrap();
        // Low-biased like classical LSM; within a few SE + a small absolute slack
        // of the (QuantLib-validated) binomial price.
        assert!(
            (est.price - tree).abs() <= 5.0 * est.standard_error + 0.30,
            "deep-lsm={} binomial={} se={}",
            est.price,
            tree,
            est.standard_error
        );
    }

    #[test]
    fn tracks_classical_lsm() {
        // At matched (paths, steps, seed) the neural continuation should land close
        // to the polynomial LSM it replaces.
        let c = cfg(0.3, 100.0);
        let deep = deep_lsm_american(OptionType::Put, &c).unwrap();
        let classical =
            lsm_american(OptionType::Put, &c.market, c.strike, c.expiry, &c.mc()).unwrap();
        assert!(
            (deep.price - classical.price).abs()
                <= 5.0 * (deep.standard_error + classical.standard_error) + 0.30,
            "deep={} classical={} se_deep={} se_cls={}",
            deep.price,
            classical.price,
            deep.standard_error,
            classical.standard_error
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let c = cfg(0.3, 110.0);
        let a = deep_lsm_american(OptionType::Put, &c).unwrap();
        let b = deep_lsm_american(OptionType::Put, &c).unwrap();
        assert_eq!(a.price.to_bits(), b.price.to_bits());
        assert_eq!(a.standard_error.to_bits(), b.standard_error.to_bits());
    }

    #[test]
    fn deep_itm_put_exercises_immediately() {
        let c = cfg(0.2, 1000.0);
        let est = deep_lsm_american(OptionType::Put, &c).unwrap();
        assert!((est.price - 900.0).abs() < 1e-9);
        assert_eq!(est.standard_error, 0.0);
    }

    #[test]
    fn rejects_bad_inputs() {
        let mut c = cfg(0.2, 100.0);
        c.hidden = vec![];
        assert!(deep_lsm_american(OptionType::Put, &c).is_err());
        let mut c = cfg(0.2, 100.0);
        c.epochs = 0;
        assert!(deep_lsm_american(OptionType::Put, &c).is_err());
        let mut c = cfg(0.2, 0.0);
        c.strike = 0.0;
        assert!(deep_lsm_american(OptionType::Put, &c).is_err());
    }

    #[test]
    #[ignore = "calibration: run with --release --ignored --nocapture to size bands"]
    fn calibrate_grid_accuracy() {
        // Mirror the green validation config (deep_lsm.json accuracy block) so the
        // printed gaps size its bands directly.
        let spots = [90.0, 100.0, 110.0];
        for s in spots {
            let mut c = cfg(0.3, 100.0); // paths 4096, steps 10, hidden [16], epochs 20
            c.market = MarketData::new(s, 0.05, 0.3, 0.0);
            let deep = deep_lsm_american(OptionType::Put, &c).unwrap();
            let classical =
                lsm_american(OptionType::Put, &c.market, c.strike, c.expiry, &c.mc()).unwrap();
            let tree = binomial(
                OptionType::Put,
                ExerciseStyle::American,
                &c.market,
                c.strike,
                c.expiry,
                2000,
            )
            .unwrap();
            eprintln!(
                "S={s:.0} deep={:.4} (se {:.4}) classical={:.4} binomial={:.4} | dΔtree={:.4} clsΔtree={:.4}",
                deep.price,
                deep.standard_error,
                classical.price,
                tree,
                (deep.price - tree).abs(),
                (classical.price - tree).abs(),
            );
        }
    }
}
