//! Deep Optimal Stopping (Becker–Cheridito–Jentzen, JMLR 2019): American pricing
//! by learning a stopping *policy* rather than a continuation value.
//!
//! At each exercise date a small network outputs a stop probability
//! `F_n(x) = sigmoid(net(x))`. Working backward from maturity, each `F_n` is
//! trained by gradient ascent on the expected discounted payoff of stopping now
//! versus following the already-learned downstream policy. The learned *hard*
//! policy (`stop ⇔ F_n > ½`) is then applied to a **fresh, independent** set of
//! paths — out-of-sample evaluation removes the optimisation bias, so the price
//! is a valid low-biased estimate (any admissible stopping time underprices the
//! American option). The network, the sigmoid output, and the policy-gradient
//! backprop are plain linear algebra over `oxis::core`; nothing here is a framework.

use crate::core::{OptionType, OxisError, mean_and_se, path_seed, splitmix64};
use crate::ml::activation::sigmoid;
use crate::ml::american::{deterministic_american, simulate_paths, validate_inputs};
use crate::ml::deep_lsm::AmericanMlConfig;
use crate::ml::mlp::Mlp;
use crate::ml::optim::{Adam, Grad, backward_value, mean_std, one_cycle_lr};
use crate::pricing::{McConfig, McEstimate};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

/// Price a 1-D American option (GBM/Black-Scholes) by Deep Optimal Stopping.
///
/// Returns a low-biased Monte-Carlo estimate (the learned policy evaluated on a
/// fresh path set) with its antithetic standard error.
///
/// # Errors
/// [`OxisError::InvalidInput`] for inputs outside the model's domain (non-positive
/// strike, negative spot/vol/time, zero paths/steps, empty hidden spec, zero epochs).
pub fn dos_american(
    option_type: OptionType,
    cfg: &AmericanMlConfig,
) -> Result<McEstimate, OxisError> {
    if cfg.hidden.is_empty() {
        return Err(OxisError::invalid_input("hidden layers must be non-empty"));
    }
    if cfg.epochs == 0 {
        return Err(OxisError::invalid_input("epochs must be >= 1"));
    }
    let mc = McConfig {
        paths: cfg.paths,
        steps: cfg.steps,
        seed: cfg.seed,
    };
    validate_inputs(cfg.strike, &cfg.market, cfg.expiry, &mc)?;

    // Fixed-path limits (T = 0, σ = 0, S = 0): the path is deterministic.
    if let Some(est) = deterministic_american(option_type, &cfg.market, cfg.strike, cfg.expiry, &mc)
    {
        return Ok(est);
    }

    let k = cfg.strike;
    let r = cfg.market.rate;

    // --- Train the per-date stopping policy on the training path set. ---
    let train = simulate_paths(&cfg.market, cfg.expiry, &mc)?;
    let n_steps = train.n_steps;
    let dt = train.dt;
    let disc_t = (-r * cfg.expiry).exp();

    // value_after[i] = PV (at t=0) of following the learned hard policy from the
    // current date onward; initialise with always-stop at maturity.
    let mut value_after: Vec<f64> = train
        .paths
        .iter()
        .map(|p| disc_t * option_type.intrinsic(p[n_steps], k))
        .collect();

    // nets[n] is the stopping network for interior date n (1..n_steps).
    let mut nets: Vec<Option<StopNet>> = (0..n_steps).map(|_| None).collect();

    for n in (1..n_steps).rev() {
        let tn = n as f64 * dt;
        let disc_n = (-r * tn).exp();
        let xs: Vec<f64> = train.paths.iter().map(|p| p[n] / k).collect();
        let exercise: Vec<f64> = train
            .paths
            .iter()
            .map(|p| disc_n * option_type.intrinsic(p[n], k))
            .collect();

        let net = fit_stopping(
            &xs,
            &exercise,
            &value_after,
            &cfg.hidden,
            cfg.epochs,
            path_seed(cfg.seed, usize::MAX - 1 - n),
            path_seed(cfg.seed, 0xD05 ^ n),
        );

        // Hard policy update: stop where the net says so, else carry continuation.
        for (i, &x) in xs.iter().enumerate() {
            if net.stop_prob(x) > 0.5 {
                value_after[i] = exercise[i];
            }
        }
        nets[n] = Some(net);
    }

    // --- Evaluate the learned policy on a fresh, independent path set. ---
    let eval_cfg = McConfig {
        paths: cfg.paths,
        steps: cfg.steps,
        seed: splitmix64(cfg.seed),
    };
    let eval = simulate_paths(&cfg.market, cfg.expiry, &eval_cfg)?;

    let payoffs: Vec<f64> = eval
        .paths
        .iter()
        .map(|p| policy_payoff(p, &nets, option_type, k, r, dt, n_steps, disc_t))
        .collect();

    let pair_means: Vec<f64> = payoffs
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

/// Discounted payoff (PV at t=0) of running one path under the learned hard policy:
/// stop at the first interior date with `F_n > ½`, else at maturity.
#[allow(clippy::too_many_arguments)]
fn policy_payoff(
    path: &[f64],
    nets: &[Option<StopNet>],
    option_type: OptionType,
    k: f64,
    r: f64,
    dt: f64,
    n_steps: usize,
    disc_t: f64,
) -> f64 {
    for n in 1..n_steps {
        if let Some(net) = &nets[n] {
            if net.stop_prob(path[n] / k) > 0.5 {
                let disc_n = (-r * (n as f64 * dt)).exp();
                return disc_n * option_type.intrinsic(path[n], k);
            }
        }
    }
    disc_t * option_type.intrinsic(path[n_steps], k)
}

/// A trained per-date stopping network plus its input standardisation; the scalar
/// linear output is squashed by a sigmoid into a stop probability.
struct StopNet {
    mlp: Mlp,
    x_mean: f64,
    x_std: f64,
}

impl StopNet {
    fn stop_prob(&self, x_raw: f64) -> f64 {
        let xn = (x_raw - self.x_mean) / self.x_std;
        let fwd = self.mlp.forward(&[xn]);
        sigmoid(self.mlp.value(&fwd))
    }
}

/// Fit one date's stopping network by maximising the expected discounted reward
/// `R = mean_i[exercise_i·F_i + cont_i·(1−F_i)]`, `F_i = sigmoid(net(x_i))` — i.e.
/// minimising `L = −R`. The per-sample output adjoint is
/// `∂L/∂y_i = −(exercise_i − cont_i)·F_i·(1−F_i)`. Adam + one-cycle LR; deterministic.
fn fit_stopping(
    xs: &[f64],
    exercise: &[f64],
    cont: &[f64],
    hidden: &[usize],
    epochs: usize,
    init_seed: u64,
    shuffle_seed: u64,
) -> StopNet {
    let m = xs.len();
    let (x_mean, x_std) = mean_std(xs);
    let xn: Vec<f64> = xs.iter().map(|&v| (v - x_mean) / x_std).collect();

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
                let f = sigmoid(mlp.value(&fwd));
                // ∂L/∂y = −(exercise − cont)·F·(1−F); the 1/m batch scaling is below.
                let dy = -(exercise[i] - cont[i]) * f * (1.0 - f);
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

    StopNet { mlp, x_mean, x_std }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{EuropeanOption, ExerciseStyle, MarketData};
    use crate::ml::mlp::Mlp;
    use crate::pricing::{binomial, black_scholes};
    use rand::rngs::SmallRng;

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

    /// The analytic policy-gradient adjoint must match central finite differences
    /// of the per-step DOS loss — the safety net for the new training objective.
    #[test]
    fn gradient_check() {
        let mut rng = SmallRng::seed_from_u64(path_seed(3, 0));
        let mlp = Mlp::new(1, &[5, 4], &mut rng);
        let (x, exercise, cont) = (0.6_f64, 3.0_f64, 1.5_f64);

        let mut grad = Grad::zeros_like(&mlp);
        let fwd = mlp.forward(&[x]);
        let f = sigmoid(mlp.value(&fwd));
        let dy = -(exercise - cont) * f * (1.0 - f);
        backward_value(&mlp, &fwd, dy, &mut grad);

        // L(net) = -(exercise·F + cont·(1−F)) for one sample, F = sigmoid(value).
        let loss_at = |m: &Mlp| -> f64 {
            let fwd = m.forward(&[x]);
            let f = sigmoid(m.value(&fwd));
            -(exercise * f + cont * (1.0 - f))
        };

        let h = 1e-6;
        let mut worst = 0.0_f64;
        for kk in 0..mlp.layers.len() {
            for i in 0..mlp.layers[kk].w.len() {
                for j in 0..mlp.layers[kk].w[i].len() {
                    let mut up = mlp.clone();
                    let mut dn = mlp.clone();
                    up.layers[kk].w[i][j] += h;
                    dn.layers[kk].w[i][j] -= h;
                    let fd = (loss_at(&up) - loss_at(&dn)) / (2.0 * h);
                    worst = worst.max((fd - grad.gw[kk][i][j]).abs());
                }
            }
            for i in 0..mlp.layers[kk].b.len() {
                let mut up = mlp.clone();
                let mut dn = mlp.clone();
                up.layers[kk].b[i] += h;
                dn.layers[kk].b[i] -= h;
                let fd = (loss_at(&up) - loss_at(&dn)) / (2.0 * h);
                worst = worst.max((fd - grad.gb[kk][i]).abs());
            }
        }
        assert!(worst < 1e-5, "worst grad mismatch {worst:.3e}");
    }

    #[test]
    fn american_put_matches_binomial() {
        let c = cfg(0.3, 100.0);
        let est = dos_american(OptionType::Put, &c).unwrap();
        let tree = binomial(
            OptionType::Put,
            ExerciseStyle::American,
            &c.market,
            c.strike,
            c.expiry,
            2000,
        )
        .unwrap();
        assert!(
            (est.price - tree).abs() <= 5.0 * est.standard_error + 0.50,
            "dos={} binomial={} se={}",
            est.price,
            tree,
            est.standard_error
        );
    }

    #[test]
    fn call_no_dividend_matches_european() {
        // Without dividends an American call is never exercised early, so the
        // learned policy should run to maturity and recover the European value.
        let c = cfg(0.2, 100.0);
        let est = dos_american(OptionType::Call, &c).unwrap();
        let euro = black_scholes(
            &EuropeanOption {
                strike: c.strike,
                expiry_years: c.expiry,
                option_type: OptionType::Call,
            },
            &c.market,
        )
        .unwrap();
        assert!(
            (est.price - euro).abs() <= 5.0 * est.standard_error + 0.50,
            "dos={} euro={} se={}",
            est.price,
            euro,
            est.standard_error
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let c = cfg(0.3, 110.0);
        let a = dos_american(OptionType::Put, &c).unwrap();
        let b = dos_american(OptionType::Put, &c).unwrap();
        assert_eq!(a.price.to_bits(), b.price.to_bits());
        assert_eq!(a.standard_error.to_bits(), b.standard_error.to_bits());
    }

    #[test]
    fn deep_itm_put_exercises_immediately() {
        let c = cfg(0.2, 1000.0);
        let est = dos_american(OptionType::Put, &c).unwrap();
        assert!((est.price - 900.0).abs() < 1e-9);
        assert_eq!(est.standard_error, 0.0);
    }

    #[test]
    fn rejects_bad_inputs() {
        let mut c = cfg(0.2, 100.0);
        c.hidden = vec![];
        assert!(dos_american(OptionType::Put, &c).is_err());
        let mut c = cfg(0.2, 100.0);
        c.epochs = 0;
        assert!(dos_american(OptionType::Put, &c).is_err());
        let mut c = cfg(0.2, 100.0);
        c.strike = 0.0;
        assert!(dos_american(OptionType::Put, &c).is_err());
    }

    #[test]
    #[ignore = "calibration: run with --release --ignored --nocapture to size bands"]
    fn calibrate_grid_accuracy() {
        let spots = [90.0, 100.0, 110.0];
        for s in spots {
            let mut c = cfg(0.3, 100.0);
            c.market = MarketData::new(s, 0.05, 0.3, 0.0);
            let dos = dos_american(OptionType::Put, &c).unwrap();
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
                "S={s:.0} dos={:.4} (se {:.4}) binomial={tree:.4} |Δtree|={:.4}",
                dos.price,
                dos.standard_error,
                (dos.price - tree).abs(),
            );
        }
    }
}
