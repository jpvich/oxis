//! The path engine: reproducible, antithetic simulation of [`Process`] paths.
//!
//! Every independent unit of work is an **antithetic pair** indexed by `i`: a
//! standard-normal stream `z` drives the "up" path and `−z` the "dn" path (for the
//! Brownian increments; Merton jumps are shared across the pair). Each pair seeds
//! its own `SmallRng` from [`crate::core::path_seed`]`(seed, i)` and the pairs are
//! collected in index order, so the simulation is **bit-reproducible** for a given
//! `(seed, paths, steps)` regardless of how `rayon` schedules the work — the same
//! determinism guarantee as the Monte Carlo / Longstaff-Schwartz pricers.

use crate::core::{OxisError, path_seed};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Poisson, StandardNormal};
use rayon::prelude::*;

use crate::stochastic::process::Process;

/// One simulated trajectory of the tracked quantity over the time grid, including
/// the initial state at index `0` (length `steps + 1`).
pub type Path = Vec<f64>;

/// Simulation configuration — mirrors the pricing engines' `McConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimConfig {
    /// Number of simulated paths (drawn as `paths / 2` antithetic pairs, rounded
    /// up to a whole pair).
    pub paths: usize,
    /// Number of time steps on the grid `t_j = j·(t/steps)`.
    pub steps: usize,
    /// RNG seed — fixes the whole simulation for reproducibility.
    pub seed: u64,
}

impl Default for SimConfig {
    /// 100k paths, 50 steps, seed 42.
    fn default() -> Self {
        Self {
            paths: 100_000,
            steps: 50,
            seed: 42,
        }
    }
}

/// Terminal states `X_t` of a simulation, kept memory-light (no full paths).
///
/// `terminals` holds every individual terminal value (`2 × n_pairs` of them) — the
/// right sample for estimating the *distribution's* variance. `pair_means` holds
/// the `n_pairs` antithetic averages — the right sample for the mean's standard
/// error (it captures the negative correlation within each pair).
#[derive(Debug, Clone)]
pub struct TerminalSample {
    /// Every terminal value, in pair order `[up₀, dn₀, up₁, dn₁, …]`.
    pub terminals: Vec<f64>,
    /// Per-pair antithetic averages `½(upᵢ + dnᵢ)`.
    pub pair_means: Vec<f64>,
}

fn validate(process: &Process, x0: f64, t: f64, cfg: &SimConfig) -> Result<(), OxisError> {
    process.validate()?;
    if !x0.is_finite() {
        return Err(OxisError::invalid_input("x0 must be finite"));
    }
    if process.requires_positive_state() && x0 <= 0.0 {
        return Err(OxisError::invalid_input("x0 must be > 0 for this process"));
    }
    if matches!(process, Process::Cir { .. }) && x0 < 0.0 {
        return Err(OxisError::invalid_input("x0 must be >= 0 for CIR"));
    }
    if !(t.is_finite() && t >= 0.0) {
        return Err(OxisError::invalid_input("t must be >= 0"));
    }
    if cfg.paths == 0 {
        return Err(OxisError::invalid_input("paths must be >= 1"));
    }
    if cfg.steps == 0 {
        return Err(OxisError::invalid_input("steps must be >= 1"));
    }
    Ok(())
}

/// Simulate the **terminal** states only — the input for moment validation.
///
/// # Errors
/// [`OxisError::InvalidInput`] for out-of-domain parameters or config.
pub fn simulate_terminal(
    process: &Process,
    x0: f64,
    t: f64,
    cfg: &SimConfig,
) -> Result<TerminalSample, OxisError> {
    validate(process, x0, t, cfg)?;
    let n_pairs = cfg.paths.div_ceil(2);
    let pairs: Vec<(f64, f64)> = (0..n_pairs)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(path_seed(cfg.seed, i));
            let (up, dn) = simulate_pair(process, x0, t, cfg.steps, &mut rng);
            (up[cfg.steps], dn[cfg.steps])
        })
        .collect();

    let mut terminals = Vec::with_capacity(2 * n_pairs);
    let mut pair_means = Vec::with_capacity(n_pairs);
    for (u, d) in pairs {
        terminals.push(u);
        terminals.push(d);
        pair_means.push(0.5 * (u + d));
    }
    Ok(TerminalSample {
        terminals,
        pair_means,
    })
}

/// Simulate **full paths**, laid out `[up₀, dn₀, up₁, dn₁, …]` so a consumer can
/// recover antithetic pairs with `chunks_exact(2)`. Used by path-dependent pricing
/// (e.g. arithmetic-average Asian options).
///
/// # Errors
/// [`OxisError::InvalidInput`] for out-of-domain parameters or config.
pub fn simulate_paths(
    process: &Process,
    x0: f64,
    t: f64,
    cfg: &SimConfig,
) -> Result<Vec<Path>, OxisError> {
    validate(process, x0, t, cfg)?;
    let n_pairs = cfg.paths.div_ceil(2);
    let nested: Vec<[Path; 2]> = (0..n_pairs)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(path_seed(cfg.seed, i));
            let (up, dn) = simulate_pair(process, x0, t, cfg.steps, &mut rng);
            [up, dn]
        })
        .collect();
    Ok(nested.into_iter().flatten().collect())
}

/// Advance one antithetic pair from `x0` over `[0, t]` in `steps` steps, returning
/// the two full paths of the tracked quantity (each of length `steps + 1`).
///
/// The "dn" path negates the Brownian increments of the "up" path; Merton's
/// Poisson jumps are drawn once and shared by both members of the pair (antithetic
/// applies to the diffusion only).
fn simulate_pair(
    process: &Process,
    x0: f64,
    t: f64,
    steps: usize,
    rng: &mut SmallRng,
) -> (Path, Path) {
    let dt = t / steps as f64;
    let sqrt_dt = dt.sqrt();
    let mut up = Vec::with_capacity(steps + 1);
    let mut dn = Vec::with_capacity(steps + 1);
    up.push(x0);
    dn.push(x0);

    match *process {
        Process::Gbm { mu, sigma } => {
            let (mut su, mut sd) = (x0, x0);
            let drift = (mu - 0.5 * sigma * sigma) * dt;
            let vol = sigma * sqrt_dt;
            for _ in 0..steps {
                let z: f64 = StandardNormal.sample(rng);
                su *= (drift + vol * z).exp();
                sd *= (drift - vol * z).exp();
                up.push(su);
                dn.push(sd);
            }
        }
        Process::OrnsteinUhlenbeck {
            kappa,
            theta,
            sigma,
        }
        | Process::Vasicek {
            kappa,
            theta,
            sigma,
        } => {
            // Exact Gaussian transition: Xₖ₊₁ = Xₖe^{-κdt} + θ(1-e^{-κdt}) + s·z.
            let e = (-kappa * dt).exp();
            let mean_shift = theta * (1.0 - e);
            let sd_step = sigma * ((1.0 - e * e) / (2.0 * kappa)).sqrt();
            let (mut xu, mut xd) = (x0, x0);
            for _ in 0..steps {
                let z: f64 = StandardNormal.sample(rng);
                xu = xu * e + mean_shift + sd_step * z;
                xd = xd * e + mean_shift - sd_step * z;
                up.push(xu);
                dn.push(xd);
            }
        }
        Process::Cir {
            kappa,
            theta,
            sigma,
        } => {
            // Full-truncation Euler: keep the raw running state (which may dip
            // below zero), but use its positive part in the drift/diffusion and
            // report the positive part as the process value.
            let (mut xu, mut xd) = (x0, x0);
            for _ in 0..steps {
                let z: f64 = StandardNormal.sample(rng);
                let pu = xu.max(0.0);
                let pd = xd.max(0.0);
                xu += kappa * (theta - pu) * dt + sigma * pu.sqrt() * sqrt_dt * z;
                xd += kappa * (theta - pd) * dt - sigma * pd.sqrt() * sqrt_dt * z;
                up.push(xu.max(0.0));
                dn.push(xd.max(0.0));
            }
        }
        Process::MertonJump {
            mu,
            sigma,
            lambda,
            jump_mean,
            jump_std,
        } => {
            let drift = (mu - 0.5 * sigma * sigma) * dt;
            let vol = sigma * sqrt_dt;
            let poisson = (lambda > 0.0).then(|| Poisson::new(lambda * dt).expect("lambda*dt > 0"));
            let (mut su, mut sd) = (x0, x0);
            for _ in 0..steps {
                let z: f64 = StandardNormal.sample(rng);
                su *= (drift + vol * z).exp();
                sd *= (drift - vol * z).exp();
                // Shared compound-Poisson jump factor for both pair members.
                if let Some(p) = &poisson {
                    let n = p.sample(rng) as u64;
                    let mut log_jump = 0.0;
                    for _ in 0..n {
                        let zj: f64 = StandardNormal.sample(rng);
                        log_jump += jump_mean + jump_std * zj;
                    }
                    let factor = log_jump.exp();
                    su *= factor;
                    sd *= factor;
                }
                up.push(su);
                dn.push(sd);
            }
        }
        Process::Heston {
            mu,
            v0,
            kappa,
            theta,
            xi,
            rho,
        } => {
            let rho_perp = (1.0 - rho * rho).max(0.0).sqrt();
            let (mut su, mut sd) = (x0, x0);
            let (mut vu, mut vd) = (v0, v0);
            for _ in 0..steps {
                let z_v: f64 = StandardNormal.sample(rng);
                let z_perp: f64 = StandardNormal.sample(rng);
                let z_s = rho * z_v + rho_perp * z_perp;

                let pu = vu.max(0.0);
                su *= ((mu - 0.5 * pu) * dt + pu.sqrt() * sqrt_dt * z_s).exp();
                vu += kappa * (theta - pu) * dt + xi * pu.sqrt() * sqrt_dt * z_v;

                // Antithetic: negate both driving normals (so z_s flips sign too).
                let pd = vd.max(0.0);
                sd *= ((mu - 0.5 * pd) * dt - pd.sqrt() * sqrt_dt * z_s).exp();
                vd += kappa * (theta - pd) * dt - xi * pd.sqrt() * sqrt_dt * z_v;

                up.push(su);
                dn.push(sd);
            }
        }
    }

    (up, dn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_across_runs() {
        let p = Process::Gbm {
            mu: 0.05,
            sigma: 0.2,
        };
        let cfg = SimConfig {
            paths: 10_000,
            steps: 20,
            seed: 7,
        };
        let a = simulate_terminal(&p, 100.0, 1.0, &cfg).unwrap();
        let b = simulate_terminal(&p, 100.0, 1.0, &cfg).unwrap();
        assert_eq!(a.terminals.len(), b.terminals.len());
        for (x, y) in a.terminals.iter().zip(b.terminals.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn paths_have_expected_shape() {
        let p = Process::Cir {
            kappa: 1.0,
            theta: 0.04,
            sigma: 0.1,
        };
        let cfg = SimConfig {
            paths: 8,
            steps: 12,
            seed: 1,
        };
        let paths = simulate_paths(&p, 0.04, 1.0, &cfg).unwrap();
        assert_eq!(paths.len(), 8); // 4 pairs × 2
        assert!(paths.iter().all(|path| path.len() == 13)); // steps + 1
        assert!(paths.iter().all(|path| path[0] == 0.04)); // initial state
        assert!(paths.iter().all(|path| path.iter().all(|&v| v >= 0.0))); // CIR stays >= 0
    }

    #[test]
    fn zero_vol_gbm_is_deterministic_drift() {
        let p = Process::Gbm {
            mu: 0.03,
            sigma: 0.0,
        };
        let cfg = SimConfig {
            paths: 100,
            steps: 10,
            seed: 3,
        };
        let sample = simulate_terminal(&p, 100.0, 2.0, &cfg).unwrap();
        let expected = 100.0 * (0.03_f64 * 2.0).exp();
        assert!(
            sample
                .terminals
                .iter()
                .all(|&x| (x - expected).abs() < 1e-9)
        );
    }

    #[test]
    fn rejects_bad_config() {
        let p = Process::Gbm {
            mu: 0.05,
            sigma: 0.2,
        };
        assert!(
            simulate_terminal(
                &p,
                100.0,
                1.0,
                &SimConfig {
                    paths: 0,
                    steps: 10,
                    seed: 1
                }
            )
            .is_err()
        );
        assert!(simulate_terminal(&p, -1.0, 1.0, &SimConfig::default()).is_err());
    }
}
