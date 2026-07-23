//! The stochastic processes and their **closed-form moments**.
//!
//! A path simulator is validated against the analytic moments of the law it
//! samples — there is no "price" to check, so the oracle is the exact mean and
//! variance of `X_t`. Each [`Process`] therefore carries [`Process::analytic_moments`],
//! used by the validation suite as ground truth. The actual stepping lives in
//! [`crate::stochastic::simulate`]; this module is the pure description of each model plus its
//! moments and parameter domain.
//!
//! Exact-in-distribution schemes (GBM, Ornstein-Uhlenbeck / Vasicek, Merton
//! jump-diffusion) have *no* time-discretization bias, so their simulated moments
//! match the closed forms up to sampling error. The square-root processes (CIR
//! and the Heston variance) use a **full-truncation Euler** scheme, which carries
//! a small `O(dt)` discretization bias; the validation bands account for it.

use crate::core::OxisError;

/// A continuous-time stochastic process, identified by its parameters.
///
/// `x0` (the initial state) and the horizon are supplied at simulation time, not
/// stored here, so one `Process` value can be simulated from several starting
/// points. For the multiplicative models (`Gbm`, `MertonJump`, `Heston`) the
/// simulated quantity is an asset price and `x0 = S₀ > 0`; for the mean-reverting
/// models (`OrnsteinUhlenbeck`, `Vasicek`, `Cir`) it is the level itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Process {
    /// Geometric Brownian motion: `dS = μS dt + σS dW`. Exact log-Euler.
    Gbm {
        /// Drift.
        mu: f64,
        /// Volatility (`≥ 0`).
        sigma: f64,
    },
    /// Ornstein-Uhlenbeck: `dX = κ(θ − X) dt + σ dW`. Exact Gaussian transition.
    OrnsteinUhlenbeck {
        /// Mean-reversion speed (`> 0`).
        kappa: f64,
        /// Long-run mean.
        theta: f64,
        /// Volatility (`≥ 0`).
        sigma: f64,
    },
    /// Vasicek short-rate model — the Ornstein-Uhlenbeck dynamics applied to an
    /// interest rate. Same exact transition; kept as a distinct variant for
    /// naming/semantics.
    Vasicek {
        /// Mean-reversion speed (`> 0`).
        kappa: f64,
        /// Long-run mean rate.
        theta: f64,
        /// Volatility (`≥ 0`).
        sigma: f64,
    },
    /// Cox-Ingersoll-Ross: `dX = κ(θ − X) dt + σ√X dW`, `X ≥ 0`. Full-truncation
    /// Euler (the Feller condition `2κθ ≥ σ²` is *not* required by this scheme).
    Cir {
        /// Mean-reversion speed (`> 0`).
        kappa: f64,
        /// Long-run mean.
        theta: f64,
        /// Volatility of the square-root diffusion (`≥ 0`).
        sigma: f64,
    },
    /// Merton jump-diffusion: GBM plus a compound-Poisson stream of log-normal
    /// jumps. Exact in distribution (exact diffusion + exact Poisson jumps).
    MertonJump {
        /// Drift of the continuous part.
        mu: f64,
        /// Diffusion volatility (`≥ 0`).
        sigma: f64,
        /// Jump intensity per unit time (`≥ 0`).
        lambda: f64,
        /// Mean of the log-jump size.
        jump_mean: f64,
        /// Standard deviation of the log-jump size (`≥ 0`).
        jump_std: f64,
    },
    /// Heston stochastic volatility: `dS = μS dt + √v S dW₁`,
    /// `dv = κ(θ − v) dt + ξ√v dW₂`, `corr(dW₁, dW₂) = ρ`. Variance via
    /// full-truncation Euler.
    Heston {
        /// Drift of the asset.
        mu: f64,
        /// Initial variance (`≥ 0`).
        v0: f64,
        /// Variance mean-reversion speed (`> 0`).
        kappa: f64,
        /// Long-run variance.
        theta: f64,
        /// Volatility of variance (`≥ 0`).
        xi: f64,
        /// Correlation between the price and variance Brownians (`∈ [−1, 1]`).
        rho: f64,
    },
}

impl Process {
    /// A short, stable identifier for output / the CLI (`"gbm"`, `"heston"`, …).
    pub fn name(&self) -> &'static str {
        match self {
            Process::Gbm { .. } => "gbm",
            Process::OrnsteinUhlenbeck { .. } => "ornstein-uhlenbeck",
            Process::Vasicek { .. } => "vasicek",
            Process::Cir { .. } => "cir",
            Process::MertonJump { .. } => "merton-jump",
            Process::Heston { .. } => "heston",
        }
    }

    /// Whether the simulated state must stay strictly positive (an asset price),
    /// so the simulator can reject `x0 ≤ 0`.
    pub fn requires_positive_state(&self) -> bool {
        matches!(
            self,
            Process::Gbm { .. } | Process::MertonJump { .. } | Process::Heston { .. }
        )
    }

    /// Closed-form mean and (where available) variance of `X_t` given `x0`.
    ///
    /// The variance is `None` for Heston — its `Var[S_t]` closed form is omitted
    /// here; the Heston dynamics are validated instead by pricing a European
    /// option over simulated paths against QuantLib's `AnalyticHestonEngine`
    /// (in `oxis::pricing`). Every other process returns an exact `(mean, var)`.
    pub fn analytic_moments(&self, x0: f64, t: f64) -> (f64, Option<f64>) {
        match *self {
            Process::Gbm { mu, sigma } => {
                let mean = x0 * (mu * t).exp();
                let var = x0 * x0 * (2.0 * mu * t).exp() * ((sigma * sigma * t).exp() - 1.0);
                (mean, Some(var))
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
                let e = (-kappa * t).exp();
                let mean = x0 * e + theta * (1.0 - e);
                let var = sigma * sigma / (2.0 * kappa) * (1.0 - (-2.0 * kappa * t).exp());
                (mean, Some(var))
            }
            Process::Cir {
                kappa,
                theta,
                sigma,
            } => {
                let e = (-kappa * t).exp();
                let mean = theta + (x0 - theta) * e;
                let s2 = sigma * sigma;
                let var = x0 * (s2 / kappa) * (e - e * e)
                    + theta * (s2 / (2.0 * kappa)) * (1.0 - e).powi(2);
                (mean, Some(var))
            }
            Process::MertonJump {
                mu,
                sigma,
                lambda,
                jump_mean,
                jump_std,
            } => {
                // Over [0, t] the jump count is Poisson(λt); E[e^{ΣJ}] = exp(λt(E[e^J]−1)).
                let k1 = (jump_mean + 0.5 * jump_std * jump_std).exp();
                let k2 = (2.0 * jump_mean + 2.0 * jump_std * jump_std).exp();
                let mean = x0 * (mu * t).exp() * (lambda * t * (k1 - 1.0)).exp();
                let e_s2 = x0
                    * x0
                    * (2.0 * mu * t + sigma * sigma * t).exp()
                    * (lambda * t * (k2 - 1.0)).exp();
                (mean, Some(e_s2 - mean * mean))
            }
            Process::Heston { mu, .. } => {
                // E[S_t] = S₀ e^{μt} exactly (the variance process does not bias the mean).
                (x0 * (mu * t).exp(), None)
            }
        }
    }

    /// Validate the process parameters, independent of `x0`/horizon.
    ///
    /// # Errors
    /// [`OxisError::InvalidInput`] for out-of-domain parameters (non-finite
    /// values, non-positive mean-reversion speed, negative volatility/intensity,
    /// correlation outside `[−1, 1]`).
    pub fn validate(&self) -> Result<(), OxisError> {
        let finite = |name: &str, v: f64| {
            if v.is_finite() {
                Ok(())
            } else {
                Err(OxisError::invalid_input(format!("{name} must be finite")))
            }
        };
        let nonneg = |name: &str, v: f64| {
            if v.is_finite() && v >= 0.0 {
                Ok(())
            } else {
                Err(OxisError::invalid_input(format!("{name} must be >= 0")))
            }
        };
        let positive = |name: &str, v: f64| {
            if v.is_finite() && v > 0.0 {
                Ok(())
            } else {
                Err(OxisError::invalid_input(format!("{name} must be > 0")))
            }
        };
        match *self {
            Process::Gbm { mu, sigma } => {
                finite("mu", mu)?;
                nonneg("sigma", sigma)
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
                positive("kappa", kappa)?;
                finite("theta", theta)?;
                nonneg("sigma", sigma)
            }
            Process::Cir {
                kappa,
                theta,
                sigma,
            } => {
                positive("kappa", kappa)?;
                nonneg("theta", theta)?;
                nonneg("sigma", sigma)
            }
            Process::MertonJump {
                mu,
                sigma,
                lambda,
                jump_mean,
                jump_std,
            } => {
                finite("mu", mu)?;
                nonneg("sigma", sigma)?;
                nonneg("lambda", lambda)?;
                finite("jump_mean", jump_mean)?;
                nonneg("jump_std", jump_std)
            }
            Process::Heston {
                mu,
                v0,
                kappa,
                theta,
                xi,
                rho,
            } => {
                finite("mu", mu)?;
                nonneg("v0", v0)?;
                positive("kappa", kappa)?;
                nonneg("theta", theta)?;
                nonneg("xi", xi)?;
                if rho.is_finite() && (-1.0..=1.0).contains(&rho) {
                    Ok(())
                } else {
                    Err(OxisError::invalid_input("rho must be in [-1, 1]"))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gbm_moments_match_lognormal() {
        let p = Process::Gbm {
            mu: 0.05,
            sigma: 0.2,
        };
        let (mean, var) = p.analytic_moments(100.0, 1.0);
        assert!((mean - 100.0 * 0.05_f64.exp()).abs() < 1e-12);
        let expected_var = 100.0_f64.powi(2) * (0.1_f64).exp() * ((0.04_f64).exp() - 1.0);
        assert!((var.unwrap() - expected_var).abs() < 1e-9);
    }

    #[test]
    fn ou_reverts_to_theta_in_the_limit() {
        let p = Process::OrnsteinUhlenbeck {
            kappa: 2.0,
            theta: 0.03,
            sigma: 0.01,
        };
        let (mean, var) = p.analytic_moments(0.10, 1000.0);
        assert!((mean - 0.03).abs() < 1e-9);
        // Stationary variance σ²/(2κ).
        assert!((var.unwrap() - 0.01_f64.powi(2) / 4.0).abs() < 1e-12);
    }

    #[test]
    fn heston_mean_is_exact_and_variance_absent() {
        let p = Process::Heston {
            mu: 0.04,
            v0: 0.04,
            kappa: 1.5,
            theta: 0.04,
            xi: 0.3,
            rho: -0.6,
        };
        let (mean, var) = p.analytic_moments(100.0, 2.0);
        assert!((mean - 100.0 * (0.08_f64).exp()).abs() < 1e-12);
        assert!(var.is_none());
    }

    #[test]
    fn rejects_bad_parameters() {
        assert!(
            Process::Cir {
                kappa: 0.0,
                theta: 0.04,
                sigma: 0.2
            }
            .validate()
            .is_err()
        );
        assert!(
            Process::Heston {
                mu: 0.0,
                v0: 0.04,
                kappa: 1.0,
                theta: 0.04,
                xi: 0.3,
                rho: 1.5
            }
            .validate()
            .is_err()
        );
        assert!(
            Process::Gbm {
                mu: 0.05,
                sigma: -0.1
            }
            .validate()
            .is_err()
        );
    }
}
