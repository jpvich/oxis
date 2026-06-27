//! `oxis simulate` — simulate a stochastic process and report its terminal moments.
//!
//! Picks a [`Process`] with `--process` and reports the simulated terminal mean /
//! std against the closed-form moments, so the output doubles as a validation
//! readout. Parameters not used by the chosen process are ignored; each has a
//! sensible default so `oxis simulate --process gbm` runs out of the box.

use oxis_core::output::render;
use oxis_core::{OxisError, RunContext};
use oxis_stochastic::{Process, ProcessResult, SimConfig, simulate_terminal};

/// Which process to simulate (`--process`).
#[derive(Clone, Copy, PartialEq, clap::ValueEnum)]
enum CliProcess {
    Gbm,
    #[value(name = "ou")]
    OrnsteinUhlenbeck,
    Vasicek,
    Cir,
    Merton,
    Heston,
}

/// Flags for `oxis simulate`.
///
/// `allow_negative_numbers` lets values like `--rho -0.6` parse instead of being
/// mistaken for flags.
#[derive(clap::Args)]
#[command(allow_negative_numbers = true)]
pub struct SimulateArgs {
    /// Process to simulate.
    #[arg(long)]
    process: CliProcess,
    /// Initial state (asset price for GBM/Merton/Heston; level otherwise).
    #[arg(long, default_value_t = 100.0)]
    x0: f64,
    /// Horizon in years.
    #[arg(long, default_value_t = 1.0)]
    t: f64,
    /// Time steps on the grid.
    #[arg(long, default_value_t = 100)]
    steps: usize,
    /// Number of simulated paths.
    #[arg(long, default_value_t = 100_000)]
    paths: usize,
    /// RNG seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Drift (GBM, Merton, Heston).
    #[arg(long, default_value_t = 0.05)]
    mu: f64,
    /// Volatility / diffusion (GBM, OU, Vasicek, CIR, Merton).
    #[arg(long, default_value_t = 0.20)]
    sigma: f64,
    /// Mean-reversion speed (OU, Vasicek, CIR, Heston).
    #[arg(long, default_value_t = 1.0)]
    kappa: f64,
    /// Long-run mean / variance (OU, Vasicek, CIR, Heston).
    #[arg(long, default_value_t = 0.04)]
    theta: f64,
    /// Jump intensity per year (Merton).
    #[arg(long, default_value_t = 0.5)]
    lambda: f64,
    /// Mean log-jump size (Merton).
    #[arg(long = "jump-mean", default_value_t = -0.10)]
    jump_mean: f64,
    /// Std of the log-jump size (Merton).
    #[arg(long = "jump-std", default_value_t = 0.15)]
    jump_std: f64,
    /// Initial variance (Heston).
    #[arg(long, default_value_t = 0.04)]
    v0: f64,
    /// Volatility of variance (Heston).
    #[arg(long, default_value_t = 0.30)]
    xi: f64,
    /// Price/variance correlation (Heston).
    #[arg(long, default_value_t = -0.60)]
    rho: f64,
}

/// Build the process, simulate, render the moment summary.
pub fn run(args: SimulateArgs, ctx: &RunContext) -> anyhow::Result<()> {
    let process = match args.process {
        CliProcess::Gbm => Process::Gbm {
            mu: args.mu,
            sigma: args.sigma,
        },
        CliProcess::OrnsteinUhlenbeck => Process::OrnsteinUhlenbeck {
            kappa: args.kappa,
            theta: args.theta,
            sigma: args.sigma,
        },
        CliProcess::Vasicek => Process::Vasicek {
            kappa: args.kappa,
            theta: args.theta,
            sigma: args.sigma,
        },
        CliProcess::Cir => Process::Cir {
            kappa: args.kappa,
            theta: args.theta,
            sigma: args.sigma,
        },
        CliProcess::Merton => Process::MertonJump {
            mu: args.mu,
            sigma: args.sigma,
            lambda: args.lambda,
            jump_mean: args.jump_mean,
            jump_std: args.jump_std,
        },
        CliProcess::Heston => Process::Heston {
            mu: args.mu,
            v0: args.v0,
            kappa: args.kappa,
            theta: args.theta,
            xi: args.xi,
            rho: args.rho,
        },
    };

    // Surface a domain error (e.g. x0 ≤ 0 for a price process) before simulating.
    process
        .validate()
        .map_err(|e: OxisError| anyhow::anyhow!(e))?;

    let cfg = SimConfig {
        paths: args.paths,
        steps: args.steps,
        seed: args.seed,
    };
    let sample = simulate_terminal(&process, args.x0, args.t, &cfg)?;
    let result = ProcessResult::from_simulation(&process, args.x0, args.t, &cfg, &sample);

    println!("{}", render(&result, ctx.format));
    Ok(())
}
