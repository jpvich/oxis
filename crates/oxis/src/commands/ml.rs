//! `oxis ml` — machine-learning pricing (the differential-ML surrogate).
//!
//! `oxis ml price` trains a small twin network on simulated payoffs and their
//! pathwise differentials, then reports its price and delta next to the classical
//! Black-Scholes values. Training runs in-process (a few seconds); for batch or
//! grid work prefer the Python API. Matrix-free by design — this is a single
//! European contract, 1-D in spot.

use super::CliOptionType;
use oxis::core::output::render;
use oxis::core::{MarketData, OxisError, RunContext};
use oxis::ml::{
    AmericanMlConfig, BsSpec, TrainConfig, deep_lsm_price, differential_ml_price, dos_price,
};

/// Flags for `oxis ml`.
#[derive(clap::Args)]
pub struct MlArgs {
    #[command(subcommand)]
    command: MlCmd,
}

#[derive(clap::Subcommand)]
enum MlCmd {
    /// Train a differential-ML surrogate and price a European option vs Black-Scholes.
    Price(PriceArgs),
    /// Price an American option with a neural engine vs the binomial baseline.
    American(AmericanArgs),
}

/// Neural American pricing method.
#[derive(clap::ValueEnum, Clone, Copy)]
enum AmericanMethod {
    /// Longstaff-Schwartz with a per-date neural continuation regression.
    DeepLsm,
    /// Deep Optimal Stopping (per-date stop-probability networks).
    Dos,
}

#[derive(clap::Args)]
#[command(allow_negative_numbers = true)]
struct PriceArgs {
    /// Spot price (also the training centre and the query point).
    #[arg(long)]
    spot: f64,
    /// Strike.
    #[arg(long)]
    strike: f64,
    /// Continuously compounded risk-free rate.
    #[arg(long)]
    rate: f64,
    /// Volatility.
    #[arg(long)]
    vol: f64,
    /// Time to maturity, in years.
    #[arg(long)]
    maturity: f64,
    /// Call or put.
    #[arg(long = "type", value_enum, default_value_t = CliOptionType::Call)]
    option_type: CliOptionType,
    /// Number of simulated training samples.
    #[arg(long, default_value_t = 4096)]
    samples: usize,
    /// Training epochs.
    #[arg(long, default_value_t = 60)]
    epochs: usize,
    /// Log-normal spread of training spots (multiple of σ√τ).
    #[arg(long, default_value_t = 2.0)]
    spread: f64,
    /// RNG seed (fixes data, init, and shuffling).
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Hidden-layer widths, comma-separated (e.g. `30,30`).
    #[arg(long, value_delimiter = ',', default_value = "30,30")]
    hidden: Vec<usize>,
}

#[derive(clap::Args)]
#[command(allow_negative_numbers = true)]
struct AmericanArgs {
    /// Pricing method.
    #[arg(long, value_enum, default_value_t = AmericanMethod::DeepLsm)]
    method: AmericanMethod,
    /// Spot price.
    #[arg(long)]
    spot: f64,
    /// Strike.
    #[arg(long)]
    strike: f64,
    /// Continuously compounded risk-free rate.
    #[arg(long)]
    rate: f64,
    /// Volatility.
    #[arg(long)]
    vol: f64,
    /// Time to maturity, in years.
    #[arg(long)]
    maturity: f64,
    /// Call or put.
    #[arg(long = "type", value_enum, default_value_t = CliOptionType::Put)]
    option_type: CliOptionType,
    /// Continuously compounded dividend yield.
    #[arg(long, default_value_t = 0.0)]
    dividend_yield: f64,
    /// Number of simulated paths (drawn as antithetic pairs).
    #[arg(long, default_value_t = 8192)]
    paths: usize,
    /// Number of exercise dates in the time grid.
    #[arg(long, default_value_t = 50)]
    steps: usize,
    /// Training epochs per exercise date.
    #[arg(long, default_value_t = 20)]
    epochs: usize,
    /// RNG seed (fixes paths, init, and shuffling).
    #[arg(long, default_value_t = 11)]
    seed: u64,
    /// Hidden-layer widths, comma-separated (e.g. `16`).
    #[arg(long, value_delimiter = ',', default_value = "16")]
    hidden: Vec<usize>,
}

/// Dispatch the `oxis ml` subcommand.
pub fn run(args: MlArgs, ctx: &RunContext) -> anyhow::Result<()> {
    match args.command {
        MlCmd::Price(a) => run_price(a, ctx),
        MlCmd::American(a) => run_american(a, ctx),
    }
}

fn run_price(a: PriceArgs, ctx: &RunContext) -> anyhow::Result<()> {
    if a.hidden.is_empty() {
        return Err(OxisError::invalid_input("--hidden must list at least one width").into());
    }
    let cfg = TrainConfig {
        spec: BsSpec {
            spot: a.spot,
            strike: a.strike,
            rate: a.rate,
            vol: a.vol,
            maturity: a.maturity,
            option_type: a.option_type.into(),
        },
        n_samples: a.samples,
        hidden: a.hidden,
        epochs: a.epochs,
        spread: a.spread,
        seed: a.seed,
    };
    let report = differential_ml_price(&cfg)?;
    println!("{}", render(&report, ctx.format));
    Ok(())
}

fn run_american(a: AmericanArgs, ctx: &RunContext) -> anyhow::Result<()> {
    if a.hidden.is_empty() {
        return Err(OxisError::invalid_input("--hidden must list at least one width").into());
    }
    let cfg = AmericanMlConfig {
        market: MarketData::new(a.spot, a.rate, a.vol, a.dividend_yield),
        strike: a.strike,
        expiry: a.maturity,
        paths: a.paths,
        steps: a.steps,
        seed: a.seed,
        hidden: a.hidden,
        epochs: a.epochs,
    };
    let report = match a.method {
        AmericanMethod::DeepLsm => deep_lsm_price(a.option_type.into(), &cfg)?,
        AmericanMethod::Dos => dos_price(a.option_type.into(), &cfg)?,
    };
    println!("{}", render(&report, ctx.format));
    Ok(())
}
