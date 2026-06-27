//! CLI subcommands. Each is a thin `run(ctx)` wrapper over a pure module core.

mod bond;
mod curve;
mod exotic;
mod greeks;
mod implied_vol;
mod portfolio;
mod price;
mod simulate;
mod stats;

use oxis_core::{OptionType, RunContext};

/// Call or put as a CLI value (`--type call|put`), shared by the commands.
/// Mapped to the core type so `clap` stays out of `oxis-core`.
#[derive(Clone, Copy, clap::ValueEnum)]
pub(crate) enum CliOptionType {
    Call,
    Put,
}

impl From<CliOptionType> for OptionType {
    fn from(value: CliOptionType) -> Self {
        match value {
            CliOptionType::Call => OptionType::Call,
            CliOptionType::Put => OptionType::Put,
        }
    }
}

/// The top-level `oxis` subcommands.
#[derive(clap::Subcommand)]
pub enum Command {
    /// Price an option (`--model black-scholes|binomial`, `--style`).
    Price(price::PriceArgs),
    /// Compute analytic Black-Scholes Greeks for a European option.
    Greeks(greeks::GreeksArgs),
    /// Solve for the implied volatility given a market price.
    #[command(name = "implied-vol")]
    ImpliedVol(implied_vol::ImpliedVolArgs),
    /// Build a yield curve and query discount / zero / forward rates.
    Curve(curve::CurveArgs),
    /// Price a fixed-rate bond and report yield / duration / convexity.
    Bond(bond::BondArgs),
    /// Price an exotic option (barrier, lookback, or Asian).
    Exotic(exotic::ExoticArgs),
    /// Simulate a stochastic process and report its terminal moments.
    Simulate(simulate::SimulateArgs),
    /// Compute descriptive, risk, and performance statistics for a series.
    Stats(stats::StatsArgs),
    /// Portfolio valuation, performance, allocation, risk, and optimization.
    Portfolio(portfolio::PortfolioArgs),
}

impl Command {
    /// Dispatch to the selected subcommand.
    pub fn run(self, ctx: &RunContext) -> anyhow::Result<()> {
        match self {
            Command::Price(args) => price::run(args, ctx),
            Command::Greeks(args) => greeks::run(args, ctx),
            Command::ImpliedVol(args) => implied_vol::run(args, ctx),
            Command::Curve(args) => curve::run(args, ctx),
            Command::Bond(args) => bond::run(args, ctx),
            Command::Exotic(args) => exotic::run(args, ctx),
            Command::Simulate(args) => simulate::run(args, ctx),
            Command::Stats(args) => stats::run(args, ctx),
            Command::Portfolio(args) => portfolio::run(args, ctx),
        }
    }
}
