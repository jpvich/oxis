//! `oxis greeks` — analytic Black-Scholes Greeks for a European option.

use super::CliOptionType;
use oxis_core::output::render;
use oxis_core::{EuropeanOption, MarketData, OptionType, RunContext};
use oxis_greeks::{GreeksResult, analytic_greeks};

/// Flags for `oxis greeks`.
#[derive(clap::Args)]
#[command(allow_negative_numbers = true)]
pub struct GreeksArgs {
    /// Spot price of the underlying.
    #[arg(long)]
    spot: f64,
    /// Strike price.
    #[arg(long)]
    strike: f64,
    /// Continuously compounded risk-free rate (e.g. 0.05).
    #[arg(long)]
    rate: f64,
    /// Volatility (annualized, e.g. 0.2).
    #[arg(long)]
    vol: f64,
    /// Time to expiry, in years.
    #[arg(long)]
    t: f64,
    /// Call or put.
    #[arg(long = "type", value_enum)]
    option_type: CliOptionType,
    /// Continuously compounded dividend yield.
    #[arg(long = "dividend-yield", default_value_t = 0.0)]
    dividend_yield: f64,
}

/// Build inputs, call the pure core, render the Greeks.
pub fn run(args: GreeksArgs, ctx: &RunContext) -> anyhow::Result<()> {
    let option_type: OptionType = args.option_type.into();
    let option = EuropeanOption {
        strike: args.strike,
        expiry_years: args.t,
        option_type,
    };
    let market = MarketData::new(args.spot, args.rate, args.vol, args.dividend_yield);

    let greeks = analytic_greeks(&option, &market)?;
    let result = GreeksResult::new("analytic", &option, &market, &greeks);

    println!("{}", render(&result, ctx.format));
    Ok(())
}
