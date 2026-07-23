//! `oxis implied-vol` — solve for the Black-Scholes implied volatility that
//! reproduces an observed market price.

use super::CliOptionType;
use oxis::core::output::render;
use oxis::core::{EuropeanOption, MarketData, OptionType, RunContext};
use oxis::pricing::{ImpliedVolResult, implied_volatility};

/// Flags for `oxis implied-vol`. Note there is no `--vol`: volatility is the
/// unknown being solved for; `--price` is the observed market price.
#[derive(clap::Args)]
#[command(allow_negative_numbers = true)]
pub struct ImpliedVolArgs {
    /// Observed market price of the option.
    #[arg(long)]
    price: f64,
    /// Spot price of the underlying.
    #[arg(long)]
    spot: f64,
    /// Strike price.
    #[arg(long)]
    strike: f64,
    /// Continuously compounded risk-free rate (e.g. 0.05).
    #[arg(long)]
    rate: f64,
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

/// Build inputs, solve, render the implied volatility.
pub fn run(args: ImpliedVolArgs, ctx: &RunContext) -> anyhow::Result<()> {
    let option_type: OptionType = args.option_type.into();
    let option = EuropeanOption {
        strike: args.strike,
        expiry_years: args.t,
        option_type,
    };
    // Volatility field is ignored by the solver (it is the unknown).
    let market = MarketData::new(args.spot, args.rate, 0.0, args.dividend_yield);

    let iv = implied_volatility(&option, args.price, &market)?;
    let result = ImpliedVolResult::new(&option, &market, args.price, iv);

    println!("{}", render(&result, ctx.format));
    Ok(())
}
