//! `oxis price` — price an option and render the result.
//!
//! One entry point, model chosen by `--model`: `black-scholes` (closed-form,
//! European only) or `binomial` (CRR tree, European or American via `--style`).
//! Invalid combinations (e.g. `black-scholes` + `american`) fail with a clear
//! error. Monte Carlo joins here as a `--model mc` in a later milestone.

use super::CliOptionType;
use oxis_core::output::render;
use oxis_core::{EuropeanOption, ExerciseStyle, MarketData, OptionType, OxisError, RunContext};
use oxis_pricing::{DEFAULT_STEPS, PriceResult, binomial, black_scholes};

/// Pricing model (`--model`).
#[derive(Clone, Copy, PartialEq, clap::ValueEnum)]
enum CliModel {
    /// Black-Scholes-Merton closed form (European only).
    BlackScholes,
    /// Cox-Ross-Rubinstein binomial tree (European or American).
    Binomial,
}

impl CliModel {
    fn as_str(self) -> &'static str {
        match self {
            CliModel::BlackScholes => "black-scholes",
            CliModel::Binomial => "binomial",
        }
    }
}

/// Exercise style (`--style`).
#[derive(Clone, Copy, clap::ValueEnum)]
enum CliExercise {
    European,
    American,
}

impl From<CliExercise> for ExerciseStyle {
    fn from(value: CliExercise) -> Self {
        match value {
            CliExercise::European => ExerciseStyle::European,
            CliExercise::American => ExerciseStyle::American,
        }
    }
}

/// Flags for `oxis price`.
///
/// `allow_negative_numbers` lets values like `--rate -0.01` parse (negative
/// interest rates are real) instead of being mistaken for flags.
#[derive(clap::Args)]
#[command(allow_negative_numbers = true)]
pub struct PriceArgs {
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
    /// Pricing model.
    #[arg(long, value_enum, default_value_t = CliModel::BlackScholes)]
    model: CliModel,
    /// Exercise style.
    #[arg(long, value_enum, default_value_t = CliExercise::European)]
    style: CliExercise,
    /// Binomial tree steps (only used by `--model binomial`).
    #[arg(long, default_value_t = DEFAULT_STEPS)]
    steps: usize,
}

/// Build the plain inputs, call the pure core, render the result.
pub fn run(args: PriceArgs, ctx: &RunContext) -> anyhow::Result<()> {
    let option_type: OptionType = args.option_type.into();
    let style: ExerciseStyle = args.style.into();
    let option = EuropeanOption {
        strike: args.strike,
        expiry_years: args.t,
        option_type,
    };
    let market = MarketData::new(args.spot, args.rate, args.vol, args.dividend_yield);

    let price = match args.model {
        CliModel::BlackScholes => {
            if matches!(style, ExerciseStyle::American) {
                return Err(OxisError::invalid_input(
                    "black-scholes prices European options only; use --model binomial for american",
                )
                .into());
            }
            black_scholes(&option, &market)?
        }
        CliModel::Binomial => {
            binomial(option_type, style, &market, args.strike, args.t, args.steps)?
        }
    };

    let result = PriceResult::new(
        args.model.as_str(),
        option_type,
        style,
        &option,
        &market,
        price,
    );

    println!("{}", render(&result, ctx.format));
    Ok(())
}
