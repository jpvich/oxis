//! `oxis price` — price an option and render the result.
//!
//! One entry point, model chosen by `--model`: `black-scholes` (closed-form,
//! European only), `binomial` (CRR tree, European or American via `--style`), or
//! `mc` (Monte Carlo — European by simulation, American via Longstaff-Schwartz).
//! Invalid combinations (e.g. `black-scholes` + `american`) fail with a clear
//! error. Monte Carlo also reports a standard error in the result.

use super::CliOptionType;
use oxis::core::output::render;
use oxis::core::{EuropeanOption, ExerciseStyle, MarketData, OptionType, OxisError, RunContext};
use oxis::pricing::{
    DEFAULT_STEPS, McConfig, PriceResult, binomial, black_scholes, lsm_american,
    monte_carlo_european,
};

/// Default time-grid size for the Longstaff-Schwartz American engine. Far
/// smaller than the binomial tree default because LSM stores every path.
const LSM_DEFAULT_STEPS: usize = 50;

/// Pricing model (`--model`).
#[derive(Clone, Copy, PartialEq, clap::ValueEnum)]
enum CliModel {
    /// Black-Scholes-Merton closed form (European only).
    BlackScholes,
    /// Cox-Ross-Rubinstein binomial tree (European or American).
    Binomial,
    /// Monte Carlo simulation (European; American via Longstaff-Schwartz).
    Mc,
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
    /// Time steps: binomial tree depth (default 1000), or Monte Carlo / LSM
    /// time-grid size (default 50). Ignored by black-scholes and European MC.
    #[arg(long)]
    steps: Option<usize>,
    /// Monte Carlo paths (only used by `--model mc`).
    #[arg(long)]
    paths: Option<usize>,
    /// Monte Carlo RNG seed (only used by `--model mc`).
    #[arg(long)]
    seed: Option<u64>,
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

    // (model label, price, optional Monte Carlo standard error).
    let (model_label, price, standard_error): (&'static str, f64, Option<f64>) = match args.model {
        CliModel::BlackScholes => {
            if matches!(style, ExerciseStyle::American) {
                return Err(OxisError::invalid_input(
                    "black-scholes prices European options only; use --model binomial or --model mc for american",
                )
                .into());
            }
            ("black-scholes", black_scholes(&option, &market)?, None)
        }
        CliModel::Binomial => {
            let steps = args.steps.unwrap_or(DEFAULT_STEPS);
            (
                "binomial",
                binomial(option_type, style, &market, args.strike, args.t, steps)?,
                None,
            )
        }
        CliModel::Mc => {
            let cfg = McConfig {
                paths: args.paths.unwrap_or_else(|| McConfig::default().paths),
                steps: args.steps.unwrap_or(LSM_DEFAULT_STEPS),
                seed: args.seed.unwrap_or_else(|| McConfig::default().seed),
            };
            match style {
                ExerciseStyle::European => {
                    let est = monte_carlo_european(&option, &market, &cfg)?;
                    ("monte-carlo", est.price, Some(est.standard_error))
                }
                ExerciseStyle::American => {
                    let est = lsm_american(option_type, &market, args.strike, args.t, &cfg)?;
                    ("longstaff-schwartz", est.price, Some(est.standard_error))
                }
            }
        }
    };

    let mut result = PriceResult::new(model_label, option_type, style, &option, &market, price);
    if let Some(se) = standard_error {
        result = result.with_standard_error(se);
    }

    println!("{}", render(&result, ctx.format));
    Ok(())
}
