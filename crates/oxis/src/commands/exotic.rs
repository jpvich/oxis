//! `oxis exotic` — price a barrier, lookback, or Asian option.
//!
//! One entry point, family chosen by `--kind`. Barrier needs `--barrier` and
//! `--barrier-type`; lookback needs `--strike-type`; Asian needs `--average`
//! (and, for `arithmetic`, Monte Carlo `--fixings`/`--paths`/`--seed`). The
//! closed-form families price exactly; the arithmetic-average Asian reports a
//! Monte Carlo standard error in the result.

use super::CliOptionType;
use oxis::core::output::render;
use oxis::core::{EuropeanOption, MarketData, OptionType, OxisError, RunContext};
use oxis::pricing::{
    BarrierType, ExoticResult, LookbackStrike, arithmetic_asian_price, barrier_price,
    geometric_asian_price, lookback_price,
};
use oxis::stochastic::SimConfig;

/// Exotic family (`--kind`).
#[derive(Clone, Copy, PartialEq, clap::ValueEnum)]
enum CliExoticKind {
    /// Single-barrier knock-in/knock-out option.
    Barrier,
    /// Continuous lookback option.
    Lookback,
    /// Average-price Asian option.
    Asian,
}

/// Barrier direction + knock sense (`--barrier-type`).
#[derive(Clone, Copy, clap::ValueEnum)]
enum CliBarrierType {
    DownIn,
    DownOut,
    UpIn,
    UpOut,
}

impl From<CliBarrierType> for BarrierType {
    fn from(value: CliBarrierType) -> Self {
        match value {
            CliBarrierType::DownIn => BarrierType::DownIn,
            CliBarrierType::DownOut => BarrierType::DownOut,
            CliBarrierType::UpIn => BarrierType::UpIn,
            CliBarrierType::UpOut => BarrierType::UpOut,
        }
    }
}

/// Lookback strike convention (`--strike-type`).
#[derive(Clone, Copy, clap::ValueEnum)]
enum CliLookbackStrike {
    Floating,
    Fixed,
}

impl From<CliLookbackStrike> for LookbackStrike {
    fn from(value: CliLookbackStrike) -> Self {
        match value {
            CliLookbackStrike::Floating => LookbackStrike::Floating,
            CliLookbackStrike::Fixed => LookbackStrike::Fixed,
        }
    }
}

/// Asian averaging convention (`--average`).
#[derive(Clone, Copy, clap::ValueEnum)]
enum CliAverage {
    /// Geometric average (closed form).
    Geometric,
    /// Arithmetic average (Monte Carlo).
    Arithmetic,
}

/// Flags for `oxis exotic`.
///
/// `allow_negative_numbers` lets values like `--rate -0.01` parse (negative rates
/// are real) instead of being mistaken for flags.
#[derive(clap::Args)]
#[command(allow_negative_numbers = true)]
pub struct ExoticArgs {
    /// Exotic family.
    #[arg(long)]
    kind: CliExoticKind,
    /// Spot price of the underlying.
    #[arg(long)]
    spot: f64,
    /// Strike price (ignored by floating-strike lookbacks).
    #[arg(long)]
    strike: f64,
    /// Continuously compounded risk-free rate.
    #[arg(long)]
    rate: f64,
    /// Volatility (annualized).
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
    /// Barrier level (barrier options).
    #[arg(long)]
    barrier: Option<f64>,
    /// Barrier direction + knock sense (barrier options).
    #[arg(long = "barrier-type", value_enum)]
    barrier_type: Option<CliBarrierType>,
    /// Strike convention (lookback options).
    #[arg(long = "strike-type", value_enum)]
    strike_type: Option<CliLookbackStrike>,
    /// Averaging convention (Asian options).
    #[arg(long, value_enum)]
    average: Option<CliAverage>,
    /// Number of fixing dates (arithmetic-average Asian).
    #[arg(long, default_value_t = 50)]
    fixings: usize,
    /// Monte Carlo paths (arithmetic-average Asian).
    #[arg(long, default_value_t = 100_000)]
    paths: usize,
    /// Monte Carlo RNG seed (arithmetic-average Asian).
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

/// Build the plain inputs, call the pure core, render the result.
pub fn run(args: ExoticArgs, ctx: &RunContext) -> anyhow::Result<()> {
    let option_type: OptionType = args.option_type.into();
    let option = EuropeanOption {
        strike: args.strike,
        expiry_years: args.t,
        option_type,
    };
    let market = MarketData::new(args.spot, args.rate, args.vol, args.dividend_yield);

    // (model label, variant label, barrier level, price, optional MC std error).
    let (model, variant, barrier, price, se): (
        &'static str,
        Option<&'static str>,
        Option<f64>,
        f64,
        Option<f64>,
    ) = match args.kind {
        CliExoticKind::Barrier => {
            let bt = args
                .barrier_type
                .ok_or_else(|| OxisError::invalid_input("barrier options need --barrier-type"))?;
            let h = args
                .barrier
                .ok_or_else(|| OxisError::invalid_input("barrier options need --barrier"))?;
            let bt: BarrierType = bt.into();
            let price = barrier_price(&option, &market, bt, h)?;
            ("barrier", Some(bt.as_str()), Some(h), price, None)
        }
        CliExoticKind::Lookback => {
            let st = args
                .strike_type
                .ok_or_else(|| OxisError::invalid_input("lookback options need --strike-type"))?;
            let st: LookbackStrike = st.into();
            let price = lookback_price(&option, &market, st)?;
            ("lookback", Some(st.as_str()), None, price, None)
        }
        CliExoticKind::Asian => {
            let avg = args
                .average
                .ok_or_else(|| OxisError::invalid_input("Asian options need --average"))?;
            match avg {
                CliAverage::Geometric => {
                    let price = geometric_asian_price(&option, &market)?;
                    ("asian", Some("geometric"), None, price, None)
                }
                CliAverage::Arithmetic => {
                    let cfg = SimConfig {
                        paths: args.paths,
                        steps: 0,
                        seed: args.seed,
                    };
                    let est = arithmetic_asian_price(&option, &market, args.fixings, &cfg)?;
                    (
                        "asian",
                        Some("arithmetic"),
                        None,
                        est.price,
                        Some(est.standard_error),
                    )
                }
            }
        }
    };

    let result = ExoticResult {
        model,
        option_type,
        spot: args.spot,
        strike: args.strike,
        rate: args.rate,
        volatility: args.vol,
        time: args.t,
        dividend_yield: args.dividend_yield,
        barrier,
        variant,
        price,
        standard_error: se,
    };

    println!("{}", render(&result, ctx.format));
    Ok(())
}
