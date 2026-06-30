//! `oxis curve` — build a yield curve and query it.
//!
//! The curve is built one of two ways: a flat rate (`--flat`), or interpolated
//! pillars (`--times` paired with either `--rates` or `--discounts`). The
//! interpolation scheme is chosen by `--interp`. The result reports the discount
//! factor and zero rate at `--at`, and — if `--forward-to` is given — the forward
//! rate from `--at` to that time.

use oxis_core::output::render;
use oxis_core::{OxisError, RunContext};
use oxis_curves::{Interpolation, YieldCurve};

/// Interpolation scheme (`--interp`).
#[derive(Clone, Copy, clap::ValueEnum)]
enum CliInterp {
    /// Linear in zero rates.
    Linear,
    /// Linear in log discount factors (piecewise-constant forwards).
    LogLinear,
    /// Natural cubic spline in zero rates.
    NaturalCubic,
}

impl From<CliInterp> for Interpolation {
    fn from(value: CliInterp) -> Self {
        match value {
            CliInterp::Linear => Interpolation::Linear,
            CliInterp::LogLinear => Interpolation::LogLinear,
            CliInterp::NaturalCubic => Interpolation::NaturalCubic,
        }
    }
}

/// Flags for `oxis curve`.
///
/// `allow_negative_numbers` lets values like `--flat -0.01` parse (negative rates
/// are real) instead of being mistaken for flags.
#[derive(clap::Args)]
#[command(allow_negative_numbers = true)]
pub struct CurveArgs {
    /// Flat continuously-compounded rate (mutually exclusive with --times).
    #[arg(long)]
    flat: Option<f64>,
    /// Comma-separated pillar times in years (e.g. `0.5,1,2,5`).
    #[arg(long, value_delimiter = ',')]
    times: Option<Vec<f64>>,
    /// Comma-separated continuously-compounded zero rates, one per pillar.
    #[arg(long, value_delimiter = ',')]
    rates: Option<Vec<f64>>,
    /// Comma-separated discount factors, one per pillar.
    #[arg(long, value_delimiter = ',')]
    discounts: Option<Vec<f64>>,
    /// Interpolation scheme (ignored for --flat).
    #[arg(long, value_enum, default_value_t = CliInterp::LogLinear)]
    interp: CliInterp,
    /// Query time in years.
    #[arg(long)]
    at: f64,
    /// If set, also report the forward rate from --at to this time.
    #[arg(long = "forward-to")]
    forward_to: Option<f64>,
}

/// Build the curve from the flags, query it, render the result.
pub fn run(args: CurveArgs, ctx: &RunContext) -> anyhow::Result<()> {
    let curve = build_curve(&args)?;
    let query = curve.query(args.at, args.forward_to)?;
    println!("{}", render(&query, ctx.format));
    Ok(())
}

/// Resolve the flag combination into a [`YieldCurve`], or a clear error.
fn build_curve(args: &CurveArgs) -> Result<YieldCurve, OxisError> {
    let interp: Interpolation = args.interp.into();
    match (&args.flat, &args.times) {
        (Some(_), Some(_)) => Err(OxisError::invalid_input(
            "use either --flat or --times, not both",
        )),
        (Some(rate), None) => {
            if args.rates.is_some() || args.discounts.is_some() {
                return Err(OxisError::invalid_input(
                    "--flat takes no --rates/--discounts",
                ));
            }
            YieldCurve::flat(*rate)
        }
        (None, Some(times)) => match (&args.rates, &args.discounts) {
            (Some(rates), None) => YieldCurve::from_zero_rates(times, rates, interp),
            (None, Some(dfs)) => YieldCurve::from_discount_factors(times, dfs, interp),
            (Some(_), Some(_)) => Err(OxisError::invalid_input(
                "use either --rates or --discounts, not both",
            )),
            (None, None) => Err(OxisError::invalid_input(
                "--times needs --rates or --discounts",
            )),
        },
        (None, None) => Err(OxisError::invalid_input(
            "provide --flat <rate> or --times <list> with --rates/--discounts",
        )),
    }
}
