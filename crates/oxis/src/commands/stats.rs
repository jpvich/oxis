//! `oxis stats` — descriptive, risk, and performance statistics for a series.
//!
//! Exactly one primary input is given: `--returns` (a periodic-returns series),
//! `--prices` (an equity/price path — returns are derived and drawdown / Calmar
//! become available), or `--values` (a generic numeric sample → descriptive
//! statistics only). An optional `--benchmark` enables beta, correlation,
//! covariance, tracking error, and information ratio; metrics whose inputs are
//! absent render as empty / `null`.

use oxis_core::output::render;
use oxis_core::{OxisError, RunContext};
use oxis_stats::{SampleKind, StatsRequest, assemble, simple_returns};

/// Flags for `oxis stats`.
///
/// `allow_negative_numbers` lets values like `--returns -0.01,0.02` parse
/// (negative returns are real) instead of being mistaken for flags.
#[derive(clap::Args)]
#[command(allow_negative_numbers = true)]
pub struct StatsArgs {
    /// Comma-separated periodic returns (e.g. `0.01,-0.02,0.015`).
    #[arg(long, value_delimiter = ',')]
    returns: Option<Vec<f64>>,
    /// Comma-separated price/equity path; returns are derived from it.
    #[arg(long, value_delimiter = ',')]
    prices: Option<Vec<f64>>,
    /// Comma-separated generic sample (descriptive statistics only).
    #[arg(long, value_delimiter = ',')]
    values: Option<Vec<f64>>,
    /// Comma-separated benchmark returns (enables beta / TE / IR / corr / cov).
    #[arg(long, value_delimiter = ',')]
    benchmark: Option<Vec<f64>>,
    /// Per-period risk-free rate / minimum acceptable return.
    #[arg(long = "risk-free", default_value_t = 0.0)]
    risk_free: f64,
    /// Periods per year (annualization factor).
    #[arg(long = "periods-per-year", default_value_t = 252.0)]
    periods_per_year: f64,
    /// Confidence level for VaR / ES.
    #[arg(long, default_value_t = 0.95)]
    confidence: f64,
    /// Extra autocorrelation lag to report.
    #[arg(long)]
    lag: Option<usize>,
}

/// Resolve the flags, compute the statistics, render the result.
pub fn run(args: StatsArgs, ctx: &RunContext) -> anyhow::Result<()> {
    // Resolve the single primary input into an owned sample + its kind.
    let (sample, kind): (Vec<f64>, SampleKind) = match (&args.returns, &args.prices, &args.values) {
        (Some(r), None, None) => (r.clone(), SampleKind::Returns),
        (None, Some(p), None) => (simple_returns(p)?, SampleKind::Returns),
        (None, None, Some(v)) => (v.clone(), SampleKind::Values),
        (None, None, None) => {
            return Err(OxisError::invalid_input(
                "provide exactly one of --returns, --prices, or --values",
            )
            .into());
        }
        _ => {
            return Err(OxisError::invalid_input(
                "use only one of --returns, --prices, or --values",
            )
            .into());
        }
    };

    let req = StatsRequest {
        sample: &sample,
        kind,
        prices: args.prices.as_deref(),
        benchmark: args.benchmark.as_deref(),
        risk_free: args.risk_free,
        periods_per_year: args.periods_per_year,
        confidence: args.confidence,
        lag: args.lag,
    };
    let report = assemble(&req)?;
    println!("{}", render(&report, ctx.format));
    Ok(())
}
