//! `oxis portfolio <subcommand>` — holdings valuation, performance, allocation,
//! risk aggregation, and Markowitz optimization.
//!
//! `value`/`twr`/`mwr`/`allocate` are convenient on the command line; `risk` and
//! `optimize` take N-asset matrices (passed as repeated row flags) and are more
//! comfortable from the Python API, which accepts lists of lists directly.

use oxis_core::output::render;
use oxis_core::{Date, OutputFormat, OxisError, RunContext};
use oxis_portfolio::{
    AllocationReport, Holding, PerformanceReport, efficient_frontier_point, min_variance_weights,
    mwr as mwr_core, optimization_report, portfolio_risk, tangency_weights, twr as twr_core,
    value_holdings, weights as alloc_weights,
};

/// Flags for `oxis portfolio`.
#[derive(clap::Args)]
pub struct PortfolioArgs {
    #[command(subcommand)]
    cmd: PortfolioCmd,
}

#[derive(clap::Subcommand)]
enum PortfolioCmd {
    /// Mark-to-market value holdings (`--holding SYM:QTY:COST:PRICE`, repeated).
    Value(ValueArgs),
    /// Time-weighted return from valuations and flows.
    Twr(TwrArgs),
    /// Money-weighted return (IRR) from dated cash flows.
    Mwr(MwrArgs),
    /// Allocation weights from market values.
    Allocate(AllocateArgs),
    /// Portfolio volatility + VaR from asset return rows and weights.
    Risk(RiskArgs),
    /// Markowitz mean-variance optimization (unconstrained).
    Optimize(OptimizeArgs),
}

#[derive(clap::Args)]
#[command(allow_negative_numbers = true)]
struct ValueArgs {
    /// A holding as `SYMBOL:QUANTITY:UNIT_COST:PRICE` (repeat for each).
    #[arg(long = "holding", required = true)]
    holdings: Vec<String>,
}

#[derive(clap::Args)]
#[command(allow_negative_numbers = true)]
struct TwrArgs {
    /// Period-boundary valuations `V0,V1,...,Vn`.
    #[arg(long, value_delimiter = ',', required = true)]
    values: Vec<f64>,
    /// External net flow at the start of each sub-period (length = values − 1).
    #[arg(long, value_delimiter = ',', required = true)]
    flows: Vec<f64>,
}

#[derive(clap::Args)]
#[command(allow_negative_numbers = true)]
struct MwrArgs {
    /// A dated cash flow as `YYYY-MM-DD:AMOUNT` (invested negative, received
    /// positive); repeat for each.
    #[arg(long = "cashflow", required = true)]
    cashflows: Vec<String>,
}

#[derive(clap::Args)]
#[command(allow_negative_numbers = true)]
struct AllocateArgs {
    /// Market values, one per position.
    #[arg(long = "market-values", value_delimiter = ',', required = true)]
    market_values: Vec<f64>,
}

#[derive(clap::Args)]
#[command(allow_negative_numbers = true)]
struct RiskArgs {
    /// One asset's return series as a comma list (repeat for each asset).
    #[arg(long = "returns-row", required = true)]
    returns_rows: Vec<String>,
    /// Portfolio weights, one per asset.
    #[arg(long, value_delimiter = ',', required = true)]
    weights: Vec<f64>,
    /// Confidence level for VaR.
    #[arg(long, default_value_t = 0.95)]
    confidence: f64,
    /// Periods per year (annualization factor).
    #[arg(long = "periods-per-year", default_value_t = 252.0)]
    periods_per_year: f64,
}

#[derive(clap::Args)]
#[command(allow_negative_numbers = true)]
struct OptimizeArgs {
    /// Expected returns, one per asset.
    #[arg(long, value_delimiter = ',', required = true)]
    mean: Vec<f64>,
    /// One row of the covariance matrix as a comma list (repeat, row-major).
    #[arg(long = "cov-row", required = true)]
    cov_rows: Vec<String>,
    /// Which portfolio: `min-variance`, `tangency`, or `frontier`.
    #[arg(long, default_value = "min-variance")]
    flavor: String,
    /// Risk-free rate (for `tangency`).
    #[arg(long, default_value_t = 0.0)]
    rf: f64,
    /// Target expected return (for `frontier`).
    #[arg(long)]
    target: Option<f64>,
}

/// Dispatch the portfolio subcommand.
pub fn run(args: PortfolioArgs, ctx: &RunContext) -> anyhow::Result<()> {
    match args.cmd {
        PortfolioCmd::Value(a) => value(a, ctx),
        PortfolioCmd::Twr(a) => {
            let r = PerformanceReport {
                twr: Some(twr_core(&a.values, &a.flows)?),
                mwr: None,
            };
            println!("{}", render(&r, ctx.format));
            Ok(())
        }
        PortfolioCmd::Mwr(a) => {
            let cfs = a
                .cashflows
                .iter()
                .map(|s| parse_cashflow(s))
                .collect::<Result<Vec<_>, _>>()?;
            let r = PerformanceReport {
                twr: None,
                mwr: Some(mwr_core(&cfs)?),
            };
            println!("{}", render(&r, ctx.format));
            Ok(())
        }
        PortfolioCmd::Allocate(a) => {
            let w = alloc_weights(&a.market_values)?;
            let r = AllocationReport {
                n_assets: w.len(),
                weights: w,
            };
            println!("{}", render(&r, ctx.format));
            Ok(())
        }
        PortfolioCmd::Risk(a) => {
            let rows = parse_rows(&a.returns_rows)?;
            let r = portfolio_risk(&rows, &a.weights, a.periods_per_year, a.confidence)?;
            println!("{}", render(&r, ctx.format));
            Ok(())
        }
        PortfolioCmd::Optimize(a) => optimize(a, ctx),
    }
}

fn value(a: ValueArgs, ctx: &RunContext) -> anyhow::Result<()> {
    let mut holdings = Vec::with_capacity(a.holdings.len());
    let mut prices = Vec::with_capacity(a.holdings.len());
    for spec in &a.holdings {
        let (h, price) = parse_holding(spec)?;
        holdings.push(h);
        prices.push(price);
    }
    let result = value_holdings(&holdings, &prices)?;
    println!("{}", render(&result, ctx.format));
    // For human / TSV, also print the per-holding breakdown (JSON already nests it).
    if ctx.format != OutputFormat::Json {
        for h in &result.holdings {
            println!("{}", render(h, ctx.format));
        }
    }
    Ok(())
}

fn optimize(a: OptimizeArgs, ctx: &RunContext) -> anyhow::Result<()> {
    let cov = parse_rows(&a.cov_rows)?;
    let (flavor, weights) = match a.flavor.as_str() {
        "min-variance" | "min-var" => ("min-variance", min_variance_weights(&cov)?),
        "tangency" => ("tangency", tangency_weights(&cov, &a.mean, a.rf)?),
        "frontier" => {
            let target = a.target.ok_or_else(|| {
                OxisError::invalid_input("optimize: --target is required for --flavor frontier")
            })?;
            ("frontier", efficient_frontier_point(&cov, &a.mean, target)?)
        }
        other => {
            return Err(OxisError::invalid_input(format!(
                "flavor must be 'min-variance', 'tangency', or 'frontier', got {other:?}"
            ))
            .into());
        }
    };
    let report = optimization_report(flavor, weights, &a.mean, &cov)?;
    println!("{}", render(&report, ctx.format));
    Ok(())
}

/// Parse `SYMBOL:QUANTITY:UNIT_COST:PRICE` into a single-lot holding + price.
fn parse_holding(spec: &str) -> Result<(Holding, f64), OxisError> {
    let parts: Vec<&str> = spec.split(':').collect();
    if parts.len() != 4 {
        return Err(OxisError::invalid_input(format!(
            "holding must be SYMBOL:QUANTITY:UNIT_COST:PRICE, got {spec:?}"
        )));
    }
    let qty = parse_f64(parts[1], "quantity")?;
    let cost = parse_f64(parts[2], "unit_cost")?;
    let price = parse_f64(parts[3], "price")?;
    Ok((Holding::single(parts[0], qty, cost), price))
}

/// Parse `YYYY-MM-DD:AMOUNT` into a dated cash flow.
fn parse_cashflow(spec: &str) -> Result<(Date, f64), OxisError> {
    // Split once from the right so the date's own dashes stay intact.
    let (date_str, amount_str) = spec.rsplit_once(':').ok_or_else(|| {
        OxisError::invalid_input(format!("cashflow must be YYYY-MM-DD:AMOUNT, got {spec:?}"))
    })?;
    let date = parse_date(date_str)?;
    Ok((date, parse_f64(amount_str, "amount")?))
}

fn parse_date(s: &str) -> Result<Date, OxisError> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 3 {
        return Err(OxisError::invalid_input(format!(
            "date must be YYYY-MM-DD, got {s:?}"
        )));
    }
    let y = parts[0]
        .parse::<i32>()
        .map_err(|_| OxisError::invalid_input(format!("bad year in {s:?}")))?;
    let m = parts[1]
        .parse::<u8>()
        .map_err(|_| OxisError::invalid_input(format!("bad month in {s:?}")))?;
    let d = parts[2]
        .parse::<u8>()
        .map_err(|_| OxisError::invalid_input(format!("bad day in {s:?}")))?;
    Date::new(y, m, d)
}

/// Parse repeated comma-list row flags into a matrix.
fn parse_rows(rows: &[String]) -> Result<Vec<Vec<f64>>, OxisError> {
    rows.iter()
        .map(|row| {
            row.split(',')
                .map(|v| parse_f64(v.trim(), "matrix entry"))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect()
}

fn parse_f64(s: &str, what: &str) -> Result<f64, OxisError> {
    s.parse::<f64>()
        .map_err(|_| OxisError::invalid_input(format!("bad {what}: {s:?}")))
}
