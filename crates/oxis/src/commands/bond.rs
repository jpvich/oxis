//! `oxis bond` — price a fixed-rate bond and report its analytics.
//!
//! Builds a regular bond from `--face`, `--coupon`, `--frequency`, and
//! `--maturity` (settling on a coupon date), then prices it one of three ways:
//! `--yield` (price + duration + convexity at that yield), `--price` (solve the
//! yield-to-maturity, then duration + convexity), or `--flat-rate` (discount on a
//! flat continuous yield curve). Exactly one mode must be given.

use oxis_bonds::{BondResult, FixedRateBond};
use oxis_core::output::render;
use oxis_core::{OxisError, RunContext};
use oxis_curves::YieldCurve;

/// Flags for `oxis bond`.
///
/// `allow_negative_numbers` lets values like `--yield -0.001` parse (negative
/// yields are real) instead of being mistaken for flags.
#[derive(clap::Args)]
#[command(allow_negative_numbers = true)]
pub struct BondArgs {
    /// Face / notional (redemption) amount.
    #[arg(long, default_value_t = 100.0)]
    face: f64,
    /// Annual coupon rate (e.g. 0.05).
    #[arg(long)]
    coupon: f64,
    /// Coupon payments per year.
    #[arg(long, default_value_t = 2)]
    frequency: u32,
    /// Maturity in years (rounded to a whole number of coupon periods).
    #[arg(long)]
    maturity: f64,
    /// Price from this flat yield (compounded at the coupon frequency).
    #[arg(long = "yield")]
    yield_: Option<f64>,
    /// Solve the yield-to-maturity that matches this clean price.
    #[arg(long)]
    price: Option<f64>,
    /// Price by discounting on a flat continuous yield curve at this rate.
    #[arg(long = "flat-rate")]
    flat_rate: Option<f64>,
}

/// Build the bond, price it per the selected mode, render the result.
pub fn run(args: BondArgs, ctx: &RunContext) -> anyhow::Result<()> {
    if args.maturity <= 0.0 {
        return Err(OxisError::invalid_input("bond: maturity must be positive").into());
    }
    let n_periods = (args.maturity * args.frequency as f64).round() as u32;
    let bond = FixedRateBond::regular(args.face, args.coupon, args.frequency, n_periods)?;

    let result = match (args.yield_, args.price, args.flat_rate) {
        (Some(y), None, None) => analytics_at_yield(&bond, y)?,
        (None, Some(clean), None) => {
            let y = bond.yield_to_maturity(clean)?;
            analytics_at_yield(&bond, y)?
        }
        (None, None, Some(rate)) => {
            let curve = YieldCurve::flat(rate)?;
            let (dirty, clean) = bond.price_from_curve(&curve)?;
            BondResult {
                face: bond.face,
                coupon_rate: bond.coupon_rate,
                frequency: bond.frequency,
                clean_price: clean,
                dirty_price: dirty,
                accrued: bond.accrued,
                bond_yield: None,
                macaulay_duration: None,
                modified_duration: None,
                convexity: None,
            }
        }
        _ => {
            return Err(OxisError::invalid_input(
                "provide exactly one of --yield, --price, or --flat-rate",
            )
            .into());
        }
    };

    println!("{}", render(&result, ctx.format));
    Ok(())
}

/// Full price + duration + convexity at a known yield.
fn analytics_at_yield(bond: &FixedRateBond, y: f64) -> Result<BondResult, OxisError> {
    Ok(BondResult {
        face: bond.face,
        coupon_rate: bond.coupon_rate,
        frequency: bond.frequency,
        clean_price: bond.clean_price_from_yield(y)?,
        dirty_price: bond.dirty_price_from_yield(y)?,
        accrued: bond.accrued,
        bond_yield: Some(y),
        macaulay_duration: Some(bond.macaulay_duration(y)?),
        modified_duration: Some(bond.modified_duration(y)?),
        convexity: Some(bond.convexity(y)?),
    })
}
