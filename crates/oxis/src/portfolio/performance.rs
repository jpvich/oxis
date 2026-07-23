//! Performance returns: time-weighted (TWR) and money-weighted (MWR / IRR).
//!
//! **TWR** removes the effect of external cash flows by chaining sub-period
//! returns geometrically — it measures the manager's return. **MWR** is the
//! internal rate of return of the dated cash flows — it measures the investor's
//! return, sensitive to flow timing.
//!
//! Conventions: a sub-period flow occurs at the **start** of the sub-period, so
//! `rᵢ = Vᵢ / (Vᵢ₋₁ + flowᵢ) − 1`. For MWR, time is the **Act/365** year fraction
//! from the first cash-flow date, and the sign convention is **money out
//! (invested) negative, money in (received) positive**.

use crate::core::{Date, OxisError, brent};

/// Time-weighted return from period-boundary `values` (`V₀..Vₙ`) and the external
/// net `flows` at the start of each of the `n` sub-periods.
///
/// # Errors
/// [`OxisError::InvalidInput`] if fewer than 2 values, `flows.len() != values.len()
/// − 1`, or any sub-period base `Vᵢ₋₁ + flowᵢ` is zero.
pub fn twr(values: &[f64], flows: &[f64]) -> Result<f64, OxisError> {
    if values.len() < 2 {
        return Err(OxisError::invalid_input("twr: need at least 2 valuations"));
    }
    if flows.len() != values.len() - 1 {
        return Err(OxisError::invalid_input(
            "twr: flows length must be values length − 1",
        ));
    }
    let mut growth = 1.0;
    for i in 0..flows.len() {
        let base = values[i] + flows[i];
        if base == 0.0 {
            return Err(OxisError::invalid_input(
                "twr: zero sub-period base (value + flow)",
            ));
        }
        growth *= values[i + 1] / base;
    }
    Ok(growth - 1.0)
}

/// Net present value of dated `cashflows` at annual rate `r`, with time measured
/// as the Act/365 year fraction from the first date.
fn npv(cashflows: &[(Date, f64)], r: f64) -> f64 {
    let first = cashflows[0].0;
    cashflows
        .iter()
        .map(|&(date, cf)| {
            let t = first.days_until(date) as f64 / 365.0;
            cf / (1.0 + r).powf(t)
        })
        .sum()
}

/// Money-weighted return (IRR) of dated `cashflows`: the annual rate where the
/// NPV is zero. Sign convention: invested amounts negative, received positive.
///
/// # Errors
/// [`OxisError::InvalidInput`] if fewer than 2 cash flows. [`OxisError::Numerical`]
/// if no internal rate brackets a sign change (e.g. all cash flows share a sign).
pub fn mwr(cashflows: &[(Date, f64)]) -> Result<f64, OxisError> {
    if cashflows.len() < 2 {
        return Err(OxisError::invalid_input("mwr: need at least 2 cash flows"));
    }

    // Scan a rate grid for a sign change in NPV, then refine with Brent. The grid
    // runs from just above −100% (where discounting blows up) upward.
    let mut prev_r = -0.9999;
    let mut prev_npv = npv(cashflows, prev_r);
    let mut r = prev_r;
    while r < 10.0 {
        r += 0.01;
        let cur = npv(cashflows, r);
        if prev_npv == 0.0 {
            return Ok(prev_r);
        }
        if prev_npv * cur < 0.0 {
            return brent(prev_r, r, 1e-12, 200, |x| npv(cashflows, x));
        }
        prev_r = r;
        prev_npv = cur;
    }
    Err(OxisError::numerical(
        "mwr: no internal rate of return found (cash flows may not change sign)",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    #[test]
    fn twr_links_subperiods() {
        // 100 → grows to 110, inject 10 → 120 base, ends 150.
        // r1 = 110/100 - 1 = 0.10; r2 = 150/(110+10) - 1 = 0.25. TWR = 1.1·1.25 - 1 = 0.375.
        let twr = twr(&[100.0, 110.0, 150.0], &[0.0, 10.0]).unwrap();
        assert!((twr - 0.375).abs() < TOL);
    }

    #[test]
    fn mwr_one_year_doubling() {
        // -1000 at t0, +1100 one year later → IRR = 10%.
        let cfs = vec![
            (Date::new(2024, 1, 1).unwrap(), -1000.0),
            (Date::new(2025, 1, 1).unwrap(), 1100.0),
        ];
        let irr = mwr(&cfs).unwrap();
        // 366 days in 2024 (leap) → Act/365 t = 366/365, so rate slightly under 0.10.
        assert!((irr - 0.0991).abs() < 1e-3);
    }

    #[test]
    fn invalid_inputs_error_not_panic() {
        assert!(twr(&[100.0], &[]).is_err());
        assert!(twr(&[100.0, 110.0], &[0.0, 0.0]).is_err());
        // All-positive cash flows → no IRR.
        let cfs = vec![
            (Date::new(2024, 1, 1).unwrap(), 100.0),
            (Date::new(2025, 1, 1).unwrap(), 100.0),
        ];
        assert!(mwr(&cfs).is_err());
    }
}
