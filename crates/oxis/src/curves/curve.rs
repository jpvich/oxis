//! The [`YieldCurve`] term structure: build once from pillars, query many times.
//!
//! A curve answers three questions, all in **continuous compounding** with time
//! measured in years (`Act/365`, matching the rest of OXIS and the QuantLib
//! oracle): the [`discount`](YieldCurve::discount) factor `P(t)`, the
//! [`zero_rate`](YieldCurve::zero_rate) `z(t) = -ln P(t) / t`, and the
//! [`forward_rate`](YieldCurve::forward_rate) `f(t₁,t₂) = (ln P(t₁) − ln P(t₂)) /
//! (t₂ − t₁)`.
//!
//! Three interpolation schemes are supported, each matching a QuantLib term
//! structure exactly:
//! - [`Interpolation::Linear`] — zero rates interpolated linearly in time
//!   (`ZeroCurve`, `Linear`).
//! - [`Interpolation::LogLinear`] — log discount factors interpolated linearly,
//!   i.e. piecewise-constant instantaneous forwards (`DiscountCurve`, default).
//! - [`Interpolation::NaturalCubic`] — zero rates on a natural cubic spline
//!   (`NaturalCubicZeroCurve`).
//!
//! The interpolation scheme is independent of how the curve is built: a curve
//! can be constructed from zero rates or from discount factors, and the inputs
//! are converted to whichever pillar quantity the scheme interpolates.
//!
//! Curves do **not** extrapolate: a query time outside `[t_first, t_last]` (other
//! than `t = 0`, where the discount factor is `1` by definition) is an
//! [`OxisError::InvalidInput`].

use crate::core::{NaturalCubicSpline, OxisError, linear_interpolate};

/// How a curve interpolates between its pillar points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// Linear in zero rates (QuantLib `ZeroCurve` with `Linear`).
    Linear,
    /// Linear in log discount factors (QuantLib `DiscountCurve`, default
    /// `LogLinear`) — equivalently, piecewise-constant instantaneous forwards.
    LogLinear,
    /// Natural cubic spline in zero rates (QuantLib `NaturalCubicZeroCurve`).
    NaturalCubic,
}

impl Interpolation {
    /// The canonical lowercase name, used in rendered output.
    pub fn as_str(self) -> &'static str {
        match self {
            Interpolation::Linear => "linear",
            Interpolation::LogLinear => "log-linear",
            Interpolation::NaturalCubic => "natural-cubic",
        }
    }
}

/// Internal representation: a flat rate, or interpolated pillars.
#[derive(Debug, Clone)]
enum Repr {
    /// A single continuously-compounded rate at all maturities.
    Flat(f64),
    /// Interpolated pillars. `values` holds **zero rates** for `Linear` /
    /// `NaturalCubic`, and **log discount factors** for `LogLinear`. `spline` is
    /// populated only for `NaturalCubic`.
    Interpolated {
        interp: Interpolation,
        times: Vec<f64>,
        values: Vec<f64>,
        spline: Option<NaturalCubicSpline>,
    },
}

/// A yield curve / term structure.
#[derive(Debug, Clone)]
pub struct YieldCurve {
    repr: Repr,
}

/// Validate pillar times: equal length, at least two, strictly increasing, all
/// non-negative and finite. A leading `t = 0` anchor is allowed (interpolated
/// curves treat the first pillar as the reference where the discount factor is
/// `1`); strict monotonicity then guarantees every later pillar is positive.
fn validate_times(times: &[f64], other_len: usize) -> Result<(), OxisError> {
    if times.len() != other_len {
        return Err(OxisError::invalid_input(
            "yield curve: times and values must have equal length",
        ));
    }
    if times.len() < 2 {
        return Err(OxisError::invalid_input(
            "yield curve: need at least two pillars",
        ));
    }
    if times[0] < 0.0 {
        return Err(OxisError::invalid_input(
            "yield curve: pillar times must be non-negative",
        ));
    }
    if times.iter().any(|t| !t.is_finite()) {
        return Err(OxisError::invalid_input(
            "yield curve: pillar times must be finite",
        ));
    }
    if times.windows(2).any(|w| w[1] <= w[0]) {
        return Err(OxisError::invalid_input(
            "yield curve: pillar times must be strictly increasing",
        ));
    }
    Ok(())
}

impl YieldCurve {
    /// A flat curve: a single continuously-compounded rate at every maturity.
    ///
    /// `rate` may be negative (negative interest rates are real). The discount
    /// factor is `e^{-rate·t}`, the zero rate is `rate` at every `t`, and every
    /// forward rate is `rate`.
    pub fn flat(rate: f64) -> Result<Self, OxisError> {
        if !rate.is_finite() {
            return Err(OxisError::invalid_input("yield curve: rate must be finite"));
        }
        Ok(Self {
            repr: Repr::Flat(rate),
        })
    }

    /// Build a curve from continuously-compounded zero rates at pillar times.
    ///
    /// # Errors
    /// [`OxisError::InvalidInput`] if the pillars are malformed (see
    /// [`validate_times`]) or a rate is non-finite.
    pub fn from_zero_rates(
        times: &[f64],
        rates: &[f64],
        interp: Interpolation,
    ) -> Result<Self, OxisError> {
        validate_times(times, rates.len())?;
        if rates.iter().any(|r| !r.is_finite()) {
            return Err(OxisError::invalid_input(
                "yield curve: zero rates must be finite",
            ));
        }
        // Canonical pillar quantity per scheme.
        let values: Vec<f64> = match interp {
            Interpolation::Linear | Interpolation::NaturalCubic => rates.to_vec(),
            // log P(t_i) = -z_i · t_i
            Interpolation::LogLinear => times
                .iter()
                .zip(rates.iter())
                .map(|(&t, &z)| -z * t)
                .collect(),
        };
        Self::from_canonical(interp, times.to_vec(), values)
    }

    /// Build a curve from discount factors at pillar times.
    ///
    /// Discount factors must be finite and strictly positive; values above `1`
    /// are accepted (they arise from negative rates), so monotonicity is not
    /// enforced.
    ///
    /// # Errors
    /// [`OxisError::InvalidInput`] if the pillars are malformed or a discount
    /// factor is non-positive / non-finite.
    pub fn from_discount_factors(
        times: &[f64],
        dfs: &[f64],
        interp: Interpolation,
    ) -> Result<Self, OxisError> {
        validate_times(times, dfs.len())?;
        if times[0] == 0.0 {
            return Err(OxisError::invalid_input(
                "yield curve: discount-factor pillars must have t > 0; use from_zero_rates for a t=0 anchor",
            ));
        }
        if dfs.iter().any(|d| !d.is_finite() || *d <= 0.0) {
            return Err(OxisError::invalid_input(
                "yield curve: discount factors must be finite and positive",
            ));
        }
        let values: Vec<f64> = match interp {
            // z_i = -ln(P_i) / t_i
            Interpolation::Linear | Interpolation::NaturalCubic => times
                .iter()
                .zip(dfs.iter())
                .map(|(&t, &d)| -d.ln() / t)
                .collect(),
            Interpolation::LogLinear => dfs.iter().map(|d| d.ln()).collect(),
        };
        Self::from_canonical(interp, times.to_vec(), values)
    }

    /// Assemble the interpolated representation, building the spline when needed.
    fn from_canonical(
        interp: Interpolation,
        times: Vec<f64>,
        values: Vec<f64>,
    ) -> Result<Self, OxisError> {
        let spline = match interp {
            Interpolation::NaturalCubic => Some(NaturalCubicSpline::new(&times, &values)?),
            _ => None,
        };
        Ok(Self {
            repr: Repr::Interpolated {
                interp,
                times,
                values,
                spline,
            },
        })
    }

    /// A short label for the curve's construction, used in rendered output:
    /// `"flat"` for a flat curve, otherwise the interpolation name.
    pub fn interpolation_label(&self) -> &'static str {
        match &self.repr {
            Repr::Flat(_) => "flat",
            Repr::Interpolated { interp, .. } => interp.as_str(),
        }
    }

    /// The discount factor `P(t)` for time `t` (years). `P(0) = 1` exactly.
    ///
    /// # Errors
    /// [`OxisError::InvalidInput`] if `t < 0`, or if `t` falls outside the pillar
    /// range (no extrapolation).
    pub fn discount(&self, t: f64) -> Result<f64, OxisError> {
        if t < 0.0 {
            return Err(OxisError::invalid_input("yield curve: t must be >= 0"));
        }
        if t == 0.0 {
            return Ok(1.0);
        }
        match &self.repr {
            Repr::Flat(r) => Ok((-r * t).exp()),
            Repr::Interpolated {
                interp,
                times,
                values,
                spline,
            } => match interp {
                Interpolation::Linear => {
                    let z = linear_interpolate(times, values, t)?;
                    Ok((-z * t).exp())
                }
                Interpolation::NaturalCubic => {
                    let z = spline
                        .as_ref()
                        .expect("natural-cubic curve has a spline")
                        .eval(t)?;
                    Ok((-z * t).exp())
                }
                Interpolation::LogLinear => {
                    let log_df = linear_interpolate(times, values, t)?;
                    Ok(log_df.exp())
                }
            },
        }
    }

    /// The continuously-compounded zero rate `z(t) = -ln P(t) / t`.
    ///
    /// At `t = 0` the rate is defined as its limit: the flat rate for a flat
    /// curve, or the average rate to the first pillar otherwise.
    ///
    /// # Errors
    /// [`OxisError::InvalidInput`] if `t < 0` or `t` is out of range.
    pub fn zero_rate(&self, t: f64) -> Result<f64, OxisError> {
        if t < 0.0 {
            return Err(OxisError::invalid_input("yield curve: t must be >= 0"));
        }
        if t == 0.0 {
            return match &self.repr {
                Repr::Flat(r) => Ok(*r),
                // Defined as the limit: use the first strictly-positive pillar
                // (always present given strictly-increasing, non-negative times).
                Repr::Interpolated { times, .. } => {
                    let tp = times
                        .iter()
                        .copied()
                        .find(|&x| x > 0.0)
                        .expect("a curve has a positive pillar");
                    self.zero_rate(tp)
                }
            };
        }
        let df = self.discount(t)?;
        Ok(-df.ln() / t)
    }

    /// The continuously-compounded forward rate over `[t1, t2]`.
    ///
    /// `f = (ln P(t1) − ln P(t2)) / (t2 − t1)`. With `t1 = 0` this reduces to the
    /// zero rate to `t2`.
    ///
    /// # Errors
    /// [`OxisError::InvalidInput`] if `t1 < 0`, `t2 <= t1`, or either endpoint is
    /// out of range.
    pub fn forward_rate(&self, t1: f64, t2: f64) -> Result<f64, OxisError> {
        if t1 < 0.0 {
            return Err(OxisError::invalid_input("yield curve: t1 must be >= 0"));
        }
        if t2 <= t1 {
            return Err(OxisError::invalid_input("yield curve: require t2 > t1"));
        }
        let df1 = self.discount(t1)?;
        let df2 = self.discount(t2)?;
        Ok((df1.ln() - df2.ln()) / (t2 - t1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b} (tol {tol})");
    }

    #[test]
    fn flat_curve_is_exact() {
        let c = YieldCurve::flat(0.03).unwrap();
        close(c.discount(0.0).unwrap(), 1.0, 1e-15);
        close(c.discount(2.0).unwrap(), (-0.06_f64).exp(), 1e-15);
        close(c.zero_rate(0.0).unwrap(), 0.03, 1e-15);
        close(c.zero_rate(5.0).unwrap(), 0.03, 1e-12);
        close(c.forward_rate(1.0, 3.0).unwrap(), 0.03, 1e-12);
    }

    #[test]
    fn flat_curve_allows_negative_rate() {
        let c = YieldCurve::flat(-0.01).unwrap();
        assert!(c.discount(2.0).unwrap() > 1.0); // negative rate -> df above 1
        close(c.zero_rate(3.0).unwrap(), -0.01, 1e-12);
    }

    #[test]
    fn zero_rates_round_trip_at_pillars() {
        let times = [0.5, 1.0, 2.0, 5.0];
        let rates = [0.02, 0.025, 0.03, 0.035];
        for interp in [
            Interpolation::Linear,
            Interpolation::LogLinear,
            Interpolation::NaturalCubic,
        ] {
            let c = YieldCurve::from_zero_rates(&times, &rates, interp).unwrap();
            for (&t, &z) in times.iter().zip(rates.iter()) {
                close(c.zero_rate(t).unwrap(), z, 1e-10);
            }
        }
    }

    #[test]
    fn discount_factors_round_trip_at_pillars() {
        let times = [1.0, 2.0, 3.0];
        let dfs = [0.97, 0.94, 0.90];
        for interp in [
            Interpolation::Linear,
            Interpolation::LogLinear,
            Interpolation::NaturalCubic,
        ] {
            let c = YieldCurve::from_discount_factors(&times, &dfs, interp).unwrap();
            for (&t, &d) in times.iter().zip(dfs.iter()) {
                close(c.discount(t).unwrap(), d, 1e-10);
            }
        }
    }

    #[test]
    fn discount_is_monotone_decreasing_for_positive_rates() {
        let times = [1.0, 2.0, 5.0];
        let rates = [0.02, 0.03, 0.04];
        let c = YieldCurve::from_zero_rates(&times, &rates, Interpolation::Linear).unwrap();
        let mut prev = 1.0;
        for t in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0] {
            let df = c.discount(t).unwrap();
            assert!(df < prev, "df not decreasing at t={t}: {df} >= {prev}");
            prev = df;
        }
    }

    #[test]
    fn log_linear_has_piecewise_constant_forwards() {
        // Linear in log P => the forward over any sub-interval inside one pillar
        // segment is constant and equals the segment's average forward.
        let times = [1.0, 2.0, 3.0];
        let dfs = [0.97, 0.94, 0.90];
        let c = YieldCurve::from_discount_factors(&times, &dfs, Interpolation::LogLinear).unwrap();
        let f_a = c.forward_rate(1.2, 1.4).unwrap();
        let f_b = c.forward_rate(1.6, 1.9).unwrap();
        close(f_a, f_b, 1e-12);
        // And it equals the whole-segment forward.
        let f_seg = c.forward_rate(1.0, 2.0).unwrap();
        close(f_a, f_seg, 1e-12);
    }

    #[test]
    fn forward_is_consistent_with_discounts() {
        let times = [0.5, 1.0, 2.0];
        let rates = [0.02, 0.025, 0.03];
        let c = YieldCurve::from_zero_rates(&times, &rates, Interpolation::NaturalCubic).unwrap();
        let (t1, t2) = (0.75, 1.5);
        let expected = (c.discount(t1).unwrap().ln() - c.discount(t2).unwrap().ln()) / (t2 - t1);
        close(c.forward_rate(t1, t2).unwrap(), expected, 1e-12);
    }

    #[test]
    fn zero_anchor_pillar_is_accepted() {
        // A leading t=0 anchor (as interpolated curves use) is valid.
        let times = [0.0, 1.0, 2.0, 5.0];
        let rates = [0.02, 0.022, 0.025, 0.03];
        for interp in [
            Interpolation::Linear,
            Interpolation::LogLinear,
            Interpolation::NaturalCubic,
        ] {
            let c = YieldCurve::from_zero_rates(&times, &rates, interp).unwrap();
            close(c.discount(0.0).unwrap(), 1.0, 1e-15);
            // Interior query lands between the anchor and later pillars.
            assert!(c.discount(0.5).unwrap() < 1.0);
            assert!(c.zero_rate(0.0).unwrap().is_finite());
        }
        // discount-factor pillars still reject a t=0 anchor (df/0 is undefined).
        assert!(
            YieldCurve::from_discount_factors(&[0.0, 1.0], &[1.0, 0.97], Interpolation::LogLinear)
                .is_err()
        );
    }

    #[test]
    fn rejects_out_of_range_and_bad_inputs() {
        let times = [1.0, 2.0, 3.0];
        let rates = [0.02, 0.025, 0.03];
        let c = YieldCurve::from_zero_rates(&times, &rates, Interpolation::Linear).unwrap();
        assert!(c.discount(0.5).is_err()); // below first pillar
        assert!(c.discount(3.5).is_err()); // above last pillar
        assert!(c.discount(-1.0).is_err());
        assert!(c.forward_rate(2.0, 1.0).is_err()); // t2 <= t1

        // Construction errors.
        assert!(YieldCurve::from_zero_rates(&[1.0], &[0.02], Interpolation::Linear).is_err());
        assert!(
            YieldCurve::from_zero_rates(&[2.0, 1.0], &[0.02, 0.03], Interpolation::Linear).is_err()
        );
        assert!(
            YieldCurve::from_discount_factors(&[1.0, 2.0], &[0.97, 0.0], Interpolation::LogLinear)
                .is_err()
        );
        assert!(YieldCurve::flat(f64::NAN).is_err());
    }
}
