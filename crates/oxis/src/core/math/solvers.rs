//! One-dimensional root finders shared by the pricing modules.
//!
//! Two complementary methods:
//!
//! - [`newton`] — quadratic convergence when a derivative is available and the
//!   start point is reasonable (used for implied volatility via vega).
//! - [`brent`] — bracketing method combining bisection, secant, and inverse
//!   quadratic interpolation; robust fallback that needs only a sign-change
//!   bracket, no derivative.
//!
//! Both return [`OxisError::Numerical`] rather than looping forever or
//! returning `NaN` when they fail to converge.

use crate::core::error::OxisError;

/// Find a root of `f` near `x0` with Newton-Raphson.
///
/// `f_and_df(x)` returns the pair `(f(x), f'(x))`. Iterates until `|f(x)| <=
/// tol` (success) or `max_iter` is exhausted / the derivative collapses
/// (failure). Returning both values together avoids evaluating the function
/// twice per step.
///
/// # Errors
/// [`OxisError::Numerical`] if the derivative underflows or the iteration does
/// not converge within `max_iter` steps.
pub fn newton<F>(x0: f64, tol: f64, max_iter: usize, mut f_and_df: F) -> Result<f64, OxisError>
where
    F: FnMut(f64) -> (f64, f64),
{
    let mut x = x0;
    for _ in 0..max_iter {
        let (fx, dfx) = f_and_df(x);
        if fx.abs() <= tol {
            return Ok(x);
        }
        if dfx.abs() < f64::EPSILON {
            return Err(OxisError::numerical(
                "newton: derivative too small to continue",
            ));
        }
        x -= fx / dfx;
        if !x.is_finite() {
            return Err(OxisError::numerical("newton: iterate left the real line"));
        }
    }
    Err(OxisError::numerical("newton: did not converge"))
}

/// Find a root of `f` in the bracket `[a, b]` with Brent's method.
///
/// Requires `f(a)` and `f(b)` to have opposite signs (a sign-change bracket).
/// Converges for any continuous `f` on the bracket without needing a
/// derivative.
///
/// # Errors
/// [`OxisError::Numerical`] if `[a, b]` is not a valid sign-change bracket or
/// the iteration does not converge within `max_iter` steps.
pub fn brent<F>(a: f64, b: f64, tol: f64, max_iter: usize, mut f: F) -> Result<f64, OxisError>
where
    F: FnMut(f64) -> f64,
{
    let (mut a, mut b) = (a, b);
    let mut fa = f(a);
    let mut fb = f(b);

    if fa.abs() <= tol {
        return Ok(a);
    }
    if fb.abs() <= tol {
        return Ok(b);
    }
    if fa * fb > 0.0 {
        return Err(OxisError::numerical(
            "brent: endpoints do not bracket a root",
        ));
    }

    // Ensure |f(b)| <= |f(a)| so b is the better estimate.
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut mflag = true;
    let mut d = a; // only read after the first iteration sets it

    for _ in 0..max_iter {
        let s = if (fa - fc).abs() > f64::EPSILON && (fb - fc).abs() > f64::EPSILON {
            // Inverse quadratic interpolation.
            a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
        } else {
            // Secant.
            b - fb * (b - a) / (fb - fa)
        };

        // Conditions under which we reject the interpolated step and bisect.
        let lo = (3.0 * a + b) / 4.0;
        let between = (s - b) * (s - lo) < 0.0; // s strictly between (3a+b)/4 and b
        let take_bisection = !between
            || (mflag && (s - b).abs() >= (b - c).abs() / 2.0)
            || (!mflag && (s - b).abs() >= (c - d).abs() / 2.0)
            || (mflag && (b - c).abs() < tol)
            || (!mflag && (c - d).abs() < tol);

        let s = if take_bisection {
            mflag = true;
            (a + b) / 2.0
        } else {
            mflag = false;
            s
        };

        let fs = f(s);
        d = c;
        c = b;
        fc = fb;

        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }

        if fb.abs() <= tol || (b - a).abs() <= tol {
            return Ok(b);
        }
    }
    Err(OxisError::numerical("brent: did not converge"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b} (tol {tol})");
    }

    #[test]
    fn newton_finds_sqrt_two() {
        // root of x^2 - 2.
        let r = newton(1.0, 1e-12, 50, |x| (x * x - 2.0, 2.0 * x)).unwrap();
        close(r, 2.0_f64.sqrt(), 1e-10);
    }

    #[test]
    fn newton_solves_cos() {
        // root of cos(x) near 1.0 -> pi/2.
        let r = newton(1.0, 1e-12, 50, |x| (x.cos(), -x.sin())).unwrap();
        close(r, std::f64::consts::FRAC_PI_2, 1e-9);
    }

    #[test]
    fn newton_reports_flat_derivative() {
        // f(x) = 1 has no root and zero derivative.
        let err = newton(0.0, 1e-12, 50, |_| (1.0, 0.0));
        assert!(err.is_err());
    }

    #[test]
    fn brent_finds_sqrt_two() {
        let r = brent(0.0, 2.0, 1e-12, 100, |x| x * x - 2.0).unwrap();
        close(r, 2.0_f64.sqrt(), 1e-10);
    }

    #[test]
    fn brent_solves_transcendental() {
        // x = cos(x) near 0.739085.
        let r = brent(0.0, 1.0, 1e-12, 100, |x| x - x.cos()).unwrap();
        close(r, 0.739_085_133_215_16, 1e-9);
    }

    #[test]
    fn brent_rejects_bad_bracket() {
        // Both endpoints positive: no sign change.
        let err = brent(1.0, 2.0, 1e-12, 100, |x| x * x + 1.0);
        assert!(err.is_err());
    }
}
