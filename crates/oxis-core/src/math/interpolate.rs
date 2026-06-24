//! One-dimensional interpolation primitives, shared by the curve modules.
//!
//! Yield-curve construction needs to interpolate between pillar points. Two
//! schemes cover the term-structure work: piecewise-[`linear_interpolate`] and a
//! [`NaturalCubicSpline`] (natural boundary, second derivative zero at the ends).
//! Both are pure `f64` and dependency-free; the cubic spline solves its
//! tridiagonal system directly (Thomas algorithm), so no matrix crate is pulled
//! into the core.
//!
//! Neither routine extrapolates: querying outside `[xs[0], xs[last]]` is an
//! [`OxisError::InvalidInput`], matching QuantLib term structures without
//! `enableExtrapolation()`.

use crate::error::OxisError;

/// Validate a set of interpolation nodes: equal-length, at least two points, and
/// strictly increasing in `xs`. Returns the node count on success.
fn validate_nodes(xs: &[f64], ys: &[f64]) -> Result<usize, OxisError> {
    if xs.len() != ys.len() {
        return Err(OxisError::invalid_input(
            "interpolate: xs and ys must have equal length",
        ));
    }
    if xs.len() < 2 {
        return Err(OxisError::invalid_input(
            "interpolate: need at least two nodes",
        ));
    }
    if xs.windows(2).any(|w| w[1] <= w[0]) {
        return Err(OxisError::invalid_input(
            "interpolate: xs must be strictly increasing",
        ));
    }
    Ok(xs.len())
}

/// Index of the segment `[xs[i], xs[i+1]]` containing `x`, after a range check.
///
/// `xs` is assumed already validated (strictly increasing, `len >= 2`). The
/// returned index is clamped to `0..=len-2` so the right endpoint maps to the
/// last segment.
fn segment_index(xs: &[f64], x: f64) -> Result<usize, OxisError> {
    let last = xs.len() - 1;
    if x < xs[0] || x > xs[last] {
        return Err(OxisError::invalid_input(
            "interpolate: x is outside the node range (no extrapolation)",
        ));
    }
    // Largest i with xs[i] <= x, clamped to the final segment for x == xs[last].
    let i = xs.partition_point(|&v| v <= x).saturating_sub(1);
    Ok(i.min(last - 1))
}

/// Piecewise-linear interpolation of `ys` over `xs`, evaluated at `x`.
///
/// # Errors
/// - [`OxisError::InvalidInput`] if `xs`/`ys` differ in length, have fewer than
///   two nodes, or `xs` is not strictly increasing.
/// - [`OxisError::InvalidInput`] if `x` lies outside `[xs[0], xs[last]]`.
pub fn linear_interpolate(xs: &[f64], ys: &[f64], x: f64) -> Result<f64, OxisError> {
    validate_nodes(xs, ys)?;
    let i = segment_index(xs, x)?;
    let t = (x - xs[i]) / (xs[i + 1] - xs[i]);
    Ok(ys[i] + t * (ys[i + 1] - ys[i]))
}

/// A natural cubic spline through a set of nodes.
///
/// "Natural" means the second derivative is zero at both endpoints. Construction
/// solves for the knot second derivatives once; [`eval`](Self::eval) then
/// evaluates the relevant cubic piece in O(log n).
#[derive(Debug, Clone)]
pub struct NaturalCubicSpline {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Second derivatives at each knot (`m[0] = m[n-1] = 0`).
    m: Vec<f64>,
}

impl NaturalCubicSpline {
    /// Build a natural cubic spline through `(xs, ys)`.
    ///
    /// # Errors
    /// [`OxisError::InvalidInput`] if `xs`/`ys` differ in length, have fewer than
    /// two nodes, or `xs` is not strictly increasing.
    pub fn new(xs: &[f64], ys: &[f64]) -> Result<Self, OxisError> {
        let n = validate_nodes(xs, ys)?;
        let mut m = vec![0.0_f64; n];

        // With only two nodes (or fewer interior unknowns) the natural spline is
        // just the straight line: both second derivatives are zero.
        if n > 2 {
            let h: Vec<f64> = xs.windows(2).map(|w| w[1] - w[0]).collect();
            // Interior system for m[1..=n-2]; m[0] = m[n-1] = 0 (natural).
            // Tridiagonal rows i = 1..=n-2:
            //   h[i-1]·m[i-1] + 2(h[i-1]+h[i])·m[i] + h[i]·m[i+1] = rhs[i].
            let interior = n - 2;
            let mut sub = vec![0.0_f64; interior]; // sub-diagonal
            let mut diag = vec![0.0_f64; interior];
            let mut sup = vec![0.0_f64; interior]; // super-diagonal
            let mut rhs = vec![0.0_f64; interior];
            for k in 0..interior {
                let i = k + 1; // knot index
                sub[k] = h[i - 1];
                diag[k] = 2.0 * (h[i - 1] + h[i]);
                sup[k] = h[i];
                let slope_hi = (ys[i + 1] - ys[i]) / h[i];
                let slope_lo = (ys[i] - ys[i - 1]) / h[i - 1];
                rhs[k] = 6.0 * (slope_hi - slope_lo);
            }
            let solved = thomas_solve(&sub, &diag, &sup, &rhs)?;
            m[1..=interior].copy_from_slice(&solved);
        }

        Ok(Self {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
            m,
        })
    }

    /// Evaluate the spline at `x`.
    ///
    /// # Errors
    /// [`OxisError::InvalidInput`] if `x` lies outside `[xs[0], xs[last]]`
    /// (the spline does not extrapolate).
    pub fn eval(&self, x: f64) -> Result<f64, OxisError> {
        let i = segment_index(&self.xs, x)?;
        let h = self.xs[i + 1] - self.xs[i];
        let a = self.xs[i + 1] - x;
        let b = x - self.xs[i];
        // Standard second-derivative form of the cubic on [x_i, x_{i+1}].
        let term = (self.m[i] * a.powi(3) + self.m[i + 1] * b.powi(3)) / (6.0 * h);
        let lin = (self.ys[i] / h - self.m[i] * h / 6.0) * a
            + (self.ys[i + 1] / h - self.m[i + 1] * h / 6.0) * b;
        Ok(term + lin)
    }
}

/// Solve a tridiagonal system (Thomas algorithm). `sub`, `diag`, `sup`, `rhs` all
/// have length `m`; `sub[0]` and `sup[m-1]` are unused (the corners).
fn thomas_solve(
    sub: &[f64],
    diag: &[f64],
    sup: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>, OxisError> {
    let m = diag.len();
    let mut c = vec![0.0_f64; m];
    let mut d = vec![0.0_f64; m];
    if diag[0].abs() < 1e-300 {
        return Err(OxisError::numerical("interpolate: singular spline system"));
    }
    c[0] = sup[0] / diag[0];
    d[0] = rhs[0] / diag[0];
    for i in 1..m {
        let denom = diag[i] - sub[i] * c[i - 1];
        if denom.abs() < 1e-300 {
            return Err(OxisError::numerical("interpolate: singular spline system"));
        }
        c[i] = sup[i] / denom;
        d[i] = (rhs[i] - sub[i] * d[i - 1]) / denom;
    }
    let mut x = vec![0.0_f64; m];
    x[m - 1] = d[m - 1];
    for i in (0..m - 1).rev() {
        x[i] = d[i] - c[i] * x[i + 1];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b} (tol {tol})");
    }

    #[test]
    fn linear_recovers_nodes_and_midpoints() {
        let xs = [0.0, 1.0, 3.0];
        let ys = [0.0, 2.0, 6.0]; // slope 2 throughout
        close(linear_interpolate(&xs, &ys, 0.0).unwrap(), 0.0, 1e-15);
        close(linear_interpolate(&xs, &ys, 3.0).unwrap(), 6.0, 1e-15);
        close(linear_interpolate(&xs, &ys, 0.5).unwrap(), 1.0, 1e-15);
        close(linear_interpolate(&xs, &ys, 2.0).unwrap(), 4.0, 1e-15);
    }

    #[test]
    fn spline_recovers_nodes() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = [1.0, 0.5, 2.0, 1.5, 3.0];
        let s = NaturalCubicSpline::new(&xs, &ys).unwrap();
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            close(s.eval(x).unwrap(), y, 1e-12);
        }
    }

    #[test]
    fn spline_is_exact_for_natural_cubic_data() {
        // A cubic with zero second derivative at the endpoints is reproduced
        // exactly by a natural spline. y = x^3 - 6x^2 + ... has y''=0 at x=2;
        // use the symmetric y = (x-2)^3 on [0,4] (y''=6(x-2): not zero at ends),
        // so instead test a line, which any natural spline reproduces exactly.
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = xs.map(|x| 2.0 - 0.5 * x);
        let s = NaturalCubicSpline::new(&xs, &ys).unwrap();
        close(s.eval(1.5).unwrap(), 2.0 - 0.5 * 1.5, 1e-12);
        close(s.eval(2.7).unwrap(), 2.0 - 0.5 * 2.7, 1e-12);
    }

    #[test]
    fn two_node_spline_is_linear() {
        let xs = [1.0, 4.0];
        let ys = [2.0, 8.0];
        let s = NaturalCubicSpline::new(&xs, &ys).unwrap();
        close(s.eval(2.5).unwrap(), 5.0, 1e-12);
    }

    #[test]
    fn rejects_out_of_range() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [0.0, 1.0, 4.0];
        assert!(linear_interpolate(&xs, &ys, -0.1).is_err());
        assert!(linear_interpolate(&xs, &ys, 2.1).is_err());
        let s = NaturalCubicSpline::new(&xs, &ys).unwrap();
        assert!(s.eval(-0.1).is_err());
        assert!(s.eval(2.1).is_err());
    }

    #[test]
    fn rejects_bad_nodes() {
        assert!(linear_interpolate(&[0.0, 1.0], &[1.0], 0.5).is_err()); // length
        assert!(linear_interpolate(&[0.0], &[1.0], 0.0).is_err()); // too few
        assert!(linear_interpolate(&[1.0, 1.0], &[1.0, 2.0], 1.0).is_err()); // not increasing
        assert!(NaturalCubicSpline::new(&[2.0, 1.0], &[1.0, 2.0]).is_err()); // decreasing
    }
}
