//! Small polynomial least-squares fit, shared by the simulation modules.
//!
//! The Longstaff-Schwartz American Monte Carlo regresses continuation value on a
//! low-degree polynomial basis of the underlying. That fit is the only linear
//! algebra the pricing core needs, so rather than pull in a matrix crate we solve
//! the (`degree+1`)×(`degree+1`) normal equations directly with Gaussian
//! elimination + partial pivoting. For the degrees LSM uses (≤ 3) this is exact,
//! fast, and dependency-free.
//!
//! Conditioning is the caller's responsibility: the normal-equations matrix has
//! entries `Σ xᵏ` up to `k = 2·degree`, so callers should pass inputs scaled to
//! `O(1)` (e.g. moneyness `S/K` rather than raw spot) to keep the system
//! well-conditioned.

use crate::core::error::OxisError;
use crate::core::math::linalg::gaussian_solve;

/// Fit `ys ≈ Σ cᵢ·xⁱ` (monomial basis, `i = 0..=degree`) by least squares.
///
/// Returns the `degree + 1` coefficients `[c₀, c₁, …, c_degree]`, lowest order
/// first (`c₀` is the constant term).
///
/// # Errors
/// - [`OxisError::InvalidInput`] if `xs` and `ys` differ in length, if `xs` is
///   empty, or if there are fewer points than coefficients (`xs.len() <=
///   degree`, an underdetermined fit).
/// - [`OxisError::Numerical`] if the normal-equations matrix is singular (e.g.
///   all `xs` identical) or the solve produces a non-finite coefficient.
pub fn poly_least_squares(xs: &[f64], ys: &[f64], degree: usize) -> Result<Vec<f64>, OxisError> {
    if xs.len() != ys.len() {
        return Err(OxisError::invalid_input(
            "poly_least_squares: xs and ys must have equal length",
        ));
    }
    if xs.is_empty() {
        return Err(OxisError::invalid_input(
            "poly_least_squares: need at least one point",
        ));
    }
    if xs.len() <= degree {
        return Err(OxisError::invalid_input(
            "poly_least_squares: fewer points than coefficients (underdetermined)",
        ));
    }

    let n = degree + 1;

    // Power sums S_k = Σ xᵏ for k = 0..=2·degree. The symmetric normal matrix is
    // M[i][j] = S_{i+j}; we accumulate the distinct sums once.
    let mut power_sums = vec![0.0_f64; 2 * degree + 1];
    // Right-hand side b[i] = Σ xⁱ·y.
    let mut rhs = vec![0.0_f64; n];
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let mut xp = 1.0;
        for ps in power_sums.iter_mut() {
            *ps += xp;
            xp *= x;
        }
        let mut xq = 1.0;
        for b in rhs.iter_mut() {
            *b += xq * y;
            xq *= x;
        }
    }

    // Assemble the dense augmented matrix [M | b].
    let mut aug = vec![vec![0.0_f64; n + 1]; n];
    for (i, row) in aug.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().take(n).enumerate() {
            *cell = power_sums[i + j];
        }
        row[n] = rhs[i];
    }

    gaussian_solve(aug, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b} (tol {tol})");
    }

    #[test]
    fn recovers_exact_line() {
        // y = 3 - 2x sampled exactly -> coefficients recovered to machine eps.
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = xs.map(|x| 3.0 - 2.0 * x);
        let c = poly_least_squares(&xs, &ys, 1).unwrap();
        close(c[0], 3.0, 1e-10);
        close(c[1], -2.0, 1e-10);
    }

    #[test]
    fn recovers_exact_quadratic() {
        // y = 1 + 0.5x - 0.25x^2.
        let xs = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let ys = xs.map(|x| 1.0 + 0.5 * x - 0.25 * x * x);
        let c = poly_least_squares(&xs, &ys, 2).unwrap();
        close(c[0], 1.0, 1e-9);
        close(c[1], 0.5, 1e-9);
        close(c[2], -0.25, 1e-9);
    }

    #[test]
    fn least_squares_fit_minimizes_residual() {
        // Noisy-ish points around y = x; the degree-1 fit should pass near them.
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = [0.1, 0.9, 2.1, 2.9];
        let c = poly_least_squares(&xs, &ys, 1).unwrap();
        close(c[0], 0.0, 0.2);
        close(c[1], 1.0, 0.1);
    }

    #[test]
    fn rejects_mismatched_lengths() {
        assert!(poly_least_squares(&[1.0, 2.0], &[1.0], 1).is_err());
    }

    #[test]
    fn rejects_underdetermined() {
        // Two points cannot determine a quadratic (3 coefficients).
        assert!(poly_least_squares(&[1.0, 2.0], &[1.0, 2.0], 2).is_err());
    }

    #[test]
    fn rejects_singular_system() {
        // All x identical -> the Vandermonde columns collapse, matrix singular.
        let xs = [2.0, 2.0, 2.0, 2.0];
        let ys = [1.0, 2.0, 3.0, 4.0];
        assert!(poly_least_squares(&xs, &ys, 2).is_err());
    }
}
