//! Small dense linear-algebra primitives: solving `A·x = b` and matrix inversion
//! by Gaussian elimination with partial pivoting.
//!
//! Kept dependency-free and deliberately simple — the systems OXIS solves are
//! tiny (a Longstaff-Schwartz normal-equations matrix of size ≤ 4, or a Markowitz
//! covariance matrix of a handful of assets), so a dense `O(n³)` solve is exact,
//! fast, and avoids pulling in a matrix crate. For larger or ill-conditioned
//! systems a dedicated BLAS/LAPACK binding would be the right tool.
//!
//! Conditioning is the caller's responsibility: scale inputs to `O(1)` to keep
//! the system well-conditioned (e.g. moneyness `S/K` rather than raw spot).

use crate::error::OxisError;

/// Solve the square linear system `A·x = b` for `x`.
///
/// `a` is a row-major square matrix (`n` rows each of length `n`); `b` has length
/// `n`. Uses Gaussian elimination with partial pivoting + back-substitution.
///
/// # Errors
/// - [`OxisError::InvalidInput`] if `a` is empty, not square, or `b`'s length does
///   not match.
/// - [`OxisError::Numerical`] if the system is singular (a pivot is effectively
///   zero) or the solve produces a non-finite value.
pub fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, OxisError> {
    let n = a.len();
    if n == 0 {
        return Err(OxisError::invalid_input(
            "solve_linear_system: empty matrix",
        ));
    }
    if a.iter().any(|row| row.len() != n) {
        return Err(OxisError::invalid_input(
            "solve_linear_system: matrix must be square",
        ));
    }
    if b.len() != n {
        return Err(OxisError::invalid_input(
            "solve_linear_system: rhs length must match matrix order",
        ));
    }

    // Assemble the augmented matrix [A | b].
    let mut aug = vec![vec![0.0_f64; n + 1]; n];
    for (i, row) in aug.iter_mut().enumerate() {
        row[..n].copy_from_slice(&a[i]);
        row[n] = b[i];
    }
    gaussian_solve(aug, n)
}

/// Invert the square matrix `a` by solving `A·xⱼ = eⱼ` for each unit column.
///
/// # Errors
/// As [`solve_linear_system`] (empty / non-square / singular).
pub fn invert(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, OxisError> {
    let n = a.len();
    if n == 0 {
        return Err(OxisError::invalid_input("invert: empty matrix"));
    }
    if a.iter().any(|row| row.len() != n) {
        return Err(OxisError::invalid_input("invert: matrix must be square"));
    }
    // Solve for each column of the inverse, then transpose into row-major form.
    let mut cols = Vec::with_capacity(n);
    for j in 0..n {
        let mut e = vec![0.0_f64; n];
        e[j] = 1.0;
        cols.push(solve_linear_system(a, &e)?);
    }
    let mut inv = vec![vec![0.0_f64; n]; n];
    for (j, col) in cols.iter().enumerate() {
        for (i, &v) in col.iter().enumerate() {
            inv[i][j] = v;
        }
    }
    Ok(inv)
}

/// Solve the `n`×`n` system in the augmented matrix `aug` (`n` rows, `n + 1`
/// columns) by Gaussian elimination with partial pivoting.
///
/// Shared by [`solve_linear_system`] and `poly_least_squares` — kept here so the
/// arithmetic order is identical for both (the LSM regression relies on it).
pub(crate) fn gaussian_solve(mut aug: Vec<Vec<f64>>, n: usize) -> Result<Vec<f64>, OxisError> {
    for col in 0..n {
        // Partial pivot: pick the row with the largest magnitude in this column.
        let mut pivot = col;
        let mut best = aug[col][col].abs();
        for (r, row) in aug.iter().enumerate().take(n).skip(col + 1) {
            let v = row[col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1e-300 {
            return Err(OxisError::numerical(
                "linear solve: singular system (degenerate inputs)",
            ));
        }
        aug.swap(col, pivot);

        // Eliminate below the pivot. Clone the pivot row so each lower row can be
        // borrowed mutably without aliasing (the matrix is tiny).
        let pivot_row = aug[col].clone();
        let pivot_diag = pivot_row[col];
        for row in aug.iter_mut().skip(col + 1) {
            let factor = row[col] / pivot_diag;
            for (cell, &pv) in row.iter_mut().zip(pivot_row.iter()).skip(col) {
                *cell -= factor * pv;
            }
        }
    }

    // Back-substitution.
    let mut coeffs = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut acc = aug[i][n];
        for j in (i + 1)..n {
            acc -= aug[i][j] * coeffs[j];
        }
        let c = acc / aug[i][i];
        if !c.is_finite() {
            return Err(OxisError::numerical("linear solve: non-finite solution"));
        }
        coeffs[i] = c;
    }
    Ok(coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b} (tol {tol})");
    }

    #[test]
    fn solves_known_3x3() {
        // 2x + y - z = 8; -3x - y + 2z = -11; -2x + y + 2z = -3  → (2, 3, -1).
        let a = vec![
            vec![2.0, 1.0, -1.0],
            vec![-3.0, -1.0, 2.0],
            vec![-2.0, 1.0, 2.0],
        ];
        let b = [8.0, -11.0, -3.0];
        let x = solve_linear_system(&a, &b).unwrap();
        close(x[0], 2.0, 1e-12);
        close(x[1], 3.0, 1e-12);
        close(x[2], -1.0, 1e-12);
    }

    #[test]
    fn inverts_known_2x2() {
        // [[4,7],[2,6]]⁻¹ = [[0.6,-0.7],[-0.2,0.4]].
        let a = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
        let inv = invert(&a).unwrap();
        close(inv[0][0], 0.6, 1e-12);
        close(inv[0][1], -0.7, 1e-12);
        close(inv[1][0], -0.2, 1e-12);
        close(inv[1][1], 0.4, 1e-12);
    }

    #[test]
    fn singular_and_malformed_error_not_panic() {
        let singular = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(solve_linear_system(&singular, &[1.0, 2.0]).is_err());
        assert!(invert(&singular).is_err());
        assert!(solve_linear_system(&[], &[]).is_err());
        assert!(solve_linear_system(&[vec![1.0, 2.0]], &[1.0]).is_err()); // non-square
    }
}
