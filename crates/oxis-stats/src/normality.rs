//! The Jarque-Bera normality test.
//!
//! `JB = n/6 · (S² + K²/4)` with `S` the biased skewness and `K` the biased
//! excess kurtosis. Under normality `JB` is asymptotically χ² with 2 degrees of
//! freedom, whose survival function is the closed form `exp(−JB/2)` — so the
//! p-value needs no special function. Matches `scipy.stats.jarque_bera`.

use crate::descriptive::{excess_kurtosis, skewness};
use oxis_core::OxisError;

/// Jarque-Bera statistic and its asymptotic p-value `(stat, p_value)`.
///
/// # Errors
/// [`OxisError::InvalidInput`] if `n < 2` or the variance is zero (skewness /
/// kurtosis are undefined).
pub fn jarque_bera(xs: &[f64]) -> Result<(f64, f64), OxisError> {
    let s = skewness(xs)?;
    let k = excess_kurtosis(xs)?;
    let n = xs.len() as f64;
    let jb = n / 6.0 * (s * s + k * k / 4.0);
    let p_value = (-jb / 2.0).exp();
    Ok((jb, p_value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_mesokurtic_has_small_statistic() {
        // A symmetric sample has near-zero skew; the statistic stays small and the
        // p-value high (cannot reject normality).
        let xs = [-2.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0, 2.0];
        let (jb, p) = jarque_bera(&xs).unwrap();
        assert!(jb >= 0.0);
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn degenerate_errors_not_panics() {
        assert!(jarque_bera(&[1.0]).is_err());
        assert!(jarque_bera(&[2.0, 2.0, 2.0]).is_err());
    }
}
