//! Portfolio allocation weights from market values.

use oxis_core::OxisError;

/// Allocation weights `wᵢ = mvᵢ / Σmv`.
///
/// Weights may be negative if a market value is negative (a short position).
///
/// # Errors
/// [`OxisError::InvalidInput`] on an empty input. When the total is zero, all
/// weights are `0.0` (not `NaN`).
pub fn weights(market_values: &[f64]) -> Result<Vec<f64>, OxisError> {
    if market_values.is_empty() {
        return Err(OxisError::invalid_input("weights: empty market values"));
    }
    let total: f64 = market_values.iter().sum();
    if total == 0.0 {
        return Ok(vec![0.0; market_values.len()]);
    }
    Ok(market_values.iter().map(|mv| mv / total).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn weights_sum_to_one() {
        let w = weights(&[1750.0, 1600.0, 650.0]).unwrap();
        assert!((w.iter().sum::<f64>() - 1.0).abs() < TOL);
        assert!((w[0] - 1750.0 / 4000.0).abs() < TOL);
    }

    #[test]
    fn zero_total_gives_zero_weights_not_nan() {
        let w = weights(&[100.0, -100.0]).unwrap();
        assert!(w.iter().all(|x| x.is_finite()));
        assert_eq!(w, vec![0.0, 0.0]);
    }

    #[test]
    fn empty_errors() {
        assert!(weights(&[]).is_err());
    }
}
