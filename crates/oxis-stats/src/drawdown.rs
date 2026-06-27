//! Maximum drawdown of a price / equity series.
//!
//! Drawdown at time `i` is the fractional decline from the running peak:
//! `(peakᵢ − priceᵢ) / peakᵢ`. The maximum drawdown is the largest such decline,
//! reported as a **positive magnitude**. The duration is the number of periods
//! from the peak that preceded the worst trough to that trough.

use oxis_core::OxisError;

/// The worst peak-to-trough decline of a price series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Drawdown {
    /// Maximum drawdown as a positive fraction (e.g. `0.20` = a 20% decline).
    pub max_drawdown: f64,
    /// Index of the running peak preceding the worst trough.
    pub peak_index: usize,
    /// Index of the worst trough.
    pub trough_index: usize,
    /// Periods from peak to trough (`trough_index − peak_index`).
    pub duration: usize,
}

/// Compute the maximum drawdown of a price/equity series.
///
/// # Errors
/// [`OxisError::InvalidInput`] if the series is empty or any price is `≤ 0`.
pub fn max_drawdown(prices: &[f64]) -> Result<Drawdown, OxisError> {
    if prices.is_empty() {
        return Err(OxisError::invalid_input("max_drawdown: empty series"));
    }
    if prices.iter().any(|&p| p <= 0.0) {
        return Err(OxisError::invalid_input(
            "max_drawdown: prices must be positive",
        ));
    }

    let mut peak = prices[0];
    let mut peak_idx = 0usize;
    let mut best = Drawdown {
        max_drawdown: 0.0,
        peak_index: 0,
        trough_index: 0,
        duration: 0,
    };

    for (i, &p) in prices.iter().enumerate() {
        if p > peak {
            peak = p;
            peak_idx = i;
        }
        let dd = (peak - p) / peak;
        if dd > best.max_drawdown {
            best = Drawdown {
                max_drawdown: dd,
                peak_index: peak_idx,
                trough_index: i,
                duration: i - peak_idx,
            };
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn simple_drawdown() {
        // Peak 100 at idx 1, trough 70 at idx 3 → 30% drawdown, duration 2.
        let prices = [90.0, 100.0, 80.0, 70.0, 95.0];
        let d = max_drawdown(&prices).unwrap();
        assert!((d.max_drawdown - 0.30).abs() < TOL);
        assert_eq!(d.peak_index, 1);
        assert_eq!(d.trough_index, 3);
        assert_eq!(d.duration, 2);
    }

    #[test]
    fn monotone_increase_has_no_drawdown() {
        let d = max_drawdown(&[1.0, 2.0, 3.0]).unwrap();
        assert!(d.max_drawdown.abs() < TOL);
    }

    #[test]
    fn invalid_inputs_error_not_panic() {
        assert!(max_drawdown(&[]).is_err());
        assert!(max_drawdown(&[100.0, -1.0]).is_err());
    }
}
