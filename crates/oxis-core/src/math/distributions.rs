//! The standard Normal distribution — high-accuracy CDF and PDF.
//!
//! Pricing accuracy depends directly on the cumulative Normal, so this is a
//! genuine high-accuracy implementation (~1e-15 over the whole range), not a
//! crude approximation. The CDF uses Graeme West's refinement of the Hart (1968)
//! rational approximation ("Better approximations to cumulative normal
//! functions", *Wilmott*, 2009), which is accurate to double precision.

use core::f64::consts::PI;

/// Probability density function of the standard Normal: `φ(x) = e^(-x²/2)/√(2π)`.
pub fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Cumulative distribution function of the standard Normal: `Φ(x) = P(Z ≤ x)`.
///
/// Accurate to ~1e-15. Saturates cleanly to 0/1 in the far tails rather than
/// under/overflowing.
pub fn normal_cdf(x: f64) -> f64 {
    let abs_x = x.abs();

    // Far tail: Φ is 0 (or 1) to within f64 precision.
    if abs_x > 37.0 {
        return if x > 0.0 { 1.0 } else { 0.0 };
    }

    let exponential = (-0.5 * abs_x * abs_x).exp();
    let tail = if abs_x < 7.071_067_811_865_475 {
        // Inner region: ratio of two polynomials in |x|.
        let mut num = 3.526_249_659_989_109e-2 * abs_x + 0.700_383_064_443_688;
        num = num * abs_x + 6.373_962_203_531_65;
        num = num * abs_x + 33.912_866_078_383;
        num = num * abs_x + 112.079_291_497_871;
        num = num * abs_x + 221.213_596_169_931;
        num = num * abs_x + 220.206_867_912_376;

        let mut den = 8.838_834_764_831_844e-2 * abs_x + 1.755_667_163_182_64;
        den = den * abs_x + 16.064_177_579_207;
        den = den * abs_x + 86.780_732_202_946_1;
        den = den * abs_x + 296.564_248_779_674;
        den = den * abs_x + 637.333_633_378_831;
        den = den * abs_x + 793.826_512_519_948;
        den = den * abs_x + 440.413_735_824_752;

        exponential * num / den
    } else {
        // Outer region: continued-fraction form.
        let mut cf = abs_x + 0.65;
        cf = abs_x + 4.0 / cf;
        cf = abs_x + 3.0 / cf;
        cf = abs_x + 2.0 / cf;
        cf = abs_x + 1.0 / cf;
        exponential / cf / 2.506_628_274_631_000_5
    };

    // `tail` is the upper-tail probability for |x|; fold back by sign.
    if x > 0.0 { 1.0 - tail } else { tail }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b} (tol {tol})");
    }

    #[test]
    fn known_cdf_values() {
        assert_close(normal_cdf(0.0), 0.5, 1e-15);
        // Φ(1.96) ≈ 0.9750021048517795
        assert_close(normal_cdf(1.96), 0.975_002_104_851_780, 1e-12);
        assert_close(normal_cdf(-1.96), 0.024_997_895_148_220, 1e-12);
        // Φ(1) ≈ 0.8413447460685429
        assert_close(normal_cdf(1.0), 0.841_344_746_068_543, 1e-12);
    }

    #[test]
    fn symmetry_holds() {
        for &x in &[0.1, 0.5, 1.0, 2.5, 5.0, 8.0, 20.0] {
            assert_close(normal_cdf(-x), 1.0 - normal_cdf(x), 1e-13);
        }
    }

    #[test]
    fn tails_saturate_without_overflow() {
        assert_eq!(normal_cdf(-40.0), 0.0);
        assert_eq!(normal_cdf(40.0), 1.0);
        assert!(normal_cdf(-10.0) > 0.0 && normal_cdf(-10.0) < 1e-20);
    }

    #[test]
    fn pdf_known_values() {
        // φ(0) = 1/√(2π) ≈ 0.3989422804014327
        assert_close(normal_pdf(0.0), 0.398_942_280_401_433, 1e-15);
        // φ is symmetric.
        assert_close(normal_pdf(-1.5), normal_pdf(1.5), 1e-15);
    }
}
