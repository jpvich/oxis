//! Sample statistics for Monte Carlo estimators.
//!
//! Stochastic engines never report a point estimate without its uncertainty, so
//! these helpers are shared across the pricing and process-simulation modules.

/// Sample mean and standard error of the mean over `samples`.
///
/// The standard error is `s / √n` with `s` the Bessel-corrected sample standard
/// deviation. Returns a standard error of `0.0` for fewer than two samples (the
/// mean of a single observation has no estimable spread).
pub fn mean_and_se(samples: &[f64]) -> (f64, f64) {
    let (mean, var) = sample_mean_var(samples);
    let n = samples.len();
    if n < 2 {
        return (mean, 0.0);
    }
    (mean, (var / n as f64).sqrt())
}

/// Sample mean and (Bessel-corrected) variance over `samples`.
///
/// Returns a variance of `0.0` for fewer than two samples. For `n ≥ 2` the
/// variance uses the `n − 1` denominator (unbiased estimator).
pub fn sample_mean_var(samples: &[f64]) -> (f64, f64) {
    let n = samples.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = samples.iter().sum::<f64>() / n as f64;
    if n < 2 {
        return (mean, 0.0);
    }
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    (mean, var)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn mean_and_variance_of_known_sample() {
        let xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (mean, var) = sample_mean_var(&xs);
        assert!(close(mean, 5.0));
        // Bessel-corrected variance of this textbook sample is 32/7.
        assert!(close(var, 32.0 / 7.0));
    }

    #[test]
    fn standard_error_is_sd_over_sqrt_n() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (_, var) = sample_mean_var(&xs);
        let (_, se) = mean_and_se(&xs);
        assert!(close(se, (var / 5.0).sqrt()));
    }

    #[test]
    fn degenerate_samples_have_zero_spread() {
        assert_eq!(mean_and_se(&[]), (0.0, 0.0));
        assert_eq!(mean_and_se(&[3.5]), (3.5, 0.0));
        assert_eq!(sample_mean_var(&[3.5]), (3.5, 0.0));
    }
}
