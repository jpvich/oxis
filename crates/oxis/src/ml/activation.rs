//! The softplus activation and its first two derivatives.
//!
//! Differential ML needs a **twice-differentiable** activation: the twin network
//! predicts the value's input-gradient through `σ'`, and training that prediction
//! backpropagates once more, bringing in `σ''`. Softplus
//! `σ(x) = ln(1 + eˣ)` is smooth, with `σ'(x) = sigmoid(x)` and
//! `σ''(x) = sigmoid(x)·(1 − sigmoid(x))` — both bounded in `(0, 1)` and `(0, ¼]`,
//! so no step ever produces `NaN`/`Inf`. All three are computed in numerically
//! stable forms that hold for large `|x|`.

/// Softplus `σ(x) = ln(1 + eˣ)`, computed as `max(x, 0) + ln(1 + e^{−|x|})` to
/// avoid overflow for large `x`.
pub fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// The logistic sigmoid `1 / (1 + e^{−x})`, which is exactly `σ'(x)`. Branchy form
/// keeps both tails overflow-free.
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// First derivative of softplus, `σ'(x) = sigmoid(x)`.
pub fn softplus_prime(x: f64) -> f64 {
    sigmoid(x)
}

/// Second derivative of softplus, `σ''(x) = sigmoid(x)·(1 − sigmoid(x))`.
pub fn softplus_second(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b} (tol {tol})");
    }

    #[test]
    fn known_values() {
        close(softplus(0.0), 2.0_f64.ln(), 1e-15);
        close(sigmoid(0.0), 0.5, 1e-15);
        close(softplus_second(0.0), 0.25, 1e-15);
    }

    #[test]
    fn stable_in_tails() {
        // No overflow/NaN for large magnitudes; softplus(x) → x as x → +∞.
        assert!(softplus(800.0).is_finite());
        close(softplus(800.0), 800.0, 1e-9);
        assert!(softplus(-800.0).is_finite());
        close(softplus(-800.0), 0.0, 1e-9);
        close(sigmoid(800.0), 1.0, 1e-12);
        close(sigmoid(-800.0), 0.0, 1e-12);
    }

    #[test]
    fn derivatives_match_finite_difference() {
        let h = 1e-6;
        for &x in &[-3.0, -0.7, 0.0, 1.3, 4.0] {
            let fd1 = (softplus(x + h) - softplus(x - h)) / (2.0 * h);
            close(softplus_prime(x), fd1, 1e-7);
            let fd2 = (softplus_prime(x + h) - softplus_prime(x - h)) / (2.0 * h);
            close(softplus_second(x), fd2, 1e-7);
        }
    }
}
