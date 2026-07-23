//! Cox-Ross-Rubinstein (CRR) binomial tree for European and American options.
//!
//! Method (with continuous dividend yield `q`, `n` steps):
//! ```text
//! dt = T/n;  u = e^(σ√dt);  d = 1/u
//! p  = 1/2 + 1/2·(r - q - σ²/2)·√dt / σ         risk-neutral up-probability
//! disc = e^(-r·dt)                              one-step discount
//! ```
//! This is the **equal-jumps** CRR probability used by QuantLib's
//! `BinomialVanillaEngine("crr")` (drift carried in `p` rather than via the
//! `(e^((r-q)dt)-d)/(u-d)` form). Both are valid CRR schemes converging to
//! Black-Scholes as `n → ∞`; this convention matches QuantLib node-for-node, so
//! the validation cross-check is exact at matched step counts.
//! Terminal payoffs are discounted by backward induction. For an American
//! option, each node takes `max(continuation, intrinsic)` so early exercise is
//! captured; for a European option only the continuation value carries back.
//!
//! This is OXIS's primary American-option pricer for Phase 1. European prices
//! converge to Black-Scholes as `n → ∞` (asserted in the tests). Edge cases
//! (`T=0`, `σ=0`, `S=0`, invalid inputs) are handled as exact limits, never
//! `NaN`/`Inf`/panic.

use crate::core::{ExerciseStyle, MarketData, OptionType, OxisError};

/// A sensible default step count: accurate to a few basis points for typical
/// inputs while staying fast. Callers override via the `steps` argument.
pub const DEFAULT_STEPS: usize = 1000;

/// Price a vanilla option with the CRR binomial tree.
///
/// `expiry` is the time to expiry in years; `steps` is the number of tree
/// steps (`>= 1`). Returns the present value.
///
/// # Errors
/// [`OxisError::InvalidInput`] for inputs outside the model's domain
/// (non-positive strike, negative spot/vol/time, zero steps).
pub fn binomial(
    option_type: OptionType,
    style: ExerciseStyle,
    market: &MarketData,
    strike: f64,
    expiry: f64,
    steps: usize,
) -> Result<f64, OxisError> {
    let MarketData {
        spot: s,
        rate: r,
        volatility: sigma,
        dividend_yield: q,
    } = *market;

    if strike <= 0.0 {
        return Err(OxisError::invalid_input("strike must be > 0"));
    }
    if s < 0.0 {
        return Err(OxisError::invalid_input("spot must be >= 0"));
    }
    if sigma < 0.0 {
        return Err(OxisError::invalid_input("volatility must be >= 0"));
    }
    if expiry < 0.0 {
        return Err(OxisError::invalid_input("time to expiry must be >= 0"));
    }
    if steps == 0 {
        return Err(OxisError::invalid_input("steps must be >= 1"));
    }

    // T = 0: value is the intrinsic payoff (both styles agree).
    if expiry == 0.0 {
        return Ok(option_type.intrinsic(s, strike));
    }
    // σ = 0: deterministic growth at (r - q). A European reduces to the
    // discounted forward intrinsic; an American may exercise immediately, so it
    // is the max of that and today's intrinsic.
    if sigma == 0.0 {
        let forward = s * ((r - q) * expiry).exp();
        let euro = (-r * expiry).exp() * option_type.intrinsic(forward, strike);
        return Ok(match style {
            ExerciseStyle::European => euro,
            ExerciseStyle::American => euro.max(option_type.intrinsic(s, strike)),
        });
    }

    let n = steps;
    let dt = expiry / n as f64;
    let sqrt_dt = dt.sqrt();
    let u = (sigma * sqrt_dt).exp();
    let d = 1.0 / u;
    let disc = (-r * dt).exp();
    // Equal-jumps CRR up-probability (QuantLib's convention): drift carried in p.
    let drift = r - q - 0.5 * sigma * sigma;
    let p = 0.5 + 0.5 * drift * sqrt_dt / sigma;

    // If the tree is degenerate (p outside [0,1] from extreme inputs), the model
    // is being pushed outside its arbitrage-free regime; report rather than
    // return a nonsense price.
    if !(0.0..=1.0).contains(&p) {
        return Err(OxisError::numerical(
            "binomial: risk-neutral probability outside [0,1] (check inputs/steps)",
        ));
    }

    // Terminal asset prices and payoffs. Node j (0..=n) has j up-moves.
    // S_T = S · u^j · d^(n-j) = S · d^n · (u/d)^j, built incrementally.
    let mut values = Vec::with_capacity(n + 1);
    let s_min = s * d.powi(n as i32);
    let u_over_d = u / d;
    for j in 0..=n {
        let st = s_min * u_over_d.powi(j as i32);
        values.push(option_type.intrinsic(st, strike));
    }

    // Backward induction.
    let american = matches!(style, ExerciseStyle::American);
    for step in (0..n).rev() {
        for j in 0..=step {
            let cont = disc * (p * values[j + 1] + (1.0 - p) * values[j]);
            values[j] = if american {
                // Spot at this node: S · d^step · (u/d)^j.
                let st = s * d.powi(step as i32) * u_over_d.powi(j as i32);
                cont.max(option_type.intrinsic(st, strike))
            } else {
                cont
            };
        }
    }

    Ok(values[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::EuropeanOption;
    use crate::pricing::black_scholes;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b} (tol {tol})");
    }

    #[test]
    fn european_converges_to_black_scholes() {
        // S=100, K=100, r=5%, σ=20%, T=1, q=0. CRR converges ~O(1/n); 2000
        // steps brings it within a couple of cents of the closed form.
        let m = MarketData::new(100.0, 0.05, 0.2, 0.0);
        let bs = black_scholes(
            &EuropeanOption {
                strike: 100.0,
                expiry_years: 1.0,
                option_type: OptionType::Call,
            },
            &m,
        )
        .unwrap();
        let tree = binomial(
            OptionType::Call,
            ExerciseStyle::European,
            &m,
            100.0,
            1.0,
            2000,
        )
        .unwrap();
        close(tree, bs, 5e-2);
    }

    #[test]
    fn european_put_converges_to_black_scholes() {
        let m = MarketData::new(100.0, 0.05, 0.2, 0.03);
        let bs = black_scholes(
            &EuropeanOption {
                strike: 105.0,
                expiry_years: 0.75,
                option_type: OptionType::Put,
            },
            &m,
        )
        .unwrap();
        let tree = binomial(
            OptionType::Put,
            ExerciseStyle::European,
            &m,
            105.0,
            0.75,
            2000,
        )
        .unwrap();
        close(tree, bs, 5e-2);
    }

    #[test]
    fn american_at_least_european() {
        // An American option is never worth less than its European twin.
        let m = MarketData::new(100.0, 0.05, 0.3, 0.04);
        for ot in [OptionType::Call, OptionType::Put] {
            let euro = binomial(ot, ExerciseStyle::European, &m, 100.0, 1.0, 500).unwrap();
            let amer = binomial(ot, ExerciseStyle::American, &m, 100.0, 1.0, 500).unwrap();
            assert!(amer >= euro - 1e-12, "{ot:?}: amer {amer} < euro {euro}");
        }
    }

    #[test]
    fn american_call_no_dividend_equals_european() {
        // Classic result: an American call on a non-dividend-paying stock is
        // never exercised early, so it equals the European call.
        let m = MarketData::new(100.0, 0.05, 0.25, 0.0);
        let euro = binomial(
            OptionType::Call,
            ExerciseStyle::European,
            &m,
            90.0,
            1.0,
            800,
        )
        .unwrap();
        let amer = binomial(
            OptionType::Call,
            ExerciseStyle::American,
            &m,
            90.0,
            1.0,
            800,
        )
        .unwrap();
        close(amer, euro, 1e-6);
    }

    #[test]
    fn edge_cases() {
        let m = MarketData::new(100.0, 0.05, 0.2, 0.0);
        // T = 0 -> intrinsic.
        assert_eq!(
            binomial(
                OptionType::Call,
                ExerciseStyle::American,
                &m,
                90.0,
                0.0,
                100
            )
            .unwrap(),
            10.0
        );
        // S = 0 -> call worthless.
        let m0 = MarketData::new(0.0, 0.05, 0.2, 0.0);
        assert_eq!(
            binomial(
                OptionType::Call,
                ExerciseStyle::European,
                &m0,
                100.0,
                1.0,
                100
            )
            .unwrap(),
            0.0
        );
        // σ = 0 European -> discounted forward intrinsic.
        let mz = MarketData::new(100.0, 0.05, 0.0, 0.0);
        let expected = (-0.05_f64).exp() * (100.0 * 0.05_f64.exp() - 100.0);
        close(
            binomial(
                OptionType::Call,
                ExerciseStyle::European,
                &mz,
                100.0,
                1.0,
                100,
            )
            .unwrap(),
            expected,
            1e-12,
        );
    }

    #[test]
    fn rejects_invalid_inputs() {
        let m = MarketData::new(100.0, 0.05, 0.2, 0.0);
        assert!(binomial(OptionType::Call, ExerciseStyle::European, &m, 0.0, 1.0, 100).is_err());
        assert!(binomial(OptionType::Call, ExerciseStyle::European, &m, 100.0, 1.0, 0).is_err());
        let bad = MarketData::new(100.0, 0.05, -0.2, 0.0);
        assert!(
            binomial(
                OptionType::Call,
                ExerciseStyle::European,
                &bad,
                100.0,
                1.0,
                100
            )
            .is_err()
        );
    }
}
