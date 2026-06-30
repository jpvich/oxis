//! Cross-engine agreement: the four American-put pricers must agree, within a
//! documented band, on one canonical in-the-money contract.
//!
//! This is the headline of M9 — the same option priced four ways:
//!   binomial CRR (oracle)  ·  classical Longstaff-Schwartz  ·  Deep LSM  ·  DOS
//! All three Monte-Carlo estimators are low-biased, so each must land within
//! `5·SE + 0.60` of the QuantLib-validated tree. The comparison table is printed.

use oxis_core::{ExerciseStyle, MarketData, OptionType};
use oxis_ml::{AmericanMlConfig, deep_lsm_american, dos_american};
use oxis_pricing::{McConfig, binomial, lsm_american};

#[test]
fn engines_agree_on_itm_put() {
    let market = MarketData::new(100.0, 0.05, 0.3, 0.0);
    let (strike, expiry) = (100.0, 1.0);
    let (paths, steps, seed) = (4096, 10, 11);

    let tree = binomial(
        OptionType::Put,
        ExerciseStyle::American,
        &market,
        strike,
        expiry,
        2000,
    )
    .unwrap();

    let mc = McConfig { paths, steps, seed };
    let classical = lsm_american(OptionType::Put, &market, strike, expiry, &mc).unwrap();

    let cfg = AmericanMlConfig {
        market,
        strike,
        expiry,
        paths,
        steps,
        seed,
        hidden: vec![16],
        epochs: 20,
    };
    let deep = deep_lsm_american(OptionType::Put, &cfg).unwrap();
    let dos = dos_american(OptionType::Put, &cfg).unwrap();

    eprintln!("American put S=100 K=100 r=0.05 σ=0.3 T=1 (paths={paths}, steps={steps}):");
    eprintln!("  binomial(2000) = {tree:.4}  (oracle)");
    eprintln!(
        "  classical LSM  = {:.4}  (se {:.4}, |Δ| {:.4})",
        classical.price,
        classical.standard_error,
        (classical.price - tree).abs()
    );
    eprintln!(
        "  Deep LSM       = {:.4}  (se {:.4}, |Δ| {:.4})",
        deep.price,
        deep.standard_error,
        (deep.price - tree).abs()
    );
    eprintln!(
        "  DOS            = {:.4}  (se {:.4}, |Δ| {:.4})",
        dos.price,
        dos.standard_error,
        (dos.price - tree).abs()
    );

    for (name, est) in [
        ("classical LSM", classical),
        ("Deep LSM", deep),
        ("DOS", dos),
    ] {
        let budget = 5.0 * est.standard_error + 0.60;
        assert!(
            (est.price - tree).abs() <= budget,
            "{name}={} vs binomial={tree} exceeds band {budget:.4}",
            est.price
        );
    }
}
