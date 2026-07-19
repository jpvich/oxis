//! Validation of `oxis-portfolio` against a numpy / scipy oracle.
//!
//! Portfolio analytics (valuation, TWR, MWR/IRR, allocation, risk aggregation,
//! Markowitz optimization) have no QuantLib equivalent, so the oracle is
//! numpy/scipy — matrix algebra via `np.linalg.solve`, IRR via
//! `scipy.optimize.brentq`. References live in `validation/reference/portfolio.json`;
//! this test recomputes each case with the OXIS functions and asserts agreement
//! within the file's tolerance (IRR uses a slightly looser band).

use oxis::core::Date;
use oxis::portfolio::{
    Holding, efficient_frontier_point, min_variance_weights, mwr, portfolio_risk, tangency_weights,
    twr, value_holdings, weights,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct PortfolioFile {
    oracle: String,
    oracle_version: String,
    tolerance: f64,
    irr_tolerance: f64,
    cases: Vec<PortfolioCase>,
}

#[derive(Debug, Deserialize)]
struct PortfolioCase {
    name: String,
    // valuation / allocation
    holdings: Option<Vec<(String, f64, f64, f64)>>,
    market_values: Option<Vec<f64>>,
    unrealized_pnls: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
    total_cost_basis: Option<f64>,
    total_market_value: Option<f64>,
    total_unrealized_pnl: Option<f64>,
    // twr
    values: Option<Vec<f64>>,
    flows: Option<Vec<f64>>,
    twr: Option<f64>,
    // mwr
    dates: Option<Vec<String>>,
    amounts: Option<Vec<f64>>,
    mwr: Option<f64>,
    // risk
    returns: Option<Vec<Vec<f64>>>,
    periods_per_year: Option<f64>,
    confidence: Option<f64>,
    variance: Option<f64>,
    volatility: Option<f64>,
    annualized_volatility: Option<f64>,
    historical_var: Option<f64>,
    parametric_var: Option<f64>,
    // optimize
    mean: Option<Vec<f64>>,
    cov: Option<Vec<Vec<f64>>>,
    rf: Option<f64>,
    target: Option<f64>,
    min_variance_weights: Option<Vec<f64>>,
    tangency_weights: Option<Vec<f64>>,
    frontier_weights: Option<Vec<f64>>,
}

fn load(name: &str) -> PortfolioFile {
    let path = format!(
        "{}/../../validation/reference/{}",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read reference data at {path}: {e}"));
    serde_json::from_str(&json).expect("reference data is valid JSON")
}

fn parse_date(s: &str) -> Date {
    let p: Vec<&str> = s.split('-').collect();
    Date::new(
        p[0].parse().unwrap(),
        p[1].parse().unwrap(),
        p[2].parse().unwrap(),
    )
    .unwrap()
}

#[test]
fn portfolio_matches_numpy_scipy() {
    let reference = load("portfolio.json");
    assert_eq!(reference.oracle, "numpy/scipy/pandas");
    assert!(!reference.cases.is_empty());
    let tol = reference.tolerance;
    let irr_tol = reference.irr_tolerance;
    let mut worst = 0.0_f64;

    let mut check = |name: &str, got: f64, exp: f64, tol: f64| {
        let err = (got - exp).abs();
        assert!(
            err <= tol,
            "{name}: error {err:.3e} > {tol:.1e} (got {got}, want {exp})"
        );
        worst = worst.max(err);
    };

    for case in &reference.cases {
        match case.name.as_str() {
            "valuation" => {
                let specs = case.holdings.as_ref().unwrap();
                let holdings: Vec<Holding> = specs
                    .iter()
                    .map(|(s, q, c, _p)| Holding::single(s.clone(), *q, *c))
                    .collect();
                let prices: Vec<f64> = specs.iter().map(|(_s, _q, _c, p)| *p).collect();
                let v = value_holdings(&holdings, &prices).unwrap();
                check(
                    "total_cost_basis",
                    v.total_cost_basis,
                    case.total_cost_basis.unwrap(),
                    tol,
                );
                check(
                    "total_market_value",
                    v.total_market_value,
                    case.total_market_value.unwrap(),
                    tol,
                );
                check(
                    "total_unrealized_pnl",
                    v.total_unrealized_pnl,
                    case.total_unrealized_pnl.unwrap(),
                    tol,
                );
                let mvs = case.market_values.as_ref().unwrap();
                let pnls = case.unrealized_pnls.as_ref().unwrap();
                let ws = case.weights.as_ref().unwrap();
                for (i, h) in v.holdings.iter().enumerate() {
                    check("market_value", h.market_value, mvs[i], tol);
                    check("unrealized_pnl", h.unrealized_pnl, pnls[i], tol);
                    check("weight", h.weight, ws[i], tol);
                }
            }
            "twr" => {
                let got = twr(case.values.as_ref().unwrap(), case.flows.as_ref().unwrap()).unwrap();
                check("twr", got, case.twr.unwrap(), tol);
            }
            "mwr" => {
                let cfs: Vec<(Date, f64)> = case
                    .dates
                    .as_ref()
                    .unwrap()
                    .iter()
                    .zip(case.amounts.as_ref().unwrap().iter())
                    .map(|(d, &a)| (parse_date(d), a))
                    .collect();
                let got = mwr(&cfs).unwrap();
                check("mwr", got, case.mwr.unwrap(), irr_tol);
            }
            "allocation" => {
                let got = weights(case.market_values.as_ref().unwrap()).unwrap();
                let exp = case.weights.as_ref().unwrap();
                for (i, &w) in got.iter().enumerate() {
                    check("allocation_weight", w, exp[i], tol);
                }
            }
            "risk" => {
                let r = portfolio_risk(
                    case.returns.as_ref().unwrap(),
                    case.weights.as_ref().unwrap(),
                    case.periods_per_year.unwrap(),
                    case.confidence.unwrap(),
                )
                .unwrap();
                check("variance", r.variance, case.variance.unwrap(), tol);
                check("volatility", r.volatility, case.volatility.unwrap(), tol);
                check(
                    "annualized_volatility",
                    r.annualized_volatility,
                    case.annualized_volatility.unwrap(),
                    tol,
                );
                check(
                    "historical_var",
                    r.historical_var,
                    case.historical_var.unwrap(),
                    tol,
                );
                check(
                    "parametric_var",
                    r.parametric_var,
                    case.parametric_var.unwrap(),
                    tol,
                );
            }
            "optimize" => {
                let cov = case.cov.as_ref().unwrap();
                let mean = case.mean.as_ref().unwrap();
                let mv = min_variance_weights(cov).unwrap();
                let tan = tangency_weights(cov, mean, case.rf.unwrap()).unwrap();
                let fr = efficient_frontier_point(cov, mean, case.target.unwrap()).unwrap();
                for (i, &w) in mv.iter().enumerate() {
                    check(
                        "min_variance_weight",
                        w,
                        case.min_variance_weights.as_ref().unwrap()[i],
                        tol,
                    );
                }
                for (i, &w) in tan.iter().enumerate() {
                    check(
                        "tangency_weight",
                        w,
                        case.tangency_weights.as_ref().unwrap()[i],
                        tol,
                    );
                }
                for (i, &w) in fr.iter().enumerate() {
                    check(
                        "frontier_weight",
                        w,
                        case.frontier_weights.as_ref().unwrap()[i],
                        tol,
                    );
                }
            }
            other => panic!("unknown portfolio case {other:?}"),
        }
    }

    eprintln!(
        "validated {} portfolio cases vs {} {} — worst |Δ| {worst:.3e}",
        reference.cases.len(),
        reference.oracle,
        reference.oracle_version,
    );
}
