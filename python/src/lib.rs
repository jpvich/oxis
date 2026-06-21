//! Python bindings for OXIS (Ring 1).
//!
//! A thin **Kind A** wrapper: it converts Python arguments into the plain core
//! types, calls the *same* pure pricing core the CLI uses, and maps
//! [`OxisError`](oxis_core::OxisError) to a Python `ValueError`. No pricing
//! logic lives here — bindings never duplicate the core.

#![forbid(unsafe_code)]

use oxis_core::{EuropeanOption, ExerciseStyle, MarketData, OptionType};
use oxis_greeks::analytic_greeks;
use oxis_pricing::{
    DEFAULT_STEPS, McConfig, binomial as binomial_core, black_scholes as bs_core,
    implied_volatility as iv_core, lsm_american as lsm_core,
    monte_carlo_european as mc_european_core,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Parse `"call"`/`"put"` (case-insensitive) into the core option type.
fn parse_option_type(s: &str) -> PyResult<OptionType> {
    match s.to_ascii_lowercase().as_str() {
        "call" | "c" => Ok(OptionType::Call),
        "put" | "p" => Ok(OptionType::Put),
        other => Err(PyValueError::new_err(format!(
            "option_type must be 'call' or 'put', got {other:?}"
        ))),
    }
}

/// Parse `"european"`/`"american"` (case-insensitive) into the core style.
fn parse_exercise(s: &str) -> PyResult<ExerciseStyle> {
    match s.to_ascii_lowercase().as_str() {
        "european" | "euro" | "e" => Ok(ExerciseStyle::European),
        "american" | "amer" | "a" => Ok(ExerciseStyle::American),
        other => Err(PyValueError::new_err(format!(
            "style must be 'european' or 'american', got {other:?}"
        ))),
    }
}

/// Price a European option with the Black-Scholes-Merton closed form.
///
/// ```python
/// import oxis
/// oxis.black_scholes(spot=100, strike=105, rate=0.05, vol=0.2, t=1.0, option_type="call")
/// # -> 8.021352235143176
/// ```
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, vol, t, option_type, dividend_yield=0.0))]
#[allow(clippy::too_many_arguments)]
fn black_scholes(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    t: f64,
    option_type: &str,
    dividend_yield: f64,
) -> PyResult<f64> {
    let option = EuropeanOption {
        strike,
        expiry_years: t,
        option_type: parse_option_type(option_type)?,
    };
    let market = MarketData::new(spot, rate, vol, dividend_yield);
    bs_core(&option, &market).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Price a European option and return a dict with the inputs and the price.
///
/// Mirrors the CLI's `PriceResult` shape so the Python and CLI surfaces agree.
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, vol, t, option_type, dividend_yield=0.0))]
#[allow(clippy::too_many_arguments)]
fn price<'py>(
    py: Python<'py>,
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    t: f64,
    option_type: &str,
    dividend_yield: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let ot = parse_option_type(option_type)?;
    let option = EuropeanOption {
        strike,
        expiry_years: t,
        option_type: ot,
    };
    let market = MarketData::new(spot, rate, vol, dividend_yield);
    let value = bs_core(&option, &market).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let d = PyDict::new(py);
    d.set_item("model", "black-scholes")?;
    d.set_item("option_type", ot.as_str())?;
    d.set_item("exercise", "european")?;
    d.set_item("spot", spot)?;
    d.set_item("strike", strike)?;
    d.set_item("rate", rate)?;
    d.set_item("volatility", vol)?;
    d.set_item("time", t)?;
    d.set_item("dividend_yield", dividend_yield)?;
    d.set_item("price", value)?;
    Ok(d)
}

/// Price a vanilla option with the CRR binomial tree (European or American).
///
/// ```python
/// oxis.binomial(spot=100, strike=110, rate=0.05, vol=0.3, t=1.0,
///               option_type="put", style="american")
/// ```
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, vol, t, option_type, style="european",
                    steps=DEFAULT_STEPS, dividend_yield=0.0))]
#[allow(clippy::too_many_arguments)]
fn binomial(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    t: f64,
    option_type: &str,
    style: &str,
    steps: usize,
    dividend_yield: f64,
) -> PyResult<f64> {
    let ot = parse_option_type(option_type)?;
    let es = parse_exercise(style)?;
    let market = MarketData::new(spot, rate, vol, dividend_yield);
    binomial_core(ot, es, &market, strike, t, steps)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Analytic Black-Scholes Greeks for a European option, as a dict
/// (`delta`, `gamma`, `vega`, `theta`, `rho`).
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, vol, t, option_type, dividend_yield=0.0))]
#[allow(clippy::too_many_arguments)]
fn greeks<'py>(
    py: Python<'py>,
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    t: f64,
    option_type: &str,
    dividend_yield: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let ot = parse_option_type(option_type)?;
    let option = EuropeanOption {
        strike,
        expiry_years: t,
        option_type: ot,
    };
    let market = MarketData::new(spot, rate, vol, dividend_yield);
    let g = analytic_greeks(&option, &market).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let d = PyDict::new(py);
    d.set_item("delta", g.delta)?;
    d.set_item("gamma", g.gamma)?;
    d.set_item("vega", g.vega)?;
    d.set_item("theta", g.theta)?;
    d.set_item("rho", g.rho)?;
    Ok(d)
}

/// Price a European option by Monte Carlo, returning a dict with the price and
/// its standard error (`{"price": ..., "standard_error": ...}`).
///
/// ```python
/// oxis.monte_carlo(spot=100, strike=105, rate=0.05, vol=0.2, t=1.0,
///                  option_type="call", paths=1_000_000, seed=42)
/// ```
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, vol, t, option_type, paths=100_000,
                    seed=42, dividend_yield=0.0))]
#[allow(clippy::too_many_arguments)]
fn monte_carlo<'py>(
    py: Python<'py>,
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    t: f64,
    option_type: &str,
    paths: usize,
    seed: u64,
    dividend_yield: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let option = EuropeanOption {
        strike,
        expiry_years: t,
        option_type: parse_option_type(option_type)?,
    };
    let market = MarketData::new(spot, rate, vol, dividend_yield);
    let cfg = McConfig {
        paths,
        steps: 1,
        seed,
    };
    let est = mc_european_core(&option, &market, &cfg)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let d = PyDict::new(py);
    d.set_item("price", est.price)?;
    d.set_item("standard_error", est.standard_error)?;
    Ok(d)
}

/// Price an American option by Longstaff-Schwartz Monte Carlo, returning a dict
/// with the price and its standard error.
///
/// ```python
/// oxis.lsm(spot=100, strike=110, rate=0.05, vol=0.3, t=1.0, option_type="put",
///          paths=200_000, steps=50, seed=42)
/// ```
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, vol, t, option_type, paths=100_000,
                    steps=50, seed=42, dividend_yield=0.0))]
#[allow(clippy::too_many_arguments)]
fn lsm<'py>(
    py: Python<'py>,
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    t: f64,
    option_type: &str,
    paths: usize,
    steps: usize,
    seed: u64,
    dividend_yield: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let ot = parse_option_type(option_type)?;
    let market = MarketData::new(spot, rate, vol, dividend_yield);
    let cfg = McConfig { paths, steps, seed };
    let est =
        lsm_core(ot, &market, strike, t, &cfg).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let d = PyDict::new(py);
    d.set_item("price", est.price)?;
    d.set_item("standard_error", est.standard_error)?;
    Ok(d)
}

/// Solve for the Black-Scholes implied volatility matching `market_price`.
#[pyfunction]
#[pyo3(signature = (market_price, spot, strike, rate, t, option_type, dividend_yield=0.0))]
fn implied_volatility(
    market_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    t: f64,
    option_type: &str,
    dividend_yield: f64,
) -> PyResult<f64> {
    let ot = parse_option_type(option_type)?;
    let option = EuropeanOption {
        strike,
        expiry_years: t,
        option_type: ot,
    };
    // Volatility field is ignored by the solver (it is the unknown).
    let market = MarketData::new(spot, rate, 0.0, dividend_yield);
    iv_core(&option, market_price, &market).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The `oxis` Python module.
#[pymodule]
fn oxis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(black_scholes, m)?)?;
    m.add_function(wrap_pyfunction!(price, m)?)?;
    m.add_function(wrap_pyfunction!(binomial, m)?)?;
    m.add_function(wrap_pyfunction!(monte_carlo, m)?)?;
    m.add_function(wrap_pyfunction!(lsm, m)?)?;
    m.add_function(wrap_pyfunction!(greeks, m)?)?;
    m.add_function(wrap_pyfunction!(implied_volatility, m)?)?;
    Ok(())
}
