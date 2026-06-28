//! Python bindings for OXIS (Ring 1).
//!
//! A thin **Kind A** wrapper: it converts Python arguments into the plain core
//! types, calls the *same* pure pricing core the CLI uses, and maps
//! [`OxisError`](oxis_core::OxisError) to a Python `ValueError`. No pricing
//! logic lives here — bindings never duplicate the core.

#![forbid(unsafe_code)]

use oxis_bonds::{Cashflow, FixedRateBond as BondCore};
use oxis_core::{EuropeanOption, ExerciseStyle, MarketData, OptionType};
use oxis_curves::{Interpolation, YieldCurve as CurveCore};
use oxis_greeks::analytic_greeks;
use oxis_ml::{BsSpec, TrainConfig, differential_ml_price};
use oxis_portfolio::{
    Holding, covariance_matrix as cov_matrix_core, efficient_frontier_point,
    min_variance_weights as min_var_core, mwr as mwr_core, portfolio_risk as portfolio_risk_core,
    tangency_weights as tangency_core, twr as twr_core, value_holdings,
    weights as alloc_weights_core,
};
use oxis_pricing::{
    BarrierType, DEFAULT_STEPS, LookbackStrike, McConfig,
    arithmetic_asian_price as arith_asian_core, barrier_price as barrier_core,
    binomial as binomial_core, black_scholes as bs_core, geometric_asian_price as geo_asian_core,
    implied_volatility as iv_core, lookback_price as lookback_core, lsm_american as lsm_core,
    monte_carlo_european as mc_european_core,
};
use oxis_stats::{
    SampleKind, StatsRequest, acf as stats_acf, assemble as stats_assemble, beta as stats_beta,
    cornish_fisher_var, historical_es, historical_var, information_ratio,
    jarque_bera as stats_jarque_bera, max_drawdown as stats_max_drawdown, parametric_es,
    parametric_var, sharpe_ratio as stats_sharpe, simple_returns as stats_simple_returns,
    sortino_ratio as stats_sortino, tracking_error as stats_tracking_error,
};
use oxis_stochastic::{Process, ProcessResult, SimConfig, simulate_terminal};
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

/// Parse a curve interpolation name (case-insensitive) into the core enum.
fn parse_interp(s: &str) -> PyResult<Interpolation> {
    match s.to_ascii_lowercase().as_str() {
        "linear" => Ok(Interpolation::Linear),
        "log-linear" | "loglinear" | "log_linear" => Ok(Interpolation::LogLinear),
        "natural-cubic" | "naturalcubic" | "natural_cubic" | "cubic" => {
            Ok(Interpolation::NaturalCubic)
        }
        other => Err(PyValueError::new_err(format!(
            "interp must be 'linear', 'log-linear', or 'natural-cubic', got {other:?}"
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

/// A yield curve / term structure: build once, query many times.
///
/// Construct with one of the static methods, then query continuously-compounded
/// quantities (time in years, `Act/365`):
///
/// ```python
/// import oxis
/// c = oxis.YieldCurve.from_zero_rates([0.5, 1, 2, 5], [0.02, 0.025, 0.03, 0.035],
///                                     interp="natural-cubic")
/// c.discount(1.5), c.zero_rate(1.5), c.forward_rate(1.5, 2.5)
/// ```
#[pyclass(name = "YieldCurve")]
pub struct YieldCurve {
    inner: CurveCore,
}

#[pymethods]
impl YieldCurve {
    /// A flat continuously-compounded curve at `rate` for every maturity.
    #[staticmethod]
    fn flat(rate: f64) -> PyResult<Self> {
        CurveCore::flat(rate)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Build from continuously-compounded zero rates at the given pillar times.
    #[staticmethod]
    #[pyo3(signature = (times, rates, interp="log-linear"))]
    fn from_zero_rates(times: Vec<f64>, rates: Vec<f64>, interp: &str) -> PyResult<Self> {
        let i = parse_interp(interp)?;
        CurveCore::from_zero_rates(&times, &rates, i)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Build from discount factors at the given pillar times.
    #[staticmethod]
    #[pyo3(signature = (times, dfs, interp="log-linear"))]
    fn from_discount_factors(times: Vec<f64>, dfs: Vec<f64>, interp: &str) -> PyResult<Self> {
        let i = parse_interp(interp)?;
        CurveCore::from_discount_factors(&times, &dfs, i)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Discount factor `P(t)` (with `P(0) = 1`).
    fn discount(&self, t: f64) -> PyResult<f64> {
        self.inner
            .discount(t)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Continuously-compounded zero rate `z(t)`.
    fn zero_rate(&self, t: f64) -> PyResult<f64> {
        self.inner
            .zero_rate(t)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Continuously-compounded forward rate over `[t1, t2]`.
    fn forward_rate(&self, t1: f64, t2: f64) -> PyResult<f64> {
        self.inner
            .forward_rate(t1, t2)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// A fixed-rate bond: build once, price and analyse from a yield or a curve.
///
/// ```python
/// import oxis
/// b = oxis.FixedRateBond.regular(face=100, coupon_rate=0.05, frequency=2, n_periods=10)
/// b.clean_price_from_yield(0.05)        # ~100 (par)
/// b.yield_to_maturity(100.0)            # ~0.05
/// b.modified_duration(0.05), b.convexity(0.05)
/// ```
#[pyclass(name = "FixedRateBond")]
pub struct FixedRateBond {
    inner: BondCore,
}

#[pymethods]
impl FixedRateBond {
    /// A regular bond settling on a coupon date: `n_periods` equal coupons at
    /// `t = k/frequency`, plus `face` redeemed at the last. Accrued is zero.
    #[staticmethod]
    #[pyo3(signature = (face, coupon_rate, frequency, n_periods))]
    fn regular(face: f64, coupon_rate: f64, frequency: u32, n_periods: u32) -> PyResult<Self> {
        BondCore::regular(face, coupon_rate, frequency, n_periods)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Build from explicit cashflows: `times` (years) and matching `amounts`,
    /// plus the accrued interest at settlement.
    #[staticmethod]
    #[pyo3(signature = (times, amounts, frequency, accrued=0.0, face=100.0, coupon_rate=0.0))]
    fn from_cashflows(
        times: Vec<f64>,
        amounts: Vec<f64>,
        frequency: u32,
        accrued: f64,
        face: f64,
        coupon_rate: f64,
    ) -> PyResult<Self> {
        if times.len() != amounts.len() {
            return Err(PyValueError::new_err(
                "times and amounts must have equal length",
            ));
        }
        let cashflows = times
            .into_iter()
            .zip(amounts)
            .map(|(time, amount)| Cashflow { time, amount })
            .collect();
        BondCore::from_cashflows(face, coupon_rate, frequency, cashflows, accrued)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Accrued interest at settlement.
    #[getter]
    fn accrued(&self) -> f64 {
        self.inner.accrued
    }

    /// Dirty (full) price from a flat yield compounded at the coupon frequency.
    fn dirty_price_from_yield(&self, y: f64) -> PyResult<f64> {
        self.inner
            .dirty_price_from_yield(y)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Clean price from a flat yield (`dirty − accrued`).
    fn clean_price_from_yield(&self, y: f64) -> PyResult<f64> {
        self.inner
            .clean_price_from_yield(y)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Yield-to-maturity from a quoted clean price.
    fn yield_to_maturity(&self, clean_price: f64) -> PyResult<f64> {
        self.inner
            .yield_to_maturity(clean_price)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Dirty and clean price discounted on a yield curve, returned as a tuple.
    fn price_from_curve(&self, curve: &YieldCurve) -> PyResult<(f64, f64)> {
        self.inner
            .price_from_curve(&curve.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Macaulay duration at yield `y`.
    fn macaulay_duration(&self, y: f64) -> PyResult<f64> {
        self.inner
            .macaulay_duration(y)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Modified duration at yield `y`.
    fn modified_duration(&self, y: f64) -> PyResult<f64> {
        self.inner
            .modified_duration(y)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Convexity at yield `y`.
    fn convexity(&self, y: f64) -> PyResult<f64> {
        self.inner
            .convexity(y)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// ----------------------------------------------------------------------------
// Exotic options (Ring 2).
// ----------------------------------------------------------------------------

fn parse_barrier_type(s: &str) -> PyResult<BarrierType> {
    match s.to_ascii_lowercase().replace('_', "-").as_str() {
        "down-in" | "downin" | "di" => Ok(BarrierType::DownIn),
        "down-out" | "downout" | "do" => Ok(BarrierType::DownOut),
        "up-in" | "upin" | "ui" => Ok(BarrierType::UpIn),
        "up-out" | "upout" | "uo" => Ok(BarrierType::UpOut),
        other => Err(PyValueError::new_err(format!(
            "barrier_type must be one of down-in/down-out/up-in/up-out, got {other:?}"
        ))),
    }
}

/// Price a continuously monitored single-barrier option (zero rebate).
///
/// ```python
/// oxis.barrier_price(spot=100, strike=100, rate=0.05, vol=0.25, t=1.0,
///                    option_type="call", barrier_type="down-out", barrier=90)
/// ```
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, vol, t, option_type, barrier_type, barrier,
                    dividend_yield=0.0))]
#[allow(clippy::too_many_arguments)]
fn barrier_price(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    t: f64,
    option_type: &str,
    barrier_type: &str,
    barrier: f64,
    dividend_yield: f64,
) -> PyResult<f64> {
    let option = EuropeanOption {
        strike,
        expiry_years: t,
        option_type: parse_option_type(option_type)?,
    };
    let market = MarketData::new(spot, rate, vol, dividend_yield);
    barrier_core(&option, &market, parse_barrier_type(barrier_type)?, barrier)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Price a continuous lookback option (freshly issued: extremum = spot).
///
/// ```python
/// oxis.lookback_price(spot=100, strike=0, rate=0.06, vol=0.3, t=1.0,
///                     option_type="call", strike_type="floating", dividend_yield=0.02)
/// ```
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, vol, t, option_type, strike_type, dividend_yield=0.0))]
#[allow(clippy::too_many_arguments)]
fn lookback_price(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    t: f64,
    option_type: &str,
    strike_type: &str,
    dividend_yield: f64,
) -> PyResult<f64> {
    let st = match strike_type.to_ascii_lowercase().as_str() {
        "floating" | "float" => LookbackStrike::Floating,
        "fixed" => LookbackStrike::Fixed,
        other => {
            return Err(PyValueError::new_err(format!(
                "strike_type must be 'floating' or 'fixed', got {other:?}"
            )));
        }
    };
    let option = EuropeanOption {
        strike,
        expiry_years: t,
        option_type: parse_option_type(option_type)?,
    };
    let market = MarketData::new(spot, rate, vol, dividend_yield);
    lookback_core(&option, &market, st).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Price an average-price Asian option, returning a dict with the price (and, for
/// the arithmetic average, its Monte Carlo standard error; `None` for geometric).
///
/// ```python
/// oxis.asian_price(spot=100, strike=100, rate=0.05, vol=0.2, t=1.0,
///                  option_type="call", average="arithmetic", n_fixings=50)
/// ```
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, vol, t, option_type, average="geometric",
                    n_fixings=50, paths=100_000, seed=42, dividend_yield=0.0))]
#[allow(clippy::too_many_arguments)]
fn asian_price<'py>(
    py: Python<'py>,
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    t: f64,
    option_type: &str,
    average: &str,
    n_fixings: usize,
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
    let d = PyDict::new(py);
    match average.to_ascii_lowercase().as_str() {
        "geometric" | "geo" => {
            let price = geo_asian_core(&option, &market)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            d.set_item("price", price)?;
            d.set_item("standard_error", py.None())?;
        }
        "arithmetic" | "arith" => {
            let cfg = SimConfig {
                paths,
                steps: 0,
                seed,
            };
            let est = arith_asian_core(&option, &market, n_fixings, &cfg)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            d.set_item("price", est.price)?;
            d.set_item("standard_error", est.standard_error)?;
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "average must be 'geometric' or 'arithmetic', got {other:?}"
            )));
        }
    }
    Ok(d)
}

// ----------------------------------------------------------------------------
// Stochastic process simulation (Ring 2).
// ----------------------------------------------------------------------------

/// Simulate a stochastic process and return a dict of its terminal moments
/// (sample vs closed-form mean/std). Parameters not used by the chosen process
/// are ignored; each has a default.
///
/// `process` is one of `gbm`, `ou`, `vasicek`, `cir`, `merton`, `heston`.
///
/// ```python
/// oxis.simulate_process("heston", x0=100, t=1.0, steps=200, paths=100_000,
///                       mu=0.04, v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=-0.6)
/// ```
#[pyfunction]
#[pyo3(signature = (process, x0=100.0, t=1.0, steps=100, paths=100_000, seed=42,
                    mu=0.05, sigma=0.20, kappa=1.0, theta=0.04, lambda_=0.5,
                    jump_mean=-0.10, jump_std=0.15, v0=0.04, xi=0.30, rho=-0.60))]
#[allow(clippy::too_many_arguments)]
fn simulate_process<'py>(
    py: Python<'py>,
    process: &str,
    x0: f64,
    t: f64,
    steps: usize,
    paths: usize,
    seed: u64,
    mu: f64,
    sigma: f64,
    kappa: f64,
    theta: f64,
    lambda_: f64,
    jump_mean: f64,
    jump_std: f64,
    v0: f64,
    xi: f64,
    rho: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let proc = match process.to_ascii_lowercase().as_str() {
        "gbm" => Process::Gbm { mu, sigma },
        "ou" | "ornstein-uhlenbeck" => Process::OrnsteinUhlenbeck {
            kappa,
            theta,
            sigma,
        },
        "vasicek" => Process::Vasicek {
            kappa,
            theta,
            sigma,
        },
        "cir" => Process::Cir {
            kappa,
            theta,
            sigma,
        },
        "merton" | "merton-jump" => Process::MertonJump {
            mu,
            sigma,
            lambda: lambda_,
            jump_mean,
            jump_std,
        },
        "heston" => Process::Heston {
            mu,
            v0,
            kappa,
            theta,
            xi,
            rho,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "process must be one of gbm/ou/vasicek/cir/merton/heston, got {other:?}"
            )));
        }
    };
    let cfg = SimConfig { paths, steps, seed };
    let sample =
        simulate_terminal(&proc, x0, t, &cfg).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let r = ProcessResult::from_simulation(&proc, x0, t, &cfg, &sample);

    let d = PyDict::new(py);
    d.set_item("process", r.process)?;
    d.set_item("x0", r.x0)?;
    d.set_item("t", r.t)?;
    d.set_item("paths", r.paths)?;
    d.set_item("steps", r.steps)?;
    d.set_item("sample_mean", r.sample_mean)?;
    d.set_item("mean_std_error", r.mean_std_error)?;
    d.set_item("sample_std", r.sample_std)?;
    d.set_item("analytic_mean", r.analytic_mean)?;
    d.set_item("analytic_std", r.analytic_std)?;
    d.set_item("mean_abs_error", r.mean_abs_error)?;
    d.set_item("std_abs_error", r.std_abs_error)?;
    Ok(d)
}

// ----------------------------------------------------------------------------
// Statistics & risk metrics (Ring 3).
// ----------------------------------------------------------------------------

/// Compute descriptive, risk, and performance statistics for a series.
///
/// Provide exactly one of `returns`, `prices`, or `values`. `prices` derives
/// returns and enables drawdown / Calmar; `values` yields descriptive statistics
/// only. A `benchmark` (returns) enables beta / correlation / covariance /
/// tracking error / information ratio. Metrics that don't apply return `None`.
///
/// ```python
/// oxis.stats(returns=[0.01, -0.02, 0.015], periods_per_year=252, confidence=0.95)
/// ```
#[pyfunction]
#[pyo3(signature = (returns=None, prices=None, values=None, benchmark=None,
                    risk_free=0.0, periods_per_year=252.0, confidence=0.95, lag=None))]
#[allow(clippy::too_many_arguments)]
fn stats<'py>(
    py: Python<'py>,
    returns: Option<Vec<f64>>,
    prices: Option<Vec<f64>>,
    values: Option<Vec<f64>>,
    benchmark: Option<Vec<f64>>,
    risk_free: f64,
    periods_per_year: f64,
    confidence: f64,
    lag: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let (sample, kind): (Vec<f64>, SampleKind) = match (&returns, &prices, &values) {
        (Some(r), None, None) => (r.clone(), SampleKind::Returns),
        (None, Some(p), None) => (
            stats_simple_returns(p).map_err(|e| PyValueError::new_err(e.to_string()))?,
            SampleKind::Returns,
        ),
        (None, None, Some(v)) => (v.clone(), SampleKind::Values),
        (None, None, None) => {
            return Err(PyValueError::new_err(
                "provide exactly one of returns, prices, or values",
            ));
        }
        _ => {
            return Err(PyValueError::new_err(
                "use only one of returns, prices, or values",
            ));
        }
    };

    let req = StatsRequest {
        sample: &sample,
        kind,
        prices: prices.as_deref(),
        benchmark: benchmark.as_deref(),
        risk_free,
        periods_per_year,
        confidence,
        lag,
    };
    let r = stats_assemble(&req).map_err(|e| PyValueError::new_err(e.to_string()))?;

    // pyo3 maps `Option<T>` to `None`, so optional metrics serialize cleanly.
    let d = PyDict::new(py);
    d.set_item("count", r.count)?;
    d.set_item("periods_per_year", r.periods_per_year)?;
    d.set_item("confidence", r.confidence)?;
    d.set_item("mean", r.mean)?;
    d.set_item("variance", r.variance)?;
    d.set_item("std_dev", r.std_dev)?;
    d.set_item("skewness", r.skewness)?;
    d.set_item("excess_kurtosis", r.excess_kurtosis)?;
    d.set_item("jarque_bera", r.jarque_bera)?;
    d.set_item("jarque_bera_pvalue", r.jarque_bera_pvalue)?;
    d.set_item("autocorr_lag1", r.autocorr_lag1)?;
    d.set_item("autocorr_at_lag", r.autocorr_at_lag)?;
    d.set_item("cumulative_return", r.cumulative_return)?;
    d.set_item("annualized_return", r.annualized_return)?;
    d.set_item("annualized_volatility", r.annualized_volatility)?;
    d.set_item("sharpe", r.sharpe)?;
    d.set_item("sortino", r.sortino)?;
    d.set_item("historical_var", r.historical_var)?;
    d.set_item("historical_es", r.historical_es)?;
    d.set_item("parametric_var", r.parametric_var)?;
    d.set_item("parametric_es", r.parametric_es)?;
    d.set_item("cornish_fisher_var", r.cornish_fisher_var)?;
    d.set_item("max_drawdown", r.max_drawdown)?;
    d.set_item("max_drawdown_duration", r.max_drawdown_duration)?;
    d.set_item("calmar", r.calmar)?;
    d.set_item("covariance", r.covariance)?;
    d.set_item("correlation", r.correlation)?;
    d.set_item("beta", r.beta)?;
    d.set_item("tracking_error", r.tracking_error)?;
    d.set_item("information_ratio", r.information_ratio)?;
    Ok(d)
}

/// Annualized Sharpe ratio of a returns series.
#[pyfunction]
#[pyo3(signature = (returns, risk_free=0.0, periods_per_year=252.0))]
fn sharpe(returns: Vec<f64>, risk_free: f64, periods_per_year: f64) -> PyResult<f64> {
    stats_sharpe(&returns, risk_free, periods_per_year)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Annualized Sortino ratio of a returns series.
#[pyfunction]
#[pyo3(signature = (returns, mar=0.0, periods_per_year=252.0))]
fn sortino(returns: Vec<f64>, mar: f64, periods_per_year: f64) -> PyResult<f64> {
    stats_sortino(&returns, mar, periods_per_year).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Maximum drawdown of a price series, as a dict.
#[pyfunction]
fn max_drawdown(py: Python<'_>, prices: Vec<f64>) -> PyResult<Bound<'_, PyDict>> {
    let dd = stats_max_drawdown(&prices).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d = PyDict::new(py);
    d.set_item("max_drawdown", dd.max_drawdown)?;
    d.set_item("peak_index", dd.peak_index)?;
    d.set_item("trough_index", dd.trough_index)?;
    d.set_item("duration", dd.duration)?;
    Ok(d)
}

/// Value-at-Risk (positive loss). `method` ∈ {`historical`, `parametric`,
/// `cornish-fisher`}.
#[pyfunction]
#[pyo3(signature = (returns, confidence=0.95, method="historical"))]
fn value_at_risk(returns: Vec<f64>, confidence: f64, method: &str) -> PyResult<f64> {
    let v = match method.to_ascii_lowercase().as_str() {
        "historical" | "hist" => historical_var(&returns, confidence),
        "parametric" | "gaussian" => parametric_var(&returns, confidence),
        "cornish-fisher" | "cf" => cornish_fisher_var(&returns, confidence),
        other => {
            return Err(PyValueError::new_err(format!(
                "method must be 'historical', 'parametric', or 'cornish-fisher', got {other:?}"
            )));
        }
    };
    v.map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Expected Shortfall (positive loss). `method` ∈ {`historical`, `parametric`}.
#[pyfunction]
#[pyo3(signature = (returns, confidence=0.95, method="historical"))]
fn expected_shortfall(returns: Vec<f64>, confidence: f64, method: &str) -> PyResult<f64> {
    let v = match method.to_ascii_lowercase().as_str() {
        "historical" | "hist" => historical_es(&returns, confidence),
        "parametric" | "gaussian" => parametric_es(&returns, confidence),
        other => {
            return Err(PyValueError::new_err(format!(
                "method must be 'historical' or 'parametric', got {other:?}"
            )));
        }
    };
    v.map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Beta of an asset's returns against a benchmark's.
#[pyfunction]
fn beta(asset: Vec<f64>, benchmark: Vec<f64>) -> PyResult<f64> {
    stats_beta(&asset, &benchmark).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Annualized tracking error of a portfolio vs a benchmark (returns series).
#[pyfunction]
#[pyo3(signature = (portfolio, benchmark, periods_per_year=252.0))]
fn tracking_error(
    portfolio: Vec<f64>,
    benchmark: Vec<f64>,
    periods_per_year: f64,
) -> PyResult<f64> {
    stats_tracking_error(&portfolio, &benchmark, periods_per_year)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Annualized information ratio of a portfolio vs a benchmark (returns series).
#[pyfunction]
#[pyo3(signature = (portfolio, benchmark, periods_per_year=252.0))]
fn info_ratio(portfolio: Vec<f64>, benchmark: Vec<f64>, periods_per_year: f64) -> PyResult<f64> {
    information_ratio(&portfolio, &benchmark, periods_per_year)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Jarque-Bera normality test: `(statistic, p_value)`.
#[pyfunction]
fn jarque_bera(values: Vec<f64>) -> PyResult<(f64, f64)> {
    stats_jarque_bera(&values).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Autocorrelation function for lags `0..=max_lag`.
#[pyfunction]
fn acf(values: Vec<f64>, max_lag: usize) -> PyResult<Vec<f64>> {
    stats_acf(&values, max_lag).map_err(|e| PyValueError::new_err(e.to_string()))
}

// ----------------------------------------------------------------------------
// Portfolio analytics (Ring 3, M7).
// ----------------------------------------------------------------------------

/// Parse an ISO `YYYY-MM-DD` string into core year/month/day.
fn parse_iso_date(s: &str) -> PyResult<oxis_core::Date> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "date must be YYYY-MM-DD, got {s:?}"
        )));
    }
    let y = parts[0]
        .parse::<i32>()
        .map_err(|_| PyValueError::new_err("bad year"))?;
    let m = parts[1]
        .parse::<u8>()
        .map_err(|_| PyValueError::new_err("bad month"))?;
    let d = parts[2]
        .parse::<u8>()
        .map_err(|_| PyValueError::new_err("bad day"))?;
    oxis_core::Date::new(y, m, d).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Mark-to-market value holdings, returning totals + a per-holding breakdown.
///
/// `holdings` is a list of `(symbol, quantity, unit_cost, price)` tuples.
///
/// ```python
/// oxis.portfolio_value([("AAPL", 10, 150, 175), ("MSFT", 5, 300, 320)])
/// ```
#[pyfunction]
fn portfolio_value<'py>(
    py: Python<'py>,
    holdings: Vec<(String, f64, f64, f64)>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut hs = Vec::with_capacity(holdings.len());
    let mut prices = Vec::with_capacity(holdings.len());
    for (sym, qty, cost, price) in &holdings {
        hs.push(Holding::single(sym.clone(), *qty, *cost));
        prices.push(*price);
    }
    let v = value_holdings(&hs, &prices).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let rows = pyo3::types::PyList::empty(py);
    for h in &v.holdings {
        let d = PyDict::new(py);
        d.set_item("symbol", &h.symbol)?;
        d.set_item("quantity", h.quantity)?;
        d.set_item("average_cost", h.average_cost)?;
        d.set_item("price", h.price)?;
        d.set_item("cost_basis", h.cost_basis)?;
        d.set_item("market_value", h.market_value)?;
        d.set_item("unrealized_pnl", h.unrealized_pnl)?;
        d.set_item("unrealized_pnl_pct", h.unrealized_pnl_pct)?;
        d.set_item("weight", h.weight)?;
        rows.append(d)?;
    }
    let out = PyDict::new(py);
    out.set_item("n_holdings", v.n_holdings)?;
    out.set_item("total_cost_basis", v.total_cost_basis)?;
    out.set_item("total_market_value", v.total_market_value)?;
    out.set_item("total_unrealized_pnl", v.total_unrealized_pnl)?;
    out.set_item("total_unrealized_pnl_pct", v.total_unrealized_pnl_pct)?;
    out.set_item("holdings", rows)?;
    Ok(out)
}

/// Time-weighted return from period-boundary valuations and sub-period flows.
#[pyfunction]
fn twr(values: Vec<f64>, flows: Vec<f64>) -> PyResult<f64> {
    twr_core(&values, &flows).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Money-weighted return (IRR) from ISO `dates` and aligned `amounts`
/// (invested negative, received positive).
#[pyfunction]
fn mwr(dates: Vec<String>, amounts: Vec<f64>) -> PyResult<f64> {
    if dates.len() != amounts.len() {
        return Err(PyValueError::new_err(
            "dates and amounts must be the same length",
        ));
    }
    let mut cfs = Vec::with_capacity(dates.len());
    for (d, a) in dates.iter().zip(amounts.iter()) {
        cfs.push((parse_iso_date(d)?, *a));
    }
    mwr_core(&cfs).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Allocation weights from market values.
#[pyfunction]
fn allocation(market_values: Vec<f64>) -> PyResult<Vec<f64>> {
    alloc_weights_core(&market_values).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The population covariance matrix of N aligned asset-return series.
#[pyfunction]
fn covariance_matrix(returns: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    cov_matrix_core(&returns).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Portfolio risk (variance, volatility, VaR) from asset returns + weights.
#[pyfunction]
#[pyo3(signature = (returns, weights, periods_per_year=252.0, confidence=0.95))]
fn portfolio_risk<'py>(
    py: Python<'py>,
    returns: Vec<Vec<f64>>,
    weights: Vec<f64>,
    periods_per_year: f64,
    confidence: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let r = portfolio_risk_core(&returns, &weights, periods_per_year, confidence)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d = PyDict::new(py);
    d.set_item("variance", r.variance)?;
    d.set_item("volatility", r.volatility)?;
    d.set_item("annualized_volatility", r.annualized_volatility)?;
    d.set_item("historical_var", r.historical_var)?;
    d.set_item("parametric_var", r.parametric_var)?;
    d.set_item("confidence", r.confidence)?;
    d.set_item("periods_per_year", r.periods_per_year)?;
    Ok(d)
}

/// Markowitz optimization: min-variance, tangency, and (if `target` is given)
/// efficient-frontier weights. Unconstrained / closed-form (shorting allowed).
#[pyfunction]
#[pyo3(signature = (mean, cov, rf=0.0, target=None))]
fn optimize<'py>(
    py: Python<'py>,
    mean: Vec<f64>,
    cov: Vec<Vec<f64>>,
    rf: f64,
    target: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let mv = min_var_core(&cov).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let tan = tangency_core(&cov, &mean, rf).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let frontier = match target {
        Some(t) => Some(
            efficient_frontier_point(&cov, &mean, t)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        ),
        None => None,
    };
    let d = PyDict::new(py);
    d.set_item("min_variance_weights", mv)?;
    d.set_item("tangency_weights", tan)?;
    d.set_item("frontier_weights", frontier)?;
    Ok(d)
}

/// Train a differential-ML surrogate for a European option and return its price
/// and delta alongside the Black-Scholes baseline. Training is deterministic given
/// `seed`. `hidden` defaults to two layers of 30 units.
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, vol, maturity, option_type="call",
    samples=4096, epochs=60, spread=2.0, seed=1, hidden=None))]
#[allow(clippy::too_many_arguments)]
fn differential_ml<'py>(
    py: Python<'py>,
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    maturity: f64,
    option_type: &str,
    samples: usize,
    epochs: usize,
    spread: f64,
    seed: u64,
    hidden: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyDict>> {
    let option_type = parse_option_type(option_type)?;
    let cfg = TrainConfig {
        spec: BsSpec {
            spot,
            strike,
            rate,
            vol,
            maturity,
            option_type,
        },
        n_samples: samples,
        hidden: hidden.unwrap_or_else(|| vec![30, 30]),
        epochs,
        spread,
        seed,
    };
    let r = differential_ml_price(&cfg).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d = PyDict::new(py);
    d.set_item("spot", r.spot)?;
    d.set_item("option_type", r.option_type)?;
    d.set_item("ml_price", r.ml_price)?;
    d.set_item("ml_delta", r.ml_delta)?;
    d.set_item("bs_price", r.bs_price)?;
    d.set_item("bs_delta", r.bs_delta)?;
    d.set_item("price_abs_err", r.price_abs_err)?;
    d.set_item("delta_abs_err", r.delta_abs_err)?;
    d.set_item("n_samples", r.n_samples)?;
    d.set_item("epochs", r.epochs)?;
    d.set_item("final_loss", r.final_loss)?;
    Ok(d)
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
    m.add_function(wrap_pyfunction!(barrier_price, m)?)?;
    m.add_function(wrap_pyfunction!(lookback_price, m)?)?;
    m.add_function(wrap_pyfunction!(asian_price, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_process, m)?)?;
    m.add_function(wrap_pyfunction!(stats, m)?)?;
    m.add_function(wrap_pyfunction!(sharpe, m)?)?;
    m.add_function(wrap_pyfunction!(sortino, m)?)?;
    m.add_function(wrap_pyfunction!(max_drawdown, m)?)?;
    m.add_function(wrap_pyfunction!(value_at_risk, m)?)?;
    m.add_function(wrap_pyfunction!(expected_shortfall, m)?)?;
    m.add_function(wrap_pyfunction!(beta, m)?)?;
    m.add_function(wrap_pyfunction!(tracking_error, m)?)?;
    m.add_function(wrap_pyfunction!(info_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(jarque_bera, m)?)?;
    m.add_function(wrap_pyfunction!(acf, m)?)?;
    m.add_function(wrap_pyfunction!(portfolio_value, m)?)?;
    m.add_function(wrap_pyfunction!(twr, m)?)?;
    m.add_function(wrap_pyfunction!(mwr, m)?)?;
    m.add_function(wrap_pyfunction!(allocation, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(portfolio_risk, m)?)?;
    m.add_function(wrap_pyfunction!(optimize, m)?)?;
    m.add_function(wrap_pyfunction!(differential_ml, m)?)?;
    m.add_class::<YieldCurve>()?;
    m.add_class::<FixedRateBond>()?;
    Ok(())
}
