"""Tests for the M6 Python bindings: oxis-stats (Ring 3).

Held to the same bar as the Rust validation suite: descriptive, risk,
performance, and relational statistics must match the numpy/scipy/pandas oracle
to ~1e-10 (with a looser band for the Jarque-Bera p-value). The reference is the
same JSON file the Rust test uses.
"""

import json
import os

import pytest

import oxis

REF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "validation", "reference")

# Scalar metric keys that appear in both a case and the stats() dict.
METRIC_KEYS = [
    "mean", "variance", "std_dev", "skewness", "excess_kurtosis",
    "jarque_bera", "cumulative_return", "annualized_return",
    "annualized_volatility", "sharpe", "sortino", "historical_var",
    "historical_es", "parametric_var", "parametric_es", "cornish_fisher_var",
    "max_drawdown", "calmar", "covariance", "correlation", "beta",
    "tracking_error", "information_ratio",
]


def _load(name):
    with open(os.path.join(REF_DIR, name)) as f:
        return json.load(f)


def _kwargs(case):
    kw = dict(
        risk_free=case["risk_free"],
        periods_per_year=case["periods_per_year"],
        confidence=case["confidence"],
    )
    for k in ("returns", "prices", "values", "benchmark"):
        if k in case:
            kw[k] = case[k]
    return kw


def test_stats_match_numpy_scipy():
    data = _load("stats.json")
    assert data["oracle"] == "numpy/scipy/pandas"
    tol = data["tolerance"]
    ptol = data["pvalue_tolerance"]
    assert data["cases"]

    for case in data["cases"]:
        got = oxis.stats(**_kwargs(case))
        for key in METRIC_KEYS:
            if case.get(key) is not None:
                assert abs(got[key] - case[key]) <= tol, (case["name"], key)
        if case.get("jarque_bera_pvalue") is not None:
            assert abs(got["jarque_bera_pvalue"] - case["jarque_bera_pvalue"]) <= ptol, case["name"]
        if case.get("max_drawdown_duration") is not None:
            assert got["max_drawdown_duration"] == case["max_drawdown_duration"], case["name"]
        if case.get("acf") is not None:
            f = oxis.acf(case["returns"], max(case["acf_lags"]))
            for lag, exp in zip(case["acf_lags"], case["acf"]):
                assert abs(f[lag] - exp) <= tol, (case["name"], "acf", lag)


def test_helper_functions_agree_with_report():
    data = _load("stats.json")
    case = next(c for c in data["cases"] if c["name"] == "returns_c95")
    r = case["returns"]
    rf, ppy, c = case["risk_free"], case["periods_per_year"], case["confidence"]
    tol = data["tolerance"]

    assert abs(oxis.sharpe(r, rf, ppy) - case["sharpe"]) <= tol
    assert abs(oxis.sortino(r, rf, ppy) - case["sortino"]) <= tol
    assert abs(oxis.value_at_risk(r, c, "historical") - case["historical_var"]) <= tol
    assert abs(oxis.value_at_risk(r, c, "parametric") - case["parametric_var"]) <= tol
    assert abs(oxis.value_at_risk(r, c, "cornish-fisher") - case["cornish_fisher_var"]) <= tol
    assert abs(oxis.expected_shortfall(r, c, "historical") - case["historical_es"]) <= tol
    assert abs(oxis.expected_shortfall(r, c, "parametric") - case["parametric_es"]) <= tol
    stat, pval = oxis.jarque_bera(r)
    assert abs(stat - case["jarque_bera"]) <= tol


def test_relational_helpers():
    data = _load("stats.json")
    case = next(c for c in data["cases"] if c["name"] == "port_vs_bench")
    port, bench = case["returns"], case["benchmark"]
    ppy = case["periods_per_year"]
    tol = data["tolerance"]

    assert abs(oxis.beta(port, bench) - case["beta"]) <= tol
    assert abs(oxis.tracking_error(port, bench, ppy) - case["tracking_error"]) <= tol
    assert abs(oxis.info_ratio(port, bench, ppy) - case["information_ratio"]) <= tol


def test_max_drawdown_helper():
    data = _load("stats.json")
    case = next(c for c in data["cases"] if c["name"] == "prices")
    dd = oxis.max_drawdown(case["prices"])
    assert abs(dd["max_drawdown"] - case["max_drawdown"]) <= data["tolerance"]
    assert dd["duration"] == case["max_drawdown_duration"]


def test_values_kind_suppresses_financial_metrics():
    got = oxis.stats(values=[1.0, 2.0, 3.0, 4.0, 5.0])
    assert got["sharpe"] is None
    assert got["historical_var"] is None
    assert got["mean"] == 3.0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        oxis.stats()  # no primary input
    with pytest.raises(ValueError):
        oxis.stats(returns=[0.01], values=[0.02])  # two inputs
    with pytest.raises(ValueError):
        oxis.value_at_risk([0.01, -0.02], 0.95, "sideways")  # bad method
    with pytest.raises(ValueError):
        oxis.sortino([0.01, 0.02, 0.03], 0.0, 252.0)  # zero downside
