"""Tests for the M7 Python bindings: oxis-portfolio (Ring 3).

Held to the same bar as the Rust validation suite: valuation, performance (TWR /
MWR), allocation, risk aggregation, and Markowitz optimization must match the
numpy/scipy oracle to ~1e-10 (IRR ~1e-9). Reference is the same JSON the Rust
test uses.
"""

import json
import os

import pytest

import oxis

REF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "validation", "reference")


def _load(name):
    with open(os.path.join(REF_DIR, name)) as f:
        return json.load(f)


def _case(data, name):
    return next(c for c in data["cases"] if c["name"] == name)


def test_portfolio_matches_numpy_scipy():
    data = _load("portfolio.json")
    assert data["oracle"] == "numpy/scipy/pandas"
    tol = data["tolerance"]
    irr_tol = data["irr_tolerance"]

    # Valuation.
    v_case = _case(data, "valuation")
    holdings = [(s, q, c, p) for (s, q, c, p) in v_case["holdings"]]
    v = oxis.portfolio_value(holdings)
    assert abs(v["total_market_value"] - v_case["total_market_value"]) <= tol
    assert abs(v["total_unrealized_pnl"] - v_case["total_unrealized_pnl"]) <= tol
    for got, mv, pnl, w in zip(
        v["holdings"], v_case["market_values"], v_case["unrealized_pnls"], v_case["weights"]
    ):
        assert abs(got["market_value"] - mv) <= tol
        assert abs(got["unrealized_pnl"] - pnl) <= tol
        assert abs(got["weight"] - w) <= tol

    # TWR.
    t_case = _case(data, "twr")
    assert abs(oxis.twr(t_case["values"], t_case["flows"]) - t_case["twr"]) <= tol

    # MWR / IRR.
    m_case = _case(data, "mwr")
    assert abs(oxis.mwr(m_case["dates"], m_case["amounts"]) - m_case["mwr"]) <= irr_tol

    # Allocation.
    a_case = _case(data, "allocation")
    for got, exp in zip(oxis.allocation(a_case["market_values"]), a_case["weights"]):
        assert abs(got - exp) <= tol

    # Risk.
    r_case = _case(data, "risk")
    r = oxis.portfolio_risk(
        r_case["returns"], r_case["weights"],
        periods_per_year=r_case["periods_per_year"], confidence=r_case["confidence"],
    )
    for key in ("variance", "volatility", "annualized_volatility", "historical_var", "parametric_var"):
        assert abs(r[key] - r_case[key]) <= tol, key

    # Optimize.
    o_case = _case(data, "optimize")
    opt = oxis.optimize(o_case["mean"], o_case["cov"], rf=o_case["rf"], target=o_case["target"])
    for got, exp in zip(opt["min_variance_weights"], o_case["min_variance_weights"]):
        assert abs(got - exp) <= tol
    for got, exp in zip(opt["tangency_weights"], o_case["tangency_weights"]):
        assert abs(got - exp) <= tol
    for got, exp in zip(opt["frontier_weights"], o_case["frontier_weights"]):
        assert abs(got - exp) <= tol


def test_covariance_matrix_symmetric():
    data = _load("portfolio.json")
    r_case = _case(data, "risk")
    cov = oxis.covariance_matrix(r_case["returns"])
    n = len(cov)
    for i in range(n):
        for j in range(n):
            assert abs(cov[i][j] - cov[j][i]) <= 1e-15


def test_optimize_without_target_omits_frontier():
    data = _load("portfolio.json")
    o_case = _case(data, "optimize")
    opt = oxis.optimize(o_case["mean"], o_case["cov"])
    assert opt["frontier_weights"] is None
    assert opt["min_variance_weights"] is not None


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        oxis.portfolio_value([])  # no holdings
    with pytest.raises(ValueError):
        oxis.twr([100.0], [])  # too few values
    with pytest.raises(ValueError):
        oxis.mwr(["2024-01-01", "2025-01-01"], [100.0, 100.0])  # no sign change → no IRR
    with pytest.raises(ValueError):
        oxis.optimize([0.1, 0.2], [[1.0, 1.0], [1.0, 1.0]])  # singular covariance
