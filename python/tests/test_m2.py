"""Tests for the M2a Python bindings: binomial, greeks, implied volatility.

Each cross-checks the Python path against the same QuantLib reference data the
Rust validation suite uses, so the fourth OXIS interface is held to the same bar.
"""

import json
import os

import pytest

import oxis

REF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "validation", "reference")


def _load(name):
    with open(os.path.join(REF_DIR, name)) as f:
        return json.load(f)


def test_binomial_matches_quantlib():
    data = _load("binomial.json")
    tol = data["tolerance"]
    for case in data["cases"]:
        got = oxis.binomial(
            spot=case["spot"],
            strike=case["strike"],
            rate=case["rate"],
            vol=case["volatility"],
            t=case["time"],
            option_type=case["option_type"],
            style=case["style"],
            steps=case["steps"],
            dividend_yield=case["dividend_yield"],
        )
        assert got == pytest.approx(case["price"], abs=tol), case


def test_greeks_match_quantlib():
    data = _load("greeks.json")
    tol = data["tolerance"]
    for case in data["cases"]:
        g = oxis.greeks(
            spot=case["spot"],
            strike=case["strike"],
            rate=case["rate"],
            vol=case["volatility"],
            t=case["time"],
            option_type=case["option_type"],
            dividend_yield=case["dividend_yield"],
        )
        for greek in ("delta", "gamma", "vega", "theta", "rho"):
            assert g[greek] == pytest.approx(case[greek], abs=tol), (greek, case)


def test_implied_vol_matches_quantlib():
    data = _load("implied_vol.json")
    tol = data["tolerance"]
    for case in data["cases"]:
        iv = oxis.implied_volatility(
            market_price=case["market_price"],
            spot=case["spot"],
            strike=case["strike"],
            rate=case["rate"],
            t=case["time"],
            option_type=case["option_type"],
            dividend_yield=case["dividend_yield"],
        )
        assert iv == pytest.approx(case["implied_volatility"], abs=tol), case


def test_american_put_exceeds_european():
    args = dict(spot=100, strike=110, rate=0.05, vol=0.3, t=1.0, option_type="put")
    euro = oxis.binomial(style="european", steps=500, **args)
    amer = oxis.binomial(style="american", steps=500, **args)
    assert amer >= euro


def test_invalid_style_raises():
    with pytest.raises(ValueError, match="style must be"):
        oxis.binomial(
            spot=100, strike=100, rate=0.05, vol=0.2, t=1.0,
            option_type="call", style="bermudan",
        )
