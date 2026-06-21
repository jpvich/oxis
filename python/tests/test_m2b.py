"""Tests for the M2b Python bindings: Monte Carlo (European) and Longstaff-
Schwartz (American).

The stochastic engines are held to the same statistical bar as the Rust
validation suite: European MC must land within a few standard errors of the
Black-Scholes closed form, and American LSM within a combined standard error of
QuantLib's own LSM engine (and a small bias band of the binomial price).
"""

import json
import math
import os

import pytest

import oxis

REF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "validation", "reference")


def _load(name):
    with open(os.path.join(REF_DIR, name)) as f:
        return json.load(f)


def test_monte_carlo_european_matches_black_scholes():
    """European MC within 4 standard errors of the exact (QuantLib BS) price."""
    data = _load("black_scholes.json")
    for case in data["cases"]:
        est = oxis.monte_carlo(
            spot=case["spot"],
            strike=case["strike"],
            rate=case["rate"],
            vol=case["volatility"],
            t=case["time"],
            option_type=case["option_type"],
            paths=1_000_000,
            seed=20240101,
            dividend_yield=case["dividend_yield"],
        )
        err = abs(est["price"] - case["price"])
        assert err <= 4.0 * est["standard_error"] + 1e-9, (err, est, case)


def test_monte_carlo_is_deterministic():
    a = oxis.monte_carlo(spot=100, strike=100, rate=0.05, vol=0.2, t=1.0,
                         option_type="call", paths=100_000, seed=7)
    b = oxis.monte_carlo(spot=100, strike=100, rate=0.05, vol=0.2, t=1.0,
                         option_type="call", paths=100_000, seed=7)
    assert a["price"] == b["price"]
    assert a["standard_error"] == b["standard_error"]


def test_lsm_american_matches_quantlib():
    """American LSM within 5 combined standard errors of QuantLib's LSM engine.

    A small absolute floor covers the deep-in-the-money cases where immediate
    exercise is optimal (OXIS returns exact intrinsic, SE -> 0).
    """
    data = _load("monte_carlo_american.json")
    for case in data["cases"]:
        est = oxis.lsm(
            spot=case["spot"],
            strike=case["strike"],
            rate=case["rate"],
            vol=case["volatility"],
            t=case["time"],
            option_type=case["option_type"],
            paths=100_000,
            steps=case["steps"],
            seed=20240102,
            dividend_yield=case["dividend_yield"],
        )
        combined_se = math.hypot(est["standard_error"], case["error_estimate"])
        assert abs(est["price"] - case["price"]) <= 5.0 * combined_se + 0.05, (est, case)


def test_lsm_american_put_has_early_exercise_premium():
    """An ITM American put is worth at least its European counterpart."""
    args = dict(spot=100, strike=110, rate=0.05, vol=0.3, t=1.0, option_type="put")
    amer = oxis.lsm(paths=100_000, steps=50, seed=1, **args)
    euro = oxis.monte_carlo(paths=100_000, seed=1, **args)
    # Allow a small statistical margin on the (independent) European estimate.
    assert amer["price"] >= euro["price"] - 4.0 * euro["standard_error"]


def test_invalid_option_type_raises():
    with pytest.raises(ValueError, match="option_type must be"):
        oxis.monte_carlo(spot=100, strike=100, rate=0.05, vol=0.2, t=1.0,
                         option_type="straddle")
