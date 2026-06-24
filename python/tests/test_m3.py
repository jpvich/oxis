"""Tests for the M3 Python binding: the YieldCurve term structure.

The curve is held to the same tight bar as the Rust validation suite: every
interpolation scheme must reproduce QuantLib's discount factor, zero rate, and
forward rate to ~1e-10 (these are deterministic closed-form interpolations, not
stochastic estimates), plus a flat-curve closed-form check.
"""

import json
import math
import os

import pytest

import oxis

REF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "validation", "reference")
TOL = 1e-10


def _load(name):
    with open(os.path.join(REF_DIR, name)) as f:
        return json.load(f)


def test_yield_curve_matches_quantlib():
    data = _load("yield_curve.json")
    assert data["cases"], "no yield-curve cases loaded"
    for case in data["cases"]:
        curve = oxis.YieldCurve.from_zero_rates(
            case["pillar_times"], case["pillar_rates"], interp=case["interpolation"]
        )
        assert abs(curve.discount(case["t"]) - case["discount"]) <= TOL, case
        assert abs(curve.zero_rate(case["t"]) - case["zero_rate"]) <= TOL, case
        if case["forward_t2"] is not None:
            f = curve.forward_rate(case["t"], case["forward_t2"])
            assert abs(f - case["forward_rate"]) <= TOL, case


def test_flat_curve_closed_form():
    c = oxis.YieldCurve.flat(0.03)
    assert abs(c.discount(2.0) - math.exp(-0.06)) <= 1e-15
    assert abs(c.zero_rate(5.0) - 0.03) <= 1e-12
    assert abs(c.forward_rate(1.0, 3.0) - 0.03) <= 1e-12


def test_discount_at_zero_is_one():
    c = oxis.YieldCurve.from_zero_rates([0.0, 1.0, 2.0], [0.02, 0.025, 0.03])
    assert c.discount(0.0) == 1.0


def test_out_of_range_raises():
    c = oxis.YieldCurve.from_zero_rates([1.0, 2.0, 3.0], [0.02, 0.025, 0.03])
    with pytest.raises(ValueError):
        c.discount(5.0)  # beyond the last pillar, no extrapolation


def test_invalid_interp_raises():
    with pytest.raises(ValueError, match="interp must be"):
        oxis.YieldCurve.from_zero_rates([1.0, 2.0], [0.02, 0.03], interp="quadratic")
