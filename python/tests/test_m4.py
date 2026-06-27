"""Tests for the M4 Python binding: the FixedRateBond.

Held to the same tight bar as the Rust validation suite: price, yield, duration,
convexity, and curve-discounted price must match QuantLib to ~1e-8 (deterministic
closed-form bond math), plus a par-bond closed-form check.
"""

import json
import os

import pytest

import oxis

REF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "validation", "reference")
TOL = 1e-8


def _load(name):
    with open(os.path.join(REF_DIR, name)) as f:
        return json.load(f)


def test_bonds_match_quantlib():
    data = _load("bonds.json")
    assert data["cases"]
    for case in data["cases"]:
        bond = oxis.FixedRateBond.from_cashflows(
            case["cashflow_times"],
            case["cashflow_amounts"],
            case["frequency"],
            accrued=case["accrued"],
            face=case["face"],
            coupon_rate=case["coupon_rate"],
        )
        y = case["test_yield"]
        assert abs(bond.dirty_price_from_yield(y) - case["dirty_price"]) <= TOL, case
        assert abs(bond.clean_price_from_yield(y) - case["clean_price"]) <= TOL, case
        assert abs(bond.yield_to_maturity(case["clean_price"]) - case["yield_roundtrip"]) <= TOL, case
        assert abs(bond.macaulay_duration(y) - case["macaulay_duration"]) <= TOL, case
        assert abs(bond.modified_duration(y) - case["modified_duration"]) <= TOL, case
        assert abs(bond.convexity(y) - case["convexity"]) <= TOL, case

        curve = oxis.YieldCurve.flat(case["flat_rate"])
        dirty, _clean = bond.price_from_curve(curve)
        assert abs(dirty - case["curve_dirty_price"]) <= TOL, case


def test_par_bond_closed_form():
    bond = oxis.FixedRateBond.regular(face=100.0, coupon_rate=0.05, frequency=2, n_periods=10)
    assert abs(bond.clean_price_from_yield(0.05) - 100.0) <= 1e-9
    assert abs(bond.yield_to_maturity(100.0) - 0.05) <= 1e-9
    assert bond.modified_duration(0.05) < 5.0  # below the 5y maturity


def test_invalid_bond_raises():
    with pytest.raises(ValueError):
        oxis.FixedRateBond.regular(face=100.0, coupon_rate=0.05, frequency=0, n_periods=10)
