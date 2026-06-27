"""Tests for the M5 Python bindings: exotic options + stochastic processes.

Held to the same bar as the Rust validation suites: the closed-form exotics
(barrier, lookback, geometric Asian) must match QuantLib to ~1e-8, the Monte
Carlo ones (arithmetic Asian) within a combined standard-error band, and the
process simulator's terminal moments within a few standard errors of the
closed form. References are the same JSON files the Rust tests use.
"""

import json
import math
import os

import pytest

import oxis

REF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "validation", "reference")
TOL = 1e-8


def _load(name):
    with open(os.path.join(REF_DIR, name)) as f:
        return json.load(f)


def test_barriers_match_quantlib():
    data = _load("barrier.json")
    assert data["cases"]
    for case in data["cases"]:
        got = oxis.barrier_price(
            spot=case["spot"], strike=case["strike"], rate=case["rate"],
            vol=case["vol"], t=case["t"], option_type=case["type"],
            barrier_type=case["barrier_type"], barrier=case["barrier"],
            dividend_yield=case["div"],
        )
        assert abs(got - case["price"]) <= TOL, case


def test_in_out_barrier_parity():
    # in + out = vanilla (zero-rebate European barrier).
    common = dict(spot=100, strike=100, rate=0.05, vol=0.25, t=1.0, option_type="call")
    vanilla = oxis.black_scholes(spot=100, strike=100, rate=0.05, vol=0.25, t=1.0, option_type="call")
    din = oxis.barrier_price(barrier_type="down-in", barrier=90, **common)
    dout = oxis.barrier_price(barrier_type="down-out", barrier=90, **common)
    assert abs(din + dout - vanilla) < 1e-10


def test_lookbacks_match_quantlib():
    data = _load("lookback.json")
    assert data["cases"]
    for case in data["cases"]:
        got = oxis.lookback_price(
            spot=case["spot"], strike=case["strike"], rate=case["rate"],
            vol=case["vol"], t=case["t"], option_type=case["type"],
            strike_type=case["strike_type"], dividend_yield=case["div"],
        )
        assert abs(got - case["price"]) <= TOL, case


def test_asians_match_quantlib():
    data = _load("asian.json")
    assert data["cases"]
    for case in data["cases"]:
        if case["average"] == "geometric":
            got = oxis.asian_price(
                spot=case["spot"], strike=case["strike"], rate=case["rate"],
                vol=case["vol"], t=case["t"], option_type=case["type"],
                average="geometric", dividend_yield=case["div"],
            )
            assert got["standard_error"] is None
            assert abs(got["price"] - case["price"]) <= TOL, case
        else:
            got = oxis.asian_price(
                spot=case["spot"], strike=case["strike"], rate=case["rate"],
                vol=case["vol"], t=case["t"], option_type=case["type"],
                average="arithmetic", n_fixings=case["n_fixings"],
                paths=case["paths"], seed=case["seed"], dividend_yield=case["div"],
            )
            se = got["standard_error"]
            combined = math.sqrt(se * se + case["ql_error"] ** 2)
            assert abs(got["price"] - case["price"]) <= 4.0 * combined + 1e-9, case


def test_simulate_process_moments_match_reference():
    data = _load("processes.json")
    assert data["cases"]
    for case in data["cases"]:
        params = {k: case[k] for k in
                  ("mu", "sigma", "kappa", "theta", "jump_mean", "jump_std", "v0", "xi", "rho")
                  if case.get(k) is not None}
        if case.get("lambda") is not None:
            params["lambda_"] = case["lambda"]
        got = oxis.simulate_process(
            case["process"], x0=case["x0"], t=case["t"], steps=case["steps"],
            paths=case["paths"], seed=case["seed"], **params,
        )
        mean_band = 5.0 * got["mean_std_error"] + 0.01 * abs(case["mean"]) + 1e-9
        assert abs(got["sample_mean"] - case["mean"]) <= mean_band, case
        if case["var"] is not None:
            ref_std = math.sqrt(case["var"])
            assert abs(got["sample_std"] - ref_std) <= case["std_rel_tol"] * ref_std + 1e-9, case


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        oxis.barrier_price(spot=100, strike=100, rate=0.05, vol=0.25, t=1.0,
                           option_type="call", barrier_type="sideways", barrier=90)
    with pytest.raises(ValueError):
        oxis.simulate_process("gbm", x0=-1.0)
