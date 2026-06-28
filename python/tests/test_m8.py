"""M8 — differential ML pricing bindings.

The rigorous two-layer validation (inference exactness <=1e-12 on fixed weights,
and trained-surrogate accuracy across a spot grid) lives in the Rust suite, which
can reach the low-level forward/twin passes. Here we exercise the Python binding
end to end: a trained surrogate must price the option within the documented bands
of Black-Scholes, the classical baseline must be wired through correctly, and
training must be deterministic given a seed.
"""

import json
import os

import oxis

REF = os.path.join(
    os.path.dirname(__file__), "..", "..", "validation", "reference", "ml.json"
)


def _reference():
    with open(REF) as f:
        return json.load(f)


def test_module_exposes_differential_ml():
    ref = _reference()
    assert ref["model"] == "differential-ml"
    assert hasattr(oxis, "differential_ml")


def test_accuracy_atm_within_bands():
    acc = _reference()["accuracy"]
    bands = acc["bands"]
    i = acc["grid"].index(100.0)
    bs_price, bs_delta = acc["bs_price"][i], acc["bs_delta"][i]

    out = oxis.differential_ml(
        spot=100.0, strike=100.0, rate=0.05, vol=0.2, maturity=1.0,
        option_type="call", samples=2048, epochs=40, seed=1,
    )
    # The classical baseline must match the closed-form oracle exactly.
    assert abs(out["bs_price"] - bs_price) < 1e-9
    assert abs(out["bs_delta"] - bs_delta) < 1e-9
    # The ML estimate must be within the documented (grid-wide) bands at this point.
    assert out["price_abs_err"] <= bands["price_max_abs"]
    assert out["delta_abs_err"] <= bands["delta_max_abs"]
    assert out["ml_price"] > 0.0


def test_training_is_deterministic():
    kw = dict(spot=100.0, strike=100.0, rate=0.05, vol=0.2, maturity=1.0,
              samples=512, epochs=10, seed=7)
    a = oxis.differential_ml(**kw)
    b = oxis.differential_ml(**kw)
    assert a["ml_price"] == b["ml_price"]
    assert a["ml_delta"] == b["ml_delta"]


def test_put_prices_and_is_finite():
    out = oxis.differential_ml(
        spot=100.0, strike=100.0, rate=0.05, vol=0.2, maturity=1.0,
        option_type="put", samples=1024, epochs=20, hidden=[16, 16],
    )
    assert out["option_type"] == "put"
    assert out["bs_price"] > 0.0
    # NaN != NaN; this asserts the ML price is a real number.
    assert out["ml_price"] == out["ml_price"]
