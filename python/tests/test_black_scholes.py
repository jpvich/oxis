"""Tests for the `oxis` Python bindings.

These verify the binding surface and, crucially, cross-check the Python path
against the same QuantLib reference data the Rust validation suite uses — so the
fourth OXIS interface (Python) is held to the same correctness bar as the others.
"""

import json
import os

import pytest

import oxis

REFERENCE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "validation",
    "reference",
    "black_scholes.json",
)


def test_textbook_values():
    call = oxis.black_scholes(
        spot=100, strike=100, rate=0.05, vol=0.2, t=1.0, option_type="call"
    )
    put = oxis.black_scholes(
        spot=100, strike=100, rate=0.05, vol=0.2, t=1.0, option_type="put"
    )
    assert call == pytest.approx(10.450583572, abs=1e-6)
    assert put == pytest.approx(5.573526022, abs=1e-6)


def test_matches_quantlib_reference():
    with open(REFERENCE) as f:
        data = json.load(f)
    tol = data["tolerance"]
    for case in data["cases"]:
        got = oxis.black_scholes(
            spot=case["spot"],
            strike=case["strike"],
            rate=case["rate"],
            vol=case["volatility"],
            t=case["time"],
            option_type=case["option_type"],
            dividend_yield=case["dividend_yield"],
        )
        assert got == pytest.approx(case["price"], abs=tol), case


def test_price_dict_shape():
    result = oxis.price(
        spot=100, strike=105, rate=0.05, vol=0.2, t=1.0, option_type="call"
    )
    assert result["model"] == "black-scholes"
    assert result["option_type"] == "call"
    assert result["exercise"] == "european"
    assert result["price"] == pytest.approx(8.021352235143176, abs=1e-9)


def test_invalid_volatility_raises():
    with pytest.raises(ValueError, match="volatility must be >= 0"):
        oxis.black_scholes(
            spot=100, strike=100, rate=0.05, vol=-0.2, t=1.0, option_type="call"
        )


def test_invalid_option_type_raises():
    with pytest.raises(ValueError, match="option_type must be"):
        oxis.black_scholes(
            spot=100, strike=100, rate=0.05, vol=0.2, t=1.0, option_type="banana"
        )
