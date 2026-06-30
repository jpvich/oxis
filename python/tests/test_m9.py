"""M9 — neural American pricing bindings (Deep LSM).

The rigorous two-layer validation (inference value <=1e-12 on fixed weights, and
trained-estimate accuracy within a band of a QuantLib CRR tree) lives in the Rust
suite, which can reach the low-level forward pass. Here we exercise the Python
binding end to end: a trained Deep-LSM estimate must price the American put within
the documented band of the binomial baseline, the baseline must be wired through
correctly, and training must be deterministic given a seed.
"""

import json
import os

import oxis

REF = os.path.join(
    os.path.dirname(__file__), "..", "..", "validation", "reference", "deep_lsm.json"
)


def _reference():
    with open(REF) as f:
        return json.load(f)


def test_module_exposes_american_ml():
    ref = _reference()
    assert ref["model"] == "deep-lsm"
    assert hasattr(oxis, "american_ml")


def test_deep_lsm_within_band_vs_binomial():
    acc = _reference()["accuracy"]
    spec, train, bands = acc["spec"], acc["train"], acc["bands"]
    i = acc["grid"].index(100.0)
    binomial_ref = acc["binomial_price"][i]

    out = oxis.american_ml(
        spot=100.0, strike=spec["strike"], rate=spec["rate"], vol=spec["vol"],
        maturity=spec["maturity"], option_type=spec["option_type"], method="deep-lsm",
        paths=train["paths"], steps=train["steps"], epochs=train["epochs"],
        seed=train["seed"], hidden=train["hidden"],
    )
    # The binomial baseline must match the independent QuantLib oracle closely
    # (both are CRR American trees: OXIS at 2000 steps, QuantLib at 2000 steps).
    assert abs(out["binomial_price"] - binomial_ref) < 0.05
    # The Deep-LSM estimate must fall within the documented band.
    budget = bands["se_mult"] * out["standard_error"] + bands["abs"]
    assert out["abs_err"] <= budget
    assert out["ml_price"] > 0.0
    assert out["method"] == "deep-lsm"


def test_deep_lsm_deterministic():
    kw = dict(spot=100.0, strike=100.0, rate=0.05, vol=0.3, maturity=1.0,
              method="deep-lsm", paths=2048, steps=8, epochs=10, seed=7, hidden=[12])
    a = oxis.american_ml(**kw)
    b = oxis.american_ml(**kw)
    assert a["ml_price"] == b["ml_price"]
    assert a["standard_error"] == b["standard_error"]


def test_deep_itm_put_intrinsic():
    # A deep-in-the-money American put exercises immediately at intrinsic, with no
    # Monte-Carlo uncertainty.
    out = oxis.american_ml(
        spot=100.0, strike=1000.0, rate=0.05, vol=0.2, maturity=1.0,
        option_type="put", method="deep-lsm", paths=2048, steps=8, epochs=10,
    )
    assert abs(out["ml_price"] - 900.0) < 1e-9
    assert out["standard_error"] == 0.0


def _dos_ref():
    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "validation", "reference", "dos.json"
    )
    with open(path) as f:
        return json.load(f)


def test_dos_within_band_vs_binomial():
    acc = _dos_ref()["accuracy"]
    spec, train, bands = acc["spec"], acc["train"], acc["bands"]
    out = oxis.american_ml(
        spot=100.0, strike=spec["strike"], rate=spec["rate"], vol=spec["vol"],
        maturity=spec["maturity"], option_type=spec["option_type"], method="dos",
        paths=train["paths"], steps=train["steps"], epochs=train["epochs"],
        seed=train["seed"], hidden=train["hidden"],
    )
    budget = bands["se_mult"] * out["standard_error"] + bands["abs"]
    assert out["abs_err"] <= budget
    assert out["method"] == "dos"
    assert out["ml_price"] > 0.0


def test_dos_deterministic():
    kw = dict(spot=100.0, strike=100.0, rate=0.05, vol=0.3, maturity=1.0,
              method="dos", paths=2048, steps=8, epochs=10, seed=7, hidden=[12])
    a = oxis.american_ml(**kw)
    b = oxis.american_ml(**kw)
    assert a["ml_price"] == b["ml_price"]
    assert a["standard_error"] == b["standard_error"]


def test_methods_agree():
    # Deep LSM and DOS must agree with the binomial baseline within a loose band.
    kw = dict(spot=100.0, strike=100.0, rate=0.05, vol=0.3, maturity=1.0,
              option_type="put", paths=4096, steps=10, epochs=20, seed=11, hidden=[16])
    deep = oxis.american_ml(method="deep-lsm", **kw)
    dos = oxis.american_ml(method="dos", **kw)
    assert deep["binomial_price"] == dos["binomial_price"]
    for out in (deep, dos):
        budget = 5.0 * out["standard_error"] + 0.60
        assert out["abs_err"] <= budget
