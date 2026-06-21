#!/usr/bin/env python3
"""Generate QuantLib reference prices for the OXIS validation suite.

This is the *oracle* side of OXIS's core promise: every pricing model is
cross-checked against QuantLib (or a documented closed form) within a tight
tolerance. This script is **not** a runtime dependency of OXIS — it runs
offline (CI or a maintainer's machine) to (re)generate `reference/*.json`,
which the Rust validation tests then assert against.

Black-Scholes-Merton, European, analytic engine. We let QuantLib pick the
exact year-fraction `t` (via Actual/365 Fixed between dates) and record *that*
`t` in the output, so the Rust side prices at the identical continuous time and
the comparison is apples-to-apples rather than fighting date arithmetic.

Usage:
    pip install -r requirements.txt
    python generate_reference.py            # writes reference/black_scholes.json
"""

from __future__ import annotations

import json
import os

import QuantLib as ql

# Fixed evaluation date for reproducibility (do NOT use "today").
EVAL_DATE = ql.Date(15, 6, 2026)
DAY_COUNT = ql.Actual365Fixed()
CALENDAR = ql.NullCalendar()

# Reference cases. `days` is the calendar tenor; QuantLib derives the exact
# `t` we record. Chosen to span OTM / ATM / ITM, both option types, a range of
# vols, rates, dividend yields, and maturities.
CASES = [
    # spot, strike, rate,  vol,  div,  days,  type
    (100.0, 100.0, 0.05, 0.20, 0.00, 365, "call"),
    (100.0, 100.0, 0.05, 0.20, 0.00, 365, "put"),
    (100.0, 105.0, 0.05, 0.20, 0.00, 365, "call"),
    (100.0, 105.0, 0.05, 0.20, 0.00, 365, "put"),
    (100.0, 95.0, 0.05, 0.20, 0.00, 365, "call"),
    (100.0, 95.0, 0.05, 0.20, 0.00, 365, "put"),
    # deep ITM / OTM
    (150.0, 100.0, 0.03, 0.25, 0.00, 365, "call"),
    (60.0, 100.0, 0.03, 0.25, 0.00, 365, "put"),
    (100.0, 200.0, 0.03, 0.25, 0.00, 365, "call"),
    # short and long maturities
    (100.0, 100.0, 0.05, 0.20, 0.00, 30, "call"),
    (100.0, 100.0, 0.05, 0.20, 0.00, 30, "put"),
    (100.0, 100.0, 0.05, 0.20, 0.00, 1825, "call"),
    (100.0, 100.0, 0.05, 0.20, 0.00, 1825, "put"),
    # continuous dividend yield
    (100.0, 100.0, 0.05, 0.20, 0.03, 365, "call"),
    (100.0, 100.0, 0.05, 0.20, 0.03, 365, "put"),
    (120.0, 110.0, 0.03, 0.35, 0.01, 274, "call"),
    (120.0, 110.0, 0.03, 0.35, 0.01, 274, "put"),
    # low and high volatility
    (100.0, 100.0, 0.05, 0.05, 0.00, 365, "call"),
    (100.0, 100.0, 0.05, 0.80, 0.00, 365, "call"),
    (100.0, 100.0, 0.05, 0.80, 0.00, 365, "put"),
    # negative rate (these are real)
    (100.0, 100.0, -0.01, 0.20, 0.00, 365, "call"),
    (100.0, 100.0, -0.01, 0.20, 0.00, 365, "put"),
]


def price_case(spot, strike, rate, vol, div, days, kind):
    """Price one European option with QuantLib's analytic engine.

    Returns (t, price) where `t` is the exact Actual/365F year fraction
    QuantLib used between the evaluation date and the exercise date.
    """
    process, _, t = _build(spot, strike, rate, vol, div, days, kind)
    option, _ = _european_option(strike, days, kind)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return t, option.NPV()


def _exercise_date(days):
    return EVAL_DATE + ql.Period(days, ql.Days)


def _process(spot, rate, vol, div):
    """A Black-Scholes-Merton process with continuously-compounded flat curves
    (discount factor exp(-r·t)), matching the OXIS continuous-time convention."""
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    rate_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(EVAL_DATE, rate, DAY_COUNT, ql.Continuous, ql.Annual)
    )
    div_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(EVAL_DATE, div, DAY_COUNT, ql.Continuous, ql.Annual)
    )
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(EVAL_DATE, CALENDAR, vol, DAY_COUNT)
    )
    return ql.BlackScholesMertonProcess(spot_handle, div_ts, rate_ts, vol_ts)


def _european_option(strike, days, kind):
    option_type = ql.Option.Call if kind == "call" else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(option_type, strike)
    exercise = ql.EuropeanExercise(_exercise_date(days))
    return ql.VanillaOption(payoff, exercise), payoff


def _american_option(strike, days, kind):
    option_type = ql.Option.Call if kind == "call" else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(option_type, strike)
    exercise = ql.AmericanExercise(EVAL_DATE, _exercise_date(days))
    return ql.VanillaOption(payoff, exercise), payoff


def _build(spot, strike, rate, vol, div, days, kind):
    ql.Settings.instance().evaluationDate = EVAL_DATE
    t = DAY_COUNT.yearFraction(EVAL_DATE, _exercise_date(days))
    return _process(spot, rate, vol, div), None, t


def gen_black_scholes():
    """European closed-form prices via AnalyticEuropeanEngine."""
    records = []
    for spot, strike, rate, vol, div, days, kind in CASES:
        t, price = price_case(spot, strike, rate, vol, div, days, kind)
        records.append(
            {
                "spot": spot, "strike": strike, "rate": rate, "volatility": vol,
                "dividend_yield": div, "time": t, "option_type": kind, "price": price,
            }
        )
    return {
        "oracle": "QuantLib", "oracle_version": ql.__version__,
        "model": "black-scholes-merton", "exercise": "european",
        "engine": "AnalyticEuropeanEngine", "day_count": "Actual365Fixed",
        "evaluation_date": str(EVAL_DATE), "tolerance": 1e-10, "cases": records,
    }


# Binomial cases reuse the same parameter grid but record the step count `N`,
# so the Rust side prices its own CRR tree at the identical `N` for an
# apples-to-apples match.
BINOMIAL_STEPS = 256


def gen_binomial():
    """European + American CRR prices via BinomialVanillaEngine at matched steps."""
    records = []
    for spot, strike, rate, vol, div, days, kind in CASES:
        ql.Settings.instance().evaluationDate = EVAL_DATE
        t = DAY_COUNT.yearFraction(EVAL_DATE, _exercise_date(days))
        process = _process(spot, rate, vol, div)
        engine = ql.BinomialVanillaEngine(process, "crr", BINOMIAL_STEPS)
        for style, factory in (("european", _european_option), ("american", _american_option)):
            option, _ = factory(strike, days, kind)
            option.setPricingEngine(engine)
            records.append(
                {
                    "spot": spot, "strike": strike, "rate": rate, "volatility": vol,
                    "dividend_yield": div, "time": t, "option_type": kind,
                    "style": style, "steps": BINOMIAL_STEPS, "price": option.NPV(),
                }
            )
    return {
        "oracle": "QuantLib", "oracle_version": ql.__version__,
        "model": "binomial-crr", "engine": "BinomialVanillaEngine(crr)",
        "day_count": "Actual365Fixed", "evaluation_date": str(EVAL_DATE),
        "tolerance": 1e-6, "cases": records,
    }


def gen_greeks():
    """Analytic Greeks via AnalyticEuropeanEngine. QuantLib conventions: vega and
    rho per unit (not per 1%), theta per year — these match the OXIS conventions."""
    records = []
    for spot, strike, rate, vol, div, days, kind in CASES:
        ql.Settings.instance().evaluationDate = EVAL_DATE
        t = DAY_COUNT.yearFraction(EVAL_DATE, _exercise_date(days))
        process = _process(spot, rate, vol, div)
        option, _ = _european_option(strike, days, kind)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        records.append(
            {
                "spot": spot, "strike": strike, "rate": rate, "volatility": vol,
                "dividend_yield": div, "time": t, "option_type": kind,
                "delta": option.delta(), "gamma": option.gamma(),
                "vega": option.vega(), "theta": option.theta(), "rho": option.rho(),
            }
        )
    return {
        "oracle": "QuantLib", "oracle_version": ql.__version__,
        "model": "black-scholes-greeks", "engine": "AnalyticEuropeanEngine",
        "conventions": "vega,rho per unit; theta per year",
        "day_count": "Actual365Fixed", "evaluation_date": str(EVAL_DATE),
        "tolerance": 1e-8, "cases": records,
    }


def gen_implied_vol():
    """Round-trip: price at a known vol, then have QuantLib recover the implied
    vol. The Rust side recovers from the same target price and must agree."""
    records = []
    for spot, strike, rate, vol, div, days, kind in CASES:
        ql.Settings.instance().evaluationDate = EVAL_DATE
        t = DAY_COUNT.yearFraction(EVAL_DATE, _exercise_date(days))
        process = _process(spot, rate, vol, div)
        option, _ = _european_option(strike, days, kind)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        target = option.NPV()
        try:
            iv = option.impliedVolatility(target, process, 1e-10, 200, 1e-7, 4.0)
        except RuntimeError:
            continue  # skip cases QuantLib's own solver cannot bracket
        records.append(
            {
                "spot": spot, "strike": strike, "rate": rate,
                "dividend_yield": div, "time": t, "option_type": kind,
                "market_price": target, "implied_volatility": iv,
            }
        )
    return {
        "oracle": "QuantLib", "oracle_version": ql.__version__,
        "model": "black-scholes-implied-vol", "engine": "VanillaOption.impliedVolatility",
        "day_count": "Actual365Fixed", "evaluation_date": str(EVAL_DATE),
        "tolerance": 1e-6, "cases": records,
    }


# Monte Carlo American (Longstaff-Schwartz) reference settings. QuantLib's
# MCAmericanEngine *is* the Longstaff-Schwartz method, so comparing it to OXIS's
# LSM is apples-to-apples (both share the same estimator bias). We record the
# price and QuantLib's own error estimate so the Rust test can use a combined
# standard-error tolerance rather than a fixed absolute one.
MC_AMERICAN_STEPS = 100
MC_AMERICAN_SAMPLES = 100_000
MC_AMERICAN_SEED = 42
MC_AMERICAN_POLYNOM_ORDER = 2
# QuantLib fits the early-exercise boundary on a separate calibration pass; its
# default (2048 paths) is far too small for deep in-the-money cases and biases
# the price low by well beyond the reported standard error. A large calibration
# sample makes the engine an accurate oracle (verified against the binomial /
# closed-form American value).
MC_AMERICAN_CALIBRATION_SAMPLES = 65_536


def gen_monte_carlo_american():
    """American prices via QuantLib's MCAmericanEngine (Longstaff-Schwartz)."""
    records = []
    for spot, strike, rate, vol, div, days, kind in CASES:
        ql.Settings.instance().evaluationDate = EVAL_DATE
        t = DAY_COUNT.yearFraction(EVAL_DATE, _exercise_date(days))
        process = _process(spot, rate, vol, div)
        option, _ = _american_option(strike, days, kind)
        engine = ql.MCAmericanEngine(
            process,
            "pseudorandom",
            timeSteps=MC_AMERICAN_STEPS,
            requiredSamples=MC_AMERICAN_SAMPLES,
            seed=MC_AMERICAN_SEED,
            polynomOrder=MC_AMERICAN_POLYNOM_ORDER,
            polynomType=ql.LsmBasisSystem.Monomial,
            antitheticVariate=True,
            nCalibrationSamples=MC_AMERICAN_CALIBRATION_SAMPLES,
        )
        option.setPricingEngine(engine)
        records.append(
            {
                "spot": spot, "strike": strike, "rate": rate, "volatility": vol,
                "dividend_yield": div, "time": t, "option_type": kind,
                "style": "american", "steps": MC_AMERICAN_STEPS,
                "samples": MC_AMERICAN_SAMPLES,
                "calibration_samples": MC_AMERICAN_CALIBRATION_SAMPLES,
                "price": option.NPV(), "error_estimate": option.errorEstimate(),
            }
        )
    return {
        "oracle": "QuantLib", "oracle_version": ql.__version__,
        "model": "longstaff-schwartz", "engine": "MCAmericanEngine",
        "method": "least-squares Monte Carlo (antithetic, polynomial order 2)",
        "day_count": "Actual365Fixed", "evaluation_date": str(EVAL_DATE),
        "cases": records,
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outputs = {
        "black_scholes.json": gen_black_scholes(),
        "binomial.json": gen_binomial(),
        "greeks.json": gen_greeks(),
        "implied_vol.json": gen_implied_vol(),
        "monte_carlo_american.json": gen_monte_carlo_american(),
    }
    for name, out in outputs.items():
        path = os.path.join(here, "reference", name)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        print(f"wrote {len(out['cases'])} cases to {name} (QuantLib {ql.__version__})")


if __name__ == "__main__":
    main()
