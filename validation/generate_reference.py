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
import math
import os

import datetime

import numpy as np
import QuantLib as ql
import scipy
from scipy import stats as sps
from scipy.optimize import brentq

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


# Yield-curve reference. QuantLib's interpolated term structures anchor t=0 at
# their first pillar (the reference date), so every curve below starts with a
# `0`-day anchor; OXIS mirrors this. Pillar tenors are in days; the exact
# Act/365 year fraction is days/365, and we query QuantLib by Time directly so
# both sides use bit-identical t. Three interpolations are covered, each matching
# a QuantLib term structure: Linear (ZeroCurve/Linear), LogLinear
# (DiscountCurve), and natural cubic (ZeroCurve with second-derivative-0 ends).
YIELD_CURVES = [
    # label,           interp,          pillar_days,                 zero_rates
    ("upward", "linear", [0, 182, 365, 730, 1825], [0.020, 0.022, 0.025, 0.030, 0.035]),
    ("upward", "log-linear", [0, 182, 365, 730, 1825], [0.020, 0.022, 0.025, 0.030, 0.035]),
    ("upward", "natural-cubic", [0, 182, 365, 730, 1825], [0.020, 0.022, 0.025, 0.030, 0.035]),
    ("humped", "natural-cubic", [0, 365, 730, 1095, 1825], [0.010, 0.030, 0.028, 0.025, 0.020]),
    ("inverted", "linear", [0, 365, 730, 1825], [0.040, 0.035, 0.030, 0.025]),
    ("negative-short", "log-linear", [0, 365, 730], [-0.005, 0.005, 0.010]),
]

# Candidate query tenors (days); filtered per curve to lie within its pillars.
QUERY_DAYS = [90, 182, 273, 365, 547, 730, 1095, 1460, 1700]


def _yield_ts(dates, zeros, interp):
    """Build the QuantLib term structure matching an OXIS interpolation scheme."""
    if interp == "linear":
        ts = ql.ZeroCurve(dates, zeros, DAY_COUNT, CALENDAR, ql.Linear())
    elif interp == "log-linear":
        dfs = [
            math.exp(-z * DAY_COUNT.yearFraction(dates[0], d))
            for z, d in zip(zeros, dates)
        ]
        ts = ql.DiscountCurve(dates, dfs, DAY_COUNT)
    elif interp == "natural-cubic":
        # NaturalCubicZeroCurve presets the Cubic interpolation to a natural
        # spline (second derivative 0 at both ends), matching OXIS exactly.
        ts = ql.NaturalCubicZeroCurve(dates, zeros, DAY_COUNT)
    else:
        raise ValueError(f"unknown interpolation {interp!r}")
    ts.enableExtrapolation()
    return ts


def gen_yield_curve():
    """Discount / zero / forward queries against QuantLib term structures."""
    ql.Settings.instance().evaluationDate = EVAL_DATE
    records = []
    for label, interp, pillar_days, zeros in YIELD_CURVES:
        dates = [_exercise_date(d) for d in pillar_days]
        ts = _yield_ts(dates, zeros, interp)
        pillar_times = [d / 365.0 for d in pillar_days]
        last_day = pillar_days[-1]
        queries = [d for d in QUERY_DAYS if 0 < d <= last_day]
        for i, d in enumerate(queries):
            t = d / 365.0
            # Forward leg runs to the next query tenor, when one exists.
            forward_t2 = None
            forward_rate = None
            if i + 1 < len(queries):
                d2 = queries[i + 1]
                forward_t2 = d2 / 365.0
                forward_rate = ts.forwardRate(
                    t, forward_t2, ql.Continuous, ql.Annual, True
                ).rate()
            records.append(
                {
                    "curve": label,
                    "interpolation": interp,
                    "pillar_times": pillar_times,
                    "pillar_rates": zeros,
                    "t": t,
                    "discount": ts.discount(t),
                    "zero_rate": ts.zeroRate(t, ql.Continuous, ql.Annual, True).rate(),
                    "forward_t2": forward_t2,
                    "forward_rate": forward_rate,
                }
            )
    return {
        "oracle": "QuantLib", "oracle_version": ql.__version__,
        "model": "yield-curve", "engine": "ZeroCurve / DiscountCurve",
        "compounding": "continuous", "day_count": "Actual365Fixed",
        "evaluation_date": str(EVAL_DATE), "tolerance": 1e-10, "cases": records,
    }


# Fixed-rate bond reference. Bonds settle on a coupon date (issue = evaluation
# date, 0 settlement days) so accrued = 0 and the 30/360 cashflow times are clean
# fractions (k/frequency), which OXIS's regular schedule reproduces exactly. We
# export the cashflow times + amounts so OXIS prices the identical stream, plus
# QuantLib's clean/dirty price, round-tripped yield, duration, convexity (all
# Compounded at the coupon frequency), and a flat-continuous-curve dirty price
# (DiscountingBondEngine) to exercise curve-based discounting.
BOND_DAY_COUNT = ql.Thirty360(ql.Thirty360.BondBasis)
BOND_FACE = 100.0
BOND_FLAT_RATE = 0.035  # continuous, for the curve-discounting leg

# coupon, frequency (coupons/yr), maturity (years), test yield
BOND_CASES = [
    (0.05, 2, 5, 0.05),    # par: yield == coupon
    (0.05, 2, 5, 0.04),    # premium
    (0.05, 2, 5, 0.06),    # discount
    (0.03, 2, 10, 0.045),
    (0.06, 2, 7, 0.05),
    (0.04, 1, 5, 0.04),    # annual
    (0.04, 1, 8, 0.035),
    (0.05, 4, 3, 0.045),   # quarterly
    (0.00, 2, 5, 0.04),    # zero-coupon
    (0.07, 2, 20, 0.05),   # long maturity
    (0.02, 2, 2, -0.001),  # negative yield
]

_FREQ_ENUM = {1: ql.Annual, 2: ql.Semiannual, 4: ql.Quarterly}


def gen_bonds():
    """Fixed-rate bond prices & analytics via QuantLib FixedRateBond / BondFunctions."""
    records = []
    for coupon, freq, years, test_yield in BOND_CASES:
        ql.Settings.instance().evaluationDate = EVAL_DATE
        freq_enum = _FREQ_ENUM[freq]
        maturity = EVAL_DATE + ql.Period(years, ql.Years)
        schedule = ql.Schedule(
            EVAL_DATE, maturity, ql.Period(freq_enum), CALENDAR,
            ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False,
        )
        bond = ql.FixedRateBond(0, BOND_FACE, schedule, [coupon], BOND_DAY_COUNT)
        settlement = bond.settlementDate()

        # Future cashflows (time from settlement via the bond day count + amount).
        # QuantLib emits the final coupon and the redemption as two cashflows on
        # the same date; merge same-date flows so times are strictly increasing
        # (matching OXIS's regular schedule, where the last flow is coupon + face).
        merged = {}
        for cf in bond.cashflows():
            if cf.date() > settlement:
                t = BOND_DAY_COUNT.yearFraction(settlement, cf.date())
                merged[t] = merged.get(t, 0.0) + cf.amount()
        times = sorted(merged)
        amounts = [merged[t] for t in times]

        rate = ql.InterestRate(test_yield, BOND_DAY_COUNT, ql.Compounded, freq_enum)
        clean = bond.cleanPrice(test_yield, BOND_DAY_COUNT, ql.Compounded, freq_enum, settlement)
        dirty = bond.dirtyPrice(test_yield, BOND_DAY_COUNT, ql.Compounded, freq_enum, settlement)
        yield_rt = bond.bondYield(
            ql.BondPrice(clean, ql.BondPrice.Clean),
            BOND_DAY_COUNT, ql.Compounded, freq_enum, settlement,
        )
        mac = ql.BondFunctions.duration(bond, rate, ql.Duration.Macaulay, settlement)
        mod = ql.BondFunctions.duration(bond, rate, ql.Duration.Modified, settlement)
        conv = ql.BondFunctions.convexity(bond, rate, settlement)

        # Curve-discounting leg: flat continuous curve, same day count.
        ts = ql.YieldTermStructureHandle(
            ql.FlatForward(settlement, BOND_FLAT_RATE, BOND_DAY_COUNT, ql.Continuous, ql.Annual)
        )
        bond.setPricingEngine(ql.DiscountingBondEngine(ts))
        curve_dirty = bond.dirtyPrice()

        records.append(
            {
                "coupon_rate": coupon, "frequency": freq, "maturity": years,
                "face": BOND_FACE, "test_yield": test_yield,
                "cashflow_times": times, "cashflow_amounts": amounts,
                "accrued": bond.accruedAmount(settlement),
                "clean_price": clean, "dirty_price": dirty, "yield_roundtrip": yield_rt,
                "macaulay_duration": mac, "modified_duration": mod, "convexity": conv,
                "flat_rate": BOND_FLAT_RATE, "curve_dirty_price": curve_dirty,
            }
        )
    return {
        "oracle": "QuantLib", "oracle_version": ql.__version__,
        "model": "fixed-rate-bond", "engine": "FixedRateBond / BondFunctions",
        "compounding": "compounded at coupon frequency; curve leg continuous",
        "day_count": "Thirty360(BondBasis)", "evaluation_date": str(EVAL_DATE),
        "tolerance": 1e-8, "cases": records,
    }


# Stochastic-process reference cases. The oracle here is the *closed-form moment*
# of each process (no QuantLib needed) — the right ground truth for a path
# simulator. `steps`/`paths`/`seed` tell the Rust simulator how to sample; the
# `std_rel_tol` is the relative band on the terminal std (looser for the
# full-truncation square-root schemes, which carry an O(dt) discretization bias).
PROCESS_CASES = [
    # process, params, x0, t, steps, paths, seed, std_rel_tol
    ("gbm", dict(mu=0.05, sigma=0.20), 100.0, 1.0, 4, 400_000, 1001, 0.05),
    ("gbm", dict(mu=0.03, sigma=0.50), 100.0, 2.0, 4, 400_000, 1002, 0.10),
    ("ornstein-uhlenbeck", dict(kappa=2.0, theta=0.05, sigma=0.02), 0.10, 1.0, 8, 400_000, 1003, 0.04),
    ("vasicek", dict(kappa=1.0, theta=0.03, sigma=0.01), 0.02, 2.0, 8, 400_000, 1004, 0.04),
    ("cir", dict(kappa=1.5, theta=0.04, sigma=0.10), 0.05, 1.0, 250, 400_000, 1005, 0.08),
    ("cir", dict(kappa=2.0, theta=0.04, sigma=0.25), 0.04, 2.0, 400, 400_000, 1006, 0.12),
    ("merton-jump", dict(mu=0.05, sigma=0.20, lambda_=0.5, jump_mean=-0.10, jump_std=0.15), 100.0, 1.0, 50, 400_000, 1007, 0.10),
    ("heston", dict(mu=0.04, v0=0.04, kappa=1.5, theta=0.04, xi=0.30, rho=-0.60), 100.0, 1.0, 250, 400_000, 1008, 0.0),
]


def _process_moments(name, p, x0, t):
    """Closed-form terminal mean and variance of a process (independent of the
    Rust implementation). Returns (mean, var) with var None where no simple closed
    form is used (Heston — validated via the AnalyticHestonEngine price tie-in)."""
    if name == "gbm":
        mu, s = p["mu"], p["sigma"]
        mean = x0 * math.exp(mu * t)
        var = x0 * x0 * math.exp(2 * mu * t) * (math.exp(s * s * t) - 1.0)
        return mean, var
    if name in ("ornstein-uhlenbeck", "vasicek"):
        k, th, s = p["kappa"], p["theta"], p["sigma"]
        e = math.exp(-k * t)
        mean = x0 * e + th * (1.0 - e)
        var = s * s / (2 * k) * (1.0 - math.exp(-2 * k * t))
        return mean, var
    if name == "cir":
        k, th, s = p["kappa"], p["theta"], p["sigma"]
        e = math.exp(-k * t)
        mean = th + (x0 - th) * e
        var = x0 * (s * s / k) * (e - e * e) + th * (s * s / (2 * k)) * (1.0 - e) ** 2
        return mean, var
    if name == "merton-jump":
        mu, s, lam = p["mu"], p["sigma"], p["lambda_"]
        jm, js = p["jump_mean"], p["jump_std"]
        k1 = math.exp(jm + 0.5 * js * js)
        k2 = math.exp(2 * jm + 2 * js * js)
        mean = x0 * math.exp(mu * t) * math.exp(lam * t * (k1 - 1.0))
        e_s2 = x0 * x0 * math.exp(2 * mu * t + s * s * t) * math.exp(lam * t * (k2 - 1.0))
        return mean, e_s2 - mean * mean
    if name == "heston":
        return x0 * math.exp(p["mu"] * t), None
    raise ValueError(f"unknown process {name}")


def gen_processes():
    """Closed-form terminal moments for the stochastic process generators."""
    records = []
    for name, p, x0, t, steps, paths, seed, std_rel_tol in PROCESS_CASES:
        mean, var = _process_moments(name, p, x0, t)
        rec = {
            "process": name, "x0": x0, "t": t,
            "steps": steps, "paths": paths, "seed": seed,
            "mean": mean, "var": var, "std_rel_tol": std_rel_tol,
            # Parameters (JSON-friendly: `lambda_` -> `lambda` is reserved, keep as is).
            "mu": p.get("mu"), "sigma": p.get("sigma"),
            "kappa": p.get("kappa"), "theta": p.get("theta"),
            "lambda": p.get("lambda_"),
            "jump_mean": p.get("jump_mean"), "jump_std": p.get("jump_std"),
            "v0": p.get("v0"), "xi": p.get("xi"), "rho": p.get("rho"),
        }
        records.append(rec)
    return {
        "oracle": "closed-form", "oracle_version": "analytic-moments",
        "model": "stochastic-process-generators",
        "note": "terminal mean/var are exact closed forms; Heston variance omitted "
                "(validated via AnalyticHestonEngine price tie-in in oxis-pricing)",
        "cases": records,
    }


# ----------------------------------------------------------------------------
# Exotic options (Ring 2): barrier, lookback, Asian.
# ----------------------------------------------------------------------------

# spot, strike, rate, vol, div, days, type, barrier_type, barrier.
# Down barriers sit below spot, up barriers above, so the option starts on the
# live side (the closed form's domain).
BARRIER_CASES = [
    (100.0, 100.0, 0.05, 0.25, 0.00, 365, "call", "down-out", 90.0),
    (100.0, 100.0, 0.05, 0.25, 0.00, 365, "call", "down-in", 90.0),
    (100.0, 100.0, 0.05, 0.25, 0.00, 365, "call", "up-out", 130.0),
    (100.0, 100.0, 0.05, 0.25, 0.00, 365, "call", "up-in", 130.0),
    (100.0, 100.0, 0.05, 0.25, 0.00, 365, "put", "down-out", 90.0),
    (100.0, 100.0, 0.05, 0.25, 0.00, 365, "put", "down-in", 90.0),
    (100.0, 100.0, 0.05, 0.25, 0.00, 365, "put", "up-out", 130.0),
    (100.0, 100.0, 0.05, 0.25, 0.00, 365, "put", "up-in", 130.0),
    # strike below / above barrier, dividends, other maturities
    (100.0, 95.0, 0.04, 0.30, 0.02, 365, "call", "down-in", 97.0),
    (100.0, 110.0, 0.04, 0.30, 0.02, 365, "call", "up-out", 120.0),
    (100.0, 105.0, 0.03, 0.20, 0.01, 180, "put", "down-out", 85.0),
    (100.0, 100.0, 0.06, 0.40, 0.00, 730, "call", "up-in", 115.0),
]


def gen_barrier():
    """Single-barrier prices via QuantLib's AnalyticBarrierEngine (rebate 0)."""
    ql.Settings.instance().evaluationDate = EVAL_DATE
    bt_map = {
        "down-in": ql.Barrier.DownIn, "down-out": ql.Barrier.DownOut,
        "up-in": ql.Barrier.UpIn, "up-out": ql.Barrier.UpOut,
    }
    records = []
    for spot, strike, rate, vol, div, days, kind, btype, barrier in BARRIER_CASES:
        process = _process(spot, rate, vol, div)
        t = DAY_COUNT.yearFraction(EVAL_DATE, _exercise_date(days))
        otype = ql.Option.Call if kind == "call" else ql.Option.Put
        payoff = ql.PlainVanillaPayoff(otype, strike)
        exercise = ql.EuropeanExercise(_exercise_date(days))
        opt = ql.BarrierOption(bt_map[btype], barrier, 0.0, payoff, exercise)
        opt.setPricingEngine(ql.AnalyticBarrierEngine(process))
        records.append({
            "spot": spot, "strike": strike, "rate": rate, "vol": vol, "div": div,
            "t": t, "type": kind, "barrier_type": btype, "barrier": barrier,
            "price": opt.NPV(),
        })
    return {
        "oracle": "QuantLib", "oracle_version": ql.__version__,
        "model": "single-barrier", "engine": "AnalyticBarrierEngine",
        "monitoring": "continuous", "rebate": 0.0,
        "tolerance": 1e-8, "cases": records,
    }


# spot, strike, rate, vol, div, days, type, strike_type.
LOOKBACK_CASES = [
    (100.0, 0.0, 0.06, 0.30, 0.02, 365, "call", "floating"),
    (100.0, 0.0, 0.06, 0.30, 0.02, 365, "put", "floating"),
    (100.0, 0.0, 0.04, 0.20, 0.00, 180, "call", "floating"),
    (100.0, 95.0, 0.06, 0.30, 0.02, 365, "call", "fixed"),
    (100.0, 105.0, 0.06, 0.30, 0.02, 365, "put", "fixed"),
    (100.0, 110.0, 0.04, 0.25, 0.01, 365, "call", "fixed"),
    (100.0, 90.0, 0.04, 0.25, 0.01, 365, "put", "fixed"),
    (100.0, 100.0, 0.05, 0.40, 0.00, 730, "call", "fixed"),
]


def gen_lookback():
    """Continuous lookback prices via QuantLib's analytic lookback engines.

    Freshly issued: the realized extremum equals the spot at inception, so the
    `minmax` argument is the spot for both floating and fixed strikes."""
    ql.Settings.instance().evaluationDate = EVAL_DATE
    records = []
    for spot, strike, rate, vol, div, days, kind, stype in LOOKBACK_CASES:
        process = _process(spot, rate, vol, div)
        t = DAY_COUNT.yearFraction(EVAL_DATE, _exercise_date(days))
        otype = ql.Option.Call if kind == "call" else ql.Option.Put
        exercise = ql.EuropeanExercise(_exercise_date(days))
        if stype == "floating":
            opt = ql.ContinuousFloatingLookbackOption(
                spot, ql.FloatingTypePayoff(otype), exercise
            )
            opt.setPricingEngine(ql.AnalyticContinuousFloatingLookbackEngine(process))
        else:
            opt = ql.ContinuousFixedLookbackOption(
                spot, ql.PlainVanillaPayoff(otype, strike), exercise
            )
            opt.setPricingEngine(ql.AnalyticContinuousFixedLookbackEngine(process))
        records.append({
            "spot": spot, "strike": strike, "rate": rate, "vol": vol, "div": div,
            "t": t, "type": kind, "strike_type": stype, "price": opt.NPV(),
        })
    return {
        "oracle": "QuantLib", "oracle_version": ql.__version__,
        "model": "continuous-lookback",
        "engine": "AnalyticContinuousFloating/FixedLookbackEngine",
        "monitoring": "continuous", "issue": "fresh (extremum = spot)",
        "tolerance": 1e-8, "cases": records,
    }


# Geometric (closed form): spot, strike, rate, vol, div, days, type.
ASIAN_GEO_CASES = [
    (100.0, 100.0, 0.05, 0.20, 0.00, 365, "call"),
    (100.0, 100.0, 0.05, 0.20, 0.00, 365, "put"),
    (100.0, 95.0, 0.05, 0.25, 0.02, 365, "call"),
    (100.0, 105.0, 0.05, 0.25, 0.02, 365, "put"),
    (100.0, 100.0, 0.03, 0.40, 0.00, 730, "call"),
]
# Arithmetic (MC): spot, strike, rate, vol, div, n_fixings, step_days, type, seed.
# days = n*step so the QuantLib fixing year-fractions are exactly i·T/n, matching
# the OXIS continuous grid.
ASIAN_ARITH_CASES = [
    (100.0, 100.0, 0.05, 0.20, 0.00, 12, 30, "call", 101),
    (100.0, 100.0, 0.05, 0.20, 0.00, 12, 30, "put", 102),
    (100.0, 95.0, 0.05, 0.25, 0.02, 25, 30, "call", 103),
    (100.0, 105.0, 0.04, 0.30, 0.01, 50, 14, "put", 104),
]
ASIAN_MC_SAMPLES = 600_000


def gen_asian():
    """Asian average-price options: geometric (closed form) + arithmetic (MC)."""
    ql.Settings.instance().evaluationDate = EVAL_DATE
    records = []
    for spot, strike, rate, vol, div, days, kind in ASIAN_GEO_CASES:
        process = _process(spot, rate, vol, div)
        t = DAY_COUNT.yearFraction(EVAL_DATE, _exercise_date(days))
        otype = ql.Option.Call if kind == "call" else ql.Option.Put
        exercise = ql.EuropeanExercise(_exercise_date(days))
        opt = ql.ContinuousAveragingAsianOption(
            ql.Average.Geometric, ql.PlainVanillaPayoff(otype, strike), exercise
        )
        opt.setPricingEngine(
            ql.AnalyticContinuousGeometricAveragePriceAsianEngine(process)
        )
        records.append({
            "average": "geometric", "spot": spot, "strike": strike, "rate": rate,
            "vol": vol, "div": div, "t": t, "type": kind, "price": opt.NPV(),
            "ql_error": None, "n_fixings": None, "paths": None, "seed": None,
        })
    for spot, strike, rate, vol, div, n, step, kind, seed in ASIAN_ARITH_CASES:
        days = n * step
        process = _process(spot, rate, vol, div)
        t = DAY_COUNT.yearFraction(EVAL_DATE, _exercise_date(days))
        otype = ql.Option.Call if kind == "call" else ql.Option.Put
        exercise = ql.EuropeanExercise(_exercise_date(days))
        fixing_dates = [_exercise_date(i * step) for i in range(1, n + 1)]
        opt = ql.DiscreteAveragingAsianOption(
            ql.Average.Arithmetic, 0.0, 0, fixing_dates,
            ql.PlainVanillaPayoff(otype, strike), exercise,
        )
        engine = ql.MCDiscreteArithmeticAPEngine(
            process, "pseudorandom", brownianBridge=False, antitheticVariate=True,
            requiredSamples=ASIAN_MC_SAMPLES, seed=seed,
        )
        opt.setPricingEngine(engine)
        records.append({
            "average": "arithmetic", "spot": spot, "strike": strike, "rate": rate,
            "vol": vol, "div": div, "t": t, "type": kind, "price": opt.NPV(),
            "ql_error": opt.errorEstimate(), "n_fixings": n,
            "paths": 600_000, "seed": seed + 5000,
        })
    return {
        "oracle": "QuantLib", "oracle_version": ql.__version__,
        "model": "average-price-asian",
        "engine": "AnalyticContinuousGeometricAveragePriceAsianEngine / "
                  "MCDiscreteArithmeticAPEngine",
        "note": "geometric is closed-form (tol 1e-8); arithmetic is MC, compared "
                "within a combined standard-error band",
        "tolerance": 1e-8, "cases": records,
    }


# European option under Heston, for the oxis-pricing path-MC tie-in.
# spot, strike, rate, div, days, type, v0, kappa, theta, xi, rho.
HESTON_CASES = [
    (100.0, 100.0, 0.04, 0.00, 365, "call", 0.04, 1.5, 0.04, 0.30, -0.60),
    (100.0, 100.0, 0.04, 0.00, 365, "put", 0.04, 1.5, 0.04, 0.30, -0.60),
    (100.0, 90.0, 0.03, 0.01, 365, "call", 0.05, 2.0, 0.04, 0.50, -0.70),
    (100.0, 110.0, 0.05, 0.00, 730, "call", 0.06, 1.0, 0.05, 0.40, -0.50),
]


def gen_heston_european():
    """European prices under Heston via the semi-analytic AnalyticHestonEngine.

    Used by oxis-pricing to validate that a Monte Carlo price over Heston paths
    from oxis-stochastic matches QuantLib within a standard-error band — the
    end-to-end check of the (hardest) Heston dynamics."""
    ql.Settings.instance().evaluationDate = EVAL_DATE
    records = []
    for spot, strike, rate, div, days, kind, v0, kappa, theta, xi, rho in HESTON_CASES:
        t = DAY_COUNT.yearFraction(EVAL_DATE, _exercise_date(days))
        sh = ql.QuoteHandle(ql.SimpleQuote(spot))
        rts = ql.YieldTermStructureHandle(
            ql.FlatForward(EVAL_DATE, rate, DAY_COUNT, ql.Continuous, ql.Annual)
        )
        dts = ql.YieldTermStructureHandle(
            ql.FlatForward(EVAL_DATE, div, DAY_COUNT, ql.Continuous, ql.Annual)
        )
        process = ql.HestonProcess(rts, dts, sh, v0, kappa, theta, xi, rho)
        model = ql.HestonModel(process)
        engine = ql.AnalyticHestonEngine(model)
        otype = ql.Option.Call if kind == "call" else ql.Option.Put
        opt = ql.VanillaOption(
            ql.PlainVanillaPayoff(otype, strike), ql.EuropeanExercise(_exercise_date(days))
        )
        opt.setPricingEngine(engine)
        records.append({
            "spot": spot, "strike": strike, "rate": rate, "div": div, "t": t,
            "type": kind, "v0": v0, "kappa": kappa, "theta": theta, "xi": xi,
            "rho": rho, "price": opt.NPV(),
            "paths": 400_000, "steps": 400, "seed": 2024,
        })
    return {
        "oracle": "QuantLib", "oracle_version": ql.__version__,
        "model": "heston-european", "engine": "AnalyticHestonEngine",
        "note": "MC over oxis-stochastic Heston paths must match within a "
                "combined standard-error band",
        "cases": records,
    }


# ----------------------------------------------------------------------------
# Statistics & risk metrics (Ring 3) — oracle is numpy / scipy, not QuantLib.
# ----------------------------------------------------------------------------

# Conventions mirror oxis-stats exactly (population/biased moments, positive-loss
# VaR/ES, geometric annualized return, √ppy scaling, numpy-linear quantile).

# Fixed deterministic series (no RNG — the oracle must be reproducible).
STATS_RETURNS = [
    0.012, -0.008, 0.020, -0.015, 0.005, 0.018, -0.010, 0.022,
    -0.003, 0.011, -0.020, 0.014, 0.007, -0.012, 0.016, -0.006,
]
STATS_PRICES = [
    100.0, 102.0, 101.0, 98.0, 95.0, 97.0, 99.0, 103.0, 96.0, 94.0, 98.0, 105.0,
]
STATS_PORT = [0.011, -0.009, 0.018, -0.014, 0.006, 0.017, -0.011, 0.021, -0.004, 0.010, -0.019, 0.013]
STATS_BENCH = [0.009, -0.007, 0.015, -0.012, 0.004, 0.014, -0.008, 0.018, -0.002, 0.008, -0.016, 0.011]

STATS_RF = 0.0002      # per-period risk-free / MAR
STATS_PPY = 252.0      # periods per year
STATS_ACF_LAGS = [1, 2, 3, 4, 5]


def _autocorr(x, lag):
    """numpy-style biased autocorrelation at `lag` (mean-centered, full denom)."""
    x = np.asarray(x, dtype=float)
    m = x.mean()
    denom = np.sum((x - m) ** 2)
    if lag == 0:
        return 1.0
    num = np.sum((x[:-lag] - m) * (x[lag:] - m))
    return float(num / denom)


def _max_drawdown(prices):
    """Replicate oxis-stats' running-peak drawdown (magnitude, indices, duration)."""
    peak, peak_idx = prices[0], 0
    best = (0.0, 0, 0, 0)  # (max_dd, peak_index, trough_index, duration)
    for i, p in enumerate(prices):
        if p > peak:
            peak, peak_idx = p, i
        dd = (peak - p) / peak
        if dd > best[0]:
            best = (dd, peak_idx, i, i - peak_idx)
    return best


def _hist_var(r, c):
    return float(-np.quantile(r, 1.0 - c, method="linear"))


def _hist_es(r, c):
    r = np.asarray(r, dtype=float)
    thr = np.quantile(r, 1.0 - c, method="linear")
    tail = r[r <= thr]
    if tail.size == 0:
        tail = np.array([r.min()])
    return float(-tail.mean())


def _param_var(r, c):
    mu, sigma = np.mean(r), np.std(r, ddof=0)
    z = sps.norm.ppf(1.0 - c)
    return float(-(mu + z * sigma))


def _param_es(r, c):
    mu, sigma = np.mean(r), np.std(r, ddof=0)
    alpha = 1.0 - c
    z = sps.norm.ppf(alpha)
    return float(-mu + sigma * sps.norm.pdf(z) / alpha)


def _cornish_fisher_var(r, c):
    mu, sigma = np.mean(r), np.std(r, ddof=0)
    s = sps.skew(r, bias=True)
    k = sps.kurtosis(r, fisher=True, bias=True)
    z = sps.norm.ppf(1.0 - c)
    z_cf = (z + (z**2 - 1) * s / 6.0 + (z**3 - 3 * z) * k / 24.0
            - (2 * z**3 - 5 * z) * s**2 / 36.0)
    return float(-(mu + z_cf * sigma))


def _descriptive(sample):
    """The descriptive + autocorrelation + JB block for any sample."""
    x = np.asarray(sample, dtype=float)
    jb_stat, jb_p = sps.jarque_bera(x)
    return {
        "mean": float(np.mean(x)),
        "variance": float(np.var(x, ddof=0)),
        "std_dev": float(np.std(x, ddof=0)),
        "skewness": float(sps.skew(x, bias=True)),
        "excess_kurtosis": float(sps.kurtosis(x, fisher=True, bias=True)),
        "jarque_bera": float(jb_stat),
        "jarque_bera_pvalue": float(jb_p),
        "acf_lags": STATS_ACF_LAGS,
        "acf": [_autocorr(x, k) for k in STATS_ACF_LAGS],
    }


def _returns_block(r, c, rf, ppy):
    """The returns / risk / VaR / ES block for a returns series."""
    r = np.asarray(r, dtype=float)
    n = r.size
    growth = float(np.prod(1.0 + r))
    sd = float(np.std(r, ddof=0))
    downside = float(np.mean(np.minimum(r - rf, 0.0) ** 2))
    dd = math.sqrt(downside)
    return {
        "cumulative_return": growth - 1.0,
        "annualized_return": growth ** (ppy / n) - 1.0,
        "annualized_volatility": sd * math.sqrt(ppy),
        "sharpe": (float(np.mean(r)) - rf) / sd * math.sqrt(ppy),
        "sortino": (float(np.mean(r)) - rf) / dd * math.sqrt(ppy),
        "historical_var": _hist_var(r, c),
        "historical_es": _hist_es(r, c),
        "parametric_var": _param_var(r, c),
        "parametric_es": _param_es(r, c),
        "cornish_fisher_var": _cornish_fisher_var(r, c),
    }


def gen_stats():
    """Descriptive, risk, performance, and relational statistics (numpy/scipy)."""
    cases = []

    # Case A — returns at 95% confidence: descriptive + returns/risk + acf + JB.
    a = {"name": "returns_c95", "returns": STATS_RETURNS, "risk_free": STATS_RF,
         "periods_per_year": STATS_PPY, "confidence": 0.95}
    a.update(_descriptive(STATS_RETURNS))
    a.update(_returns_block(STATS_RETURNS, 0.95, STATS_RF, STATS_PPY))
    cases.append(a)

    # Case A2 — same returns at 99% confidence: tail (VaR/ES) coverage.
    a2 = {"name": "returns_c99", "returns": STATS_RETURNS, "risk_free": STATS_RF,
          "periods_per_year": STATS_PPY, "confidence": 0.99}
    a2.update(_returns_block(STATS_RETURNS, 0.99, STATS_RF, STATS_PPY))
    cases.append(a2)

    # Case B — price path: drawdown + Calmar.
    mdd, _peak, _trough, dur = _max_drawdown(STATS_PRICES)
    prices = np.asarray(STATS_PRICES, dtype=float)
    simple = prices[1:] / prices[:-1] - 1.0
    growth = float(np.prod(1.0 + simple))
    ann = growth ** (STATS_PPY / simple.size) - 1.0
    b = {"name": "prices", "prices": STATS_PRICES, "risk_free": STATS_RF,
         "periods_per_year": STATS_PPY, "confidence": 0.95,
         "max_drawdown": mdd, "max_drawdown_duration": dur, "calmar": ann / mdd}
    cases.append(b)

    # Case C — portfolio vs benchmark: relational + active-return metrics.
    port = np.asarray(STATS_PORT, dtype=float)
    bench = np.asarray(STATS_BENCH, dtype=float)
    active = port - bench
    c = {"name": "port_vs_bench", "returns": STATS_PORT, "benchmark": STATS_BENCH,
         "risk_free": STATS_RF, "periods_per_year": STATS_PPY, "confidence": 0.95,
         "covariance": float(np.cov(port, bench, bias=True)[0, 1]),
         "correlation": float(np.corrcoef(port, bench)[0, 1]),
         "beta": float(np.cov(port, bench, bias=True)[0, 1] / np.var(bench, ddof=0)),
         "tracking_error": float(np.std(active, ddof=0) * math.sqrt(STATS_PPY)),
         "information_ratio": float(np.mean(active) / np.std(active, ddof=0) * math.sqrt(STATS_PPY))}
    cases.append(c)

    return {
        "oracle": "numpy/scipy/pandas",
        "oracle_version": f"numpy {np.__version__}, scipy {scipy.__version__}",
        "model": "statistics",
        "tolerance": 1e-10,
        "pvalue_tolerance": 1e-7,
        "cases": cases,
    }


# ----------------------------------------------------------------------------
# Portfolio analytics (Ring 3, M7) — oracle is numpy / scipy.
# ----------------------------------------------------------------------------

# Conventions mirror oxis-portfolio exactly: f64 money; TWR sub-period flows at
# start; MWR = Act/365 NPV root; population covariance (np.cov bias=True);
# Markowitz via np.linalg.solve (not inv); positive-loss VaR.

# (symbol, quantity, unit_cost, price)
PORT_HOLDINGS = [("AAPL", 10.0, 150.0, 175.0), ("MSFT", 5.0, 300.0, 320.0), ("NVDA", 8.0, 400.0, 650.0)]
PORT_VALUES = [100000.0, 102000.0, 101500.0, 108000.0]
PORT_FLOWS = [0.0, 5000.0, -2000.0]
PORT_CF_DATES = ["2024-01-01", "2024-07-01", "2025-01-01"]
PORT_CF_AMOUNTS = [-10000.0, -5000.0, 16000.0]
OPT_MEAN = [0.08, 0.10, 0.13]
OPT_COV = [
    [0.0100, 0.0018, 0.0011],
    [0.0018, 0.0109, 0.0026],
    [0.0011, 0.0026, 0.0199],
]
OPT_RF = 0.02
OPT_TARGET = 0.11
RISK_RETURNS = [
    [0.012, -0.008, 0.020, -0.015, 0.005, 0.018, -0.010, 0.022],
    [0.009, -0.005, 0.014, -0.011, 0.004, 0.013, -0.007, 0.017],
    [0.020, 0.002, -0.012, 0.025, -0.018, 0.010, 0.006, -0.009],
]
RISK_WEIGHTS = [0.40, 0.35, 0.25]
RISK_PPY = 252.0
RISK_CONF = 0.95


def _npv(amounts, days, r):
    return sum(cf / (1.0 + r) ** (d / 365.0) for cf, d in zip(amounts, days))


def gen_portfolio():
    """Portfolio valuation, performance, allocation, risk, optimization."""
    cases = []

    # Valuation.
    mvs = [q * p for (_s, q, _c, p) in PORT_HOLDINGS]
    bases = [q * c for (_s, q, c, _p) in PORT_HOLDINGS]
    total_mv, total_cost = sum(mvs), sum(bases)
    cases.append({
        "name": "valuation",
        "holdings": [[s, q, c, p] for (s, q, c, p) in PORT_HOLDINGS],
        "market_values": mvs,
        "unrealized_pnls": [mv - b for mv, b in zip(mvs, bases)],
        "weights": [mv / total_mv for mv in mvs],
        "total_cost_basis": total_cost,
        "total_market_value": total_mv,
        "total_unrealized_pnl": total_mv - total_cost,
    })

    # TWR.
    twr = 1.0
    for i in range(len(PORT_FLOWS)):
        twr *= PORT_VALUES[i + 1] / (PORT_VALUES[i] + PORT_FLOWS[i])
    cases.append({"name": "twr", "values": PORT_VALUES, "flows": PORT_FLOWS, "twr": twr - 1.0})

    # MWR (IRR via Act/365 NPV root).
    first = datetime.date.fromisoformat(PORT_CF_DATES[0])
    days = [(datetime.date.fromisoformat(d) - first).days for d in PORT_CF_DATES]
    irr = brentq(lambda r: _npv(PORT_CF_AMOUNTS, days, r), -0.999, 10.0, xtol=1e-14, rtol=1e-15)
    cases.append({"name": "mwr", "dates": PORT_CF_DATES, "amounts": PORT_CF_AMOUNTS, "mwr": irr})

    # Allocation.
    cases.append({"name": "allocation", "market_values": mvs, "weights": [mv / total_mv for mv in mvs]})

    # Risk aggregation.
    R = np.asarray(RISK_RETURNS, dtype=float)
    w = np.asarray(RISK_WEIGHTS, dtype=float)
    cov = np.cov(R, bias=True)
    variance = float(w @ cov @ w)
    vol = math.sqrt(variance)
    port = w @ R  # weighted portfolio return series
    hist_var = float(-np.quantile(port, 1.0 - RISK_CONF, method="linear"))
    z = sps.norm.ppf(1.0 - RISK_CONF)
    param_var = float(-(np.mean(port) + z * np.std(port, ddof=0)))
    cases.append({
        "name": "risk",
        "returns": RISK_RETURNS, "weights": RISK_WEIGHTS,
        "periods_per_year": RISK_PPY, "confidence": RISK_CONF,
        "variance": variance, "volatility": vol,
        "annualized_volatility": vol * math.sqrt(RISK_PPY),
        "historical_var": hist_var, "parametric_var": param_var,
    })

    # Markowitz optimization (np.linalg.solve, not inv).
    C = np.asarray(OPT_COV, dtype=float)
    mu = np.asarray(OPT_MEAN, dtype=float)
    ones = np.ones(len(mu))
    x1 = np.linalg.solve(C, ones)
    xmu = np.linalg.solve(C, mu)
    A, B, Cc = float(ones @ x1), float(ones @ xmu), float(mu @ xmu)
    D = A * Cc - B * B
    min_var = (x1 / A).tolist()
    z_tan = np.linalg.solve(C, mu - OPT_RF * ones)
    tangency = (z_tan / z_tan.sum()).tolist()
    frontier = (x1 * (Cc - B * OPT_TARGET) / D + xmu * (A * OPT_TARGET - B) / D).tolist()
    cases.append({
        "name": "optimize",
        "mean": OPT_MEAN, "cov": OPT_COV, "rf": OPT_RF, "target": OPT_TARGET,
        "min_variance_weights": min_var, "tangency_weights": tangency,
        "frontier_weights": frontier,
    })

    return {
        "oracle": "numpy/scipy/pandas",
        "oracle_version": f"numpy {np.__version__}, scipy {scipy.__version__}",
        "model": "portfolio",
        "tolerance": 1e-10,
        "irr_tolerance": 1e-9,
        "cases": cases,
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outputs = {
        "black_scholes.json": gen_black_scholes(),
        "binomial.json": gen_binomial(),
        "greeks.json": gen_greeks(),
        "implied_vol.json": gen_implied_vol(),
        "monte_carlo_american.json": gen_monte_carlo_american(),
        "yield_curve.json": gen_yield_curve(),
        "bonds.json": gen_bonds(),
        "processes.json": gen_processes(),
        "barrier.json": gen_barrier(),
        "lookback.json": gen_lookback(),
        "asian.json": gen_asian(),
        "heston_european.json": gen_heston_european(),
        "stats.json": gen_stats(),
        "portfolio.json": gen_portfolio(),
    }
    for name, out in outputs.items():
        path = os.path.join(here, "reference", name)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        print(f"wrote {len(out['cases'])} cases to {name} (oracle: {out['oracle']})")


if __name__ == "__main__":
    main()
