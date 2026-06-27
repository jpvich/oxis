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
    }
    for name, out in outputs.items():
        path = os.path.join(here, "reference", name)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        print(f"wrote {len(out['cases'])} cases to {name} (QuantLib {ql.__version__})")


if __name__ == "__main__":
    main()
