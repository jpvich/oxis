# Models: Method & Validation Status

Every pricing model documents its method, assumptions, and validation status here.
**No model is "done" until it has a passing validation test against QuantLib (or a
closed form) within a documented tolerance** — that bar is the project's whole
value proposition versus unvalidated Rust libraries.

Status values: **planned** · **building** · **validated** (a passing
QuantLib/closed-form test exists in CI at the stated tolerance).

## Ring 1 — pricing core

| Model | Crate | Method | Reference | Tolerance | Status |
|---|---|---|---|---|---|
| Black-Scholes European | `oxis-pricing` | BSM closed-form (continuous dividend) | QuantLib `AnalyticEuropeanEngine` (1.42.1, 22 cases) | ≤ 1e-10 (max 2.5e-14) | **validated** |
| Binomial (European) | `oxis-pricing` | Cox-Ross-Rubinstein tree (equal-jumps) | QuantLib `BinomialVanillaEngine` (CRR), N=256 + BS limit | ≤ 1e-6 @ matched steps (max 1.1e-10) | **validated** |
| Binomial (American) | `oxis-pricing` | CRR tree + early-exercise check | QuantLib `BinomialVanillaEngine` (CRR), N=256 | ≤ 1e-6 @ matched steps (max 1.1e-10) | **validated** |
| Monte Carlo (European) | `oxis-pricing` | GBM terminal simulation, antithetic variates, + SE | Black-Scholes (QuantLib-validated), 22 cases | ≤ 4 standard errors (worst ~1.7σ) | **validated** |
| Longstaff-Schwartz (American) | `oxis-pricing` | LSM regression (`{1, S/K, (S/K)²}`) on ITM paths, antithetic | QuantLib `MCAmericanEngine` (LSM) + binomial, 22 cases | ≤ 5 combined SE + 0.05 vs QuantLib; ≤ 5 SE + 2.5% vs binomial | **validated** |
| Analytic Greeks | `oxis-greeks` | Closed-form BSM derivatives | QuantLib `AnalyticEuropeanEngine` Greeks (22 cases ×5) | ≤ 1e-8 (max 1.0e-13) | **validated** |
| Finite-difference Greeks | `oxis-greeks` | Central differences (bump: spot 1e-4 rel; vol/rate/time 1e-4 abs) | analytic Greeks (agree ≤1e-4) | documented | **implemented** |
| Implied volatility | `oxis-pricing` | Newton-Raphson on vega + Brent fallback | round-trip + QuantLib `impliedVolatility` (22 cases) | ≤ 1e-6 (max 1.2e-11) | **validated** |

## Ring 2 — term structures

| Model | Crate | Method | Reference | Tolerance | Status |
|---|---|---|---|---|---|
| Yield curve / term structure | `oxis-curves` | interpolated discount/zero/forward — linear (zero rates), log-linear (discount factors), natural cubic (zero rates) | QuantLib `ZeroCurve` / `DiscountCurve` / `NaturalCubicZeroCurve` (1.42.1, 51 queries) | ≤ 1e-10 (max ~4.3e-16) | **validated** |
| Fixed-rate bond | `oxis-bonds` | cashflow PV (flat yield compounded@freq, or curve discounting); YTM (Brent); Macaulay/modified duration; convexity | QuantLib `FixedRateBond` / `BondFunctions` (1.42.1, 11 bonds) | ≤ 1e-8 (max ~9.5e-9) | **validated** |

## Core numerics

| Primitive | Crate | Method | Reference | Status |
|---|---|---|---|---|
| Normal CDF / PDF | `oxis-core` | high-accuracy `erf`/`erfc`-based (~1e-15) | known values (unit-tested) + cross-checked through Black-Scholes (≤2.5e-14 vs QuantLib) | **validated** |
| Polynomial least-squares | `oxis-core` | normal equations + Gaussian elimination (LSM regression) | known polynomials (unit-tested) | **implemented** |
| 1-D interpolation | `oxis-core` | piecewise-linear + natural cubic spline (tridiagonal solve) | known functions (unit-tested) + cross-checked through yield curves (≤1e-10 vs QuantLib) | **validated** |
| Day-count year fraction | `oxis-core` | Act/365, Act/360, 30/360 (US) | hand-checked | **implemented** (unit-tested) |

## Conventions

- **Binomial CRR.** Uses the *equal-jumps* probability `p = ½ + ½·(r−q−σ²/2)·√dt/σ`
  (drift carried in `p`), the same scheme as QuantLib's `BinomialVanillaEngine("crr")`.
  This matches the oracle node-for-node at matched step counts (hence the ~1e-10
  agreement) while still converging to Black-Scholes as `N → ∞`. The alternative
  textbook form `p = (e^{(r−q)dt}−d)/(u−d)` is an equally valid CRR scheme but
  differs at O(1/N), so it is *not* used.
- **Greek conventions** (match QuantLib's `AnalyticEuropeanEngine`): delta `∂V/∂S`,
  gamma `∂²V/∂S²`, **vega `∂V/∂σ` per unit volatility** (not per 1%), **theta
  `∂V/∂t` per year** (`−∂V/∂T`; divide by 365 for per-day), **rho `∂V/∂r` per unit
  rate**.
- **Implied volatility** is conditioning-limited for deep-ITM/OTM low-vol options
  (vega → 0): recovered σ error scales like (price tolerance)/vega. The solver
  uses a 1e-12 price residual so σ stays accurate to ≤1e-6 even there; ATM
  recovers σ to ~1e-9.
- **Monte Carlo** simulates GBM terminal prices in one exact log-normal jump (no
  time discretization for European), reduces variance with **antithetic variates**
  (each draw `z` paired with `−z`), and always reports a **standard error** (over
  the per-pair averages, so the antithetic correlation is captured). Runs are
  **bit-reproducible** for a fixed `(seed, paths)` regardless of thread count:
  each antithetic pair seeds its own RNG from a `splitmix64` mix of `(seed,
  index)`, and results are reduced in index order — `rayon` parallelism never
  changes the number.
- **Longstaff-Schwartz** is a **lower-bound** estimator: the early-exercise policy
  comes from a least-squares regression of discounted continuation values on a
  degree-2 monomial basis of moneyness `S/K` (matching QuantLib's default
  `MCAmericanEngine`), fit over in-the-money paths only. A regression-based policy
  is necessarily suboptimal, so the price sits *below* the true (binomial) value —
  by ≤2.5% here, largest for high-σ / dividend cases where the continuation
  surface is hardest to fit. Validation is therefore two-pronged: apples-to-apples
  against QuantLib's *own* LSM engine (same method, same bias) within a combined
  standard error, plus a bias-banded cross-check against the QuantLib-validated
  binomial price. QuantLib's engine needs a large calibration sample
  (`nCalibrationSamples = 65536`) to be an accurate oracle for deep-ITM cases.
- **Yield curves** are continuously compounded with time in years (`Act/365`),
  matching `MarketData.rate` and the QuantLib oracle. Three interpolation schemes
  each mirror a QuantLib term structure: **linear** in zero rates (`ZeroCurve`,
  `Linear`), **log-linear** in discount factors (`DiscountCurve` — piecewise-
  constant instantaneous forwards), and **natural cubic** in zero rates
  (`NaturalCubicZeroCurve`, second derivative zero at both ends). The
  interpolation scheme is independent of how the curve is built (from zero rates
  or discount factors). QuantLib's interpolated curves anchor `t = 0` at their
  first pillar (the reference date where the discount factor is `1`), so OXIS
  curves accept a leading `t = 0` pillar; this is required for the natural cubic
  spline (a global fit) to match node-for-node. Curves do **not** extrapolate: a
  query outside `[t_first, t_last]` (other than `t = 0`) is an error, matching
  QuantLib without `enableExtrapolation()`.
- **Bonds** are modelled by their cashflows `(t_i, amount_i)` from settlement plus
  the accrued interest (so the financial math is exact and validateable without
  reimplementing calendar/schedule machinery, which is deferred). Yield-to-maturity,
  Macaulay/modified duration, and convexity use **compounding at the coupon
  frequency** `(1 + y/f)^(−f·t)` — the market and QuantLib `Compounded@freq`
  convention; **curve discounting** (via `oxis-curves`) stays continuous. Prices
  are quoted per face with `clean = dirty − accrued`. Validation settles on a coupon
  date (`accrued = 0`) with a `Thirty360(BondBasis)`, `NullCalendar`, `Unadjusted`
  schedule, so the ergonomic `regular` builder reproduces QuantLib's cashflows
  exactly. Bootstrapping a curve from bond/swap quotes is a separate, later module.

## Edge-case contract (applies to every pricing model)

- `σ = 0` → discounted intrinsic value (no `NaN`/`Inf`/panic).
- `T = 0` → intrinsic value `max(S-K,0)` / `max(K-S,0)`.
- Deep ITM/OTM and very high `σ` handled as correct limits.
- European put-call parity holds (tested).
- American price ≥ corresponding European price (tested).
- Binomial: risk-neutral `p ∉ [0,1]` (extreme inputs/too-few steps) → error, not
  a nonsense price.
- Monte Carlo / LSM: the deterministic limits (`T = 0`, `σ = 0`, `S = 0`) return
  the exact value with a standard error of exactly `0.0` (no sampling). Deep-ITM
  American options return the exact intrinsic when immediate exercise dominates.

Later rings (stats, portfolio, ML, market-data) document their methods and
validation here as they land.
