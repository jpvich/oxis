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
| Barrier option (8 single-barrier types) | `oxis-pricing` | Reiner-Rubinstein closed form, continuous monitoring, zero rebate (`out = vanilla − in`) | QuantLib `AnalyticBarrierEngine` (1.42.1, 12 cases) | ≤ 1e-8 (max ~2.5e-14) | **validated** |
| Lookback option (fixed & floating strike) | `oxis-pricing` | continuous closed form — Goldman-Sosin-Gatto (floating) / Conze-Viswanathan (fixed), fresh issue (extremum = spot) | QuantLib `AnalyticContinuous{Floating,Fixed}LookbackEngine` (1.42.1, 8 cases) | ≤ 1e-8 (max ~2.1e-14) | **validated** |
| Asian option — geometric average | `oxis-pricing` | Kemna-Vorst closed form (vol `σ/√3`, carry `½(b−σ²/6)`), continuous averaging | QuantLib `AnalyticContinuousGeometricAveragePriceAsianEngine` (1.42.1) | ≤ 1e-8 | **validated** |
| Asian option — arithmetic average | `oxis-pricing` | Monte Carlo over GBM paths (`oxis-stochastic`), discrete fixings, + SE | QuantLib `MCDiscreteArithmeticAPEngine` (1.42.1) | ≤ 4 combined SE | **validated** |

## Ring 2 — stochastic processes

`oxis-stochastic` is a pure path-generation primitive (no pricing inside it) that
`oxis-pricing` consumes for path-dependent exotics, and the later ML / portfolio
rings will consume for training / scenario simulation. A simulator has no "price"
to validate, so the oracle is the **closed-form terminal moment** of each process.

| Process | Crate | Scheme | Reference | Tolerance | Status |
|---|---|---|---|---|---|
| Geometric Brownian motion | `oxis-stochastic` | exact log-Euler | closed-form lognormal mean/variance | ≤ ~4 SE (mean) / rel. band (std) | **validated** |
| Ornstein-Uhlenbeck / Vasicek | `oxis-stochastic` | exact Gaussian transition | closed-form OU mean/variance | ≤ ~4 SE / rel. band | **validated** |
| Cox-Ingersoll-Ross | `oxis-stochastic` | full-truncation Euler | closed-form CIR mean/variance | ≤ ~5 SE / rel. band | **validated** |
| Merton jump-diffusion | `oxis-stochastic` | exact diffusion + compound-Poisson jumps | closed-form mean/variance | ≤ ~5 SE / rel. band | **validated** |
| Heston stochastic vol | `oxis-stochastic` | full-truncation Euler (correlated) | mean `S₀e^{μt}`; European price vs QuantLib `AnalyticHestonEngine` | mean rel. band; price ≤ 5 SE + 0.15 | **validated** |

## Ring 3 — statistics & risk

`oxis-stats` is a pure-compute module of descriptive, risk, performance, and
relational statistics over return / price series, consumed by the later portfolio
and ML rings. QuantLib has thin coverage here, so the oracle is **numpy / scipy /
pandas** (industry-standard, closed-form where applicable) — all moments are
population / biased (÷n) to match the oracle exactly, VaR/ES are positive loss
magnitudes, and annualization scales per-period inputs by `√ppy` (geometric for
returns).

| Family | Crate | Method | Reference | Tolerance | Status |
|---|---|---|---|---|---|
| Descriptive moments | `oxis-stats` | mean, population variance/std, biased skew, Fisher excess kurtosis | `numpy.var(ddof=0)` / `scipy.stats.skew,kurtosis(bias=True)` | ≤ 1e-10 (max ~1.8e-15) | **validated** |
| Returns & volatility | `oxis-stats` | simple/log/cumulative returns, geometric annualized return, annualized vol | numpy closed form | ≤ 1e-10 | **validated** |
| Risk-adjusted ratios | `oxis-stats` | Sharpe, Sortino (downside dev vs MAR, full-`n`), Calmar | numpy closed form | ≤ 1e-10 | **validated** |
| Drawdown | `oxis-stats` | running-peak max drawdown + duration | numpy reference | exact (duration) / ≤ 1e-10 | **validated** |
| VaR / Expected Shortfall | `oxis-stats` | historical (numpy-linear quantile), parametric Gaussian, Cornish-Fisher | `numpy.quantile` / `scipy.stats.norm` | ≤ 1e-10 | **validated** |
| Relational & active-return | `oxis-stats` | covariance, correlation, beta, tracking error, information ratio | `numpy.cov(bias=True)` / `corrcoef` | ≤ 1e-10 | **validated** |
| Autocorrelation | `oxis-stats` | biased ACF (mean-centered, full denom), lags 1–5 | numpy reference | ≤ 1e-10 | **validated** |
| Jarque-Bera | `oxis-stats` | `n/6·(S² + K²/4)`, χ²₂ p-value `exp(−JB/2)` | `scipy.stats.jarque_bera` | stat ≤ 1e-10; p-value ≤ 1e-7 | **validated** |

## Ring 3 — portfolio analytics

`oxis-portfolio` is the first **aggregate** module: it consumes `oxis-stats`
(covariance, VaR) and the core's linear-algebra solver to value holdings and
compute performance, allocation, risk, and Markowitz optimization. It is pure and
sync — it operates on price/return records passed in, not a live data source.
**Money is `f64`** (a deliberate, documented deviation from the spec's
"decimal-precise money": the analytics are ratios / linear algebra validated
against a float numpy oracle, where a decimal type cannot do `sqrt`/`exp`/matrix
solves; exact-cent accounting belongs to a future transaction-ledger module).
Oracle = numpy/scipy: matrix algebra via `np.linalg.solve`, IRR via
`scipy.optimize.brentq`.

| Family | Crate | Method | Reference | Tolerance | Status |
|---|---|---|---|---|---|
| Holdings valuation | `oxis-portfolio` | lot-tracked cost basis, mark-to-market value, unrealized P&L, weight | numpy closed form | ≤ 1e-10 | **validated** |
| Time-weighted return | `oxis-portfolio` | geometric linking of sub-period returns `Vᵢ/(Vᵢ₋₁+flowᵢ)−1` | numpy `np.prod` | ≤ 1e-10 | **validated** |
| Money-weighted return (IRR) | `oxis-portfolio` | root of Act/365 NPV `Σcf/(1+r)^t` via Brent | `scipy.optimize.brentq` | ≤ 1e-9 | **validated** |
| Allocation weights | `oxis-portfolio` | `mvᵢ/Σmv` | numpy | ≤ 1e-10 | **validated** |
| Risk aggregation | `oxis-portfolio` | population covariance matrix, `wᵀΣw` vol, portfolio VaR (reuses `oxis-stats`) | `np.cov(bias=True)`, `np.quantile`, `scipy.stats.norm` | ≤ 1e-10 | **validated** |
| Markowitz optimization | `oxis-portfolio` | unconstrained closed form — min-variance / tangency / efficient-frontier via `Σ⁻¹` (solved, shorting allowed) | `numpy.linalg.solve` | ≤ 1e-10 (max ~8.7e-16) | **validated** |

## Ring 4 — machine-learning pricing

`oxis-ml` is the OXIS differentiator: validated ML pricing, **hand-rolled with no
ML framework** (`candle`/`burn`/`tch`), so the binary stays portable and every
number is auditable. The first model is **Differential Machine Learning** (Huge &
Savine, 2020): a *twin network* — a softplus MLP whose forward pass predicts an
option's price and whose backprop pass predicts its delta — trained on simulated
payoffs *and* their pathwise differentials. The network, its twin (input-gradient)
pass, and the doubled-network training gradient are plain linear algebra over
`oxis-core`; training is Adam with a one-cycle schedule and is bit-reproducible for
a fixed seed. Inference is **Kind A** (pure compute); it sits on top of the
classical engines (`oxis-pricing`, `oxis-greeks`) so its accuracy is measured
against a trusted baseline.

**Validation contract — two layers.** An ML model is an *approximation* and will
not match Black-Scholes to `1e-10`; the non-negotiable ("no model is done without a
test vs a trusted baseline") is preserved by splitting the test:

1. **Inference exactness** — the forward value and input-gradient of a *fixed-weight*
   net match an independent numpy reference to **≤ 1e-12** (proves the math is right).
2. **Model accuracy** — the *trained* net's price and delta lie within a **documented
   error band** vs Black-Scholes over a held-out spot grid (proves the model is
   accurate, not exact).

| Family | Crate | Method | Reference | Tolerance / band | Status |
|---|---|---|---|---|---|
| Network inference | `oxis-ml` | softplus MLP forward value + twin input-gradient (hand-rolled) | numpy forward/backprop on fixed weights | ≤ 1e-12 | **validated** |
| Differential-ML pricing (European, 1-D spot) | `oxis-ml` | twin network trained on pathwise payoff + differential labels; price + delta | Black-Scholes price/delta over an `[80,120]` spot grid | price max-abs ≤ 1.5 / RMSE ≤ 1.0; delta max-abs ≤ 0.10 / RMSE ≤ 0.06 (observed ~0.63/0.45, ~0.046/0.029) | **validated** |
| Deep LSM (American put, 1-D spot) | `oxis-ml` | Longstaff-Schwartz with a per-date neural continuation regression (a fresh softplus MLP of `S/K` per exercise date replaces the `{1, S/K, (S/K)²}` polynomial; same ITM-only regression, antithetic pairs, and per-pair seeding) | QuantLib CRR American tree (2000 steps) over a `{90,100,110}` spot grid | `\|price − binomial\| ≤ 5·SE + 0.40` (observed `\|Δ\|` ~0.21–0.23, SE ~0.09–0.12; gap dominated by the 10-step Bermudan-vs-American exercise discretization, which classical LSM shares) | **validated** |

**Deep LSM.** The estimate is low-biased exactly like classical Longstaff-Schwartz,
so the trusted-baseline contract is a band against the QuantLib-validated binomial
tree, not exactness. The substitution is local — only the continuation regression
changes — so at matched `(paths, steps, seed)` the neural price tracks the polynomial
LSM it replaces. The accuracy band is sized for a deliberately light config (4096
paths, 10 exercise dates) to keep the test fast; the discretization gap shrinks with
more steps.

Neural optimal stopping (Deep Optimal Stopping, the headline American method),
multi-dimensional pricing surfaces, higher Greeks, and a GPU backend are deferred to
later Ring-4 milestones.

## Core numerics

| Primitive | Crate | Method | Reference | Status |
|---|---|---|---|---|
| Normal CDF / PDF | `oxis-core` | high-accuracy `erf`/`erfc`-based (~1e-15) | known values (unit-tested) + cross-checked through Black-Scholes (≤2.5e-14 vs QuantLib) | **validated** |
| Polynomial least-squares | `oxis-core` | normal equations + Gaussian elimination (LSM regression) | known polynomials (unit-tested) | **implemented** |
| Linear solve / matrix inverse | `oxis-core` | dense Gaussian elimination + partial pivoting (`solve_linear_system`, `invert`); shared by LSM regression and Markowitz | known systems (unit-tested) + cross-checked through Markowitz (≤1e-10 vs `numpy.linalg.solve`) | **validated** |
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
- **Exotic options** are **continuously monitored** (the closed-form domain). Barrier
  options use the Reiner-Rubinstein formulas with **zero rebate**, and each knock-out
  is recovered from the exact European parity `in + out = vanilla` (reusing the
  validated Black-Scholes vanilla), so the eight types share four coded knock-in
  formulas. Lookbacks are priced **freshly issued** — the realized running extremum
  equals the spot at inception, matching QuantLib's `minmax` argument. Asians are
  **average-price** (fixed strike): the geometric average is closed-form (Kemna-Vorst),
  the arithmetic average is Monte Carlo over `oxis-stochastic` GBM paths with discrete
  fixings aligned to QuantLib's (`days = n·step`, so fixing year-fractions are exactly
  `i·T/n`). Average-strike (floating) Asians and discrete-monitoring barriers/lookbacks
  are deferred.
- **Stochastic processes** live in `oxis-stochastic` with **no pricing inside** them, so
  downstream rings consume raw paths without depending on `oxis-pricing`. Paths are
  **bit-reproducible** for a fixed `(seed, paths, steps)` via the same per-index
  `splitmix64` seeding + antithetic pairing + ordered reduction as the Monte Carlo
  pricers (the seeding/sample helpers now live in `oxis-core::math`). GBM, OU/Vasicek,
  and Merton jump-diffusion are **exact in distribution** (no time-discretization bias),
  so their moments match the closed form up to sampling error; CIR and the Heston
  variance use a **full-truncation Euler** scheme (the Feller condition `2κθ ≥ σ²` is
  *not* required), which carries a small `O(dt)` bias absorbed by the validation bands.
  Heston's mean is `S₀e^{μt}` exactly; its full dynamics are validated end-to-end by
  pricing a European option over simulated paths against QuantLib's `AnalyticHestonEngine`.

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
