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
| Monte Carlo (European) | `oxis-pricing` | GBM terminal simulation + SE | Black-Scholes | within 3 standard errors | planned (M2b) |
| Longstaff-Schwartz (American) | `oxis-pricing` | LSM regression on basis functions | QuantLib `MCAmericanEngine` + binomial | within MC error + documented bias | planned (M2b) |
| Analytic Greeks | `oxis-greeks` | Closed-form BSM derivatives | QuantLib `AnalyticEuropeanEngine` Greeks (22 cases ×5) | ≤ 1e-8 (max 1.0e-13) | **validated** |
| Finite-difference Greeks | `oxis-greeks` | Central differences (bump: spot 1e-4 rel; vol/rate/time 1e-4 abs) | analytic Greeks (agree ≤1e-4) | documented | **implemented** |
| Implied volatility | `oxis-pricing` | Newton-Raphson on vega + Brent fallback | round-trip + QuantLib `impliedVolatility` (22 cases) | ≤ 1e-6 (max 1.2e-11) | **validated** |

## Core numerics

| Primitive | Crate | Method | Reference | Status |
|---|---|---|---|---|
| Normal CDF / PDF | `oxis-core` | high-accuracy `erf`/`erfc`-based (~1e-15) | known values (unit-tested) + cross-checked through Black-Scholes (≤2.5e-14 vs QuantLib) | **validated** |
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

## Edge-case contract (applies to every pricing model)

- `σ = 0` → discounted intrinsic value (no `NaN`/`Inf`/panic).
- `T = 0` → intrinsic value `max(S-K,0)` / `max(K-S,0)`.
- Deep ITM/OTM and very high `σ` handled as correct limits.
- European put-call parity holds (tested).
- American price ≥ corresponding European price (tested).
- Binomial: risk-neutral `p ∉ [0,1]` (extreme inputs/too-few steps) → error, not
  a nonsense price.

Later rings (stats, portfolio, ML, market-data) document their methods and
validation here as they land.
