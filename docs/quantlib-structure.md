# QuantLib: Structure, and What OXIS Borrows

A map of how QuantLib is organized, and an explicit decision for every major
subsystem: **copy** (steal the design), **adapt** (same idea, OXIS-native
shape), **defer** (want it, later ring), or **skip** (out of scope). This is a
reference document — it explains *why* OXIS looks the way it does relative to the
library it validates against.

> Scope note. QuantLib is huge: ~2,400 source files and ~436,000 lines of C++ in
> `ql/`, plus a 198-file test suite. OXIS is deliberately **not** a port. We copy
> QuantLib's best *ideas* and its role as a validation oracle; we do not copy its
> size, its template-heavy C++ ergonomics, or its global mutable state. See
> `docs/architecture.md` for the OXIS side of this story.

---

## 1. QuantLib's philosophy in one paragraph

QuantLib separates **what** a financial instrument *is* from **how** it is
priced. An `Instrument` (e.g. a `VanillaOption`) holds its contractual terms; a
`PricingEngine` holds a numerical method (analytic, tree, Monte Carlo, finite
difference). You attach an engine to an instrument and ask for its `NPV()`.
Market inputs flow in through `Handle`s to `Quote`s and `TermStructure`s, and an
`Observable`/`Observer` web automatically recomputes results when inputs change.
Time is modeled richly (calendars, day counters, schedules) around a single
**global evaluation date**. This is a powerful, battle-tested design — and also
the source of QuantLib's steepest learning curve.

---

## 2. The `ql/` tree, subsystem by subsystem

Top-level layout under `references/QuantLib/ql/`:

| Directory | What lives there | OXIS decision |
|---|---|---|
| `math/` | numerics: distributions, solvers, interpolation, optimization, RNGs, linear algebra | **copy** (selectively) |
| `methods/` | numerical methods: `montecarlo/`, `lattices/`, `finitedifferences/` | **copy** (MC + lattices); **skip** FD for now |
| `pricingengines/` | the engines: `vanilla/`, `exotic/`, `barrier/`, `asian/`, `bond/`, ... | **adapt** (the engine *concept*) |
| `instruments/` | instrument definitions: options, bonds, swaps, swaptions, ... | **adapt** (typed structs, not class tree) |
| `processes/` | stochastic processes: GBM, Heston, Hull-White, Bates, ... | **defer** (Ring 2) |
| `termstructures/` | yield curves, vol surfaces, credit/inflation curves + bootstrapping | **defer** (Ring 2) |
| `time/` | `date.hpp`, `calendar`, `daycounter`, `schedule`, `period`, calendars | **adapt** (date + day-count done; calendars deferred) |
| `cashflows/` | coupons, legs, cashflow pricers (fixed/float/CMS/inflation) | **defer** (Ring 2, fixed income) |
| `indexes/` | IBOR / swap / inflation / equity indices | **defer** (Ring 2) |
| `models/` | calibratable models: short-rate, equity (Heston), market models | **defer** (Ring 2+) |
| `currencies/` | currency definitions by region + crypto | **adapt** (minimal `Currency`/`Money` done) |
| `patterns/` | `observable`, `lazyobject`, `singleton`, `visitor`, `curiouslyrecurring` | **skip** (see §4) |
| `instruments/`, `experimental/` | bleeding-edge / unstable additions | **skip** |
| `Examples/`, `test-suite/` | example programs and the C++ test suite | reference only |

### 2.1 `math/` — the numerics layer (copy, selectively)

This is the part of QuantLib most worth studying, because correct numerics are
the foundation of correct prices.

- `math/distributions/` — `normaldistribution.{hpp,cpp}` holds
  `CumulativeNormalDistribution` (the normal CDF) and its inverse, plus
  bivariate normal, chi-square, gamma, Student-t, Poisson, binomial. **OXIS
  copies the normal CDF/PDF** (high-accuracy `erf`-based) — already implemented
  in `oxis::core::math` and cross-checked to ~1e-14 through Black-Scholes.
- `math/solvers1d/` — root finders: `brent`, `newton`, `newtonsafe`,
  `bisection`, `ridder`, `secant`, `falseposition`, `halley`. **OXIS copies
  Newton + Brent** (needed for implied vol, M2).
- `math/interpolations/` — linear, cubic spline, log-linear, SABR, and more.
  **OXIS copies linear + cubic** when curves arrive (Ring 2).
- `math/optimization/` — Levenberg-Marquardt, BFGS, conjugate gradient,
  differential evolution, line searches. **OXIS defers** (needed for model
  calibration and portfolio optimization, Ring 2–3).
- `math/randomnumbers/` — Mersenne Twister, Sobol, Halton, Box-Muller,
  inverse-cumulative. **OXIS copies** an MT19937 + inverse-CDF Gaussian path for
  Monte Carlo (M2).
- `math/matrix`, `math/integrals`, `math/copulas`, `math/statistics` — linear
  algebra, quadrature, copulas, incremental statistics. **OXIS adapts** the
  statistics pieces in `oxis::stats` (Ring 3); copulas/quadrature deferred.

### 2.2 `methods/` — numerical methods (copy MC + lattices)

- `methods/montecarlo/` — `pathgenerator`, `montecarlomodel`, `pathpricer`,
  `longstaffschwartzpathpricer`, `brownianbridge`, `lsmbasissystem`. This is the
  blueprint for **OXIS's Monte Carlo + Longstaff-Schwartz** (M2). The clean
  split of *path generation* / *path pricing* / *statistics accumulation* is
  worth copying directly.
- `methods/lattices/` — `binomialtree`, `trinomialtree`, `bsmlattice`,
  `tflattice`. The blueprint for **OXIS's CRR binomial** (European + American,
  M2).
- `methods/finitedifferences/` — PDE solvers (Crank-Nicolson, explicit/implicit
  Euler, operators, meshers, schemes). Powerful but heavy. **OXIS skips FD for
  now**; trees + MC cover the early-exercise cases we need first.

### 2.3 `pricingengines/` — the engines (adapt the concept)

QuantLib's `pricingengines/vanilla/` alone holds dozens of engines:
`analyticeuropeanengine` (our M1 oracle), `binomialengine`,
`baroneadesiwhaleyengine`, `bjerksundstenslandengine`, `analytichestonengine`,
`fdblackscholesvanillaengine`, `mceuropeanengine`, and more — plus whole
subdirectories for `barrier/`, `asian/`, `basket/`, `lookback/`, `bond/`,
`swap/`, `swaption/`, `credit/`, `inflation/`, `quanto/`.

**OXIS adapts the Instrument↔Engine split conceptually** but not literally:
where QuantLib uses an abstract `PricingEngine` polymorphism wired through an
observer web, OXIS uses **plain functions over plain typed structs**
(`black_scholes(&EuropeanOption, &MarketData) -> Result<f64>`). Same separation
of "contract" from "method", far less machinery. The engine catalog tells us
*what to build and in what order* (see `docs/parity.md`).

### 2.4 `time/` — dates and day counts (adapt; partly done)

`time/date.hpp`, `daycounters/`, `calendar.hpp` + `calendars/` (dozens of
national calendars), `schedule.hpp`, `period.hpp`. **OXIS has copied the
essentials**: a validated `Date` (proleptic Gregorian, serial arithmetic) and
`DayCount` (Act/365, Act/360, 30/360) in `oxis::core::types`. Business-day
calendars and schedules are **deferred** to Ring 2 (needed for fixed income, not
for vanilla options).

### 2.5 `instruments/`, `processes/`, `termstructures/`, `cashflows/`, `models/`

These are the breadth of QuantLib: bonds, swaps, swaptions, caps/floors, the
full stochastic-process zoo (Heston, Hull-White, Bates, G2++), yield-curve and
vol-surface bootstrapping, coupon legs. **All deferred to Ring 2+**, in the
order set by `docs/parity.md`. They are exactly where OXIS aims to reach
QuantLib's coverage — *after* the validated pricing core is solid.

---

## 3. QuantLib design patterns — what to steal, what to avoid

QuantLib's `ql/patterns/` directory is the heart of its architecture.

| Pattern | What it does in QuantLib | OXIS stance |
|---|---|---|
| **Instrument ↔ PricingEngine** | decouples contract from numerical method | **steal** (as functions over typed structs, not a class hierarchy) |
| **Handle / Quote** | a shared, swappable pointer to a live market input | **avoid the indirection** — OXIS passes plain `MarketData` values; no live-mutation web |
| **Observable / Observer (`lazyobject`)** | auto-recompute when an input changes | **avoid** — pure functions recompute on call; no hidden invalidation graph |
| **Singleton `Settings` (global eval date)** | one process-wide "today" every calc reads | **avoid** — OXIS threads time explicitly (`expiry_years`, recorded `t`); no global mutable state |
| **`TermStructure` hierarchy** | curves with reference dates + interpolation | **adapt later** (Ring 2) as typed curve structs |
| **Visitor / CRTP templates** | C++ double-dispatch and static polymorphism | **skip** — Rust enums + traits cover this without the template pain |

The single most important *avoid* is the **global evaluation date**. QuantLib's
`Settings::instance().evaluationDate` is a process-wide mutable "today" that
every pricing call implicitly reads. It is ergonomic for scripts and a notorious
source of bugs in concurrent/served contexts. OXIS makes time an explicit input
— which is also what lets our pricing core stay pure and thread-safe.

---

## 4. The summary table: OXIS vs QuantLib by decision

| QuantLib subsystem | Copy | Adapt | Defer | Skip | Notes |
|---|:--:|:--:|:--:|:--:|---|
| Normal CDF/PDF & special functions | ● | | | | done in `oxis::core::math` |
| 1-D root finders (Newton/Brent) | ● | | | | **done** in `oxis::core::math::solvers` (M2a) |
| Interpolation (linear/cubic) | ● | | | | Ring 2 (curves) |
| Optimization (LM/BFGS/...) | | | ● | | calibration + portfolio opt |
| RNGs (MT/Sobol) | ● | | | | **`rand` SmallRng done** (M2b, counter-seeded for determinism); Sobol later |
| Monte Carlo framework | ● | | | | **done** (M2b) — European MC + LSM American, mirrors `MCAmericanEngine`; antithetic, `rayon`-parallel |
| Lattices (binomial/trinomial) | ● | | | | **CRR done** (M2a), mirrors `BinomialVanillaEngine("crr")`; trinomial later |
| Finite differences (PDE) | | | | ● | trees + MC cover our needs first |
| Instrument ↔ Engine split | | ● | | | functions over typed structs |
| Vanilla analytic engines | ● | | | | **BS + Greeks + implied-vol done** (M1/M2a), mirror `AnalyticEuropeanEngine` |
| Exotics (barrier/Asian/lookback/...) | | | ● | | Ring 2 |
| Stochastic processes (Heston/HW/...) | | | ● | | Ring 2 |
| Term structures / bootstrapping | | | ● | | Ring 2 |
| Cashflows / bonds / swaps | | | ● | | Ring 2 (fixed income) |
| Calibratable models | | | ● | | Ring 2+ |
| Date & day-count | | ● | | | done in `oxis::core::types` |
| Calendars & schedules | | | ● | | Ring 2 |
| Currency / Money | | ● | | | minimal version done |
| Observable/Observer web | | | | ● | pure functions instead |
| Global `Settings` eval date | | | | ● | explicit time inputs instead |
| Handle/Quote indirection | | | | ● | plain values instead |
| Visitor / CRTP templates | | | | ● | Rust enums + traits |
| `experimental/` | | | | ● | unstable by definition |

---

## 5. Where OXIS deliberately goes *beyond* QuantLib

These have no real QuantLib counterpart and are the reason OXIS exists as more
than "QuantLib in Rust" (tracked in `docs/parity.md`):

- **Validation as a first-class deliverable.** QuantLib *is* the reference; it
  has a test suite but no notion of "every model ships with a cross-check
  against an independent oracle at a documented tolerance." OXIS makes that the
  merge gate (see `validation/` and `crates/oxis/tests/pricing_validation_tests.rs`).
- **One pure core, four interfaces.** The same I/O-free functions back the Rust
  crate, the `oxis` CLI, the PyO3 Python module, and (later) the REPL. QuantLib's
  Python comes via SWIG over the stateful C++ object graph.
- **Two declared module kinds + serde interchange.** Compute modules stay pure;
  service modules (market-data, later) isolate I/O behind a trait. Typed
  `serde`-friendly structs are the universal contract — see `docs/architecture.md`.
- **Platform reach.** Stats/time-series, portfolio & risk, ML-based pricing, and
  a provider-agnostic market-data API are planned rings. QuantLib is intentionally
  scoped to pricing/risk libraries, not a data or ML platform.

---

## 6. How to use this document

- Building a new pricing model? Find QuantLib's engine for it under
  `references/QuantLib/ql/pricingengines/`, read it for the method and edge-case
  handling, then implement the OXIS-native pure function and add a QuantLib
  cross-check to `validation/`.
- Unsure whether something is in scope? Check the table in §4 and the ring
  ordering in `docs/parity.md`.
- The relationship in one line: **OXIS borrows QuantLib's numerics and its role
  as an oracle, and rejects its global state and template machinery.**
