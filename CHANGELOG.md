# Changelog

All notable changes to OXIS are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and OXIS adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Pre-1.0, the public API may change in any `0.x` release; breaking changes are
called out explicitly under **Changed**.

## [Unreleased]

## [0.1.0] — 2026-07-22

First public release. Every pricing model listed below ships with a passing
validation test against QuantLib, a closed form, or numpy/scipy, enforced in CI
on Linux and macOS.

### Added

**Pricing (`oxis::pricing`)**
- Black-Scholes European closed form, validated vs QuantLib `AnalyticEuropeanEngine`.
- Binomial CRR trees, European and American, validated vs QuantLib `BinomialVanillaEngine`.
- Monte Carlo European with antithetic variates; every estimate reports a standard error.
- Longstaff-Schwartz American Monte Carlo, validated vs QuantLib `MCAmericanEngine` and the tree.
- Exotics: barrier, lookback, and Asian options.
- Implied-volatility solver (round-trip and QuantLib validated).

**Greeks (`oxis::greeks`)**
- Analytic Δ Γ ν Θ ρ where a closed form exists, finite-difference fallback with a
  documented bump size, validated vs QuantLib.

**Term structures (`oxis::curves`, `oxis::bonds`)**
- Yield curves and discount/zero/forward interpolation, validated vs QuantLib
  `ZeroCurve` / `DiscountCurve`.
- Fixed-rate bonds: price, yield-to-maturity, duration, convexity, validated vs
  QuantLib `FixedRateBond` / `BondFunctions` (≤1e-8).

**Processes (`oxis::stochastic`)**
- GBM, Ornstein-Uhlenbeck, Vasicek, CIR, Merton jump-diffusion, and Heston path
  generators, validated against closed-form terminal moments.

**Statistics & portfolio (`oxis::stats`, `oxis::portfolio`)**
- Returns, volatility, Sharpe/Sortino, VaR and expected shortfall, drawdown, beta,
  Jarque-Bera, autocorrelation — validated vs numpy/scipy/pandas.
- Portfolio valuation, TWR/MWR performance, allocation, risk, and Markowitz
  optimization.

**Machine learning (`oxis::ml`)** — the differentiating module, hand-rolled with no
ML framework (`candle`/`burn`/`tch`), so every number is auditable and the binary
stays portable. Validated two ways: inference exact to ≤1e-12 against a reference
forward pass, and trained accuracy within documented bands against a trusted engine.
- Differential ML (Huge-Savine twin network) for European price and delta, banded
  against Black-Scholes.
- Deep LSM — Longstaff-Schwartz with a neural continuation regression, banded
  against the QuantLib CRR American tree.
- Deep Optimal Stopping (Becker-Cheridito-Jentzen) — per-date stop-probability
  networks, priced out-of-sample for a valid low-biased estimate, banded against
  the same tree.
- A cross-engine comparison suite pricing one American put four ways
  (binomial ↔ classical LSM ↔ Deep LSM ↔ DOS).

**Interfaces**
- Rust library: one `oxis` crate, every domain behind a Cargo feature
  (`pricing`, `greeks`, `curves`, `bonds`, `stochastic`, `stats`, `portfolio`,
  `ml`; `full` enables all, `cli` adds the binary and is on by default).
- CLI: `oxis price`, `greeks`, `implied-vol`, `exotic`, `curve`, `bond`, `simulate`,
  `stats`, `portfolio`, `ml`; human-readable on a TTY, `--json` / `--tsv` when piped.
- REPL: run `oxis` with no subcommand. Tab opens an IDE-style completion dropdown
  computed by walking the live clap command tree, so completion never drifts from
  the real parser; history, per-line output flags, and a build-metadata banner.
- Python bindings via PyO3 (`pip install oxis`), abi3 wheels for CPython 3.9+.

**Project**
- Validation harness: `validation/generate_reference.py` regenerates the QuantLib
  reference JSON that the Rust test suites read. QuantLib-Python is needed only to
  regenerate references, never at runtime.
- Documentation: `docs/models.md` (per-model method and validation status),
  `docs/parity.md` (coverage vs RustQuant/QuantLib), `docs/architecture.md`.
- Dual-licensed MIT OR Apache-2.0.

[Unreleased]: https://github.com/jpvich/oxis/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jpvich/oxis/releases/tag/v0.1.0
