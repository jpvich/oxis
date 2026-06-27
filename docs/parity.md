# Feature Parity & Coverage Matrix

Validation proves *correctness*; this matrix tracks *coverage* — what OXIS
implements relative to its references. It is the single source of truth for the
"match them, then exceed them" claim, and is updated as each module lands.

Legend: ✅ first-class · ◐ partial / limited (and, for RustQuant, **unvalidated**)
· ❌ absent · 🔜 planned · 🧪 in progress.

## Pricing & Greeks (Ring 1–2)

| Capability | QuantLib | RustQuant | OXIS status |
|---|---|---|---|
| Black-Scholes European (closed-form) | ✅ | ✅ | ✅ Ring 1 — **validated vs QuantLib** (≤2.5e-14) |
| Binomial CRR (European + American) | ✅ | ◐ | ✅ Ring 1 — **validated vs QuantLib** (≤1.1e-10) |
| Monte Carlo European | ✅ | ✅ | ✅ Ring 1 — **validated vs Black-Scholes** (≤4σ; antithetic, reports SE) |
| Longstaff-Schwartz (American MC) | ✅ | ❌ | ✅ Ring 1 — **validated vs QuantLib LSM + binomial** (combined-SE / ≤2.5% bias band) |
| Analytic Greeks | ✅ | ◐ | ✅ Ring 1 — **validated vs QuantLib** (≤1.0e-13) |
| Implied volatility solver | ✅ | ◐ | ✅ Ring 1 — **validated vs QuantLib** (≤1.2e-11) |
| Exotic options (barrier, Asian, lookback) | ✅ | ◐ | ✅ Ring 2 — **validated vs QuantLib** (barrier/lookback/geometric-Asian closed-form ≤1e-13; arithmetic-Asian MC within combined SE) |
| Yield curves / term structures | ✅ | ◐ | ✅ Ring 2 — **validated vs QuantLib** (≤1e-10; linear / log-linear / natural-cubic; discount/zero/forward) |
| Bonds & fixed income | ✅ | ❌ | ✅ Ring 2 — **validated vs QuantLib** (≤1e-8; fixed-rate price/YTM/duration/convexity; curve or yield discounting). Bootstrapping planned. |
| Stochastic process generators | ✅ | ✅ | ✅ Ring 2 — **validated vs closed-form moments** (GBM, OU, Vasicek, CIR, Merton-jump, Heston; reproducible antithetic paths) |

## Platform breadth (Ring 3+ — where OXIS aims past the references)

| Capability | QuantLib | RustQuant | OXIS status |
|---|---|---|---|
| Statistics & time-series analytics | ◐ | ◐ | ✅ Ring 3 (`oxis-stats`) — **validated vs numpy/scipy** (≤1e-10; descriptive, returns, risk-adjusted ratios, drawdown, autocorrelation, Jarque-Bera, beta/TE/IR) |
| VaR / Expected Shortfall | ◐ | ❌ | ✅ Ring 3 (`oxis-stats`) — **validated vs numpy/scipy** (≤1e-10; historical, parametric Gaussian, Cornish-Fisher) |
| Portfolio: holdings / valuation / performance (TWR, MWR) | ❌ | ◐ | ✅ Ring 3 (`oxis-portfolio`) — **validated vs numpy/scipy** (≤1e-10; lot cost basis, mark-to-market, TWR, MWR/IRR, risk aggregation) |
| Portfolio optimization / allocation | ❌ | ◐ | ✅ Ring 3 (`oxis-portfolio`) — **validated vs numpy** (≤1e-10; Markowitz min-variance / tangency / efficient frontier, unconstrained) |
| **ML / neural option pricing** | ❌ | ❌ | 🔜 **Ring 4 — differentiator** (`oxis-ml`) |
| Market-data API (provider-agnostic) | ❌ | ◐ (Yahoo only) | 🔜 long-term (`oxis-data`) |

## Ergonomics & engineering (the consistent edge)

| Dimension | QuantLib | RustQuant | OXIS |
|---|---|---|---|
| Validated against a reference | n/a (is the reference) | ❌ (hobby project) | ✅ **every model, in CI** |
| Rust crate | ❌ | ✅ | ✅ |
| Python package | ◐ (SWIG) | ◐ (partial PyO3) | 🔜 first-class PyO3 |
| Scriptable CLI | ❌ | ❌ | 🔜 |
| Interactive REPL | ❌ | ❌ | 🔜 |
| Machine-readable output (JSON/TSV) | ❌ | ❌ | ✅ (output layer) |
| Lean, portable single binary | ❌ (heavy C++) | ◐ (Polars everywhere) | ✅ (lean core) |

The bottom rows and the ML row are where OXIS is built to be *better*, not just
equal. Breadth vs QuantLib is a long-game target, not a day-one claim.
