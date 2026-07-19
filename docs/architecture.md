# OXIS Architecture

OXIS is a **stable core + a growing catalog of modules**. It starts as a
validated option-pricing library and is built to grow — through architecture, not
heroics — into a platform spanning statistics, portfolio & risk, ML-based pricing,
and (long-term) a market-data API. This document is the operational map of how the
pieces fit and the rules that keep them from tangling.

OXIS ships as a **single `oxis` crate**: each domain below is an internal module
(`oxis::pricing`, `oxis::ml`, …) gated behind a Cargo feature of the same name, and
the `core` module is the always-present contract every other builds on. The
boundaries here are a design discipline enforced by the module layout and the
one-way dependency rule — not by crate walls.

## Layered design

```
╔══════════════════════════════════════════════════════════════════╗
║  MODULE LAYER (grows)                                              ║
║                                                                    ║
║  Kind A — compute (pure, I/O-free):                                ║
║    oxis::pricing · oxis::greeks · oxis::stats · oxis::ml (infer.)  ║
║  Kind B — service (stateful, I/O behind a trait):                  ║
║    oxis::data (market data) · (future: storage, live AI)          ║
║  Aggregate:                                                        ║
║    oxis::portfolio (consumes data + pricing + stats)              ║
╚══════════════════════════════════════════════════════════════════╝
                     │ depends on ▼ (one direction only)
┌────────────────────────────────────────────────────────────────────┐
│  STABLE CORE — oxis::core (fixed contracts)                         │
│  • Financial types (option, money, date/day-count)                  │
│  • Market-data inputs (MarketData; curves later)                    │
│  • Typed time-series interchange (Ohlcv, TimeSeries<T>, DateRange)  │
│  • Output layer: Tabular → human / JSON / TSV                       │
│  • Error type (OxisError), RunContext                               │
│  • DataSource trait (the service-module contract)                   │
│  • (Ring 0+) distributions, RNG, root-finding, interpolation        │
└────────────────────────────────────────────────────────────────────┘
```

**Dependency rule — module → core only.** No module imports another module's
internals; shared logic belongs in the core. An *aggregate* module (e.g.
`oxis::portfolio`) may consume another module's **public result types** and the
core's trait contracts — never its internals. The feature graph encodes this: a
module's feature enables exactly the modules it builds on.

## The two module kinds

The single most important design decision. OXIS modules come in two kinds,
distinguished by their relationship to I/O. The full contract for each lives in
[`oxis::core::contract`](../crates/oxis/src/core/contract.rs).

### Kind A — Compute modules (pure, I/O-free)

Pricing, Greeks, stats, ML *inference*. Plain functions over plain types that
never touch the network, disk, the clock, stdout/stderr, or `clap`. This purity is
what lets the *same* function serve the Rust API, the CLI, the REPL, and the Python
bindings with no duplicated logic — and what makes exact validation against
QuantLib possible. Every pricing model in a compute module must have a passing
validation test before it merges.

### Kind B — Service modules (stateful, I/O-bound)

Market-data, storage, live AI. These genuinely need I/O, so the contract is honest
about it rather than forcing an awkward "pure" shape. A service module implements a
**capability trait defined in the core** (e.g.
[`DataSource`](../crates/oxis/src/core/source.rs)), owns its client/cache/config,
and **confines all I/O to its own module** (its runtime/HTTP deps behind that
module's feature flag, off by default). Consumers depend on the *trait*, not the
concrete provider — so the market-data ring can land last without anything waiting
on it (dependency inversion).

## Guardrails

These keep the lean, validated character of the project intact as it grows:

- **The core stays lean and runtime-agnostic:** no Polars/Arrow, no async runtime,
  no HTTP in `oxis::core`. Its only always-on deps are serde/serde_json/thiserror;
  everything heavier is pulled in by a specific module's feature, so a
  `default-features = false` build with just `core` compiles none of it.
- **Polars is opt-in and local.** Heavy columnar machinery may be used *inside* the
  stats/data modules behind a Cargo feature flag — never in the core or in a
  compute module's public API. The typed structs in
  [`oxis::core::series`](../crates/oxis/src/core/series.rs) are the universal
  interchange that crosses module boundaries; Polars is an internal compute detail.
- **`f64` for pricing; decimal for accounting.** Prices/rates are `f64`. The
  portfolio ring introduces decimal-precise money for accounting on top of the
  core types.
- **Output only through the layer.** Modules return `Tabular` result types; only
  app edges (CLI/REPL/PyO3) write to stdout. Errors go to stderr as
  `error: <message>` (lowercase).

## Growth in rings

Breadth is earned incrementally; each ring begins once the previous is solid.

| Ring | Scope | Modules | Status |
|---|---|---|---|
| **1** | Validated pricing core | `core`, `pricing`, `greeks` (+ the CLI/REPL) | in progress |
| **2** | Breadth: exotics, term structures, bonds, more processes | `curves`, `bonds`, `stochastic` (extend pricing/core) | planned |
| **3** | Risk & portfolio; statistics & time-series | `stats`, `portfolio` | reserved (skeleton) |
| **4** | ML-based pricing (the differentiator) | `ml` | reserved (skeleton) |
| **long-term** | Market-data API | `data` | reserved (skeleton) |

The reserved modules exist today as **skeletons carrying their boundary contract**
(not empty placeholders): each compiles, proves the `module → core` direction, and
defines the trait/types its ring will fill. This is the "middle path" — the
platform's shape is fixed now, while only Ring 1 is fleshed out.

## Why this shape (reference lessons)

- **QuantLib** — adopt the Instrument↔Engine separation (contract vs. valuation,
  swappable) and the time/math breadth as a target; avoid its global evaluation
  date, observer web, and template/build pain.
- **RustQuant** — adopt zero-dep foundation + trait extensibility; avoid its empty
  placeholder crates and the Polars-everywhere `data` crate that became a coupling
  bottleneck.
- **akshare** — for the market-data ring: typed records + a provider-agnostic
  `DataSource` trait + caching/rate-limiting, instead of fragile untyped scrapers.
- **wealthfolio** — trait-first core, leaf service crates plugged in at the edge,
  decimal money, daily-snapshot caching for the portfolio ring.
