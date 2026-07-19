# OXIS

[![CI](https://github.com/jpvich/oxis/actions/workflows/ci.yml/badge.svg)](https://github.com/jpvich/oxis/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/oxis.svg)](https://crates.io/crates/oxis)
[![Downloads](https://img.shields.io/crates/d/oxis.svg)](https://crates.io/crates/oxis)
[![Docs.rs](https://docs.rs/oxis/badge.svg)](https://docs.rs/oxis)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org/)
[![Dependencies](https://deps.rs/repo/github/jpvich/oxis/status.svg)](https://deps.rs/repo/github/jpvich/oxis)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

**Open eXtensible Instruments & Statistics** — a modular, validated quantitative finance library written in Rust.

OXIS is built to be used four ways from a single validated core: as a **Rust crate**, a **Python package** (via PyO3), a **scriptable CLI**, and an **interactive terminal REPL**. **Every pricing model is validated against [QuantLib](https://www.quantlib.org/) — the industry-standard reference — (or a closed form, or numpy/scipy for statistics) to a documented numerical tolerance.**

> The name nods to both its foundation and its character: *oxidation* (the Rust ecosystem) and the Greek root *oxys* (ὀξύς, "sharp, precise") — precision being the whole point of a pricing library.

> [!WARNING]
> OXIS is in active development. APIs are unstable and may change without notice until the first tagged release.

## Why OXIS

Quantitative pricing code is only trustworthy if it is validated — a plausible-but-wrong price is worse than no price. OXIS is designed from the start to be a **validated, professional-grade, ergonomic** quant library that works seamlessly from Rust, Python, the command line, and an interactive terminal.

- **Correctness first.** Every pricing function is validated against a known reference (closed-form where one exists, otherwise QuantLib) before it is considered done. *A model without a passing validation test is not merged.*
- **Numerical rigor.** `f64` throughout; high-accuracy `erf`-based Normal CDF (~1e-15); explicit, correct handling of edge cases (σ=0, T=0, deep ITM/OTM, very high σ) — never `NaN`/`Inf`/panic. Monte Carlo always reports a standard error.
- **Four interfaces, one core.** All logic lives in a pure, I/O-free core that the Rust API, Python bindings, CLI, and REPL all wrap — no duplicated pricing logic.
- **Hand-rolled where it matters.** The ML engines are written from scratch over the core's linear algebra — no `candle`/`burn`/`tch` — so the binary stays portable and every number is auditable.
- **Machine-readable output everywhere.** Human-readable on a TTY; `--json` / `--tsv` when piped.

## Status

Every row below is validated against its reference in CI (Linux + macOS).

| Capability | Module | Reference | Status |
|---|---|---|---|
| Black-Scholes European (closed-form) | `oxis::pricing` | QuantLib `AnalyticEuropeanEngine` | ✅ |
| Binomial CRR (European + American) | `oxis::pricing` | QuantLib `BinomialVanillaEngine` | ✅ |
| Monte Carlo European | `oxis::pricing` | Black-Scholes | ✅ |
| Longstaff-Schwartz (American MC) | `oxis::pricing` | QuantLib `MCAmericanEngine` + binomial | ✅ |
| Exotics — barrier, lookback, Asian | `oxis::pricing` | QuantLib analytic / MC | ✅ |
| Analytic Greeks (Δ Γ ν Θ ρ) | `oxis::greeks` | QuantLib `AnalyticEuropeanEngine` | ✅ |
| Implied-volatility solver | `oxis::pricing` | round-trip + QuantLib | ✅ |
| Yield curves / term structures | `oxis::curves` | QuantLib `ZeroCurve` / `DiscountCurve` | ✅ |
| Fixed-rate bonds (price, YTM, duration, convexity) | `oxis::bonds` | QuantLib `FixedRateBond` / `BondFunctions` | ✅ |
| Stochastic processes (GBM, OU, Vasicek, CIR, Merton, Heston) | `oxis::stochastic` | closed-form terminal moments | ✅ |
| Statistics & risk (returns, vol, Sharpe/Sortino, VaR/ES, drawdown, β, JB, ACF) | `oxis::stats` | numpy / scipy / pandas | ✅ |
| Portfolio (valuation, TWR/MWR, allocation, risk, Markowitz) | `oxis::portfolio` | numpy / scipy | ✅ |
| **ML — differential ML (European price + delta)** | `oxis::ml` | Black-Scholes (inference ≤1e-12; trained within bands) | ✅ |
| **ML — Deep LSM (American put)** | `oxis::ml` | QuantLib CRR American tree (within bands) | ✅ |
| **ML — Deep Optimal Stopping (American put)** | `oxis::ml` | QuantLib CRR American tree (within bands) | ✅ |

Per-model method and validation status live in [`docs/models.md`](docs/models.md); a living capability-coverage matrix vs RustQuant/QuantLib is in [`docs/parity.md`](docs/parity.md).

## Install / build

OXIS is a **single `oxis` crate** — the library, the CLI, and the REPL are one
package, and every domain is an internal module (`oxis::pricing`, `oxis::ml`, …).

Not yet published to crates.io / PyPI. Until the first release, depend on it
straight from git, or build from source:

```toml
# Cargo.toml — track the repo until the crates.io release:
oxis = { git = "https://github.com/jpvich/oxis" }
```

```bash
# Rust workspace
cargo build --workspace --release        # builds the `oxis` binary at target/release/oxis
cargo test  --workspace                   # run the full validated test suite

# install the CLI/REPL binary onto your PATH
cargo install --path crates/oxis          # provides the `oxis` command

# Python bindings (PyO3 + maturin) — from a virtualenv
pip install maturin
cd python && maturin develop              # builds and installs the `oxis` module
```

### Use OXIS as a Rust library

Add the one crate and reach every module through it (`oxis::pricing`,
`oxis::ml`, …):

```toml
# Cargo.toml — the whole library:
oxis = "0.1"
# …or only the modules you need (drops the CLI/REPL deps too):
oxis = { version = "0.1", default-features = false, features = ["pricing", "ml"] }
```

```rust
use oxis::core::{EuropeanOption, MarketData, OptionType};
use oxis::pricing::black_scholes;

let market = MarketData::new(100.0, 0.05, 0.2, 0.0);
let option = EuropeanOption { strike: 100.0, expiry_years: 1.0, option_type: OptionType::Call };
let price = black_scholes(&option, &market).unwrap();
```

Per-module features: `pricing`, `greeks`, `curves`, `bonds`, `stochastic`,
`stats`, `portfolio`, `ml` (`full` enables all; `cli` adds the binary, on by
default).

Toolchain: Rust ≥ 1.85 (edition 2024). QuantLib-Python is **only** needed to *regenerate* validation reference data, never at runtime.

## Usage

### Interactive REPL

Run `oxis` with no subcommand to open the REPL. It opens with a banner (logo, version/build metadata, and live command/module counts from the parser), then takes the same commands as the CLI (without the leading `oxis`). It has history (↑/↓) and per-line output flags. `help` prints the full command listing; `quit`, `exit`, or Ctrl-D leaves.

**Tab** opens an IDE-style completion dropdown under the cursor (powered by [`reedline`](https://github.com/nushell/reedline)): navigate with ↑/↓ and press Enter to accept. Each entry carries its clap help text as a description. The candidates are computed by walking the clap command tree at the cursor, so they track the real parser at any depth: top-level commands and REPL builtins for the first word (`pri`↹ → `price`), a command's nested subcommands next (`ml `↹ → `american · price`), and its long flags — including the global `--json`/`--tsv`/`--quiet`/`--verbose` — after that (`ml american --met`↹ → `--method`). Completion never drifts from the real commands.

```text
$ oxis
   ██████╗  ██╗  ██╗ ██╗ ███████╗
  ██╔═══██╗ ╚██╗██╔╝ ██║ ██╔════╝
  ██║   ██║  ╚███╔╝  ██║ ███████╗
  ██║   ██║  ██╔██╗  ██║ ╚════██║
  ╚██████╔╝ ██╔╝ ██╗ ██║ ███████║
   ╚═════╝  ╚═╝  ╚═╝ ╚═╝ ╚══════╝

  Open eXtensible Instruments & Statistics
  ────────────────────────────────────────────────────
  v0.0.0   ·   built 2026-06-29   ·   52d8809
  validated quantitative finance, in Rust
  10 commands · 8 modules — mirror the CLI (drop the `oxis`)
  github.com/jpvich/oxis
  ⇥ tab opens the completion menu · type `help` · `quit` to exit

oxis> price --spot 100 --strike 100 --rate 0.05 --vol 0.2 --t 1 --type call
price: 10.45058357218555
oxis> --json greeks --spot 100 --strike 100 --rate 0.05 --vol 0.2 --t 1 --type call
{ "delta": 0.6368..., "gamma": 0.0187..., ... }
oxis> ml american --method dos --spot 100 --strike 100 --rate 0.05 --vol 0.3 --maturity 1
oxis> quit
```

### CLI

```bash
# European call via Black-Scholes; American put via binomial tree
oxis price --spot 100 --strike 105 --rate 0.05 --vol 0.2 --t 1 --type call
oxis price --spot 100 --strike 105 --rate 0.05 --vol 0.2 --t 1 \
           --type put --style american --model binomial --steps 1000

# Greeks (JSON for piping), and the implied-vol solver
oxis greeks --spot 100 --strike 105 --rate 0.05 --vol 0.2 --t 1 --type call --json
oxis implied-vol --price 10.45 --spot 100 --strike 100 --rate 0.05 --t 1 --type call

# Exotics: barrier / lookback / Asian
oxis exotic --kind barrier --spot 100 --strike 100 --rate 0.05 --vol 0.2 --t 1 \
            --type call --barrier 120 --barrier-type up-out

# Yield curve: discount / zero / forward at t=1.5 (natural-cubic interpolation)
oxis curve --times 0.5,1,2,5 --rates 0.02,0.025,0.03,0.035 \
           --interp natural-cubic --at 1.5 --forward-to 2.5

# Fixed-rate bond priced from a flat yield: price, YTM, duration, convexity
oxis bond --coupon 0.05 --maturity 10 --face 100 --frequency 2 --yield 0.04

# Simulate a stochastic process and report terminal moments
oxis simulate --process heston --x0 100 --t 1 --paths 100000 --seed 42

# Statistics on a return series; Markowitz mean-variance optimization
oxis stats --returns 0.01,-0.02,0.015,0.03 --periods-per-year 252
oxis portfolio optimize --mean 0.10,0.15 --cov-row 0.04,0.01 --cov-row 0.01,0.09 \
                --flavor min-variance

# ML pricing — differential ML (European) and neural American (Deep LSM / DOS)
oxis ml price --spot 100 --strike 100 --rate 0.05 --vol 0.2 --maturity 1 --type call
oxis ml american --method deep-lsm --spot 100 --strike 100 --rate 0.05 --vol 0.3 --maturity 1 --type put
oxis ml american --method dos      --spot 100 --strike 100 --rate 0.05 --vol 0.3 --maturity 1 --type put
```

Global flags `--json` / `--tsv` (before or after the subcommand) switch the output format; default is human-readable. Run `oxis <command> --help` for every flag.

### Python

```python
import oxis

# Closed-form price + Greeks
price = oxis.black_scholes(spot=100, strike=105, rate=0.05, vol=0.2, t=1.0,
                           dividend_yield=0.0, option_type="call")
g = oxis.greeks(spot=100, strike=105, rate=0.05, vol=0.2, t=1.0, option_type="call")

# American pricing: binomial tree, classical LSM, and the neural engines
tree = oxis.binomial(spot=100, strike=100, rate=0.05, vol=0.3, t=1.0,
                     option_type="put", style="american", steps=1000)
lsm  = oxis.lsm(spot=100, strike=100, rate=0.05, vol=0.3, t=1.0, option_type="put")
dl   = oxis.american_ml(spot=100, strike=100, rate=0.05, vol=0.3, maturity=1.0,
                        option_type="put", method="deep-lsm")
dos  = oxis.american_ml(spot=100, strike=100, rate=0.05, vol=0.3, maturity=1.0,
                        option_type="put", method="dos")
print(dl["ml_price"], dl["binomial_price"], dl["abs_err"])

# Differential-ML European price + delta vs Black-Scholes
ml = oxis.differential_ml(spot=100, strike=100, rate=0.05, vol=0.2, maturity=1.0,
                          option_type="call")

# Yield curve — build once, query discount / zero / forward
curve = oxis.YieldCurve.from_zero_rates([0.5, 1, 2, 5], [0.02, 0.025, 0.03, 0.035],
                                        interp="natural-cubic")
print(curve.discount(1.5), curve.zero_rate(1.5), curve.forward_rate(1.5, 2.5))

# Statistics
s = oxis.stats(returns=[0.01, -0.02, 0.015, 0.03], periods_per_year=252)
```

### Rust

```rust
use oxis::core::{EuropeanOption, MarketData, OptionType};
use oxis::pricing::black_scholes;

let option = EuropeanOption { strike: 105.0, expiry_years: 1.0, option_type: OptionType::Call };
let market = MarketData { spot: 100.0, rate: 0.05, volatility: 0.2, dividend_yield: 0.0 };
let price = black_scholes(&option, &market)?;
```

Every module is reached through the one crate: `oxis::core`, `oxis::pricing`, `oxis::greeks`, `oxis::curves`, `oxis::bonds`, `oxis::stochastic`, `oxis::stats`, `oxis::portfolio`, `oxis::ml`.

## Architecture

OXIS ships as a **single `oxis` crate**, but inside it is a **stable core** plus a **module layer** that grows. The dependency direction is one-way — **module → core only**; a module never imports another module's internals, and shared logic belongs in the core. Each module is published behind a Cargo feature and re-exported as `oxis::<module>`, so the internal crate split stays an implementation detail consumers never name.

Modules come in **two kinds**:

- **Compute modules** (pricing, greeks, curves, bonds, exotics, stochastic, stats, portfolio, ML) are pure and I/O-free — the CLI `run()`, the REPL, and the PyO3 bindings are thin wrappers around that *same* core.
- **Service modules** (market data and, later, storage / live AI) are stateful and do I/O behind a trait defined in the core, confined to their own crate.

The core stays lean and runtime-agnostic on purpose (no Polars/Arrow, no async runtime, no HTTP); heavy machinery is opt-in and local to the module that needs it.

Capability grows in **rings**: **Ring 1** the validated pricing core; **Ring 2** breadth (curves, fixed income, exotics, stochastic processes); **Ring 3** statistics, risk & portfolio; **Ring 4** the differentiating ML-based pricing (differential ML + neural American). A market-data API follows long-term.

See [docs/architecture.md](docs/architecture.md) for the full map, [CONTRIBUTING.md](CONTRIBUTING.md) for the module contract, and [docs/parity.md](docs/parity.md) / [docs/models.md](docs/models.md) for coverage and per-model validation status.

## Validation

Validation is at the heart of OXIS:

1. `validation/generate_reference.py` uses **QuantLib-Python** (and numpy/scipy/pandas for statistics) to price an edge-case-rich parameter grid for each model and writes the results to `validation/reference/<model>.json` (checked into the repo).
2. Rust tests load each reference JSON, run the corresponding OXIS pure core on the same inputs, and assert agreement within a documented tolerance.
3. ML models use a **two-layer contract** (an approximation can't match to 1e-10): *(a)* the inference math of a fixed-weight network matches numpy to ≤1e-12, and *(b)* the trained model lands within a documented error band of the trusted baseline.
4. QuantLib-Python is **only** a validation-time dependency — never a runtime dependency of OXIS.

```bash
cd validation && pip install -r requirements.txt && python generate_reference.py
```

## Contributing

Contributions are welcome — a module is the unit of contribution. See [CONTRIBUTING.md](CONTRIBUTING.md) for the module contract, the validation requirement, and the workflow. Please also read [SECURITY.md](SECURITY.md) for how to report vulnerabilities.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
