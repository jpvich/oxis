# OXIS

**Open eXtensible Instruments & Statistics** — a modular, validated quantitative finance library written in Rust.

OXIS is built to be used four ways from a single validated core: as a **Rust crate**, a **Python package** (via PyO3), a **scriptable CLI**, and an **interactive terminal REPL**. **Every pricing model is validated against [QuantLib](https://www.quantlib.org/), the industry-standard reference, to a documented numerical tolerance.**

> The name nods to both its foundation and its character: *oxidation* (the Rust ecosystem) and the Greek root *oxys* (ὀξύς, "sharp, precise") — precision being the whole point of a pricing library.

> [!WARNING]
> OXIS is in early development (Phase 1). APIs are unstable and may change without notice until the first tagged release.

## Why OXIS

Quantitative pricing code is only trustworthy if it is validated — a plausible-but-wrong price is worse than no price. OXIS is designed from the start to be a **validated, professional-grade, ergonomic** quantitative finance library in Rust that works seamlessly from Rust, Python, the command line, and an interactive terminal.

It does this with:

- **Correctness first.** Every pricing function is validated against a known reference (closed-form where one exists, otherwise QuantLib) before it is considered done. A model without a passing validation test is not merged.
- **Numerical rigor.** `f64` throughout; high-accuracy `erf`-based Normal CDF (~1e-15); explicit, correct handling of edge cases (σ=0, T=0, deep ITM/OTM, very high σ) — never `NaN`/`Inf`/panic. Monte Carlo always reports a standard error.
- **Four interfaces, one core.** All logic lives in a pure, I/O-free core that the Rust API, Python bindings, CLI, and REPL all wrap — no duplicated pricing logic.
- **Modular and readable.** A stable core plus a growing catalog of self-contained modules built against a fixed contract. New domains arrive as new modules without changing the core.
- **Machine-readable output everywhere.** Human-readable on a TTY; `--json` / `--tsv` when piped.

## Status — Phase 1

Ring 1 is a complete, validated pricing core; Ring 2 has begun with yield curves.
Every row below is validated against its reference in CI:

| Capability | Reference | Status |
|---|---|---|
| Black-Scholes European (closed-form) | QuantLib `AnalyticEuropeanEngine` | ✅ Validated (Ring 1) |
| Binomial CRR (European + American) | QuantLib `BinomialVanillaEngine` | ✅ Validated (Ring 1) |
| Monte Carlo European | Black-Scholes | ✅ Validated (Ring 1) |
| Longstaff-Schwartz (American MC) | QuantLib `MCAmericanEngine` + binomial | ✅ Validated (Ring 1) |
| Analytic Greeks | QuantLib `AnalyticEuropeanEngine` | ✅ Validated (Ring 1) |
| Implied volatility solver | round-trip + QuantLib | ✅ Validated (Ring 1) |
| Yield curves / term structures | QuantLib `ZeroCurve` / `DiscountCurve` | ✅ Validated (Ring 2) |

Per-model method and validation status are tracked in [`docs/models.md`](docs/models.md), and a living capability-coverage matrix in [`docs/parity.md`](docs/parity.md).

## Usage

### CLI

```bash
# European call via Black-Scholes
oxis price --spot 100 --strike 105 --rate 0.05 --vol 0.2 --t 1.0 --type call

# American put via binomial tree
oxis price --spot 100 --strike 105 --rate 0.05 --vol 0.2 --t 1.0 \
           --type put --style american --model binomial --steps 1000

# Greeks, with JSON output for piping
oxis greeks --spot 100 --strike 105 --rate 0.05 --vol 0.2 --t 1.0 --type call --json

# Yield curve: discount / zero / forward at t=1.5 (natural-cubic interpolation)
oxis curve --times 0.5,1,2,5 --rates 0.02,0.025,0.03,0.035 \
           --interp natural-cubic --at 1.5 --forward-to 2.5
```

Running `oxis` with no subcommand on a TTY opens an interactive REPL with completion and history.

### Python

```python
import oxis

price = oxis.black_scholes(
    spot=100, strike=105, rate=0.05, vol=0.2, t=1.0,
    dividend_yield=0.0, option_type="call",
)

g = oxis.greeks(spot=100, strike=105, rate=0.05, vol=0.2, t=1.0, option_type="call")
print(g["delta"], g["gamma"], g["vega"], g["theta"], g["rho"])

# Yield curve — build once, query discount / zero / forward rates
curve = oxis.YieldCurve.from_zero_rates(
    [0.5, 1, 2, 5], [0.02, 0.025, 0.03, 0.035], interp="natural-cubic",
)
print(curve.discount(1.5), curve.zero_rate(1.5), curve.forward_rate(1.5, 2.5))
```

### Rust

```rust
use oxis_core::{EuropeanOption, MarketData, OptionType};
use oxis_pricing::black_scholes;

let option = EuropeanOption { strike: 105.0, expiry_years: 1.0, option_type: OptionType::Call };
let market = MarketData { spot: 100.0, rate: 0.05, volatility: 0.2, dividend_yield: 0.0 };
let price = black_scholes(&option, &market)?;
```

## Architecture

OXIS is a **stable core** plus a **module layer** that grows. The dependency direction is one-way — **module → core only**; a module never imports another module's internals, and shared logic belongs in the core.

Modules come in **two kinds**:

- **Compute modules** (pricing, greeks, stats, ML inference) are pure and I/O-free — the CLI `run()`, the REPL, and the PyO3 bindings are thin wrappers around that *same* core, and every pricing model is validated against QuantLib.
- **Service modules** (market-data and, later, storage / live AI) are stateful and do I/O behind a trait defined in the core, confined to their own crate — so consumers depend on the *capability*, not a concrete provider.

The core stays lean and runtime-agnostic on purpose (no Polars/Arrow, no async runtime, no HTTP); heavy columnar machinery is opt-in and local to the stats/data modules. This is what lets OXIS start as a validated pricing library and grow — through architecture, not heroics — toward statistics, portfolio & risk, ML-based pricing, and (long-term) a market-data API.

Capability grows in **rings**: Ring 1 is the validated pricing core (in progress); Ring 2 adds breadth (exotics, curves, fixed income); Ring 3 adds risk/portfolio and statistics; Ring 4 adds the differentiating ML-based pricing; a market-data API follows long-term. The later-ring crates already exist as skeletons carrying their boundary contracts.

See [docs/architecture.md](docs/architecture.md) for the full map, [CONTRIBUTING.md](CONTRIBUTING.md) for the two module-kind contracts, and [docs/parity.md](docs/parity.md) / [docs/models.md](docs/models.md) for coverage and per-model validation status.

## Validation

Validation against QuantLib — the industry-standard reference — is at the heart of OXIS:

1. `validation/generate_reference.py` uses **QuantLib-Python** to price a realistic, edge-case-rich parameter grid for each model and writes the results to `validation/reference/<model>.json` (checked into the repo).
2. Rust validation tests load each reference JSON, run the corresponding OXIS pure core on the same inputs, and assert agreement within a documented tolerance.
3. QuantLib-Python is **only** a validation-time dependency — never a runtime dependency of OXIS.

## Building

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check

# Python bindings (PyO3 + maturin)
cd python && maturin develop

# Regenerate QuantLib reference data (needs QuantLib-Python)
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
