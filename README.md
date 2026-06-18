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

## Status — Phase 1 (in progress)

Phase 1 establishes a usable, validated pricing core. Target surface:

| Capability | Reference | Status |
|---|---|---|
| Black-Scholes European (closed-form) | QuantLib `AnalyticEuropeanEngine` | Planned |
| Binomial CRR (European + American) | QuantLib `BinomialVanillaEngine` | Planned |
| Monte Carlo European | Black-Scholes | Planned |
| Longstaff-Schwartz (American MC) | QuantLib `MCAmericanEngine` + binomial | Planned |
| Analytic Greeks | QuantLib `AnalyticEuropeanEngine` | Planned |
| Implied volatility solver | round-trip + QuantLib | Planned |

Per-model method and validation status are tracked in [`docs/models.md`](docs/models.md), and a living capability-coverage matrix in [`docs/parity.md`](docs/parity.md).

## Planned usage

### CLI

```bash
# European call via Black-Scholes
oxis price --spot 100 --strike 105 --rate 0.05 --vol 0.2 --t 1.0 --type call

# American put via binomial tree
oxis price --spot 100 --strike 105 --rate 0.05 --vol 0.2 --t 1.0 \
           --type put --style american --model binomial --steps 1000

# Greeks, with JSON output for piping
oxis greeks --spot 100 --strike 105 --rate 0.05 --vol 0.2 --t 1.0 --type call --json
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
print(g.delta, g.gamma, g.vega, g.theta, g.rho)
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

OXIS is a **stable core** plus a **module layer** that grows. The dependency direction is one-way — **module → core only**; a module never imports another module's internals, and shared logic belongs in the core. Every module's logic lives in a pure, I/O-free core; the CLI `run()`, the REPL, and the PyO3 bindings are thin wrappers around that *same* core.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the module contract every module implements, and `docs/` for per-model and architectural notes as modules land.

## Validation

Validation against QuantLib — the industry-standard reference — is at the heart of OXIS:

1. `validation/generate_reference.py` uses **QuantLib-Python** to price a realistic, edge-case-rich parameter grid for each model and writes the results to `validation/reference_data/<model>.csv` (checked into the repo).
2. Rust validation tests load each reference CSV, run the corresponding OXIS pure core on the same inputs, and assert agreement within a documented tolerance.
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
