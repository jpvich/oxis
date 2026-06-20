# Contributing to OXIS

Thanks for your interest in OXIS. This project has one defining quality bar — **no pricing model is "done" until it is validated against QuantLib (or a closed form) within a documented tolerance** — and a deliberately simple way to grow: a stable core plus a catalog of self-contained modules. This guide explains how to add to it.

## Ground rules

- **Correctness > features > speed.** A fast wrong price is a liability. Optimize only after correctness is proven and tested.
- **Pure-core principle (compute modules).** A *compute* module's logic lives in a pure, I/O-free core: plain functions over plain types, never touching files / stdout / stderr / network / `clap`. The CLI `run()`, the REPL, and the PyO3 bindings are thin wrappers around the *same* core. I/O-bound work belongs in a *service* module instead (see "the two module kinds" below).
- **One-way dependencies.** Module → core only. A module never imports another module's internals; shared logic belongs in the core. An *aggregate* module may consume another module's public result types, never its internals.
- **Lean core.** No Polars/Arrow, no async runtime, no HTTP in `oxis-core`. Polars is opt-in and local to the stats/data modules behind a feature flag.
- **Consistency is the source of truth.** Follow the shape and conventions of the existing modules; if a pattern seems wrong, flag it in an issue rather than introducing a competing one.

## The two module kinds

A module is the unit of contribution. OXIS has **two kinds of module**, distinguished by their relationship to I/O. Both depend on the core and only the core. The authoritative contract lives in [`oxis_core::contract`](crates/oxis-core/src/contract.rs); the summary:

### Kind A — Compute modules (pure, I/O-free)

Pricing, Greeks, stats, ML *inference*. To add one, implement against the stable core and touch nothing else:

1. **A pure, I/O-free core** — plain functions over plain types (e.g. `price(option, market, model) -> oxis_core::Result<PriceResult>`). This is the part validated against QuantLib and wrapped by Python.
2. **A result type** that derives `serde::Serialize` and implements `output::Tabular`. The core renders human / TSV / JSON automatically — modules never format output by hand.
3. **A `run(args, ctx: &RunContext) -> anyhow::Result<()>`** thin wrapper for the CLI: parse args → build plain inputs → call the pure core → print via the output layer. (`anyhow` lives only at this edge.)
4. **A `clap::Args` struct** for the command's flags and inputs.
5. **Tests** — unit tests for the pure core, closed-form checks where a formula exists, and **at least one validation test against QuantLib for every pricing model**.

The `pricing` and `greeks` modules are the reference Kind-A implementations — mirror their shape.

### Kind B — Service modules (stateful, I/O-bound)

Market-data, storage, live AI. These need I/O, so they are honest about it:

1. **A capability trait**, defined in the core where it is a shared contract (e.g. [`DataSource`](crates/oxis-core/src/source.rs)), so consumers depend on the capability rather than the concrete provider.
2. **A concrete service type** built from a `RunContext`, owning its client / cache / config — **all I/O confined to the crate.**
3. **Result types** that are the typed interchange records from `oxis_core::series` (or implement `Tabular` for CLI output) — never a provider-specific shape leaking out.
4. **Tests** against a *mock* implementation of the capability trait, so they need no network.

`oxis-data` is the reference Kind-B implementation.

Any module **must not**: import another module's internals, write progress to stdout (use stderr), bypass the output layer, or pull Polars/async-runtime/HTTP into the core.

## Numerical correctness requirements (non-negotiable)

- The standard Normal CDF must be high-accuracy (`erf`-based, ~1e-15), not a crude approximation.
- Every model handles σ=0 and T=0 as correct mathematical limits — never `NaN` / `Inf` / panic. Handle deep ITM/OTM and very high σ explicitly.
- Monte Carlo returns a standard error; never report an MC price without its uncertainty.
- Greeks: analytic closed-form where it exists; finite-difference fallback must document its bump size.
- Put-call parity holds for European options (tested); American price ≥ corresponding European price (tested).
- Every model documents its method, assumptions, and validation status in [`docs/models.md`](docs/models.md).

## Setting up your environment

You can develop locally or in the cloud — pick one, you don't need both:

- **Local.** Install [`rustup`](https://rustup.rs/) and clone the repo. The pinned
  toolchain and components (`rustfmt`, `clippy`) are declared in
  `rust-toolchain.toml`, so rustup installs the right versions automatically on
  your first `cargo` command — no manual setup. Optionally install
  [`just`](https://github.com/casey/just) (`cargo install just`) to use the task
  shortcuts below.
- **Codespaces / Dev Container.** Click *Code → Codespaces → Create* on GitHub (or
  "Reopen in Container" in VS Code) to get a ready-made environment with Rust,
  Python, and maturin already installed, via `.devcontainer/`. Zero local setup.

Either way, the same `rust-toolchain.toml` pins the compiler, so your local checks
match CI exactly.

### Task shortcuts (`just`)

```bash
just            # list recipes
just check      # the full CI gate: fmt-check + clippy + build + test
just fmt        # apply rustfmt
```

`just check` runs the identical steps CI enforces, so run it before opening a PR.

## Development workflow

1. **Open an issue first** for anything non-trivial, so the design can be discussed before code is written.
2. **Branch from `dev`.** PRs target the `dev` branch by default.
3. **Keep PRs focused** — one logical change per PR.
4. **Write tests as you go.** A model without a passing validation test is not merged.
5. **Update the docs that travel with the code:** [`docs/models.md`](docs/models.md) (method + validation status) and [`docs/parity.md`](docs/parity.md) (the parity matrix) when a module lands.

### Before you open a PR, everything must be green:

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

Commit `Cargo.lock`.

## Validation

If you add or change a pricing model, regenerate and check in the QuantLib reference data, and add the corresponding Rust validation test:

```bash
cd validation && pip install -r requirements.txt && python generate_reference.py
```

QuantLib-Python is a validation-time dependency only — never a runtime dependency of OXIS.

## Commit messages

- Write clear, conventional commit messages that describe the change — nothing else.
- **Do not** add `Co-Authored-By` trailers, tool attributions, or any "generated with" / assistant references to commit messages, PR descriptions, code comments, or docs.

## Code style & conventions

- `f64` throughout for prices and rates.
- Errors to stderr as `error: <message>` (lowercase); non-zero exit on error.
- Public APIs are named after the finance concept, not the implementation detail.
- Minimal, justified dependencies; prefer pure-Rust implementations so the binary stays portable. A new dependency requires justification.
- Prefer composition over inheritance; keep responsibilities single and clear.

## Licensing of contributions

By contributing, you agree that your contributions will be dual licensed under **MIT OR Apache-2.0** (inbound = outbound), without any additional terms or conditions and with no CLA required. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).

## Code of conduct

Be respectful and constructive. Harassment or abusive behavior is not tolerated. Report concerns to the maintainers via a private channel (see [SECURITY.md](SECURITY.md) for the contact address).
