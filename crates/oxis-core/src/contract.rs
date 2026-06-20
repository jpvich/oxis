//! The module contract — the fixed shape every OXIS module is built against.
//!
//! OXIS has **two kinds of module**. Both depend on the core and only the core;
//! neither imports another module's internals. The distinction is about *I/O*.
//!
//! # Kind A — Compute modules (pure, I/O-free)
//!
//! Pricing, Greeks, stats, ML *inference*. These are the heart of the validated
//! library. A compute module provides:
//!
//! 1. **A pure core**: plain functions over plain types, e.g.
//!    `fn price(option: &EuropeanOption, market: &MarketData) -> oxis_core::Result<PriceResult>`.
//!    It never touches files, stdout/stderr, the network, the clock, or `clap`.
//!    This is the part validated against QuantLib and wrapped by PyO3.
//! 2. **A result type** that derives [`serde::Serialize`] and implements
//!    [`Tabular`](crate::output::Tabular). The core renders human / JSON / TSV;
//!    modules never format output by hand.
//! 3. **A `run(args, ctx: &RunContext) -> anyhow::Result<()>`** thin CLI wrapper:
//!    parse args → build plain inputs → call the pure core → render via the
//!    output layer. (`anyhow` lives at this edge, not in the pure core.)
//! 4. **A `clap::Args` struct** for the command's flags.
//! 5. **Tests**: unit tests for the pure core, closed-form checks where a formula
//!    exists, and **≥1 QuantLib validation test per pricing model**.
//!
//! Because the core is pure, the *same* function serves the Rust API, the CLI,
//! the REPL, and the Python bindings with no duplicated logic.
//!
//! # Kind B — Service modules (stateful, I/O-bound)
//!
//! Market-data, storage, live AI. These genuinely need I/O, so the contract is
//! honest about it instead of pretending otherwise. A service module provides:
//!
//! 1. **A capability trait** (often `async`), defined in the core where it is a
//!    shared contract — see [`DataSource`](crate::source::DataSource) — so
//!    consumers depend on the capability, not the concrete provider.
//! 2. **A concrete service type** constructed from a [`RunContext`](crate::context::RunContext),
//!    owning its client / cache / config. **All I/O is confined to this crate.**
//! 3. **Result types** that are the typed interchange records from
//!    [`crate::series`] (or implement [`Tabular`](crate::output::Tabular) for CLI
//!    output) — never a provider-specific shape leaking outward.
//! 4. **Tests** that exercise logic against a *mock* implementation of the
//!    capability trait, so they need no network.
//!
//! # Guardrails (enforced by review)
//!
//! - The **core** has no Polars/Arrow, no async runtime, no HTTP.
//! - **Polars** may be used only inside stats/data modules, behind a feature
//!   flag, never in the core or in a compute module's public API.
//! - A module **never** imports another module's internals; shared logic moves to
//!   the core. A service/aggregate module may consume another module's *public
//!   result types* only.
//!
//! The `oxis-pricing` and `oxis-greeks` modules are the reference Kind-A
//! implementations; `oxis-data` is the reference Kind-B implementation.
