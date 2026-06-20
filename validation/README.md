# OXIS Validation Suite

This directory is the **oracle** side of OXIS's core promise:

> No pricing model is "done" without a validation test against QuantLib (or a
> closed form) within a documented tolerance.

QuantLib is **not** a runtime dependency of OXIS. It is used here, offline, only
to *generate reference prices*. The Rust side (`crates/oxis-pricing/tests/
validation_tests.rs`) reads the committed reference JSON and asserts that the
OXIS price matches the oracle within the recorded tolerance.

## Layout

```
validation/
  generate_reference.py     # QuantLib script that produces reference/*.json
  requirements.txt          # pinned QuantLib version (generation only)
  reference/
    black_scholes.json      # committed reference data (the oracle output)
```

## Regenerating reference data

```bash
cd validation
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python generate_reference.py        # rewrites reference/black_scholes.json
```

Then run the Rust side:

```bash
cargo test -p oxis-pricing --test validation_tests -- --nocapture
```

## How the comparison stays exact

QuantLib derives the year-fraction `t` from dates (Actual/365 Fixed). Rather
than fight date arithmetic, the generator records the **exact `t` QuantLib
used** in each case, and the Rust test prices at that identical continuous time.
Flat curves are built with continuous compounding so the discount factor is
`exp(-r·t)`, matching the OXIS convention. The result: agreement to ~1e-14
(machine precision), far inside the 1e-10 tolerance.

## Current status

| Model                       | Oracle           | Cases  | Max abs error | Tolerance |
| --------------------------- | ---------------- | ------ | ------------- | --------- |
| Black-Scholes-Merton (Euro) | QuantLib 1.42.1  | 22     | ~2.5e-14      | 1e-10     |
| Binomial CRR (Euro+American)| QuantLib 1.42.1  | 44     | ~1.1e-10      | 1e-6      |
| Analytic Greeks (×5)        | QuantLib 1.42.1  | 22     | ~1.0e-13      | 1e-8      |
| Implied volatility          | QuantLib 1.42.1  | 22     | ~1.2e-11      | 1e-6      |

Edge cases (`T=0`, `σ=0`, `S=0`) are validated separately as exact mathematical
limits in the unit tests of `crates/oxis-pricing`, since the analytic engine's
behavior at the boundary is a documented limit rather than an oracle lookup.
