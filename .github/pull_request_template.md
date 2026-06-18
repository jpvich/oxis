<!--
Thanks for contributing to OXIS! Please fill out this template.
PRs target the `dev` branch by default. See CONTRIBUTING.md for the module contract.
-->

## Summary

<!-- What does this PR change, and why? Link any related issue: "Closes #123". -->

## Type of change

- [ ] New module (against the fixed module contract)
- [ ] New pricing model / capability in an existing module
- [ ] Bug fix (numerical correctness, edge case, or other)
- [ ] Documentation / tooling / CI
- [ ] Refactor (no behavior change)

## Validation (required for any pricing model)

> No pricing model is "done" until it is validated against QuantLib (or a closed form) within a documented tolerance.

- [ ] Added/updated a validation test against QuantLib reference data, **or** a closed-form check
- [ ] Documented the tolerance and the reference engine used
- [ ] Regenerated `validation/reference_data/` if inputs/grid changed
- [ ] N/A — this PR does not touch a pricing model

## Numerical correctness

- [ ] Edge cases handled as correct limits (σ=0, T=0, deep ITM/OTM, very high σ) — no `NaN`/`Inf`/panic
- [ ] Monte Carlo results report a standard error (if applicable)
- [ ] Put-call parity / American ≥ European invariants hold where relevant
- [ ] N/A

## Checklist

- [ ] Logic lives in the pure, I/O-free core (no files/stdout/stderr/clap in the core)
- [ ] Result type derives `serde::Serialize` and implements `output::Tabular` (no hand-formatted output)
- [ ] `cargo build --workspace` passes
- [ ] `cargo test --workspace` passes
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` is clean
- [ ] `cargo fmt --all --check` is clean
- [ ] `Cargo.lock` committed
- [ ] Updated `docs/models.md` (method + validation status) and `docs/parity.md` if a module landed
- [ ] No competing patterns introduced; follows existing conventions
- [ ] Commit messages describe only the change (no tool/assistant attributions)

## Notes for reviewers

<!-- Anything that needs special attention: tradeoffs, open questions, follow-ups. -->
