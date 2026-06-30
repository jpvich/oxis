---
name: Bug report
about: Report a defect — including an incorrect or imprecise price
title: "[bug] "
labels: bug
assignees: ''
---

## Description

<!-- A clear description of the bug. For a wrong/imprecise price, say what you expected vs. what you got. -->

## Affected component

- [ ] `oxis-core` (types / math / output)
- [ ] `oxis-pricing`
- [ ] `oxis-greeks`
- [ ] `oxis` (facade crate / CLI / REPL)
- [ ] Python bindings
- [ ] Validation / tooling
- [ ] Other:

## Reproduction

<!--
Exact inputs so we can reproduce. For a pricing issue, give every parameter.
e.g. CLI:  oxis price --spot 100 --strike 105 --rate 0.05 --vol 0.2 --t 1.0 --type call
or Python / Rust snippet.
-->

```text
```

## Expected vs. actual

- **Expected:** <!-- value, and the reference (QuantLib engine, closed form, textbook) -->
- **Actual:** <!-- value OXIS produced -->
- **Tolerance/discrepancy:** <!-- absolute or relative difference, if known -->

## Environment

- OXIS version / commit:
- Interface: Rust crate / Python / CLI / REPL
- OS + arch:
- Rust version (`rustc --version`), if building from source:

## Additional context

<!-- Logs, stderr output, screenshots, anything else. -->
