# Security Policy

## Supported versions

OXIS is in early development (Phase 1) and has not yet had a tagged release. Until `1.0`, only the latest commit on the `dev` and `main` branches is supported. Security fixes are applied to the latest code; older snapshots are not maintained.

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Report them privately using one of:

- **GitHub Security Advisories** — use the [**Report a vulnerability**](https://github.com/jpvich/oxis/security/advisories/new) button on the repository's *Security* tab (preferred).
- **Email** — `repo_security@themaic.com`. Encrypt if you can; otherwise send a description and we will arrange a secure channel.

Please include, as far as you can:

- a description of the issue and its impact,
- the affected component, version, or commit,
- steps to reproduce or a proof of concept,
- any suggested mitigation.

## What to expect

- **Acknowledgement** of your report within **3 business days**.
- An initial **assessment** within **10 business days**, including whether we accept the report and an expected timeline.
- We will keep you informed of progress and coordinate a disclosure date. We support **coordinated disclosure** and ask that you give us reasonable time to release a fix before any public disclosure.
- With your consent, we will credit you in the advisory and release notes.

## Scope

In scope:

- Memory-safety issues, panics reachable from untrusted input, and `unsafe` misuse in the Rust core or the PyO3 bindings.
- Vulnerabilities in the CLI, REPL, or Python bindings (e.g. input handling, deserialization of `--json` input, file handling).
- Issues in the build, release, or distribution pipeline (crates.io, PyPI wheels, GitHub Releases).
- Vulnerable dependencies that are exploitable through OXIS.

Out of scope (please file these as **regular issues**, not security reports):

- **Numerical-correctness bugs** — a wrong or imprecise price is a correctness defect, not a security vulnerability. These are critical to us, but they are tracked publicly so they can be validated against QuantLib in the open. File them as normal issues with reproduction inputs.
- Theoretical issues without a practical, demonstrated impact.

## Safe harbor

We will not pursue or support legal action against researchers who, in good faith, discover and report vulnerabilities in accordance with this policy and who avoid privacy violations, data destruction, and service disruption while doing so.
