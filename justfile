# OXIS task runner — https://github.com/casey/just
# Install: `cargo install just` (or `brew install just`). Run `just` to list recipes.

# List available recipes.
default:
    @just --list

# Run the full CI gate locally: format check, clippy, build, test.
check: fmt-check clippy build test

# Build the whole workspace.
build:
    cargo build --workspace

# Run all tests in the workspace.
test:
    cargo test --workspace

# Lint with clippy, denying warnings (matches CI exactly).
clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Check formatting without modifying files (matches CI).
fmt-check:
    cargo fmt --all --check

# Apply rustfmt to the whole workspace.
fmt:
    cargo fmt --all

# Build and install the Python bindings into the active environment (needs maturin).
# (Available once the `python/` PyO3 crate exists.)
py-dev:
    cd python && maturin develop

# Regenerate the QuantLib reference data (needs QuantLib-Python in validation/).
# (Available once the `validation/` tooling exists.)
gen-reference:
    cd validation && python generate_reference.py

# Dry-run the crates.io publish: packages `oxis` and compiles the packaged copy.
# Publishing itself is done by pushing a `v*` tag (see .github/workflows/release.yml).
publish-check:
    cargo publish -p oxis --locked --dry-run

# Everything a release tag will be checked against, locally.
release-check: check publish-check
