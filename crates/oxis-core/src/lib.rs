//! # oxis-core
//!
//! The stable core of OXIS — the platform that every OXIS module builds against.
//!
//! It will provide the fixed contracts shared across modules: financial types
//! (money, rates, dates, options), high-accuracy distributions and numerical
//! methods, market-data types, the error type, and the output layer that renders
//! results as human-readable text, JSON, or TSV.
//!
//! This crate is under active development. See <https://github.com/jpvich/oxis>.

#![forbid(unsafe_code)]

#[cfg(test)]
mod tests {
    /// Smoke test: the workspace compiles and the test harness runs.
    /// Real unit and QuantLib-validation tests land with the core modules.
    #[test]
    fn workspace_builds() {
        assert_eq!(2 + 2, 4);
    }
}
