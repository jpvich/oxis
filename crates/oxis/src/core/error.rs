//! The library error type, [`OxisError`].
//!
//! The core uses `thiserror` for a structured, domain-specific error type — never
//! `anyhow` (that belongs at app edges such as the CLI). Variants are named after
//! the failure, not the call site, so callers can match on them.

use thiserror::Error;

/// The error type returned by OXIS core and compute modules.
///
/// Keep variants domain-specific. Add new variants as new domains land rather
/// than overloading a generic catch-all.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OxisError {
    /// An input was outside its valid domain (e.g. negative volatility, a strike
    /// of zero where it is not allowed). `what` names the offending input.
    #[error("invalid input: {what}")]
    InvalidInput {
        /// Human-readable description of what was invalid.
        what: String,
    },

    /// A numerical routine failed to converge or produced a non-finite result.
    #[error("numerical error: {0}")]
    Numerical(String),

    /// A capability has been declared but is not yet implemented (used by
    /// skeleton modules whose ring has not landed).
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),

    /// A *service* module (e.g. market-data) failed to retrieve or parse data.
    /// This variant is the boundary type for I/O-bound modules; the underlying
    /// cause is flattened to a string so the core stays free of I/O-crate types.
    #[error("data source error: {0}")]
    DataSource(String),
}

impl OxisError {
    /// Construct an [`OxisError::InvalidInput`] from anything string-like.
    pub fn invalid_input(what: impl Into<String>) -> Self {
        Self::InvalidInput { what: what.into() }
    }

    /// Construct an [`OxisError::Numerical`] from anything string-like.
    pub fn numerical(msg: impl Into<String>) -> Self {
        Self::Numerical(msg.into())
    }

    /// Construct an [`OxisError::DataSource`] from anything string-like.
    pub fn data_source(msg: impl Into<String>) -> Self {
        Self::DataSource(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_are_lowercase_and_prefixed() {
        // CLI convention: `error: <message>` with a lowercase message.
        assert_eq!(
            OxisError::invalid_input("volatility must be >= 0").to_string(),
            "invalid input: volatility must be >= 0"
        );
        assert_eq!(
            OxisError::NotImplemented("binomial pricing").to_string(),
            "not implemented: binomial pricing"
        );
    }
}
