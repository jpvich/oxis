//! [`RunContext`] — shared run-time configuration threaded through `run()` wrappers.
//!
//! The context is deliberately minimal today (output format + verbosity). It is
//! the extension point that *service* modules will later read for things like a
//! cache directory, an HTTP client configuration, or a model registry — so adding
//! those is a field addition, not a signature churn across every module.

use serde::{Deserialize, Serialize};

/// How a result should be rendered. Selected by global CLI flags (`--json` /
/// `--tsv`); defaults to human-readable on a TTY.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// Aligned, human-readable text (the default on a TTY).
    #[default]
    Human,
    /// Structured JSON, for piping into other tools.
    Json,
    /// Tab-separated values, for spreadsheets and `cut`/`awk`.
    Tsv,
}

/// Shared run-time configuration passed to a module's `run()` wrapper.
///
/// Constructed once at an app edge (CLI/REPL) and borrowed by modules. Extend
/// with new fields as service modules land; keep it cheap to clone.
#[derive(Debug, Clone, Default)]
pub struct RunContext {
    /// The output format to render results in.
    pub format: OutputFormat,
    /// Suppress non-essential output.
    pub quiet: bool,
    /// Emit extra diagnostic output (to stderr).
    pub verbose: bool,
}

impl RunContext {
    /// A context that renders human-readable output with default verbosity.
    pub fn human() -> Self {
        Self::default()
    }

    /// Set the output format (builder-style).
    #[must_use]
    pub fn with_format(mut self, format: OutputFormat) -> Self {
        self.format = format;
        self
    }
}
