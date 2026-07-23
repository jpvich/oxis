//! The output layer: the [`Tabular`] contract plus human / JSON / TSV renderers.
//!
//! Every module's result type implements [`Tabular`] (and derives
//! `serde::Serialize`). The core renders it in the format chosen by
//! [`RunContext`](crate::core::context::RunContext) — modules never format output by
//! hand, and never write to stdout/stderr themselves.
//!
//! A `Tabular` describes a **single record**: [`columns`](Tabular::columns) gives
//! the field names and [`cells`](Tabular::cells) the matching values, in order.

mod human;
mod json;
mod tsv;

use crate::core::context::OutputFormat;

pub use self::json::to_json_value;

/// A single rendered value within a result record.
#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    /// Text.
    Str(String),
    /// A floating-point number (prices, rates, Greeks).
    F64(f64),
    /// An integer (counts, step counts, path counts).
    Int(i64),
    /// A boolean flag.
    Bool(bool),
    /// An absent value (renders as empty text / JSON `null`).
    Null,
}

impl Cell {
    /// Convenience constructor for a string cell.
    pub fn str(s: impl Into<String>) -> Self {
        Self::Str(s.into())
    }
}

impl From<f64> for Cell {
    fn from(v: f64) -> Self {
        Cell::F64(v)
    }
}
impl From<i64> for Cell {
    fn from(v: i64) -> Self {
        Cell::Int(v)
    }
}
impl From<bool> for Cell {
    fn from(v: bool) -> Self {
        Cell::Bool(v)
    }
}
impl From<&str> for Cell {
    fn from(v: &str) -> Self {
        Cell::Str(v.to_owned())
    }
}
impl From<String> for Cell {
    fn from(v: String) -> Self {
        Cell::Str(v)
    }
}
impl<T: Into<Cell>> From<Option<T>> for Cell {
    fn from(v: Option<T>) -> Self {
        match v {
            Some(x) => x.into(),
            None => Cell::Null,
        }
    }
}

/// A column header for a result record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Column {
    /// The field name shown in headers / human labels / JSON keys.
    pub name: &'static str,
}

impl Column {
    /// Construct a column from its name.
    pub const fn new(name: &'static str) -> Self {
        Self { name }
    }
}

/// The contract every module result type implements so the core can render it.
///
/// Implementors return [`columns`](Self::columns) and [`cells`](Self::cells) of
/// equal length and matching order.
pub trait Tabular {
    /// The column headers, in order.
    fn columns(&self) -> Vec<Column>;
    /// The cell values, in the same order as [`columns`](Self::columns).
    fn cells(&self) -> Vec<Cell>;
}

/// Render a result in the requested format, returning the string to print.
///
/// This is the single entry point app edges (CLI/REPL/PyO3) use; they decide
/// where the string goes (stdout), keeping modules I/O-free.
pub fn render(value: &impl Tabular, format: OutputFormat) -> String {
    match format {
        OutputFormat::Human => human::render(value),
        OutputFormat::Json => json::render(value),
        OutputFormat::Tsv => tsv::render(value),
    }
}

#[cfg(test)]
pub(crate) struct Sample;

#[cfg(test)]
impl Tabular for Sample {
    fn columns(&self) -> Vec<Column> {
        vec![
            Column::new("model"),
            Column::new("price"),
            Column::new("steps"),
            Column::new("standard_error"),
        ]
    }
    fn cells(&self) -> Vec<Cell> {
        vec![
            Cell::str("black-scholes"),
            Cell::F64(8.021352),
            Cell::Int(1000),
            Cell::Null,
        ]
    }
}
