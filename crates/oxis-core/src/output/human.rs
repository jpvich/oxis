//! Human-readable rendering: aligned `name: value` lines.

use super::{Cell, Tabular};

/// Render as left-aligned `name:` labels followed by values, one per line.
pub(super) fn render(value: &impl Tabular) -> String {
    let columns = value.columns();
    let cells = value.cells();

    // Width of the widest label, so values line up in a column.
    let label_width = columns.iter().map(|c| c.name.len()).max().unwrap_or(0);

    let mut out = String::new();
    for (col, cell) in columns.iter().zip(cells.iter()) {
        // `label:` padded to the widest label + one space.
        let label = format!("{}:", col.name);
        out.push_str(&format!(
            "{label:<width$} {value}\n",
            label = label,
            width = label_width + 1,
            value = cell_to_string(cell),
        ));
    }
    out
}

/// Render a cell as plain text. `f64` uses Rust's shortest round-trippable form
/// (lossless), so prices keep their precision.
pub(super) fn cell_to_string(cell: &Cell) -> String {
    match cell {
        Cell::Str(s) => s.clone(),
        Cell::F64(v) => v.to_string(),
        Cell::Int(v) => v.to_string(),
        Cell::Bool(v) => v.to_string(),
        Cell::Null => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use crate::context::OutputFormat;
    use crate::output::{Sample, render as render_format};

    #[test]
    fn aligns_labels_and_keeps_precision() {
        let out = render_format(&Sample, OutputFormat::Human);
        let lines: Vec<&str> = out.lines().collect();

        // One line per field, label then value.
        assert!(lines[0].starts_with("model:"));
        assert!(lines[0].ends_with("black-scholes"));
        // f64 keeps full precision (not truncated to 4 dp).
        assert!(lines[1].starts_with("price:"));
        assert!(lines[1].ends_with("8.021352"));
        assert!(lines[2].ends_with("1000"));
        // Null renders as empty: the line is just the padded label.
        assert!(lines[3].trim() == "standard_error:");

        // Values are column-aligned: the value starts at the same offset on every
        // line (label padded to the widest label, "standard_error").
        let value_col = |line: &str| line.find("black-scholes").or_else(|| line.find("8.021352"));
        assert_eq!(value_col(lines[0]), value_col(lines[1]));
    }
}
