//! TSV rendering: a tab-separated header row followed by one value row.

use super::{Tabular, human::cell_to_string};

/// Render as two tab-separated lines: column names, then values.
pub(super) fn render(value: &impl Tabular) -> String {
    let header = value
        .columns()
        .iter()
        .map(|c| c.name)
        .collect::<Vec<_>>()
        .join("\t");
    let row = value
        .cells()
        .iter()
        .map(cell_to_string)
        .collect::<Vec<_>>()
        .join("\t");
    format!("{header}\n{row}\n")
}

#[cfg(test)]
mod tests {
    use crate::context::OutputFormat;
    use crate::output::{Sample, render as render_format};

    #[test]
    fn header_then_values_tab_separated() {
        let out = render_format(&Sample, OutputFormat::Tsv);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "model\tprice\tsteps\tstandard_error");
        // Null renders as an empty field (trailing tab-empty).
        assert_eq!(lines[1], "black-scholes\t8.021352\t1000\t");
    }
}
