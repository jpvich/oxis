//! JSON rendering: a `Tabular` record becomes a JSON object `{ name: value }`.

use super::{Cell, Tabular};
use serde_json::{Map, Number, Value};

/// Build a [`serde_json::Value`] object from a `Tabular` record. Exposed so app
/// edges (e.g. PyO3) can embed the value rather than re-parse a string.
pub fn to_json_value(value: &impl Tabular) -> Value {
    let mut map = Map::new();
    for (col, cell) in value.columns().iter().zip(value.cells().iter()) {
        map.insert(col.name.to_owned(), cell_to_value(cell));
    }
    Value::Object(map)
}

/// Render as pretty-printed JSON.
pub(super) fn render(value: &impl Tabular) -> String {
    // Pretty by default for human-piped readability; callers wanting compact
    // output can use `to_json_value` and serialize themselves.
    serde_json::to_string_pretty(&to_json_value(value)).unwrap_or_else(|_| "{}".to_owned())
}

fn cell_to_value(cell: &Cell) -> Value {
    match cell {
        Cell::Str(s) => Value::String(s.clone()),
        // Non-finite f64 (NaN/Inf) has no JSON representation; emit null rather
        // than panic. OXIS models must not produce these, but render defensively.
        Cell::F64(v) => Number::from_f64(*v)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        Cell::Int(v) => Value::Number(Number::from(*v)),
        Cell::Bool(v) => Value::Bool(*v),
        Cell::Null => Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use crate::core::context::OutputFormat;
    use crate::core::output::{Sample, render as render_format};

    #[test]
    fn renders_object_with_typed_values() {
        let out = render_format(&Sample, OutputFormat::Json);
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["model"], "black-scholes");
        assert_eq!(parsed["price"], 8.021352);
        assert_eq!(parsed["steps"], 1000);
        assert!(parsed["standard_error"].is_null());
    }
}
