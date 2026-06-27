//! Lot-tracked holdings and cost basis.
//!
//! A [`Holding`] is a position in one symbol, made of one or more [`Lot`]s (each
//! a quantity bought at a unit cost). Money is `f64` throughout — these are
//! accounting *inputs* to analytics validated against a float oracle, not a
//! transaction ledger (see the crate-level note).

use oxis_core::OxisError;
use serde::Serialize;

/// A single purchase lot: `quantity` units acquired at `unit_cost` each.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Lot {
    /// Number of units in this lot.
    pub quantity: f64,
    /// Cost per unit when acquired.
    pub unit_cost: f64,
}

/// A position in one `symbol`, composed of its purchase lots.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Holding {
    /// The instrument identifier (e.g. `"AAPL"`).
    pub symbol: String,
    /// The purchase lots making up the position.
    pub lots: Vec<Lot>,
}

impl Holding {
    /// Construct a holding from a single lot.
    pub fn single(symbol: impl Into<String>, quantity: f64, unit_cost: f64) -> Self {
        Self {
            symbol: symbol.into(),
            lots: vec![Lot {
                quantity,
                unit_cost,
            }],
        }
    }
}

/// Total quantity held across all lots.
pub fn total_quantity(h: &Holding) -> f64 {
    h.lots.iter().map(|l| l.quantity).sum()
}

/// Total cost basis: `Σ quantity·unit_cost` across lots.
pub fn cost_basis(h: &Holding) -> f64 {
    h.lots.iter().map(|l| l.quantity * l.unit_cost).sum()
}

/// Average unit cost: `cost_basis / total_quantity`.
///
/// # Errors
/// [`OxisError::InvalidInput`] if the total quantity is zero (no position).
pub fn average_unit_cost(h: &Holding) -> Result<f64, OxisError> {
    let q = total_quantity(h);
    if q == 0.0 {
        return Err(OxisError::invalid_input(
            "average_unit_cost: zero total quantity",
        ));
    }
    Ok(cost_basis(h) / q)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn aggregates_lots() {
        let h = Holding {
            symbol: "AAPL".into(),
            lots: vec![
                Lot {
                    quantity: 10.0,
                    unit_cost: 150.0,
                },
                Lot {
                    quantity: 5.0,
                    unit_cost: 180.0,
                },
            ],
        };
        assert!((total_quantity(&h) - 15.0).abs() < TOL);
        assert!((cost_basis(&h) - (1500.0 + 900.0)).abs() < TOL);
        assert!((average_unit_cost(&h).unwrap() - 2400.0 / 15.0).abs() < TOL);
    }

    #[test]
    fn empty_position_errors_not_panics() {
        let h = Holding {
            symbol: "X".into(),
            lots: vec![],
        };
        assert_eq!(total_quantity(&h), 0.0);
        assert!(average_unit_cost(&h).is_err());
    }
}
