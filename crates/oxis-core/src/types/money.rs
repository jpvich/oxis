//! Lightweight money & currency types.
//!
//! Intentionally minimal for the pricing core (which works in `f64`). When the
//! portfolio ring lands it will introduce decimal-precise accounting on top of
//! these; pricing never needs sub-cent exactness on a model price.

use serde::{Deserialize, Serialize};

/// An ISO-4217-style currency code (e.g. `USD`, `EUR`). Stored as a short code;
/// not validated against the full ISO list yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Currency(pub [u8; 3]);

impl Currency {
    /// Construct from a 3-letter code, upper-casing ASCII letters.
    pub fn new(code: &str) -> Option<Self> {
        let bytes = code.as_bytes();
        if bytes.len() != 3 || !bytes.iter().all(|b| b.is_ascii_alphabetic()) {
            return None;
        }
        let mut out = [0u8; 3];
        for (i, b) in bytes.iter().enumerate() {
            out[i] = b.to_ascii_uppercase();
        }
        Some(Currency(out))
    }

    /// The code as a string slice.
    pub fn code(&self) -> &str {
        // Safe: constructed only from ASCII alphabetic bytes.
        core::str::from_utf8(&self.0).unwrap_or("???")
    }
}

/// An amount in a given currency.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Money {
    /// The numeric amount.
    pub amount: f64,
    /// The currency it is denominated in.
    pub currency: Currency,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn currency_normalizes_and_validates() {
        assert_eq!(Currency::new("usd").unwrap().code(), "USD");
        assert!(Currency::new("US").is_none());
        assert!(Currency::new("US1").is_none());
    }
}
