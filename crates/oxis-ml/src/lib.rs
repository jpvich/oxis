//! # oxis-ml — ML-based pricing (Ring 4, reserved) — the OXIS differentiator
//!
//! Planned home of validated machine-learning pricing — e.g. neural-network
//! pricing of American options / the early-exercise boundary — measured against
//! the classical engines already in `oxis-pricing`. Model *inference* is a
//! **Kind A** (pure compute) concern; any model *loading / training* I/O is a
//! **Kind B** (service) concern confined behind a trait.
//!
//! This is the capability neither RustQuant nor QuantLib offers; it lands on top
//! of the validated classical core so its accuracy is measurable against a
//! trusted baseline.
//!
//! **Status: reserved skeleton.** No models implemented yet.

#![forbid(unsafe_code)]

#[cfg(test)]
mod tests {
    use oxis_core::{AmericanOption, OptionType};

    /// Proves the `module → core` dependency direction compiles.
    #[test]
    fn builds_against_core_types() {
        let opt = AmericanOption {
            strike: 100.0,
            expiry_years: 1.0,
            option_type: OptionType::Put,
        };
        assert_eq!(opt.option_type, OptionType::Put);
    }
}
