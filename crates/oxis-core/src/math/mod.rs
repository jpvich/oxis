//! Numerical primitives shared by the pricing modules.
//!
//! Phase 1 provides the standard Normal distribution (high-accuracy CDF/PDF)
//! and one-dimensional root finders (Newton + Brent). Interpolation and
//! integration land alongside the models that need them.

mod distributions;
mod solvers;

pub use distributions::{normal_cdf, normal_pdf};
pub use solvers::{brent, newton};
