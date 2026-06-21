//! Numerical primitives shared by the pricing modules.
//!
//! Phase 1 provides the standard Normal distribution (high-accuracy CDF/PDF),
//! one-dimensional root finders (Newton + Brent), and a small polynomial
//! least-squares fit (for Longstaff-Schwartz regression). Interpolation and
//! integration land alongside the models that need them.

mod distributions;
mod regression;
mod solvers;

pub use distributions::{normal_cdf, normal_pdf};
pub use regression::poly_least_squares;
pub use solvers::{brent, newton};
