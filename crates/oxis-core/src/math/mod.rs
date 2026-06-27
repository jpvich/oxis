//! Numerical primitives shared by the pricing modules.
//!
//! Phase 1 provides the standard Normal distribution (high-accuracy CDF/PDF),
//! one-dimensional root finders (Newton + Brent), a small polynomial
//! least-squares fit (for Longstaff-Schwartz regression), and one-dimensional
//! interpolation (for yield curves). Integration lands alongside the models that
//! need it.

mod distributions;
mod interpolate;
mod regression;
mod solvers;

pub use distributions::{normal_cdf, normal_pdf};
pub use interpolate::{NaturalCubicSpline, linear_interpolate};
pub use regression::poly_least_squares;
pub use solvers::{brent, newton};
