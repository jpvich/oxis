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
mod rng;
mod sample;
mod solvers;

pub use distributions::{normal_cdf, normal_pdf};
pub use interpolate::{NaturalCubicSpline, linear_interpolate};
pub use regression::poly_least_squares;
pub use rng::{path_seed, splitmix64};
pub use sample::{mean_and_se, sample_mean_var};
pub use solvers::{brent, newton};
