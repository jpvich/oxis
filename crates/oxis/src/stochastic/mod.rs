//! # oxis::stochastic — stochastic process generators (Ring 2)
//!
//! A **Kind A** module: a pure, I/O-free path-simulation primitive with *no
//! pricing inside it*. Given a [`Process`] (GBM, Ornstein-Uhlenbeck, Vasicek, CIR,
//! Merton jump-diffusion, Heston) it produces reproducible sample paths over a time
//! grid, which downstream modules consume — `oxis::pricing` for path-dependent
//! exotics today, and the ML / portfolio rings later. Keeping the generator free of
//! pricing is what lets those consumers depend on raw paths without pulling in
//! `oxis::pricing`.
//!
//! ## Validation
//! A simulator has no "price" to check, so the oracle is the **closed-form moment**:
//! each process exposes [`Process::analytic_moments`], and the validation suite
//! asserts the simulated terminal mean/variance match within a standard-error band.
//! The exact-in-distribution schemes (GBM, OU/Vasicek, Merton) match up to sampling
//! error; the full-truncation square-root schemes (CIR, Heston variance) carry a
//! small discretization bias accounted for in the bands. Heston's dynamics are
//! additionally validated by pricing a European option over its paths against
//! QuantLib's `AnalyticHestonEngine` (in `oxis::pricing`).
//!
//! ## Reproducibility
//! Simulations are **bit-reproducible** for a given `(seed, paths, steps)`
//! regardless of thread count — antithetic pairs are seeded per-index via
//! [`crate::core::path_seed`] and reduced in order, exactly like the Monte Carlo /
//! Longstaff-Schwartz pricers.

mod process;
mod result;
mod simulate;

pub use process::Process;
pub use result::ProcessResult;
pub use simulate::{Path, SimConfig, TerminalSample, simulate_paths, simulate_terminal};
