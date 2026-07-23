//! [`ProcessResult`] — the renderable summary of a path simulation.
//!
//! A simulator is judged by how well its sampled terminal distribution matches the
//! process's closed-form moments, so the result is self-describing: it carries the
//! process, the simulation size, the **sample** terminal mean/std (with the mean's
//! standard error), the **analytic** mean/std, and the absolute differences. Where
//! a process has no closed-form variance (Heston), the analytic-std and std-error
//! fields render as JSON `null` (the `Option` → `Cell::Null` convention shared with
//! `PriceResult.standard_error`).

use crate::core::{Cell, Column, Tabular, mean_and_se, sample_mean_var};
use serde::Serialize;

use crate::stochastic::process::Process;
use crate::stochastic::simulate::{SimConfig, TerminalSample};

/// The outcome of simulating a process and comparing its terminal moments to the
/// closed form.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProcessResult {
    /// Process identifier (`"gbm"`, `"heston"`, …).
    pub process: &'static str,
    /// Initial state `x0`.
    pub x0: f64,
    /// Horizon in years.
    pub t: f64,
    /// Number of simulated paths.
    pub paths: usize,
    /// Number of time steps on the grid.
    pub steps: usize,
    /// Sample mean of the terminal state.
    pub sample_mean: f64,
    /// Standard error of the sample mean (over antithetic pair averages).
    pub mean_std_error: f64,
    /// Sample standard deviation of the terminal state.
    pub sample_std: f64,
    /// Closed-form `E[X_t]`.
    pub analytic_mean: f64,
    /// Closed-form `Std[X_t]`, if available (`None` for Heston).
    pub analytic_std: Option<f64>,
    /// `|sample_mean − analytic_mean|`.
    pub mean_abs_error: f64,
    /// `|sample_std − analytic_std|`, if an analytic std exists.
    pub std_abs_error: Option<f64>,
}

impl ProcessResult {
    /// Summarize a terminal sample against the process's analytic moments.
    pub fn from_simulation(
        process: &Process,
        x0: f64,
        t: f64,
        cfg: &SimConfig,
        sample: &TerminalSample,
    ) -> Self {
        let (sample_mean, mean_std_error) = mean_and_se(&sample.pair_means);
        let (_, var) = sample_mean_var(&sample.terminals);
        let sample_std = var.sqrt();

        let (analytic_mean, analytic_var) = process.analytic_moments(x0, t);
        let analytic_std = analytic_var.map(f64::sqrt);

        Self {
            process: process.name(),
            x0,
            t,
            paths: sample.terminals.len(),
            steps: cfg.steps,
            sample_mean,
            mean_std_error,
            sample_std,
            analytic_mean,
            analytic_std,
            mean_abs_error: (sample_mean - analytic_mean).abs(),
            std_abs_error: analytic_std.map(|s| (sample_std - s).abs()),
        }
    }
}

impl Tabular for ProcessResult {
    fn columns(&self) -> Vec<Column> {
        vec![
            Column::new("process"),
            Column::new("x0"),
            Column::new("t"),
            Column::new("paths"),
            Column::new("steps"),
            Column::new("sample_mean"),
            Column::new("mean_std_error"),
            Column::new("sample_std"),
            Column::new("analytic_mean"),
            Column::new("analytic_std"),
            Column::new("mean_abs_error"),
            Column::new("std_abs_error"),
        ]
    }

    fn cells(&self) -> Vec<Cell> {
        vec![
            Cell::str(self.process),
            Cell::F64(self.x0),
            Cell::F64(self.t),
            Cell::Int(self.paths as i64),
            Cell::Int(self.steps as i64),
            Cell::F64(self.sample_mean),
            Cell::F64(self.mean_std_error),
            Cell::F64(self.sample_std),
            Cell::F64(self.analytic_mean),
            self.analytic_std.into(),
            Cell::F64(self.mean_abs_error),
            self.std_abs_error.into(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stochastic::simulate::simulate_terminal;

    #[test]
    fn renders_through_output_layer_with_optional_nulls() {
        let process = Process::Heston {
            mu: 0.03,
            v0: 0.04,
            kappa: 1.5,
            theta: 0.04,
            xi: 0.3,
            rho: -0.5,
        };
        let cfg = SimConfig {
            paths: 2_000,
            steps: 20,
            seed: 9,
        };
        let sample = simulate_terminal(&process, 100.0, 1.0, &cfg).unwrap();
        let r = ProcessResult::from_simulation(&process, 100.0, 1.0, &cfg, &sample);

        assert_eq!(r.columns().len(), r.cells().len());
        let json = crate::core::output::render(&r, crate::core::OutputFormat::Json);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["process"], "heston");
        // Heston has no closed-form variance here → null std fields.
        assert!(parsed["analytic_std"].is_null());
        assert!(parsed["std_abs_error"].is_null());
        assert!(parsed["analytic_mean"].as_f64().unwrap() > 0.0);
    }
}
