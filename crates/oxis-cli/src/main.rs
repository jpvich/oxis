//! The `oxis` command-line interface (Ring 1).
//!
//! A thin edge over the pure module cores: it parses args, builds plain inputs,
//! calls the core, and renders the result through [`oxis_core::output`]. Global
//! flags select the output format; per-command flags carry the inputs.
//!
//! Implemented: `oxis price` (Black-Scholes European). The REPL and the
//! `greeks` / `implied-vol` / `completions` subcommands land in later milestones.

#![forbid(unsafe_code)]

mod commands;

use clap::Parser;
use commands::Command;
use oxis_core::{OutputFormat, RunContext};
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "oxis",
    version,
    about = "OXIS — Open eXtensible Instruments & Statistics",
    propagate_version = true
)]
struct Cli {
    #[command(flatten)]
    global: GlobalArgs,

    #[command(subcommand)]
    command: Option<Command>,
}

/// Output/verbosity flags, available before or after the subcommand.
#[derive(clap::Args)]
struct GlobalArgs {
    /// Emit structured JSON.
    #[arg(long, global = true)]
    json: bool,
    /// Emit tab-separated values.
    #[arg(long, global = true)]
    tsv: bool,
    /// Suppress non-essential output.
    #[arg(long, global = true)]
    quiet: bool,
    /// Emit extra diagnostics to stderr.
    #[arg(long, global = true)]
    verbose: bool,
}

impl GlobalArgs {
    fn context(&self) -> RunContext {
        let format = if self.json {
            OutputFormat::Json
        } else if self.tsv {
            OutputFormat::Tsv
        } else {
            OutputFormat::Human
        };
        RunContext {
            format,
            quiet: self.quiet,
            verbose: self.verbose,
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let ctx = cli.global.context();

    let result = match cli.command {
        Some(command) => command.run(&ctx),
        None => {
            // No subcommand: the interactive REPL lands later. For now, point the
            // user at the available commands.
            eprintln!("oxis: run `oxis price --help` to get started (REPL coming soon).");
            return ExitCode::SUCCESS;
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            // CLI convention: `error: <message>` (lowercase) to stderr, non-zero exit.
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
