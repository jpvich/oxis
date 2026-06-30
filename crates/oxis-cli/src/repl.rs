//! The interactive `oxis` REPL.
//!
//! A thin loop over the *same* clap definitions as the CLI: each entered line is
//! tokenized (quote-aware, via `shlex`), re-parsed by [`crate::Cli`], and
//! dispatched through [`Command::run`] — so the REPL and the CLI share one parser
//! and one set of command implementations, with no duplicated logic. The line
//! editor ([`rustyline`]) supplies history and tab-completion of command names
//! and their long flags (introspected from clap, so completion never drifts).

use clap::{CommandFactory, Parser};
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::{Editor, Helper, Highlighter, Hinter, Validator};

use crate::Cli;

/// Words handled by the REPL itself rather than the clap parser.
const BUILTINS: &[&str] = &["help", "quit", "exit"];

/// Run the interactive REPL until EOF (Ctrl-D), `quit`, or `exit`.
pub fn run() -> anyhow::Result<()> {
    println!("OXIS interactive REPL — type `help`, `<command> --help`, or `quit`.");
    println!("Commands are identical to the CLI (without the leading `oxis`).");

    let helper = ReplHelper::new();
    let mut editor: Editor<ReplHelper, rustyline::history::DefaultHistory> =
        Editor::new().map_err(|e| anyhow::anyhow!("could not start REPL: {e}"))?;
    editor.set_helper(Some(helper));

    loop {
        match editor.readline("oxis> ") {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let _ = editor.add_history_entry(trimmed);
                match trimmed {
                    "quit" | "exit" => break,
                    "help" => {
                        print_help();
                        continue;
                    }
                    _ => run_line(trimmed),
                }
            }
            // Ctrl-C cancels the current line; Ctrl-D / EOF exits.
            Err(ReadlineError::Interrupted) => continue,
            Err(ReadlineError::Eof) => break,
            Err(e) => {
                eprintln!("error: {e}");
                break;
            }
        }
    }
    Ok(())
}

/// Parse one REPL line as a full `oxis` invocation and dispatch it.
fn run_line(line: &str) {
    let Some(tokens) = shlex::split(line) else {
        eprintln!("error: unbalanced quotes");
        return;
    };
    // Re-parse with the same clap definitions as the CLI (program name prepended).
    let argv = std::iter::once("oxis".to_string()).chain(tokens);
    match Cli::try_parse_from(argv) {
        Ok(cli) => {
            // Per-line global flags (e.g. `--json`) select the output format.
            let ctx = cli.global.context();
            if let Some(command) = cli.command {
                if let Err(err) = command.run(&ctx) {
                    eprintln!("error: {err}");
                }
            }
        }
        // clap formats help/usage/validation errors; print them as-is.
        Err(err) => {
            let _ = err.print();
        }
    }
}

fn print_help() {
    println!("REPL commands:");
    println!("  help            show this message");
    println!("  quit | exit     leave the REPL (or press Ctrl-D)");
    println!();
    println!("Anything else is run as an `oxis` command. Examples:");
    println!("  price --spot 100 --strike 100 --rate 0.05 --vol 0.2 --t 1 --type call");
    println!("  greeks --spot 100 --strike 100 --rate 0.05 --vol 0.2 --t 1 --type call --json");
    println!(
        "  ml american --method dos --spot 100 --strike 100 --rate 0.05 --vol 0.3 --maturity 1"
    );
    println!();
    println!("Run `<command> --help` for a command's flags. Tab completes commands and flags.");
}

/// rustyline helper: tab-completion of command names (first word) and the long
/// flags of the active command (later words), introspected from clap.
#[derive(Helper, Hinter, Highlighter, Validator)]
struct ReplHelper {
    commands: Vec<String>,
}

impl ReplHelper {
    fn new() -> Self {
        let mut commands: Vec<String> = Cli::command()
            .get_subcommands()
            .map(|c| c.get_name().to_string())
            .collect();
        commands.extend(BUILTINS.iter().map(|s| s.to_string()));
        commands.sort();
        Self { commands }
    }
}

/// Long flags (`--flag`) for the subcommand named `name`, if any.
fn flags_for(name: &str) -> Vec<String> {
    let cli = Cli::command();
    let Some(sub) = cli.get_subcommands().find(|c| c.get_name() == name) else {
        return Vec::new();
    };
    // For commands with their own nested subcommands (e.g. `ml`, `portfolio`),
    // offer the nested command names; otherwise offer the long flags.
    let nested: Vec<String> = sub
        .get_subcommands()
        .map(|c| c.get_name().to_string())
        .collect();
    if !nested.is_empty() {
        return nested;
    }
    sub.get_arguments()
        .filter_map(|a| a.get_long().map(|l| format!("--{l}")))
        .collect()
}

impl Completer for ReplHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        // Start of the word under the cursor = just after the last whitespace.
        let start = line[..pos].rfind(char::is_whitespace).map_or(0, |i| i + 1);
        let word = &line[start..pos];
        let preceding = line[..start].trim();

        let candidates: Vec<String> = if preceding.is_empty() {
            // First word: complete command names + REPL builtins.
            self.commands.clone()
        } else {
            // Later word: complete the first token's flags / nested commands.
            let first = preceding.split_whitespace().next().unwrap_or("");
            flags_for(first)
        };

        let matches = candidates
            .into_iter()
            .filter(|c| c.starts_with(word))
            .map(|c| Pair {
                display: c.clone(),
                replacement: c,
            })
            .collect();
        Ok((start, matches))
    }
}
