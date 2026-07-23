//! The interactive `oxis` REPL.
//!
//! A thin loop over the *same* clap definitions as the CLI: each entered line is
//! tokenized (quote-aware, via `shlex`), re-parsed by [`crate::Cli`], and
//! dispatched through [`Command::run`] — so the REPL and the CLI share one parser
//! and one set of command implementations, with no duplicated logic. The line
//! editor ([`reedline`]) supplies history and an interactive completion menu (the
//! `IdeMenu` dropdown, opened with Tab). Completion walks the clap command tree at
//! the cursor position — top-level commands, then a command's nested subcommands
//! (e.g. `ml american`), then that command's long flags plus the global flags —
//! so it descends to any depth and never drifts from the real parser, and each
//! entry carries its clap help text as the dropdown description.

use std::borrow::Cow;
use std::io::IsTerminal;

use clap::{CommandFactory, Parser};
use reedline::{
    Completer, Emacs, IdeMenu, KeyCode, KeyModifiers, MenuBuilder, Prompt, PromptEditMode,
    PromptHistorySearch, PromptHistorySearchStatus, Reedline, ReedlineEvent, ReedlineMenu, Signal,
    Span, Suggestion, default_emacs_keybindings,
};

use crate::Cli;

/// Name binding the Tab key to the completion dropdown.
const MENU_NAME: &str = "completion_menu";

/// REPL builtins (name + description) handled by the loop rather than clap.
const BUILTINS: &[(&str, &str)] = &[
    ("help", "show the command listing"),
    ("quit", "leave the REPL"),
    ("exit", "leave the REPL"),
];

/// Big "OXIS" wordmark (ANSI Shadow style) shown when the REPL opens.
const LOGO: [&str; 6] = [
    r"   ██████╗  ██╗  ██╗ ██╗ ███████╗",
    r"  ██╔═══██╗ ╚██╗██╔╝ ██║ ██╔════╝",
    r"  ██║   ██║  ╚███╔╝  ██║ ███████╗",
    r"  ██║   ██║  ██╔██╗  ██║ ╚════██║",
    r"  ╚██████╔╝ ██╔╝ ██╗ ██║ ███████║",
    r"   ╚═════╝  ╚═╝  ╚═╝ ╚═╝ ╚══════╝",
];

/// The re-exported `oxis::<module>` modules, for the live count in the banner.
const MODULES: &[&str] = &[
    "pricing",
    "greeks",
    "curves",
    "bonds",
    "stochastic",
    "stats",
    "portfolio",
    "ml",
];

/// Repository shown in the banner.
const REPO: &str = "github.com/jpvich/oxis";

/// ANSI accent/dim/reset codes, or empty strings when color is disabled
/// (output is not a terminal, or `NO_COLOR` is set) so pipes stay clean.
fn colors() -> (&'static str, &'static str, &'static str) {
    let enabled = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    if enabled {
        ("\x1b[1;36m", "\x1b[2m", "\x1b[0m") // bright cyan · dim · reset
    } else {
        ("", "", "")
    }
}

/// Run the interactive REPL until EOF (Ctrl-D), `quit`, or `exit`.
pub fn run() -> anyhow::Result<()> {
    // The REPL drives an interactive line editor; it needs a real terminal.
    if !std::io::stdin().is_terminal() {
        anyhow::bail!(
            "the interactive REPL requires a terminal; pass a subcommand to use oxis non-interactively (see `oxis --help`)"
        );
    }

    print_banner();

    // Tab opens an IDE-style dropdown (a bordered menu under the cursor) and then
    // moves through it; ↑/↓ navigate, Enter accepts.
    let menu = Box::new(IdeMenu::default().with_name(MENU_NAME));
    let mut keybindings = default_emacs_keybindings();
    keybindings.add_binding(
        KeyModifiers::NONE,
        KeyCode::Tab,
        ReedlineEvent::UntilFound(vec![
            ReedlineEvent::Menu(MENU_NAME.to_string()),
            ReedlineEvent::MenuNext,
        ]),
    );

    let mut editor = Reedline::create()
        .with_completer(Box::new(OxisCompleter))
        .with_menu(ReedlineMenu::EngineCompleter(menu))
        .with_edit_mode(Box::new(Emacs::new(keybindings)));
    let prompt = OxisPrompt;

    loop {
        match editor.read_line(&prompt) {
            Ok(Signal::Success(line)) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                match trimmed {
                    "quit" | "exit" => break,
                    "help" => {
                        print_help();
                        continue;
                    }
                    _ => run_line(trimmed),
                }
            }
            // Ctrl-C cancels the current line; Ctrl-D exits.
            Ok(Signal::CtrlC) => continue,
            Ok(Signal::CtrlD) => break,
            // `Signal` is non-exhaustive; treat any future variant as a no-op.
            Ok(_) => continue,
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

/// Print the opening banner: big logo, version/build metadata, a one-line value
/// proposition, live command/module counts, the repo link, and usage hints.
fn print_banner() {
    let (accent, dim, reset) = colors();
    let rule = "─".repeat(52);

    for line in LOGO {
        println!("{accent}{line}{reset}");
    }
    println!();
    println!("  Open eXtensible Instruments & Statistics");
    println!("  {dim}{rule}{reset}");

    // Version + build metadata (date and short commit come from build.rs).
    let mut meta = format!("v{}", env!("CARGO_PKG_VERSION"));
    let date = env!("OXIS_BUILD_DATE");
    if !date.is_empty() {
        meta.push_str(&format!("   ·   built {date}"));
    }
    let sha = env!("OXIS_GIT_SHA");
    if !sha.is_empty() {
        meta.push_str(&format!("   ·   {sha}"));
    }
    println!("  {dim}{meta}{reset}");
    println!("  {dim}validated quantitative finance, in Rust{reset}");

    let commands = Cli::command().get_subcommands().count();
    println!(
        "  {dim}{commands} commands · {} modules — mirror the CLI (drop the `oxis`){reset}",
        MODULES.len()
    );
    println!("  {dim}{REPO}{reset}");
    println!("  {dim}⇥ tab opens the completion menu · type `help` · `quit` to exit{reset}");
    println!();
}

/// Name + one-line description of every top-level command, introspected from
/// clap so the listing tracks the real command set (never hand-maintained).
fn command_overview() -> Vec<(String, String)> {
    Cli::command()
        .get_subcommands()
        .map(|c| {
            let about = c.get_about().map(|s| s.to_string()).unwrap_or_default();
            (c.get_name().to_string(), about)
        })
        .collect()
}

/// Print the available commands as an aligned, dynamically built table.
fn print_commands() {
    let cmds = command_overview();
    let width = cmds.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
    println!("Commands:");
    for (name, about) in cmds {
        // Keep the description to its first line for a compact overview.
        let about = about.lines().next().unwrap_or("");
        println!("  {name:<width$}  {about}");
    }
}

fn print_help() {
    println!("REPL builtins:");
    println!("  help            show this message");
    println!("  quit | exit     leave the REPL (or press Ctrl-D)");
    println!();
    print_commands();
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

/// Global flags declared on [`Cli`] with `global = true`; available on every
/// command, so completion offers them (with their help text) at any depth.
const GLOBAL_FLAGS: &[(&str, &str)] = &[
    ("--json", "Emit structured JSON"),
    ("--tsv", "Emit tab-separated values"),
    ("--quiet", "Suppress non-essential output"),
    ("--verbose", "Emit extra diagnostics to stderr"),
];

/// A completion candidate: the text inserted plus an optional one-line
/// description shown next to it in the dropdown.
struct Candidate {
    value: String,
    description: Option<String>,
}

/// First line of a (possibly multi-line) clap help string.
fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").to_string()
}

/// Walk the clap command tree following the subcommand names already typed,
/// returning the deepest command reached. Tokens that aren't subcommand names
/// (flags, flag values) are skipped, so `ml american --spot 100` resolves to the
/// `american` command.
fn resolve_command<'a>(root: &'a clap::Command, preceding: &[&str]) -> &'a clap::Command {
    let mut current = root;
    for tok in preceding {
        if tok.starts_with('-') {
            continue;
        }
        if let Some(sub) = current.find_subcommand(tok) {
            current = sub;
        }
    }
    current
}

/// Nested subcommand names of `cmd`, each with its one-line about text.
fn subcommand_candidates(cmd: &clap::Command) -> Vec<Candidate> {
    cmd.get_subcommands()
        .map(|c| Candidate {
            value: c.get_name().to_string(),
            description: c.get_about().map(|s| first_line(&s.to_string())),
        })
        .collect()
}

/// Long flags (`--flag`) declared on `cmd` with their help text, plus the
/// always-available globals.
fn flag_candidates(cmd: &clap::Command) -> Vec<Candidate> {
    let mut flags: Vec<Candidate> = cmd
        .get_arguments()
        .filter_map(|a| {
            a.get_long().map(|l| Candidate {
                value: format!("--{l}"),
                description: a.get_help().map(|s| first_line(&s.to_string())),
            })
        })
        .collect();
    for (flag, desc) in GLOBAL_FLAGS {
        if !flags.iter().any(|c| c.value == *flag) {
            flags.push(Candidate {
                value: (*flag).to_string(),
                description: Some((*desc).to_string()),
            });
        }
    }
    flags
}

/// Compute the completion candidates for the word under the cursor: returns the
/// word's start offset and the sorted, prefix-filtered candidate list. Pure and
/// I/O-free so it can be unit-tested without a live line editor.
fn candidates_at(line: &str, pos: usize) -> (usize, Vec<Candidate>) {
    // Start of the word under the cursor = just after the last whitespace.
    let start = line[..pos].rfind(char::is_whitespace).map_or(0, |i| i + 1);
    let word = &line[start..pos];
    let preceding: Vec<&str> = line[..start].split_whitespace().collect();

    let root = Cli::command();
    let current = resolve_command(&root, &preceding);

    let mut candidates: Vec<Candidate> = if word.starts_with('-') {
        // Completing a flag: the resolved command's long flags (+ globals).
        flag_candidates(current)
    } else if preceding.is_empty() {
        // First word: top-level command names + REPL builtins.
        let mut c = subcommand_candidates(current);
        c.extend(BUILTINS.iter().map(|(name, desc)| Candidate {
            value: (*name).to_string(),
            description: Some((*desc).to_string()),
        }));
        c
    } else {
        // Later word: nested subcommand names if the command has any,
        // otherwise fall back to its flags.
        let subs = subcommand_candidates(current);
        if subs.is_empty() {
            flag_candidates(current)
        } else {
            subs
        }
    };
    candidates.retain(|c| c.value.starts_with(word));
    candidates.sort_by(|a, b| a.value.cmp(&b.value));
    (start, candidates)
}

/// reedline completer: turns [`candidates_at`] into dropdown suggestions.
struct OxisCompleter;

impl Completer for OxisCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        let (start, candidates) = candidates_at(line, pos);
        candidates
            .into_iter()
            .map(|c| Suggestion {
                value: c.value,
                description: c.description,
                style: None,
                extra: None,
                span: Span { start, end: pos },
                append_whitespace: true,
                display_override: None,
                match_indices: None,
            })
            .collect()
    }
}

/// Minimal `oxis> ` prompt for reedline.
struct OxisPrompt;

impl Prompt for OxisPrompt {
    fn render_prompt_left(&self) -> Cow<'_, str> {
        Cow::Borrowed("oxis> ")
    }
    fn render_prompt_right(&self) -> Cow<'_, str> {
        Cow::Borrowed("")
    }
    fn render_prompt_indicator(&self, _mode: PromptEditMode) -> Cow<'_, str> {
        Cow::Borrowed("")
    }
    fn render_prompt_multiline_indicator(&self) -> Cow<'_, str> {
        Cow::Borrowed("... ")
    }
    fn render_prompt_history_search_indicator(
        &self,
        history_search: PromptHistorySearch,
    ) -> Cow<'_, str> {
        let failing = matches!(history_search.status, PromptHistorySearchStatus::Failing);
        let prefix = if failing { "failing " } else { "" };
        Cow::Owned(format!("({prefix}search: {}) ", history_search.term))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Candidate strings for a line completed at its end.
    fn complete(line: &str) -> Vec<String> {
        candidates_at(line, line.len())
            .1
            .into_iter()
            .map(|c| c.value)
            .collect()
    }

    #[test]
    fn first_word_offers_commands_and_builtins() {
        let cands = complete("");
        assert!(cands.contains(&"price".to_string()));
        assert!(cands.contains(&"ml".to_string()));
        assert!(cands.contains(&"quit".to_string()));
    }

    #[test]
    fn first_word_is_prefix_filtered() {
        let cands = complete("pri");
        assert_eq!(cands, vec!["price".to_string()]);
    }

    #[test]
    fn nested_subcommands_complete() {
        // `ml ` should offer its nested engines, not top-level commands.
        let cands = complete("ml ");
        assert!(cands.contains(&"american".to_string()));
        assert!(cands.contains(&"price".to_string()));
        assert!(!cands.contains(&"greeks".to_string()));
    }

    #[test]
    fn nested_command_flags_complete() {
        // The gap we fixed: flags of a *nested* subcommand.
        let cands = complete("ml american --met");
        assert_eq!(cands, vec!["--method".to_string()]);
    }

    #[test]
    fn flags_include_globals_at_any_depth() {
        let cands = complete("ml american --");
        assert!(cands.contains(&"--spot".to_string()));
        assert!(cands.contains(&"--json".to_string()));
    }

    #[test]
    fn flag_values_do_not_derail_resolution() {
        // Tokens after a consumed flag value must not break tree descent.
        let cands = complete("ml american --spot 100 --meth");
        assert_eq!(cands, vec!["--method".to_string()]);
    }
}
