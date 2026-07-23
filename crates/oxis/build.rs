//! Build script: capture the short git commit hash and its date so the REPL
//! banner can show real build metadata. Both are best-effort — if git is not
//! available (e.g. building from a published crate), the values are left empty
//! and the banner simply omits them.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

fn main() {
    let sha = git(&["rev-parse", "--short", "HEAD"]).unwrap_or_default();
    let date = git(&["log", "-1", "--format=%cd", "--date=short"]).unwrap_or_default();

    println!("cargo:rustc-env=OXIS_GIT_SHA={sha}");
    println!("cargo:rustc-env=OXIS_BUILD_DATE={date}");

    // Rebuild the banner metadata when HEAD moves (best-effort; the workspace
    // .git lives two levels up from this crate).
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
}
