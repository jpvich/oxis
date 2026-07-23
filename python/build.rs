//! Make the PyO3 extension (a `cdylib`) link with plain `cargo build` on
//! macOS/iOS: Python symbols are resolved at import time, not link time, so the
//! linker must defer unresolved symbols (`-undefined dynamic_lookup`).
//!
//! This is emitted as a **cdylib-specific link arg**, which (unlike
//! `[target.*].rustflags` in `.cargo/config.toml`) is *not* clobbered when
//! `RUSTFLAGS` is set in the environment — CI sets `RUSTFLAGS="-D warnings"`,
//! which would otherwise drop the flag and break the link. Scoped to this crate
//! only, so the CLI binary and other crates are unaffected.

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" || target_os == "ios" {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
