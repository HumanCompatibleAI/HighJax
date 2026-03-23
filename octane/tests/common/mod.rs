//! Shared helpers for integration tests.
//!
//! - `airbus`: Trek fixtures, behavior fixtures, and airbus CLI wrappers.
//! - `svg`: SVG state builders, rendering helpers, and SVG parsing.

#![allow(dead_code, unused_imports)]

#[cfg(target_os = "linux")]
mod airbus;
mod svg;

#[cfg(target_os = "linux")]
pub use airbus::*;
pub use svg::*;

/// Get path to the octane binary built by cargo.
pub fn binary_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_octane"))
}
