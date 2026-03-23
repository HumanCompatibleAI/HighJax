//! Shared utility functions.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// Find a parquet file by stem. Tries `.pq` first, falls back to `.parquet`.
/// Returns the path if it exists, or `None`.
pub(crate) fn find_parquet_file(dir: &Path, stem: &str) -> Option<PathBuf> {
    let pq_path = dir.join(format!("{}.pq", stem));
    if pq_path.exists() {
        return Some(pq_path);
    }
    // Backward compat: old treks used .parquet extension
    let parquet_path = dir.join(format!("{}.parquet", stem));
    if parquet_path.exists() {
        return Some(parquet_path);
    }
    None
}

/// Whether the `posh` executable is available on this machine.
/// Checked once and cached for the lifetime of the process.
fn posh_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        Command::new("posh")
            .arg("/dev/null")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// Shorten a path using the `posh` executable if available.
/// Returns the original path string if posh is not installed.
pub fn posh_path(path: &Path) -> String {
    let path_str = path.to_string_lossy().to_string();
    if !posh_available() {
        return path_str;
    }
    match Command::new("posh").arg(&path_str).output() {
        Ok(output) if output.status.success() => {
            String::from_utf8(output.stdout)
                .map(|s| s.trim().to_string())
                .unwrap_or(path_str)
        }
        _ => path_str,
    }
}

/// Batch-shorten multiple paths using a single `posh` invocation.
/// Returns a vec of poshed strings in the same order as input.
pub fn posh_paths(paths: &[&Path]) -> Vec<String> {
    let path_strs: Vec<String> = paths.iter().map(|p| p.to_string_lossy().to_string()).collect();
    if !posh_available() || paths.is_empty() {
        return path_strs;
    }
    match Command::new("posh").args(&path_strs).output() {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let poshed: Vec<String> = stdout.lines().map(|l| l.to_string()).collect();
            if poshed.len() == paths.len() {
                poshed
            } else {
                path_strs
            }
        }
        _ => path_strs,
    }
}
