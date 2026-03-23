//! Log pointer files and posh path formatting.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Convert a path to poshed form using the `posh` executable.
/// Falls back to the original path string if posh is not available or fails.
/// Caches the availability check so we only probe once per process.
fn posh_path(path: &Path) -> String {
    use std::sync::OnceLock;
    static POSH_AVAILABLE: OnceLock<bool> = OnceLock::new();

    let path_str = path.to_string_lossy().to_string();
    if !*POSH_AVAILABLE.get_or_init(|| {
        Command::new("posh").arg("/dev/null").output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }) {
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

/// Print the log path to stderr in dim gray.
/// Uses poshed path if posh executable is available.
pub(crate) fn print_log_path(log_path: &Path) {
    let poshed = posh_path(log_path);
    // Dim gray ANSI: \x1b[90m, reset: \x1b[0m
    eprintln!("\x1b[90m{}\x1b[0m", poshed);
}

/// Get the pointers directory (~/.leona/pointers)
fn pointers_dir() -> PathBuf {
    dirs::home_dir()
        .expect("Could not find home directory")
        .join(".leona")
        .join("pointers")
}

/// Get the tmux pane GUID by looking up the pane ID in byobu's pane-guids directory.
fn get_pane_guid() -> Option<String> {
    // Get pane ID from tmux
    let output = Command::new("tmux")
        .args(["display-message", "-p", "#{pane_id}"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let pane_id = String::from_utf8(output.stdout).ok()?.trim().to_string();
    if pane_id.is_empty() {
        return None;
    }

    // Look up GUID in ~/.byobu/pane-guids/{pane_id}
    let home = dirs::home_dir()?;
    let guid_path = home.join(".byobu").join("pane-guids").join(&pane_id);

    if guid_path.exists() {
        fs::read_to_string(&guid_path)
            .ok()
            .map(|s| s.trim().to_string())
    } else {
        None
    }
}

/// Get ttyname using libc.
fn get_ttyname() -> Option<String> {
    // Use /proc/self/fd/0 to get tty path
    fs::read_link("/proc/self/fd/0")
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}

/// Sanitize a string for use as a filename (replace slashes with underscores).
fn sanitize_for_filename(s: &str) -> String {
    s.replace('/', "_")
}

/// Write log pointer files to ~/.leona/pointers/ for various terminal identifiers.
/// These files allow keybindings to fetch the current log path.
pub(crate) fn write_log_pointers(log_path: &Path) {
    let pointers = pointers_dir();

    // Create pointers directory if it doesn't exist
    if fs::create_dir_all(&pointers).is_err() {
        return;
    }

    let poshed = posh_path(log_path);

    // Helper to write a single pointer file
    let write_pointer = |filename: &str| {
        let path = pointers.join(filename);
        let _ = fs::write(&path, &poshed);
    };

    // 1. Pane GUID (tmux/byobu)
    if let Some(guid) = get_pane_guid() {
        write_pointer(&format!("leona_log_{}", guid));
    }

    // 2. Windows Terminal session (first 8 chars)
    if let Ok(wt_session) = env::var("WT_SESSION") {
        let prefix = if wt_session.len() >= 8 {
            &wt_session[..8]
        } else {
            &wt_session
        };
        write_pointer(&format!("leona_log_wt_{}", prefix));
    }

    // 3. TTY from env var
    if let Ok(tty) = env::var("TTY") {
        let sanitized = sanitize_for_filename(&tty);
        write_pointer(&format!("leona_log_{}", sanitized));
    }

    // 4. ttyname from OS
    if let Some(ttyname) = get_ttyname() {
        let sanitized = sanitize_for_filename(&ttyname);
        write_pointer(&format!("leona_log_{}", sanitized));
    }
}
