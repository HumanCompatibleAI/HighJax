//! Configuration types, constants, and log level management.

use std::fmt;
use std::fs;

use chrono::Local;
use tracing::info;
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::EnvFilter;

use crate::paths::config_dir_for;
use crate::RELOAD_HANDLE;

/// Maximum age of log files in days
pub(crate) const MAX_LOG_AGE_DAYS: i64 = 7;

/// Maximum total size of log directory in bytes (100 MB)
pub(crate) const MAX_LOG_DIR_SIZE: u64 = 100 * 1024 * 1024;

/// Configuration for logging initialization.
#[derive(Clone)]
pub struct LogConfig {
    /// Application name (used for config dir and env var).
    pub app_name: String,
    /// Enable verbose console output.
    pub verbose: bool,
    /// Command name to record in log header.
    pub command: String,
    /// Version/build info string to record in log header (e.g. git hash).
    pub version: String,
}

impl LogConfig {
    /// Create a new LogConfig with the given app name.
    pub fn new(app_name: impl Into<String>) -> Self {
        Self {
            app_name: app_name.into(),
            verbose: false,
            command: String::new(),
            version: String::new(),
        }
    }

    /// Set verbose mode (enables console output).
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set the command name for the log header.
    pub fn command(mut self, command: impl Into<String>) -> Self {
        self.command = command.into();
        self
    }

    /// Set the version/build info string for the log header.
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }
}

/// Custom timer that formats timestamps as: "2026-01-15T00:38:42.492"
pub(crate) struct LeonaTimer;

impl FormatTime for LeonaTimer {
    fn format_time(&self, w: &mut Writer<'_>) -> fmt::Result {
        let now = Local::now();
        write!(w, "{}", now.format("%Y-%m-%dT%H:%M:%S%.3f"))
    }
}

/// Load log_level from config.json for the given app.
pub(crate) fn load_log_level_from_config(app_name: &str) -> Option<String> {
    let config_path = config_dir_for(app_name).join("config.json");
    if !config_path.exists() {
        return None;
    }

    let content = fs::read_to_string(&config_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;
    json.get("log_level")?.as_str().map(|s| s.to_string())
}

/// Reload log level from config.json.
///
/// Called when config.json is modified. Uses the app name from initialization.
pub fn reload_log_level() {
    let app_name = match crate::APP_NAME.get() {
        Some(name) => name,
        None => return,
    };

    let new_level = load_log_level_from_config(app_name).unwrap_or_else(|| "info".to_string());

    if let Ok(guard) = RELOAD_HANDLE.lock() {
        if let Some(ref handle) = *guard {
            match EnvFilter::try_new(&new_level) {
                Ok(new_filter) => {
                    if handle.reload(new_filter).is_ok() {
                        info!("Log level changed to: {}", new_level);
                    }
                }
                Err(e) => {
                    info!("Invalid log level '{}': {}", new_level, e);
                }
            }
        }
    }
}
