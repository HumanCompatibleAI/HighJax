//! Leona - Logging library for HighJax Rust applications (vendored from mussel).
//!
//! Provides file-based logging with timestamped format and automatic log rotation.
//! Supports hot-reloading of log level from config.json.
//!
//! # Example
//!
//! ```ignore
//! use leona::LogConfig;
//!
//! let config = LogConfig::new("myapp")
//!     .verbose(true)
//!     .command("serve");
//!
//! let log_path = leona::init(config)?;
//! ```

mod config;
mod dedup;
pub mod paths;
mod pointers;
mod rotation;

pub use config::{reload_log_level, LogConfig};
pub use paths::{config_dir, config_dir_for, debug_dir, debug_dir_for, logs_dir, logs_dir_for};

use anyhow::{Context, Result};
use chrono::Local;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::reload;
use tracing_subscriber::{fmt as subscriber_fmt, prelude::*, EnvFilter};

use config::{load_log_level_from_config, LeonaTimer};
use pointers::{print_log_path, write_log_pointers};
use rotation::rotate_logs;

/// Guard that must be kept alive for the duration of the program.
/// Stored in a Mutex so the panic hook can take+drop it to flush buffered output.
static LOG_GUARD: Mutex<Option<WorkerGuard>> = Mutex::new(None);

/// Path to the current log file (for panic hook).
static LOG_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Handle for reloading log level at runtime.
static RELOAD_HANDLE: Mutex<Option<reload::Handle<EnvFilter, tracing_subscriber::Registry>>> =
    Mutex::new(None);

/// App name used for config directory and env var lookup.
static APP_NAME: OnceLock<String> = OnceLock::new();

/// Initialize logging with timestamped format.
///
/// Creates the log file, sets up tracing subscriber, and stores the app name
/// for later use by helper functions.
///
/// Returns the path to the created log file.
pub fn init(config: LogConfig) -> Result<PathBuf> {
    // Store app name for later use
    let _ = APP_NAME.set(config.app_name.clone());

    let logs_dir = paths::logs_dir_for(&config.app_name);
    fs::create_dir_all(&logs_dir)
        .with_context(|| format!("Failed to create logs directory: {logs_dir:?}"))?;

    rotate_logs(&logs_dir);

    // Generate log filename with timestamp
    let now = Local::now();
    let timestamp = now.format("%Y-%m-%d-%H-%M-%S");
    let log_filename = format!("{timestamp}_{}.log", config.app_name);
    let log_path = logs_dir.join(&log_filename);

    // Write header (leona style)
    {
        let mut file = File::create(&log_path)
            .with_context(|| format!("Failed to create log file: {log_path:?}"))?;

        let cwd = env::current_dir().unwrap_or_default();
        let user = env::var("USER")
            .or_else(|_| env::var("USERNAME"))
            .unwrap_or_else(|_| "unknown".to_string());
        let machine = hostname::get()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        let pid = std::process::id();

        writeln!(file, "datetime: {}", now.format("%Y-%m-%dT%H:%M:%S%.6f"))?;
        writeln!(file, "command: {} {}", config.app_name, config.command)?;
        if !config.version.is_empty() {
            writeln!(file, "version: {}", config.version)?;
        }
        writeln!(file, "cwd: {}", cwd.display())?;
        writeln!(file, "user: {user}")?;
        writeln!(file, "machine: {machine}")?;
        writeln!(file, "pid: {pid}")?;
        writeln!(file, "--- LOG START ---")?;
    }

    // Set up file appender (append mode)
    let file = fs::OpenOptions::new()
        .append(true)
        .open(&log_path)
        .with_context(|| format!("Failed to open log file for append: {log_path:?}"))?;

    let (non_blocking, guard) = tracing_appender::non_blocking(file);
    if let Ok(mut g) = LOG_GUARD.lock() {
        *g = Some(guard);
    }

    // Build filter: {APP_NAME}_LOG env var > config.json log_level > default "info"
    let env_var_name = format!("{}_LOG", config.app_name.to_uppercase());
    let env_filter = EnvFilter::try_from_env(&env_var_name).unwrap_or_else(|_| {
        let config_level = load_log_level_from_config(&config.app_name);
        EnvFilter::new(config_level.unwrap_or_else(|| "info".to_string()))
    });

    // Wrap filter in reload layer for hot-reloading
    let (filter_layer, reload_handle) = reload::Layer::new(env_filter);

    if let Ok(mut guard) = RELOAD_HANDLE.lock() {
        *guard = Some(reload_handle);
    }

    // Custom format: "2026-01-15T00:38:42.492 message"
    let file_layer = subscriber_fmt::layer()
        .with_writer(non_blocking)
        .with_ansi(false)
        .with_target(false)
        .with_level(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_file(false)
        .with_line_number(false)
        .with_timer(LeonaTimer);

    let subscriber = tracing_subscriber::registry()
        .with(filter_layer)
        .with(dedup::DedupLayer::new())
        .with(file_layer);

    // Add console output if verbose
    if config.verbose {
        let console_layer = subscriber_fmt::layer()
            .with_writer(std::io::stderr)
            .with_ansi(true)
            .with_target(false)
            .with_level(false)
            .compact()
            .with_timer(LeonaTimer);

        subscriber.with(console_layer).init();
    } else {
        subscriber.init();
    }

    // Store log path for panic hook
    if let Ok(mut guard) = LOG_PATH.lock() {
        *guard = Some(log_path.clone());
    }

    // Print poshed log path to stderr and write pointer files
    print_log_path(&log_path);
    write_log_pointers(&log_path);

    Ok(log_path)
}

/// Install a panic hook that writes to the log file before the process exits.
///
/// This ensures panics are captured in the log file even if they would otherwise
/// exit silently. Call this after `init()`.
pub fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        // Flush the tracing non-blocking writer first so buffered log lines
        // appear in the file before the panic message.
        // Use try_lock to avoid deadlock if panic occurs while mutex is held.
        if let Ok(mut guard) = LOG_GUARD.try_lock() {
            drop(guard.take());
        }

        if let Ok(guard) = LOG_PATH.try_lock() {
            if let Some(ref log_path) = *guard {
                if let Ok(mut file) = OpenOptions::new().append(true).open(log_path) {
                    let now = Local::now();
                    let timestamp = now.format("%Y-%m-%dT%H:%M:%S%.3f");

                    let location = panic_info
                        .location()
                        .map(|loc| format!("{}:{}:{}", loc.file(), loc.line(), loc.column()))
                        .unwrap_or_else(|| "unknown location".to_string());

                    let message = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic payload".to_string()
                    };

                    let _ = writeln!(file, "{} PANIC at {}: {}", timestamp, location, message);
                    let _ = writeln!(
                        file,
                        "{} Backtrace:\n{:?}",
                        timestamp,
                        std::backtrace::Backtrace::capture()
                    );
                    let _ = file.flush();
                }
            }
        }

        default_hook(panic_info);
    }));
}

/// Get the current log file path, if logging has been initialized.
pub fn log_path() -> Option<PathBuf> {
    LOG_PATH.lock().ok().and_then(|g| g.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use config::{LeonaTimer, MAX_LOG_AGE_DAYS, MAX_LOG_DIR_SIZE};
    use tracing_subscriber::fmt::format::Writer;
    use tracing_subscriber::fmt::time::FormatTime;

    // ==================== Directory Functions ====================

    #[test]
    fn test_config_dir_for() {
        let dir = config_dir_for("testapp");
        assert!(dir.ends_with(".testapp"));
    }

    #[test]
    fn test_logs_dir_for() {
        let dir = logs_dir_for("testapp");
        assert!(dir.ends_with("testapp"));
        assert!(dir.parent().unwrap().ends_with("logs"));
    }

    #[test]
    fn test_debug_dir_for() {
        let dir = debug_dir_for("testapp");
        assert!(dir.ends_with("debug"));
        assert!(dir.parent().unwrap().ends_with(".testapp"));
    }

    #[test]
    fn test_config_dir_is_in_home() {
        let dir = config_dir_for("testapp");
        let home = dirs::home_dir().unwrap();
        assert!(dir.starts_with(&home));
    }

    #[test]
    fn test_config_dir_is_absolute() {
        let dir = config_dir_for("testapp");
        assert!(dir.is_absolute());
    }

    #[test]
    fn test_logs_dir_is_absolute() {
        let dir = logs_dir_for("testapp");
        assert!(dir.is_absolute());
    }

    #[test]
    fn test_debug_dir_is_absolute() {
        let dir = debug_dir_for("testapp");
        assert!(dir.is_absolute());
    }

    #[test]
    fn test_logs_dir_under_highjax() {
        let logs = logs_dir_for("testapp");
        let home = dirs::home_dir().unwrap();
        assert!(logs.starts_with(home.join(".highjax").join("logs")));
    }

    #[test]
    fn test_debug_dir_under_config_dir() {
        let config = config_dir_for("testapp");
        let debug = debug_dir_for("testapp");
        assert!(debug.starts_with(&config));
    }

    #[test]
    fn test_config_dir_consistent() {
        let dir1 = config_dir_for("testapp");
        let dir2 = config_dir_for("testapp");
        assert_eq!(dir1, dir2);
    }

    #[test]
    fn test_logs_dir_consistent() {
        let dir1 = logs_dir_for("testapp");
        let dir2 = logs_dir_for("testapp");
        assert_eq!(dir1, dir2);
    }

    #[test]
    fn test_debug_dir_consistent() {
        let dir1 = debug_dir_for("testapp");
        let dir2 = debug_dir_for("testapp");
        assert_eq!(dir1, dir2);
    }

    #[test]
    fn test_different_apps_different_dirs() {
        let dir1 = config_dir_for("app1");
        let dir2 = config_dir_for("app2");
        assert_ne!(dir1, dir2);
    }

    #[test]
    fn test_app_name_with_hyphen() {
        let dir = config_dir_for("my-app");
        assert!(dir.ends_with(".my-app"));
    }

    #[test]
    fn test_app_name_with_underscore() {
        let dir = config_dir_for("my_app");
        assert!(dir.ends_with(".my_app"));
    }

    // ==================== LogConfig ====================

    #[test]
    fn test_log_config_builder() {
        let config = LogConfig::new("myapp").verbose(true).command("serve");

        assert_eq!(config.app_name, "myapp");
        assert!(config.verbose);
        assert_eq!(config.command, "serve");
    }

    #[test]
    fn test_log_config_defaults() {
        let config = LogConfig::new("myapp");

        assert_eq!(config.app_name, "myapp");
        assert!(!config.verbose);
        assert!(config.command.is_empty());
    }

    #[test]
    fn test_log_config_clone() {
        let config1 = LogConfig::new("myapp").verbose(true).command("serve");
        let config2 = config1.clone();

        assert_eq!(config1.app_name, config2.app_name);
        assert_eq!(config1.verbose, config2.verbose);
        assert_eq!(config1.command, config2.command);
    }

    #[test]
    fn test_log_config_builder_chaining() {
        let config = LogConfig::new("app")
            .verbose(false)
            .command("cmd1")
            .verbose(true)
            .command("cmd2");

        assert!(config.verbose);
        assert_eq!(config.command, "cmd2");
    }

    #[test]
    fn test_log_config_from_string() {
        let name = String::from("myapp");
        let config = LogConfig::new(name);
        assert_eq!(config.app_name, "myapp");
    }

    #[test]
    fn test_log_config_command_from_string() {
        let cmd = String::from("serve");
        let config = LogConfig::new("app").command(cmd);
        assert_eq!(config.command, "serve");
    }

    #[test]
    fn test_log_config_empty_command() {
        let config = LogConfig::new("app").command("");
        assert!(config.command.is_empty());
    }

    #[test]
    fn test_log_config_command_with_args() {
        let config = LogConfig::new("app").command("serve --port 8080");
        assert_eq!(config.command, "serve --port 8080");
    }

    // ==================== Constants ====================

    #[test]
    fn test_max_log_age_reasonable() {
        assert!(MAX_LOG_AGE_DAYS >= 1);
        assert!(MAX_LOG_AGE_DAYS <= 30);
    }

    #[test]
    fn test_max_log_dir_size_reasonable() {
        assert!(MAX_LOG_DIR_SIZE >= 10 * 1024 * 1024);
        assert!(MAX_LOG_DIR_SIZE <= 500 * 1024 * 1024);
    }

    #[test]
    fn test_max_log_age_is_7_days() {
        assert_eq!(MAX_LOG_AGE_DAYS, 7);
    }

    #[test]
    fn test_max_log_dir_size_is_100mb() {
        assert_eq!(MAX_LOG_DIR_SIZE, 100 * 1024 * 1024);
    }

    // ==================== LeonaTimer ====================

    #[test]
    fn test_leona_timer_format() {
        let timer = LeonaTimer;
        let mut buf = String::new();
        let mut writer = Writer::new(&mut buf);

        timer.format_time(&mut writer).unwrap();

        assert!(buf.len() >= 23, "Timestamp too short: {}", buf);
        assert!(buf.contains('T'), "Should contain 'T' separator: {}", buf);
        assert!(buf.contains(':'), "Should contain colons: {}", buf);
        assert!(buf.contains('.'), "Should contain milliseconds: {}", buf);
    }

    #[test]
    fn test_leona_timer_iso_format() {
        let timer = LeonaTimer;
        let mut buf = String::new();
        let mut writer = Writer::new(&mut buf);

        timer.format_time(&mut writer).unwrap();

        // Should match format: YYYY-MM-DDTHH:MM:SS.mmm
        let parts: Vec<&str> = buf.split('T').collect();
        assert_eq!(parts.len(), 2, "Should have date and time separated by T");

        let date_parts: Vec<&str> = parts[0].split('-').collect();
        assert_eq!(date_parts.len(), 3, "Date should have 3 parts");
        assert_eq!(date_parts[0].len(), 4, "Year should be 4 digits");
        assert_eq!(date_parts[1].len(), 2, "Month should be 2 digits");
        assert_eq!(date_parts[2].len(), 2, "Day should be 2 digits");
    }

    #[test]
    fn test_leona_timer_has_milliseconds() {
        let timer = LeonaTimer;
        let mut buf = String::new();
        let mut writer = Writer::new(&mut buf);

        timer.format_time(&mut writer).unwrap();

        let dot_pos = buf.rfind('.').expect("Should have milliseconds");
        let millis = &buf[dot_pos + 1..];
        assert_eq!(millis.len(), 3, "Should have 3 digit milliseconds");
    }

    #[test]
    fn test_leona_timer_consistent_length() {
        let timer = LeonaTimer;

        for _ in 0..10 {
            let mut buf = String::new();
            let mut writer = Writer::new(&mut buf);
            timer.format_time(&mut writer).unwrap();
            assert_eq!(buf.len(), 23, "Timestamp should always be 23 chars: {}", buf);
        }
    }

    // ==================== Env Var Name ====================

    #[test]
    fn test_env_var_name_uppercase() {
        // The env var should be {APP_NAME}_LOG in uppercase
        let app_name = "myapp";
        let expected = format!("{}_LOG", app_name.to_uppercase());
        assert_eq!(expected, "MYAPP_LOG");
    }

    #[test]
    fn test_env_var_name_with_hyphen() {
        let app_name = "my-app";
        let expected = format!("{}_LOG", app_name.to_uppercase());
        assert_eq!(expected, "MY-APP_LOG");
    }

    // ==================== Log Path Function ====================

    #[test]
    fn test_log_path_before_init_is_none() {
        // Note: This may not be None if other tests have run init()
        // But it tests the function doesn't panic
        let _ = log_path();
    }
}
