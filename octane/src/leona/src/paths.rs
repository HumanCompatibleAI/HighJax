//! Directory path helpers for log, config, and debug directories.

use std::path::PathBuf;

use crate::APP_NAME;

/// Get the config directory for an app (~/.{app_name})
pub fn config_dir_for(app_name: &str) -> PathBuf {
    dirs::home_dir()
        .expect("Could not find home directory")
        .join(format!(".{}", app_name))
}

/// Get the config directory for the initialized app.
/// Panics if logging hasn't been initialized.
pub fn config_dir() -> PathBuf {
    let app_name = APP_NAME
        .get()
        .expect("Logging not initialized - call leona::init() first");
    config_dir_for(app_name)
}

/// Get the logs directory for an app (~/.highjax/logs/{app_name})
pub fn logs_dir_for(app_name: &str) -> PathBuf {
    dirs::home_dir()
        .expect("Could not find home directory")
        .join(".highjax")
        .join("logs")
        .join(app_name)
}

/// Get the logs directory for the initialized app.
/// Panics if logging hasn't been initialized.
pub fn logs_dir() -> PathBuf {
    let app_name = APP_NAME
        .get()
        .expect("Logging not initialized - call leona::init() first");
    logs_dir_for(app_name)
}

/// Get the debug directory for an app (~/.{app_name}/debug)
pub fn debug_dir_for(app_name: &str) -> PathBuf {
    config_dir_for(app_name).join("debug")
}

/// Get the debug directory for the initialized app.
/// Panics if logging hasn't been initialized.
#[allow(dead_code)]
pub fn debug_dir() -> PathBuf {
    let app_name = APP_NAME
        .get()
        .expect("Logging not initialized - call leona::init() first");
    debug_dir_for(app_name)
}
