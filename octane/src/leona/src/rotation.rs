//! Log file rotation: age-based and size-based cleanup.

use std::fs;
use std::path::PathBuf;

use chrono::{Duration, Local};

use crate::config::{MAX_LOG_AGE_DAYS, MAX_LOG_DIR_SIZE};

/// Rotate old log files - delete logs older than MAX_LOG_AGE_DAYS
/// and enforce MAX_LOG_DIR_SIZE limit.
pub(crate) fn rotate_logs(logs_dir: &PathBuf) {
    let cutoff = Local::now() - Duration::days(MAX_LOG_AGE_DAYS);
    let cutoff_time = cutoff.timestamp();

    let mut log_files: Vec<(PathBuf, u64, i64)> = Vec::new();
    let mut deleted_count = 0;
    let mut deleted_bytes = 0u64;

    if let Ok(entries) = fs::read_dir(logs_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "log") {
                if let Ok(metadata) = entry.metadata() {
                    let mtime = metadata
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs() as i64)
                        .unwrap_or(0);

                    if mtime < cutoff_time {
                        let size = metadata.len();
                        if fs::remove_file(&path).is_ok() {
                            deleted_count += 1;
                            deleted_bytes += size;
                        }
                    } else {
                        log_files.push((path, metadata.len(), mtime));
                    }
                }
            }
        }
    }

    // Enforce size limit - delete oldest files first
    log_files.sort_by_key(|(_, _, mtime)| *mtime);

    let mut total_size: u64 = log_files.iter().map(|(_, size, _)| size).sum();

    for (path, size, _) in &log_files {
        if total_size <= MAX_LOG_DIR_SIZE {
            break;
        }
        if fs::remove_file(path).is_ok() {
            deleted_count += 1;
            deleted_bytes += size;
            total_size -= size;
        }
    }

    if deleted_count > 0 {
        eprintln!(
            "Log rotation: deleted {} files ({:.1} MB)",
            deleted_count,
            deleted_bytes as f64 / 1024.0 / 1024.0
        );
    }
}
