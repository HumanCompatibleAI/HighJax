//! File-based IPC transport for zoltar.
//!
//! Creates a session directory at `/tmp/octane-zoltar/{pid}/` with `in` and `out`
//! files. A background thread polls the `in` file for JSON line requests and
//! writes JSON line responses to `out`. Same channel pattern as socket.rs.
//!
//! Works on all platforms, default on Windows.

use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use tracing::{debug, info, warn};

use super::protocol::{ZoltarRequest, ZoltarResponse};
use super::ZoltarRequestEnvelope;

/// Session directory for current process.
fn session_dir() -> PathBuf {
    super::zoltar_dir().join(format!("{}", std::process::id()))
}

/// Start the file IPC listener in a background thread.
pub fn start_listener(request_tx: mpsc::Sender<ZoltarRequestEnvelope>) -> anyhow::Result<()> {
    let dir = session_dir();
    fs::create_dir_all(&dir)?;

    let in_path = dir.join("in");
    let out_path = dir.join("out");

    // Create empty files (truncate if stale from previous run)
    fs::write(&in_path, "")?;
    fs::write(&out_path, "")?;

    // Write discovery file
    let latest = super::latest_path();
    fs::write(&latest, dir.to_string_lossy().as_bytes())?;

    info!("Zoltar file IPC: {}", dir.display());

    thread::Builder::new()
        .name("zoltar-file-ipc".into())
        .spawn(move || {
            let mut in_pos: u64 = 0;

            loop {
                match read_next_command(&in_path, &mut in_pos) {
                    Ok(Some(line)) => {
                        let request: ZoltarRequest = match serde_json::from_str(&line) {
                            Ok(req) => req,
                            Err(e) => {
                                let resp =
                                    ZoltarResponse::error(format!("invalid JSON: {}", e));
                                if let Err(e) = write_response(&out_path, &resp) {
                                    warn!("Failed to write error response: {}", e);
                                }
                                continue;
                            }
                        };

                        debug!("Zoltar file IPC request: {:?}", request);

                        let (response_tx, response_rx) = mpsc::channel();
                        let envelope = ZoltarRequestEnvelope {
                            request,
                            response_tx,
                        };

                        if request_tx.send(envelope).is_err() {
                            break; // Main loop has exited
                        }

                        match response_rx.recv_timeout(Duration::from_secs(5)) {
                            Ok(response) => {
                                if let Err(e) = write_response(&out_path, &response) {
                                    warn!("Failed to write response: {}", e);
                                }
                            }
                            Err(_) => {
                                let resp =
                                    ZoltarResponse::error("timeout waiting for app response");
                                let _ = write_response(&out_path, &resp);
                            }
                        }
                    }
                    Ok(None) => {
                        thread::sleep(Duration::from_millis(50));
                    }
                    Err(e) => {
                        warn!("Zoltar file IPC read error: {}", e);
                        thread::sleep(Duration::from_millis(50));
                    }
                }
            }
        })?;

    Ok(())
}

/// Read the next complete JSON line from the `in` file, starting at `pos`.
/// Only advances `pos` past a complete line (terminated by `\n`).
fn read_next_command(in_path: &Path, pos: &mut u64) -> anyhow::Result<Option<String>> {
    let metadata = fs::metadata(in_path)?;
    if metadata.len() <= *pos {
        return Ok(None);
    }

    let mut file = OpenOptions::new().read(true).open(in_path)?;
    file.seek(SeekFrom::Start(*pos))?;

    let mut content = String::new();
    file.read_to_string(&mut content)?;

    // Find the first complete line (terminated by \n)
    if let Some(newline_pos) = content.find('\n') {
        let line = content[..newline_pos].trim().to_string();
        *pos += (newline_pos + 1) as u64;
        if line.is_empty() {
            return Ok(None);
        }
        return Ok(Some(line));
    }

    // No complete line yet (partial write)
    Ok(None)
}

/// Write a JSON line response to the `out` file.
fn write_response(out_path: &Path, response: &ZoltarResponse) -> anyhow::Result<()> {
    let mut json = serde_json::to_string(response)?;
    json.push('\n');

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_path)?;
    file.write_all(json.as_bytes())?;
    file.flush()?;
    Ok(())
}

/// Remove the session directory and discovery file.
pub fn cleanup() {
    let dir = session_dir();
    if dir.exists() {
        let _ = fs::remove_dir_all(&dir);
        debug!("Removed zoltar file IPC dir: {}", dir.display());
    }
    // Only remove latest if it points to our directory
    let latest = super::latest_path();
    if let Ok(content) = fs::read_to_string(&latest) {
        if content.trim() == dir.to_string_lossy() {
            let _ = fs::remove_file(&latest);
        }
    }
}
