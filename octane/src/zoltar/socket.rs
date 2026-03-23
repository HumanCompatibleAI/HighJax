//! Unix socket transport for zoltar.
//!
//! Spawns a background thread that listens on `/tmp/octane-zoltar/{pid}.sock`.
//! Each connection is handled in its own thread: reads JSON lines, forwards
//! them as ZoltarRequestEnvelope to the main loop, and writes back responses.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use tracing::{debug, info, warn};

use super::protocol::{ZoltarRequest, ZoltarResponse};
use super::ZoltarRequestEnvelope;

/// Socket path for current process.
fn socket_path() -> PathBuf {
    super::zoltar_dir().join(format!("{}.sock", std::process::id()))
}

/// Start the Unix socket listener in a background thread.
pub fn start_listener(request_tx: mpsc::Sender<ZoltarRequestEnvelope>) -> anyhow::Result<()> {
    let dir = super::zoltar_dir();
    std::fs::create_dir_all(&dir)?;

    let sock = socket_path();

    // Clean up stale socket
    let _ = std::fs::remove_file(&sock);

    let listener = UnixListener::bind(&sock)?;
    listener.set_nonblocking(true)?;

    // Write discovery file
    std::fs::write(super::latest_path(), sock.to_string_lossy().as_bytes())?;

    info!("Zoltar socket: {}", sock.display());

    // Spawn listener thread
    thread::Builder::new()
        .name("zoltar-listener".into())
        .spawn(move || {
            loop {
                match listener.accept() {
                    Ok((stream, _)) => {
                        debug!("Zoltar client connected");
                        let tx = request_tx.clone();
                        thread::Builder::new()
                            .name("zoltar-conn".into())
                            .spawn(move || {
                                if let Err(e) = handle_connection(stream, tx) {
                                    debug!("Zoltar connection ended: {}", e);
                                }
                            })
                            .ok();
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(50));
                    }
                    Err(e) => {
                        warn!("Zoltar accept error: {}", e);
                        break;
                    }
                }
            }
        })?;

    Ok(())
}

/// Handle a single client connection: read JSON lines, forward to main loop.
fn handle_connection(
    stream: std::os::unix::net::UnixStream,
    request_tx: mpsc::Sender<ZoltarRequestEnvelope>,
) -> anyhow::Result<()> {
    let reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let request: ZoltarRequest = match serde_json::from_str(&line) {
            Ok(req) => req,
            Err(e) => {
                let resp = ZoltarResponse::error(format!("invalid JSON: {}", e));
                write_response(&mut writer, &resp)?;
                continue;
            }
        };

        debug!("Zoltar request: {:?}", request);

        // Forward to main loop and wait for response
        let (response_tx, response_rx) = mpsc::channel();
        let envelope = ZoltarRequestEnvelope {
            request,
            response_tx,
        };

        if request_tx.send(envelope).is_err() {
            // Main loop has exited
            break;
        }

        match response_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(response) => {
                write_response(&mut writer, &response)?;
            }
            Err(_) => {
                let resp = ZoltarResponse::error("timeout waiting for app response");
                write_response(&mut writer, &resp)?;
            }
        }
    }

    Ok(())
}

fn write_response(writer: &mut impl Write, response: &ZoltarResponse) -> anyhow::Result<()> {
    let mut json = serde_json::to_string(response)?;
    json.push('\n');
    writer.write_all(json.as_bytes())?;
    writer.flush()?;
    Ok(())
}

/// Remove the socket file and discovery file.
pub fn cleanup() {
    let sock = socket_path();
    if sock.exists() {
        let _ = std::fs::remove_file(&sock);
        debug!("Removed zoltar socket: {}", sock.display());
    }
    // Only remove latest if it points to our socket
    let latest = super::latest_path();
    if let Ok(content) = std::fs::read_to_string(&latest) {
        if content.trim() == sock.to_string_lossy() {
            let _ = std::fs::remove_file(&latest);
        }
    }
}
