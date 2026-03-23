//! Zoltar: IPC query interface for external tools.
//!
//! Zoltar lets external tools (airbus, Claude Code, scripts) query Octane's
//! semantic state while it's running. A background thread listens via Unix
//! socket (default on Linux) or file-based IPC (default on Windows). Requests
//! are forwarded to the main event loop via mpsc channels.

pub mod file_ipc;
pub mod protocol;
#[cfg(unix)]
pub mod socket;

use std::path::PathBuf;
use std::sync::mpsc;

pub use protocol::{ZoltarRequest, ZoltarResponse};

/// Base directory for zoltar sockets and file IPC sessions.
/// Reads `ZOLTAR_DIR` env var if set (e.g. when launched inside an airbus bwrap sandbox),
/// otherwise defaults to `{temp}/octane-zoltar/`.
pub(crate) fn zoltar_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("ZOLTAR_DIR") {
        PathBuf::from(dir)
    } else {
        std::env::temp_dir().join("octane-zoltar")
    }
}

/// Discovery file path.
pub(crate) fn latest_path() -> PathBuf {
    zoltar_dir().join("latest")
}

/// Channel pair for zoltar communication.
/// The main event loop receives requests and sends back responses.
pub struct ZoltarChannels {
    /// Receives requests from the zoltar listener thread.
    pub request_rx: mpsc::Receiver<ZoltarRequestEnvelope>,
    /// Kept alive so the listener thread can clone response senders per-request.
    _keepalive: (),
}

/// A request envelope: the parsed request plus a channel to send the response back.
pub struct ZoltarRequestEnvelope {
    pub request: ZoltarRequest,
    pub response_tx: mpsc::Sender<ZoltarResponse>,
}

/// Start the zoltar listener. Returns channels for the main loop to use.
///
/// Transport selection:
/// - `use_file_ipc = false` (default on Linux): Unix socket at `/tmp/octane-zoltar/{pid}.sock`
/// - `use_file_ipc = true` (default on Windows, or `--zoltar-file-ipc`): file IPC at `/tmp/octane-zoltar/{pid}/`
pub fn start(disabled: bool, use_file_ipc: bool) -> Option<ZoltarChannels> {
    if disabled {
        return None;
    }

    // Advertise our zoltar directory via env var so external tools (airbus)
    // can discover it by reading /proc/{pid}/environ.
    let dir = zoltar_dir();
    std::env::set_var("ZOLTAR_DIR", &dir);

    let (request_tx, request_rx) = mpsc::channel();

    let result = if use_file_ipc {
        file_ipc::start_listener(request_tx)
    } else {
        #[cfg(unix)]
        { socket::start_listener(request_tx) }
        #[cfg(not(unix))]
        { file_ipc::start_listener(request_tx) }
    };

    match result {
        Ok(()) => {
            tracing::info!("Zoltar listener started");
            Some(ZoltarChannels {
                request_rx,
                _keepalive: (),
            })
        }
        Err(e) => {
            tracing::warn!("Failed to start zoltar listener: {}", e);
            None
        }
    }
}

/// Clean up zoltar resources on exit. Safe to call for either transport.
pub fn cleanup() {
    #[cfg(unix)]
    socket::cleanup();
    file_ipc::cleanup();
}
