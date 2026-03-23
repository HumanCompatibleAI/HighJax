//! CLI client for querying a running Octane instance via zoltar IPC.

use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::cli::args::ZoltarCommands;

/// Base directory for zoltar sockets and file IPC (same as zoltar/mod.rs).
fn zoltar_dir() -> PathBuf {
    std::env::temp_dir().join("octane-zoltar")
}

/// Discovered transport for a running Octane instance.
enum Transport {
    #[cfg(unix)]
    Socket(PathBuf),
    FileIpc(PathBuf),
}

/// Discover the zoltar transport for a running Octane instance.
/// Tries socket first (Unix only), then file IPC directory.
fn discover_transport(pid: Option<u32>) -> Result<Transport> {
    if let Some(pid) = pid {
        // Direct PID: try socket (Unix), then file IPC dir
        #[cfg(unix)]
        {
            let sock = zoltar_dir().join(format!("{}.sock", pid));
            if sock.exists() {
                return Ok(Transport::Socket(sock));
            }
        }
        let dir = zoltar_dir().join(format!("{}", pid));
        if dir.is_dir() {
            return Ok(Transport::FileIpc(dir));
        }
        anyhow::bail!("No zoltar transport for PID {}", pid);
    }

    // Auto-discover via latest file
    let latest = zoltar_dir().join("latest");
    let content = fs::read_to_string(&latest)
        .context("No zoltar discovery file")?;
    let path = PathBuf::from(content.trim());

    if path.is_dir() {
        Ok(Transport::FileIpc(path))
    } else if path.exists() {
        #[cfg(unix)]
        { Ok(Transport::Socket(path)) }
        #[cfg(not(unix))]
        { anyhow::bail!("Discovery path is not a directory (sockets not supported on this platform): {}", path.display()) }
    } else {
        anyhow::bail!("Discovery path does not exist: {}", path.display());
    }
}

/// Send a JSON request via the appropriate transport and return the response.
fn query(transport: &Transport, request: &serde_json::Value) -> Result<serde_json::Value> {
    match transport {
        #[cfg(unix)]
        Transport::Socket(path) => query_socket(path, request),
        Transport::FileIpc(dir) => query_file_ipc(dir, request),
    }
}

/// Send a JSON request over Unix socket.
#[cfg(unix)]
fn query_socket(socket_path: &Path, request: &serde_json::Value) -> Result<serde_json::Value> {
    use std::io::BufRead;
    use std::os::unix::net::UnixStream;

    let stream = UnixStream::connect(socket_path)
        .context(format!("Failed to connect to zoltar at {}", socket_path.display()))?;

    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;

    let mut writer = stream.try_clone()?;
    let mut json = serde_json::to_string(request)?;
    json.push('\n');
    writer.write_all(json.as_bytes())?;
    writer.flush()?;

    let mut reader = std::io::BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line)?;

    if line.is_empty() {
        anyhow::bail!("Empty response from zoltar");
    }

    let value: serde_json::Value = serde_json::from_str(line.trim())?;
    Ok(value)
}

/// Send a JSON request over file IPC.
fn query_file_ipc(dir: &Path, request: &serde_json::Value) -> Result<serde_json::Value> {
    let in_path = dir.join("in");
    let out_path = dir.join("out");

    // Record current out file size before sending
    let out_pos = fs::metadata(&out_path)
        .map(|m| m.len())
        .unwrap_or(0);

    // Append JSON line to in file
    {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&in_path)
            .context("Failed to open zoltar 'in' file")?;
        let mut json = serde_json::to_string(request)?;
        json.push('\n');
        file.write_all(json.as_bytes())?;
        file.flush()?;
    }

    // Poll out file for response
    let timeout = Duration::from_secs(10);
    let start = Instant::now();

    loop {
        if start.elapsed() > timeout {
            anyhow::bail!("Timeout waiting for zoltar response ({}s)", timeout.as_secs());
        }

        let metadata = fs::metadata(&out_path)?;
        if metadata.len() > out_pos {
            let mut file = OpenOptions::new().read(true).open(&out_path)?;
            file.seek(SeekFrom::Start(out_pos))?;

            let mut content = String::new();
            file.read_to_string(&mut content)?;

            // Find first complete line
            if let Some(newline_pos) = content.find('\n') {
                let line = content[..newline_pos].trim();
                if !line.is_empty() {
                    let value: serde_json::Value = serde_json::from_str(line)
                        .context("Failed to parse zoltar response")?;
                    return Ok(value);
                }
            }
        }

        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Run the zoltar CLI subcommand.
pub(crate) fn run_zoltar(pid: Option<u32>, command: ZoltarCommands) -> Result<()> {
    let transport = discover_transport(pid)?;

    let request = match &command {
        ZoltarCommands::Genco => serde_json::json!({"cmd": "genco"}),
        ZoltarCommands::Ping => serde_json::json!({"cmd": "ping"}),
        ZoltarCommands::Press { keys } => serde_json::json!({"cmd": "press", "keys": keys}),
        ZoltarCommands::Navigate { epoch, episode, timestep } => {
            let mut req = serde_json::json!({"cmd": "navigate"});
            if let Some(e) = epoch { req["epoch"] = serde_json::json!(e); }
            if let Some(e) = episode { req["episode"] = serde_json::json!(e); }
            if let Some(t) = timestep { req["timestep"] = serde_json::json!(t); }
            req
        }
        ZoltarCommands::Pane { pane, query: q, path } => {
            let mut req = serde_json::json!({"cmd": "pane", "pane": pane, "query": q});
            if let Some(p) = path { req["path"] = serde_json::json!(p); }
            req
        }
    };

    let response = query(&transport, &request)?;
    println!("{}", serde_json::to_string_pretty(&response)?);
    Ok(())
}
