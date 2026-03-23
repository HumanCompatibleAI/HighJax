//! Trek fixture generation, behavior fixtures, and airbus CLI wrappers.

use std::path::Path;
use std::process::Command;
use std::sync::Arc;

/// Ensure the `airbus` binary is available on PATH. Panics with a clear message if not found.
fn require_airbus() {
    use std::sync::Once;
    static CHECK: Once = Once::new();
    CHECK.call_once(|| {
        if Command::new("airbus").arg("--version").output().is_err() {
            panic!(
                "airbus not found on PATH. Install airbus to run TUI tests: see Octane docs."
            );
        }
    });
}

use arrow::array::{BooleanArray, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

// ─── Trek fixture generation ────────────────────────────────────────────────

/// Generate a simple trek: 1 epoch, 1 episode, 10 timesteps, 2 NPCs, no crash.
pub fn generate_simple_trek(dir: &Path) {
    write_meta_yaml(dir);

    let n_npcs = 2;
    let mut rows = Vec::new();

    for t in 0..10 {
        let t_f = t as f64;
        let ego_x = t_f * 2.0;
        let mut npcs = Vec::new();
        // NPC0: same lane, ahead
        npcs.push((ego_x + 30.0, 2.0, 0.0, 18.0));
        // NPC1: adjacent lane
        npcs.push((ego_x + 15.0, 6.0, 0.0, 22.0));
        rows.push(FrameRow {
            epoch: 0, e: 0, t: t_f,
            crashed: false,
            ego_x, ego_y: 2.0, ego_heading: 0.0, ego_speed: 20.0,
            npcs,
            reward: Some(0.1),
            score: Some(0.5),
        });
    }

    write_parquet(dir, n_npcs, &rows);
}

/// Generate a crash trek: 2 epochs × 2 episodes × 10 timesteps, 5 NPCs.
/// Epoch 0, episode 1 crashes at t=7.
pub fn generate_crash_trek(dir: &Path) {
    write_meta_yaml(dir);

    let n_npcs = 5;
    let mut rows = Vec::new();

    for epoch in 0..2i64 {
        for ep in 0..2i64 {
            let crashes = epoch == 0 && ep == 1;
            for t in 0..10 {
                let t_f = t as f64;
                let ego_x = t_f * 2.0;
                let crashed = crashes && t >= 7;
                let ego_speed = if crashed { 0.0 } else { 20.0 };

                let mut npcs = Vec::new();
                for n in 0..n_npcs {
                    let lane_y = 2.0 + (n as f64) * 4.0; // spread across lanes
                    let npc_x = ego_x + 20.0 + (n as f64) * 10.0;
                    let npc_speed = 15.0 + (n as f64) * 2.0;
                    npcs.push((npc_x, lane_y, 0.0, npc_speed));
                }

                let reward = if crashed { Some(-50.0) } else { Some(0.1) };
                let score = Some(0.5 + epoch as f64 * 0.1);

                rows.push(FrameRow {
                    epoch, e: ep, t: t_f,
                    crashed,
                    ego_x, ego_y: 2.0, ego_heading: 0.0, ego_speed,
                    npcs,
                    reward,
                    score,
                });
            }
        }
    }

    write_parquet(dir, n_npcs, &rows);
}

struct FrameRow {
    epoch: i64,
    e: i64,
    t: f64,
    crashed: bool,
    ego_x: f64,
    ego_y: f64,
    ego_heading: f64,
    ego_speed: f64,
    npcs: Vec<(f64, f64, f64, f64)>, // (x, y, heading, speed)
    reward: Option<f64>,
    score: Option<f64>,
}

fn write_meta_yaml(dir: &Path) {
    std::fs::write(
        dir.join("meta.yaml"),
        "commands:\n  2.highway:\n    seconds_per_t: 0.1\n    n_lanes: 4\n    lane_width: 4.0\n",
    ).unwrap();
}

fn write_parquet(dir: &Path, n_npcs: usize, rows: &[FrameRow]) {
    write_parquet_at(&dir.join("sample_es.parquet"), n_npcs, rows);
}

fn write_parquet_at(path: &Path, n_npcs: usize, rows: &[FrameRow]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let schema = build_schema(n_npcs);
    let batch = build_batch(&schema, n_npcs, rows);

    let file = std::fs::File::create(path).unwrap();
    let props = WriterProperties::builder()
        .set_compression(parquet::basic::Compression::UNCOMPRESSED)
        .build();
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Generate a trek with both sample_es.parquet and a breakdown run.
/// sample_es: 1 epoch (0), 1 episode, 10 timesteps, 2 NPCs.
/// breakdown/test-run/es.parquet: 2 epochs (0, 1), 1 episode each, 5 timesteps, 1 NPC.
pub fn generate_breakdown_trek(dir: &Path) {
    // Write sample_es.p (same as simple trek)
    generate_simple_trek(dir);

    // Write breakdown/test-run/es.p with different data
    let n_npcs = 1;
    let mut rows = Vec::new();
    for epoch in 0..2i64 {
        for t in 0..5 {
            let t_f = t as f64;
            rows.push(FrameRow {
                epoch, e: 0, t: t_f,
                crashed: false,
                ego_x: t_f * 3.0, ego_y: 2.0, ego_heading: 0.0, ego_speed: 25.0,
                npcs: vec![(t_f * 3.0 + 20.0, 6.0, 0.0, 20.0)],
                reward: Some(0.2),
                score: Some(0.8),
            });
        }
    }
    write_parquet_at(&dir.join("breakdown/test-run/es.parquet"), n_npcs, &rows);
}

fn build_schema(n_npcs: usize) -> Schema {
    let mut fields = vec![
        Field::new("epoch", DataType::Int64, false),
        Field::new("e", DataType::Int64, false),
        Field::new("t", DataType::Float64, false),
        Field::new("reward", DataType::Float64, true),
        Field::new("score", DataType::Float64, true),
        Field::new("state.crashed", DataType::Boolean, false),
        Field::new("state.ego_x", DataType::Float64, false),
        Field::new("state.ego_y", DataType::Float64, false),
        Field::new("state.ego_heading", DataType::Float64, false),
        Field::new("state.ego_speed", DataType::Float64, false),
    ];
    for n in 0..n_npcs {
        fields.push(Field::new(format!("state.npc{n}_x"), DataType::Float64, false));
        fields.push(Field::new(format!("state.npc{n}_y"), DataType::Float64, false));
        fields.push(Field::new(format!("state.npc{n}_heading"), DataType::Float64, false));
        fields.push(Field::new(format!("state.npc{n}_speed"), DataType::Float64, false));
    }
    Schema::new(fields)
}

fn build_batch(schema: &Schema, n_npcs: usize, rows: &[FrameRow]) -> RecordBatch {
    let epochs = Int64Array::from(rows.iter().map(|r| r.epoch).collect::<Vec<_>>());
    let episodes = Int64Array::from(rows.iter().map(|r| r.e).collect::<Vec<_>>());
    let timesteps = Float64Array::from(rows.iter().map(|r| r.t).collect::<Vec<_>>());
    let rewards = Float64Array::from(rows.iter().map(|r| r.reward).collect::<Vec<_>>());
    let scores = Float64Array::from(rows.iter().map(|r| r.score).collect::<Vec<_>>());
    let crashed = BooleanArray::from(rows.iter().map(|r| r.crashed).collect::<Vec<_>>());
    let ego_x = Float64Array::from(rows.iter().map(|r| r.ego_x).collect::<Vec<_>>());
    let ego_y = Float64Array::from(rows.iter().map(|r| r.ego_y).collect::<Vec<_>>());
    let ego_heading = Float64Array::from(rows.iter().map(|r| r.ego_heading).collect::<Vec<_>>());
    let ego_speed = Float64Array::from(rows.iter().map(|r| r.ego_speed).collect::<Vec<_>>());

    let mut columns: Vec<Arc<dyn arrow::array::Array>> = vec![
        Arc::new(epochs),
        Arc::new(episodes),
        Arc::new(timesteps),
        Arc::new(rewards),
        Arc::new(scores),
        Arc::new(crashed),
        Arc::new(ego_x),
        Arc::new(ego_y),
        Arc::new(ego_heading),
        Arc::new(ego_speed),
    ];

    for npc_idx in 0..n_npcs {
        let xs = Float64Array::from(
            rows.iter().map(|r| r.npcs.get(npc_idx).map(|p| p.0).unwrap_or(0.0)).collect::<Vec<_>>()
        );
        let ys = Float64Array::from(
            rows.iter().map(|r| r.npcs.get(npc_idx).map(|p| p.1).unwrap_or(0.0)).collect::<Vec<_>>()
        );
        let headings = Float64Array::from(
            rows.iter().map(|r| r.npcs.get(npc_idx).map(|p| p.2).unwrap_or(0.0)).collect::<Vec<_>>()
        );
        let speeds = Float64Array::from(
            rows.iter().map(|r| r.npcs.get(npc_idx).map(|p| p.3).unwrap_or(0.0)).collect::<Vec<_>>()
        );
        columns.push(Arc::new(xs));
        columns.push(Arc::new(ys));
        columns.push(Arc::new(headings));
        columns.push(Arc::new(speeds));
    }

    RecordBatch::try_new(Arc::new(schema.clone()), columns).unwrap()
}

/// Generate a trek with a corrupt (truncated) sample_es.parquet.
/// The parquet has valid magic bytes but invalid content.
pub fn generate_corrupt_trek(dir: &Path) {
    write_meta_yaml(dir);
    // Write a truncated parquet: valid magic at start, but corrupt footer
    let pq_path = dir.join("sample_es.parquet");
    let mut data = Vec::new();
    data.extend_from_slice(b"PAR1"); // magic
    data.extend_from_slice(&[0u8; 100]); // garbage
    data.extend_from_slice(b"PAR1"); // footer magic
    std::fs::write(pq_path, data).unwrap();
}

/// Generate a trek with non-sequential episode numbers (e=10, e=20, e=30).
/// 1 epoch, 3 episodes, 5 timesteps each, 1 NPC.
pub fn generate_sparse_episode_trek(dir: &Path) {
    write_meta_yaml(dir);

    let n_npcs = 1;
    let mut rows = Vec::new();
    for &ep_e in &[10i64, 20, 30] {
        for t in 0..5 {
            let t_f = t as f64;
            rows.push(FrameRow {
                epoch: 0, e: ep_e, t: t_f,
                crashed: false,
                ego_x: t_f * 2.0, ego_y: 2.0, ego_heading: 0.0, ego_speed: 20.0,
                npcs: vec![(t_f * 2.0 + 20.0, 6.0, 0.0, 18.0)],
                reward: Some(0.1),
                score: Some(0.5),
            });
        }
    }
    write_parquet(dir, n_npcs, &rows);
}

/// Get the isolated home directory path for a test session.
pub fn isolated_home(session_name: &str) -> String {
    format!("/tmp/octane-test-{}", session_name)
}

// ─── Behavior fixture generation ─────────────────────────────────────────────

/// Write a behavior JSON file into the isolated home's ~/.highjax/behaviors/highway/ dir.
/// Returns the home directory path.
fn write_behavior_json(session_name: &str, filename: &str, json: serde_json::Value) -> String {
    let home = isolated_home(session_name);
    let behaviors_dir = format!("{}/.highjax/behaviors/highway", home);
    std::fs::create_dir_all(&behaviors_dir).unwrap();
    let path = format!("{}/{}", behaviors_dir, filename);
    std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    home
}

/// Write a test behavior JSON into the isolated home's ~/.highjax/behaviors/highway/ dir.
/// Returns the home directory path so airbus can use it.
pub fn write_test_behavior(session_name: &str) -> String {
    write_behavior_json(session_name, "test-behavior.json", serde_json::json!({
        "name": "test-behavior",
        "description": "Test behavior for airbus tests",
        "scenarios": [
            {
                "actions": ["left", "right"],
                "weight": 1.0,
                "state": {
                    "ego_x": 100.0,
                    "ego_y": 8.0,
                    "ego_speed": 20.0,
                    "ego_heading": 0.0,
                    "npc0_x": 130.0,
                    "npc0_y": 12.0,
                    "npc0_speed": 18.0,
                    "npc0_heading": 0.0,
                    "npc1_x": 80.0,
                    "npc1_y": 4.0,
                    "npc1_speed": 22.0,
                    "npc1_heading": 0.0
                }
            },
            {
                "actions": ["idle"],
                "weight": 0.5,
                "state": {
                    "ego_x": 100.0,
                    "ego_y": 8.0,
                    "ego_speed": 30.0,
                    "ego_heading": 0.0
                }
            }
        ]
    }))
}

/// Write a test behavior JSON with source info for scenario-to-source navigation tests.
/// Creates a behavior called "sourced-behavior" with two scenarios:
/// - Scenario 0: source epoch=0, episode=0, t=5
/// - Scenario 1: no source info (for testing the "no source" path)
pub fn write_test_behavior_with_source(session_name: &str) -> String {
    write_behavior_json(session_name, "sourced-behavior.json", serde_json::json!({
        "name": "sourced-behavior",
        "description": "Behavior with source info for navigation tests",
        "scenarios": [
            {
                "actions": ["left"],
                "weight": 1.0,
                "state": {
                    "ego_x": 10.0,
                    "ego_y": 2.0,
                    "ego_speed": 20.0,
                    "ego_heading": 0.0,
                    "npc0_x": 40.0,
                    "npc0_y": 2.0,
                    "npc0_speed": 18.0,
                    "npc0_heading": 0.0,
                    "npc1_x": 25.0,
                    "npc1_y": 6.0,
                    "npc1_speed": 22.0,
                    "npc1_heading": 0.0
                },
                "source": {
                    "epoch": 0,
                    "episode": 0,
                    "t": 5
                }
            },
            {
                "actions": ["idle"],
                "weight": 0.5,
                "state": {
                    "ego_x": 100.0,
                    "ego_y": 8.0,
                    "ego_speed": 30.0,
                    "ego_heading": 0.0
                }
            }
        ]
    }))
}

/// Write a sourced behavior targeting the crash frame in `generate_crash_trek`.
/// Source: epoch=0, episode=1, t=8 (crashed, ego_speed=0, reward=-50).
pub fn write_crash_source_behavior(session_name: &str) -> String {
    write_behavior_json(session_name, "sourced-behavior.json", serde_json::json!({
        "name": "sourced-behavior",
        "description": "Behavior sourced from crash frame",
        "scenarios": [
            {
                "actions": ["left"],
                "weight": 1.0,
                "state": {
                    "ego_x": 16.0,
                    "ego_y": 2.0,
                    "ego_speed": 0.0,
                    "ego_heading": 0.0,
                    "npc0_x": 36.0,
                    "npc0_y": 2.0,
                    "npc0_speed": 15.0,
                    "npc0_heading": 0.0,
                    "npc1_x": 46.0,
                    "npc1_y": 6.0,
                    "npc1_speed": 17.0,
                    "npc1_heading": 0.0,
                    "npc2_x": 56.0,
                    "npc2_y": 10.0,
                    "npc2_speed": 19.0,
                    "npc2_heading": 0.0,
                    "npc3_x": 66.0,
                    "npc3_y": 14.0,
                    "npc3_speed": 21.0,
                    "npc3_heading": 0.0,
                    "npc4_x": 76.0,
                    "npc4_y": 18.0,
                    "npc4_speed": 23.0,
                    "npc4_heading": 0.0
                },
                "source": {
                    "epoch": 0,
                    "episode": 1,
                    "t": 8
                }
            },
            {
                "actions": ["idle"],
                "weight": 0.5,
                "state": {
                    "ego_x": 100.0,
                    "ego_y": 8.0,
                    "ego_speed": 30.0,
                    "ego_heading": 0.0
                }
            }
        ]
    }))
}

/// Write a sourced behavior with `"target"` key pointing to the given trek directory.
/// Source: epoch=0, episode=1, t=8 (crash frame in `generate_crash_trek`).
pub fn write_target_source_behavior(session_name: &str, trek_path: &Path) -> String {
    write_behavior_json(session_name, "sourced-behavior.json", serde_json::json!({
        "name": "sourced-behavior",
        "scenarios": [
            {
                "actions": ["left"],
                "weight": 1.0,
                "state": {
                    "ego_x": 16.0,
                    "ego_y": 2.0,
                    "ego_speed": 0.0,
                    "ego_heading": 0.0,
                    "npc0_x": 36.0,
                    "npc0_y": 2.0,
                    "npc0_speed": 15.0,
                    "npc0_heading": 0.0
                },
                "source": {
                    "target": trek_path.display().to_string(),
                    "epoch": 0,
                    "episode": 1,
                    "t": 8
                }
            },
            {
                "actions": ["idle"],
                "weight": 0.5,
                "state": {
                    "ego_x": 100.0,
                    "ego_y": 8.0,
                    "ego_speed": 30.0,
                    "ego_heading": 0.0
                }
            }
        ]
    }))
}

/// Write a sourced behavior using legacy `"trek"` key (backward compat test).
/// Source: epoch=0, episode=1, t=8 (crash frame in `generate_crash_trek`).
pub fn write_legacy_trek_source_behavior(session_name: &str, trek_path: &Path) -> String {
    write_behavior_json(session_name, "sourced-behavior.json", serde_json::json!({
        "name": "sourced-behavior",
        "scenarios": [
            {
                "actions": ["left"],
                "weight": 1.0,
                "state": {
                    "ego_x": 16.0,
                    "ego_y": 2.0,
                    "ego_speed": 0.0,
                    "ego_heading": 0.0,
                    "npc0_x": 36.0,
                    "npc0_y": 2.0,
                    "npc0_speed": 15.0,
                    "npc0_heading": 0.0
                },
                "source": {
                    "trek": trek_path.display().to_string(),
                    "epoch": 0,
                    "episode": 1,
                    "t": 8
                }
            },
            {
                "actions": ["idle"],
                "weight": 0.5,
                "state": {
                    "ego_x": 100.0,
                    "ego_y": 8.0,
                    "ego_speed": 30.0,
                    "ego_heading": 0.0
                }
            }
        ]
    }))
}

/// Write a sourced behavior with `"target"` pointing to a parquet file.
/// Source: epoch=0, episode=1, t=8 (crash frame in `generate_crash_trek`).
pub fn write_pq_target_source_behavior(session_name: &str, trek_path: &Path) -> String {
    let pq_path = trek_path.join("sample_es.parquet");
    write_behavior_json(session_name, "sourced-behavior.json", serde_json::json!({
        "name": "sourced-behavior",
        "scenarios": [
            {
                "actions": ["left"],
                "weight": 1.0,
                "state": {
                    "ego_x": 16.0,
                    "ego_y": 2.0,
                    "ego_speed": 0.0,
                    "ego_heading": 0.0,
                    "npc0_x": 36.0,
                    "npc0_y": 2.0,
                    "npc0_speed": 15.0,
                    "npc0_heading": 0.0
                },
                "source": {
                    "target": pq_path.display().to_string(),
                    "epoch": 0,
                    "episode": 1,
                    "t": 8
                }
            },
            {
                "actions": ["idle"],
                "weight": 0.5,
                "state": {
                    "ego_x": 100.0,
                    "ego_y": 8.0,
                    "ego_speed": 30.0,
                    "ego_heading": 0.0
                }
            }
        ]
    }))
}

// ─── Airbus CLI wrappers ────────────────────────────────────────────────────

/// Sandbox mode for the current platform.
fn sandbox_mode() -> &'static str {
    if cfg!(target_os = "linux") { "yes" } else { "gutter" }
}

/// Spawn octane in an airbus session with sandboxing and config isolation.
///
/// On Linux: uses bwrap sandbox (`--sandbox yes`) with `HOME=...` env prefix.
/// On Windows: uses gutter sandbox (`--sandbox gutter`) which redirects HOME
/// to a temp dir. The `HOME=...` env prefix is extracted by the gutter server.
/// Returns the session name on success.
pub fn spawn_octane(trek_path: &Path, session_name: &str, extra_args: &[&str]) {
    require_airbus();
    let octane_bin = super::binary_path();
    let isolated_home = isolated_home(session_name);
    std::fs::create_dir_all(&isolated_home).ok();

    let mut cmd_parts = vec![
        format!("HOME={}", isolated_home),
        octane_bin.display().to_string(),
        "-t".to_string(),
        trek_path.display().to_string(),
    ];
    for arg in extra_args {
        cmd_parts.push(arg.to_string());
    }
    let cmd_str = cmd_parts.join(" ");

    let output = Command::new("airbus")
        .args(["spawn", &cmd_str, "--sandbox", sandbox_mode(), "--name", session_name, "--size", "120x40"])
        .output()
        .expect("Failed to run airbus spawn — is airbus installed?");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!("airbus spawn failed:\nstdout: {}\nstderr: {}", stdout, stderr);
    }
}

/// Resolve a path written inside a sandboxed airbus session to its actual location.
///
/// On Linux (bwrap): writes land in `/tmp/airbus-overlay-{session}/{category}/upper/{rest}`.
/// On Windows (gutter): writes to HOME land in `{temp}/airbus-gutter-{session}/home/{rest}`.
pub fn overlay_path(session_name: &str, sandboxed_path: &str) -> std::path::PathBuf {
    if cfg!(target_os = "linux") {
        // Bwrap overlay categories
        let categories = ["/tmp/", "/home/", "/etc/", "/root/", "/var/", "/opt/", "/srv/"];
        for cat in &categories {
            if let Some(rest) = sandboxed_path.strip_prefix(cat) {
                let cat_name = cat.trim_matches('/');
                return std::path::PathBuf::from(format!(
                    "/tmp/airbus-overlay-{}/{}/upper/{}",
                    session_name, cat_name, rest
                ));
            }
        }
        panic!(
            "Path '{}' is not under an overlay-mounted directory ({:?})",
            sandboxed_path, categories
        );
    } else {
        // Gutter: the file is under the gutter home.
        // The sandboxed_path is "{isolated_home}/relative/path". Strip the
        // isolated_home prefix and prepend the gutter home.
        let home = isolated_home(session_name);
        let relative = sandboxed_path.strip_prefix(&home)
            .unwrap_or_else(|| panic!(
                "Gutter overlay_path: '{}' does not start with isolated home '{}'",
                sandboxed_path, home
            ));
        let relative = relative.trim_start_matches('/').trim_start_matches('\\');
        let gutter_home = std::env::temp_dir()
            .join(format!("airbus-gutter-{}", session_name))
            .join("home");
        gutter_home.join(relative)
    }
}

/// Send keypresses to an airbus session.
pub fn press(session: &str, keys: &[&str]) {
    let mut args = vec!["-s", session, "press"];
    args.extend_from_slice(keys);

    let output = Command::new("airbus")
        .args(&args)
        .output()
        .expect("Failed to run airbus press");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("airbus press failed: {}", stderr);
    }
}

/// Type text as keypresses (handles uppercase correctly unlike `press`).
pub fn type_text(session: &str, text: &str) {
    let output = Command::new("airbus")
        .args(["-s", session, "type", text])
        .output()
        .expect("Failed to run airbus type");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("airbus type failed: {}", stderr);
    }
}

/// Like type_text but uses `--` to prevent flag parsing (needed for text starting with `-`).
pub fn type_text_raw(session: &str, text: &str) {
    let output = Command::new("airbus")
        .args(["-s", session, "type", "--", text])
        .output()
        .expect("Failed to run airbus type");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("airbus type_raw failed: {}", stderr);
    }
}

/// Send keypresses with a delay between each.
pub fn press_with_delay(session: &str, keys: &[&str], delay_ms: u32) {
    let mut args = vec!["-s".to_string(), session.to_string(), "press".to_string(),
                        "--delay".to_string(), delay_ms.to_string()];
    for k in keys {
        args.push(k.to_string());
    }

    let output = Command::new("airbus")
        .args(&args)
        .output()
        .expect("Failed to run airbus press");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("airbus press failed: {}", stderr);
    }
}

/// Dump plain text from an airbus session.
pub fn dump_text(session: &str) -> String {
    let output = Command::new("airbus")
        .args(["-s", session, "dump", "text"])
        .output()
        .expect("Failed to run airbus dump");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("airbus dump text failed: {}", stderr);
    }

    String::from_utf8_lossy(&output.stdout).to_string()
}

/// Run a batch of commands, return final dump.
pub fn batch(session: &str, commands: &str) -> String {
    let output = Command::new("airbus")
        .args(["-s", session, "batch", commands])
        .output()
        .expect("Failed to run airbus batch");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("airbus batch failed: {}", stderr);
    }

    String::from_utf8_lossy(&output.stdout).to_string()
}

/// Stop an airbus session and clean up.
pub fn stop_session(session: &str) {
    let _ = Command::new("airbus")
        .args(["-s", session, "stop", "--cleanup"])
        .output();
}

/// Sleep briefly to let the TUI render a frame.
/// Only used when no specific text predicate is available (e.g. Tab cycling
/// that changes highlighting but not text content).
pub fn settle() {
    std::thread::sleep(std::time::Duration::from_millis(200));
}

/// Wait until the screen dump contains the given text, or panic after timeout.
pub fn wait_for_text(session: &str, needle: &str, timeout_secs: u64) {
    let timeout_secs = timeout_secs.max(30); // floor: dump_text shells out and is slow under parallel load
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    loop {
        let screen = dump_text(session);
        if screen.contains(needle) {
            return;
        }
        if std::time::Instant::now() >= deadline {
            panic!("Timed out waiting for '{}' in screen:\n{}", needle, screen);
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// Wait until the screen dump contains any of the given texts, or panic after timeout.
/// Returns the screen content on success.
pub fn wait_for_any_text(session: &str, needles: &[&str], timeout_secs: u64) -> String {
    let timeout_secs = timeout_secs.max(30);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    loop {
        let screen = dump_text(session);
        if needles.iter().any(|n| screen.contains(n)) {
            return screen;
        }
        if std::time::Instant::now() >= deadline {
            panic!("Timed out waiting for any of {:?} in screen:\n{}", needles, screen);
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// Run a zoltar command against an airbus session, returning (success, stdout, stderr).
pub fn zoltar(session: &str, args: &[&str]) -> (bool, String, String) {
    let mut cmd_args = vec!["-s", session, "zoltar"];
    cmd_args.extend_from_slice(args);

    let output = Command::new("airbus")
        .args(&cmd_args)
        .output()
        .expect("Failed to run airbus zoltar");

    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

/// Run a zoltar query and parse the JSON response. Panics on failure.
pub fn zoltar_json(session: &str, args: &[&str]) -> serde_json::Value {
    let (success, stdout, stderr) = zoltar(session, args);
    assert!(
        success,
        "zoltar {:?} failed.\nstdout: {}\nstderr: {}",
        args, stdout, stderr
    );
    serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("Failed to parse zoltar JSON: {}\nstdout: {}", e, stdout))
}

/// Get a genco state snapshot via zoltar.
pub fn zoltar_genco(session: &str) -> serde_json::Value {
    zoltar_json(session, &["genco"])
}

/// Try to get a genco snapshot, returning None on connection/parse failure
/// (e.g. zoltar socket not ready yet).
pub fn try_zoltar_genco(session: &str) -> Option<serde_json::Value> {
    let (success, stdout, _stderr) = zoltar(session, &["genco"]);
    if !success {
        return None;
    }
    serde_json::from_str(stdout.trim()).ok()
}

/// Try to query a pane, returning None on connection/parse failure.
pub fn try_zoltar_pane(session: &str, pane: &str, query: &str) -> Option<serde_json::Value> {
    let (success, stdout, _stderr) = zoltar(session, &["pane", pane, query]);
    if !success {
        return None;
    }
    serde_json::from_str(stdout.trim()).ok()
}

/// Query a pane via zoltar.
pub fn zoltar_pane(session: &str, pane: &str, query: &str) -> serde_json::Value {
    zoltar_json(session, &["pane", pane, query])
}

/// Poll a zoltar pane until a predicate is satisfied, or panic after timeout.
pub fn wait_for_pane(
    session: &str,
    pane: &str,
    predicate: impl Fn(&serde_json::Value) -> bool,
    description: &str,
    timeout_secs: u64,
) {
    let timeout_secs = timeout_secs.max(30);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    let mut last_data = None;
    loop {
        if let Some(data) = try_zoltar_pane(session, pane, "data") {
            if predicate(&data) {
                return;
            }
            last_data = Some(data);
        }
        if std::time::Instant::now() >= deadline {
            panic!(
                "Timed out waiting for pane '{}' predicate '{}'. Last data:\n{}",
                pane, description,
                last_data.map(|d| serde_json::to_string_pretty(&d).unwrap())
                    .unwrap_or_else(|| "(zoltar never connected)".to_string())
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// Poll genco until a predicate is satisfied, or panic after timeout.
pub fn wait_for_genco(
    session: &str,
    description: &str,
    predicate: impl Fn(&serde_json::Value) -> bool,
    timeout_secs: u64,
) {
    let timeout_secs = timeout_secs.max(30);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    let mut last_genco = None;
    loop {
        if let Some(genco) = try_zoltar_genco(session) {
            if predicate(&genco) {
                return;
            }
            last_genco = Some(genco);
        }
        if std::time::Instant::now() >= deadline {
            panic!(
                "Timed out waiting for genco predicate '{}'. Last genco:\n{}",
                description,
                last_genco.map(|g| serde_json::to_string_pretty(&g).unwrap())
                    .unwrap_or_else(|| "(zoltar never connected)".to_string())
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// Wait until genco's `last_toast` field contains the given text, or panic after timeout.
/// Unlike `wait_for_text`, this checks the persistent `last_toast_text` field in the app state
/// rather than the screen buffer, so it's immune to short-lived toast messages expiring before
/// a slow `dump_text` can capture them.
pub fn wait_for_toast(session: &str, needle: &str, timeout_secs: u64) {
    let timeout_secs = timeout_secs.max(30);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    let mut last_toast_val = None;
    loop {
        if let Some(genco) = try_zoltar_genco(session) {
            if let Some(toast) = genco["last_toast"].as_str() {
                if toast.contains(needle) {
                    return;
                }
            }
            last_toast_val = Some(genco["last_toast"].clone());
        }
        if std::time::Instant::now() >= deadline {
            let screen = dump_text(session);
            panic!(
                "Timed out waiting for toast '{}'. last_toast: {:?}\nscreen:\n{}",
                needle,
                last_toast_val.unwrap_or(serde_json::Value::Null),
                screen
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// Wait until the screen dump no longer contains the given text, or panic after timeout.
pub fn wait_for_text_absent(session: &str, needle: &str, timeout_secs: u64) {
    let timeout_secs = timeout_secs.max(30);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    loop {
        let screen = dump_text(session);
        if !screen.contains(needle) {
            return;
        }
        if std::time::Instant::now() >= deadline {
            panic!("Timed out waiting for '{}' to disappear from screen:\n{}", needle, screen);
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}
