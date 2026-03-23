//! Behavior file discovery and loading utilities.
//!
//! Shared between the draw CLI command and the behavior explorer.

use std::path::{Path, PathBuf};

use tracing::{info, warn};

use crate::data::{ActionDistribution, FrameState, VehicleState};
use crate::envs::EnvType;

/// Get the user behaviors directory path for a specific environment.
pub fn behaviors_dir(env_type: EnvType) -> PathBuf {
    let root = if let Ok(highjax_home) = std::env::var("HIGHJAX_HOME") {
        PathBuf::from(highjax_home).join("behaviors")
    } else {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        PathBuf::from(home).join(".highjax").join("behaviors")
    };
    root.join(env_type.env_name())
}

/// Find a behavior JSON file by name, checking user dir and preset locations.
/// If `env_type` is provided, searches that env only; otherwise searches all env types.
pub fn find_behavior_json(name: &str, env_type: Option<EnvType>) -> Option<PathBuf> {
    let filename = format!("{}.json", name);
    let env = env_type.unwrap_or(EnvType::Highway);

    // 1. User behaviors dir (~/.highjax/behaviors/{env_name}/)
    let user_path = behaviors_dir(env).join(&filename);
    if user_path.exists() {
        return Some(user_path);
    }

    // 2. Preset behaviors via Python package path
    if let Ok(output) = std::process::Command::new("python3")
        .args(["-c", "import highjax; print(highjax.__path__[0])"])
        .output()
    {
        if output.status.success() {
            let pkg_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let preset = PathBuf::from(&pkg_path)
                .join(env.behaviors_subpath())
                .join(&filename);
            if preset.exists() {
                return Some(preset);
            }
        }
    }

    None
}

/// Find the preset behaviors directory via the Python package path.
/// If `env_type` is provided, returns that env's preset dir; otherwise defaults to Highway.
pub fn preset_behaviors_dir(env_type: Option<EnvType>) -> Option<PathBuf> {
    let subpath = env_type.unwrap_or(EnvType::Highway).behaviors_subpath();
    let output = std::process::Command::new("python3")
        .args(["-c", "import highjax; print(highjax.__path__[0])"])
        .output()
        .ok()?;
    if output.status.success() {
        let pkg_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let dir = PathBuf::from(pkg_path).join(subpath);
        if dir.is_dir() {
            return Some(dir);
        }
    }
    None
}

/// Build a FrameState from a behavior scenario's flat state dict.
///
/// Reads absolute positions from keys like `ego_x`, `ego_y`, `npc0_x`, etc.
/// Falls back to sensible defaults when keys are missing.
pub fn build_frame_from_state(
    state: &serde_json::Value,
    n_lanes: usize,
    lane_width: f64,
) -> FrameState {
    let ego_x = state.get("ego_x").and_then(|v| v.as_f64()).unwrap_or(100.0);
    let ego_y = state.get("ego_y").and_then(|v| v.as_f64())
        .unwrap_or((n_lanes / 2) as f64 * lane_width);
    let ego_speed = state.get("ego_speed").and_then(|v| v.as_f64())
        .unwrap_or(20.0);
    let ego_heading = state.get("ego_heading").and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    let ego = VehicleState {
        x: ego_x,
        y: ego_y,
        heading: ego_heading,
        speed: ego_speed,
        acceleration: 0.0,
        attention: None,
    };

    let mut npcs = Vec::new();
    for i in 0.. {
        let key = format!("npc{}_x", i);
        let Some(npc_x) = state.get(&key).and_then(|v| v.as_f64()) else {
            break;
        };
        let npc_y = state.get(&format!("npc{}_y", i))
            .and_then(|v| v.as_f64()).unwrap_or(ego_y);
        let npc_speed = state.get(&format!("npc{}_speed", i))
            .and_then(|v| v.as_f64()).unwrap_or(ego_speed);
        let npc_heading = state.get(&format!("npc{}_heading", i))
            .and_then(|v| v.as_f64()).unwrap_or(0.0);

        npcs.push(VehicleState {
            x: npc_x,
            y: npc_y,
            heading: npc_heading,
            speed: npc_speed,
            acceleration: 0.0,
            attention: None,
        });
    }

    FrameState {
        crashed: false,
        ego,
        npcs,
        action_distribution: None,
        chosen_action: None,
        old_action_distribution: None,
    }
}

/// Infer which lane the ego should be on so that all NPC positions
/// (ego_y + rel_y) fall within road bounds [−lane_width/2 .. (n_lanes−0.5)*lane_width].
pub fn infer_ego_lane(rel_ys: &[f64], n_lanes: usize, lane_width: f64) -> usize {
    if rel_ys.is_empty() || n_lanes == 0 {
        return n_lanes / 2;
    }
    let min_ry = rel_ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ry = rel_ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // ego_y = L * lane_width
    // Need: ego_y + min_ry >= -lane_width/2  →  L >= (-lane_width/2 - min_ry) / lane_width
    // Need: ego_y + max_ry <= (n_lanes-0.5)*lane_width  →  L <= ((n_lanes-0.5)*lane_width - max_ry) / lane_width
    let lower = (-lane_width / 2.0 - min_ry) / lane_width;
    let upper = ((n_lanes as f64 - 0.5) * lane_width - max_ry) / lane_width;

    let lane = lower.ceil() as i64;
    let lane = lane.max(0).min(n_lanes as i64 - 1) as usize;

    // If both bounds can't be satisfied (NPCs span more than road width),
    // pick the midpoint of the feasible range.
    if lower > upper {
        let mid = ((lower + upper) / 2.0).round() as i64;
        return mid.max(0).min(n_lanes as i64 - 1) as usize;
    }

    lane
}

/// List all behavior entries (user + preset) for a specific env type.
/// Sorted: user first, then presets.
pub fn list_all_behaviors(env_type: EnvType) -> Vec<(String, PathBuf, usize, String, bool)> {
    let mut entries = Vec::new();

    // User behaviors for this env
    let user_dir = behaviors_dir(env_type);
    if user_dir.is_dir() {
        if let Ok(read_dir) = std::fs::read_dir(&user_dir) {
            for entry in read_dir.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "json") {
                    if let Some(info) = parse_behavior_file(&path) {
                        entries.push((info.0, path, info.1, info.2, false));
                    }
                }
            }
        }
    }

    // Preset behaviors for this env
    if let Some(preset_dir) = preset_behaviors_dir(Some(env_type)) {
        if let Ok(read_dir) = std::fs::read_dir(&preset_dir) {
            for entry in read_dir.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "json") {
                    // Skip if user already has a behavior with same name
                    let name = path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_string();
                    if entries.iter().any(|(n, ..)| n == &name) {
                        continue;
                    }
                    if let Some(info) = parse_behavior_file(&path) {
                        entries.push((info.0, path, info.1, info.2, true));
                    }
                }
            }
        }
    }

    // Sort: user behaviors first (alphabetical), then presets (alphabetical)
    entries.sort_by(|a, b| {
        a.4.cmp(&b.4).then_with(|| a.0.cmp(&b.0))
    });

    entries
}

fn parse_behavior_file(path: &std::path::Path) -> Option<(String, usize, String)> {
    let text = std::fs::read_to_string(path).ok()?;
    // Python's json.dump can write NaN for float('nan'), which isn't valid JSON.
    let text = sanitize_json_nans(&text);
    let data: serde_json::Value = serde_json::from_str(&text).ok()?;
    let name = data["name"].as_str().unwrap_or("?").to_string();
    let n_scenarios = data["scenarios"].as_array().map(|a| a.len()).unwrap_or(0);
    let description = data["description"].as_str().unwrap_or("").to_string();
    Some((name, n_scenarios, description))
}

/// Load scenarios from a behavior JSON file.
pub fn load_behavior_scenarios(path: &std::path::Path) -> Option<Vec<serde_json::Value>> {
    let text = std::fs::read_to_string(path).ok()?;
    let text = sanitize_json_nans(&text);
    let data: serde_json::Value = serde_json::from_str(&text).ok()?;
    data["scenarios"].as_array().cloned()
}

/// Replace JSON-invalid NaN/Infinity literals with null.
/// Python's json.dump with allow_nan=True (the default) can produce these.
fn sanitize_json_nans(text: &str) -> String {
    // Replace `: NaN` with `: null` (covers NaN as a JSON value)
    // Also handle Infinity and -Infinity
    let mut result = String::with_capacity(text.len());
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        result.push(c);
        if c == ':' {
            // Skip whitespace after colon
            let mut ws = String::new();
            while let Some(&(_, wc)) = chars.peek() {
                if wc == ' ' || wc == '\t' {
                    ws.push(wc);
                    chars.next();
                } else {
                    break;
                }
            }
            result.push_str(&ws);

            // Check what follows
            let rest = &text[i + 1 + ws.len()..];
            if rest.starts_with("NaN") {
                let after = rest.as_bytes().get(3).copied().unwrap_or(b',');
                if !after.is_ascii_alphanumeric() {
                    result.push_str("null");
                    chars.nth(2); // skip "NaN"
                    continue;
                }
            }
            if rest.starts_with("-Infinity") || rest.starts_with("Infinity") {
                let skip = if rest.starts_with('-') { 9 } else { 8 };
                result.push_str("null");
                for _ in 0..skip {
                    chars.next();
                }
                continue;
            }
        }
    }
    result
}

/// Call Python to get action probabilities for all scenarios of a behavior.
///
/// Runs `highjax brain p-by-action` with JSON output and parses the result
/// into one `ActionDistribution` per scenario.
pub fn fetch_action_probs(
    trek_path: &Path,
    epoch: i64,
    behavior_name: &str,
) -> Option<Vec<ActionDistribution>> {
    info!(
        "Fetching action probs for behavior '{}' at epoch {}",
        behavior_name, epoch
    );
    let output = std::process::Command::new("python3")
        .args([
            "-m",
            "highjax",
            "brain",
            "-t",
            &trek_path.to_string_lossy(),
            "--epoch",
            &epoch.to_string(),
            "-b",
            behavior_name,
            "-f",
            "json",
            "--value-only",
            "p-by-action",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        warn!("highjax brain p-by-action failed: {}", stderr.trim());
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let result = parse_action_probs_json(stdout.trim())?;
    info!(
        "Got action probs for {} scenarios of '{}'",
        result.len(),
        behavior_name
    );
    Some(result)
}

/// Parse the JSON output of `highjax brain p-by-action` into action distributions.
fn parse_action_probs_json(json_str: &str) -> Option<Vec<ActionDistribution>> {
    let data: Vec<serde_json::Value> = serde_json::from_str(json_str).ok()?;

    let mut result = Vec::new();
    for entry in &data {
        let obj = entry.as_object()?;
        let probs: Vec<(String, f64)> = obj
            .iter()
            .filter(|(k, _)| k.as_str() != "scenario")
            .map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(0.0)))
            .collect();
        result.push(ActionDistribution { probs });
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_frame_from_state_explicit_position() {
        let state = serde_json::json!({
            "ego_x": 100.0,
            "ego_y": 8.0,
            "ego_speed": 25.0,
            "ego_heading": 0.0
        });
        let frame = build_frame_from_state(&state, 4, 4.0);
        assert_eq!(frame.ego.x, 100.0);
        assert_eq!(frame.ego.y, 8.0);
        assert_eq!(frame.ego.speed, 25.0);
        assert!(!frame.crashed);
        assert!(frame.npcs.is_empty());
    }

    #[test]
    fn test_build_frame_from_state_with_npcs() {
        let state = serde_json::json!({
            "ego_x": 100.0,
            "ego_y": 8.0,
            "ego_speed": 20.0,
            "npc0_x": 130.0,
            "npc0_y": 12.0,
            "npc0_speed": 18.0,
            "npc1_x": 80.0,
            "npc1_y": 4.0,
            "npc1_speed": 22.0
        });
        let frame = build_frame_from_state(&state, 4, 4.0);
        assert_eq!(frame.npcs.len(), 2);
        assert_eq!(frame.npcs[0].x, 130.0);
        assert_eq!(frame.npcs[0].y, 12.0);
        assert_eq!(frame.npcs[0].speed, 18.0);
        assert_eq!(frame.npcs[1].x, 80.0);
        assert_eq!(frame.npcs[1].y, 4.0);
        assert_eq!(frame.npcs[1].speed, 22.0);
    }

    #[test]
    fn test_build_frame_from_state_defaults() {
        // No ego_x/ego_y → defaults: x=100, y = n_lanes/2 * lane_width
        let state = serde_json::json!({
            "ego_speed": 15.0
        });
        let frame = build_frame_from_state(&state, 3, 4.0);
        assert_eq!(frame.ego.speed, 15.0);
        assert_eq!(frame.ego.x, 100.0);
        assert_eq!(frame.ego.y, 4.0); // 3/2 = 1, 1 * 4.0 = 4.0
        assert!(frame.npcs.is_empty());
    }

    #[test]
    fn test_build_frame_from_state_npc_defaults() {
        let state = serde_json::json!({
            "ego_x": 100.0,
            "ego_y": 8.0,
            "ego_speed": 20.0,
            "npc0_x": 130.0
        });
        let frame = build_frame_from_state(&state, 4, 4.0);
        assert_eq!(frame.npcs.len(), 1);
        assert_eq!(frame.npcs[0].x, 130.0);
        assert_eq!(frame.npcs[0].y, 8.0); // defaults to ego_y
        assert_eq!(frame.npcs[0].speed, 20.0); // defaults to ego_speed
        assert_eq!(frame.npcs[0].heading, 0.0);
    }

    #[test]
    fn test_build_frame_from_state_npc_gap_stops_iteration() {
        // npc0 exists, npc1 missing, npc2 exists → only npc0 is loaded
        let state = serde_json::json!({
            "ego_x": 100.0,
            "ego_y": 8.0,
            "ego_speed": 20.0,
            "npc0_x": 130.0,
            "npc0_y": 8.0,
            "npc2_x": 200.0,
            "npc2_y": 8.0
        });
        let frame = build_frame_from_state(&state, 4, 4.0);
        assert_eq!(frame.npcs.len(), 1);
    }

    #[test]
    fn test_parse_action_probs_json() {
        let json = r#"[
            {"scenario": 0, "left": 0.01, "idle": 0.02, "right": 0.03, "faster": 0.90, "slower": 0.04},
            {"scenario": 1, "left": 0.10, "idle": 0.60, "right": 0.10, "faster": 0.10, "slower": 0.10}
        ]"#;
        let result = parse_action_probs_json(json).unwrap();
        assert_eq!(result.len(), 2);
        // First scenario should have 5 action probs (scenario key filtered out)
        assert_eq!(result[0].probs.len(), 5);
        assert_eq!(result[1].probs.len(), 5);
        // Check a specific value
        let faster = result[0].probs.iter().find(|(k, _)| k == "faster").unwrap();
        assert!((faster.1 - 0.90).abs() < 1e-10);
    }

    #[test]
    fn test_parse_action_probs_json_empty() {
        let result = parse_action_probs_json("[]").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_action_probs_json_invalid() {
        assert!(parse_action_probs_json("not json").is_none());
    }

    #[test]
    fn test_infer_ego_lane_empty() {
        assert_eq!(infer_ego_lane(&[], 4, 4.0), 2);
    }

    #[test]
    fn test_infer_ego_lane_all_below() {
        assert_eq!(infer_ego_lane(&[-12.0, -8.0, -4.0], 4, 4.0), 3);
    }

    #[test]
    fn test_infer_ego_lane_all_above() {
        assert_eq!(infer_ego_lane(&[4.0, 8.0, 12.0], 4, 4.0), 0);
    }

    #[test]
    fn test_infer_ego_lane_symmetric() {
        assert_eq!(infer_ego_lane(&[-4.0, 4.0], 4, 4.0), 1);
    }
}
