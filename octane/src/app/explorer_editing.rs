//! Explorer scenario editor — field editing, save/create/delete operations.

use tracing::info;

use std::path::Path;

use super::{App, EditorField, EditorFieldKind, ExplorerMode, ACTION_NAMES};
use crate::util::posh_path;
use crate::data::behaviors;
use crate::worlds::SceneEpisode;

fn read_behavior_json(path: &Path) -> Result<serde_json::Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("Read failed: {}", e))?;
    serde_json::from_str(&text).map_err(|e| format!("Parse failed: {}", e))
}

fn write_behavior_json(path: &Path, data: &serde_json::Value) -> Result<(), String> {
    std::fs::write(path, serde_json::to_string_pretty(data).unwrap())
        .map_err(|e| format!("Write failed: {}", e))
}

fn default_scenario_json() -> serde_json::Value {
    serde_json::json!({
        "action_weights": {"idle": 1},
        "state": {
            "ego_x": 100.0,
            "ego_y": 8.0,
            "ego_speed": 20.0,
            "ego_heading": 0.0
        }
    })
}

impl App {
    /// Navigate editor fields up/down.
    pub fn editor_navigate_fields(&mut self, delta: isize) {
        if self.explorer.editor_fields.is_empty() {
            return;
        }
        let len = self.explorer.editor_fields.len();
        let new = (self.explorer.editor_selected_field as isize + delta).clamp(0, len as isize - 1);
        self.explorer.editor_selected_field = new as usize;
    }

    /// Adjust the selected editor field's value.
    /// `direction` is +1.0 (right/increase) or -1.0 (left/decrease).
    /// `fine` uses smaller step sizes (Shift held).
    pub fn editor_adjust_value(&mut self, direction: f64, fine: bool) {
        let Some(field) = self.explorer.editor_fields.get_mut(self.explorer.editor_selected_field) else {
            return;
        };

        let step = match field.kind {
            EditorFieldKind::ActionWeight(_) => if fine { 0.1 } else { 0.5 },
            EditorFieldKind::EgoSpeed => if fine { 0.1 } else { 1.0 },
            EditorFieldKind::NpcRelX(_) => if fine { 1.0 } else { 5.0 },
            EditorFieldKind::NpcRelY(_) => if fine { 0.1 } else { 1.0 },
            EditorFieldKind::NpcSpeed(_) => if fine { 0.1 } else { 1.0 },
        };

        field.value += direction * step;

        // Clamp to reasonable ranges
        field.value = match field.kind {
            EditorFieldKind::ActionWeight(_) => field.value.clamp(-10.0, 10.0),
            EditorFieldKind::EgoSpeed | EditorFieldKind::NpcSpeed(_) => field.value.clamp(0.0, 50.0),
            EditorFieldKind::NpcRelX(_) => field.value.clamp(-200.0, 200.0),
            EditorFieldKind::NpcRelY(_) => field.value.clamp(-20.0, 20.0),
        };

        self.explorer.editor_dirty = true;
        self.explorer.preview_cache = None;
    }

    /// Save editor changes back to the behavior JSON file.
    pub fn editor_save(&mut self) {
        let Some(behavior) = self.explorer.behaviors.get(self.explorer.selected_behavior) else {
            return;
        };
        let path = behavior.path.clone();

        let mut data = match read_behavior_json(&path) {
            Ok(d) => d,
            Err(e) => { self.show_toast(e, std::time::Duration::from_secs(3)); return; }
        };

        let scenario_idx = self.explorer.selected_scenario;
        let Some(scenario) = data["scenarios"].as_array_mut()
            .and_then(|arr| arr.get_mut(scenario_idx)) else {
            self.show_toast("Scenario not found".to_string(), std::time::Duration::from_secs(3));
            return;
        };

        // Rebuild action_weights from editor fields (omit zeros, use int when possible)
        let action_names = ["left", "idle", "right", "faster", "slower"];
        let mut aw_map = serde_json::Map::new();
        for field in &self.explorer.editor_fields {
            if let EditorFieldKind::ActionWeight(i) = field.kind {
                if field.value != 0.0 {
                    let val = if field.value == field.value.round() {
                        serde_json::json!(field.value as i64)
                    } else {
                        serde_json::json!(field.value)
                    };
                    if let Some(name) = action_names.get(i) {
                        aw_map.insert(name.to_string(), val);
                    }
                }
            }
        }
        scenario["action_weights"] = serde_json::Value::Object(aw_map);
        scenario["edited"] = serde_json::json!(true);
        // Remove old format keys if present
        if let Some(obj) = scenario.as_object_mut() {
            obj.remove("actions");
            obj.remove("weight");
        }

        // Rebuild state dict from editor fields
        let n_lanes = self.config.octane.road.n_lanes;
        let lane_width = self.config.octane.road.lane_width;
        let state = build_state_from_editor_fields(
            &self.explorer.editor_fields, n_lanes, lane_width,
        );
        scenario["state"] = state;

        match write_behavior_json(&path, &data) {
            Ok(()) => {
                self.explorer.editor_dirty = false;
                self.explorer.mode = ExplorerMode::Browse;
                self.explorer.editor_fields.clear();
                self.show_toast("Saved".to_string(), std::time::Duration::from_secs(2));
                self.load_explorer_scenarios();
                info!("Saved scenario {} to {}", scenario_idx, posh_path(&path));
            }
            Err(e) => {
                self.show_toast(e, std::time::Duration::from_secs(3));
            }
        }
    }

    /// Add a new NPC at default position in edit mode.
    pub fn editor_add_npc(&mut self) {
        let next_index = self.explorer.editor_fields.iter().filter_map(|f| match f.kind {
            EditorFieldKind::NpcRelX(i) | EditorFieldKind::NpcRelY(i) | EditorFieldKind::NpcSpeed(i) => Some(i),
            _ => None,
        }).max().map(|i| i + 1).unwrap_or(0);

        for (kind, value) in [
            (EditorFieldKind::NpcRelX(next_index), 30.0),
            (EditorFieldKind::NpcRelY(next_index), 0.0),
            (EditorFieldKind::NpcSpeed(next_index), 20.0),
        ] {
            self.explorer.editor_fields.push(EditorField {
                label: kind.label(), value, kind,
            });
        }

        // Move cursor to the new NPC's first field
        self.explorer.editor_selected_field = self.explorer.editor_fields.len() - 3;
        self.explorer.editor_dirty = true;
        self.explorer.preview_cache = None;
        info!("Added NPC {} in editor", next_index);
    }

    /// Get the NPC index that the currently selected editor field belongs to.
    pub fn npc_index_of_selected_field(&self) -> Option<usize> {
        self.explorer.editor_fields.get(self.explorer.editor_selected_field).and_then(|f| match f.kind {
            EditorFieldKind::NpcRelX(i) | EditorFieldKind::NpcRelY(i) | EditorFieldKind::NpcSpeed(i) => Some(i),
            _ => None,
        })
    }

    /// Remove the NPC whose field is currently selected, renumbering remaining NPCs.
    pub fn editor_remove_npc(&mut self) {
        let Some(remove_idx) = self.npc_index_of_selected_field() else {
            self.show_toast(
                "Select an NPC field to remove".to_string(),
                std::time::Duration::from_secs(2),
            );
            return;
        };

        // Remove all fields for this NPC
        self.explorer.editor_fields.retain(|f| match f.kind {
            EditorFieldKind::NpcRelX(i) | EditorFieldKind::NpcRelY(i) | EditorFieldKind::NpcSpeed(i) => i != remove_idx,
            _ => true,
        });

        // Renumber NPCs above the removed one
        for field in &mut self.explorer.editor_fields {
            field.kind = match field.kind {
                EditorFieldKind::NpcRelX(i) if i > remove_idx => EditorFieldKind::NpcRelX(i - 1),
                EditorFieldKind::NpcRelY(i) if i > remove_idx => EditorFieldKind::NpcRelY(i - 1),
                EditorFieldKind::NpcSpeed(i) if i > remove_idx => EditorFieldKind::NpcSpeed(i - 1),
                other => other,
            };
            field.label = field.kind.label();
        }

        // Clamp cursor
        if self.explorer.editor_selected_field >= self.explorer.editor_fields.len() {
            self.explorer.editor_selected_field = self.explorer.editor_fields.len().saturating_sub(1);
        }

        self.explorer.editor_dirty = true;
        self.explorer.preview_cache = None;
        self.explorer.npc_remove_pending = false;
        info!("Removed NPC {} from editor", remove_idx);
    }

    /// Create a new empty scenario in the current behavior.
    pub fn explorer_new_scenario(&mut self) {
        let Some(behavior) = self.explorer.behaviors.get(self.explorer.selected_behavior) else {
            return;
        };
        let path = behavior.path.clone();

        let mut data = match read_behavior_json(&path) {
            Ok(d) => d,
            Err(e) => { self.show_toast(e, std::time::Duration::from_secs(3)); return; }
        };

        if let Some(arr) = data["scenarios"].as_array_mut() {
            arr.push(default_scenario_json());
        }

        match write_behavior_json(&path, &data) {
            Ok(()) => {
                let new_idx = data["scenarios"].as_array().map(|a| a.len() - 1).unwrap_or(0);
                self.load_explorer_scenarios();
                self.explorer.selected_scenario = new_idx;
                self.explorer.preview_cache = None;
                // Also update behavior n_scenarios count
                if let Some(b) = self.explorer.behaviors.get_mut(self.explorer.selected_behavior) {
                    b.n_scenarios += 1;
                }
                self.show_toast("New scenario created".to_string(), std::time::Duration::from_secs(2));
                info!("Created new scenario {} in {}", new_idx, posh_path(&path));
            }
            Err(e) => {
                self.show_toast(e, std::time::Duration::from_secs(3));
            }
        }
    }

    /// Duplicate the selected scenario in the current behavior.
    pub fn explorer_duplicate_scenario(&mut self) {
        let Some(behavior) = self.explorer.behaviors.get(self.explorer.selected_behavior) else {
            return;
        };
        let path = behavior.path.clone();

        let mut data = match read_behavior_json(&path) {
            Ok(d) => d,
            Err(e) => { self.show_toast(e, std::time::Duration::from_secs(3)); return; }
        };

        let idx = self.explorer.selected_scenario;
        let Some(arr) = data["scenarios"].as_array_mut() else { return; };
        if idx >= arr.len() { return; }
        let clone = arr[idx].clone();
        arr.insert(idx + 1, clone);

        match write_behavior_json(&path, &data) {
            Ok(()) => {
                if let Some(b) = self.explorer.behaviors.get_mut(self.explorer.selected_behavior) {
                    b.n_scenarios += 1;
                }
                self.load_explorer_scenarios();
                self.explorer.selected_scenario = idx + 1;
                self.explorer.preview_cache = None;
                self.show_toast("Scenario duplicated".to_string(), std::time::Duration::from_secs(2));
                info!("Duplicated scenario {} in {}", idx, posh_path(&path));
            }
            Err(e) => {
                self.show_toast(e, std::time::Duration::from_secs(3));
            }
        }
    }

    /// Delete the selected scenario from the current behavior.
    pub fn explorer_delete_scenario(&mut self) {
        let Some(behavior) = self.explorer.behaviors.get(self.explorer.selected_behavior) else {
            return;
        };
        if self.explorer.scenarios.len() <= 1 {
            self.show_toast("Cannot delete last scenario".to_string(), std::time::Duration::from_secs(2));
            return;
        }
        let path = behavior.path.clone();

        let mut data = match read_behavior_json(&path) {
            Ok(d) => d,
            Err(e) => { self.show_toast(e, std::time::Duration::from_secs(3)); return; }
        };

        let idx = self.explorer.selected_scenario;
        if let Some(arr) = data["scenarios"].as_array_mut() {
            if idx < arr.len() {
                arr.remove(idx);
            }
        }

        match write_behavior_json(&path, &data) {
            Ok(()) => {
                if let Some(b) = self.explorer.behaviors.get_mut(self.explorer.selected_behavior) {
                    b.n_scenarios = b.n_scenarios.saturating_sub(1);
                }
                self.load_explorer_scenarios();
                if self.explorer.selected_scenario >= self.explorer.scenarios.len() {
                    self.explorer.selected_scenario = self.explorer.scenarios.len().saturating_sub(1);
                }
                self.explorer.preview_cache = None;
                self.explorer.delete_pending = false;
                self.show_toast("Scenario deleted".to_string(), std::time::Duration::from_secs(2));
                info!("Deleted scenario {} from {}", idx, posh_path(&path));
            }
            Err(e) => {
                self.show_toast(e, std::time::Duration::from_secs(3));
            }
        }
    }

    /// Delete the currently selected behavior (file on disk).
    pub fn explorer_delete_behavior(&mut self) {
        let Some(behavior) = self.explorer.behaviors.get(self.explorer.selected_behavior) else {
            return;
        };
        if behavior.is_preset {
            self.show_toast("Cannot delete preset behaviors".to_string(), std::time::Duration::from_secs(2));
            self.explorer.delete_pending = false;
            return;
        }
        let name = behavior.name.clone();
        let path = behavior.path.clone();

        match std::fs::remove_file(&path) {
            Ok(()) => {
                self.explorer.delete_pending = false;
                self.load_explorer_behaviors();
                self.show_toast(format!("Deleted behavior '{}'", name), std::time::Duration::from_secs(2));
                info!("Deleted behavior '{}' at {}", name, posh_path(&path));
            }
            Err(e) => {
                self.show_toast(format!("Delete failed: {}", e), std::time::Duration::from_secs(3));
            }
        }
    }

    /// Enter new behavior text input mode.
    pub fn enter_new_behavior_mode(&mut self) {
        self.explorer.new_behavior_name.clear();
        self.explorer.mode = ExplorerMode::NewBehavior;
        info!("Entered NewBehavior text input mode");
    }

    /// Create the new behavior file from the entered name.
    pub fn confirm_new_behavior(&mut self) {
        let name = self.explorer.new_behavior_name.trim().to_string();
        if name.is_empty() {
            self.show_toast("Name cannot be empty".to_string(), std::time::Duration::from_secs(2));
            return;
        }

        let env_type = self.trek.env_type.unwrap_or(crate::envs::EnvType::Highway);
        let dir = behaviors::behaviors_dir(env_type);
        std::fs::create_dir_all(&dir).ok();

        let filename = format!("{}.json", name);
        let path = dir.join(&filename);

        if path.exists() {
            self.show_toast(format!("Behavior '{}' already exists", name), std::time::Duration::from_secs(3));
            return;
        }

        let data = serde_json::json!({
            "name": name,
            "description": "",
            "action_names": ACTION_NAMES.iter()
                .map(|a| a.to_lowercase())
                .collect::<Vec<_>>(),
            "scenarios": [default_scenario_json()]
        });

        match write_behavior_json(&path, &data) {
            Ok(()) => {
                self.explorer.mode = ExplorerMode::Browse;
                self.load_explorer_behaviors();
                // Select the new behavior
                if let Some(idx) = self.explorer.behaviors.iter().position(|b| b.name == name) {
                    self.explorer.selected_behavior = idx;
                    self.load_explorer_scenarios();
                }
                self.show_toast(format!("Created behavior '{}'", name), std::time::Duration::from_secs(2));
                info!("Created new behavior '{}' at {}", name, posh_path(&path));
            }
            Err(e) => {
                self.show_toast(e, std::time::Duration::from_secs(3));
            }
        }
    }

    /// Render preview from editor fields (instead of from disk).
    /// Cache is invalidated by setting `explorer_preview_cache = None` (done in `editor_adjust_value`).
    pub fn render_editor_preview(&mut self, cols: u32, rows: u32) -> Option<String> {
        if self.explorer.preview_cache.is_some() && self.explorer.preview_dims == (cols, rows) {
            return self.explorer.preview_cache.clone();
        }

        let frame = build_frame_from_editor_fields(&self.explorer.editor_fields, &self.config);

        let scene = SceneEpisode::from_frames(vec![frame]);
        let ansi = self.render_static_scene(scene, cols, rows)?;

        self.explorer.preview_cache = Some(ansi.clone());
        self.explorer.preview_dims = (cols, rows);
        Some(ansi)
    }
}

/// Build editor fields from a scenario JSON (including action_weights and state).
///
/// The state dict has absolute positions (`ego_x`, `npc0_x`, etc.).
/// The editor works in relative NPC positions, so we compute
/// `rel_x = npc_x - ego_x` and `rel_y = npc_y - ego_y`.
pub fn build_editor_fields(scenario: &serde_json::Value) -> Vec<EditorField> {
    let mut fields = Vec::new();
    let action_names = ["left", "idle", "right", "faster", "slower"];

    // Action weight fields
    let aw = scenario.get("action_weights").and_then(|v| v.as_object());
    // Backward compat: old format with actions list + weight
    let old_actions = scenario.get("actions").and_then(|v| v.as_array());
    let old_weight = scenario.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0);

    for (i, name) in action_names.iter().enumerate() {
        let w = if let Some(aw_map) = aw {
            aw_map.get(*name)
                .and_then(|v| v.as_f64().or_else(|| v.as_i64().map(|n| n as f64)))
                .unwrap_or(0.0)
        } else if let Some(actions) = old_actions {
            if actions.iter().any(|a| a.as_str() == Some(name)) {
                old_weight
            } else {
                0.0
            }
        } else {
            0.0
        };
        let kind = EditorFieldKind::ActionWeight(i);
        fields.push(EditorField { label: kind.label(), value: w, kind });
    }

    let state = scenario.get("state").unwrap_or(scenario);

    let ego_x = state.get("ego_x").and_then(|v| v.as_f64()).unwrap_or(100.0);
    let ego_y = state.get("ego_y").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let ego_speed = state.get("ego_speed").and_then(|v| v.as_f64())
        .unwrap_or(20.0);

    fields.push(EditorField {
        label: EditorFieldKind::EgoSpeed.label(),
        value: ego_speed,
        kind: EditorFieldKind::EgoSpeed,
    });

    let mut i = 0;
    loop {
        let key = format!("npc{}_x", i);
        let Some(npc_x) = state.get(&key).and_then(|v| v.as_f64()) else {
            break;
        };
        let npc_y = state.get(&format!("npc{}_y", i))
            .and_then(|v| v.as_f64()).unwrap_or(ego_y);
        let npc_speed = state.get(&format!("npc{}_speed", i))
            .and_then(|v| v.as_f64()).unwrap_or(ego_speed);

        for (kind, value) in [
            (EditorFieldKind::NpcRelX(i), npc_x - ego_x),
            (EditorFieldKind::NpcRelY(i), npc_y - ego_y),
            (EditorFieldKind::NpcSpeed(i), npc_speed),
        ] {
            fields.push(EditorField { label: kind.label(), value, kind });
        }
        i += 1;
    }

    fields
}

struct ResolvedNpc {
    rel_x: f64,
    rel_y: f64,
    speed: f64,
}

struct ResolvedFields {
    ego_x: f64,
    ego_y: f64,
    ego_speed: f64,
    npcs: Vec<ResolvedNpc>,
}

fn resolve_fields(fields: &[EditorField], n_lanes: usize, lane_width: f64) -> ResolvedFields {
    use crate::data::behaviors::infer_ego_lane;

    let ego_x = 100.0;
    let ego_speed = fields.iter()
        .find(|f| f.kind == EditorFieldKind::EgoSpeed)
        .map(|f| f.value)
        .unwrap_or(20.0);

    let max_npc = fields.iter().filter_map(|f| match f.kind {
        EditorFieldKind::NpcRelX(i) | EditorFieldKind::NpcRelY(i)
        | EditorFieldKind::NpcSpeed(i) => Some(i),
        _ => None,
    }).max();

    let mut npcs = Vec::new();
    if let Some(max_i) = max_npc {
        for i in 0..=max_i {
            let rel_x = fields.iter()
                .find(|f| f.kind == EditorFieldKind::NpcRelX(i))
                .map(|f| f.value).unwrap_or(0.0);
            let rel_y = fields.iter()
                .find(|f| f.kind == EditorFieldKind::NpcRelY(i))
                .map(|f| f.value).unwrap_or(0.0);
            let speed = fields.iter()
                .find(|f| f.kind == EditorFieldKind::NpcSpeed(i))
                .map(|f| f.value).unwrap_or(ego_speed);
            npcs.push(ResolvedNpc { rel_x, rel_y, speed });
        }
    }

    let npc_rel_ys: Vec<f64> = npcs.iter().map(|n| n.rel_y).collect();
    let ego_lane = infer_ego_lane(&npc_rel_ys, n_lanes, lane_width);
    let ego_y = ego_lane as f64 * lane_width;

    ResolvedFields { ego_x, ego_y, ego_speed, npcs }
}

/// Build a flat state dict from editor fields (for saving back to disk).
fn build_state_from_editor_fields(
    fields: &[EditorField],
    n_lanes: usize,
    lane_width: f64,
) -> serde_json::Value {
    let r = resolve_fields(fields, n_lanes, lane_width);

    let mut map = serde_json::Map::new();
    map.insert("ego_x".into(), serde_json::json!(r.ego_x));
    map.insert("ego_y".into(), serde_json::json!(r.ego_y));
    map.insert("ego_speed".into(), serde_json::json!(r.ego_speed));
    map.insert("ego_heading".into(), serde_json::json!(0.0));

    for (i, npc) in r.npcs.iter().enumerate() {
        map.insert(format!("npc{}_x", i), serde_json::json!(r.ego_x + npc.rel_x));
        map.insert(format!("npc{}_y", i), serde_json::json!(r.ego_y + npc.rel_y));
        map.insert(format!("npc{}_speed", i), serde_json::json!(npc.speed));
        map.insert(format!("npc{}_heading", i), serde_json::json!(0.0));
    }

    serde_json::Value::Object(map)
}

/// Build a FrameState from current editor field values.
fn build_frame_from_editor_fields(
    fields: &[EditorField],
    config: &crate::config::Config,
) -> crate::data::FrameState {
    use crate::data::{FrameState, VehicleState};

    let r = resolve_fields(fields, config.octane.road.n_lanes, config.octane.road.lane_width);

    let ego = VehicleState {
        x: r.ego_x, y: r.ego_y, heading: 0.0, speed: r.ego_speed,
        acceleration: 0.0, attention: None,
    };

    let npcs = r.npcs.iter().map(|npc| VehicleState {
        x: r.ego_x + npc.rel_x, y: r.ego_y + npc.rel_y,
        heading: 0.0, speed: npc.speed,
        acceleration: 0.0, attention: None,
    }).collect();

    FrameState { crashed: false, ego, npcs, action_distribution: None, chosen_action: None, old_action_distribution: None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_editor_fields_ego_only() {
        let scenario = serde_json::json!({
            "action_weights": {"faster": 1},
            "state": {
                "ego_x": 100.0, "ego_y": 8.0, "ego_speed": 25.0, "ego_heading": 0.0
            }
        });
        let fields = build_editor_fields(&scenario);
        // 5 action weights + 1 ego speed = 6
        assert_eq!(fields.len(), 6);
        assert_eq!(fields[0].kind, EditorFieldKind::ActionWeight(0));
        assert_eq!(fields[3].kind, EditorFieldKind::ActionWeight(3)); // faster
        assert_eq!(fields[3].value, 1.0);
        assert_eq!(fields[5].kind, EditorFieldKind::EgoSpeed);
        assert_eq!(fields[5].value, 25.0);
    }

    #[test]
    fn test_build_editor_fields_with_npcs() {
        let scenario = serde_json::json!({
            "action_weights": {"left": 1, "idle": -0.5},
            "state": {
                "ego_x": 100.0, "ego_y": 8.0, "ego_speed": 20.0,
                "npc0_x": 130.0, "npc0_y": 12.0, "npc0_speed": 18.0,
                "npc1_x": 90.0, "npc1_y": 4.0, "npc1_speed": 22.0
            }
        });
        let fields = build_editor_fields(&scenario);
        // 5 action weights + 1 ego speed + 2 NPCs * 3 fields = 12
        assert_eq!(fields.len(), 12);
        assert_eq!(fields[0].kind, EditorFieldKind::ActionWeight(0)); // left
        assert_eq!(fields[0].value, 1.0);
        assert_eq!(fields[1].kind, EditorFieldKind::ActionWeight(1)); // idle
        assert_eq!(fields[1].value, -0.5);
        assert_eq!(fields[5].kind, EditorFieldKind::EgoSpeed);
        assert_eq!(fields[6].kind, EditorFieldKind::NpcRelX(0));
        assert_eq!(fields[6].value, 30.0); // 130 - 100
        assert_eq!(fields[7].kind, EditorFieldKind::NpcRelY(0));
        assert_eq!(fields[7].value, 4.0); // 12 - 8
        assert_eq!(fields[8].kind, EditorFieldKind::NpcSpeed(0));
        assert_eq!(fields[8].value, 18.0);
        assert_eq!(fields[9].kind, EditorFieldKind::NpcRelX(1));
        assert_eq!(fields[9].value, -10.0); // 90 - 100
    }

    #[test]
    fn test_editor_step_sizes() {
        let scenario = serde_json::json!({
            "action_weights": {"idle": 1},
            "state": {
                "ego_x": 100.0, "ego_y": 8.0, "ego_speed": 20.0,
                "npc0_x": 150.0, "npc0_y": 12.0, "npc0_speed": 18.0
            }
        });
        let mut fields = build_editor_fields(&scenario);

        // ActionWeight: coarse step = 0.5
        fields[1].value += 0.5; // idle, was 1.0
        assert_eq!(fields[1].value, 1.5);

        // EgoSpeed: coarse step = 1.0
        fields[5].value += 1.0;
        assert_eq!(fields[5].value, 21.0);

        // NpcRelX: coarse step = 5.0
        fields[6].value += 5.0;
        assert_eq!(fields[6].value, 55.0);

        // NpcRelY: coarse step = 1.0
        fields[7].value += 1.0;
        assert_eq!(fields[7].value, 5.0);

        // Fine: EgoSpeed step = 0.1
        fields[5].value += 0.1;
        assert!((fields[5].value - 21.1).abs() < 1e-9);
    }

    #[test]
    fn test_editor_value_clamping() {
        let mut field = EditorField {
            label: "Ego speed".to_string(),
            value: 49.0,
            kind: EditorFieldKind::EgoSpeed,
        };
        field.value = (field.value + 5.0).clamp(0.0, 50.0);
        assert_eq!(field.value, 50.0);

        field.value = (field.value - 100.0).clamp(0.0, 50.0);
        assert_eq!(field.value, 0.0);

        let mut rel_x = EditorField {
            label: "NPC 0 rel_x".to_string(),
            value: 195.0,
            kind: EditorFieldKind::NpcRelX(0),
        };
        rel_x.value = (rel_x.value + 10.0).clamp(-200.0, 200.0);
        assert_eq!(rel_x.value, 200.0);
    }

    #[test]
    fn test_state_roundtrip() {
        let scenario = serde_json::json!({
            "action_weights": {"left": 1, "faster": -0.5},
            "state": {
                "ego_x": 100.0, "ego_y": 8.0, "ego_speed": 25.0, "ego_heading": 0.0,
                "npc0_x": 130.0, "npc0_y": 12.0, "npc0_speed": 18.0, "npc0_heading": 0.0,
                "npc1_x": 90.0, "npc1_y": 4.0, "npc1_speed": 22.0, "npc1_heading": 0.0
            }
        });
        let fields = build_editor_fields(&scenario);
        let rebuilt = build_state_from_editor_fields(&fields, 4, 4.0);

        assert_eq!(rebuilt["ego_speed"].as_f64().unwrap(), 25.0);
        // NPC absolute positions are reconstructed from rel + ego
        let ego_x = rebuilt["ego_x"].as_f64().unwrap();
        let ego_y = rebuilt["ego_y"].as_f64().unwrap();
        assert_eq!(rebuilt["npc0_x"].as_f64().unwrap() - ego_x, 30.0);
        assert_eq!(rebuilt["npc0_y"].as_f64().unwrap() - ego_y, 4.0);
        assert_eq!(rebuilt["npc0_speed"].as_f64().unwrap(), 18.0);
        assert_eq!(rebuilt["npc1_x"].as_f64().unwrap() - ego_x, -10.0);
        assert_eq!(rebuilt["npc1_y"].as_f64().unwrap() - ego_y, -4.0);
        assert_eq!(rebuilt["npc1_speed"].as_f64().unwrap(), 22.0);
    }

    #[test]
    fn test_npc_add_produces_default_fields() {
        let scenario = serde_json::json!({
            "action_weights": {"idle": 1},
            "state": {
                "ego_x": 100.0, "ego_y": 8.0, "ego_speed": 20.0, "ego_heading": 0.0
            }
        });
        let mut fields = build_editor_fields(&scenario);
        assert_eq!(fields.len(), 6); // 5 action weights + ego speed

        // Simulate editor_add_npc logic
        let next_index = 0;
        fields.push(EditorField {
            label: format!("NPC {} rel_x", next_index),
            value: 30.0,
            kind: EditorFieldKind::NpcRelX(next_index),
        });
        fields.push(EditorField {
            label: format!("NPC {} rel_y", next_index),
            value: 0.0,
            kind: EditorFieldKind::NpcRelY(next_index),
        });
        fields.push(EditorField {
            label: format!("NPC {} speed", next_index),
            value: 20.0,
            kind: EditorFieldKind::NpcSpeed(next_index),
        });

        assert_eq!(fields.len(), 9);
        let rebuilt = build_state_from_editor_fields(&fields, 4, 4.0);
        let ego_x = rebuilt["ego_x"].as_f64().unwrap();
        let ego_y = rebuilt["ego_y"].as_f64().unwrap();
        assert_eq!(rebuilt["npc0_x"].as_f64().unwrap() - ego_x, 30.0);
        assert_eq!(rebuilt["npc0_y"].as_f64().unwrap() - ego_y, 0.0);
        assert_eq!(rebuilt["npc0_speed"].as_f64().unwrap(), 20.0);
    }

    #[test]
    fn test_npc_remove_updates_field_indices() {
        let scenario = serde_json::json!({
            "action_weights": {"idle": 1},
            "state": {
                "ego_x": 100.0, "ego_y": 8.0, "ego_speed": 20.0,
                "npc0_x": 110.0, "npc0_y": 8.0, "npc0_speed": 18.0,
                "npc1_x": 130.0, "npc1_y": 12.0, "npc1_speed": 22.0,
                "npc2_x": 150.0, "npc2_y": 4.0, "npc2_speed": 25.0
            }
        });
        let mut fields = build_editor_fields(&scenario);
        assert_eq!(fields.len(), 15); // 5 action weights + 1 ego + 3 NPCs * 3 fields

        // Remove NPC 1 (middle NPC)
        let remove_idx = 1;
        fields.retain(|f| match f.kind {
            EditorFieldKind::NpcRelX(i) | EditorFieldKind::NpcRelY(i)
            | EditorFieldKind::NpcSpeed(i) => i != remove_idx,
            _ => true,
        });

        for field in &mut fields {
            field.kind = match field.kind {
                EditorFieldKind::NpcRelX(i) if i > remove_idx => {
                    EditorFieldKind::NpcRelX(i - 1)
                }
                EditorFieldKind::NpcRelY(i) if i > remove_idx => {
                    EditorFieldKind::NpcRelY(i - 1)
                }
                EditorFieldKind::NpcSpeed(i) if i > remove_idx => {
                    EditorFieldKind::NpcSpeed(i - 1)
                }
                other => other,
            };
        }

        assert_eq!(fields.len(), 12); // 5 action weights + 1 ego + 2 NPCs * 3 fields

        let rebuilt = build_state_from_editor_fields(&fields, 4, 4.0);
        let ego_x = rebuilt["ego_x"].as_f64().unwrap();
        // NPC 0 (was 0): rel_x = 10
        assert_eq!(rebuilt["npc0_x"].as_f64().unwrap() - ego_x, 10.0);
        // NPC 1 (was 2): rel_x = 50
        assert_eq!(rebuilt["npc1_x"].as_f64().unwrap() - ego_x, 50.0);
    }

    #[test]
    fn test_new_scenario_json_structure() {
        let new_scenario = serde_json::json!({
            "action_weights": {"idle": 1},
            "state": {
                "ego_x": 100.0,
                "ego_y": 8.0,
                "ego_speed": 20.0,
                "ego_heading": 0.0
            }
        });

        assert_eq!(
            new_scenario["action_weights"]["idle"].as_i64().unwrap(), 1,
        );
        assert_eq!(
            new_scenario["state"]["ego_speed"].as_f64().unwrap(), 20.0,
        );

        let fields = build_editor_fields(&new_scenario);
        // 5 action weights + 1 ego speed = 6
        assert_eq!(fields.len(), 6);
        assert_eq!(fields[1].kind, EditorFieldKind::ActionWeight(1)); // idle
        assert_eq!(fields[1].value, 1.0);
        assert_eq!(fields[5].kind, EditorFieldKind::EgoSpeed);
        assert_eq!(fields[5].value, 20.0);
    }
}
