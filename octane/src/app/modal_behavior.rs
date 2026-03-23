//! Behavior scenario modal key handling.

use crossterm::event::{KeyCode, KeyEvent};
use tracing::info;

use super::{App, BehaviorScenarioField, ACTION_NAMES};
use crate::data::behaviors;

impl App {
    /// Commit the text in `action_input` to the current action's weight.
    fn commit_action_input(&mut self) {
        let i = self.behavior_modal.action_cursor;
        if let Ok(v) = self.behavior_modal.action_input.parse::<f64>() {
            self.behavior_modal.action_weights[i] = v;
        }
        self.behavior_modal.action_input.clear();
    }

    /// Sync action_input buffer from the current action's stored weight.
    fn sync_action_input(&mut self) {
        let i = self.behavior_modal.action_cursor;
        let w = self.behavior_modal.action_weights[i];
        self.behavior_modal.action_input = if w == 0.0 {
            String::new()
        } else if w == w.round() {
            format!("{}", w as i64)
        } else {
            format!("{}", w)
        };
    }

    /// Handle Behavior Scenario modal keys.
    pub(super) fn handle_behavior_scenario_key(&mut self, key: KeyEvent) {
        match self.behavior_modal.focus {
            BehaviorScenarioField::Actions => match key.code {
                KeyCode::Up => {
                    self.commit_action_input();
                    self.behavior_modal.action_cursor =
                        self.behavior_modal.action_cursor.saturating_sub(1);
                    self.sync_action_input();
                }
                KeyCode::Down => {
                    self.commit_action_input();
                    self.behavior_modal.action_cursor =
                        (self.behavior_modal.action_cursor + 1).min(ACTION_NAMES.len() - 1);
                    self.sync_action_input();
                }
                KeyCode::Char(' ') => {
                    // Quick toggle: 0 → 1, nonzero → 0
                    let i = self.behavior_modal.action_cursor;
                    if self.behavior_modal.action_weights[i] == 0.0 {
                        self.behavior_modal.action_weights[i] = 1.0;
                        self.behavior_modal.action_input = "1".to_string();
                    } else {
                        self.behavior_modal.action_weights[i] = 0.0;
                        self.behavior_modal.action_input.clear();
                    }
                }
                KeyCode::Char(c) if c.is_ascii_digit() || c == '.' || c == '-' => {
                    self.behavior_modal.action_input.push(c);
                    // Live-parse into weight (ignore if invalid mid-typing, e.g. "-")
                    if let Ok(v) = self.behavior_modal.action_input.parse::<f64>() {
                        let i = self.behavior_modal.action_cursor;
                        self.behavior_modal.action_weights[i] = v;
                    }
                }
                KeyCode::Backspace => {
                    self.behavior_modal.action_input.pop();
                    let i = self.behavior_modal.action_cursor;
                    if self.behavior_modal.action_input.is_empty() {
                        self.behavior_modal.action_weights[i] = 0.0;
                    } else if let Ok(v) = self.behavior_modal.action_input.parse::<f64>() {
                        self.behavior_modal.action_weights[i] = v;
                    }
                }
                KeyCode::Tab => {
                    self.commit_action_input();
                    self.behavior_modal.focus = self.behavior_modal.focus.next();
                }
                KeyCode::BackTab => {
                    self.commit_action_input();
                    self.behavior_modal.focus = self.behavior_modal.focus.prev();
                }
                KeyCode::Enter => {
                    self.commit_action_input();
                    self.confirm_behavior_scenario();
                }
                _ => {}
            },
            BehaviorScenarioField::Name => match key.code {
                KeyCode::Char(c) if c.is_alphanumeric() || c == '_' || c == '-' => {
                    self.behavior_modal.name.push(c);
                    self.behavior_modal.suggestion_cursor = None;
                }
                KeyCode::Backspace => {
                    self.behavior_modal.name.pop();
                    self.behavior_modal.suggestion_cursor = None;
                }
                KeyCode::Down => {
                    let n = self.filtered_suggestions().len();
                    if n > 0 {
                        self.behavior_modal.suggestion_cursor = Some(
                            match self.behavior_modal.suggestion_cursor {
                                None => 0,
                                Some(i) => (i + 1).min(n - 1),
                            }
                        );
                    }
                }
                KeyCode::Up => {
                    self.behavior_modal.suggestion_cursor = match self.behavior_modal.suggestion_cursor {
                        None | Some(0) => None,
                        Some(i) => Some(i - 1),
                    };
                }
                KeyCode::Enter => {
                    if let Some(idx) = self.behavior_modal.suggestion_cursor {
                        let filtered = self.filtered_suggestions();
                        if let Some(&name) = filtered.get(idx) {
                            self.behavior_modal.name = name.to_string();
                            self.behavior_modal.suggestion_cursor = None;
                        }
                    } else {
                        self.confirm_behavior_scenario();
                    }
                }
                KeyCode::Tab => {
                    self.behavior_modal.focus = self.behavior_modal.focus.next();
                    self.behavior_modal.suggestion_cursor = None;
                }
                KeyCode::BackTab => {
                    self.behavior_modal.focus = self.behavior_modal.focus.prev();
                    self.behavior_modal.suggestion_cursor = None;
                }
                _ => {}
            },
        }
    }

    /// Load behavior names from disk for autocomplete suggestions.
    pub(super) fn load_behavior_suggestions(&mut self) {
        let env_type = self.trek.env_type.unwrap_or(crate::envs::EnvType::Highway);
        let behaviors_dir = behaviors::behaviors_dir(env_type);
        let mut names = Vec::new();

        if behaviors_dir.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&behaviors_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().is_some_and(|e| e == "json") {
                        if let Some(stem) = path.file_stem() {
                            names.push(stem.to_string_lossy().to_string());
                        }
                    }
                }
            }
        }

        names.sort();
        self.behavior_modal.suggestions = names;
        self.behavior_modal.suggestion_cursor = None;
    }

    /// Get filtered suggestions matching the current name input.
    pub fn filtered_suggestions(&self) -> Vec<&str> {
        let input = &self.behavior_modal.name;
        if input.is_empty() {
            self.behavior_modal.suggestions.iter().map(|s| s.as_str()).collect()
        } else {
            self.behavior_modal.suggestions.iter()
                .filter(|s| s.starts_with(input.as_str()))
                .map(|s| s.as_str())
                .collect()
        }
    }

    /// Confirm and execute the behavior scenario capture.
    /// Builds the scenario state dict directly from frame data and writes JSON,
    /// bypassing the Python subprocess (no snapshot required).
    fn confirm_behavior_scenario(&mut self) {
        // Validate
        if !self.behavior_modal.action_weights.iter().any(|&w| w != 0.0) {
            self.show_toast("Select at least one action".to_string(),
                            std::time::Duration::from_secs(3));
            return;
        }
        if self.behavior_modal.name.is_empty() {
            self.show_toast("Behavior name is required".to_string(),
                            std::time::Duration::from_secs(3));
            return;
        }

        // Get vehicle state from current frame
        let vehicle_state = match self.current_frame()
            .and_then(|f| f.vehicle_state.clone())
        {
            Some(vs) => vs,
            None => {
                self.show_toast("No vehicle state on this frame".to_string(),
                                std::time::Duration::from_secs(3));
                return;
            }
        };

        // Build action_weights map (omit zeros, use int when possible)
        let mut action_weights = serde_json::Map::new();
        for (i, &w) in self.behavior_modal.action_weights.iter().enumerate() {
            if w != 0.0 {
                let val = if w == w.round() {
                    serde_json::json!(w as i64)
                } else {
                    serde_json::json!(w)
                };
                action_weights.insert(
                    ACTION_NAMES[i].to_lowercase(),
                    val,
                );
            }
        }

        let name = self.behavior_modal.name.clone();

        // Build flat state dict from vehicle_state
        let ego = &vehicle_state.ego;
        let mut state_map = serde_json::Map::new();
        let r2 = |v: f64| (v * 100.0).round() / 100.0;
        let r4 = |v: f64| (v * 10000.0).round() / 10000.0;
        state_map.insert("ego_x".into(), serde_json::json!(r2(ego.x)));
        state_map.insert("ego_y".into(), serde_json::json!(r2(ego.y)));
        state_map.insert("ego_speed".into(), serde_json::json!(r2(ego.speed)));
        state_map.insert(
            "ego_heading".into(), serde_json::json!(r4(ego.heading)),
        );

        for (i, npc) in vehicle_state.npcs.iter().enumerate() {
            state_map.insert(
                format!("npc{}_x", i), serde_json::json!(r2(npc.x)),
            );
            state_map.insert(
                format!("npc{}_y", i), serde_json::json!(r2(npc.y)),
            );
            state_map.insert(
                format!("npc{}_speed", i), serde_json::json!(r2(npc.speed)),
            );
            state_map.insert(
                format!("npc{}_heading", i),
                serde_json::json!(r4(npc.heading)),
            );
        }

        let state = serde_json::Value::Object(state_map);

        // Source info — store the target (parquet file or trek dir)
        let target_path = self.parquet_sources.get(self.selected_parquet)
            .map(|s| s.path.display().to_string())
            .unwrap_or_else(|| self.trek.path.display().to_string());
        let epoch = self.trek.epochs.get(self.selected_epoch)
            .map(|e| e.epoch_number)
            .unwrap_or(0);
        let n_sub = self.effective_n_sub;
        let source_t = self.frame_index as f64 / n_sub as f64;

        let scenario = serde_json::json!({
            "source": {
                "target": target_path,
                "epoch": epoch,
                "episode": self.selected_episode,
                "t": source_t,
            },
            "action_weights": action_weights,
            "added": chrono::Local::now().to_rfc3339(),
            "state": state,
        });

        // Load or create behavior file
        let env_type = self.trek.env_type.unwrap_or(crate::envs::EnvType::Highway);
        let dir = behaviors::behaviors_dir(env_type);
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join(format!("{}.json", name));

        let mut data: serde_json::Value = if path.exists() {
            match std::fs::read_to_string(&path).ok()
                .and_then(|text| serde_json::from_str(&text).ok())
            {
                Some(d) => d,
                None => {
                    self.show_toast("Failed to read behavior file".to_string(),
                                    std::time::Duration::from_secs(3));
                    return;
                }
            }
        } else {
            serde_json::json!({
                "name": name,
                "description": "",
                "action_names": ACTION_NAMES.iter()
                    .map(|a| a.to_lowercase())
                    .collect::<Vec<_>>(),
                "scenarios": [],
            })
        };

        // Append scenario
        if let Some(scenarios) = data["scenarios"].as_array_mut() {
            scenarios.push(scenario);
        } else {
            data["scenarios"] = serde_json::json!([scenario]);
        }

        // Ensure directory exists and write
        if let Err(e) = std::fs::create_dir_all(&dir) {
            self.show_toast(format!("Failed to create dir: {}", e),
                            std::time::Duration::from_secs(3));
            return;
        }
        match std::fs::write(&path, serde_json::to_string_pretty(&data).unwrap()) {
            Ok(_) => {
                let n_npcs = vehicle_state.npcs.len();
                let msg = format!("Scenario added to '{}' ({} NPCs)", name, n_npcs);
                info!("{}", msg);
                self.show_toast(msg, std::time::Duration::from_secs(3));
                self.active_modal = None;
                self.trigger_behavior_reload();
            }
            Err(e) => {
                let msg = format!("Failed to write behavior: {}", e);
                info!("{}", msg);
                self.show_toast(msg, std::time::Duration::from_secs(5));
            }
        }
    }
}
