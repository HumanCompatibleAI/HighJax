//! Behavior explorer logic — loading, navigation, and preview rendering.

use tracing::info;

use std::sync::mpsc;

use super::{App, BehaviorEntry, BehaviorLoadResult, ExplorerMode, Focus, ScenarioEntry, Screen};
use super::navigation::DEFAULT_PAGE_SIZE;
use crate::data::behaviors;
use crate::worlds::SceneEpisode;

impl App {
    /// Spawn a background thread to load behaviors.
    /// Safe to call multiple times — only spawns if no load is already pending.
    pub fn spawn_behavior_load(&mut self) {
        if self.explorer.load_rx.is_some() {
            return; // Already loading
        }
        let env_type = self.trek.env_type.unwrap_or(crate::envs::EnvType::Highway);
        let (tx, rx) = mpsc::channel();
        self.explorer.load_rx = Some(rx);
        std::thread::spawn(move || {
            let result = behaviors::list_all_behaviors(env_type);
            let _ = tx.send(result);
        });
        info!("Spawned background behavior load");
    }

    /// Check if background behavior load has completed.
    /// Called each iteration of the main loop.
    pub fn check_behavior_load(&mut self) {
        let Some(rx) = &self.explorer.load_rx else {
            return;
        };
        match rx.try_recv() {
            Ok(raw) => {
                self.explorer.load_rx = None;
                self.apply_behavior_load(raw);
                self.explorer.loaded = true;
                info!("Background behavior load complete");
            }
            Err(mpsc::TryRecvError::Empty) => {} // Still loading
            Err(mpsc::TryRecvError::Disconnected) => {
                // Thread died without sending — clear receiver
                self.explorer.load_rx = None;
                info!("Background behavior load thread disconnected");
            }
        }
    }

    /// Apply loaded behavior data, preserving selection where possible.
    fn apply_behavior_load(&mut self, raw: BehaviorLoadResult) {
        // Remember current selection name for re-selection after reload
        let prev_name = self.explorer.behaviors
            .get(self.explorer.selected_behavior)
            .map(|b| b.name.clone());

        self.explorer.behaviors = raw
            .into_iter()
            .map(|(name, path, n_scenarios, _description, is_preset)| BehaviorEntry {
                name,
                path,
                n_scenarios,
                is_preset,
            })
            .collect();

        // Try to restore previous selection by name
        if let Some(ref prev) = prev_name {
            if let Some(idx) = self.explorer.behaviors.iter().position(|b| &b.name == prev) {
                self.explorer.selected_behavior = idx;
            }
        }

        // Clamp selection
        if self.explorer.selected_behavior >= self.explorer.behaviors.len() {
            self.explorer.selected_behavior = self.explorer.behaviors.len().saturating_sub(1);
        }

        info!(
            "Applied {} explorer behaviors",
            self.explorer.behaviors.len()
        );

        // Also load scenarios for current selection
        self.load_explorer_scenarios();
    }

    /// Trigger a behavior reload (e.g. after capturing a scenario).
    /// Starts a fresh background load.
    pub fn trigger_behavior_reload(&mut self) {
        self.explorer.load_rx = None; // Drop any pending receiver
        self.spawn_behavior_load();
    }

    /// Synchronous behavior load (used by operations that need immediate results,
    /// e.g. after deleting a behavior or creating a new one).
    pub fn load_explorer_behaviors(&mut self) {
        let env_type = self.trek.env_type.unwrap_or(crate::envs::EnvType::Highway);
        let raw = behaviors::list_all_behaviors(env_type);
        self.apply_behavior_load(raw);
        self.explorer.loaded = true;
    }

    pub fn load_explorer_scenarios(&mut self) {
        self.explorer.scenarios.clear();
        self.explorer.selected_scenario = 0;
        self.explorer.scenarios_scroll = 0;
        self.explorer.preview_cache = None;
        // Drop any in-flight action probs fetch so a new one can start
        self.explorer.action_probs_rx = None;

        let Some(behavior) = self.explorer.behaviors.get(self.explorer.selected_behavior) else {
            return;
        };

        let Some(scenarios) = behaviors::load_behavior_scenarios(&behavior.path) else {
            return;
        };

        for (i, scenario) in scenarios.iter().enumerate() {
            let has_state = scenario.get("state").is_some();

            let (ego_speed, ego_heading, n_npcs) = if has_state {
                let state = &scenario["state"];
                let speed = state.get("ego_speed")
                    .and_then(|v| v.as_f64()).unwrap_or(0.0);
                let heading = state.get("ego_heading")
                    .and_then(|v| v.as_f64()).unwrap_or(0.0);
                let mut n = 0;
                while state.get(&format!("npc{}_x", n)).is_some() {
                    n += 1;
                }
                (speed, heading, n)
            } else {
                (0.0, 0.0, 0)
            };

            let action_names = ["left", "idle", "right", "faster", "slower"];
            let mut action_weights = [0.0f64; 5];
            if let Some(aw) = scenario.get("action_weights").and_then(|v| v.as_object()) {
                for (idx, name) in action_names.iter().enumerate() {
                    if let Some(v) = aw.get(*name) {
                        action_weights[idx] = v.as_f64()
                            .unwrap_or(v.as_i64().map(|n| n as f64).unwrap_or(0.0));
                    }
                }
            } else {
                // Backward compat: old format with actions list + weight
                let weight = scenario["weight"].as_f64().unwrap_or(1.0);
                if let Some(actions) = scenario["actions"].as_array() {
                    for a in actions.iter().filter_map(|v| v.as_str()) {
                        if let Some(idx) = action_names.iter().position(|&n| n == a) {
                            action_weights[idx] = weight;
                        }
                    }
                }
            };

            let source = scenario.get("source");
            let source_target = source
                .and_then(|s| s.get("target").or_else(|| s.get("trek")))
                .and_then(|v| v.as_str())
                .map(|s| std::path::PathBuf::from(s));
            let source_epoch = source
                .and_then(|s| s.get("epoch"))
                .and_then(|v| v.as_i64());
            let source_episode = source
                .and_then(|s| s.get("episode"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);
            let source_t = source
                .and_then(|s| s.get("t"))
                .and_then(|v| v.as_f64());

            let edited = scenario.get("edited")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            self.explorer.scenarios.push(ScenarioEntry {
                index: i,
                action_weights,
                has_state,
                ego_speed,
                ego_heading,
                n_npcs,
                source_target,
                source_epoch,
                source_episode,
                source_t,
                edited,
            });
        }
    }

    pub fn explorer_navigate_behaviors(&mut self, delta: isize) {
        let len = self.explorer.behaviors.len();
        let visible = self.explorer.behaviors_visible_rows.unwrap_or(DEFAULT_PAGE_SIZE);
        let (new_sel, new_scroll) = super::navigation::navigate_list(
            self.explorer.selected_behavior, self.explorer.behaviors_scroll,
            len, visible, delta,
        );
        if new_sel != self.explorer.selected_behavior {
            self.explorer.selected_behavior = new_sel;
            self.explorer.behaviors_scroll = new_scroll;
            self.load_explorer_scenarios();
        }
    }

    pub fn explorer_navigate_scenarios(&mut self, delta: isize) {
        let len = self.explorer.scenarios.len();
        let visible = self.explorer.scenarios_visible_rows.unwrap_or(DEFAULT_PAGE_SIZE);
        let (new_sel, new_scroll) = super::navigation::navigate_list(
            self.explorer.selected_scenario, self.explorer.scenarios_scroll,
            len, visible, delta,
        );
        if new_sel != self.explorer.selected_scenario {
            self.explorer.selected_scenario = new_sel;
            self.explorer.scenarios_scroll = new_scroll;
            self.explorer.preview_cache = None;
        }
    }

    /// Render the preview for the currently selected scenario.
    /// Returns the cached ANSI string, rendering fresh if needed.
    pub fn render_explorer_preview(&mut self, cols: u32, rows: u32) -> Option<String> {
        // Check cache — if dimensions match and cache exists, return it
        if self.explorer.preview_cache.is_some() && self.explorer.preview_dims == (cols, rows) {
            return self.explorer.preview_cache.clone();
        }

        let scenario_entry = self.explorer.scenarios.get(self.explorer.selected_scenario)?;
        if !scenario_entry.has_state {
            return None;
        }

        let behavior = self.explorer.behaviors.get(self.explorer.selected_behavior)?;
        let behavior_name = behavior.name.clone();
        let scenarios = behaviors::load_behavior_scenarios(&behavior.path)?;
        let scenario = scenarios.get(self.explorer.selected_scenario)?;
        let state_json = scenario.get("state")?;

        let n_lanes = self.config.octane.road.n_lanes;
        let lane_width = self.config.octane.road.lane_width;
        let mut frame = behaviors::build_frame_from_state(state_json, n_lanes, lane_width);

        // Inject action distribution from policy if current epoch has a snapshot
        self.ensure_explorer_action_probs(&behavior_name);
        if let Some(epoch) = self.current_snapshot_epoch() {
            let key = (behavior_name.clone(), epoch);
            if let Some(probs) = self.explorer.action_probs_cache.get(&key) {
                if let Some(dist) = probs.get(self.explorer.selected_scenario) {
                    frame.action_distribution = Some(dist.clone());
                }
            }
        }

        let scene = SceneEpisode::from_frames(vec![frame]);
        let ansi = self.render_explorer_scene(scene, cols, rows)?;

        self.explorer.preview_cache = Some(ansi.clone());
        self.explorer.preview_dims = (cols, rows);
        info!("Rendered explorer preview {}x{}", cols, rows);
        Some(ansi)
    }

    /// Ensure action probs are being fetched for the given behavior.
    /// Returns cached result instantly if available, otherwise spawns background fetch.
    fn ensure_explorer_action_probs(&mut self, behavior_name: &str) {
        let Some(epoch) = self.current_snapshot_epoch() else {
            self.explorer.action_probs_rx = None;
            return;
        };

        let key = (behavior_name.to_string(), epoch);

        // Already cached — nothing to do
        if self.explorer.action_probs_cache.contains_key(&key) {
            return;
        }

        // Already fetching — wait for it
        if self.explorer.action_probs_rx.is_some() {
            return;
        }

        // Spawn background fetch
        let trek_path = self.trek.path.clone();
        let behavior_name_owned = behavior_name.to_string();
        let (tx, rx) = mpsc::channel();
        self.explorer.action_probs_rx = Some(rx);
        std::thread::spawn(move || {
            let probs = behaviors::fetch_action_probs(&trek_path, epoch, &behavior_name_owned);
            let _ = tx.send(super::ActionProbsResult {
                behavior_name: behavior_name_owned,
                epoch,
                probs,
            });
        });
        info!("Spawned background action probs fetch for '{}' epoch {}", behavior_name, epoch);
    }

    /// Check if background action probs fetch has completed.
    /// Called each iteration of the main loop.
    pub fn check_action_probs_load(&mut self) {
        let rx = match self.explorer.action_probs_rx.take() {
            Some(rx) => rx,
            None => return,
        };
        match rx.try_recv() {
            Ok(result) => {
                let key = (result.behavior_name.clone(), result.epoch);
                if let Some(probs) = result.probs {
                    self.explorer.action_probs_cache.insert(key.clone(), probs);
                }
                // Invalidate preview if this result is for the current selection
                let current_behavior = self.explorer.behaviors
                    .get(self.explorer.selected_behavior)
                    .map(|b| b.name.as_str());
                let current_epoch = self.current_snapshot_epoch();
                if current_behavior == Some(&key.0) && current_epoch == Some(key.1) {
                    self.explorer.preview_cache = None;
                }
                info!("Cached action probs for '{}' epoch {}", key.0, key.1);
            }
            Err(mpsc::TryRecvError::Empty) => {
                self.explorer.action_probs_rx = Some(rx);
            }
            Err(mpsc::TryRecvError::Disconnected) => {}
        }
    }

    /// Get the epoch number if the currently selected epoch has a snapshot.
    pub(crate) fn current_snapshot_epoch(&self) -> Option<i64> {
        let epoch = self.trek.epochs.get(self.selected_epoch)?;
        if self.trek.snapshot_epochs.contains(&epoch.epoch_number) {
            Some(epoch.epoch_number)
        } else {
            None
        }
    }

    /// Render a static scene with action distribution overlay enabled.
    fn render_explorer_scene(
        &self,
        scene: SceneEpisode,
        cols: u32,
        rows: u32,
    ) -> Option<String> {
        use crate::mango::{render_svg_to_ansi, MangoConfig};
        use crate::worlds::{SvgConfig, SvgEpisode, ViewportConfig};

        let env_type = self.trek.env_type?;

        let vp_config = ViewportConfig {
            zoom: self.zoom,
            corn_aspro: self.config.octane.rendering.corn_aspro,
            ..ViewportConfig::default()
        };
        let viewport =
            env_type.build_viewport(scene, vp_config, &self.trek, &self.config);
        let svg_config = SvgConfig::new(cols, rows, self.config.octane.rendering.corn_aspro);
        let svg_episode = SvgEpisode::new(viewport, svg_config);

        // Build render config with action distribution forced on when we have probs
        let has_probs = self.current_snapshot_epoch().is_some_and(|epoch| {
            let behavior_name = self.explorer.behaviors
                .get(self.explorer.selected_behavior)
                .map(|b| b.name.clone())
                .unwrap_or_default();
            self.explorer.action_probs_cache.contains_key(&(behavior_name, epoch))
        });
        let render_config = {
            let mut rc = crate::envs::highway::build_render_config(
                &self.config,
                &self.trek,
                cols,
                rows,
                &self.highway_prefs,
                true, // is_paused
                false,
                self.scene_theme,
            );
            if has_probs {
                let crate::render::SceneRenderConfig::Highway(ref mut hrc) = &mut rc;
                hrc.show_action_distribution = true;
                hrc.show_action_distribution_text = true;
            }
            rc
        };

        let svg = render_config.render_svg(&svg_episode, 0.0)?;

        let mango_config = MangoConfig {
            n_cols: cols,
            n_rows: rows,
            use_sextants: self.use_sextants,
            use_octants: self.use_octants,
        };
        render_svg_to_ansi(&svg, &mango_config).ok()
    }

    /// Navigate to the source location of the selected scenario in the Runs tab.
    pub fn navigate_to_scenario_source(&mut self) {
        let Some(entry) = self.explorer.scenarios.get(self.explorer.selected_scenario) else {
            return;
        };

        let Some(epoch_number) = entry.source_epoch else {
            self.show_toast(
                "No source info on this scenario".to_string(),
                std::time::Duration::from_secs(3),
            );
            return;
        };
        let episode_idx = entry.source_episode.unwrap_or(0);
        let policy_t = entry.source_t.unwrap_or(0.0);
        let source_target = entry.source_target.clone();

        // If source has a target path, switch to the right trek/parquet
        if let Some(ref target) = source_target {
            let is_pq_file = target.extension().is_some_and(|e| e == "pq")
                || (target.exists() && target.is_file());
            let target_trek = if is_pq_file {
                // Walk up to find containing trek directory (meta.yaml)
                let resolved = target.canonicalize().unwrap_or_else(|_| target.clone());
                let mut dir = resolved.parent();
                let mut found = None;
                while let Some(d) = dir {
                    if d.join("meta.yaml").exists() {
                        found = Some(d.to_path_buf());
                        break;
                    }
                    dir = d.parent();
                }
                found
            } else if target.is_dir() || target.join("meta.yaml").exists() {
                target.canonicalize().ok().or_else(|| Some(target.clone()))
            } else {
                None
            };

            if let Some(ref trek_dir) = target_trek {
                // Check if we need to switch treks
                let current_canon = self.trek.path.canonicalize()
                    .unwrap_or_else(|_| self.trek.path.clone());
                let target_canon = trek_dir.canonicalize()
                    .unwrap_or_else(|_| trek_dir.clone());
                if current_canon != target_canon {
                    // Find matching trek in trek_entries
                    let trek_idx = self.trek_entries.iter().position(|e| {
                        e.path.canonicalize().unwrap_or_else(|_| e.path.clone())
                            == target_canon
                    });
                    if let Some(idx) = trek_idx {
                        self.selected_trek = idx;
                        self.load_selected_trek();
                    } else {
                        self.show_toast(
                            format!("Trek not found in trek list: {}",
                                    crate::util::posh_path(trek_dir)),
                            std::time::Duration::from_secs(3),
                        );
                        return;
                    }
                }
            }

            // If target was a pq file, switch to that parquet source
            if is_pq_file {
                if let Ok(canonical) = target.canonicalize() {
                    if let Some(idx) = self.parquet_sources.iter().position(|s| {
                        s.path.canonicalize().ok().as_ref() == Some(&canonical)
                    }) {
                        if self.selected_parquet != idx {
                            self.selected_parquet = idx;
                            self.adjust_parquets_scroll();
                            self.switch_parquet_source();
                        }
                    }
                }
            } else {
                // Trek dir target — default to sample_es (parquet index 0)
                if self.selected_parquet != 0 {
                    self.selected_parquet = 0;
                    self.adjust_parquets_scroll();
                    self.switch_parquet_source();
                }
            }
        } else {
            // No target — default to sample_es
            if self.selected_parquet != 0 {
                self.selected_parquet = 0;
                self.adjust_parquets_scroll();
                self.switch_parquet_source();
            }
        }

        // Find epoch by epoch_number
        let epoch_pos = self.trek.epochs.iter()
            .position(|e| e.epoch_number == epoch_number);
        let Some(epoch_idx) = epoch_pos else {
            self.show_toast(
                format!("Epoch {} not found in current trek", epoch_number),
                std::time::Duration::from_secs(3),
            );
            return;
        };

        // Validate episode index
        let n_episodes = self.trek.epochs[epoch_idx].episodes.len();
        if episode_idx >= n_episodes {
            self.show_toast(
                format!("Episode {} not found in epoch {}", episode_idx, epoch_number),
                std::time::Duration::from_secs(3),
            );
            return;
        };

        // Navigate
        self.save_episode_position();
        self.selected_epoch = epoch_idx;
        self.selected_episode = episode_idx;
        self.pending_timestep = Some(policy_t);
        self.stop_playback();
        self.adjust_epochs_scroll();
        self.adjust_episodes_scroll();

        // Invalidate cached episode so it reloads at the new location
        self.cached_episode_key = (usize::MAX, usize::MAX);
        self.current_episode = None;
        self.invalidate_viewport();
        self.reset_effective_timing();

        // Switch to Runs tab
        self.screen = Screen::Browse;
        self.focus = Focus::Highway;

        info!(
            "Navigated to scenario source: epoch={}, episode={}, t={}",
            epoch_number, episode_idx, policy_t
        );
    }

    /// Enter edit mode for the currently selected scenario.
    pub fn enter_explorer_edit(&mut self) {
        let Some(scenario_entry) = self.explorer.scenarios.get(self.explorer.selected_scenario) else {
            return;
        };
        if !scenario_entry.has_state {
            self.show_toast(
                "Cannot edit observation-only scenarios".to_string(),
                std::time::Duration::from_secs(3),
            );
            return;
        }

        let Some(behavior) = self.explorer.behaviors.get(self.explorer.selected_behavior) else {
            return;
        };
        let Some(scenarios) = behaviors::load_behavior_scenarios(&behavior.path) else {
            return;
        };
        let Some(scenario) = scenarios.get(self.explorer.selected_scenario) else {
            return;
        };
        if scenario.get("state").is_none() {
            return;
        }

        self.explorer.editor_fields = super::explorer_editing::build_editor_fields(scenario);
        self.explorer.editor_selected_field = 0;
        self.explorer.editor_dirty = false;
        self.explorer.mode = ExplorerMode::Edit;
        self.explorer.preview_cache = None;
        info!("Entered editor with {} fields", self.explorer.editor_fields.len());
    }

    /// Exit edit mode, discarding unsaved changes.
    pub fn exit_explorer_edit(&mut self) {
        self.explorer.mode = ExplorerMode::Browse;
        self.explorer.editor_fields.clear();
        self.explorer.editor_dirty = false;
        self.explorer.preview_cache = None;
        info!("Exited editor");
    }
}
