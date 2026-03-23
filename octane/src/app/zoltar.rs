//! Zoltar query handlers for the App.
//!
//! Implements the server side of zoltar commands: genco (state snapshot),
//! press (key injection), navigate (jump to position), and pane queries.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::data::behaviors;
use crate::worlds::{SceneEpisode, SvgConfig, SvgEpisode, ViewportConfig};
use crate::zoltar::protocol::{ZoltarRequest, ZoltarResponse};

impl super::App {
    /// Dispatch a zoltar request to the appropriate handler.
    pub(super) fn handle_zoltar_request(&mut self, request: &ZoltarRequest) -> ZoltarResponse {
        match request {
            ZoltarRequest::Ping => ZoltarResponse::pong(),
            ZoltarRequest::Genco => self.handle_genco(),
            ZoltarRequest::Press { keys } => self.handle_press(keys),
            ZoltarRequest::Navigate { epoch, episode, timestep } => {
                self.handle_navigate(*epoch, *episode, *timestep)
            }
            ZoltarRequest::Pane { pane, query, path } => {
                self.handle_pane(pane, query, path.as_deref())
            }
        }
    }

    /// Genco: return a semantic state snapshot.
    fn handle_genco(&self) -> ZoltarResponse {
        let epoch = self.trek.epochs.get(self.selected_epoch);
        let epoch_number = epoch.map(|e| e.epoch_number);
        let episode_meta = epoch.and_then(|e| e.episodes.get(self.selected_episode));

        let n_frames = self.current_episode.as_ref().map(|ep| ep.frames.len()).unwrap_or(0);

        // Compute t-value from frame_index using the same logic as format_timestep
        let timestep: Option<f64> = if n_frames > 0 {
            let n_sub = self.effective_n_sub;
            if n_sub > 1 {
                let policy_t = self.frame_index / n_sub;
                let fraction = (self.frame_index % n_sub) as f64 / n_sub as f64;
                Some(policy_t as f64 + fraction)
            } else {
                Some(self.frame_index as f64)
            }
        } else {
            None
        };

        let trek_path = self.trek.path.to_string_lossy().to_string();

        let json = serde_json::json!({
            "screen": format!("{:?}", self.screen),
            "focus": format!("{:?}", self.focus),
            "epoch": epoch_number,
            "epoch_index": self.selected_epoch,
            "episode": episode_meta.and_then(|m| m.es_episode),
            "episode_index": self.selected_episode,
            "frame_index": self.frame_index,
            "timestep": timestep,
            "n_frames": n_frames,
            "n_epochs": self.trek.epochs.len(),
            "playing": self.playback.playing,
            "trek_path": trek_path,
            "last_toast": self.last_toast_text,
            "explorer": {
                "pane_focus": format!("{:?}", self.explorer.pane_focus),
                "mode": format!("{:?}", self.explorer.mode),
                "selected_behavior": self.explorer.selected_behavior,
                "selected_scenario": self.explorer.selected_scenario,
                "action_probs_cache_size": self.explorer.action_probs_cache.len(),
                "has_action_probs": self.current_snapshot_epoch().map(|epoch| {
                    let behavior_name = self.explorer.behaviors
                        .get(self.explorer.selected_behavior)
                        .map(|b| b.name.clone())
                        .unwrap_or_default();
                    self.explorer.action_probs_cache.contains_key(&(behavior_name, epoch))
                }).unwrap_or(false),
                "action_probs_epoch": self.current_snapshot_epoch(),
                "action_probs_behavior": self.current_snapshot_epoch().and_then(|_| {
                    self.explorer.behaviors
                        .get(self.explorer.selected_behavior)
                        .map(|b| b.name.clone())
                }),
            },
        });

        ZoltarResponse::Json(json)
    }

    /// Press: inject key events, return genco after processing.
    fn handle_press(&mut self, keys: &[String]) -> ZoltarResponse {
        for key_str in keys {
            if let Some(key_event) = parse_key_string(key_str) {
                self.handle_key(key_event);
            }
        }
        self.handle_genco()
    }

    /// Navigate: jump to a specific epoch/episode/timestep, return genco.
    fn handle_navigate(
        &mut self,
        epoch: Option<u64>,
        episode: Option<u64>,
        timestep: Option<u64>,
    ) -> ZoltarResponse {
        if let Some(epoch_num) = epoch {
            // Find epoch by number
            if let Some(idx) = self.trek.epochs.iter().position(|e| e.epoch_number == epoch_num as i64) {
                self.selected_epoch = idx;
                // Reset episode selection when changing epoch
                self.selected_episode = 0;
                self.frame_index = 0;
            } else {
                return ZoltarResponse::error(format!("epoch {} not found", epoch_num));
            }
        }

        if let Some(ep_num) = episode {
            // Find episode by es_episode number
            let epoch = &self.trek.epochs[self.selected_epoch];
            if let Some(idx) = epoch.episodes.iter().position(|e| e.es_episode == Some(ep_num as i64)) {
                self.selected_episode = idx;
                self.frame_index = 0;
            } else {
                return ZoltarResponse::error(format!("episode {} not found in epoch", ep_num));
            }
        }

        if let Some(t) = timestep {
            // Store as pending timestep — will be resolved to frame_index
            // after the episode loads (deferred lookup via binary search on t values)
            self.pending_timestep = Some(t as f64);
        }

        self.handle_genco()
    }

    /// Dispatch a pane query to the appropriate handler.
    fn handle_pane(&self, pane: &str, query: &str, path: Option<&str>) -> ZoltarResponse {
        match (pane, query) {
            ("scene", "svg") => self.pane_scene_svg(),
            ("scene", "screenshot") => self.pane_scene_screenshot(path),
            ("metrics", "data") => self.pane_metrics_data(),
            ("epochs", "data") => self.pane_epochs_data(),
            ("episodes", "data") => self.pane_episodes_data(),
            ("treks", "data") => self.pane_treks_data(),
            ("parquets", "data") => self.pane_parquets_data(),
            ("explorer.behaviors", "data") => self.pane_explorer_behaviors_data(),
            ("explorer.scenarios", "data") => self.pane_explorer_scenarios_data(),
            ("explorer.preview", "svg") => self.pane_explorer_preview_svg(),
            ("explorer.preview", "screenshot") => self.pane_explorer_preview_screenshot(path),
            ("explorer.info", "data") => self.pane_explorer_info_data(),
            ("explorer.editor", "data") => self.pane_explorer_editor_data(),
            _ => ZoltarResponse::error(format!("unknown pane/query: {} {}", pane, query)),
        }
    }

    /// Pane scene svg: return the SVG string for the current frame.
    fn pane_scene_svg(&self) -> ZoltarResponse {
        let Some(ref svg_episode) = self.svg_episode else {
            return ZoltarResponse::error("no scene available (episode not loaded or no dimensions)");
        };
        let scene_time = self.playback.scene_time;
        let n_cols = svg_episode.config().n_cols;
        let n_rows = svg_episode.config().n_rows;
        let Some(render_config) = self.scene_render_config(n_cols, n_rows) else {
            return ZoltarResponse::error("unknown env type");
        };
        match render_config.render_svg(svg_episode, scene_time) {
            Some(svg) => ZoltarResponse::Json(serde_json::json!({"svg": svg})),
            None => ZoltarResponse::error("failed to render scene SVG"),
        }
    }

    /// Pane scene screenshot: render to PNG, write to path, return path.
    fn pane_scene_screenshot(&self, path: Option<&str>) -> ZoltarResponse {
        let Some(path) = path else {
            return ZoltarResponse::error("screenshot requires 'path' field");
        };
        let Some(ref svg_episode) = self.svg_episode else {
            return ZoltarResponse::error("no scene available (episode not loaded or no dimensions)");
        };
        let scene_time = self.playback.scene_time;
        let n_cols = svg_episode.config().n_cols;
        let n_rows = svg_episode.config().n_rows;
        let Some(render_config) = self.scene_render_config(n_cols, n_rows) else {
            return ZoltarResponse::error("unknown env type");
        };
        let Some(svg) = render_config.render_svg(svg_episode, scene_time) else {
            return ZoltarResponse::error("failed to render scene SVG");
        };

        // Parse SVG and render to PNG using resvg
        let tree = match resvg::usvg::Tree::from_str(
            &svg,
            &resvg::usvg::Options::default(),
        ) {
            Ok(tree) => tree,
            Err(e) => return ZoltarResponse::error(format!("SVG parse error: {}", e)),
        };
        let size = tree.size();
        let width = size.width() as u32;
        let height = size.height() as u32;
        let Some(mut pixmap) = resvg::tiny_skia::Pixmap::new(width, height) else {
            return ZoltarResponse::error("failed to create pixmap");
        };
        resvg::render(&tree, resvg::tiny_skia::Transform::default(), &mut pixmap.as_mut());
        match pixmap.save_png(path) {
            Ok(()) => ZoltarResponse::Json(serde_json::json!({"ok": true, "path": path})),
            Err(e) => ZoltarResponse::error(format!("failed to save PNG: {}", e)),
        }
    }

    /// Pane metrics data: return per-frame metrics for the current frame.
    fn pane_metrics_data(&self) -> ZoltarResponse {
        let frame = self.current_episode.as_ref()
            .and_then(|ep| ep.frames.get(self.frame_index));

        let Some(frame) = frame else {
            return ZoltarResponse::error("no current frame");
        };

        let ego = frame.vehicle_state.as_ref().map(|vs| &vs.ego);
        let env_type = self.trek.env_type.unwrap_or(crate::envs::EnvType::Highway);
        let action_name = frame.action_name.as_deref()
            .or_else(|| frame.vehicle_state.as_ref()
                .and_then(|vs| vs.chosen_action)
                .map(|a| env_type.action_name(a)));

        ZoltarResponse::Json(serde_json::json!({
            "timestep": self.format_timestep(self.frame_index),
            "reward": frame.reward,
            "crash_reward": frame.crash_reward,
            "action": frame.action,
            "action_name": action_name,
            "v": frame.v,
            "return": frame.return_value,
            "logp": frame.tendency,
            "advantage": frame.advantage,
            "nz_advantage": frame.nz_advantage,
            "ego_speed": ego.map(|e| e.speed),
            "ego_heading": ego.map(|e| e.heading),
            "ego_x": ego.map(|e| e.x),
            "ego_y": ego.map(|e| e.y),
            "crashed": frame.vehicle_state.as_ref().map(|vs| vs.crashed).unwrap_or(false),
        }))
    }

    /// Pane epochs data: return epoch list with selection.
    fn pane_epochs_data(&self) -> ZoltarResponse {
        let epochs: Vec<serde_json::Value> = self.trek.epochs.iter().map(|e| {
            serde_json::json!({
                "number": e.epoch_number,
                "n_episodes": e.episode_count(),
                "mean_nreturn": e.mean_nreturn(),
                "survival_fraction": e.epochia_alive_fraction,
                "has_snapshot": self.trek.snapshot_epochs.contains(&e.epoch_number),
            })
        }).collect();

        ZoltarResponse::Json(serde_json::json!({
            "selected": self.selected_epoch,
            "epochs": epochs,
        }))
    }

    /// Pane episodes data: return episode list for current epoch with selection.
    fn pane_episodes_data(&self) -> ZoltarResponse {
        let max_policy_frames = self.trek.epochs.iter()
            .flat_map(|e| e.episodes.iter())
            .map(|ep| ep.n_policy_frames)
            .max()
            .unwrap_or(1);
        let epoch = self.trek.epochs.get(self.selected_epoch);
        let episodes: Vec<serde_json::Value> = epoch.map(|e| {
            e.episodes.iter().map(|ep| {
                serde_json::json!({
                    "es_episode": ep.es_episode,
                    "n_frames": ep.n_frames,
                    "n_policy_frames": ep.n_policy_frames,
                    "reward": ep.total_reward,
                    "survival": ep.n_policy_frames as f64 / max_policy_frames as f64,
                })
            }).collect()
        }).unwrap_or_default();

        ZoltarResponse::Json(serde_json::json!({
            "selected": self.selected_episode,
            "episodes": episodes,
        }))
    }

    /// Pane treks data: return trek list with selection.
    fn pane_treks_data(&self) -> ZoltarResponse {
        let treks: Vec<serde_json::Value> = self.trek_entries.iter().map(|t| {
            serde_json::json!({
                "display": t.display,
                "path": t.path.to_string_lossy(),
            })
        }).collect();

        ZoltarResponse::Json(serde_json::json!({
            "selected": self.selected_trek,
            "treks": treks,
        }))
    }

    /// Pane parquets data: return parquet source list with selection.
    fn pane_parquets_data(&self) -> ZoltarResponse {
        let parquets: Vec<serde_json::Value> = self.parquet_sources.iter().map(|p| {
            serde_json::json!({
                "display": p.display,
                "relative_path": p.relative_path,
            })
        }).collect();

        ZoltarResponse::Json(serde_json::json!({
            "selected": self.selected_parquet,
            "parquets": parquets,
        }))
    }

    /// Pane explorer.behaviors data: return behavior list with selection.
    fn pane_explorer_behaviors_data(&self) -> ZoltarResponse {
        let behaviors: Vec<serde_json::Value> = self.explorer.behaviors.iter().map(|b| {
            serde_json::json!({
                "name": b.name,
                "n_scenarios": b.n_scenarios,
                "is_preset": b.is_preset,
            })
        }).collect();

        ZoltarResponse::Json(serde_json::json!({
            "selected": self.explorer.selected_behavior,
            "behaviors": behaviors,
        }))
    }

    /// Pane explorer.scenarios data: return scenario list with selection.
    fn pane_explorer_scenarios_data(&self) -> ZoltarResponse {
        let scenarios: Vec<serde_json::Value> = self.explorer.scenarios.iter().map(|s| {
            serde_json::json!({
                "index": s.index,
                "ego_speed": s.ego_speed,
                "ego_heading": s.ego_heading,
                "n_npcs": s.n_npcs,
                "action_weights": s.action_weights,
                "edited": s.edited,
                "has_state": s.has_state,
                "source_target": s.source_target.as_ref().map(|p| p.display().to_string()),
                "source_epoch": s.source_epoch,
                "source_episode": s.source_episode,
                "source_t": s.source_t,
            })
        }).collect();

        ZoltarResponse::Json(serde_json::json!({
            "selected": self.explorer.selected_scenario,
            "scenarios": scenarios,
        }))
    }

    /// Pane explorer.preview svg: render the selected scenario preview as SVG.
    fn pane_explorer_preview_svg(&self) -> ZoltarResponse {
        let scenario_entry = match self.explorer.scenarios.get(self.explorer.selected_scenario) {
            Some(e) if e.has_state => e,
            Some(_) => return ZoltarResponse::error("selected scenario has no state"),
            None => return ZoltarResponse::error("no scenario selected"),
        };
        let _ = scenario_entry; // used for has_state check above

        let behavior = match self.explorer.behaviors.get(self.explorer.selected_behavior) {
            Some(b) => b,
            None => return ZoltarResponse::error("no behavior selected"),
        };
        let scenarios = match behaviors::load_behavior_scenarios(&behavior.path) {
            Some(s) => s,
            None => return ZoltarResponse::error("failed to load behavior scenarios"),
        };
        let scenario = match scenarios.get(self.explorer.selected_scenario) {
            Some(s) => s,
            None => return ZoltarResponse::error("scenario index out of range"),
        };
        let state_json = match scenario.get("state") {
            Some(s) => s,
            None => return ZoltarResponse::error("scenario has no state"),
        };

        let n_lanes = self.config.octane.road.n_lanes;
        let lane_width = self.config.octane.road.lane_width;
        let frame = behaviors::build_frame_from_state(state_json, n_lanes, lane_width);
        let scene = SceneEpisode::from_frames(vec![frame]);

        let vp_config = ViewportConfig {
            zoom: self.zoom,
            corn_aspro: self.config.octane.rendering.corn_aspro,
            ..ViewportConfig::default()
        };
        let Some(env_type) = self.trek.env_type else {
            return ZoltarResponse::error("unknown env type");
        };
        let viewport = env_type.build_viewport(
            scene, vp_config, &self.trek, &self.config,
        );
        let svg_config = SvgConfig::new(80, 24, self.config.octane.rendering.corn_aspro);
        let svg_episode = SvgEpisode::new(viewport, svg_config);
        let Some(render_config) = self.scene_render_config(80, 24) else {
            return ZoltarResponse::error("unknown env type");
        };

        match render_config.render_svg(&svg_episode, 0.0) {
            Some(svg) => ZoltarResponse::Json(serde_json::json!({"svg": svg})),
            None => ZoltarResponse::error("failed to render explorer preview SVG"),
        }
    }

    /// Pane explorer.preview screenshot: render preview to PNG.
    fn pane_explorer_preview_screenshot(&self, path: Option<&str>) -> ZoltarResponse {
        let Some(path) = path else {
            return ZoltarResponse::error("screenshot requires 'path' field");
        };
        // Get the SVG first
        let svg_resp = self.pane_explorer_preview_svg();
        let svg = match &svg_resp {
            ZoltarResponse::Json(json) => match json["svg"].as_str() {
                Some(s) => s.to_string(),
                None => return ZoltarResponse::error("internal: no svg in preview response"),
            },
            ZoltarResponse::Error { .. } => return svg_resp,
        };

        let tree = match resvg::usvg::Tree::from_str(&svg, &resvg::usvg::Options::default()) {
            Ok(tree) => tree,
            Err(e) => return ZoltarResponse::error(format!("SVG parse error: {}", e)),
        };
        let size = tree.size();
        let width = size.width() as u32;
        let height = size.height() as u32;
        let Some(mut pixmap) = resvg::tiny_skia::Pixmap::new(width, height) else {
            return ZoltarResponse::error("failed to create pixmap");
        };
        resvg::render(&tree, resvg::tiny_skia::Transform::default(), &mut pixmap.as_mut());
        match pixmap.save_png(path) {
            Ok(()) => ZoltarResponse::Json(serde_json::json!({"ok": true, "path": path})),
            Err(e) => ZoltarResponse::error(format!("failed to save PNG: {}", e)),
        }
    }

    /// Pane explorer.info data: return info about the selected scenario.
    fn pane_explorer_info_data(&self) -> ZoltarResponse {
        let scenario = self.explorer.scenarios.get(self.explorer.selected_scenario);
        match scenario {
            Some(s) => ZoltarResponse::Json(serde_json::json!({
                "ego_speed": s.ego_speed,
                "ego_heading": s.ego_heading,
                "n_npcs": s.n_npcs,
            })),
            None => ZoltarResponse::error("no scenario selected"),
        }
    }

    /// Pane explorer.editor data: return editor fields and state.
    fn pane_explorer_editor_data(&self) -> ZoltarResponse {
        let fields: Vec<serde_json::Value> = self.explorer.editor_fields.iter().map(|f| {
            serde_json::json!({
                "label": f.label,
                "value": f.value,
                "kind": format!("{:?}", f.kind),
            })
        }).collect();

        ZoltarResponse::Json(serde_json::json!({
            "fields": fields,
            "selected_field": self.explorer.editor_selected_field,
            "dirty": self.explorer.editor_dirty,
        }))
    }
}

/// Parse a key string (like "g", "enter", "shift+b") into a KeyEvent.
fn parse_key_string(s: &str) -> Option<KeyEvent> {
    let s = s.to_lowercase();
    let (modifiers, key_part) = if let Some(rest) = s.strip_prefix("shift+") {
        (KeyModifiers::SHIFT, rest)
    } else if let Some(rest) = s.strip_prefix("ctrl+") {
        (KeyModifiers::CONTROL, rest)
    } else if let Some(rest) = s.strip_prefix("alt+") {
        (KeyModifiers::ALT, rest)
    } else {
        (KeyModifiers::NONE, s.as_str())
    };

    let code = match key_part {
        "enter" | "return" => KeyCode::Enter,
        "esc" | "escape" => KeyCode::Esc,
        "tab" => KeyCode::Tab,
        "backspace" | "bs" => KeyCode::Backspace,
        "space" => KeyCode::Char(' '),
        "up" => KeyCode::Up,
        "down" => KeyCode::Down,
        "left" => KeyCode::Left,
        "right" => KeyCode::Right,
        "home" => KeyCode::Home,
        "end" => KeyCode::End,
        "pageup" | "pgup" => KeyCode::PageUp,
        "pagedown" | "pgdn" => KeyCode::PageDown,
        "delete" | "del" => KeyCode::Delete,
        s if s.len() == 1 => {
            let ch = s.chars().next().unwrap();
            if modifiers.contains(KeyModifiers::SHIFT) {
                KeyCode::Char(ch.to_ascii_uppercase())
            } else {
                KeyCode::Char(ch)
            }
        }
        _ => return None,
    };

    Some(KeyEvent::new(code, modifiers))
}
