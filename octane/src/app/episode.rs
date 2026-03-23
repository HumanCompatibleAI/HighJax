//! Episode loading, viewport building, and config application.

use tracing::info;

use crate::config::{self, Config};
use crate::data::FrameState;
use crate::render::SceneRenderConfig;
use crate::worlds::{SceneEpisode, SvgConfig, SvgEpisode, ViewportConfig};

use super::App;

impl App {
    /// Build a SceneRenderConfig appropriate for the current env type.
    /// Applies config defaults, trek overrides, per-env prefs, and app state.
    /// Returns None if env type is unknown.
    pub fn scene_render_config(&self, n_cols: u32, n_rows: u32) -> Option<SceneRenderConfig> {
        let _env_type = self.trek.env_type?;
        let is_paused = !self.playback.playing;
        Some(crate::envs::highway::build_render_config(
            &self.config, &self.trek, n_cols, n_rows,
            &self.highway_prefs, is_paused, self.show_debug,
            self.scene_theme,
        ))
    }

    /// Apply a new config (from hot-reload). Updates all derived fields.
    pub fn apply_config(&mut self, new_config: Config) {
        self.playback.speed = new_config.octane.rendering.playback_speed;
        self.playback.fps = new_config.octane.rendering.fps;
        self.use_sextants = new_config.octane.rendering.use_sextants;
        self.use_octants = new_config.octane.rendering.use_octants;
        self.scene_theme = new_config.octane.rendering.theme;
        self.omega = new_config.octane.podium.omega;
        self.highway_prefs.apply_config(&new_config, self.scene_theme);
        self.config = new_config;
        // Rebuild viewport with new config values
        self.invalidate_viewport();
        info!("Applied hot-reloaded config");
    }

    /// Check for config hot-reload updates and apply if available.
    pub fn check_config_updates(&mut self) {
        if let Some(new_config) = config::check_config_updates(&self.config_rx) {
            self.apply_config(new_config);
        }
    }

    /// Build viewport and SVG episodes with specific scene dimensions.
    pub fn compute_episode_viewport_with_dims(&mut self, cols: u32, rows: u32) {
        self.viewport_episode = None;
        self.svg_episode = None;
        self.scene_dims = (cols, rows);

        let Some(ref episode) = self.current_episode else {
            return;
        };

        // Convert episode frames to FrameState for SceneEpisode
        let frame_states: Vec<FrameState> = episode
            .frames
            .iter()
            .filter_map(|frame| frame.vehicle_state.clone())
            .collect();

        if frame_states.is_empty() {
            return;
        }

        let n_frames = frame_states.len();

        // Build SceneEpisode (dt depends on whether episode has sub-step rows)
        let mut scene = SceneEpisode::from_frames_with_timestep(frame_states, self.effective_seconds_per_frame);
        scene.set_acceleration_lookback_seconds(
            self.config.octane.rendering.brakelight_deceleration_lookback_seconds,
        );

        // Build ViewportConfig from central config + runtime overrides
        let vp_config = ViewportConfig {
            zoom: self.zoom,
            omega: self.omega,
            corn_aspro: self.config.octane.rendering.corn_aspro,
            podium_fraction: self.config.octane.podium.offset,
            damping_ratio: self.config.octane.podium.damping_ratio,
        };

        // Build ViewportEpisode (env-type specific)
        let Some(env_type) = self.trek.env_type else {
            return;
        };
        let viewport = env_type.build_viewport(
            scene, vp_config, &self.trek, &self.config,
        );

        info!(
            "Built ViewportEpisode: {} frames, zoom={:.2}, env={:?}",
            n_frames, self.zoom, self.trek.env_type
        );

        // Build SvgEpisode if we have valid dimensions
        if cols > 0 && rows > 0 {
            let svg_config = SvgConfig::new(cols, rows, self.config.octane.rendering.corn_aspro);
            let svg_episode = SvgEpisode::new(viewport.clone(), svg_config);
            self.svg_episode = Some(svg_episode);
            info!("Built SvgEpisode: {}x{}", cols, rows);
        }

        self.viewport_episode = Some(viewport);
    }

    /// Invalidate viewport to trigger rebuild on next frame.
    /// Call this when zoom or omega changes.
    pub fn invalidate_viewport(&mut self) {
        self.viewport_episode = None;
        self.svg_episode = None;
    }

    /// Render a single-frame scene to an ANSI string.
    ///
    /// Shared by the behavior explorer preview and editor preview.
    /// Uses the app's current graphics settings (zoom, sextants, etc.)
    /// and proper road geometry from config.
    pub fn render_static_scene(
        &self,
        scene: SceneEpisode,
        cols: u32,
        rows: u32,
    ) -> Option<String> {
        use crate::mango::{render_svg_to_ansi, MangoConfig};

        let env_type = self.trek.env_type?;

        let vp_config = ViewportConfig {
            zoom: self.zoom,
            corn_aspro: self.config.octane.rendering.corn_aspro,
            ..ViewportConfig::default()
        };
        let viewport = env_type.build_viewport(
            scene, vp_config, &self.trek, &self.config,
        );
        let svg_config = SvgConfig::new(cols, rows, self.config.octane.rendering.corn_aspro);
        let svg_episode = SvgEpisode::new(viewport, svg_config);

        let render_config = self.scene_render_config(cols, rows)?;
        let svg = render_config.render_svg(&svg_episode, 0.0)?;

        let mango_config = MangoConfig {
            n_cols: cols,
            n_rows: rows,
            use_sextants: self.use_sextants,
            use_octants: self.use_octants,
        };
        render_svg_to_ansi(&svg, &mango_config).ok()
    }

    /// Get current ego speed (estimated from viewport velocity).
    pub fn current_ego_speed(&self) -> Option<f64> {
        let vp = self.viewport_episode.as_ref()?;
        let state = vp.scene().state_at(self.playback.scene_time)?;
        Some(state.ego.speed)
    }

    /// Get current ego heading in radians.
    pub fn current_ego_heading(&self) -> Option<f64> {
        let vp = self.viewport_episode.as_ref()?;
        let state = vp.scene().state_at(self.playback.scene_time)?;
        Some(state.ego.heading)
    }

    /// Ensure the current episode is loaded into cache.
    pub fn ensure_episode_loaded(&mut self) {
        let key = (self.selected_epoch, self.selected_episode);
        if self.cached_episode_key == key && self.current_episode.is_some() {
            // Episode already cached — still check for deferred timestep
            self.apply_pending_timestep();
            return;
        }

        // Extract es_epoch/es_episode from metadata to avoid borrow conflicts
        let es_keys = self.trek.epochs
            .get(self.selected_epoch)
            .and_then(|e| e.episodes.get(self.selected_episode))
            .and_then(|ep| match (ep.es_epoch, ep.es_episode) {
                (Some(a), Some(b)) => Some((a, b)),
                _ => None,
            });

        info!("Loading episode {}:{}", key.0, key.1);

        // Try parquet index (with retry for concurrent writer)
        if let (Some(ref pq_index), Some((es_epoch, es_episode)), Some(env_type)) = (&self.trek.es_parquet_index, es_keys, self.trek.env_type) {
            let format_name = "sample_es parquet";
            let max_retries = 3;
            for attempt in 0..max_retries {
                match pq_index.load_episode(es_epoch, es_episode, env_type) {
                    Ok(frames_data) => {
                        {
                            let t0 = frames_data.get(0).map(|f| f.t);
                            let t1 = frames_data.get(1).map(|f| f.t);
                            let fc = frames_data.len();
                            let np = frames_data.iter()
                                .filter(|f| (f.t - f.t.floor()).abs() < 1e-6)
                                .count();
                            self.detect_effective_timing(t0, t1, fc, np);
                        }
                        // Apply deferred timestep (from --timestep CLI or scenario source nav)
                        if let Some(ts) = self.pending_timestep.take() {
                            let idx = match frames_data.binary_search_by(|f| {
                                f.t.partial_cmp(&ts).unwrap()
                            }) {
                                Ok(i) => i,
                                Err(i) => {
                                    if i == 0 {
                                        0
                                    } else if i >= frames_data.len() {
                                        frames_data.len() - 1
                                    } else if (frames_data[i - 1].t - ts).abs()
                                        <= (frames_data[i].t - ts).abs()
                                    {
                                        i - 1
                                    } else {
                                        i
                                    }
                                }
                            };
                            let found_t = frames_data[idx].t;
                            if (found_t - ts).abs() > 0.5 {
                                let msg = format!(
                                    "Timestep {} not found (nearest: {})",
                                    ts, found_t
                                );
                                tracing::warn!("{}", msg);
                                self.show_toast(
                                    msg,
                                    std::time::Duration::from_secs(5),
                                );
                            }
                            self.frame_index = idx;
                            self.reset_playback_timing();
                            info!(
                                "Timestep {} resolved to frame {} (t={})",
                                ts, idx, found_t
                            );
                        }

                        let frame_count = frames_data.len();
                        let total_reward: f64 = frames_data.iter()
                            .filter_map(|f| f.reward)
                            .sum::<f64>()
                            + frames_data.last()
                                .and_then(|f| f.crash_reward)
                                .unwrap_or(0.0);
                        let mut last_action: u8 = 0; // default to first action
                        let mut last_action_name: Option<String> = None;
                        let mut last_reward: f64 = 0.0;
                        let mut last_v: Option<f64> = None;
                        let mut last_return: Option<f64> = None;
                        let mut last_tendency: Option<f64> = None;
                        let mut last_adv: Option<f64> = None;
                        let mut last_nadv: Option<f64> = None;
                        let frames: Vec<crate::data::Frame> = frames_data
                            .into_iter()
                            .enumerate()
                            .map(|(i, es_frame)| {
                                let is_last = i == frame_count - 1;
                                if let Some(ref name) = es_frame.action_name {
                                    let env = self.trek.env_type.unwrap_or(crate::envs::EnvType::Highway);
                                    last_action = env.parse_action_name(name);
                                    last_action_name = Some(name.clone());
                                }
                                if let Some(r) = es_frame.reward {
                                    last_reward = r;
                                }
                                if let Some(v) = es_frame.v {
                                    last_v = Some(v);
                                }
                                if let Some(r) = es_frame.return_value {
                                    last_return = Some(r);
                                }
                                if let Some(t) = es_frame.tendency {
                                    last_tendency = Some(t);
                                }
                                if let Some(a) = es_frame.advantage {
                                    last_adv = Some(a);
                                }
                                if let Some(a) = es_frame.nz_advantage {
                                    last_nadv = Some(a);
                                }
                                let mut vs = es_frame.state.clone();
                                vs.chosen_action = Some(last_action);
                                crate::data::Frame {
                                    observation: [[0u8; 5]; 5],
                                    action: last_action,
                                    action_name: last_action_name.clone(),
                                    reward: last_reward,
                                    done: is_last,
                                    vehicle_state: Some(vs),
                                    crash_reward: es_frame.crash_reward,
                                    v: last_v,
                                    return_value: last_return,
                                    tendency: last_tendency,
                                    advantage: last_adv,
                                    nz_advantage: last_nadv,
                                }
                            })
                            .collect();

                        let ep_meta = self.trek.epochs
                            .get_mut(self.selected_epoch)
                            .and_then(|e| e.episodes.get_mut(self.selected_episode))
                            .expect("episode must exist since we just read es_keys from it");
                        ep_meta.n_frames = frame_count;
                        ep_meta.total_reward = Some(total_reward);
                        let meta = ep_meta.clone();

                        let loaded_episode = crate::data::Episode { meta, frames };
                        self.current_episode = Some(loaded_episode);
                        self.cached_episode_key = key;
                        info!(
                            "Loaded episode {}:{} from {} with {} frames, reward {:.1}",
                            self.selected_epoch, self.selected_episode, format_name,
                            frame_count, total_reward
                        );
                        return;
                    }
                    Err(e) => {
                        if attempt + 1 < max_retries {
                            info!(
                                "Parquet read attempt {}/{} for epoch {} episode {} failed (writer busy?), retrying: {}",
                                attempt + 1, max_retries, self.selected_epoch, self.selected_episode, e
                            );
                            std::thread::sleep(std::time::Duration::from_millis(50));
                        } else {
                            let msg = format!(
                                "Failed to load epoch {} episode {}: {}",
                                self.selected_epoch, self.selected_episode, e
                            );
                            tracing::warn!("{}", msg);
                            self.show_toast(msg, std::time::Duration::from_secs(3));
                            self.current_episode = None;
                            self.cached_episode_key = key;
                            return;
                        }
                    }
                }
            }
        }

        // Fall back to loading from episode JSON files (legacy format)
        if let Some(epoch) = self.trek.epochs.get(self.selected_epoch) {
            if let Some(ep_meta) = epoch.episodes.get(self.selected_episode) {
                match ep_meta.load() {
                    Ok(episode) => {
                        self.current_episode = Some(episode);
                        self.cached_episode_key = key;
                        info!("Loaded episode {}:{}", self.selected_epoch, self.selected_episode);
                    }
                    Err(e) => {
                        let msg = format!(
                            "Failed to load epoch {} episode {}: {}",
                            self.selected_epoch, self.selected_episode, e
                        );
                        tracing::warn!("{}", msg);
                        self.show_toast(msg, std::time::Duration::from_secs(3));
                        self.current_episode = None;
                        self.cached_episode_key = key;
                    }
                }
            } else {
                self.current_episode = None;
            }
        } else {
            self.current_episode = None;
        }
    }

    /// Get the current frame, if available.
    pub fn current_frame(&self) -> Option<&crate::data::Frame> {
        self.current_episode
            .as_ref()
            .and_then(|ep| ep.frames.get(self.frame_index))
    }
}
