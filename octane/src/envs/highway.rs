//! Highway (highway) environment adapter.
//!
//! Viewport construction and render config for the Highway highway env.
//! The Highway SVG renderer lives in render/highway_svg.rs due to its size
//! and tight coupling with the rendering subsystem (car templates, brakelights,
//! headlights, terrain, etc.).

use crate::config::{self, Config};
use crate::data::trek::Trek;
use crate::render::highway_svg::HighwayRenderConfig;
use crate::render::SceneRenderConfig;
use crate::worlds::{SceneEpisode, ViewportConfig, ViewportEpisode};

// ── Prefs ────────────────────────────────────────────────────────────────────

/// Per-env preferences for Highway: runtime toggles that affect rendering.
/// Initialized from config defaults, overridable via CLI and in-app keybindings.
#[derive(Debug, Clone)]
pub struct HighwayPrefs {
    pub show_podium_marker: bool,
    pub show_scala: bool,
    pub show_attention: bool,
    pub debug_eye: bool,
    pub light_blend_mode: String,
    pub velocity_arrows: config::DisplayMode,
    pub action_distribution: config::DisplayMode,
    pub action_distribution_text: config::DisplayMode,
    pub npc_text: config::DisplayMode,
}

impl HighwayPrefs {
    /// Initialize from config defaults.
    pub fn from_config(config: &Config, theme: config::SceneTheme) -> Self {
        let scene = match theme {
            config::SceneTheme::Dark => &config.octane.colors.scene_themes.dark,
            config::SceneTheme::Light => &config.octane.colors.scene_themes.light,
        };
        Self {
            show_podium_marker: config.octane.podium.show_marker,
            show_scala: config.octane.rendering.show_scala,
            show_attention: config.octane.attention.show,
            debug_eye: false,
            light_blend_mode: scene.light_blend_mode.clone(),
            velocity_arrows: config.octane.rendering.velocity_arrows,
            action_distribution: config.octane.rendering.action_distribution,
            action_distribution_text: config.octane.rendering.action_distribution_text,
            npc_text: config.octane.rendering.npc_text,
        }
    }

    /// Apply CLI overrides from comma-separated key=value pairs.
    /// Returns an error message for the first unrecognized key, if any.
    pub fn apply_overrides(&mut self, raw: &str) -> Result<(), String> {
        for pair in raw.split(',') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }
            let (key, val) = pair.split_once('=')
                .ok_or_else(|| format!("invalid pref '{}' (expected key=value)", pair))?;
            let key = key.trim();
            let val = val.trim();
            match key {
                "podium_marker" => self.show_podium_marker = parse_bool(val, key)?,
                "scala" => self.show_scala = parse_bool(val, key)?,
                "attention" => self.show_attention = parse_bool(val, key)?,
                "debug_eye" => self.debug_eye = parse_bool(val, key)?,
                "light_blend" => self.light_blend_mode = val.to_string(),
                "velocity_arrows" => self.velocity_arrows = parse_display_mode(val, key)?,
                "action_distribution" => self.action_distribution = parse_display_mode(val, key)?,
                "action_distribution_text" => self.action_distribution_text = parse_display_mode(val, key)?,
                "npc_text" => self.npc_text = parse_display_mode(val, key)?,
                _ => return Err(format!("unknown Highway pref '{}'", key)),
            }
        }
        Ok(())
    }

    /// Update prefs from a hot-reloaded config (preserves user-toggled values
    /// by only updating fields that match config semantics).
    pub fn apply_config(&mut self, config: &Config, theme: config::SceneTheme) {
        let scene = match theme {
            config::SceneTheme::Dark => &config.octane.colors.scene_themes.dark,
            config::SceneTheme::Light => &config.octane.colors.scene_themes.light,
        };
        self.show_podium_marker = config.octane.podium.show_marker;
        self.show_scala = config.octane.rendering.show_scala;
        self.show_attention = config.octane.attention.show;
        self.light_blend_mode = scene.light_blend_mode.clone();
        self.velocity_arrows = config.octane.rendering.velocity_arrows;
        self.action_distribution = config.octane.rendering.action_distribution;
        self.action_distribution_text = config.octane.rendering.action_distribution_text;
        self.npc_text = config.octane.rendering.npc_text;
    }
}

fn parse_bool(val: &str, key: &str) -> Result<bool, String> {
    match val {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => Err(format!("invalid bool for '{}': '{}' (expected true/false)", key, val)),
    }
}

fn parse_display_mode(val: &str, key: &str) -> Result<config::DisplayMode, String> {
    match val {
        "on-pause" | "on_pause" | "onpause" => Ok(config::DisplayMode::OnPause),
        "always" => Ok(config::DisplayMode::Always),
        "never" => Ok(config::DisplayMode::Never),
        _ => Err(format!(
            "invalid display mode for '{}': '{}' (expected on-pause/always/never)",
            key, val,
        )),
    }
}

// ── Parquet columns ──────────────────────────────────────────────────────────

/// NPC state columns extracted from a parquet batch.
pub(crate) struct NpcColumns {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub heading: Vec<f64>,
    pub speed: Vec<f64>,
    pub attention: Option<Vec<f64>>,
}

/// Highway-specific columns extracted from a parquet batch.
pub(crate) struct HighwayParquetColumns {
    pub ego_x: Vec<f64>,
    pub ego_y: Vec<f64>,
    pub ego_heading: Vec<f64>,
    pub ego_speed: Vec<f64>,
    pub npcs: Vec<NpcColumns>,
}

impl HighwayParquetColumns {
    /// Count NPC columns from the parquet schema.
    pub fn count_npcs(columns: &[String]) -> usize {
        let mut n = 0;
        while columns.contains(&format!("state.npc{}_x", n)) {
            n += 1;
        }
        n
    }

    /// Extract Highway-specific columns from a record batch.
    pub fn extract(
        batch: &arrow::record_batch::RecordBatch,
        columns: &[String],
    ) -> Self {
        use crate::data::es_parquet::{get_f64, get_f64_opt};

        let n_npcs = Self::count_npcs(columns);
        let npcs: Vec<NpcColumns> = (0..n_npcs).map(|i| {
            NpcColumns {
                x: get_f64(batch, &format!("state.npc{}_x", i)),
                y: get_f64(batch, &format!("state.npc{}_y", i)),
                heading: get_f64(batch, &format!("state.npc{}_heading", i)),
                speed: get_f64(batch, &format!("state.npc{}_speed", i)),
                attention: get_f64_opt(batch, &format!("state.npc{}_attention", i)),
            }
        }).collect();

        Self {
            ego_x: get_f64(batch, "state.ego_x"),
            ego_y: get_f64(batch, "state.ego_y"),
            ego_heading: get_f64(batch, "state.ego_heading"),
            ego_speed: get_f64(batch, "state.ego_speed"),
            npcs,
        }
    }

    /// Build ego and NPC vehicle states from row data.
    pub fn build_frame_state(
        &self,
        row: usize,
    ) -> (crate::data::jsonla::VehicleState, Vec<crate::data::jsonla::VehicleState>) {
        use crate::data::jsonla::VehicleState;

        let ego = VehicleState {
            x: self.ego_x[row],
            y: self.ego_y[row],
            heading: self.ego_heading[row],
            speed: self.ego_speed[row],
            acceleration: 0.0,
            attention: None,
        };
        let npcs: Vec<VehicleState> = self.npcs.iter().map(|npc| {
            VehicleState {
                x: npc.x[row],
                y: npc.y[row],
                heading: npc.heading[row],
                speed: npc.speed[row],
                acceleration: 0.0,
                attention: npc.attention.as_ref().map(|a| a[row]),
            }
        }).collect();
        (ego, npcs)
    }
}

// ── Detection ────────────────────────────────────────────────────────────────

/// Detect Highway from meta.yaml commands.
pub fn detect_from_meta(doc: &serde_yaml::Value) -> bool {
    doc.get("commands")
        .and_then(|c| c.as_mapping())
        .map(|commands| {
            commands.keys().any(|k| {
                k.as_str()
                    .map(|s| s.contains("highway") || s.contains("highjax"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

// ── Viewport ─────────────────────────────────────────────────────────────────

/// Build spring-tracked viewport for Highway highway.
pub fn build_viewport(
    scene: SceneEpisode,
    vp_config: ViewportConfig,
    trek: &Trek,
    app_config: &Config,
) -> ViewportEpisode {
    let n_lanes = trek.n_lanes.unwrap_or(app_config.octane.road.n_lanes);
    let lane_width = app_config.octane.road.lane_width;
    let road_center_y = (n_lanes - 1) as f64 * lane_width / 2.0;
    ViewportEpisode::new(scene, vp_config, road_center_y)
}

// ── Render config ────────────────────────────────────────────────────────────

/// Build a full SceneRenderConfig for Highway.
/// Applies config defaults, trek overrides, prefs toggles, and app state.
pub fn build_render_config(
    config: &Config,
    trek: &Trek,
    n_cols: u32,
    n_rows: u32,
    prefs: &HighwayPrefs,
    is_paused: bool,
    show_debug: bool,
    theme: config::SceneTheme,
) -> SceneRenderConfig {
    let mut hrc = HighwayRenderConfig::from_config(config, theme);
    hrc.n_cols = n_cols;
    hrc.n_rows = n_rows;
    // Trek overrides
    if let Some(n_lanes) = trek.n_lanes {
        hrc.n_lanes = n_lanes;
    }
    if let Some((min, max)) = trek.ego_speed_range {
        hrc.ego_speed_range = (min, max);
    }
    // Prefs toggles
    hrc.show_podium_marker = prefs.show_podium_marker;
    hrc.show_scala = prefs.show_scala;
    hrc.show_attention = prefs.show_attention;
    hrc.debug_eye = prefs.debug_eye;
    hrc.light_blend_mode = prefs.light_blend_mode.clone();
    hrc.show_velocity_arrows = prefs.velocity_arrows.resolve(!is_paused);
    hrc.show_action_distribution = prefs.action_distribution.resolve(!is_paused);
    hrc.show_action_distribution_text = prefs.action_distribution_text.resolve(!is_paused);
    hrc.show_npc_text = prefs.npc_text.resolve(!is_paused);
    hrc.is_paused = is_paused;
    // Debug overrides
    if show_debug {
        hrc.show_action_distribution = true;
        hrc.show_action_distribution_text = true;
        hrc.show_npc_text = true;
    }
    SceneRenderConfig::Highway(hrc)
}

/// Build a base SceneRenderConfig for Highway (without prefs/app state).
/// Used by draw and animate commands which don't have a TUI prefs context.
pub fn scene_render_config(config: &Config, trek: &Trek, n_cols: u32, n_rows: u32, theme: config::SceneTheme) -> SceneRenderConfig {
    let prefs = HighwayPrefs::from_config(config, theme);
    build_render_config(config, trek, n_cols, n_rows, &prefs, true, false, theme)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scala_prefs_override() {
        let config = Config::default();
        let mut prefs = HighwayPrefs::from_config(&config, config::SceneTheme::Dark);
        assert!(prefs.show_scala);

        prefs.apply_overrides("scala=false").unwrap();
        assert!(!prefs.show_scala);

        prefs.apply_overrides("scala=true").unwrap();
        assert!(prefs.show_scala);
    }
}
