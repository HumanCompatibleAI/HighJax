//! Configuration for Octane, loaded from ~/.highjax/config.json.
//!
//! Supports hot-reload via file watcher (notify crate). All fields have defaults
//! matching the original hardcoded values, so a partial config file works fine.

use anyhow::{Context, Result};
use notify::{Config as NotifyConfig, RecommendedWatcher, RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::util::posh_path;

// =============================================================================
// Sub-configs
// =============================================================================

/// Top-level colors configuration, split into scene themes and UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ColorsConfig {
    /// Dark and light scene color themes.
    pub scene_themes: SceneThemesConfig,
    pub ui: UiColorConfig,
}

impl Default for ColorsConfig {
    fn default() -> Self {
        Self {
            scene_themes: SceneThemesConfig::default(),
            ui: UiColorConfig::default(),
        }
    }
}

/// Dark and light scene color themes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SceneThemesConfig {
    pub dark: SceneColorConfig,
    pub light: SceneColorConfig,
}

impl Default for SceneThemesConfig {
    fn default() -> Self {
        Self {
            dark: SceneColorConfig::default(),
            light: SceneColorConfig::default_light(),
        }
    }
}

/// Color configuration for the highway scene (SVG rendering).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SceneColorConfig {
    pub background: String,
    pub terrain: String,
    pub road_surface: String,
    pub road_edge: String,
    pub lane_divider: String,
    pub lane_divider_dashed: bool,
    pub ego: String,
    pub ego_crashed: String,
    pub npc_lightness: f64,
    pub npc_chroma: f64,
    pub window: String,
    pub scale_bar_fg: String,
    pub scale_bar_bg: String,
    pub headlights: bool,
    pub headlight: String,
    pub brakelight: String,
    pub light_blend_mode: String,
    pub action_color: String,
    pub action_chosen_color: String,
    pub hardcoded_arrow_color: String,
}

impl Default for SceneColorConfig {
    fn default() -> Self {
        Self {
            background: "#030100".into(),
            terrain: "#001000".into(),
            road_surface: "#2a2a2a".into(),
            road_edge: "#787878".into(),
            lane_divider: "#646464".into(),
            lane_divider_dashed: false,
            ego: "#b4b4b4".into(),
            ego_crashed: "#ff6464".into(),
            npc_lightness: 0.75,
            npc_chroma: 0.15,
            window: "#1a1a2e".into(),
            scale_bar_fg: "#cccccc".into(),
            scale_bar_bg: "#000000".into(),
            headlights: true,
            headlight: "#ffffcc".into(),
            brakelight: "#ff3333".into(),
            light_blend_mode: "lighten".into(),
            action_color: "#00ffff".into(),
            action_chosen_color: "#ffd700".into(),
            hardcoded_arrow_color: "#ffd700".into(),
        }
    }
}

impl SceneColorConfig {
    /// Default light theme colors.
    pub fn default_light() -> Self {
        Self {
            background: "#f0ece0".into(),
            terrain: "#f0ece0".into(),
            road_surface: "#e3e3e3".into(),
            road_edge: "#404040".into(),
            lane_divider: "#555555".into(),
            lane_divider_dashed: true,
            ego: "#8d8d8d".into(),
            ego_crashed: "#ff3333".into(),
            npc_lightness: 0.82,
            npc_chroma: 0.1,
            window: "#505060".into(),
            scale_bar_fg: "#333333".into(),
            scale_bar_bg: "#f0ece0".into(),
            headlights: false,
            headlight: "#aaaa66".into(),
            brakelight: "#ff3333".into(),
            light_blend_mode: "lighten".into(),
            action_color: "#0088aa".into(),
            action_chosen_color: "#cc8800".into(),
            hardcoded_arrow_color: "#442d00".into(),
        }
    }
}

/// Color configuration for the TUI (ratatui).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UiColorConfig {
    pub focused_border: String,
    pub unfocused_border: String,
    pub selected_bg: String,
    pub selected_fg: String,
    pub mnemonic: String,
    pub title_bar_fg: String,
    pub title_bar_bg: String,
    pub scrollbar_track_multiplier: f64,
    pub scrollbar_thumb_multiplier: f64,
    pub toast_border: String,
    pub toast_prefix: String,
    pub modal_help_border: String,
    pub modal_border: String,
    pub debug_border: String,
    pub modal_bg: String,
    pub seeker_played: String,
    pub seeker_remaining: String,
    pub status_bar_bg: String,
    pub status_bar_key: String,
    pub status_bar_text: String,
}

impl Default for UiColorConfig {
    fn default() -> Self {
        Self {
            focused_border: "#b09000".into(),
            unfocused_border: "#808080".into(),
            selected_bg: "#444444".into(),
            selected_fg: "#ffffff".into(),
            mnemonic: "#ffd700".into(),
            title_bar_fg: "#ffffff".into(),
            title_bar_bg: "#14143c".into(),
            scrollbar_track_multiplier: 0.25,
            scrollbar_thumb_multiplier: 0.5,
            toast_border: "#ffff00".into(),
            toast_prefix: "#888888".into(),
            modal_help_border: "#00ffff".into(),
            modal_border: "#00ff00".into(),
            debug_border: "#ff00ff".into(),
            modal_bg: "#000000".into(),
            seeker_played: "#ffff00".into(),
            seeker_remaining: "#808080".into(),
            status_bar_bg: "#111111".into(),
            status_bar_key: "#fea62b".into(),
            status_bar_text: "#e0e0e0".into(),
        }
    }
}

/// Podium configuration (ego vehicle screen position and camera tracking).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PodiumConfig {
    /// Fraction of visible width to offset ego from center (0.3 = ego at 20% from left).
    pub offset: f64,
    /// Whether to show the podium marker (red vertical line).
    pub show_marker: bool,
    /// Podium marker stroke width.
    pub marker_stroke: f64,
    /// Natural frequency for viewport smoothing. Lower = weaker spring, more drift.
    pub omega: f64,
    /// Damping ratio. 1.0 = critically damped (no oscillation), <1.0 = underdamped.
    pub damping_ratio: f64,
}

impl Default for PodiumConfig {
    fn default() -> Self {
        Self {
            offset: 0.3,
            show_marker: false,
            marker_stroke: 2.0,
            omega: 0.5,
            damping_ratio: 1.0,
        }
    }
}

/// Terrain blob configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TerrainConfig {
    /// Grid size in meters for terrain blob placement.
    pub scale: f64,
    /// Noise threshold for blob placement (0-1, higher = fewer blobs).
    pub density: f64,
    /// Minimum blob radius as fraction of terrain scale.
    pub blob_size_min: f64,
    /// Blob radius variation range as fraction of terrain scale.
    pub blob_size_range: f64,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            scale: 15.0,
            density: 0.55,
            blob_size_min: 0.45,
            blob_size_range: 0.9,
        }
    }
}

/// Road and vehicle configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RoadConfig {
    /// Number of lanes.
    pub n_lanes: usize,
    /// Lane width in meters.
    pub lane_width: f64,
    /// Vehicle length in meters.
    pub vehicle_length: f64,
    /// Vehicle width in meters.
    pub vehicle_width: f64,
    /// Road edge stroke width.
    pub edge_stroke: f64,
    /// Lane divider stroke width.
    pub lane_stroke: f64,
    /// Multiplier for road edge border stroke.
    pub edge_border_multiplier: f64,
}

impl Default for RoadConfig {
    fn default() -> Self {
        Self {
            n_lanes: 4,
            lane_width: 4.0,
            vehicle_length: 5.0,
            vehicle_width: 2.0,
            edge_stroke: 5.0,
            lane_stroke: 2.5,
            edge_border_multiplier: 1.5,
        }
    }
}

/// Scene color theme (dark or light background).
///
/// Each environment interprets this independently. Dark is the default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SceneTheme {
    #[default]
    Dark,
    Light,
}

impl SceneTheme {
    /// Cycle to the next theme.
    pub fn next(self) -> Self {
        match self {
            Self::Dark => Self::Light,
            Self::Light => Self::Dark,
        }
    }

    /// Display label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Dark => "dark",
            Self::Light => "light",
        }
    }

}

/// When to show a display overlay (velocity arrows, action distribution, NPC text, etc.).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum DisplayMode {
    /// Show only when paused.
    OnPause,
    /// Always show.
    Always,
    /// Never show.
    #[default]
    Never,
}

impl DisplayMode {
    /// Cycle to the next mode.
    pub fn next(self) -> Self {
        match self {
            Self::OnPause => Self::Always,
            Self::Always => Self::Never,
            Self::Never => Self::OnPause,
        }
    }

    /// Cycle to the previous mode.
    pub fn prev(self) -> Self {
        match self {
            Self::OnPause => Self::Never,
            Self::Always => Self::OnPause,
            Self::Never => Self::Always,
        }
    }

    /// Display name for UI.
    pub fn label(self) -> &'static str {
        match self {
            Self::OnPause => "on pause",
            Self::Always => "always",
            Self::Never => "never",
        }
    }

    /// Resolve to a boolean given the current playback state.
    pub fn resolve(self, playing: bool) -> bool {
        match self {
            Self::Always => true,
            Self::Never => false,
            Self::OnPause => !playing,
        }
    }
}

/// Velocity arrow visual configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VelocityArrowsConfig {
    /// Arrow opacity (0.0-1.0).
    pub opacity: f64,
    /// Arrow length in meters per m/s of speed.
    pub length_scale: f64,
    /// Minimum speed (m/s) below which arrows are hidden.
    pub min_speed: f64,
    /// Arrowhead half-angle in radians.
    pub head_angle: f64,

    // -- Stroke width (pixels) --
    /// Base stroke width in pixels.
    pub stroke_width: f64,
    /// Additional stroke scaling per m/s: final = stroke_width * (1 + speed * stroke_speed_factor).
    pub stroke_speed_factor: f64,
    /// Minimum stroke width in pixels (floor after scaling).
    pub stroke_min: f64,
    /// Maximum stroke width in pixels (cap after scaling).
    pub stroke_max: f64,

    // -- Head size --
    /// Base arrowhead size as fraction of arrow length.
    pub head_size: f64,
    /// Additional head scaling per m/s: final_frac = head_size * (1 + speed * head_speed_factor).
    pub head_speed_factor: f64,
    /// Minimum arrowhead length in meters (floor).
    pub head_min_meters: f64,
    /// Maximum arrowhead length in meters (cap).
    pub head_max_meters: f64,
}

impl Default for VelocityArrowsConfig {
    fn default() -> Self {
        Self {
            opacity: 0.2,
            length_scale: 0.3,
            min_speed: 0.1,
            head_angle: 0.5,

            stroke_width: 20.0,
            stroke_speed_factor: 0.05,
            stroke_min: 1.0,
            stroke_max: 30.0,

            head_size: 0.25,
            head_speed_factor: 0.03,
            head_min_meters: 0.5,
            head_max_meters: 5.0,
        }
    }
}

/// Attention overlay configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AttentionConfig {
    /// Whether to show attention overlay.
    pub show: bool,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            show: false,
        }
    }
}

/// Action distribution overlay configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ActionDistributionConfig {
    /// Arrow color (hex string).
    pub color: String,
    /// Color for the chosen/selected action (hex string).
    pub chosen_color: String,
    /// Idle circle color (hex string).
    pub circle_color: String,
    /// Maximum arrow length in meters (for probability=1.0).
    pub max_arrow_length: f64,
    /// Arrow stroke width in pixels.
    pub stroke_width: f64,
    /// Minimum circle radius in meters (floor, even at low idle probability).
    pub min_circle_radius: f64,
    /// Maximum circle radius in meters (for probability=1.0).
    pub max_circle_radius: f64,
    /// Minimum opacity for arrows at lowest probability.
    pub min_opacity: f64,
    /// Maximum opacity for arrows at highest probability.
    pub max_opacity: f64,
    /// Minimum opacity for idle circle at lowest probability.
    pub circle_min_opacity: f64,
    /// Maximum opacity for idle circle at highest probability.
    pub circle_max_opacity: f64,
    /// Arrowhead half-angle in radians.
    pub head_angle: f64,
    /// Arrowhead length as fraction of arrow length.
    pub head_size: f64,
    /// Font family for percentage labels.
    pub font_family: String,
    /// Font size in world meters.
    pub font_size_meters: f64,
    /// Text color (hex string). If empty, uses arrow/circle color.
    pub text_color: String,
    /// Text color for the chosen action (hex string). If empty, uses chosen_color.
    pub chosen_text_color: String,
    /// Text background color (hex string). If empty, no background.
    pub text_bg: String,
    /// Distance from arrow tip to text label, in meters.
    pub text_offset_meters: f64,
}

impl Default for ActionDistributionConfig {
    fn default() -> Self {
        Self {
            color: "#00ffff".into(),
            chosen_color: "#ffd700".into(),
            circle_color: "#00ffff".into(),
            max_arrow_length: 10.0,
            stroke_width: 20.0,
            min_circle_radius: 1.0,
            max_circle_radius: 1.0,
            min_opacity: 0.1,
            max_opacity: 0.8,
            circle_min_opacity: 0.0,
            circle_max_opacity: 1.0,
            head_angle: 1.0,
            head_size: 0.5,
            font_family: "'Cascadia Mono', monospace".into(),
            font_size_meters: 2.0,
            text_color: "#cccccc".into(),
            chosen_text_color: "".into(),
            text_bg: "#000000".into(),
            text_offset_meters: 1.0,
        }
    }
}

/// NPC text label configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NpcTextConfig {
    /// Font family for NPC text labels.
    pub font_family: String,
    /// Font size in world meters.
    pub font_size_meters: f64,
    /// Background color (hex string).
    pub bg_color: String,
}

impl Default for NpcTextConfig {
    fn default() -> Self {
        Self {
            font_family: "'DejaVu Sans Mono', monospace".into(),
            font_size_meters: 1.2,
            bg_color: "#000000".into(),
        }
    }
}

/// Rendering parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RenderingConfig {
    /// Scene color theme (dark or light).
    pub theme: SceneTheme,
    /// Playback speed multiplier (1.0 = realtime, 2.0 = 2x speed).
    pub playback_speed: f64,
    /// Frames per second for TUI playback.
    pub fps: u32,
    /// Whether to use sextant characters (higher resolution).
    pub use_sextants: bool,
    /// Whether to use octant characters (highest resolution, requires Unicode 16.0).
    pub use_octants: bool,
    /// When to show velocity arrows on vehicles.
    pub velocity_arrows: DisplayMode,
    /// When to show action distribution on ego vehicle.
    pub action_distribution: DisplayMode,
    /// When to show action distribution percentage text.
    pub action_distribution_text: DisplayMode,
    /// When to show NPC text labels on vehicles.
    pub npc_text: DisplayMode,
    /// Whether to show the scale bar in the bottom-right corner.
    pub show_scala: bool,
    /// Terminal character aspect ratio (height/width). Typical: 1.875 (15px/8px).
    pub corn_aspro: f64,
    /// Raster resolution per character diagonal.
    pub pixels_per_corn_diagonal: f64,
    /// Scene distance (meters) the canvas diagonal spans at zoom=1.0.
    pub unzoomed_canvas_diagonal_in_meters: f64,
    /// Headlight cone base opacity (0.0-1.0).
    pub headlight_opacity: f64,
    /// Brakelight cone base opacity (0.0-1.0).
    pub brakelight_opacity: f64,
    /// Deceleration threshold (m/s², positive) for brakelight activation.
    /// Brakelights turn on when acceleration < -threshold.
    pub brakelight_deceleration_threshold_m_s2: f64,
    /// How far back (in seconds) to look when computing deceleration for brakelights.
    pub brakelight_deceleration_lookback_seconds: f64,
}

impl Default for RenderingConfig {
    fn default() -> Self {
        Self {
            theme: SceneTheme::default(),
            playback_speed: 2.0,
            fps: 30,
            use_sextants: true,
            use_octants: true,
            velocity_arrows: DisplayMode::default(),
            action_distribution: DisplayMode::default(),
            action_distribution_text: DisplayMode::default(),
            npc_text: DisplayMode::default(),
            show_scala: true,
            corn_aspro: 1.875,
            pixels_per_corn_diagonal: 25.3,
            unzoomed_canvas_diagonal_in_meters: 180.0,
            headlight_opacity: 0.12,
            brakelight_opacity: 0.4,
            brakelight_deceleration_threshold_m_s2: 5.0,
            brakelight_deceleration_lookback_seconds: 0.3,
        }
    }
}

/// Hardcoded action arrow configuration (for --hardcoded-action in draw).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HardcodedArrowConfig {
    /// Arrow length in meters.
    pub length: f64,
    /// Shaft width in meters.
    pub shaft_width: f64,
    /// Arrowhead width in meters (full width of the triangle base).
    pub head_width: f64,
    /// Arrowhead length in meters.
    pub head_length: f64,
    /// Arrow opacity (0.0-1.0).
    pub opacity: f64,
}

impl Default for HardcodedArrowConfig {
    fn default() -> Self {
        Self {
            length: 5.0,
            shaft_width: 1.2,
            head_width: 3.0,
            head_length: 2.0,
            opacity: 0.85,
        }
    }
}

// =============================================================================
// Main Config
// =============================================================================

/// Octane-specific configuration (everything under the "octane" key).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct OctaneConfig {
    pub colors: ColorsConfig,
    pub podium: PodiumConfig,
    pub terrain: TerrainConfig,
    pub road: RoadConfig,
    pub rendering: RenderingConfig,
    pub velocity_arrows: VelocityArrowsConfig,
    pub attention: AttentionConfig,
    pub action_distribution: ActionDistributionConfig,
    pub hardcoded_arrow: HardcodedArrowConfig,
    pub npc_text: NpcTextConfig,
}

/// Top-level configuration, loaded from ~/.highjax/config.json.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub octane: OctaneConfig,
}

impl Config {
    /// Config file path: ~/.highjax/config.json
    pub fn path() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        PathBuf::from(home).join(".highjax").join("config.json")
    }

    /// Create config file with defaults if missing, or fill in missing fields.
    ///
    /// If the file doesn't exist, writes full defaults.
    /// If it exists, loads it (which fills defaults via serde), then writes back.
    pub fn fill_defaults() -> Result<()> {
        let config_path = Self::path();

        // Ensure parent directory exists
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Load existing (fills in defaults) or create fresh defaults
        let config = if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            serde_json::from_str::<Config>(&content)
                .context("Failed to parse existing config")?
        } else {
            Config::default()
        };

        // Write back with all fields populated
        let json = serde_json::to_string_pretty(&config)?;
        std::fs::write(&config_path, &json)?;
        println!("Config written to: {}", config_path.display());
        Ok(())
    }

    /// Load config from file, falling back to defaults for missing fields.
    pub fn load() -> Self {
        let config_path = Self::path();
        if !config_path.exists() {
            info!("No config file at {}, using defaults", posh_path(&config_path));
            return Self::default();
        }

        match std::fs::read_to_string(&config_path) {
            Ok(content) => match serde_json::from_str::<Config>(&content) {
                Ok(config) => {
                    info!("Loaded config from {}", posh_path(&config_path));
                    config
                }
                Err(e) => {
                    warn!("Failed to parse config {}: {}", posh_path(&config_path), e);
                    eprintln!("Warning: Failed to parse config file: {}", e);
                    Self::default()
                }
            },
            Err(e) => {
                warn!("Failed to read config {}: {}", posh_path(&config_path), e);
                Self::default()
            }
        }
    }
}

// =============================================================================
// Hot Reload
// =============================================================================

/// Start a file watcher that sends new Config through the channel on changes.
pub fn start_config_watcher() -> Option<Receiver<Config>> {
    let config_path = Config::path();
    if !config_path.exists() {
        debug!("Config file doesn't exist, skipping watcher");
        return None;
    }

    // Follow symlink to get the real path (important for Dropbox-synced configs)
    let watch_path = match std::fs::canonicalize(&config_path) {
        Ok(p) => p,
        Err(e) => {
            warn!("Failed to canonicalize config path: {:?}", e);
            config_path.clone()
        }
    };

    // Watch the parent directory (handles atomic saves: write tmp + rename)
    let watch_dir = match watch_path.parent() {
        Some(dir) => dir.to_path_buf(),
        None => {
            warn!("Config path has no parent directory");
            return None;
        }
    };

    let (tx, rx) = mpsc::channel();

    std::thread::spawn(move || {
        let tx_clone = tx.clone();
        let watch_path_clone = watch_path.clone();

        let mut watcher = match RecommendedWatcher::new(
            move |res: Result<notify::Event, notify::Error>| {
                if let Ok(event) = res {
                    if event.kind.is_modify() || event.kind.is_create() {
                        // Check if the event is for our config file
                        let is_our_file = event.paths.iter().any(|p| {
                            p == &watch_path_clone
                                || p.file_name() == Some(std::ffi::OsStr::new("config.json"))
                        });
                        if !is_our_file {
                            return;
                        }

                        // Small delay to let the file finish writing
                        std::thread::sleep(Duration::from_millis(100));

                        let new_config = Config::load();
                        info!("Config reloaded");
                        let _ = tx_clone.send(new_config);
                    }
                }
            },
            NotifyConfig::default(),
        ) {
            Ok(w) => w,
            Err(e) => {
                warn!("Failed to create config watcher: {:?}", e);
                return;
            }
        };

        if let Err(e) = watcher.watch(&watch_dir, RecursiveMode::NonRecursive) {
            warn!("Failed to watch config directory: {:?}", e);
            return;
        }

        info!("Config watcher started for {}", posh_path(&watch_dir));

        // Keep the thread alive (watcher drops when thread exits)
        loop {
            std::thread::sleep(Duration::from_secs(1));
        }
    });

    Some(rx)
}

/// Check for config updates. Returns Some(new_config) if the file changed.
pub fn check_config_updates(rx: &Option<Receiver<Config>>) -> Option<Config> {
    let rx = rx.as_ref()?;
    // Drain all pending updates, keep only the latest
    let mut latest = None;
    loop {
        match rx.try_recv() {
            Ok(config) => latest = Some(config),
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                warn!("Config watcher disconnected");
                break;
            }
        }
    }
    latest
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.octane.podium.omega, 0.5);
        assert_eq!(config.octane.road.n_lanes, 4);
        assert_eq!(config.octane.rendering.fps, 30);
        assert_eq!(config.octane.colors.scene_themes.dark.background, "#030100");
        assert_eq!(config.octane.colors.scene_themes.dark.npc_lightness, 0.75);
        assert_eq!(config.octane.colors.scene_themes.dark.npc_chroma, 0.15);
        assert_eq!(config.octane.colors.ui.mnemonic, "#ffd700");
    }

    #[test]
    fn test_config_parse_empty() {
        let config: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(config.octane.podium.omega, 0.5);
        assert_eq!(config.octane.colors.scene_themes.dark.background, "#030100");
        assert_eq!(config.octane.colors.ui.focused_border, "#b09000");
    }

    #[test]
    fn test_config_parse_partial() {
        let json = r#"{ "octane": { "podium": { "omega": 0.5 } } }"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.octane.podium.omega, 0.5);
        // Other fields should be defaults
        assert_eq!(config.octane.road.n_lanes, 4);
        assert_eq!(config.octane.colors.scene_themes.dark.background, "#030100");
    }

    #[test]
    fn test_config_parse_partial_colors() {
        let json = r##"{ "octane": { "colors": { "scene_themes": { "dark": { "background": "#ff0000" } } } } }"##;
        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.octane.colors.scene_themes.dark.background, "#ff0000");
        // Other color fields should be defaults
        assert_eq!(config.octane.colors.scene_themes.dark.terrain, "#001000");
        assert_eq!(config.octane.colors.scene_themes.dark.ego, "#b4b4b4");
        assert_eq!(config.octane.colors.ui.mnemonic, "#ffd700");
    }

    #[test]
    fn test_config_roundtrip() {
        let config = Config::default();
        let json = serde_json::to_string_pretty(&config).unwrap();
        let parsed: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.octane.podium.omega, config.octane.podium.omega);
        assert_eq!(parsed.octane.colors.scene_themes.dark.background, config.octane.colors.scene_themes.dark.background);
        assert_eq!(parsed.octane.colors.ui.status_bar_bg, config.octane.colors.ui.status_bar_bg);
    }

    #[test]
    fn test_config_path() {
        let path = Config::path();
        assert!(path.ends_with("config.json"));
    }

    #[test]
    fn test_scene_theme_default() {
        assert_eq!(SceneTheme::default(), SceneTheme::Dark);
        let config = Config::default();
        assert_eq!(config.octane.rendering.theme, SceneTheme::Dark);
    }

    #[test]
    fn test_scene_theme_serde() {
        let json = r#"{ "octane": { "rendering": { "theme": "light" } } }"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.octane.rendering.theme, SceneTheme::Light);
    }

    #[test]
    fn test_scene_theme_roundtrip() {
        let mut config = Config::default();
        config.octane.rendering.theme = SceneTheme::Light;
        let json = serde_json::to_string(&config).unwrap();
        let parsed: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.octane.rendering.theme, SceneTheme::Light);
    }

    #[test]
    fn test_scene_theme_palettes_differ() {
        let dark = SceneColorConfig::default();
        let light = SceneColorConfig::default_light();
        assert_ne!(dark.road_surface, light.road_surface);
        assert_ne!(dark.ego, light.ego);
        assert!(light.npc_lightness > dark.npc_lightness);
    }

    #[test]
    fn test_scene_theme_cycling() {
        assert_eq!(SceneTheme::Dark.next(), SceneTheme::Light);
        assert_eq!(SceneTheme::Light.next(), SceneTheme::Dark);
    }
}
