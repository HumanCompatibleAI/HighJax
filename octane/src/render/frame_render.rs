//! Frame rendering using the mango module.

use anyhow::Result;
use tracing::warn;

use crate::mango::{render_svg_to_ansi, MangoConfig};
use crate::worlds::SvgEpisode;

/// Environment-specific render configuration.
pub enum SceneRenderConfig {
    Highway(super::highway_svg::HighwayRenderConfig),
}

impl SceneRenderConfig {
    pub fn n_cols(&self) -> u32 {
        match self {
            Self::Highway(c) => c.n_cols,
        }
    }
    pub fn n_rows(&self) -> u32 {
        match self {
            Self::Highway(c) => c.n_rows,
        }
    }

    /// Render SVG for the appropriate environment.
    /// Single dispatch site — all consumers (TUI, draw, animate) call this.
    pub fn render_svg(&self, svg_episode: &SvgEpisode, scene_time: f64) -> Option<String> {
        match self {
            Self::Highway(hrc) => {
                super::highway_svg::render_highway_svg_from_episode(svg_episode, scene_time, hrc)
            }
        }
    }
}

/// Render a frame using the SvgEpisode and mango.
pub fn render_frame_with_svg_episode(
    svg_episode: &SvgEpisode,
    scene_time: f64,
    render_config: &SceneRenderConfig,
    use_sextants: bool,
    use_octants: bool,
) -> Result<String> {
    let svg = render_config.render_svg(svg_episode, scene_time)
        .ok_or_else(|| anyhow::anyhow!("Failed to render SVG at scene_time={}", scene_time))?;

    // Debug: save SVG to file for comparison
    if std::env::var("OCTANE_DEBUG_SVG").is_ok() {
        let _ = std::fs::write("/tmp/octane-debug.svg", &svg);
    }

    let n_cols = render_config.n_cols();
    let n_rows = render_config.n_rows();
    let mango_config = MangoConfig { n_cols, n_rows, use_sextants, use_octants };
    let result = render_svg_to_ansi(&svg, &mango_config)
        .map_err(|e| anyhow::anyhow!("Mango render failed: {}", e))?;

    Ok(result)
}

/// Render using SvgEpisode, falling back to error message if render fails.
pub fn render_frame_with_svg_episode_or_fallback(
    svg_episode: &SvgEpisode,
    scene_time: f64,
    render_config: &SceneRenderConfig,
    use_sextants: bool,
    use_octants: bool,
    fallback: &str,
) -> String {
    match render_frame_with_svg_episode(svg_episode, scene_time, render_config, use_sextants, use_octants) {
        Ok(output) => output,
        Err(e) => {
            warn!("Episode-based rendering failed: {}", e);
            fallback.to_string()
        }
    }
}

/// Configuration for mango rendering.
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// Whether mango rendering is enabled (false = ASCII mode).
    pub enabled: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl RenderConfig {
    /// Create config with rendering disabled.
    pub fn disabled() -> Self {
        Self { enabled: false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{FrameState, VehicleState};
    use crate::worlds::{
        SceneEpisode, SvgConfig, ViewportConfig, ViewportEpisode, DEFAULT_CORN_ASPRO,
    };

    fn make_test_svg_episode() -> SvgEpisode {
        let state = FrameState {
            crashed: false,
            ego: VehicleState {
                x: 50.0,
                y: 4.0,
                heading: 0.0,
                speed: 0.0,
                acceleration: 0.0,
                attention: None,
            },
            npcs: vec![],
            ..Default::default()
        };
        let scene = SceneEpisode::from_frames(vec![state]);
        let viewport = ViewportEpisode::new_default(scene, ViewportConfig::default());
        SvgEpisode::new(viewport, SvgConfig::new(20, 10, DEFAULT_CORN_ASPRO))
    }

    #[test]
    fn test_render_config_default() {
        let config = RenderConfig::default();
        assert!(config.enabled);
    }

    #[test]
    fn test_render_config_disabled() {
        let config = RenderConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_render_frame_with_svg_episode_basic() {
        use crate::render::highway_svg::HighwayRenderConfig;
        let svg_episode = make_test_svg_episode();
        let hrc = HighwayRenderConfig { n_cols: 20, n_rows: 10, ..Default::default() };
        let config = SceneRenderConfig::Highway(hrc);
        let result = render_frame_with_svg_episode(&svg_episode, 0.0, &config, true, true);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("\x1b[")); // Contains ANSI codes
    }

    #[test]
    fn test_render_frame_with_svg_episode_or_fallback() {
        use crate::render::highway_svg::HighwayRenderConfig;
        let svg_episode = make_test_svg_episode();
        let hrc = HighwayRenderConfig { n_cols: 20, n_rows: 10, ..Default::default() };
        let config = SceneRenderConfig::Highway(hrc);
        let output = render_frame_with_svg_episode_or_fallback(
            &svg_episode,
            0.0,
            &config,
            true,
            true,
            "fallback",
        );
        assert!(output.contains("\x1b[")); // Contains ANSI codes, not fallback
    }
}
