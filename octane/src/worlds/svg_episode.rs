//! SVG episode - scene to SVG coordinate transform.
//!
//! SvgEpisode is a thin wrapper providing scene-to-SVG coordinate transformation.
//! It delegates camera tracking to ViewportEpisode.

use crate::data::FrameState;
use crate::worlds::coords::{SceneBounds, ScenePoint, SvgPoint};
use crate::worlds::viewport_episode::ViewportEpisode;
use crate::worlds::DEFAULT_CORN_ASPRO;

/// Configuration for SVG coordinate space.
#[derive(Debug, Clone)]
pub struct SvgConfig {
    /// SVG width in normalized units (always 1.0).
    pub width: f64,
    /// SVG height in normalized units (depends on aspect ratio).
    pub height: f64,
    /// Terminal columns.
    pub n_cols: u32,
    /// Terminal rows.
    pub n_rows: u32,
    /// Corn aspect ratio used for height calculation.
    pub corn_aspro: f64,
}

impl Default for SvgConfig {
    fn default() -> Self {
        Self::new(80, 24, DEFAULT_CORN_ASPRO)
    }
}

impl SvgConfig {
    /// Create SVG config from terminal dimensions.
    ///
    /// # Arguments
    /// * `n_cols` - Terminal width in characters
    /// * `n_rows` - Terminal height in characters
    /// * `corn_aspro` - Character cell aspect ratio (height/width)
    pub fn new(n_cols: u32, n_rows: u32, corn_aspro: f64) -> Self {
        // SVG width is always 1.0
        // SVG height = n_rows / n_cols * corn_aspro (normalized)
        let width = 1.0;
        let height = (n_rows as f64 / n_cols as f64) * corn_aspro;

        Self {
            width,
            height,
            n_cols,
            n_rows,
            corn_aspro,
        }
    }

    /// Get the diagonal length in normalized SVG units.
    pub fn diagonal(&self) -> f64 {
        (self.width * self.width + self.height * self.height).sqrt()
    }

    /// Get dimensions as tuple (width, height).
    pub fn dims(&self) -> (f64, f64) {
        (self.width, self.height)
    }
}

/// Scene-to-SVG coordinate transformer.
///
/// This layer is stateless - it simply transforms coordinates based on
/// the current viewport position (which is stateful in ViewportEpisode).
#[derive(Debug, Clone)]
pub struct SvgEpisode {
    /// Underlying viewport (with camera tracking).
    viewport: ViewportEpisode,
    /// SVG configuration.
    config: SvgConfig,
}

impl SvgEpisode {
    /// Create SvgEpisode from viewport and config.
    pub fn new(viewport: ViewportEpisode, config: SvgConfig) -> Self {
        Self { viewport, config }
    }

    /// Create with default SVG config.
    pub fn new_default(viewport: ViewportEpisode) -> Self {
        Self::new(viewport, SvgConfig::default())
    }

    /// Transform scene point to SVG coordinates at given scene time.
    ///
    /// # Arguments
    /// * `point` - Point in scene coordinates (meters)
    /// * `scene_time` - Time in the episode (seconds)
    ///
    /// # Returns
    /// Point in SVG coordinates, or None if scene_time is invalid.
    pub fn scene_to_svg(&self, point: ScenePoint, scene_time: f64) -> Option<SvgPoint> {
        let view_x = self.viewport.view_x_at(scene_time)?;
        let view_y = self.viewport.view_y_at(scene_time)?;
        let scale = self.viewport.scale();

        // Transform: center scene at viewport, scale, then offset to SVG center
        let svg_x = (point.x - view_x) * scale + self.config.width / 2.0;
        let svg_y = (point.y - view_y) * scale + self.config.height / 2.0;

        Some(SvgPoint::new(svg_x, svg_y))
    }

    /// Transform scene point to SVG at discrete timestep.
    pub fn scene_to_svg_at_index(&self, point: ScenePoint, index: usize) -> Option<SvgPoint> {
        let view_x = self.viewport.view_x(index)?;
        let view_y = self.viewport.view_y(index)?;
        let scale = self.viewport.scale();

        let svg_x = (point.x - view_x) * scale + self.config.width / 2.0;
        let svg_y = (point.y - view_y) * scale + self.config.height / 2.0;

        Some(SvgPoint::new(svg_x, svg_y))
    }

    /// Get SVG bounds (width, height).
    pub fn bounds(&self) -> (f64, f64) {
        self.config.dims()
    }

    /// Access underlying viewport.
    pub fn viewport(&self) -> &ViewportEpisode {
        &self.viewport
    }

    /// Access SVG config.
    pub fn config(&self) -> &SvgConfig {
        &self.config
    }

    /// Time step between frames.
    pub fn seconds_per_timestep(&self) -> f64 {
        self.viewport.scene().seconds_per_timestep()
    }

    /// Get interpolated state at scene time (delegates).
    pub fn state_at(&self, scene_time: f64) -> Option<FrameState> {
        self.viewport.state_at(scene_time)
    }

    /// Get visible scene bounds at scene time.
    pub fn visible_bounds_at(&self, scene_time: f64) -> Option<SceneBounds> {
        self.viewport
            .visible_bounds_at(scene_time, self.config.dims())
    }

    /// Get ego position in SVG coordinates at scene time.
    pub fn ego_svg_at(&self, scene_time: f64) -> Option<SvgPoint> {
        let state = self.viewport.state_at(scene_time)?;
        let ego_scene = ScenePoint::new(state.ego.x, state.ego.y);
        self.scene_to_svg(ego_scene, scene_time)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::VehicleState;
    use crate::worlds::scene_episode::SceneEpisode;
    use crate::worlds::viewport_episode::ViewportConfig;

    fn make_frame(ego_x: f64, ego_y: f64) -> FrameState {
        FrameState {
            crashed: false,
            ego: VehicleState {
                x: ego_x,
                y: ego_y,
                heading: 0.0,
                speed: 0.0,
                acceleration: 0.0,
                attention: None,
            },
            npcs: vec![],
            ..Default::default()
        }
    }

    fn make_scene(n: usize, velocity: f64) -> SceneEpisode {
        let dt = 0.1;
        let frames: Vec<FrameState> = (0..n)
            .map(|i| make_frame(i as f64 * velocity * dt, 6.0))
            .collect();
        SceneEpisode::from_frames(frames)
    }

    fn make_svg_episode() -> SvgEpisode {
        let scene = make_scene(10, 10.0);
        let viewport = ViewportEpisode::new_default(scene, ViewportConfig::with_omega(3.0));
        SvgEpisode::new_default(viewport)
    }

    // =========================================================================
    // SvgConfig tests
    // =========================================================================

    #[test]
    fn test_svg_config_default() {
        let config = SvgConfig::default();
        assert!((config.width - 1.0).abs() < 1e-9);
        assert!(config.height > 0.0);
        assert_eq!(config.n_cols, 80);
        assert_eq!(config.n_rows, 24);
    }

    #[test]
    fn test_svg_config_new() {
        let config = SvgConfig::new(100, 30, 2.0);
        assert!((config.width - 1.0).abs() < 1e-9);
        // height = 30/100 * 2.0 = 0.6
        assert!((config.height - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_svg_config_diagonal() {
        let config = SvgConfig::new(80, 24, DEFAULT_CORN_ASPRO);
        let diag = config.diagonal();
        // sqrt(1^2 + height^2)
        let expected = (1.0_f64 + config.height * config.height).sqrt();
        assert!((diag - expected).abs() < 1e-9);
    }

    #[test]
    fn test_svg_config_different_rows_different_height() {
        let config1 = SvgConfig::new(80, 24, DEFAULT_CORN_ASPRO);
        let config2 = SvgConfig::new(80, 48, DEFAULT_CORN_ASPRO);
        assert!(config2.height > config1.height);
        // Should be exactly double
        assert!((config2.height / config1.height - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_svg_config_different_aspro_different_height() {
        let config1 = SvgConfig::new(80, 24, 1.0);
        let config2 = SvgConfig::new(80, 24, 2.0);
        assert!(config2.height > config1.height);
    }

    // =========================================================================
    // SvgEpisode transform tests
    // =========================================================================

    #[test]
    fn test_scene_to_svg_returns_some() {
        let svg_ep = make_svg_episode();
        let point = ScenePoint::new(0.0, 6.0);
        let result = svg_ep.scene_to_svg(point, 0.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_viewport_center_maps_to_svg_center() {
        let scene = make_scene(10, 10.0);
        let viewport = ViewportEpisode::new(scene, ViewportConfig::with_omega(3.0), 6.0);
        let svg_ep = SvgEpisode::new_default(viewport);

        let scene_time = 0.5;
        let view_center = svg_ep.viewport().view_center_at(scene_time).unwrap();
        let svg_point = svg_ep.scene_to_svg(view_center, scene_time).unwrap();

        // Should map to SVG center (0.5, height/2)
        let (w, h) = svg_ep.bounds();
        assert!(
            (svg_point.x - w / 2.0).abs() < 1e-6,
            "Expected x={}, got {}",
            w / 2.0,
            svg_point.x
        );
        assert!(
            (svg_point.y - h / 2.0).abs() < 1e-6,
            "Expected y={}, got {}",
            h / 2.0,
            svg_point.y
        );
    }

    #[test]
    fn test_point_left_of_viewport_maps_left_of_svg_center() {
        let svg_ep = make_svg_episode();
        let scene_time = 0.5;

        let view_center = svg_ep.viewport().view_center_at(scene_time).unwrap();

        // Point 10m to the left of viewport center
        let left_point = ScenePoint::new(view_center.x - 10.0, view_center.y);
        let svg_point = svg_ep.scene_to_svg(left_point, scene_time).unwrap();

        let (w, _) = svg_ep.bounds();
        assert!(
            svg_point.x < w / 2.0,
            "Expected x < {}, got {}",
            w / 2.0,
            svg_point.x
        );
    }

    #[test]
    fn test_point_right_of_viewport_maps_right_of_svg_center() {
        let svg_ep = make_svg_episode();
        let scene_time = 0.5;

        let view_center = svg_ep.viewport().view_center_at(scene_time).unwrap();

        // Point 10m to the right of viewport center
        let right_point = ScenePoint::new(view_center.x + 10.0, view_center.y);
        let svg_point = svg_ep.scene_to_svg(right_point, scene_time).unwrap();

        let (w, _) = svg_ep.bounds();
        assert!(
            svg_point.x > w / 2.0,
            "Expected x > {}, got {}",
            w / 2.0,
            svg_point.x
        );
    }

    #[test]
    fn test_point_above_viewport_maps_above_svg_center() {
        let svg_ep = make_svg_episode();
        let scene_time = 0.5;

        let view_center = svg_ep.viewport().view_center_at(scene_time).unwrap();

        // Point 5m above viewport center (larger y in scene = larger y in SVG)
        let above_point = ScenePoint::new(view_center.x, view_center.y + 5.0);
        let svg_point = svg_ep.scene_to_svg(above_point, scene_time).unwrap();

        let (_, h) = svg_ep.bounds();
        assert!(
            svg_point.y > h / 2.0,
            "Expected y > {}, got {}",
            h / 2.0,
            svg_point.y
        );
    }

    #[test]
    fn test_transform_is_linear() {
        let svg_ep = make_svg_episode();
        let scene_time = 0.5;

        let view_center = svg_ep.viewport().view_center_at(scene_time).unwrap();

        let p1 = ScenePoint::new(view_center.x - 10.0, view_center.y);
        let p2 = ScenePoint::new(view_center.x + 10.0, view_center.y);
        let mid = ScenePoint::new(view_center.x, view_center.y);

        let svg1 = svg_ep.scene_to_svg(p1, scene_time).unwrap();
        let svg2 = svg_ep.scene_to_svg(p2, scene_time).unwrap();
        let svg_mid = svg_ep.scene_to_svg(mid, scene_time).unwrap();

        // Midpoint in scene should map to midpoint in SVG
        let expected_mid_x = (svg1.x + svg2.x) / 2.0;
        assert!(
            (svg_mid.x - expected_mid_x).abs() < 1e-6,
            "Expected midpoint x={}, got {}",
            expected_mid_x,
            svg_mid.x
        );
    }

    #[test]
    fn test_same_point_different_times_different_svg() {
        let svg_ep = make_svg_episode();

        // Same scene point at different times should give different SVG coords
        // (because viewport moves)
        let point = ScenePoint::new(5.0, 6.0);
        let svg1 = svg_ep.scene_to_svg(point, 0.0).unwrap();
        let svg2 = svg_ep.scene_to_svg(point, 0.5).unwrap();

        // The viewport moves, so the same scene point should appear at different SVG positions
        assert!(
            (svg1.x - svg2.x).abs() > 0.01,
            "Expected different SVG x coords, got {} and {}",
            svg1.x,
            svg2.x
        );
    }

    // =========================================================================
    // SvgEpisode delegation tests
    // =========================================================================

    #[test]
    fn test_bounds_returns_dims() {
        let config = SvgConfig::new(100, 50, 2.0);
        let scene = make_scene(5, 10.0);
        let viewport = ViewportEpisode::new_default(scene, ViewportConfig::default());
        let svg_ep = SvgEpisode::new(viewport, config.clone());

        let (w, h) = svg_ep.bounds();
        assert!((w - config.width).abs() < 1e-9);
        assert!((h - config.height).abs() < 1e-9);
    }

    #[test]
    fn test_state_at_delegates() {
        let svg_ep = make_svg_episode();
        let state = svg_ep.state_at(0.0);
        assert!(state.is_some());
    }

    #[test]
    fn test_visible_bounds_at() {
        let svg_ep = make_svg_episode();
        let bounds = svg_ep.visible_bounds_at(0.5);
        assert!(bounds.is_some());
    }

    #[test]
    fn test_ego_svg_at() {
        let svg_ep = make_svg_episode();
        let ego_svg = svg_ep.ego_svg_at(0.5);
        assert!(ego_svg.is_some());

        // Ego should be somewhere in the SVG bounds (probably near podium position)
        let (w, h) = svg_ep.bounds();
        let pos = ego_svg.unwrap();
        // Allow some margin outside bounds due to podium offset
        assert!(pos.x >= -0.5 && pos.x <= w + 0.5);
        assert!(pos.y >= -0.5 && pos.y <= h + 0.5);
    }

    #[test]
    fn test_scene_to_svg_at_index() {
        let svg_ep = make_svg_episode();
        let point = ScenePoint::new(0.0, 6.0);

        let result_time = svg_ep.scene_to_svg(point, 0.0).unwrap();
        let result_index = svg_ep.scene_to_svg_at_index(point, 0).unwrap();

        // Should be the same
        assert!((result_time.x - result_index.x).abs() < 1e-9);
        assert!((result_time.y - result_index.y).abs() < 1e-9);
    }

    #[test]
    fn test_empty_viewport() {
        let scene = SceneEpisode::from_frames(vec![]);
        let viewport = ViewportEpisode::new_default(scene, ViewportConfig::default());
        let svg_ep = SvgEpisode::new_default(viewport);

        assert!(svg_ep.scene_to_svg(ScenePoint::new(0.0, 0.0), 0.0).is_none());
        assert!(svg_ep.ego_svg_at(0.0).is_none());
    }
}
