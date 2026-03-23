//! Viewport episode - precomputed camera trajectory.
//!
//! ViewportEpisode is the only stateful layer in the coordinate system architecture.
//! It precomputes the entire camera trajectory on construction using
//! critically damped spring dynamics.
//! See docs/Octane/ for coordinate system design details.

use crate::data::FrameState;
use crate::worlds::coords::{SceneBounds, ScenePoint};
use crate::worlds::scene_episode::SceneEpisode;
use crate::worlds::{DEFAULT_CORN_ASPRO, UNZOOMED_CANVAS_DIAGONAL_IN_METERS};

/// Default natural frequency for viewport smoothing.
/// Lower values = weaker spring, more drift, slower return to podium.
pub const DEFAULT_OMEGA: f64 = 0.5;

/// Configuration for viewport behavior.
#[derive(Debug, Clone)]
pub struct ViewportConfig {
    /// Zoom multiplier (1.0 = default, 2.0 = 2x magnification).
    pub zoom: f64,
    /// Spring constant for camera smoothing (lower = more drift).
    pub omega: f64,
    /// Corn aspect ratio for scale calculation.
    pub corn_aspro: f64,
    /// Fraction of visible width for podium offset (default 0.3).
    pub podium_fraction: f64,
    /// Spring damping ratio (default 1.0 = critically damped).
    pub damping_ratio: f64,
}

impl Default for ViewportConfig {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            omega: DEFAULT_OMEGA,
            corn_aspro: DEFAULT_CORN_ASPRO,
            podium_fraction: 0.3,
            damping_ratio: 1.0,
        }
    }
}

impl ViewportConfig {
    /// Create config with specified zoom.
    pub fn with_zoom(zoom: f64) -> Self {
        Self {
            zoom,
            ..Default::default()
        }
    }

    /// Create config with specified omega.
    pub fn with_omega(omega: f64) -> Self {
        Self {
            omega,
            ..Default::default()
        }
    }

    /// Builder: set zoom.
    pub fn zoom(mut self, zoom: f64) -> Self {
        self.zoom = zoom;
        self
    }

    /// Builder: set omega.
    pub fn omega(mut self, omega: f64) -> Self {
        self.omega = omega;
        self
    }
}

/// Camera tracking system with precomputed trajectory.
///
/// This is the only layer with history-dependence. On construction, it
/// processes the entire SceneEpisode and precomputes the viewport x-position
/// for every timestep using critically damped spring dynamics.
#[derive(Debug, Clone)]
pub struct ViewportEpisode {
    /// Underlying scene data.
    scene: SceneEpisode,
    /// Configuration.
    config: ViewportConfig,
    /// Precomputed viewport x positions, one per timestep.
    trajectory_x: Vec<f64>,
    /// Precomputed viewport y positions, one per timestep.
    trajectory_y: Vec<f64>,
    /// Precomputed scale factor (SVG units per meter).
    scale: f64,
    /// Visible width in meters at current zoom.
    visible_width: f64,
    /// Podium offset (where ego should appear relative to viewport center).
    podium_offset: f64,
}

impl ViewportEpisode {
    /// Construct and precompute entire trajectory.
    ///
    /// # Arguments
    /// * `scene` - The scene episode to track
    /// * `config` - Viewport configuration
    /// * `road_center_y` - Y coordinate of road center (typically `(n_lanes - 1) * lane_width / 2`)
    pub fn new(scene: SceneEpisode, config: ViewportConfig, road_center_y: f64) -> Self {
        let scale = compute_scale(config.zoom, config.corn_aspro);
        let visible_width = 1.0 / scale; // SVG width is 1.0
        let podium_offset = -visible_width * config.podium_fraction;

        let trajectory_x = precompute_trajectory(&scene, &config, podium_offset);

        // Constant y trajectory (highway: road doesn't move)
        let n = trajectory_x.len();
        let trajectory_y = vec![road_center_y; n];

        Self {
            scene,
            config,
            trajectory_x,
            trajectory_y,
            scale,
            visible_width,
            podium_offset,
        }
    }

    /// Construct a static viewport with fixed center position (no tracking).
    /// Used for envs where the camera shouldn't follow the agent.
    pub fn new_static(scene: SceneEpisode, config: ViewportConfig, center: ScenePoint) -> Self {
        let scale = compute_scale(config.zoom, config.corn_aspro);
        let visible_width = 1.0 / scale;

        let n = scene.n_frames();
        let trajectory_x = vec![center.x; n];
        let trajectory_y = vec![center.y; n];

        Self {
            scene,
            config,
            trajectory_x,
            trajectory_y,
            scale,
            visible_width,
            podium_offset: 0.0,
        }
    }

    /// Construct with ego always at horizontal center, fixed vertical position.
    /// Used for envs that need direct ego tracking without spring dynamics.
    pub fn new_centered(scene: SceneEpisode, config: ViewportConfig, center_y: f64) -> Self {
        let scale = compute_scale(config.zoom, config.corn_aspro);
        let visible_width = 1.0 / scale;

        // Direct tracking: viewport x = ego x (no spring, no podium offset)
        let trajectory_x: Vec<f64> = scene.ego_xs();
        let n = trajectory_x.len();
        let trajectory_y = vec![center_y; n];

        Self {
            scene,
            config,
            trajectory_x,
            trajectory_y,
            scale,
            visible_width,
            podium_offset: 0.0,
        }
    }

    /// Construct with default road center (4-lane road with 4m lanes).
    pub fn new_default(scene: SceneEpisode, config: ViewportConfig) -> Self {
        // Default: 4 lanes, 4m width, center at (4-1) * 4 / 2 = 6m
        let road_center_y = 6.0;
        Self::new(scene, config, road_center_y)
    }

    /// Get viewport x at discrete timestep.
    pub fn view_x(&self, index: usize) -> Option<f64> {
        self.trajectory_x.get(index).copied()
    }

    /// Get interpolated viewport x at scene time.
    pub fn view_x_at(&self, scene_time: f64) -> Option<f64> {
        if self.trajectory_x.is_empty() {
            return None;
        }

        if self.trajectory_x.len() == 1 {
            return Some(self.trajectory_x[0]);
        }

        let dt = self.scene.seconds_per_timestep();
        let duration = self.scene.duration();
        let t = scene_time.clamp(0.0, duration);

        let index_f = t / dt;
        let index_a = (index_f.floor() as usize).min(self.trajectory_x.len() - 1);
        let index_b = (index_a + 1).min(self.trajectory_x.len() - 1);
        let lerp_t = index_f.fract();

        if lerp_t < 1e-9 || index_a == index_b {
            return Some(self.trajectory_x[index_a]);
        }

        let x_a = self.trajectory_x[index_a];
        let x_b = self.trajectory_x[index_b];
        Some(x_a + (x_b - x_a) * lerp_t)
    }

    /// Get viewport y at discrete timestep.
    pub fn view_y(&self, index: usize) -> Option<f64> {
        self.trajectory_y.get(index).copied()
    }

    /// Get interpolated viewport y at scene time.
    pub fn view_y_at(&self, scene_time: f64) -> Option<f64> {
        if self.trajectory_y.is_empty() {
            return None;
        }

        if self.trajectory_y.len() == 1 {
            return Some(self.trajectory_y[0]);
        }

        let dt = self.scene.seconds_per_timestep();
        let duration = self.scene.duration();
        let t = scene_time.clamp(0.0, duration);

        let index_f = t / dt;
        let index_a = (index_f.floor() as usize).min(self.trajectory_y.len() - 1);
        let index_b = (index_a + 1).min(self.trajectory_y.len() - 1);
        let lerp_t = index_f.fract();

        if lerp_t < 1e-9 || index_a == index_b {
            return Some(self.trajectory_y[index_a]);
        }

        let y_a = self.trajectory_y[index_a];
        let y_b = self.trajectory_y[index_b];
        Some(y_a + (y_b - y_a) * lerp_t)
    }

    /// Get viewport center at scene time.
    pub fn view_center_at(&self, scene_time: f64) -> Option<ScenePoint> {
        let x = self.view_x_at(scene_time)?;
        let y = self.view_y_at(scene_time)?;
        Some(ScenePoint::new(x, y))
    }

    /// Get scale factor (SVG units per meter).
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Get visible width in meters.
    pub fn visible_width_meters(&self) -> f64 {
        self.visible_width
    }

    /// Get visible height in meters for given SVG height.
    pub fn visible_height_meters(&self, svg_height: f64) -> f64 {
        svg_height / self.scale
    }

    /// Get visible scene bounds at scene time.
    pub fn visible_bounds_at(&self, scene_time: f64, svg_dims: (f64, f64)) -> Option<SceneBounds> {
        let view_x = self.view_x_at(scene_time)?;
        let view_y = self.view_y_at(scene_time)?;
        let half_w = self.visible_width / 2.0;
        let half_h = self.visible_height_meters(svg_dims.1) / 2.0;

        Some(SceneBounds::new(
            view_x - half_w,
            view_x + half_w,
            view_y - half_h,
            view_y + half_h,
        ))
    }

    /// Access underlying scene.
    pub fn scene(&self) -> &SceneEpisode {
        &self.scene
    }

    /// Get interpolated state at scene time (delegates to scene).
    pub fn state_at(&self, scene_time: f64) -> Option<FrameState> {
        self.scene.state_at(scene_time)
    }

    /// Access configuration.
    pub fn config(&self) -> &ViewportConfig {
        &self.config
    }

    /// Get podium offset (negative = ego appears right of center).
    pub fn podium_offset(&self) -> f64 {
        self.podium_offset
    }

    /// Number of frames in the trajectory.
    pub fn n_frames(&self) -> usize {
        self.trajectory_x.len()
    }
}

/// Compute scale factor from zoom and corn_aspro.
fn compute_scale(zoom: f64, corn_aspro: f64) -> f64 {
    let canvas_diagonal = (1.0_f64 + corn_aspro * corn_aspro).sqrt();
    let scene_diagonal = UNZOOMED_CANVAS_DIAGONAL_IN_METERS / zoom;
    canvas_diagonal / scene_diagonal
}

/// Compute podium offset from scale.
#[allow(dead_code)]
fn compute_podium_offset(scale: f64) -> f64 {
    let visible_width = 1.0 / scale;
    -visible_width * 0.3
}

/// Precompute viewport trajectory using spring dynamics.
fn precompute_trajectory(
    scene: &SceneEpisode,
    config: &ViewportConfig,
    podium_offset: f64,
) -> Vec<f64> {
    if scene.is_empty() {
        return Vec::new();
    }

    let dt = scene.seconds_per_timestep();
    let ego_xs = scene.ego_xs();
    let ego_vs = scene.ego_velocities();
    let n = ego_xs.len();

    // Initialize viewport so ego is at podium
    let mut vp_x = ego_xs[0] - podium_offset;
    let mut vp_v = ego_vs[0];

    let mut trajectory = Vec::with_capacity(n);
    trajectory.push(vp_x);

    let omega = config.omega;
    let omega_sq = omega * omega;

    for i in 1..n {
        // Move viewport (using velocity from last frame)
        vp_x += vp_v * dt;

        // Spring dynamics
        // error = ego_x - vp_x - podium_offset
        // Note: podium_offset is negative, so this means ego should be at vp_x + |podium_offset|
        let error = ego_xs[i] - vp_x - podium_offset;
        let v_error = ego_vs[i] - vp_v;

        // Critically damped spring: update velocity
        vp_v += (omega_sq * error + 2.0 * config.damping_ratio * omega * v_error) * dt;

        trajectory.push(vp_x);
    }

    trajectory
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::VehicleState;

    fn make_frame(ego_x: f64) -> FrameState {
        FrameState {
            crashed: false,
            ego: VehicleState {
                x: ego_x,
                y: 4.0,
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
            .map(|i| make_frame(i as f64 * velocity * dt))
            .collect();
        SceneEpisode::from_frames(frames)
    }

    #[test]
    fn test_construction_doesnt_panic() {
        let scene = make_scene(10, 10.0);
        let config = ViewportConfig::default();
        let _vp = ViewportEpisode::new_default(scene, config);
    }

    #[test]
    fn test_trajectory_has_correct_length() {
        let scene = make_scene(10, 10.0);
        let config = ViewportConfig::default();
        let vp = ViewportEpisode::new_default(scene, config);
        assert_eq!(vp.n_frames(), 10);
    }

    #[test]
    fn test_view_y_is_road_center() {
        let scene = make_scene(5, 10.0);
        let config = ViewportConfig::default();
        let vp = ViewportEpisode::new(scene, config, 8.0);
        assert!((vp.view_y(0).unwrap() - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_scale_is_positive() {
        let scene = make_scene(5, 10.0);
        let config = ViewportConfig::default();
        let vp = ViewportEpisode::new_default(scene, config);
        assert!(vp.scale() > 0.0);
    }

    #[test]
    fn test_scale_increases_with_zoom() {
        let scene1 = make_scene(5, 10.0);
        let scene2 = make_scene(5, 10.0);
        let vp1 = ViewportEpisode::new_default(scene1, ViewportConfig::with_zoom(1.0));
        let vp2 = ViewportEpisode::new_default(scene2, ViewportConfig::with_zoom(2.0));
        assert!(vp2.scale() > vp1.scale());
    }

    #[test]
    fn test_construction_with_default_config() {
        let scene = make_scene(5, 10.0);
        let vp = ViewportEpisode::new_default(scene, ViewportConfig::default());
        assert_eq!(vp.n_frames(), 5);
    }

    #[test]
    fn test_construction_with_custom_zoom() {
        let scene = make_scene(5, 10.0);
        let config = ViewportConfig::with_zoom(2.0);
        let vp = ViewportEpisode::new_default(scene, config);
        assert!((vp.config().zoom - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_construction_with_custom_omega() {
        let scene = make_scene(5, 10.0);
        let config = ViewportConfig::with_omega(0.5);
        let vp = ViewportEpisode::new_default(scene, config);
        assert!((vp.config().omega - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_view_x_returns_value() {
        let scene = make_scene(5, 10.0);
        let vp = ViewportEpisode::new_default(scene, ViewportConfig::default());
        assert!(vp.view_x(0).is_some());
        assert!(vp.view_x(4).is_some());
        assert!(vp.view_x(10).is_none());
    }

    #[test]
    fn test_view_x_at_zero_equals_view_x_0() {
        let scene = make_scene(5, 10.0);
        let vp = ViewportEpisode::new_default(scene, ViewportConfig::default());
        let x_discrete = vp.view_x(0).unwrap();
        let x_interp = vp.view_x_at(0.0).unwrap();
        assert!((x_discrete - x_interp).abs() < 1e-9);
    }

    #[test]
    fn test_view_x_at_interpolates() {
        let scene = make_scene(5, 10.0);
        let vp = ViewportEpisode::new_default(scene, ViewportConfig::default());
        let x0 = vp.view_x(0).unwrap();
        let x1 = vp.view_x(1).unwrap();
        let x_mid = vp.view_x_at(0.05).unwrap(); // dt=0.1, so t=0.05 is midpoint
        // Should be between x0 and x1
        assert!(
            (x_mid > x0.min(x1) - 1e-9) && (x_mid < x0.max(x1) + 1e-9),
            "x_mid={} should be between x0={} and x1={}",
            x_mid, x0, x1
        );
    }

    #[test]
    fn test_view_x_at_clamps() {
        let scene = make_scene(5, 10.0);
        let vp = ViewportEpisode::new_default(scene, ViewportConfig::default());
        let x_neg = vp.view_x_at(-1.0).unwrap();
        let x_0 = vp.view_x(0).unwrap();
        assert!((x_neg - x_0).abs() < 1e-9);

        let x_big = vp.view_x_at(100.0).unwrap();
        let x_last = vp.view_x(4).unwrap();
        assert!((x_big - x_last).abs() < 1e-9);
    }

    #[test]
    fn test_visible_width_decreases_with_zoom() {
        let scene1 = make_scene(5, 10.0);
        let scene2 = make_scene(5, 10.0);
        let vp1 = ViewportEpisode::new_default(scene1, ViewportConfig::with_zoom(1.0));
        let vp2 = ViewportEpisode::new_default(scene2, ViewportConfig::with_zoom(2.0));
        assert!(vp2.visible_width_meters() < vp1.visible_width_meters());
        // Should be roughly half
        let ratio = vp2.visible_width_meters() / vp1.visible_width_meters();
        assert!((ratio - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_visible_bounds_contains_ego() {
        let scene = make_scene(10, 10.0);
        let vp = ViewportEpisode::new_default(scene.clone(), ViewportConfig::with_omega(3.0));

        // After settling, ego should be within visible bounds
        let scene_time = 0.5; // After some settling
        let bounds = vp.visible_bounds_at(scene_time, (1.0, 0.5)).unwrap();
        let ego_pos = scene.ego_position_at(scene_time).unwrap();

        assert!(
            bounds.contains(&ego_pos),
            "Ego at {:?} should be in bounds {:?}",
            ego_pos, bounds
        );
    }

    #[test]
    fn test_empty_scene() {
        let scene = SceneEpisode::from_frames(vec![]);
        let vp = ViewportEpisode::new_default(scene, ViewportConfig::default());
        assert_eq!(vp.n_frames(), 0);
        assert!(vp.view_x(0).is_none());
        assert!(vp.view_x_at(0.0).is_none());
    }

    #[test]
    fn test_single_frame_scene() {
        let scene = SceneEpisode::from_frames(vec![make_frame(50.0)]);
        let vp = ViewportEpisode::new_default(scene, ViewportConfig::default());
        assert_eq!(vp.n_frames(), 1);
        assert!(vp.view_x(0).is_some());
    }

    #[test]
    fn test_state_at_delegates() {
        let scene = make_scene(5, 10.0);
        let vp = ViewportEpisode::new_default(scene.clone(), ViewportConfig::default());
        let state_direct = scene.state_at(0.05).unwrap();
        let state_via_vp = vp.state_at(0.05).unwrap();
        assert!((state_direct.ego.x - state_via_vp.ego.x).abs() < 1e-9);
    }
}
