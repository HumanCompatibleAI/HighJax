//! Scene episode - raw episode data with on-demand interpolation.
//!
//! SceneEpisode wraps the raw frame data from a Highway episode and provides
//! interpolation for smooth playback at arbitrary scene times.

use crate::data::FrameState;
use crate::worlds::coords::ScenePoint;
use crate::worlds::DEFAULT_SECONDS_PER_TIMESTEP;

/// Raw episode data with on-demand interpolation.
///
/// This is the base layer of the coordinate system architecture.
/// It holds the discrete frame states and provides interpolation for
/// arbitrary scene times.
#[derive(Debug, Clone)]
pub struct SceneEpisode {
    /// All frame states in this episode.
    frames: Vec<FrameState>,
    /// Time step between frames (typically 0.1s).
    seconds_per_timestep: f64,
    /// How far back (in seconds) to look when computing acceleration.
    acceleration_lookback_seconds: f64,
}

/// Default lookback window for acceleration computation.
const DEFAULT_ACCELERATION_LOOKBACK_SECONDS: f64 = 0.3;

#[allow(dead_code)]
impl SceneEpisode {
    /// Create a SceneEpisode from a vector of frame states.
    pub fn from_frames(frames: Vec<FrameState>) -> Self {
        let mut ep = Self {
            frames,
            seconds_per_timestep: DEFAULT_SECONDS_PER_TIMESTEP,
            acceleration_lookback_seconds: DEFAULT_ACCELERATION_LOOKBACK_SECONDS,
        };
        ep.compute_accelerations();
        ep
    }

    /// Create a SceneEpisode with custom timestep.
    pub fn from_frames_with_timestep(frames: Vec<FrameState>, seconds_per_timestep: f64) -> Self {
        let mut ep = Self {
            frames,
            seconds_per_timestep,
            acceleration_lookback_seconds: DEFAULT_ACCELERATION_LOOKBACK_SECONDS,
        };
        ep.compute_accelerations();
        ep
    }

    /// Set the acceleration lookback window and recompute.
    pub fn set_acceleration_lookback_seconds(&mut self, seconds: f64) {
        self.acceleration_lookback_seconds = seconds;
        self.compute_accelerations();
    }

    /// Compute acceleration for each vehicle by comparing speed over a lookback window.
    ///
    /// Rounds the lookback to the nearest frame count (minimum 1 frame back).
    /// For early frames where the lookback reaches before the episode start,
    /// compares against frame 0. Frame 0 copies from frame 1.
    fn compute_accelerations(&mut self) {
        let dt = self.seconds_per_timestep;
        if dt <= 0.0 || self.frames.len() < 2 {
            return;
        }
        let lookback = (self.acceleration_lookback_seconds / dt).round() as usize;
        let lookback = lookback.max(1);
        for i in 1..self.frames.len() {
            // Use frame 0 if lookback reaches before episode start
            let prev_i = i.saturating_sub(lookback);
            let delta_t = (i - prev_i) as f64 * dt;
            let inv_delta_t = 1.0 / delta_t;
            let (prev_slice, rest) = self.frames.split_at_mut(i);
            let prev = &prev_slice[prev_i];
            let curr = &mut rest[0];
            curr.ego.acceleration = (curr.ego.speed - prev.ego.speed) * inv_delta_t;
            for (npc, prev_npc) in curr.npcs.iter_mut().zip(prev.npcs.iter()) {
                npc.acceleration = (npc.speed - prev_npc.speed) * inv_delta_t;
            }
        }
        // Frame 0: copy from frame 1
        if self.frames.len() >= 2 {
            let accel_ego = self.frames[1].ego.acceleration;
            self.frames[0].ego.acceleration = accel_ego;
            let n_npcs = self.frames[0].npcs.len().min(self.frames[1].npcs.len());
            for j in 0..n_npcs {
                self.frames[0].npcs[j].acceleration = self.frames[1].npcs[j].acceleration;
            }
        }
    }

    /// Number of discrete timesteps/frames.
    pub fn n_frames(&self) -> usize {
        self.frames.len()
    }

    /// Check if episode is empty.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Total episode duration in seconds.
    pub fn duration(&self) -> f64 {
        if self.frames.is_empty() {
            0.0
        } else {
            (self.frames.len() - 1) as f64 * self.seconds_per_timestep
        }
    }

    /// Time step between frames.
    pub fn seconds_per_timestep(&self) -> f64 {
        self.seconds_per_timestep
    }

    /// Get state at discrete timestep index.
    pub fn state(&self, index: usize) -> Option<&FrameState> {
        self.frames.get(index)
    }

    /// Get interpolated state at arbitrary scene time.
    ///
    /// Clamps to [0, duration] and linearly interpolates between adjacent frames.
    pub fn state_at(&self, scene_time: f64) -> Option<FrameState> {
        if self.frames.is_empty() {
            return None;
        }

        if self.frames.len() == 1 {
            return Some(self.frames[0].clone());
        }

        // Clamp to valid range
        let duration = self.duration();
        let t = scene_time.clamp(0.0, duration);

        // Compute frame indices and interpolation factor
        let index_f = t / self.seconds_per_timestep;
        let index_a = (index_f.floor() as usize).min(self.frames.len() - 1);
        let index_b = (index_a + 1).min(self.frames.len() - 1);
        let lerp_t = index_f.fract();

        // If at exact frame boundary or at last frame, return without interpolation
        if lerp_t < 1e-9 || index_a == index_b {
            return Some(self.frames[index_a].clone());
        }

        // Interpolate between frames
        Some(self.frames[index_a].lerp(&self.frames[index_b], lerp_t))
    }

    /// Find the index of the first crashed frame, if any.
    pub fn crash_frame(&self) -> Option<usize> {
        self.frames.iter().position(|f| f.crashed)
    }

    /// Get ego position at discrete timestep index.
    pub fn ego_position(&self, index: usize) -> Option<ScenePoint> {
        self.frames.get(index).map(|f| ScenePoint::new(f.ego.x, f.ego.y))
    }

    /// Get interpolated ego position at arbitrary scene time.
    pub fn ego_position_at(&self, scene_time: f64) -> Option<ScenePoint> {
        self.state_at(scene_time)
            .map(|f| ScenePoint::new(f.ego.x, f.ego.y))
    }

    /// Get all ego x positions (for viewport trajectory computation).
    pub fn ego_xs(&self) -> Vec<f64> {
        self.frames.iter().map(|f| f.ego.x).collect()
    }

    /// Get ego velocities via finite differences.
    ///
    /// Uses forward difference for all frames except last (backward diff).
    pub fn ego_velocities(&self) -> Vec<f64> {
        if self.frames.is_empty() {
            return Vec::new();
        }

        if self.frames.len() == 1 {
            return vec![0.0];
        }

        let dt = self.seconds_per_timestep;
        let xs = self.ego_xs();
        let n = xs.len();
        let mut velocities = Vec::with_capacity(n);

        // Forward difference for all but last
        for i in 0..n - 1 {
            velocities.push((xs[i + 1] - xs[i]) / dt);
        }

        // Backward difference for last
        velocities.push((xs[n - 1] - xs[n - 2]) / dt);

        velocities
    }

    /// Access the underlying frames.
    pub fn frames(&self) -> &[FrameState] {
        &self.frames
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::VehicleState;

    fn make_frame(ego_x: f64, crashed: bool) -> FrameState {
        FrameState {
            crashed,
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

    fn make_episode(n: usize) -> SceneEpisode {
        let frames: Vec<FrameState> = (0..n)
            .map(|i| make_frame(i as f64 * 10.0, false))
            .collect();
        SceneEpisode::from_frames(frames)
    }

    #[test]
    fn test_n_frames_returns_correct_count() {
        let ep = make_episode(5);
        assert_eq!(ep.n_frames(), 5);
    }

    #[test]
    fn test_duration_calculation() {
        let ep = make_episode(10);
        // 10 frames, 9 intervals of 0.1s each
        assert!((ep.duration() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_state_returns_correct_frame() {
        let ep = make_episode(5);
        let frame = ep.state(2).unwrap();
        assert!((frame.ego.x - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_state_returns_none_for_out_of_bounds() {
        let ep = make_episode(5);
        assert!(ep.state(10).is_none());
    }

    #[test]
    fn test_state_at_zero_returns_first_frame() {
        let ep = make_episode(5);
        let state = ep.state_at(0.0).unwrap();
        assert!((state.ego.x - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_state_at_duration_returns_last_frame() {
        let ep = make_episode(5);
        let state = ep.state_at(ep.duration()).unwrap();
        assert!((state.ego.x - 40.0).abs() < 1e-9);
    }

    #[test]
    fn test_state_at_interpolation_midpoint() {
        let ep = make_episode(3);
        // Frames at x=0, 10, 20
        // At t=0.05 (midpoint of first interval), should get x=5
        let state = ep.state_at(0.05).unwrap();
        assert!((state.ego.x - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_state_at_interpolation_quarter() {
        let ep = make_episode(3);
        // At t=0.025 (25% of first interval), should get x=2.5
        let state = ep.state_at(0.025).unwrap();
        assert!((state.ego.x - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_state_at_clamps_negative_time() {
        let ep = make_episode(5);
        let state = ep.state_at(-1.0).unwrap();
        assert!((state.ego.x - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_state_at_clamps_beyond_duration() {
        let ep = make_episode(5);
        let state = ep.state_at(100.0).unwrap();
        assert!((state.ego.x - 40.0).abs() < 1e-9);
    }

    #[test]
    fn test_crash_frame_finds_crash() {
        let frames = vec![
            make_frame(0.0, false),
            make_frame(10.0, false),
            make_frame(20.0, true),
            make_frame(30.0, true),
        ];
        let ep = SceneEpisode::from_frames(frames);
        assert_eq!(ep.crash_frame(), Some(2));
    }

    #[test]
    fn test_crash_frame_returns_none_when_no_crash() {
        let ep = make_episode(5);
        assert_eq!(ep.crash_frame(), None);
    }

    #[test]
    fn test_ego_position_returns_scene_point() {
        let ep = make_episode(3);
        let pos = ep.ego_position(1).unwrap();
        assert!((pos.x - 10.0).abs() < 1e-9);
        assert!((pos.y - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_ego_position_at_interpolates() {
        let ep = make_episode(3);
        let pos = ep.ego_position_at(0.05).unwrap();
        assert!((pos.x - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_episode() {
        let ep = SceneEpisode::from_frames(vec![]);
        assert!(ep.is_empty());
        assert_eq!(ep.n_frames(), 0);
        assert!((ep.duration() - 0.0).abs() < 1e-9);
        assert!(ep.state(0).is_none());
        assert!(ep.state_at(0.0).is_none());
        assert!(ep.crash_frame().is_none());
    }

    #[test]
    fn test_single_frame_episode() {
        let ep = SceneEpisode::from_frames(vec![make_frame(50.0, false)]);
        assert_eq!(ep.n_frames(), 1);
        assert!((ep.duration() - 0.0).abs() < 1e-9);

        let state = ep.state_at(0.0).unwrap();
        assert!((state.ego.x - 50.0).abs() < 1e-9);

        // Any time should return the same frame
        let state2 = ep.state_at(1.0).unwrap();
        assert!((state2.ego.x - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_ego_velocities() {
        let ep = make_episode(4);
        // x = 0, 10, 20, 30 at dt=0.1
        // v = 100, 100, 100, 100 m/s
        let vels = ep.ego_velocities();
        assert_eq!(vels.len(), 4);
        for v in vels {
            assert!((v - 100.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_ego_velocities_single_frame() {
        let ep = SceneEpisode::from_frames(vec![make_frame(50.0, false)]);
        let vels = ep.ego_velocities();
        assert_eq!(vels.len(), 1);
        assert!((vels[0] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_ego_velocities_empty() {
        let ep = SceneEpisode::from_frames(vec![]);
        let vels = ep.ego_velocities();
        assert!(vels.is_empty());
    }
}
