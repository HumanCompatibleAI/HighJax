//! Data types for episode frames (vehicle states, interpolation, env-specific data).

use serde::Serialize;

/// Vehicle state: position, heading, speed, and optional attention.
#[derive(Debug, Clone, Default, Serialize)]
pub struct VehicleState {
    pub x: f64,
    pub y: f64,
    pub heading: f64,
    pub speed: f64,
    /// Longitudinal acceleration in m/s² (negative = decelerating).
    /// Computed from consecutive frame speeds during episode loading.
    pub acceleration: f64,
    /// Attention weight from ego agent (None if NPC not observed).
    pub attention: Option<f64>,
}

impl VehicleState {
    /// Linear interpolation between two vehicle states.
    /// lerp_t=0 returns self, lerp_t=1 returns other.
    pub fn lerp(&self, other: &VehicleState, lerp_t: f64) -> VehicleState {
        VehicleState {
            x: self.x + (other.x - self.x) * lerp_t,
            y: self.y + (other.y - self.y) * lerp_t,
            heading: lerp_angle(self.heading, other.heading, lerp_t),
            speed: self.speed + (other.speed - self.speed) * lerp_t,
            acceleration: self.acceleration
                + (other.acceleration - self.acceleration) * lerp_t,
            attention: match (self.attention, other.attention) {
                (Some(a), Some(b)) => Some(a + (b - a) * lerp_t),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            },
        }
    }
}

/// Lerp between two angles, handling wraparound.
fn lerp_angle(a: f64, b: f64, t: f64) -> f64 {
    use std::f64::consts::PI;
    let mut diff = b - a;
    // Normalize to [-PI, PI]
    while diff > PI { diff -= 2.0 * PI; }
    while diff < -PI { diff += 2.0 * PI; }
    a + diff * t
}

/// Action distribution: dynamic list of (deed_name, probability) pairs.
/// Discovered from parquet schema columns matching `p.*`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ActionDistribution {
    pub probs: Vec<(String, f64)>,
}

impl ActionDistribution {
    /// Linear interpolation between two distributions.
    /// Both must have the same deed names in the same order.
    pub fn lerp(&self, other: &ActionDistribution, t: f64) -> ActionDistribution {
        let probs = self.probs.iter().zip(other.probs.iter())
            .map(|((name, a), (_, b))| (name.clone(), a + (b - a) * t))
            .collect();
        ActionDistribution { probs }
    }

    /// Look up probability by deed name (case-insensitive).
    pub fn get(&self, name: &str) -> Option<f64> {
        let lower = name.to_lowercase();
        self.probs.iter()
            .find(|(n, _)| n.to_lowercase() == lower)
            .map(|(_, p)| *p)
    }
}

/// Full frame state.
#[derive(Debug, Clone, Default, Serialize)]
pub struct FrameState {
    pub crashed: bool,
    pub ego: VehicleState,
    pub npcs: Vec<VehicleState>,
    /// Action distribution for the ego agent (None if not available).
    pub action_distribution: Option<ActionDistribution>,
    /// Chosen action index (0=LEFT, 1=IDLE, 2=RIGHT, 3=FASTER, 4=SLOWER).
    pub chosen_action: Option<u8>,
    /// Old action distribution for delta overlay. When present,
    /// `action_distribution` holds new probs and this holds old probs.
    pub old_action_distribution: Option<ActionDistribution>,
}

impl FrameState {
    /// Linear interpolation between two frame states.
    /// lerp_t=0 returns self, lerp_t=1 returns other.
    pub fn lerp(&self, other: &FrameState, lerp_t: f64) -> FrameState {
        let ego = self.ego.lerp(&other.ego, lerp_t);
        let npcs: Vec<VehicleState> = self.npcs
            .iter()
            .zip(other.npcs.iter())
            .map(|(a, b)| a.lerp(b, lerp_t))
            .collect();

        let action_distribution = match (&self.action_distribution, &other.action_distribution) {
            (Some(a), Some(b)) => Some(a.lerp(b, lerp_t)),
            (Some(a), None) => Some(a.clone()),
            (None, Some(b)) => Some(b.clone()),
            (None, None) => None,
        };

        // Keep chosen_action from the earlier frame (doesn't interpolate)
        let chosen_action = self.chosen_action.or(other.chosen_action);

        FrameState {
            crashed: self.crashed || other.crashed, // Crashed if either is crashed
            ego,
            npcs,
            action_distribution,
            chosen_action,
            old_action_distribution: None, // Delta overlay doesn't interpolate
        }
    }
}

/// Frame data with state and episode info.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct EsFrame {
    pub epoch: i64,
    pub episode: i64,
    /// Fractional timestep: floor(t) = policy step, frac(t) encodes sub-step.
    pub t: f64,
    pub reward: Option<f64>,
    /// Crash reward from `crash_reward` column. Only on last frame of crashed episodes.
    pub crash_reward: Option<f64>,
    /// Crash score from `crash_score` column. Only on last frame of crashed episodes.
    pub crash_score: Option<f64>,
    /// JSON-encoded action (from `action`).
    pub action: Option<String>,
    /// Human-readable action name (from `action_name`).
    pub action_name: Option<String>,
    pub state: FrameState,
    /// Value function estimate V(s) from the critic (None if not available).
    pub v: Option<f64>,
    /// Discounted return from `return` column.
    pub return_value: Option<f64>,
    /// Log-probability of chosen action (from microbes parquet `tendency` column).
    pub tendency: Option<f64>,
    /// GAE advantage from es parquet `advantage` column.
    pub advantage: Option<f64>,
    /// Normalized advantage (from microbes parquet `nz_advantage` column).
    pub nz_advantage: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Interpolation Tests
    // =========================================================================

    #[test]
    fn test_vehicle_lerp_at_t0_returns_first() {
        let v1 = VehicleState { x: 0.0, y: 0.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None };
        let v2 = VehicleState { x: 10.0, y: 5.0, heading: 1.0, speed: 0.0, acceleration: 0.0, attention: None };
        let result = v1.lerp(&v2, 0.0);

        assert!((result.x - 0.0).abs() < 1e-9);
        assert!((result.y - 0.0).abs() < 1e-9);
        assert!((result.heading - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_vehicle_lerp_at_t1_returns_second() {
        let v1 = VehicleState { x: 0.0, y: 0.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None };
        let v2 = VehicleState { x: 10.0, y: 5.0, heading: 1.0, speed: 0.0, acceleration: 0.0, attention: None };
        let result = v1.lerp(&v2, 1.0);

        assert!((result.x - 10.0).abs() < 1e-9);
        assert!((result.y - 5.0).abs() < 1e-9);
        assert!((result.heading - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_vehicle_lerp_at_midpoint() {
        let v1 = VehicleState { x: 0.0, y: 0.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None };
        let v2 = VehicleState { x: 10.0, y: 6.0, heading: 1.0, speed: 0.0, acceleration: 0.0, attention: None };
        let result = v1.lerp(&v2, 0.5);

        assert!((result.x - 5.0).abs() < 1e-9);
        assert!((result.y - 3.0).abs() < 1e-9);
        assert!((result.heading - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_vehicle_lerp_at_quarter_and_three_quarter() {
        let v1 = VehicleState { x: 0.0, y: 0.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None };
        let v2 = VehicleState { x: 100.0, y: 40.0, heading: 2.0, speed: 0.0, acceleration: 0.0, attention: None };

        let q1 = v1.lerp(&v2, 0.25);
        assert!((q1.x - 25.0).abs() < 1e-9);
        assert!((q1.y - 10.0).abs() < 1e-9);
        assert!((q1.heading - 0.5).abs() < 1e-9);

        let q3 = v1.lerp(&v2, 0.75);
        assert!((q3.x - 75.0).abs() < 1e-9);
        assert!((q3.y - 30.0).abs() < 1e-9);
        assert!((q3.heading - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_angle_lerp_no_wraparound() {
        // Angles close together, no wraparound needed
        let result = super::lerp_angle(0.0, 1.0, 0.5);
        assert!((result - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_angle_lerp_wraps_through_positive_pi() {
        // Angles on opposite sides of +PI: should wrap through PI, not through 0
        // a = 2.5 (close to PI), b = -2.5 (close to -PI)
        // The short path goes through ±PI, not through 0
        let a = 2.5;
        let b = -2.5;
        let result = super::lerp_angle(a, b, 0.5);

        // Midpoint should be near ±PI, not near 0
        assert!(
            result.abs() > 2.0,
            "Angle lerp should wrap through PI: a={}, b={}, result={}",
            a, b, result
        );
    }

    #[test]
    fn test_angle_lerp_wraps_through_negative_pi() {
        // Another wraparound case
        let a = -3.0;
        let b = 3.0;
        let result = super::lerp_angle(a, b, 0.5);

        // Midpoint should be near ±PI
        assert!(
            result.abs() > 2.5,
            "Angle lerp should wrap: a={}, b={}, result={}",
            a, b, result
        );
    }

    #[test]
    fn test_angle_lerp_does_not_wrap_when_close() {
        // Angles 0 and 1 are close, should not wrap
        let result = super::lerp_angle(0.0, 1.0, 0.5);
        assert!((result - 0.5).abs() < 1e-9);

        // Angles -0.5 and 0.5 are close
        let result2 = super::lerp_angle(-0.5, 0.5, 0.5);
        assert!((result2 - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_action_distribution_lerp() {
        let a = ActionDistribution {
            probs: vec![
                ("left".into(), 0.1), ("idle".into(), 0.5), ("right".into(), 0.1),
                ("faster".into(), 0.2), ("slower".into(), 0.1),
            ],
        };
        let b = ActionDistribution {
            probs: vec![
                ("left".into(), 0.3), ("idle".into(), 0.1), ("right".into(), 0.3),
                ("faster".into(), 0.2), ("slower".into(), 0.1),
            ],
        };
        let mid = a.lerp(&b, 0.5);
        assert_eq!(mid.probs.len(), 5);
        assert!((mid.probs[0].1 - 0.2).abs() < 1e-9);
        assert!((mid.probs[1].1 - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_action_distribution_get() {
        let dist = ActionDistribution {
            probs: vec![("left".into(), 0.1), ("idle".into(), 0.5)],
        };
        assert_eq!(dist.get("left"), Some(0.1));
        assert_eq!(dist.get("LEFT"), Some(0.1));
        assert_eq!(dist.get("missing"), None);
    }

    #[test]
    fn test_frame_lerp_with_matching_npc_counts() {
        let f1 = FrameState {
            crashed: false,
            ego: VehicleState { x: 0.0, y: 0.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None },
            npcs: vec![
                VehicleState { x: 10.0, y: 4.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None },
                VehicleState { x: 20.0, y: 8.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None },
            ],
            ..Default::default()
        };
        let f2 = FrameState {
            crashed: false,
            ego: VehicleState { x: 10.0, y: 2.0, heading: 0.5, speed: 0.0, acceleration: 0.0, attention: None },
            npcs: vec![
                VehicleState { x: 15.0, y: 4.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None },
                VehicleState { x: 30.0, y: 8.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None },
            ],
            ..Default::default()
        };

        let result = f1.lerp(&f2, 0.5);

        // Ego interpolated
        assert!((result.ego.x - 5.0).abs() < 1e-9);
        assert!((result.ego.y - 1.0).abs() < 1e-9);

        // NPCs interpolated
        assert_eq!(result.npcs.len(), 2);
        assert!((result.npcs[0].x - 12.5).abs() < 1e-9);
        assert!((result.npcs[1].x - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_frame_lerp_preserves_crashed_if_either_crashed() {
        let f1 = FrameState {
            crashed: false,
            ego: VehicleState::default(),
            npcs: vec![],
            ..Default::default()
        };
        let f2 = FrameState {
            crashed: true,
            ego: VehicleState::default(),
            npcs: vec![],
            ..Default::default()
        };

        // If second is crashed, result is crashed
        let result1 = f1.lerp(&f2, 0.5);
        assert!(result1.crashed, "Should be crashed if second frame crashed");

        // If first is crashed, result is crashed
        let result2 = f2.lerp(&f1, 0.5);
        assert!(result2.crashed, "Should be crashed if first frame crashed");
    }

    #[test]
    fn test_frame_lerp_with_empty_npc_lists() {
        let f1 = FrameState {
            crashed: false,
            ego: VehicleState { x: 0.0, y: 0.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None },
            npcs: vec![],
            ..Default::default()
        };
        let f2 = FrameState {
            crashed: false,
            ego: VehicleState { x: 10.0, y: 5.0, heading: 1.0, speed: 0.0, acceleration: 0.0, attention: None },
            npcs: vec![],
            ..Default::default()
        };

        let result = f1.lerp(&f2, 0.5);
        assert!(result.npcs.is_empty());
        assert!((result.ego.x - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_frame_lerp_with_many_npcs() {
        let make_npc = |i: usize| VehicleState {
            x: i as f64 * 10.0,
            y: 4.0,
            heading: 0.0,
            speed: 0.0,
            acceleration: 0.0,
            attention: None,
        };

        let f1 = FrameState {
            crashed: false,
            ego: VehicleState::default(),
            npcs: (0..10).map(make_npc).collect(),
            ..Default::default()
        };
        let f2 = FrameState {
            crashed: false,
            ego: VehicleState::default(),
            npcs: (0..10).map(|i| VehicleState {
                x: i as f64 * 10.0 + 5.0,
                y: 4.0,
                heading: 0.0,
                speed: 0.0,
                acceleration: 0.0,
                attention: None,
            }).collect(),
            ..Default::default()
        };

        let result = f1.lerp(&f2, 0.5);
        assert_eq!(result.npcs.len(), 10);

        // Check interpolation for first and last NPC
        assert!((result.npcs[0].x - 2.5).abs() < 1e-9);
        assert!((result.npcs[9].x - 92.5).abs() < 1e-9);
    }
}
