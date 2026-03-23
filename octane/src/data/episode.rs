//! Episode and frame data structures.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

/// Metadata about an episode (without loading full frame data).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct EpisodeMeta {
    /// Episode index within the epoch.
    pub index: usize,
    /// Path to the episode JSON file.
    pub path: PathBuf,
    /// Number of frames in this episode (including sub-step frames).
    pub n_frames: usize,
    /// Number of policy-step frames (sub_t=0 only).
    /// Equals n_frames when there are no sub-steps.
    pub n_policy_frames: usize,
    /// Number of alive (not crashed) policy frames. Used for survival calculation.
    pub n_alive_policy_frames: usize,
    /// Total reward accumulated in this episode (None = not yet computed).
    pub total_reward: Option<f64>,
    /// Normalized discounted return: (1-gamma) * sum(gamma^t * reward_t).
    pub nz_return: Option<f64>,
    /// Original epoch number from sample_es parquet (for index lookup).
    pub es_epoch: Option<i64>,
    /// Original episode number from sample_es parquet (for index lookup).
    pub es_episode: Option<i64>,
}

/// A fully loaded episode with all frames.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Episode {
    /// Metadata for this episode.
    pub meta: EpisodeMeta,
    /// All frames in this episode.
    pub frames: Vec<Frame>,
}

/// A single frame/step in an episode.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Frame {
    /// 5×5 observation grid (legacy format).
    pub observation: [[u8; 5]; 5],
    /// Action taken (0-4: left, idle, right, faster, slower).
    pub action: u8,
    /// Human-readable action name from the environment (e.g. "left", "RIGHT").
    pub action_name: Option<String>,
    /// Reward received after this action.
    pub reward: f64,
    /// Whether this frame ends the episode.
    pub done: bool,
    /// Full vehicle state from sample_es parquet (if available).
    pub vehicle_state: Option<super::jsonla::FrameState>,
    /// Crash reward: the reward from the action that caused the crash.
    /// Only set on the last frame of a crash episode.
    pub crash_reward: Option<f64>,
    /// Value function estimate V(s) from the critic (None if not available).
    pub v: Option<f64>,
    /// Discounted return from es parquet `return` column.
    pub return_value: Option<f64>,
    /// Log-probability of chosen action (from microbes parquet `tendency` column).
    pub tendency: Option<f64>,
    /// GAE advantage from es parquet `advantage` column.
    pub advantage: Option<f64>,
    /// Normalized advantage (from microbes parquet `nz_advantage` column).
    pub nz_advantage: Option<f64>,
}

/// Raw JSON structure for episode files.
#[derive(Debug, Deserialize)]
struct EpisodeJson {
    observations: Vec<Vec<Vec<u8>>>,
    actions: Vec<u8>,
    rewards: Vec<f64>,
    dones: Vec<bool>,
}

impl EpisodeMeta {
    /// Create metadata by peeking at episode file without full load.
    pub fn from_path(path: PathBuf, index: usize) -> Result<Self> {
        let content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read episode file: {:?}", path))?;
        let json: EpisodeJson = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse episode JSON: {:?}", path))?;

        let n_frames = json.observations.len();
        let total_reward: f64 = json.rewards.iter().sum();

        Ok(Self {
            index,
            path,
            n_frames,
            n_policy_frames: n_frames,
            n_alive_policy_frames: n_frames,
            total_reward: Some(total_reward),
            nz_return: None,
            es_epoch: None,
            es_episode: None,
        })
    }

    /// Load the full episode data.
    pub fn load(&self) -> Result<Episode> {
        let content = fs::read_to_string(&self.path)
            .with_context(|| format!("Failed to read episode file: {:?}", self.path))?;
        let json: EpisodeJson = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse episode JSON: {:?}", self.path))?;

        let frames = json
            .observations
            .iter()
            .enumerate()
            .map(|(i, observation_grid)| {
                let mut observation = [[0u8; 5]; 5];
                for (row_idx, row) in observation_grid.iter().enumerate().take(5) {
                    for (col_idx, &val) in row.iter().enumerate().take(5) {
                        observation[row_idx][col_idx] = val;
                    }
                }

                Frame {
                    observation,
                    action: json.actions.get(i).copied().unwrap_or(0),
                    action_name: None,
                    reward: json.rewards.get(i).copied().unwrap_or(0.0),
                    done: json.dones.get(i).copied().unwrap_or(false),
                    vehicle_state: None,
                    crash_reward: None,
                    v: None,
                    return_value: None,
                    tendency: None,
                    advantage: None,
                    nz_advantage: None,
                }
            })
            .collect();

        Ok(Episode {
            meta: self.clone(),
            frames,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_frame_default_values() {
        let frame = Frame {
            observation: [[0; 5]; 5],
            action: 0,
            action_name: None,
            reward: 0.0,
            done: false,
            vehicle_state: None,
            crash_reward: None,
            v: None,
            return_value: None,
            tendency: None,
            advantage: None,
            nz_advantage: None,
        };
        assert_eq!(frame.action, 0);
        assert!(!frame.done);
    }

    #[test]
    fn test_frame_with_observation() {
        let mut observation = [[0u8; 5]; 5];
        observation[2][2] = 1; // car at center
        observation[0][0] = 2; // obstacle
        let frame = Frame {
            observation,
            action: 3, // up
            action_name: None,
            reward: 1.0,
            done: false,
            vehicle_state: None,
            crash_reward: None,
            v: None,
            return_value: None,
            tendency: None,
            advantage: None,
            nz_advantage: None,
        };
        assert_eq!(frame.observation[2][2], 1);
        assert_eq!(frame.observation[0][0], 2);
        assert_eq!(frame.action, 3);
    }

    #[test]
    fn test_episode_meta_clone() {
        let meta = EpisodeMeta {
            index: 0,
            path: PathBuf::from("/tmp/test.json"),
            n_frames: 100,
            n_policy_frames: 100,
            n_alive_policy_frames: 100,
            total_reward: Some(42.5),
            nz_return: None,
            es_epoch: None,
            es_episode: None,
        };
        let cloned = meta.clone();
        assert_eq!(cloned.index, meta.index);
        assert_eq!(cloned.n_frames, meta.n_frames);
        assert_eq!(cloned.total_reward, meta.total_reward);
    }

    #[test]
    fn test_episode_meta_from_valid_json() {
        let json = r#"{
            "observations": [[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]],
            "actions": [0],
            "rewards": [1.0],
            "dones": [false]
        }"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(json.as_bytes()).unwrap();
        let path = file.path().to_path_buf();

        let meta = EpisodeMeta::from_path(path, 0).unwrap();
        assert_eq!(meta.index, 0);
        assert_eq!(meta.n_frames, 1);
        assert_eq!(meta.total_reward, Some(1.0));
    }

    #[test]
    fn test_episode_meta_from_multi_frame_json() {
        let json = r#"{
            "observations": [
                [[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]],
                [[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
                [[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
            ],
            "actions": [3, 3, 3],
            "rewards": [0.5, 0.5, 1.0],
            "dones": [false, false, true]
        }"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(json.as_bytes()).unwrap();
        let path = file.path().to_path_buf();

        let meta = EpisodeMeta::from_path(path, 5).unwrap();
        assert_eq!(meta.index, 5);
        assert_eq!(meta.n_frames, 3);
        assert_eq!(meta.total_reward, Some(2.0));
    }

    #[test]
    fn test_episode_load_frames() {
        let json = r#"{
            "observations": [
                [[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]],
                [[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
            ],
            "actions": [3, 0],
            "rewards": [0.5, -0.1],
            "dones": [false, true]
        }"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(json.as_bytes()).unwrap();
        let path = file.path().to_path_buf();

        let meta = EpisodeMeta::from_path(path, 0).unwrap();
        let episode = meta.load().unwrap();

        assert_eq!(episode.frames.len(), 2);
        assert_eq!(episode.frames[0].action, 3);
        assert_eq!(episode.frames[0].observation[2][2], 1);
        assert!(!episode.frames[0].done);
        assert_eq!(episode.frames[1].action, 0);
        assert!(episode.frames[1].done);
    }

    #[test]
    fn test_episode_meta_invalid_json() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"not valid json").unwrap();
        let path = file.path().to_path_buf();

        let result = EpisodeMeta::from_path(path, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_episode_meta_nonexistent_file() {
        let result = EpisodeMeta::from_path(PathBuf::from("/nonexistent/file.json"), 0);
        assert!(result.is_err());
    }
}
