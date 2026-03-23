//! Playback control and timing for the Octane TUI app.

use std::time::Instant;
use tracing::{debug, info, trace};

use super::App;

impl App {
    /// Get the frame count for the currently selected episode.
    pub fn current_episode_frame_count(&self) -> Option<usize> {
        self.trek
            .epochs
            .get(self.selected_epoch)
            .and_then(|epoch| epoch.episodes.get(self.selected_episode))
            .map(|ep| ep.n_frames)
    }

    /// Reset effective timing to trek defaults (before episode data is inspected).
    pub(crate) fn reset_effective_timing(&mut self) {
        self.effective_seconds_per_frame = self.trek.seconds_per_sub_t;
        self.effective_n_sub = self.trek.n_sub_ts_per_t;
    }

    /// Detect whether loaded episode frames have sub-steps and update timing.
    ///
    /// Two parquet formats exist:
    ///   - New (float64 t): policy steps at t=0.0, 1.0, ... and sub-steps at 0.2, 0.4, ...
    ///     Detected via `frame_count > n_policy_frames`.
    ///   - Old (int64 t): sub-steps share the same integer t (0, 0, 0, 0, 0, 1, ...).
    ///     Detected via first two t values being equal.
    ///   - Breakdown (int64 t, no sub-steps): sequential integers (0, 1, 2, ...).
    ///
    /// When no sub-steps are present, we use `seconds_per_t` so that lerp interpolation
    /// in SceneEpisode produces correct-speed playback.
    pub(crate) fn detect_effective_timing(
        &mut self,
        t0: Option<f64>,
        t1: Option<f64>,
        frame_count: usize,
        n_policy_frames: usize,
    ) {
        // New format: fractional t values make n_policy_frames < frame_count
        let has_sub_steps = if frame_count > n_policy_frames {
            true
        } else if let (Some(t0), Some(t1)) = (t0, t1) {
            // Old format: sub-steps share the same integer t value
            (t0 - t1).abs() < 1e-6
        } else {
            // Not enough frames to tell; assume trek defaults
            self.reset_effective_timing();
            return;
        };

        if has_sub_steps {
            self.effective_seconds_per_frame = self.trek.seconds_per_sub_t;
            self.effective_n_sub = self.trek.n_sub_ts_per_t;
        } else {
            self.effective_seconds_per_frame = self.trek.seconds_per_t;
            self.effective_n_sub = 1;
        }
        debug!(
            "detect_effective_timing: has_sub_steps={} (frames={}, policy={}), dt={:.3}, n_sub={}",
            has_sub_steps, frame_count, n_policy_frames,
            self.effective_seconds_per_frame, self.effective_n_sub
        );
    }

    /// Update scene_time to current instant (call before drawing for accurate display).
    pub fn sync_scene_time(&mut self) {
        if !self.playback.playing {
            return;
        }

        let Some(start_time) = self.playback.start else {
            return;
        };

        let Some(n_states) = self.current_episode_frame_count() else {
            return;
        };

        let dt = self.effective_seconds_per_frame;

        // Calculate current scene time from wall clock
        let elapsed_real = start_time.elapsed().as_secs_f64();
        let elapsed_scene = elapsed_real * self.playback.speed;
        let old_scene_time = self.playback.scene_time;
        let old_frame_index = self.frame_index;
        self.playback.scene_time = (self.playback.start_scene_time + elapsed_scene).max(0.0);

        // Calculate max scene time (end of last state)
        let max_scene_time = (n_states - 1) as f64 * dt;

        if self.playback.scene_time >= max_scene_time {
            // Reached end of episode
            self.playback.scene_time = max_scene_time;
            self.frame_index = n_states - 1;
            self.playback.playing = false;
            self.playback.start = None;
            trace!(
                "ANIM: sync_scene_time END elapsed_real={:.4} scene_time={:.4} frame_index={}",
                elapsed_real, self.playback.scene_time, self.frame_index
            );
        } else {
            // Update frame_index
            self.frame_index = (self.playback.scene_time / dt).floor() as usize;

            let scene_time_delta = self.playback.scene_time - old_scene_time;
            let frame_changed = self.frame_index != old_frame_index;
            trace!(
                "ANIM: sync_scene_time elapsed_real={:.4} speed={:.2} start_scene={:.4} scene_time={:.4} delta={:.4} frame_index={} frame_changed={}",
                elapsed_real, self.playback.speed, self.playback.start_scene_time, self.playback.scene_time, scene_time_delta, self.frame_index, frame_changed
            );

        }
    }

    /// Advance frame during playback (periodic tick for logging/prerender).
    pub fn tick(&mut self) {
        if !self.playback.playing {
            return;
        }

        // Sync time first
        self.sync_scene_time();

        // Log for debugging
        if let Some(start_time) = self.playback.start {
            let elapsed_real = start_time.elapsed().as_secs_f64();
            let (state_idx, sub_idx) = self.current_subdivision();
            trace!(
                "ANIM: tick elapsed_real={:.4} speed={:.1} scene_time={:.4} state_idx={} sub_idx={} frame_index={}",
                elapsed_real, self.playback.speed, self.playback.scene_time, state_idx, sub_idx, self.frame_index
            );
        }
    }

    /// Format a frame index as a timestep string.
    /// With sub-steps: "35.40" (policy_t.fraction). Without: "35".
    pub fn format_timestep(&self, frame_idx: usize) -> String {
        let n_sub = self.effective_n_sub;
        if n_sub > 1 {
            let policy_t = frame_idx / n_sub;
            let fraction = (frame_idx % n_sub) as f64 / n_sub as f64;
            format!("{}{}", policy_t, &format!("{:.2}", fraction)[1..])
        } else {
            format!("{}", frame_idx)
        }
    }

    /// Get the current (state_idx, sub_idx) for cache lookup based on scene_time.
    pub fn current_subdivision(&self) -> (usize, usize) {
        use crate::render::n_subdivisions;

        let n_subs = n_subdivisions();
        let dt = self.effective_seconds_per_frame;

        // State index is floor(scene_time / dt)
        let state_idx = (self.playback.scene_time / dt).floor() as usize;

        // Fraction within the state (0.0 to 1.0)
        let state_start = state_idx as f64 * dt;
        let fraction = (self.playback.scene_time - state_start) / dt;

        // Subdivision index: round to nearest subdivision
        let sub_idx = (fraction * n_subs as f64).round() as usize;

        // Clamp sub_idx to valid range (0 to n_subs-1)
        let sub_idx = sub_idx.min(n_subs - 1);

        trace!(
            "ANIM: current_subdivision scene_time={:.4} state_idx={} fraction={:.3} sub_idx={}",
            self.playback.scene_time, state_idx, fraction, sub_idx
        );

        (state_idx, sub_idx)
    }

    /// Reset playback timing (call when frame position changes externally).
    /// Also syncs scene_time from frame_index for consistent state.
    pub(crate) fn reset_playback_timing(&mut self) {
        let dt = self.effective_seconds_per_frame;
        // Sync scene_time from frame_index (navigation sets frame_index directly)
        let old_scene_time = self.playback.scene_time;
        self.playback.scene_time = self.frame_index as f64 * dt;

        if self.playback.playing {
            debug!(
                "reset_playback_timing: old_scene_time={:.3}, new_scene_time={:.3}, frame_idx={}",
                old_scene_time, self.playback.scene_time, self.frame_index
            );
            self.playback.start = Some(Instant::now());
            self.playback.start_scene_time = self.playback.scene_time;
        }
    }

    /// Apply a deferred timestep (from `--timestep` CLI or scenario source navigation).
    /// Uses `effective_n_sub` arithmetic to convert the `t` value to a frame index.
    /// Called when the episode is already cached (no access to raw parquet `t` values).
    pub(crate) fn apply_pending_timestep(&mut self) {
        let Some(ts) = self.pending_timestep.take() else {
            return;
        };
        let max_frame = self.current_episode_frame_count()
            .unwrap_or(1)
            .saturating_sub(1);
        let n_sub = self.effective_n_sub;
        let frame = if n_sub > 1 {
            let policy = ts.floor() as usize;
            let frac = ts - ts.floor();
            let sub = (frac * n_sub as f64).round() as usize;
            policy * n_sub + sub
        } else {
            ts.round() as usize
        };
        let frame = frame.min(max_frame);
        self.frame_index = frame;
        self.reset_playback_timing();
        info!(
            "Timestep {} resolved to frame {} (n_sub={})",
            ts, frame, n_sub
        );
    }

    /// Toggle playback on/off, recording or clearing timing state.
    pub(crate) fn toggle_playback(&mut self) {
        self.playback.playing = !self.playback.playing;
        if self.playback.playing {
            self.playback.start = Some(Instant::now());
            self.playback.start_scene_time = self.playback.scene_time;
            debug!(
                "ANIM: PLAY_START frame_index={} scene_time={:.4} speed={:.1}",
                self.frame_index, self.playback.scene_time, self.playback.speed
            );
        } else {
            self.playback.start = None;
            debug!(
                "ANIM: PLAY_STOP frame_index={} scene_time={:.4}",
                self.frame_index, self.playback.scene_time
            );
        }
    }

    /// Stop playback and clear all timing state to prevent delayed actions.
    pub(crate) fn stop_playback(&mut self) {
        let dt = self.effective_seconds_per_frame;
        self.playback.playing = false;
        self.playback.start = None;
        self.playback.scene_time = self.frame_index as f64 * dt;
    }
}
