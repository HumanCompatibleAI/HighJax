//! Clipboard and toast utilities for the Octane TUI app.

use std::time::{Duration, Instant};
use tracing::info;

use super::{App, Toast};

impl App {
    /// Format debug info text (shared by overlay and clipboard).
    pub(crate) fn debug_info_text(&self) -> String {
        let render_mode = if self.render_config.enabled { "Mango" } else { "ASCII" };
        let frame_count = self.current_episode_frame_count().unwrap_or(0);

        let timestep_str = self.format_timestep(self.frame_index);
        let max_str = self.format_timestep(frame_count.saturating_sub(1));

        format!(
            "Epoch:    {:>4} / {:<4}\n\
             Episode:  {:>4} / {:<4}\n\
             Timestep: {:>7} / {:<7}\n\
             SceneT:   {:>7.3}s\n\
             FPS:      {:>3}\n\
             Speed:    {:>5.2}×\n\
             Mode:     {}\n\
             Playing:  {}",
            self.trek.epochs.get(self.selected_epoch)
                .map(|e| e.epoch_number).unwrap_or(0),
            self.trek.epochs.last()
                .map(|e| e.epoch_number).unwrap_or(0),
            self.trek.epochs.get(self.selected_epoch)
                .and_then(|e| e.episodes.get(self.selected_episode))
                .and_then(|ep| ep.es_episode)
                .unwrap_or(self.selected_episode as i64),
            self.trek.epochs.get(self.selected_epoch)
                .and_then(|e| e.episodes.last())
                .and_then(|ep| ep.es_episode)
                .unwrap_or(0),
            timestep_str,
            max_str,
            self.playback.scene_time,
            self.playback.fps,
            self.playback.speed,
            render_mode,
            self.playback.playing,
        )
    }

    /// Copy text to clipboard and show a toast. This is the unified entry point
    /// for all copy-to-clipboard operations.
    ///
    /// When `label` equals `content`, the toast shows "Copied: " (dim) + content
    /// (bright) so the user sees what was copied. When `label` is a summary
    /// description, the toast shows "Copied {label}" in uniform color.
    pub(crate) fn copy_to_clipboard(&mut self, content: &str, label: &str) {
        if let Err(e) = Self::try_clipboard_copy(content) {
            tracing::warn!("Failed to copy to clipboard: {}", e);
            self.show_toast("Clipboard copy failed", Duration::from_secs(2));
        } else {
            info!("Copied {} to clipboard", label);
            let verbatim = content == label;
            let (message, ellipsis_at) = if verbatim {
                Self::truncate_middle(label, 40)
            } else {
                (format!("Copied {}", label), None)
            };
            self.push_toast(Toast {
                prefix: if verbatim { Some("Copied: ".to_string()) } else { None },
                message,
                expiry: Instant::now() + Duration::from_secs(3),
                clipboard_text: Some(content.to_string()),
                ellipsis_at,
            });
        }
    }

    /// Build an octane command line that reopens the current view.
    pub(crate) fn copy_command_to_clipboard(&mut self) {
        let trek_path_display = Self::posh_path(&self.trek.path.display().to_string());

        let epoch_number = self.trek.epochs.get(self.selected_epoch)
            .map(|e| e.epoch_number).unwrap_or(0);

        let es_episode = self.trek.epochs.get(self.selected_epoch)
            .and_then(|e| e.episodes.get(self.selected_episode))
            .and_then(|ep| ep.es_episode)
            .unwrap_or(self.selected_episode as i64);

        let timestep_str = self.format_timestep(self.frame_index);

        // Use the parquet file path as target if a non-default source is selected,
        // otherwise use the trek path.
        let target = if self.selected_parquet > 0 {
            if let Some(source) = self.parquet_sources.get(self.selected_parquet) {
                Self::posh_path(&source.path.display().to_string())
            } else {
                trek_path_display.clone()
            }
        } else {
            trek_path_display.clone()
        };

        let cmd = format!(
            "octane -t {} --epoch {} -e {} --timestep {}",
            target,
            epoch_number,
            es_episode,
            timestep_str,
        );

        self.copy_to_clipboard(&cmd, &cmd);
    }

    /// Copy debug info to clipboard with trek path appended.
    pub(crate) fn copy_debug_to_clipboard(&mut self) {
        let trek_path_display = Self::posh_path(&self.trek.path.display().to_string());
        let debug_text = format!("{}\nPath:     {}", self.debug_info_text(), trek_path_display);
        self.copy_to_clipboard(&debug_text, "debug info");
    }

    /// Get the current FrameState from whichever view is active.
    ///
    /// In BehaviorExplorer: builds FrameState from the selected scenario's state dict.
    /// In Browse: returns the current frame's vehicle_state.
    pub(crate) fn current_frame_state(&self) -> Option<crate::data::FrameState> {
        use super::state::Screen;
        match self.screen {
            Screen::BehaviorExplorer => {
                let entry = self.explorer.scenarios
                    .get(self.explorer.selected_scenario)?;
                if !entry.has_state { return None; }
                let behavior = self.explorer.behaviors
                    .get(self.explorer.selected_behavior)?;
                let scenarios = crate::data::behaviors::load_behavior_scenarios(
                    &behavior.path,
                )?;
                let scenario = scenarios.get(self.explorer.selected_scenario)?;
                let state_json = scenario.get("state")?;
                Some(crate::data::behaviors::build_frame_from_state(
                    state_json,
                    self.config.octane.road.n_lanes,
                    self.config.octane.road.lane_width,
                ))
            }
            Screen::Browse => {
                self.current_frame()?.vehicle_state.clone()
            }
        }
    }

    /// Dump the current FrameState to `octane-frame-state.json` in the trek directory.
    pub(crate) fn dump_frame_state(&mut self) {
        let Some(state) = self.current_frame_state() else {
            self.show_toast("No frame state to dump", Duration::from_secs(2));
            return;
        };
        let path = self.trek.path.join("octane-frame-state.json");
        match serde_json::to_string_pretty(&state) {
            Ok(json) => match std::fs::write(&path, &json) {
                Ok(()) => {
                    info!("Dumped frame state to {}", path.display());
                    self.show_toast("Frame state dumped", Duration::from_secs(2));
                }
                Err(e) => {
                    tracing::warn!("Failed to write frame state: {}", e);
                    self.show_toast(format!("Write failed: {}", e), Duration::from_secs(2));
                }
            },
            Err(e) => {
                tracing::warn!("Failed to serialize frame state: {}", e);
                self.show_toast(format!("Serialize failed: {}", e), Duration::from_secs(2));
            }
        }
    }

    /// Copy the current log file path to clipboard.
    pub(crate) fn copy_log_path_to_clipboard(&mut self) {
        match leona::log_path() {
            Some(path) => {
                let poshed = Self::posh_path(&path.display().to_string());
                self.copy_to_clipboard(&poshed, &poshed);
            }
            None => {
                self.show_toast("No log file active", Duration::from_secs(2));
            }
        }
    }

    /// Show a plain toast message for a given duration.
    pub(crate) fn show_toast(&mut self, msg: impl Into<String>, duration: Duration) {
        self.push_toast(Toast {
            prefix: None,
            message: msg.into(),
            expiry: Instant::now() + duration,
            clipboard_text: None,
            ellipsis_at: None,
        });
    }

    /// Truncate a string to `max_len` chars with a middle ellipsis.
    /// Returns (truncated_string, Some(split_pos)) or (original, None) if short enough.
    fn truncate_middle(s: &str, max_len: usize) -> (String, Option<usize>) {
        if s.len() <= max_len {
            return (s.to_string(), None);
        }
        // Each half gets (max_len - 1) / 2 chars (1 char reserved for ellipsis)
        let half = (max_len - 1) / 2;
        let left = &s[..half];
        let right = &s[s.len() - half..];
        (format!("{}{}", left, right), Some(half))
    }

    /// Push a toast onto the queue (max 3 visible at once).
    ///
    /// - Plain toasts (no prefix) replace any existing plain toast.
    /// - Prefixed toasts replace an existing toast with the same prefix.
    /// - Different prefixes coexist (e.g. "Copied:" alongside "Neo:").
    pub(crate) fn push_toast(&mut self, toast: Toast) {
        self.last_toast_text = Some(toast.display_text());
        let replace_idx = self.toasts.iter().position(|t| {
            match (&t.prefix, &toast.prefix) {
                (None, None) => true,
                (Some(a), Some(b)) => a == b,
                _ => false,
            }
        });
        if let Some(idx) = replace_idx {
            self.toasts[idx].message = toast.message;
            self.toasts[idx].expiry = toast.expiry;
            self.toasts[idx].clipboard_text = toast.clipboard_text;
            self.toasts[idx].ellipsis_at = toast.ellipsis_at;
            return;
        }
        const MAX_TOASTS: usize = 3;
        while self.toasts.len() >= MAX_TOASTS {
            self.toasts.pop_front();
        }
        self.toasts.push_back(toast);
    }

    /// Remove expired toasts from the queue.
    pub(crate) fn gc_toasts(&mut self) {
        let now = Instant::now();
        self.toasts.retain(|t| now < t.expiry);
    }

    /// Shorten a path using the `posh` executable.
    fn posh_path(path: &str) -> String {
        crate::util::posh_path(std::path::Path::new(path))
    }

    /// Attempt to copy text to clipboard.
    pub(crate) fn try_clipboard_copy(text: &str) -> std::io::Result<()> {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let commands: &[(&str, &[&str])] = &[
            ("clip", &[]),
            ("xclip", &["-selection", "clipboard"]),
            ("xsel", &["--clipboard", "--input"]),
            ("wl-copy", &[]),
        ];

        for (cmd, args) in commands {
            let result = Command::new(cmd)
                .args(*args)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .and_then(|mut child| {
                    if let Some(stdin) = child.stdin.as_mut() {
                        stdin.write_all(text.as_bytes())?;
                    }
                    child.wait()
                });
            if result.is_ok() {
                return Ok(());
            }
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No clipboard utility found (clip, xclip, xsel, or wl-copy)",
        ))
    }
}
