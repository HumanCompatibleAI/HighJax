//! Jump-to-epoch/episode/frame modal key handling.

use crossterm::event::{KeyCode, KeyEvent};
use tracing::info;

use super::App;

impl App {
    /// Handle jump input keys (common logic for all jump modals).
    pub(super) fn handle_jump_input(&mut self, key: KeyEvent) -> Option<usize> {
        match key.code {
            KeyCode::Char(c) if c.is_ascii_digit() => {
                self.jump_input.push(c);
                None
            }
            KeyCode::Backspace => {
                self.jump_input.pop();
                None
            }
            KeyCode::Enter => {
                let value = self.jump_input.parse::<usize>().ok();
                if value.is_some() {
                    self.last_jump = value;
                }
                self.active_modal = None;
                value
            }
            // Use last jump value with 'n' for repeat
            KeyCode::Char('n') => {
                self.active_modal = None;
                self.last_jump
            }
            _ => None,
        }
    }

    /// Handle Jump Epoch modal keys.
    pub(super) fn handle_jump_epoch_key(&mut self, key: KeyEvent) {
        if let Some(target) = self.handle_jump_input(key) {
            // Find epoch by epoch_number, fall back to clamped index
            let idx = self.trek.epochs.iter()
                .position(|e| e.epoch_number == target as i64)
                .unwrap_or_else(|| target.min(self.trek.epochs.len().saturating_sub(1)));
            self.selected_epoch = idx;
            self.adjust_epochs_scroll();
            self.save_episode_position();
            self.selected_episode = 0;
            self.restore_episode_position();
            self.stop_playback();
            info!("Jumped to epoch {} (index {})", target, idx);
        }
    }

    /// Handle Jump Episode modal keys.
    pub(super) fn handle_jump_episode_key(&mut self, key: KeyEvent) {
        if let Some(target) = self.handle_jump_input(key) {
            // Look up by es_episode first (matches --episode CLI and episodes pane display),
            // fall back to clamped index.
            let max_episode = self
                .trek
                .epochs
                .get(self.selected_epoch)
                .map(|e| e.episodes.len().saturating_sub(1))
                .unwrap_or(0);
            let idx = self.trek.epochs.get(self.selected_epoch)
                .and_then(|e| e.episodes.iter()
                    .position(|em| em.es_episode == Some(target as i64)))
                .unwrap_or_else(|| target.min(max_episode));
            self.save_episode_position();
            self.selected_episode = idx;
            self.restore_episode_position();
            self.stop_playback();
            info!("Jumped to episode {} (index {})", target, idx);
        }
    }

    /// Handle Jump Frame modal keys.
    pub(super) fn handle_jump_frame_key(&mut self, key: KeyEvent) {
        if let Some(target) = self.handle_jump_input(key) {
            let max_frame = self.current_episode_frame_count().unwrap_or(1).saturating_sub(1);
            self.frame_index = target.min(max_frame);
            info!("Jumped to frame {}", self.frame_index);
        }
    }
}
