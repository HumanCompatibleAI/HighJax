//! Navigation and scroll management for the Octane TUI app.

use super::{App, Focus};

/// Fallback page size when visible rows aren't known yet.
pub(crate) const DEFAULT_PAGE_SIZE: usize = 10;

/// Navigate a scrollable list by `delta` items, returning `(new_selected, new_scroll)`.
/// Clamps selection to `[0, len-1]` and keeps it within the visible window.
pub(crate) fn navigate_list(
    selected: usize, scroll: usize, len: usize,
    visible_rows: usize, delta: isize,
) -> (usize, usize) {
    if len == 0 {
        return (0, 0);
    }
    let new_sel = (selected as isize + delta).clamp(0, len as isize - 1) as usize;
    let new_scroll = adjust_scroll(new_sel, scroll, len, visible_rows);
    (new_sel, new_scroll)
}

/// Adjust scroll offset so `selected` stays within the visible window.
pub(crate) fn adjust_scroll(
    selected: usize, scroll: usize, len: usize, visible_rows: usize,
) -> usize {
    let mut s = scroll;
    if selected < s {
        s = selected;
    }
    if selected >= s + visible_rows {
        s = selected - visible_rows + 1;
    }
    let max_scroll = len.saturating_sub(visible_rows);
    s.min(max_scroll)
}

/// Page size: 80% of visible rows, minimum 1.
pub(crate) fn page_size(visible_rows: usize) -> usize {
    (visible_rows * 4 / 5).max(1)
}

impl App {
    /// Save the current frame position for the current (epoch, episode).
    pub(crate) fn save_episode_position(&mut self) {
        self.episode_frame_positions.insert(
            (self.selected_epoch, self.selected_episode),
            self.frame_index,
        );
    }

    /// Restore a previously saved frame position, clamped to the episode length.
    /// Falls back to frame 0 if no position was saved.
    pub(crate) fn restore_episode_position(&mut self) {
        let saved = self.episode_frame_positions
            .get(&(self.selected_epoch, self.selected_episode))
            .copied()
            .unwrap_or(0);
        let max = self.current_episode_frame_count()
            .unwrap_or(1)
            .saturating_sub(1);
        self.frame_index = saved.min(max);
    }

    /// Navigate up in the current focused pane.
    pub(crate) fn navigate_up(&mut self) {
        match self.focus {
            Focus::Treks => {
                if self.selected_trek > 0 {
                    self.selected_trek -= 1;
                    self.adjust_treks_scroll();
                    self.load_selected_trek();
                }
            }
            Focus::Parquets => {
                if self.selected_parquet > 0 {
                    self.selected_parquet -= 1;
                    self.adjust_parquets_scroll();
                    self.switch_parquet_source();
                }
            }
            Focus::Epochs => {
                if self.selected_epoch > 0 {
                    self.save_episode_position();
                    self.selected_epoch -= 1;
                    self.selected_episode = 0;
                    self.restore_episode_position();
                    self.stop_playback();
                    self.adjust_epochs_scroll();
                }
            }
            Focus::Episodes => {
                if self.selected_episode > 0 {
                    self.save_episode_position();
                    self.selected_episode -= 1;
                    self.restore_episode_position();
                    self.stop_playback();
                    self.adjust_episodes_scroll();
                }
            }
            Focus::Highway => {
                // Up in highway/metrics doesn't scroll - use left/right for frames
            }
        }
    }

    /// Navigate down in the current focused pane.
    pub(crate) fn navigate_down(&mut self) {
        match self.focus {
            Focus::Treks => {
                if self.selected_trek + 1 < self.trek_entries.len() {
                    self.selected_trek += 1;
                    self.adjust_treks_scroll();
                    self.load_selected_trek();
                }
            }
            Focus::Parquets => {
                if self.selected_parquet + 1 < self.parquet_sources.len() {
                    self.selected_parquet += 1;
                    self.adjust_parquets_scroll();
                    self.switch_parquet_source();
                }
            }
            Focus::Epochs => {
                if self.selected_epoch + 1 < self.trek.epochs.len() {
                    self.save_episode_position();
                    self.selected_epoch += 1;
                    self.selected_episode = 0;
                    self.restore_episode_position();
                    self.stop_playback();
                    self.adjust_epochs_scroll();
                }
            }
            Focus::Episodes => {
                let n_episodes = self.trek.epochs.get(self.selected_epoch)
                    .map(|e| e.episodes.len()).unwrap_or(0);
                if self.selected_episode + 1 < n_episodes {
                    self.save_episode_position();
                    self.selected_episode += 1;
                    self.restore_episode_position();
                    self.stop_playback();
                    self.adjust_episodes_scroll();
                }
            }
            Focus::Highway => {
                // Down in highway/metrics doesn't scroll - use left/right for frames
            }
        }
    }

    /// Navigate left (previous frame in highway).
    pub(crate) fn navigate_left(&mut self) {
        if matches!(self.focus, Focus::Highway) && self.frame_index > 0 {
            self.stop_playback();
            self.frame_index -= 1;
            self.reset_playback_timing();
        }
    }

    /// Navigate right (next frame in highway).
    pub(crate) fn navigate_right(&mut self) {
        if matches!(self.focus, Focus::Highway) {
            if let Some(n_frames) = self.current_episode_frame_count() {
                if self.frame_index + 1 < n_frames {
                    self.stop_playback();
                    self.frame_index += 1;
                    self.reset_playback_timing();
                }
            }
        }
    }

    /// Navigate frames by a signed step (negative = backward, positive = forward).
    pub(crate) fn navigate_frames(&mut self, step: i32) {
        if let Some(n_frames) = self.current_episode_frame_count() {
            let new_index = if step < 0 {
                self.frame_index.saturating_sub((-step) as usize)
            } else {
                (self.frame_index + step as usize).min(n_frames.saturating_sub(1))
            };
            if new_index != self.frame_index {
                self.stop_playback();
                self.frame_index = new_index;
                self.reset_playback_timing();
            }
        }
    }

    /// Go to the first frame.
    pub(crate) fn go_to_first_frame(&mut self) {
        if self.frame_index != 0 {
            self.stop_playback();
            self.frame_index = 0;
            self.reset_playback_timing();
        }
    }

    /// Go to the last frame.
    pub(crate) fn go_to_last_frame(&mut self) {
        if let Some(n_frames) = self.current_episode_frame_count() {
            let last = n_frames.saturating_sub(1);
            if self.frame_index != last {
                self.stop_playback();
                self.frame_index = last;
                self.reset_playback_timing();
            }
        }
    }

    /// Page size for the current focus: 80% of visible rows.
    fn page_size(&self) -> usize {
        let visible = match self.focus {
            Focus::Treks => self.treks_visible_rows.unwrap_or(DEFAULT_PAGE_SIZE),
            Focus::Parquets => self.parquets_visible_rows.unwrap_or(DEFAULT_PAGE_SIZE),
            Focus::Epochs => self.epochs_visible_rows.unwrap_or(DEFAULT_PAGE_SIZE),
            Focus::Episodes => self.episodes_visible_rows.unwrap_or(DEFAULT_PAGE_SIZE),
            Focus::Highway => DEFAULT_PAGE_SIZE,
        };
        (visible * 4 / 5).max(1)
    }

    /// Page up navigation.
    pub(crate) fn page_up(&mut self) {
        let step = self.page_size();
        match self.focus {
            Focus::Treks => {
                self.selected_trek = self.selected_trek.saturating_sub(step);
                self.adjust_treks_scroll();
                self.load_selected_trek();
            }
            Focus::Parquets => {
                let old = self.selected_parquet;
                self.selected_parquet = self.selected_parquet.saturating_sub(step);
                self.adjust_parquets_scroll();
                if self.selected_parquet != old {
                    self.switch_parquet_source();
                }
            }
            Focus::Epochs => {
                self.save_episode_position();
                self.selected_epoch = self.selected_epoch.saturating_sub(step);
                self.selected_episode = 0;
                self.restore_episode_position();
                self.stop_playback();
                self.adjust_epochs_scroll();
            }
            Focus::Episodes => {
                self.save_episode_position();
                self.selected_episode = self.selected_episode.saturating_sub(step);
                self.restore_episode_position();
                self.stop_playback();
                self.adjust_episodes_scroll();
            }
            Focus::Highway => {
                self.frame_index = self.frame_index.saturating_sub(step);
                self.reset_playback_timing();
            }
        }
    }

    /// Page down navigation.
    pub(crate) fn page_down(&mut self) {
        let step = self.page_size();
        match self.focus {
            Focus::Treks => {
                let max = self.trek_entries.len().saturating_sub(1);
                self.selected_trek = (self.selected_trek + step).min(max);
                self.adjust_treks_scroll();
                self.load_selected_trek();
            }
            Focus::Parquets => {
                let old = self.selected_parquet;
                let max = self.parquet_sources.len().saturating_sub(1);
                self.selected_parquet = (self.selected_parquet + step).min(max);
                self.adjust_parquets_scroll();
                if self.selected_parquet != old {
                    self.switch_parquet_source();
                }
            }
            Focus::Epochs => {
                let max_epoch = self.trek.epochs.len().saturating_sub(1);
                self.save_episode_position();
                self.selected_epoch = (self.selected_epoch + step).min(max_epoch);
                self.selected_episode = 0;
                self.restore_episode_position();
                self.stop_playback();
                self.adjust_epochs_scroll();
            }
            Focus::Episodes => {
                let max_episode = self.trek.epochs.get(self.selected_epoch)
                    .map(|e| e.episodes.len().saturating_sub(1)).unwrap_or(0);
                self.save_episode_position();
                self.selected_episode = (self.selected_episode + step).min(max_episode);
                self.restore_episode_position();
                self.stop_playback();
                self.adjust_episodes_scroll();
            }
            Focus::Highway => {
                if let Some(n_frames) = self.current_episode_frame_count() {
                    let max_frame = n_frames.saturating_sub(1);
                    self.frame_index = (self.frame_index + step).min(max_frame);
                    self.reset_playback_timing();
                }
            }
        }
    }

    /// Update treks visible rows (called from UI).
    pub fn update_treks_scroll_bounds(&mut self, visible_rows: usize) {
        self.treks_visible_rows = Some(visible_rows);
    }

    /// Visible rows in epochs panel (updated during rendering).
    fn epochs_visible_rows(&self) -> usize {
        self.epochs_visible_rows.unwrap_or(10)
    }

    /// Update scroll bounds based on actual visible area (called from UI).
    pub fn update_epochs_scroll_bounds(&mut self, visible_rows: usize) {
        self.epochs_visible_rows = Some(visible_rows);
        self.adjust_epochs_scroll();
    }

    pub(crate) fn adjust_epochs_scroll(&mut self) {
        let visible_rows = self.epochs_visible_rows();
        let n_epochs = self.trek.epochs.len();
        // Keep selected epoch in visible range
        if self.selected_epoch < self.epochs_scroll {
            self.epochs_scroll = self.selected_epoch;
        }
        if self.selected_epoch >= self.epochs_scroll + visible_rows {
            self.epochs_scroll = self.selected_epoch - visible_rows + 1;
        }
        // Don't scroll past what fills the pane
        let max_scroll = n_epochs.saturating_sub(visible_rows);
        self.epochs_scroll = self.epochs_scroll.min(max_scroll);
    }

    /// Visible rows in episodes panel (updated during rendering).
    fn episodes_visible_rows(&self) -> usize {
        self.episodes_visible_rows.unwrap_or(5)
    }

    /// Update scroll bounds based on actual visible area (called from UI).
    pub fn update_episodes_scroll_bounds(&mut self, visible_rows: usize) {
        self.episodes_visible_rows = Some(visible_rows);
        self.adjust_episodes_scroll();
    }

    /// Update parquets visible rows (called from UI).
    pub fn update_parquets_scroll_bounds(&mut self, visible_rows: usize) {
        self.parquets_visible_rows = Some(visible_rows);
        self.adjust_parquets_scroll();
    }

    /// Adjust parquets scroll to keep selection visible.
    pub(crate) fn adjust_parquets_scroll(&mut self) {
        let visible_rows = self.parquets_visible_rows.unwrap_or(3);
        let n = self.parquet_sources.len();
        if self.selected_parquet < self.parquets_scroll {
            self.parquets_scroll = self.selected_parquet;
        }
        if self.selected_parquet >= self.parquets_scroll + visible_rows {
            self.parquets_scroll = self.selected_parquet - visible_rows + 1;
        }
        let max_scroll = n.saturating_sub(visible_rows);
        self.parquets_scroll = self.parquets_scroll.min(max_scroll);
    }

    /// Adjust episodes scroll to keep selection visible.
    pub(crate) fn adjust_episodes_scroll(&mut self) {
        let visible_rows = self.episodes_visible_rows();
        if self.selected_episode < self.episodes_scroll {
            self.episodes_scroll = self.selected_episode;
        }
        if self.selected_episode >= self.episodes_scroll + visible_rows {
            self.episodes_scroll = self.selected_episode - visible_rows + 1;
        }
    }
}
