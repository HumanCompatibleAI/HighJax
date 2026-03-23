//! Trek management for the Octane TUI app.

use std::time::Duration;
use tracing::info;

use super::{App, TrekViewState};
use crate::data::Trek;

impl App {
    /// Reload the trek from disk, preserving current selection where possible.
    /// Also re-discovers the treks list, preserving the selected trek.
    pub(crate) fn refresh_trek(&mut self) {
        let path = self.trek.path.clone();
        info!("Refreshing trek from {}", crate::util::posh_path(&path));
        match Trek::load(path.clone()) {
            Ok(new_trek) => {
                self.trek = new_trek;

                // Clamp selections
                if self.trek.epochs.is_empty() {
                    self.selected_epoch = 0;
                    self.selected_episode = 0;
                } else {
                    self.selected_epoch = self.selected_epoch.min(self.trek.epochs.len() - 1);
                    let n_episodes = self.trek.epochs[self.selected_epoch].episodes.len();
                    if n_episodes == 0 {
                        self.selected_episode = 0;
                    } else {
                        self.selected_episode = self.selected_episode.min(n_episodes - 1);
                    }
                }

                // Re-discover treks, preserving selection
                let old_selected = self.selected_trek;
                self.trek_entries = crate::data::discover_treks();
                self.selected_trek = self
                    .trek_entries
                    .iter()
                    .position(|e| e.path == path)
                    .unwrap_or(old_selected.min(
                        self.trek_entries.len().saturating_sub(1),
                    ));
                self.adjust_treks_scroll();

                // Re-discover parquet sources
                self.parquet_sources = crate::data::discover_parquet_sources(&path);
                self.selected_parquet = self.selected_parquet
                    .min(self.parquet_sources.len().saturating_sub(1));

                // Invalidate cached episode so it reloads
                self.cached_episode_key = (usize::MAX, usize::MAX);
                self.current_episode = None;
                self.invalidate_viewport();
                self.frame_index = 0;
                self.reset_effective_timing();
                self.stop_playback();

                let msg = format!(
                    "Refreshed: {} epochs",
                    self.trek.epochs.len()
                );
                info!("{}", msg);
                self.show_toast(msg, Duration::from_secs(2));
            }
            Err(e) => {
                let msg = format!("Refresh failed: {}", e);
                tracing::warn!("{}", msg);
                self.show_toast(msg, Duration::from_secs(3));
            }
        }
    }

    /// Switch to a different parquet source and rebuild epochs/episodes from its index.
    pub(crate) fn switch_parquet_source(&mut self) {
        let Some(source) = self.parquet_sources.get(self.selected_parquet) else {
            return;
        };

        let source_path = source.path.clone();
        let source_display = source.display.clone();

        match crate::data::es_parquet::EsParquetIndex::build(&source_path) {
            Ok(index) => {
                self.trek.epochs = crate::data::trek::epochs_from_parquet_index(
                    &index, &source_path, &self.trek.path,
                );
                self.trek.es_parquet_index = Some(index);

                // Reset selections
                self.selected_epoch = self.trek.epochs.len().saturating_sub(1);
                self.selected_episode = 0;
                self.epochs_scroll = 0;
                self.episodes_scroll = 0;
                self.frame_index = 0;
                self.reset_effective_timing();
                self.stop_playback();
                self.current_episode = None;
                self.cached_episode_key = (usize::MAX, usize::MAX);
                self.invalidate_viewport();
                self.episode_frame_positions.clear();
                self.adjust_epochs_scroll();

                info!("Switched to parquet source '{}': {} epochs", source_display, self.trek.epochs.len());
            }
            Err(e) => {
                let msg = format!("Failed to load parquet '{}': {}", source_display, e);
                tracing::warn!("{}", msg);
                self.show_toast(msg, std::time::Duration::from_secs(3));
            }
        }
    }

    /// Adjust treks scroll to keep selection visible.
    pub(crate) fn adjust_treks_scroll(&mut self) {
        if self.selected_trek < self.treks_scroll {
            self.treks_scroll = self.selected_trek;
        }
        // We don't know visible rows here; UI will clamp if needed
    }

    /// Save the current trek's selection state for later restoration.
    fn save_trek_view_state(&mut self) {
        self.trek_view_states.insert(
            self.trek.path.clone(),
            TrekViewState {
                selected_parquet: self.selected_parquet,
                selected_epoch: self.selected_epoch,
                selected_episode: self.selected_episode,
                epochs_scroll: self.epochs_scroll,
                episodes_scroll: self.episodes_scroll,
            },
        );
    }

    /// Load the currently selected trek entry, replacing the active trek.
    /// Restores previously saved selection state if the trek was visited before;
    /// otherwise selects the last epoch and first episode.
    pub(crate) fn load_selected_trek(&mut self) {
        if let Some(entry) = self.trek_entries.get(self.selected_trek) {
            let new_path = entry.path.clone();
            match Trek::load(new_path.clone()) {
                Ok(new_trek) => {
                    info!("Loaded trek: {}", entry.display);

                    // Save current trek's state before switching
                    self.save_trek_view_state();

                    self.trek = new_trek;
                    self.reset_effective_timing();
                    self.stop_playback();
                    self.current_episode = None;
                    self.cached_episode_key = (usize::MAX, usize::MAX);
                    self.viewport_episode = None;
                    self.svg_episode = None;
                    self.episode_frame_positions.clear();

                    // Re-discover parquet sources for the new trek
                    self.parquet_sources = crate::data::discover_parquet_sources(&new_path);

                    if let Some(saved) = self.trek_view_states.get(&new_path) {
                        // Restore previously saved state, clamping to valid ranges
                        self.selected_parquet = saved.selected_parquet
                            .min(self.parquet_sources.len().saturating_sub(1));
                        self.selected_epoch = saved.selected_epoch
                            .min(self.trek.epochs.len().saturating_sub(1));
                        let n_episodes = self.trek.epochs
                            .get(self.selected_epoch)
                            .map(|e| e.episodes.len())
                            .unwrap_or(0);
                        self.selected_episode = saved.selected_episode
                            .min(n_episodes.saturating_sub(1));
                        self.epochs_scroll = saved.epochs_scroll;
                        self.episodes_scroll = saved.episodes_scroll;
                    } else {
                        // First visit: select first parquet, last epoch, first episode
                        self.selected_parquet = 0;
                        self.selected_epoch = self.trek.epochs.len().saturating_sub(1);
                        self.selected_episode = 0;
                        self.episodes_scroll = 0;
                    }
                    self.parquets_scroll = 0;

                    self.adjust_epochs_scroll();
                    self.frame_index = 0;
                    self.ensure_episode_loaded();
                }
                Err(e) => {
                    info!("Failed to load trek: {}", e);
                    self.show_toast(
                        format!("Failed to load trek: {}", e),
                        Duration::from_secs(3),
                    );
                }
            }
        }
    }
}
