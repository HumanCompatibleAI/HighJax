//! Input handling for the Octane TUI app.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use tracing::info;

use super::{App, Focus, GraphicsControl, Modal, BehaviorScenarioField, Screen, ExplorerPane, ExplorerMode};
use super::navigation::{self, DEFAULT_PAGE_SIZE};

impl App {
    /// Handle a key event.
    pub fn handle_key(&mut self, key: KeyEvent) {
    // Ctrl+C always quits, regardless of mode
    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        self.should_quit = true;
        return;
    }

    // Ctrl+L copies log path to clipboard, regardless of mode
    if key.code == KeyCode::Char('l') && key.modifiers.contains(KeyModifiers::CONTROL) {
        self.copy_log_path_to_clipboard();
        return;
    }

    // 'c' copies the last toast's displayed text to clipboard (even if expired)
    if key.code == KeyCode::Char('c') && key.modifiers.is_empty() {
        if let Some(display_text) = self.last_toast_text.clone() {
            self.copy_to_clipboard(&display_text, &display_text);
            return;
        }
    }

    // Tab switching: 1/2 work from any context except text input, edit modes, or neo mode
    let in_text_mode = self.screen == Screen::BehaviorExplorer
        && matches!(self.explorer.mode, ExplorerMode::Edit | ExplorerMode::NewBehavior);
    let in_modal = self.active_modal.is_some();

    if !in_text_mode && !in_modal {
        match key.code {
            KeyCode::Char('1') => {
                if self.screen != Screen::Browse {
                    self.screen = Screen::Browse;
                    info!("Switched to Runs tab");
                }
                return;
            }
            KeyCode::Char('2') => {
                if self.screen != Screen::BehaviorExplorer {
                    self.screen = Screen::BehaviorExplorer;
                    info!("Switched to Behaviors tab");
                }
                return;
            }
            _ => {}
        }
    }

    // Delegate to tab-specific handler
    match self.screen {
        Screen::Browse => self.handle_browse_key(key),
        Screen::BehaviorExplorer => self.handle_explorer_key(key),
    }
}

    fn handle_explorer_key(&mut self, key: KeyEvent) {
        // Handle modal if active (e.g., Graphics modal)
        if let Some(modal) = self.active_modal {
            self.handle_modal_key(key, modal);
            // Invalidate preview cache when graphics settings may have changed
            self.explorer.preview_cache = None;
            return;
        }

        match self.explorer.mode {
            ExplorerMode::Edit => self.handle_explorer_edit_key(key),
            ExplorerMode::Browse => self.handle_explorer_browse_key(key),
            ExplorerMode::NewBehavior => self.handle_explorer_new_behavior_key(key),
        }
    }

    fn handle_explorer_edit_key(&mut self, key: KeyEvent) {
        // Reset NPC remove pending on any key except x
        if key.code != KeyCode::Char('x') {
            self.explorer.npc_remove_pending = false;
        }

        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => {
                self.exit_explorer_edit();
            }
            KeyCode::Up => self.editor_navigate_fields(-1),
            KeyCode::Down => self.editor_navigate_fields(1),
            KeyCode::Home => {
                if !self.explorer.editor_fields.is_empty() {
                    self.explorer.editor_selected_field = 0;
                }
            }
            KeyCode::End => {
                if !self.explorer.editor_fields.is_empty() {
                    self.explorer.editor_selected_field = self.explorer.editor_fields.len() - 1;
                }
            }
            KeyCode::Right => {
                let fine = key.modifiers.contains(KeyModifiers::SHIFT);
                self.editor_adjust_value(1.0, fine);
            }
            KeyCode::Left => {
                let fine = key.modifiers.contains(KeyModifiers::SHIFT);
                self.editor_adjust_value(-1.0, fine);
            }
            KeyCode::Enter => {
                self.editor_save();
            }
            KeyCode::Char('a') => {
                self.editor_add_npc();
            }
            KeyCode::Char('x') => {
                if self.explorer.npc_remove_pending {
                    self.editor_remove_npc();
                } else if self.npc_index_of_selected_field().is_some() {
                    self.explorer.npc_remove_pending = true;
                    self.show_toast(
                        "Press x again to remove NPC".to_string(),
                        std::time::Duration::from_secs(3),
                    );
                } else {
                    self.show_toast(
                        "Select an NPC field to remove".to_string(),
                        std::time::Duration::from_secs(2),
                    );
                }
            }
            _ => {}
        }
    }

    fn handle_explorer_browse_key(&mut self, key: KeyEvent) {
        // Reset delete pending on any key except d
        if key.code != KeyCode::Char('d') {
            self.explorer.delete_pending = false;
        }

        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => {
                self.screen = Screen::Browse;
                info!("Exited BehaviorExplorer via q/Esc");
            }
            // Pane mnemonics
            KeyCode::Char('b') => {
                self.explorer.pane_focus = ExplorerPane::Behaviors;
                info!("Explorer focus: Behaviors (mnemonic)");
            }
            KeyCode::Char('s') => {
                self.explorer.pane_focus = ExplorerPane::Scenarios;
                info!("Explorer focus: Scenarios (mnemonic)");
            }
            KeyCode::Char('p') => {
                self.explorer.pane_focus = ExplorerPane::Preview;
                info!("Explorer focus: Preview (mnemonic)");
            }
            KeyCode::Tab => {
                self.explorer.pane_focus = self.explorer.pane_focus.next();
                info!("Explorer focus: {:?}", self.explorer.pane_focus);
            }
            KeyCode::BackTab => {
                self.explorer.pane_focus = self.explorer.pane_focus.prev();
                info!("Explorer focus: {:?}", self.explorer.pane_focus);
            }
            KeyCode::Enter => match self.explorer.pane_focus {
                ExplorerPane::Behaviors => {
                    self.explorer.pane_focus = ExplorerPane::Scenarios;
                    info!("Explorer focus: Scenarios (via Enter)");
                }
                ExplorerPane::Scenarios => {
                    self.navigate_to_scenario_source();
                }
                ExplorerPane::Preview => {}
            },
            KeyCode::Char('e') => {
                self.enter_explorer_edit();
            }
            KeyCode::Char('n') => {
                if self.explorer.pane_focus == ExplorerPane::Scenarios {
                    self.explorer_new_scenario();
                }
            }
            KeyCode::Char('c') => {
                if self.explorer.pane_focus == ExplorerPane::Scenarios {
                    self.explorer_duplicate_scenario();
                }
            }
            KeyCode::Char('d') => {
                match self.explorer.pane_focus {
                    ExplorerPane::Scenarios => {
                        if self.explorer.delete_pending {
                            self.explorer_delete_scenario();
                        } else {
                            self.explorer.delete_pending = true;
                            self.show_toast(
                                "Press d again to delete scenario".to_string(),
                                std::time::Duration::from_secs(3),
                            );
                        }
                    }
                    ExplorerPane::Behaviors => {
                        if self.explorer.delete_pending {
                            self.explorer_delete_behavior();
                        } else {
                            self.explorer.delete_pending = true;
                            self.show_toast(
                                "Press d again to delete behavior".to_string(),
                                std::time::Duration::from_secs(3),
                            );
                        }
                    }
                    ExplorerPane::Preview => {}
                }
            }
            KeyCode::Char('N') => {
                if self.explorer.pane_focus == ExplorerPane::Behaviors {
                    self.enter_new_behavior_mode();
                }
            }
            KeyCode::Up => match self.explorer.pane_focus {
                ExplorerPane::Behaviors => self.explorer_navigate_behaviors(-1),
                ExplorerPane::Scenarios => self.explorer_navigate_scenarios(-1),
                ExplorerPane::Preview => {}
            },
            KeyCode::Down => match self.explorer.pane_focus {
                ExplorerPane::Behaviors => self.explorer_navigate_behaviors(1),
                ExplorerPane::Scenarios => self.explorer_navigate_scenarios(1),
                ExplorerPane::Preview => {}
            },
            KeyCode::Home => match self.explorer.pane_focus {
                ExplorerPane::Behaviors => {
                    self.explorer_navigate_behaviors(-(self.explorer.selected_behavior as isize));
                }
                ExplorerPane::Scenarios => {
                    self.explorer_navigate_scenarios(-(self.explorer.selected_scenario as isize));
                }
                ExplorerPane::Preview => {}
            },
            KeyCode::End => match self.explorer.pane_focus {
                ExplorerPane::Behaviors => {
                    let last = self.explorer.behaviors.len().saturating_sub(1);
                    self.explorer_navigate_behaviors(
                        last as isize - self.explorer.selected_behavior as isize);
                }
                ExplorerPane::Scenarios => {
                    let last = self.explorer.scenarios.len().saturating_sub(1);
                    self.explorer_navigate_scenarios(
                        last as isize - self.explorer.selected_scenario as isize);
                }
                ExplorerPane::Preview => {}
            },
            KeyCode::PageUp => match self.explorer.pane_focus {
                ExplorerPane::Behaviors => {
                    let step = navigation::page_size(
                        self.explorer.behaviors_visible_rows.unwrap_or(DEFAULT_PAGE_SIZE));
                    self.explorer_navigate_behaviors(-(step as isize));
                }
                ExplorerPane::Scenarios => {
                    let step = navigation::page_size(
                        self.explorer.scenarios_visible_rows.unwrap_or(DEFAULT_PAGE_SIZE));
                    self.explorer_navigate_scenarios(-(step as isize));
                }
                ExplorerPane::Preview => {}
            },
            KeyCode::PageDown => match self.explorer.pane_focus {
                ExplorerPane::Behaviors => {
                    let step = navigation::page_size(
                        self.explorer.behaviors_visible_rows.unwrap_or(DEFAULT_PAGE_SIZE));
                    self.explorer_navigate_behaviors(step as isize);
                }
                ExplorerPane::Scenarios => {
                    let step = navigation::page_size(
                        self.explorer.scenarios_visible_rows.unwrap_or(DEFAULT_PAGE_SIZE));
                    self.explorer_navigate_scenarios(step as isize);
                }
                ExplorerPane::Preview => {}
            },
            KeyCode::Char('r') => {
                self.active_modal = Some(Modal::Graphics);
                self.graphics_selection = GraphicsControl::Zoom;
                info!("Opened Graphics modal from explorer");
            }
            KeyCode::Char('y') => {
                self.dump_frame_state();
            }
            _ => {}
        }
    }

    fn handle_explorer_new_behavior_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.explorer.mode = ExplorerMode::Browse;
                info!("Cancelled new behavior creation");
            }
            KeyCode::Enter => {
                self.confirm_new_behavior();
            }
            KeyCode::Backspace => {
                self.explorer.new_behavior_name.pop();
            }
            KeyCode::Char(c) => {
                // Allow alphanumeric, dashes, underscores
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    self.explorer.new_behavior_name.push(c);
                }
            }
            _ => {}
        }
    }

    fn handle_browse_key(&mut self, key: KeyEvent) {
        // Handle modal-specific keys first
        if let Some(modal) = self.active_modal {
            self.handle_modal_key(key, modal);
            return;
        }

        match key.code {
            // Quit
            KeyCode::Char('q') | KeyCode::Esc => {
                self.should_quit = true;
            }
            // Cycle focus forward
            KeyCode::Tab => {
                self.focus = self.focus.next();
                info!("Focus changed to {:?}", self.focus);
            }
            // Cycle focus backward
            KeyCode::BackTab => {
                self.focus = self.focus.prev();
                info!("Focus changed to {:?}", self.focus);
            }
            // Direct pane focus (mnemonic keys shown in pane titles)
            KeyCode::Char('t') => {
                self.focus = Focus::Treks;
                info!("Focus changed to Treks (mnemonic)");
            }
            KeyCode::Char('a') => {
                self.focus = Focus::Parquets;
                info!("Focus changed to Parquets (mnemonic)");
            }
            KeyCode::Char('o') => {
                self.focus = Focus::Epochs;
                info!("Focus changed to Epochs (mnemonic)");
            }
            KeyCode::Char('e') => {
                self.focus = Focus::Episodes;
                info!("Focus changed to Episodes (mnemonic)");
            }
            KeyCode::Char('s') => {
                self.focus = Focus::Highway;
                info!("Focus changed to Scene (mnemonic)");
            }
            // Navigation within focused pane
            KeyCode::Up => {
                self.navigate_up();
            }
            KeyCode::Down => {
                self.navigate_down();
            }
            KeyCode::Left => {
                self.navigate_left();
            }
            KeyCode::Right => {
                self.navigate_right();
            }
            // j/k: frame back/forward (J/K = 5 frames)
            KeyCode::Char('j') => {
                self.navigate_frames(-1);
            }
            KeyCode::Char('J') => {
                self.navigate_frames(-5);
            }
            KeyCode::Char('k') => {
                self.navigate_frames(1);
            }
            KeyCode::Char('K') => {
                self.navigate_frames(5);
            }
            // h/l: go to start/end
            KeyCode::Char('h') => {
                self.go_to_first_frame();
            }
            KeyCode::Char('l') => {
                self.go_to_last_frame();
            }
            // Play/pause
            KeyCode::Char('p') => {
                self.toggle_playback();
            }
            // Home/End for navigation
            KeyCode::Home => {
                match self.focus {
                    Focus::Treks => {
                        self.selected_trek = 0;
                        self.adjust_treks_scroll();
                        self.load_selected_trek();
                    }
                    Focus::Parquets => {
                        if self.selected_parquet != 0 {
                            self.selected_parquet = 0;
                            self.adjust_parquets_scroll();
                            self.switch_parquet_source();
                        }
                    }
                    Focus::Highway => {
                        self.frame_index = 0;
                        self.reset_playback_timing();
                    }
                    Focus::Epochs => {
                        self.save_episode_position();
                        self.selected_epoch = 0;
                        self.selected_episode = 0;
                        self.restore_episode_position();
                        self.stop_playback();
                        self.adjust_epochs_scroll();
                    }
                    Focus::Episodes => {
                        self.save_episode_position();
                        self.selected_episode = 0;
                        self.restore_episode_position();
                        self.stop_playback();
                        self.adjust_episodes_scroll();
                    }
                }
            }
            KeyCode::End => {
                match self.focus {
                    Focus::Treks => {
                        if !self.trek_entries.is_empty() {
                            self.selected_trek = self.trek_entries.len() - 1;
                            self.adjust_treks_scroll();
                            self.load_selected_trek();
                        }
                    }
                    Focus::Parquets => {
                        if !self.parquet_sources.is_empty() {
                            let last = self.parquet_sources.len() - 1;
                            if self.selected_parquet != last {
                                self.selected_parquet = last;
                                self.adjust_parquets_scroll();
                                self.switch_parquet_source();
                            }
                        }
                    }
                    Focus::Highway => {
                        if let Some(n_frames) = self.current_episode_frame_count() {
                            self.frame_index = n_frames.saturating_sub(1);
                            self.reset_playback_timing();
                        }
                    }
                    Focus::Epochs => {
                        if !self.trek.epochs.is_empty() {
                            self.save_episode_position();
                            self.selected_epoch = self.trek.epochs.len() - 1;
                            self.selected_episode = 0;
                            self.restore_episode_position();
                            self.stop_playback();
                            self.adjust_epochs_scroll();
                        }
                    }
                    Focus::Episodes => {
                        let n_episodes = self.trek.epochs.get(self.selected_epoch)
                            .map(|e| e.episodes.len()).unwrap_or(0);
                        if n_episodes > 0 {
                            self.save_episode_position();
                            self.selected_episode = n_episodes - 1;
                            self.restore_episode_position();
                            self.stop_playback();
                            self.adjust_episodes_scroll();
                        }
                    }
                }
            }
            // Page navigation
            KeyCode::PageUp => {
                self.page_up();
            }
            KeyCode::PageDown => {
                self.page_down();
            }
            // Help dialog
            KeyCode::Char('?') => {
                self.active_modal = Some(Modal::Help);
                info!("Opened Help modal");
            }
            // Graphics dialog (unified settings)
            KeyCode::Char('r') => {
                self.active_modal = Some(Modal::Graphics);
                info!("Opened Graphics modal");
            }
            // Jump dialogs
            KeyCode::Char('g') => {
                self.jump_input.clear();
                self.active_modal = Some(Modal::JumpEpoch);
                info!("Opened Jump Epoch modal");
            }
            KeyCode::Char('G') => {
                self.jump_input.clear();
                self.active_modal = Some(Modal::JumpEpisode);
                info!("Opened Jump Episode modal");
            }
            KeyCode::Char('/') => {
                self.jump_input.clear();
                self.active_modal = Some(Modal::JumpFrame);
                info!("Opened Jump Frame modal");
            }
            // Debug overlay
            KeyCode::Char('d') => {
                self.show_debug = !self.show_debug;
                info!("Debug overlay: {}", if self.show_debug { "shown" } else { "hidden" });
            }
            // Copy octane command to clipboard (Shift+C)
            KeyCode::Char('C') => {
                self.copy_command_to_clipboard();
            }
            // Copy debug info to clipboard (Shift+D)
            KeyCode::Char('D') => {
                self.copy_debug_to_clipboard();
            }
            // Refresh trek from disk (Shift+R)
            KeyCode::Char('R') => {
                self.refresh_trek();
                self.spawn_behavior_load();
            }
            // Dump current frame state to JSON file
            KeyCode::Char('y') => {
                self.dump_frame_state();
            }
            // Behavior scenario capture modal
            KeyCode::Char('b') => {
                // Check if current frame has vehicle state (traffic geometry)
                let has_vehicle_state = self.current_frame()
                    .and_then(|f| f.vehicle_state.as_ref())
                    .is_some();
                if has_vehicle_state {
                    self.behavior_modal.action_weights = [0.0; 5];
                    self.behavior_modal.action_input.clear();
                    self.behavior_modal.focus = BehaviorScenarioField::Actions;
                    self.behavior_modal.action_cursor = 0;
                    self.load_behavior_suggestions();
                    // Keep existing name for convenience (user may add multiple scenarios)
                    self.active_modal = Some(Modal::BehaviorScenario);
                    info!("Opened Behavior Scenario modal");
                } else {
                    self.show_toast(
                        "No vehicle state data on this frame".to_string(),
                        std::time::Duration::from_secs(3),
                    );
                }
            }
            // Behavior explorer tab (Shift+B)
            KeyCode::Char('B') => {
                self.screen = Screen::BehaviorExplorer;
                info!("Switched to Behaviors tab via B");
            }
            // Sidebar width dialog (Shift+W) - opens Graphics modal focused on width
            KeyCode::Char('W') => {
                self.graphics_selection = GraphicsControl::SidebarWidth;
                self.active_modal = Some(Modal::Graphics);
                info!("Opened Graphics modal (sidebar width)");
            }
            _ => {}
        }
    }

    /// Handle key events when a modal is active.
    pub(super) fn handle_modal_key(&mut self, key: KeyEvent, modal: Modal) {
    // BehaviorScenario accepts text input — only Esc closes it, not 'q'.
    if modal == Modal::BehaviorScenario {
        if key.code == KeyCode::Esc {
            self.active_modal = None;
        } else {
            self.handle_behavior_scenario_key(key);
        }
        return;
    }
    match key.code {
        // Close modal
        KeyCode::Esc | KeyCode::Char('q') => {
            self.active_modal = None;
        }
        // Modal-specific handling
        _ => match modal {
            Modal::Help => {} // No special keys, Esc/q closes
            Modal::Graphics => self.handle_graphics_modal_key(key),
            Modal::JumpEpoch => self.handle_jump_epoch_key(key),
            Modal::JumpEpisode => self.handle_jump_episode_key(key),
            Modal::JumpFrame => self.handle_jump_frame_key(key),
            Modal::BehaviorScenario => unreachable!(),
        },
    }
}
}
