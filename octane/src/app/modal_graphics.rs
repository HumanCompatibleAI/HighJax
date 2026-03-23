//! Graphics settings modal key handling.

use crossterm::event::{KeyCode, KeyEvent};
use tracing::info;

use super::{App, GraphicsControl, Screen, LIGHT_BLEND_MODES};

impl App {
    /// Get the graphics control set for the current screen.
    pub(crate) fn graphics_control_set(&self) -> &[GraphicsControl] {
        let in_explorer = self.screen == Screen::BehaviorExplorer;
        if in_explorer {
            &GraphicsControl::EXPLORER
        } else {
            &GraphicsControl::ALL
        }
    }

    /// Handle Graphics modal keys.
    pub(super) fn handle_graphics_modal_key(&mut self, key: KeyEvent) {
        let set = self.graphics_control_set();

        match key.code {
            // Navigate between controls
            KeyCode::Up | KeyCode::Char('k') => {
                self.graphics_selection = self.graphics_selection.prev_in(set);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.graphics_selection = self.graphics_selection.next_in(set);
            }
            // Adjust current control
            KeyCode::Left | KeyCode::Char('h') => {
                self.adjust_graphics_control(-1);
            }
            KeyCode::Right | KeyCode::Char('l') => {
                self.adjust_graphics_control(1);
            }
            // Mnemonics — jump directly to control (only if in active set)
            KeyCode::Char('f') if set.contains(&GraphicsControl::Fps) => {
                self.graphics_selection = GraphicsControl::Fps;
            }
            KeyCode::Char('s') if set.contains(&GraphicsControl::PlaybackSpeed) => {
                self.graphics_selection = GraphicsControl::PlaybackSpeed;
            }
            KeyCode::Char('z') if set.contains(&GraphicsControl::Zoom) => {
                self.graphics_selection = GraphicsControl::Zoom;
            }
            KeyCode::Char('w') if set.contains(&GraphicsControl::SidebarWidth) => {
                self.graphics_selection = GraphicsControl::SidebarWidth;
            }
            KeyCode::Char('p') if set.contains(&GraphicsControl::PodiumMarker) => {
                self.graphics_selection = GraphicsControl::PodiumMarker;
            }
            KeyCode::Char('c') if set.contains(&GraphicsControl::Scala) => {
                self.graphics_selection = GraphicsControl::Scala;
            }
            KeyCode::Char('x') if set.contains(&GraphicsControl::Sextants) => {
                self.graphics_selection = GraphicsControl::Sextants;
            }
            KeyCode::Char('o') if set.contains(&GraphicsControl::Octants) => {
                self.graphics_selection = GraphicsControl::Octants;
            }
            KeyCode::Char('v') if set.contains(&GraphicsControl::VelocityArrows) => {
                self.graphics_selection = GraphicsControl::VelocityArrows;
            }
            KeyCode::Char('i') if set.contains(&GraphicsControl::ActionDistribution) => {
                self.graphics_selection = GraphicsControl::ActionDistribution;
            }
            KeyCode::Char('t') if set.contains(&GraphicsControl::ActionDistributionText) => {
                self.graphics_selection = GraphicsControl::ActionDistributionText;
            }
            KeyCode::Char('a') if set.contains(&GraphicsControl::Attention) => {
                self.graphics_selection = GraphicsControl::Attention;
            }
            KeyCode::Char('n') if set.contains(&GraphicsControl::NpcText) => {
                self.graphics_selection = GraphicsControl::NpcText;
            }
            KeyCode::Char('d') if set.contains(&GraphicsControl::DebugEye) => {
                self.graphics_selection = GraphicsControl::DebugEye;
            }
            KeyCode::Char('b') if set.contains(&GraphicsControl::LightBlendMode) => {
                self.graphics_selection = GraphicsControl::LightBlendMode;
            }
            KeyCode::Char('e') if set.contains(&GraphicsControl::Theme) => {
                self.graphics_selection = GraphicsControl::Theme;
            }
            KeyCode::Char('r') | KeyCode::Enter => {
                self.active_modal = None;
            }
            _ => {}
        }
    }

    /// Adjust the currently selected graphics control.
    pub(super) fn adjust_graphics_control(&mut self, delta: i32) {
        match self.graphics_selection {
            GraphicsControl::Theme => {
                self.scene_theme = self.scene_theme.next();
                info!("Scene theme: {}", self.scene_theme.label());
            }
            GraphicsControl::Fps => {
                if delta > 0 {
                    self.playback.fps = (self.playback.fps + 1).min(60);
                } else {
                    self.playback.fps = self.playback.fps.saturating_sub(1).max(1);
                }
                info!("FPS set to {}", self.playback.fps);
            }
            GraphicsControl::PlaybackSpeed => {
                if delta > 0 {
                    self.playback.speed = (self.playback.speed + 0.25).min(10.0);
                } else {
                    self.playback.speed = (self.playback.speed - 0.25).max(0.25);
                }
                // Re-anchor wall-clock reference without snapping scene_time
                // (reset_playback_timing would snap to frame_index, losing sub-frame position)
                if self.playback.playing {
                    self.playback.start = Some(std::time::Instant::now());
                    self.playback.start_scene_time = self.playback.scene_time;
                }
                info!("Playback speed set to {}×", self.playback.speed);
            }
            GraphicsControl::Zoom => {
                if delta > 0 {
                    self.zoom = (self.zoom * 1.1).min(10.0);
                } else {
                    self.zoom = (self.zoom / 1.1).max(0.1);
                }
                // Invalidate viewport so it rebuilds with new zoom
                self.invalidate_viewport();
                info!("Zoom: {:.2}", self.zoom);
            }
            GraphicsControl::SidebarWidth => {
                if delta > 0 {
                    self.sidebar_width = (self.sidebar_width + 2).min(80);
                } else {
                    self.sidebar_width = self.sidebar_width.saturating_sub(2).max(30);
                }
                self.svg_episode = None; // Force scene re-render
                info!("Sidebar width: {}", self.sidebar_width);
            }
            GraphicsControl::PodiumMarker => {
                let p = &mut self.highway_prefs;
                p.show_podium_marker = !p.show_podium_marker;
                info!("Podium marker: {}", if p.show_podium_marker { "on" } else { "off" });
            }
            GraphicsControl::Scala => {
                let p = &mut self.highway_prefs;
                p.show_scala = !p.show_scala;
                info!("Scala: {}", if p.show_scala { "on" } else { "off" });
            }
            GraphicsControl::Sextants => {
                self.use_sextants = !self.use_sextants;
                info!("Sextants: {}", if self.use_sextants { "on" } else { "off" });
            }
            GraphicsControl::Octants => {
                self.use_octants = !self.use_octants;
                info!("Octants: {}", if self.use_octants { "on" } else { "off" });
            }
            GraphicsControl::VelocityArrows => {
                let p = &mut self.highway_prefs;
                if delta > 0 {
                    p.velocity_arrows = p.velocity_arrows.next();
                } else {
                    p.velocity_arrows = p.velocity_arrows.prev();
                }
                info!("Velocity arrows: {}", p.velocity_arrows.label());
            }
            GraphicsControl::ActionDistribution => {
                let p = &mut self.highway_prefs;
                if delta > 0 {
                    p.action_distribution = p.action_distribution.next();
                } else {
                    p.action_distribution = p.action_distribution.prev();
                }
                info!("Action distribution: {}", p.action_distribution.label());
            }
            GraphicsControl::ActionDistributionText => {
                let p = &mut self.highway_prefs;
                if delta > 0 {
                    p.action_distribution_text = p.action_distribution_text.next();
                } else {
                    p.action_distribution_text = p.action_distribution_text.prev();
                }
                info!("Action dist text: {}", p.action_distribution_text.label());
            }
            GraphicsControl::NpcText => {
                let p = &mut self.highway_prefs;
                if delta > 0 {
                    p.npc_text = p.npc_text.next();
                } else {
                    p.npc_text = p.npc_text.prev();
                }
                info!("NPC text: {}", p.npc_text.label());
            }
            GraphicsControl::Attention => {
                let p = &mut self.highway_prefs;
                p.show_attention = !p.show_attention;
                info!("Attention: {}", if p.show_attention { "on" } else { "off" });
            }
            GraphicsControl::DebugEye => {
                let p = &mut self.highway_prefs;
                p.debug_eye = !p.debug_eye;
                info!("Debug eye: {}", if p.debug_eye { "on" } else { "off" });
            }
            GraphicsControl::LightBlendMode => {
                let p = &mut self.highway_prefs;
                let modes = LIGHT_BLEND_MODES;
                let cur = modes.iter().position(|&m| m == p.light_blend_mode).unwrap_or(0);
                let next = if delta > 0 {
                    (cur + 1) % modes.len()
                } else {
                    (cur + modes.len() - 1) % modes.len()
                };
                p.light_blend_mode = modes[next].to_string();
                info!("Light blend mode: {}", p.light_blend_mode);
            }
        }
    }
}
