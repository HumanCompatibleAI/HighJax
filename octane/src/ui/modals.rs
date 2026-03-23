//! Modal dialog rendering (Help, Graphics, Jump, BehaviorScenario).

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, Paragraph};

use crate::app::{App, GraphicsControl, Modal, BehaviorScenarioField, ACTION_NAMES};
use crate::config::UiColorConfig;
use crate::render::color::hex_to_ratatui;

/// Build a styled footer line with hotkeys highlighted like the status bar.
fn footer_spans(parts: &[(&str, bool)], ui: &UiColorConfig) -> Line<'static> {
    let key_style = Style::default()
        .fg(hex_to_ratatui(&ui.status_bar_key))
        .bold();
    let text_style = Style::default()
        .fg(hex_to_ratatui(&ui.status_bar_text));
    let spans: Vec<Span<'static>> = parts
        .iter()
        .map(|(text, is_key)| {
            Span::styled(
                text.to_string(),
                if *is_key { key_style } else { text_style },
            )
        })
        .collect();
    Line::from(spans)
}

/// Draw a modal dialog.
pub(crate) fn draw_modal(frame: &mut Frame, area: Rect, modal: Modal, app: &App, ui: &UiColorConfig) {
    let (title, content, width, height): (&str, Text<'static>, u16, u16) = match modal {
        Modal::Help => {
            let help_text = r#"
  Highway TUI - Keybindings

  j/k  ←/→   Timestep back/forward
  J/K         5 timesteps back/forward
  h/l         First/last timestep
  ↑/↓         Navigate list
  Tab         Cycle focus
  t/a/o/e/s   Focus Treks/Parquets/Epochs/Episodes/Scene
  p           Play/pause
  Home/End    First/last item
  PgUp/PgDn   Jump ~80% of visible rows

  g  Epoch    G  Episode   /  Timestep
  r  Graphics

  b  Behavior scenario  B  Manage behaviors
  d  Debug overlay
  D  Copy debug      y  Copy state
  ^L Copy log path   W  Sidebar width
  ?  This help       q / Esc  Quit
"#;
            let content_lines = help_text.lines().count() as u16;
            (" Help ", Text::from(help_text.to_string()), 50, content_lines + 2)
        }
        Modal::Graphics => {
            let sel = app.graphics_selection;
            let arrow = |ctrl: GraphicsControl| if sel == ctrl { "►" } else { " " };
            let set = app.graphics_control_set();
            let has = |c: GraphicsControl| set.contains(&c);

            let p = &app.highway_prefs;
            let mut lines = String::from("\n");
            let mut n_items = 0u16;

            if has(GraphicsControl::Theme) {
                lines.push_str(&format!(
                    "  {} [e] Theme:         {:^12}\n", arrow(GraphicsControl::Theme),
                    format!("◄ {} ►", app.scene_theme.label())));
                n_items += 1;
            }
            if has(GraphicsControl::Fps) {
                lines.push_str(&format!(
                    "  {} [f] FPS:           {:^12}\n", arrow(GraphicsControl::Fps),
                    format!("◄ {} ►", app.playback.fps)));
                n_items += 1;
            }
            if has(GraphicsControl::PlaybackSpeed) {
                lines.push_str(&format!(
                    "  {} [s] Speed:         {:^12}\n", arrow(GraphicsControl::PlaybackSpeed),
                    format!("◄ {:.2}× ►", app.playback.speed)));
                n_items += 1;
            }
            if has(GraphicsControl::Zoom) {
                lines.push_str(&format!(
                    "  {} [z] Zoom:          {:^12}\n", arrow(GraphicsControl::Zoom),
                    format!("◄ {:.1}× ►", app.zoom)));
                n_items += 1;
            }
            if has(GraphicsControl::SidebarWidth) {
                lines.push_str(&format!(
                    "  {} [w] Sidebar width: {:^12}\n", arrow(GraphicsControl::SidebarWidth),
                    format!("◄ {} ►", app.sidebar_width)));
                n_items += 1;
            }
            if has(GraphicsControl::PodiumMarker) {
                lines.push_str(&format!(
                    "  {} [p] Podium marker: {:^12}\n", arrow(GraphicsControl::PodiumMarker),
                    if p.show_podium_marker { "◄ on ►" } else { "◄ off ►" }));
                n_items += 1;
            }
            if has(GraphicsControl::Scala) {
                lines.push_str(&format!(
                    "  {} [c] Scala:         {:^12}\n", arrow(GraphicsControl::Scala),
                    if p.show_scala { "◄ on ►" } else { "◄ off ►" }));
                n_items += 1;
            }
            if has(GraphicsControl::Sextants) {
                lines.push_str(&format!(
                    "  {} [x] Sextants:      {:^12}\n", arrow(GraphicsControl::Sextants),
                    if app.use_sextants { "◄ on ►" } else { "◄ off ►" }));
                n_items += 1;
            }
            if has(GraphicsControl::Octants) {
                lines.push_str(&format!(
                    "  {} [o] Octants:       {:^12}\n", arrow(GraphicsControl::Octants),
                    if app.use_octants { "◄ on ►" } else { "◄ off ►" }));
                n_items += 1;
            }
            if has(GraphicsControl::VelocityArrows) {
                lines.push_str(&format!(
                    "  {} [v] Vel. arrows:   {:^12}\n", arrow(GraphicsControl::VelocityArrows),
                    format!("◄ {} ►", p.velocity_arrows.label())));
                n_items += 1;
            }
            if has(GraphicsControl::ActionDistribution) {
                lines.push_str(&format!(
                    "  {} [i] Action dist:   {:^12}\n", arrow(GraphicsControl::ActionDistribution),
                    format!("◄ {} ►", p.action_distribution.label())));
                n_items += 1;
            }
            if has(GraphicsControl::ActionDistributionText) {
                lines.push_str(&format!(
                    "  {} [t] Act.dist text: {:^12}\n", arrow(GraphicsControl::ActionDistributionText),
                    format!("◄ {} ►", p.action_distribution_text.label())));
                n_items += 1;
            }
            if has(GraphicsControl::Attention) {
                lines.push_str(&format!(
                    "  {} [a] Attention:     {:^12}\n", arrow(GraphicsControl::Attention),
                    if p.show_attention { "◄ on ►" } else { "◄ off ►" }));
                n_items += 1;
            }
            if has(GraphicsControl::NpcText) {
                lines.push_str(&format!(
                    "  {} [n] NPC text:      {:^12}\n", arrow(GraphicsControl::NpcText),
                    format!("◄ {} ►", p.npc_text.label())));
                n_items += 1;
            }
            if has(GraphicsControl::DebugEye) {
                lines.push_str(&format!(
                    "  {} [d] Debug eye:     {:^12}\n", arrow(GraphicsControl::DebugEye),
                    if p.debug_eye { "◄ on ►" } else { "◄ off ►" }));
                n_items += 1;
            }
            if has(GraphicsControl::LightBlendMode) {
                lines.push_str(&format!(
                    "  {} [b] Light blend:   {:^14}\n", arrow(GraphicsControl::LightBlendMode),
                    format!("◄ {} ►", p.light_blend_mode)));
                n_items += 1;
            }

            let mut text_lines: Vec<Line<'static>> = lines
                .lines()
                .map(|l| Line::from(l.to_string()))
                .collect();
            text_lines.push(Line::from(""));
            text_lines.push(footer_spans(&[
                ("  ", false), ("←/→", true), (" adjust   ", false),
                ("Enter", true), (" close", false),
            ], ui));

            let height = n_items + 5; // items + header/footer padding
            (" Graphics ", Text::from(text_lines), 46, height as u16)
        }
        Modal::JumpEpoch => {
            let first_epoch = app.trek.epochs.first().map(|e| e.epoch_number).unwrap_or(0);
            let last_epoch = app.trek.epochs.last().map(|e| e.epoch_number).unwrap_or(0);
            let mut lines: Vec<Line<'static>> = vec![
                Line::from(""),
                Line::from(format!("  Enter epoch ({}-{}):", first_epoch, last_epoch)),
                Line::from(format!("  > {}_", app.jump_input)),
                Line::from(""),
                footer_spans(&[("  ", false), ("Enter", true), ("  Confirm", false)], ui),
                footer_spans(&[("  ", false), ("Esc", true), ("    Cancel", false)], ui),
            ];
            if let Some(v) = app.last_jump {
                lines.push(footer_spans(&[("  ", false), ("n", true), (&format!("    Repeat ({})", v), false)], ui));
            }
            (" Jump to Epoch ", Text::from(lines), 32, 11)
        }
        Modal::JumpEpisode => {
            let max = app
                .trek
                .epochs
                .get(app.selected_epoch)
                .map(|e| e.episodes.len().saturating_sub(1))
                .unwrap_or(0);
            let mut lines: Vec<Line<'static>> = vec![
                Line::from(""),
                Line::from(format!("  Enter episode (0-{}):", max)),
                Line::from(format!("  > {}_", app.jump_input)),
                Line::from(""),
                footer_spans(&[("  ", false), ("Enter", true), ("  Confirm", false)], ui),
                footer_spans(&[("  ", false), ("Esc", true), ("    Cancel", false)], ui),
            ];
            if let Some(v) = app.last_jump {
                lines.push(footer_spans(&[("  ", false), ("n", true), (&format!("    Repeat ({})", v), false)], ui));
            }
            (" Jump to Episode ", Text::from(lines), 32, 11)
        }
        Modal::JumpFrame => {
            let max = app.current_episode_frame_count().unwrap_or(1).saturating_sub(1);
            let mut lines: Vec<Line<'static>> = vec![
                Line::from(""),
                Line::from(format!("  Enter timestep (0-{}):", max)),
                Line::from(format!("  > {}_", app.jump_input)),
                Line::from(""),
                footer_spans(&[("  ", false), ("Enter", true), ("  Confirm", false)], ui),
                footer_spans(&[("  ", false), ("Esc", true), ("    Cancel", false)], ui),
            ];
            if let Some(v) = app.last_jump {
                lines.push(footer_spans(&[("  ", false), ("n", true), (&format!("    Repeat ({})", v), false)], ui));
            }
            (" Jump to Timestep ", Text::from(lines), 32, 11)
        }
        Modal::BehaviorScenario => {
            let action_dist = app.current_frame()
                .and_then(|f| f.vehicle_state.as_ref())
                .and_then(|vs| vs.action_distribution.as_ref());

            let trek_name = app.trek.path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "?".to_string());
            let epoch = app.trek.epochs.get(app.selected_epoch)
                .map(|e| e.epoch_number)
                .unwrap_or(0);

            let focus_actions = app.behavior_modal.focus == BehaviorScenarioField::Actions;
            let focus_name = app.behavior_modal.focus == BehaviorScenarioField::Name;

            let n_sub = app.effective_n_sub;
            let t_val = app.frame_index as f64 / n_sub as f64;
            let t_str = if n_sub > 1 {
                format!("{:.2}", t_val)
            } else {
                format!("{}", app.frame_index)
            };
            let mut lines = format!(
                "\n  Trek: {}\n  Epoch: {}  Episode: {}  t: {}\n\n  Actions:",
                trek_name, epoch, app.selected_episode, t_str
            );

            for (i, action_name) in ACTION_NAMES.iter().enumerate() {
                let w = app.behavior_modal.action_weights[i];
                let is_active = focus_actions && app.behavior_modal.action_cursor == i;
                let cursor = if is_active { "►" } else { " " };

                // Show input buffer when actively editing, otherwise show stored weight
                let weight_str = if is_active && !app.behavior_modal.action_input.is_empty() {
                    format!("{}_", app.behavior_modal.action_input)
                } else if is_active && w == 0.0 {
                    "_".to_string()
                } else if w == 0.0 {
                    " ".to_string()
                } else if w == w.round() {
                    format!("{}", w as i64)
                } else {
                    format!("{}", w)
                };

                let prob_str = action_dist
                    .and_then(|ad| ad.get(action_name))
                    .map(|p| format!("{:5.1}%", p * 100.0))
                    .unwrap_or_else(|| "    ?%".to_string());
                lines.push_str(&format!(
                    "\n  {} [{:>5}] {:<7} ({})", cursor, weight_str, action_name, prob_str
                ));
            }

            let name_cursor = if focus_name { "_" } else { "" };
            let name_marker = if focus_name { "►" } else { " " };

            lines.push_str(&format!(
                "\n\n  {} Behavior: [{}{}]",
                name_marker, app.behavior_modal.name, name_cursor,
            ));

            // Show name suggestions when Name field is focused
            let filtered = app.filtered_suggestions();
            let n_suggestions = if focus_name && !filtered.is_empty() {
                let max_show = 4;
                let show = filtered.len().min(max_show);
                for (i, name) in filtered.iter().take(max_show).enumerate() {
                    let marker = if app.behavior_modal.suggestion_cursor == Some(i) {
                        "►"
                    } else {
                        " "
                    };
                    lines.push_str(&format!("\n    {} {}", marker, name));
                }
                show
            } else {
                0
            };

            let mut text_lines: Vec<Line<'static>> = lines
                .lines()
                .map(|l| Line::from(l.to_string()))
                .collect();
            text_lines.push(Line::from(""));
            text_lines.push(footer_spans(&[
                ("  ", false), ("Enter", true), (" Save  ", false),
                ("Tab", true), (" Next  ", false),
                ("Space", true), (" Toggle", false),
            ], ui));
            text_lines.push(footer_spans(&[
                ("  Type weight (e.g. 1, -0.5)  ", false),
                ("Esc", true), (" Cancel", false),
            ], ui));

            let height = 19 + n_suggestions as u16;
            (" Add Behavior Scenario ", Text::from(text_lines), 52, height)
        }
    };

    let border_color = match modal {
        Modal::Help => hex_to_ratatui(&ui.modal_help_border),
        _ => hex_to_ratatui(&ui.modal_border),
    };

    // Position the modal: Graphics in top-right, others centered
    let (x, y) = if matches!(modal, Modal::Graphics) {
        // Top-right corner (with some padding)
        let x = area.width.saturating_sub(width + 2);
        let y = 1_u16;
        (x, y)
    } else {
        // Centered
        let x = (area.width.saturating_sub(width)) / 2;
        let y = (area.height.saturating_sub(height)) / 2;
        (x, y)
    };
    let modal_area = Rect::new(x, y, width.min(area.width), height.min(area.height));

    frame.render_widget(Clear, modal_area);

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color))
        .style(Style::default().bg(hex_to_ratatui(&ui.modal_bg)));

    let para = Paragraph::new(content).block(block);
    frame.render_widget(para, modal_area);
}
