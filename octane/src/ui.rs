//! UI rendering for the TUI.

mod dashboard;
mod explorer;
mod modals;
mod panels;
mod styles;
mod widgets;

use ratatui::prelude::*;
use ratatui::widgets::Paragraph;

use crate::app::{App, ExplorerMode, ExplorerPane, Screen};
use crate::render::color::hex_to_ratatui;
use styles::make_tab_bar;

use explorer::draw_explorer;
use modals::draw_modal;
use panels::{
    draw_epochs_panel, draw_episodes_panel, draw_highway_panel, draw_parquets_panel,
    draw_status_bar, draw_treks_panel,
};
use widgets::{draw_debug_overlay, draw_toasts};

/// Draw the complete UI.
pub fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();

    // Top-level layout: tab bar + tab content
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Tab bar
            Constraint::Min(3),   // Tab content
        ])
        .split(area);

    // Tab bar
    let ui = &app.config.octane.colors.ui;
    let tabs = make_tab_bar(app.screen, ui);
    frame.render_widget(tabs, outer[0]);

    let tab_area = outer[1];

    match app.screen {
        Screen::BehaviorExplorer => {
            draw_explorer_screen(frame, app, tab_area);
            return;
        }
        Screen::Browse => {}
    }

    // Runs tab: title bar + content area + status bar
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Title bar (trek path)
            Constraint::Min(3),   // Content
            Constraint::Length(1), // Status bar
        ])
        .split(tab_area);

    // Title bar: poshed trek path
    let title_text = app
        .trek_entries
        .get(app.selected_trek)
        .map(|e| e.display.clone())
        .unwrap_or_default();
    let title_bar = Paragraph::new(title_text)
        .alignment(Alignment::Center)
        .style(Style::default().fg(hex_to_ratatui(&ui.title_bar_fg)).bg(hex_to_ratatui(&ui.title_bar_bg)).add_modifier(Modifier::BOLD));
    frame.render_widget(title_bar, main_chunks[0]);

    // Two-column layout: left sidebar (epochs/episodes stacked) + highway
    let content_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(app.sidebar_width), // Sidebar (toggled with Shift+W)
            Constraint::Min(40),    // Highway (gets remaining space)
        ])
        .split(main_chunks[1]);

    // Sidebar: treks on top, parquets, epochs, episodes below
    let sidebar_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8), // Treks (compact)
            Constraint::Length(8), // Parquets
            Constraint::Min(5),    // Epochs (takes most space)
            Constraint::Length(6), // Episodes (small)
        ])
        .split(content_chunks[0]);

    draw_treks_panel(frame, app, sidebar_chunks[0]);
    draw_parquets_panel(frame, app, sidebar_chunks[1]);
    draw_epochs_panel(frame, app, sidebar_chunks[2]);
    draw_episodes_panel(frame, app, sidebar_chunks[3]);
    draw_highway_panel(frame, app, content_chunks[1]);
    draw_status_bar(frame, app, main_chunks[2]);

    // Debug overlay
    if app.show_debug {
        draw_debug_overlay(frame, area, app, &app.config.octane.colors.ui);
    }

    // Modal dialogs
    if let Some(modal) = app.active_modal {
        draw_modal(frame, area, modal, app, &app.config.octane.colors.ui);
    }

    // Toast notifications (drawn last, on top of everything)
    app.gc_toasts();
    if !app.toasts.is_empty() {
        draw_toasts(frame, area, &app.toasts, &app.config.octane.colors.ui);
    }
}

fn draw_explorer_screen(frame: &mut Frame, app: &mut App, area: Rect) {
    // Screen too small check
    if area.width < 80 || area.height < 20 {
        let msg = Paragraph::new("Terminal too small for behavior explorer")
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Yellow));
        frame.render_widget(msg, area);
        return;
    }

    // Loading state: show message while background load is still in progress
    if !app.explorer.loaded {
        let msg = Paragraph::new("Loading behaviors...")
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Yellow));
        let centered = Rect {
            x: area.x,
            y: area.y + area.height / 2,
            width: area.width,
            height: 1,
        };
        frame.render_widget(msg, centered);
        return;
    }

    // Clone color config to avoid borrow conflicts
    let status_fg = hex_to_ratatui(&app.config.octane.colors.ui.status_bar_text);
    let status_bg = hex_to_ratatui(&app.config.octane.colors.ui.status_bar_bg);

    // Content + status bar (tab bar is rendered by the caller)
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(1),
        ])
        .split(area);

    // Main content
    draw_explorer(frame, app, chunks[0]);

    // Mode-aware status bar (styled like the Runs tab footer)
    let ui = &app.config.octane.colors.ui;
    let key_style = Style::default().fg(hex_to_ratatui(&ui.status_bar_key)).bg(status_bg).bold();
    let text_style = Style::default().fg(status_fg).bg(status_bg);

    let spans: Vec<Span> = match app.explorer.mode {
        ExplorerMode::Edit => vec![
            Span::styled(" ", text_style),
            Span::styled("←/→", key_style), Span::styled(" Adjust  ", text_style),
            Span::styled("↑↓", key_style), Span::styled(" Fields  ", text_style),
            Span::styled("a", key_style), Span::styled(" Add NPC  ", text_style),
            Span::styled("x", key_style), Span::styled(" Remove NPC  ", text_style),
            Span::styled("Enter", key_style), Span::styled(" Save  ", text_style),
            Span::styled("Esc", key_style), Span::styled(" Cancel", text_style),
        ],
        ExplorerMode::NewBehavior => vec![
            Span::styled(" Type name, ", text_style),
            Span::styled("Enter", key_style), Span::styled(" Create  ", text_style),
            Span::styled("Esc", key_style), Span::styled(" Cancel", text_style),
        ],
        ExplorerMode::Browse => {
            let mut s = vec![Span::styled(" ", text_style)];
            match app.explorer.pane_focus {
                ExplorerPane::Behaviors => {
                    s.extend([
                        Span::styled("↑↓", key_style), Span::styled(" Navigate  ", text_style),
                        Span::styled("e", key_style), Span::styled(" Edit  ", text_style),
                        Span::styled("N", key_style), Span::styled(" New  ", text_style),
                        Span::styled("d", key_style), Span::styled(" Delete  ", text_style),
                    ]);
                }
                ExplorerPane::Scenarios => {
                    s.extend([
                        Span::styled("↑↓", key_style), Span::styled(" Navigate  ", text_style),
                        Span::styled("Enter", key_style), Span::styled(" Source  ", text_style),
                        Span::styled("e", key_style), Span::styled(" Edit  ", text_style),
                        Span::styled("n", key_style), Span::styled(" New  ", text_style),
                        Span::styled("c", key_style), Span::styled(" Clone  ", text_style),
                        Span::styled("d", key_style), Span::styled(" Delete  ", text_style),
                    ]);
                }
                ExplorerPane::Preview => {
                    s.extend([
                        Span::styled("e", key_style), Span::styled(" Edit  ", text_style),
                    ]);
                }
            }
            s.extend([
                Span::styled("Tab", key_style), Span::styled(" Next pane  ", text_style),
                Span::styled("r", key_style), Span::styled(" Graphics  ", text_style),
                Span::styled("1", key_style), Span::styled(" Runs  ", text_style),
                Span::styled("q", key_style), Span::styled(" Quit", text_style),
            ]);
            s
        }
    };

    let status_bar = Paragraph::new(Line::from(spans)).style(Style::default().bg(status_bg));
    frame.render_widget(status_bar, chunks[1]);

    // Modal dialogs
    if let Some(modal) = app.active_modal {
        let ui = &app.config.octane.colors.ui;
        draw_modal(frame, area, modal, app, ui);
    }

    // Toast notifications
    app.gc_toasts();
    if !app.toasts.is_empty() {
        let ui = &app.config.octane.colors.ui;
        draw_toasts(frame, area, &app.toasts, ui);
    }
}

#[cfg(test)]
mod tests {
    use crate::config::UiColorConfig;
    use crate::render::color::hex_to_ratatui;

    use super::styles::{focused_style, make_block, selected_style, unfocused_style};

    fn test_ui() -> UiColorConfig {
        UiColorConfig::default()
    }

    #[test]
    fn test_focused_style_is_bright_gray() {
        let ui = test_ui();
        let style = focused_style(&ui);
        assert_eq!(style.fg, Some(hex_to_ratatui(&ui.focused_border)));
    }

    #[test]
    fn test_unfocused_style_is_dark_gray() {
        let ui = test_ui();
        let style = unfocused_style(&ui);
        assert_eq!(style.fg, Some(hex_to_ratatui(&ui.unfocused_border)));
    }

    #[test]
    fn test_selected_style_has_background() {
        let ui = test_ui();
        let style = selected_style(&ui);
        assert_eq!(style.bg, Some(hex_to_ratatui(&ui.selected_bg)));
    }

    #[test]
    fn test_make_block_focused() {
        let ui = test_ui();
        let block = make_block("Test", true, None, &ui);
        // Block was created without panic - basic sanity check
        assert!(format!("{:?}", block).contains("Test"));
    }

    #[test]
    fn test_make_block_unfocused() {
        let ui = test_ui();
        let block = make_block("Test", false, None, &ui);
        assert!(format!("{:?}", block).contains("Test"));
    }

    #[test]
    fn test_make_block_with_mnemonic() {
        let ui = test_ui();
        let block = make_block("Epochs", true, Some('E'), &ui);
        let debug = format!("{:?}", block);
        assert!(debug.contains("Epochs") || debug.contains("pochs"));
    }
}
