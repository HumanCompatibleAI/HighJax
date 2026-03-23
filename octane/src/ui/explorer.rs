//! Behavior Explorer screen rendering.

use ansi_to_tui::IntoText;
use ratatui::prelude::*;
use ratatui::widgets::Paragraph;

use crate::app::{App, ExplorerMode, ExplorerPane};
use crate::config::UiColorConfig;
use crate::render::color::hex_to_ratatui;

use super::dashboard::{DashboardContext, dashboard_height, draw_dashboard};
use super::styles::{make_block, selected_style, unfocused_style, render_scrollbar};

pub(crate) fn draw_explorer(frame: &mut Frame, app: &mut App, area: Rect) {
    match app.explorer.mode {
        ExplorerMode::Browse => draw_explorer_browse(frame, app, area),
        ExplorerMode::Edit => draw_explorer_edit(frame, app, area),
        ExplorerMode::NewBehavior => draw_explorer_new_behavior(frame, app, area),
    }
}

fn draw_explorer_browse(frame: &mut Frame, app: &mut App, area: Rect) {
    // Two-column layout: left (behaviors + scenarios stacked) | right (preview + info)
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(36),
            Constraint::Min(30),
        ])
        .split(area);

    // Split left column vertically: behaviors (top) | scenarios (bottom)
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(40),
            Constraint::Percentage(60),
        ])
        .split(columns[0]);

    // Extract ui colors before mutable borrow for preview
    let ui = app.config.octane.colors.ui.clone();

    draw_behaviors_pane(frame, app, left[0], &ui);
    draw_scenarios_pane(frame, app, left[1], &ui);
    draw_preview_pane(frame, app, columns[1], &ui);
}

fn draw_explorer_edit(frame: &mut Frame, app: &mut App, area: Rect) {
    // Two-column layout: editor fields | (live preview + metrics)
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(35),
            Constraint::Min(30),
        ])
        .split(area);

    let ui = app.config.octane.colors.ui.clone();

    draw_editor_fields(frame, app, columns[0], &ui);

    // Split right column: preview + dashboard
    let db_height = dashboard_height(app);
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(db_height)])
        .split(columns[1]);

    draw_editor_preview(frame, app, right[0], &ui);
    draw_dashboard(frame, app, right[1], DashboardContext::Explorer, &ui);
}

fn draw_editor_fields(frame: &mut Frame, app: &App, area: Rect, ui: &UiColorConfig) {
    let block = make_block("editor", true, None, ui);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.explorer.editor_fields.is_empty() {
        let p = Paragraph::new("No fields")
            .style(Style::default().fg(unfocused_style(ui).fg.unwrap_or(Color::DarkGray)));
        frame.render_widget(p, inner);
        return;
    }

    let sel_style = selected_style(ui);
    let highlight_fg = hex_to_ratatui(&ui.focused_border);
    let dirty_indicator = if app.explorer.editor_dirty { " [modified]" } else { "" };

    let mut lines = Vec::new();
    lines.push(Line::styled(
        format!(" Scenario {}{}", app.explorer.selected_scenario, dirty_indicator),
        Style::default().fg(highlight_fg).add_modifier(Modifier::BOLD),
    ));
    lines.push(Line::raw(""));

    for (i, field) in app.explorer.editor_fields.iter().enumerate() {
        let arrow = if i == app.explorer.editor_selected_field { ">" } else { " " };
        let label = format!("{} {}: {:>8.1}", arrow, field.label, field.value);
        let style = if i == app.explorer.editor_selected_field { sel_style } else { Style::default() };
        lines.push(Line::styled(label, style));
    }

    lines.push(Line::raw(""));
    lines.push(Line::styled(
        " ←/→: adjust  a: add NPC",
        Style::default().fg(unfocused_style(ui).fg.unwrap_or(Color::DarkGray)),
    ));
    lines.push(Line::styled(
        " x: remove NPC  Enter: save",
        Style::default().fg(unfocused_style(ui).fg.unwrap_or(Color::DarkGray)),
    ));
    lines.push(Line::styled(
        " Esc: cancel",
        Style::default().fg(unfocused_style(ui).fg.unwrap_or(Color::DarkGray)),
    ));

    let paragraph = Paragraph::new(lines);
    frame.render_widget(paragraph, inner);
}

fn draw_editor_preview(frame: &mut Frame, app: &mut App, area: Rect, ui: &UiColorConfig) {
    let block = make_block("preview", true, None, ui);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let inner_cols = inner.width as u32;
    let inner_rows = inner.height as u32;

    if inner_cols < 4 || inner_rows < 2 {
        return;
    }

    if let Some(ansi_output) = app.render_editor_preview(inner_cols, inner_rows) {
        let styled_text = ansi_output.into_text()
            .unwrap_or_else(|_| Text::raw(&ansi_output));
        let para = Paragraph::new(styled_text);
        frame.render_widget(para, inner);
    } else {
        let p = Paragraph::new("Render failed")
            .style(Style::default().fg(unfocused_style(ui).fg.unwrap_or(Color::DarkGray)))
            .alignment(Alignment::Center);
        frame.render_widget(p, inner);
    }
}

fn draw_behaviors_pane(frame: &mut Frame, app: &mut App, area: Rect, ui: &UiColorConfig) {
    let focused = app.explorer.pane_focus == ExplorerPane::Behaviors;
    let block = make_block("behaviors", focused, Some('b'), ui);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.explorer.behaviors.is_empty() {
        let msg = Paragraph::new("No behaviors found.\nCapture with 'b' in\nbrowse mode, or\ncreate with 'N'.")
            .style(Style::default().fg(unfocused_style(ui).fg.unwrap_or(Color::DarkGray)));
        frame.render_widget(msg, inner);
        return;
    }

    let visible_rows = inner.height as usize;
    app.explorer.behaviors_visible_rows = Some(visible_rows);
    let scroll = app.explorer.behaviors_scroll;
    let sel = app.explorer.selected_behavior;
    let sel_style = selected_style(ui);

    let mut lines = Vec::new();
    for (i, behavior) in app.explorer.behaviors.iter().enumerate().skip(scroll).take(visible_rows) {
        let prefix = if behavior.is_preset { "* " } else { "  " };
        let label = format!("{}{} ({})", prefix, behavior.name, behavior.n_scenarios);
        let style = if i == sel { sel_style } else { Style::default() };
        lines.push(Line::styled(label, style));
    }

    let paragraph = Paragraph::new(lines);
    frame.render_widget(paragraph, inner);
    render_scrollbar(frame, area, app.explorer.behaviors.len(), visible_rows, scroll, focused, ui);
}

fn draw_scenarios_pane(frame: &mut Frame, app: &mut App, area: Rect, ui: &UiColorConfig) {
    let focused = app.explorer.pane_focus == ExplorerPane::Scenarios;
    let block = make_block("scenarios", focused, Some('s'), ui);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.explorer.scenarios.is_empty() {
        let msg = if app.explorer.behaviors.is_empty() {
            ""
        } else {
            "No scenarios"
        };
        let p = Paragraph::new(msg)
            .style(Style::default().fg(unfocused_style(ui).fg.unwrap_or(Color::DarkGray)));
        frame.render_widget(p, inner);
        return;
    }

    let visible_rows = inner.height as usize;
    app.explorer.scenarios_visible_rows = Some(visible_rows);
    let scroll = app.explorer.scenarios_scroll;
    let sel = app.explorer.selected_scenario;
    let sel_style = selected_style(ui);
    let dim_style = Style::default().fg(unfocused_style(ui).fg.unwrap_or(Color::DarkGray));

    // Header row
    let mut lines = Vec::new();
    lines.push(Line::styled(
        " # spd npc   L   I   R   F   S  E",
        dim_style,
    ));

    for (i, scenario) in app.explorer.scenarios.iter().enumerate().skip(scroll).take(visible_rows.saturating_sub(1)) {
        let aw = &scenario.action_weights;
        let fmt_w = |w: f64| -> String {
            if w == 0.0 { "  · ".to_string() }
            else if w == w.round() { format!("{:>3} ", w as i64) }
            else { format!("{:>4.1}", w) }
        };
        let edited = if scenario.edited { " *" } else { "  " };
        let label = format!(
            "{}{}.{:>4.0} {:>2} {}{}{}{}{}{}",
            if i == sel { ">" } else { " " },
            scenario.index,
            scenario.ego_speed,
            scenario.n_npcs,
            fmt_w(aw[0]), fmt_w(aw[1]), fmt_w(aw[2]), fmt_w(aw[3]), fmt_w(aw[4]),
            edited,
        );
        let style = if i == sel { sel_style } else { Style::default() };
        lines.push(Line::styled(label, style));
    }

    let paragraph = Paragraph::new(lines);
    frame.render_widget(paragraph, inner);
    render_scrollbar(frame, area, app.explorer.scenarios.len(), visible_rows, scroll, focused, ui);
}

fn draw_preview_pane(frame: &mut Frame, app: &mut App, area: Rect, ui: &UiColorConfig) {
    // Split into preview (top) + dashboard (bottom)
    let db_height = dashboard_height(app);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(db_height)])
        .split(area);

    draw_preview_scene(frame, app, chunks[0], ui);
    draw_dashboard(frame, app, chunks[1], DashboardContext::Explorer, ui);
}

fn draw_preview_scene(frame: &mut Frame, app: &mut App, area: Rect, ui: &UiColorConfig) {
    let focused = app.explorer.pane_focus == ExplorerPane::Preview;
    let block = make_block("preview", focused, Some('p'), ui);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let inner_cols = inner.width as u32;
    let inner_rows = inner.height as u32;

    if inner_cols < 4 || inner_rows < 2 {
        return;
    }

    // Check if we have a scenario to preview
    let has_state = app.explorer.scenarios
        .get(app.explorer.selected_scenario)
        .map(|s| s.has_state)
        .unwrap_or(false);

    if app.explorer.scenarios.is_empty() {
        let p = Paragraph::new("Select a scenario to preview")
            .style(Style::default().fg(unfocused_style(ui).fg.unwrap_or(Color::DarkGray)))
            .alignment(Alignment::Center);
        frame.render_widget(p, inner);
        return;
    }

    if !has_state {
        let p = Paragraph::new("Preview unavailable\n(observation-only scenario)")
            .style(Style::default().fg(unfocused_style(ui).fg.unwrap_or(Color::DarkGray)))
            .alignment(Alignment::Center);
        frame.render_widget(p, inner);
        return;
    }

    // Render the preview (uses cache when possible)
    if let Some(ansi_output) = app.render_explorer_preview(inner_cols, inner_rows) {
        let styled_text = ansi_output.into_text()
            .unwrap_or_else(|_| Text::raw(&ansi_output));
        let para = Paragraph::new(styled_text);
        frame.render_widget(para, inner);
    } else {
        let p = Paragraph::new("Render failed")
            .style(Style::default().fg(unfocused_style(ui).fg.unwrap_or(Color::DarkGray)))
            .alignment(Alignment::Center);
        frame.render_widget(p, inner);
    }
}

fn draw_explorer_new_behavior(frame: &mut Frame, app: &App, area: Rect) {
    let ui = &app.config.octane.colors.ui;
    let block = make_block("new behavior", true, None, ui);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let highlight_fg = hex_to_ratatui(&ui.focused_border);

    let mut lines = Vec::new();
    lines.push(Line::raw(""));
    lines.push(Line::styled(
        " Enter a name for the new behavior:",
        Style::default().fg(highlight_fg).add_modifier(Modifier::BOLD),
    ));
    lines.push(Line::raw(""));
    lines.push(Line::styled(
        format!(" > {}_", app.explorer.new_behavior_name),
        Style::default().fg(Color::White),
    ));
    lines.push(Line::raw(""));
    lines.push(Line::styled(
        " Enter: create  Esc: cancel",
        Style::default().fg(unfocused_style(ui).fg.unwrap_or(Color::DarkGray)),
    ));
    lines.push(Line::styled(
        " (letters, digits, dashes, underscores)",
        Style::default().fg(unfocused_style(ui).fg.unwrap_or(Color::DarkGray)),
    ));

    let paragraph = Paragraph::new(lines);
    frame.render_widget(paragraph, inner);
}
