//! Main panel drawing: treks, epochs, episodes, highway/scene, info pane, status bar.

use ansi_to_tui::IntoText;
use ratatui::prelude::*;
use ratatui::widgets::{Paragraph, Row, Table};
use std::time::Instant;
use tracing::trace;

use crate::app::{App, Focus};
use crate::render::color::hex_to_ratatui;
use crate::render::render_frame_with_svg_episode_or_fallback;

use super::dashboard::{DashboardContext, dashboard_height, draw_dashboard};
use super::styles::{make_block, render_scrollbar, selected_style};

/// Draw the treks panel (top of sidebar).
pub(crate) fn draw_treks_panel(frame: &mut Frame, app: &mut App, area: Rect) {
    let visible_rows = area.height.saturating_sub(2) as usize; // 2 borders

    // Update visible rows and adjust scroll (must precede ui borrow)
    app.update_treks_scroll_bounds(visible_rows);
    if app.selected_trek < app.treks_scroll {
        app.treks_scroll = app.selected_trek;
    } else if app.selected_trek >= app.treks_scroll + visible_rows {
        app.treks_scroll = app.selected_trek.saturating_sub(visible_rows.saturating_sub(1));
    }

    let ui = &app.config.octane.colors.ui;
    let focused = app.focus == Focus::Treks;
    let block = make_block("treks", focused, Some('t'), ui);

    let inner_width = area.width.saturating_sub(2) as usize;

    let lines: Vec<Line> = app
        .trek_entries
        .iter()
        .enumerate()
        .skip(app.treks_scroll)
        .take(visible_rows)
        .map(|(i, entry)| {
            let name = entry.display.clone();
            let truncated = if name.len() > inner_width {
                name[..inner_width].to_string()
            } else {
                name
            };
            let style = if i == app.selected_trek {
                selected_style(ui)
            } else {
                Style::default()
            };
            Line::styled(truncated, style)
        })
        .collect();

    let total_treks = app.trek_entries.len();
    let text = Text::from(lines);
    let para = Paragraph::new(text).block(block);
    frame.render_widget(para, area);
    render_scrollbar(frame, area, total_treks, visible_rows, app.treks_scroll, focused, ui);
}

/// Draw the parquets panel (between treks and epochs in sidebar).
pub(crate) fn draw_parquets_panel(frame: &mut Frame, app: &mut App, area: Rect) {
    let visible_rows = area.height.saturating_sub(2) as usize; // 2 borders

    app.update_parquets_scroll_bounds(visible_rows);

    let ui = &app.config.octane.colors.ui;
    let focused = app.focus == Focus::Parquets;
    let block = make_block("parquets", focused, Some('a'), ui);

    let inner_width = area.width.saturating_sub(2) as usize;

    let lines: Vec<Line> = app
        .parquet_sources
        .iter()
        .enumerate()
        .skip(app.parquets_scroll)
        .take(visible_rows)
        .map(|(i, source)| {
            let name = &source.display;
            let truncated = if name.len() > inner_width {
                name[..inner_width].to_string()
            } else {
                name.clone()
            };
            let style = if i == app.selected_parquet {
                selected_style(ui)
            } else {
                Style::default()
            };
            Line::styled(truncated, style)
        })
        .collect();

    let total = app.parquet_sources.len();
    let text = Text::from(lines);
    let para = Paragraph::new(text).block(block);
    frame.render_widget(para, area);
    render_scrollbar(frame, area, total, visible_rows, app.parquets_scroll, focused, ui);
}

pub(crate) fn draw_epochs_panel(frame: &mut Frame, app: &mut App, area: Rect) {
    // Calculate visible rows (area height minus borders and header)
    let visible_rows = area.height.saturating_sub(3) as usize; // 2 borders + 1 header

    // Update scroll bounds based on actual visible area (must precede ui borrow)
    app.update_epochs_scroll_bounds(visible_rows);

    let ui = &app.config.octane.colors.ui;
    let focused = app.focus == Focus::Epochs;
    let block = make_block("epochs", focused, Some('o'), ui);

    // Build rows for visible epochs only
    let rows: Vec<Row> = app
        .trek
        .epochs
        .iter()
        .enumerate()
        .skip(app.epochs_scroll)
        .take(visible_rows)
        .map(|(i, epoch)| {
            let style = if i == app.selected_epoch {
                selected_style(ui)
            } else {
                Style::default()
            };
            let has_epochia = epoch.epochia_nz_return.is_some();
            let mut cells = vec![
                format!("{:>3}", epoch.epoch_number),
                format!("{:>3}", epoch.episode_count()),
            ];
            if has_epochia {
                cells.push(match epoch.epochia_nz_return {
                    Some(r) => format!("{:>8.2}", r),
                    None => format!("{:>8}", "?"),
                });
                cells.push(match epoch.epochia_alive_fraction {
                    Some(f) => format!("{:>5.0}%", f * 100.0),
                    None => format!("{:>6}", "?"),
                });
            }
            Row::new(cells)
            .style(style)
        })
        .collect();

    // Check if any epoch has epochia data
    let has_epochia = app.trek.epochs.iter().any(|e| e.epochia_nz_return.is_some());

    let mut constraints = vec![
        Constraint::Length(4),  // #
        Constraint::Length(4),  // Eps
    ];
    let mut header_cells = vec!["#", "Eps"];
    if has_epochia {
        constraints.push(Constraint::Length(9));  // NReturn
        constraints.push(Constraint::Length(8));  // Survival
        header_cells.push("NReturn");
        header_cells.push("Survival");
    }

    let table = Table::new(rows, constraints)
    .header(
        Row::new(header_cells)
            .style(Style::default().bold())
            .bottom_margin(0),
    )
    .block(block);

    let total_epochs = app.trek.epochs.len();
    frame.render_widget(table, area);
    render_scrollbar(frame, area, total_epochs, visible_rows, app.epochs_scroll, focused, ui);
}

/// Draw the episodes panel (middle).
pub(crate) fn draw_episodes_panel(frame: &mut Frame, app: &mut App, area: Rect) {
    // Calculate visible rows (area height minus borders and header)
    let visible_rows = area.height.saturating_sub(3) as usize;

    // Update scroll bounds based on actual visible area (must precede ui borrow)
    app.update_episodes_scroll_bounds(visible_rows);

    let ui = &app.config.octane.colors.ui;
    let focused = app.focus == Focus::Episodes;
    let block = make_block("episodes", focused, Some('e'), ui);

    let epoch = app
        .trek
        .epochs
        .get(app.selected_epoch);
    let episodes = epoch
        .map(|e| &e.episodes[..])
        .unwrap_or(&[]);
    let n_ts_per_e = app.trek.n_ts_per_e;

    let rows: Vec<Row> = episodes
        .iter()
        .enumerate()
        .skip(app.episodes_scroll)
        .take(visible_rows)
        .map(|(i, ep)| {
            let style = if i == app.selected_episode {
                selected_style(ui)
            } else {
                Style::default()
            };
            let display_num = ep.es_episode.unwrap_or(i as i64);
            Row::new(vec![
                format!("{:>3}", display_num),
                format!("{:>4}", ep.n_frames),
                match ep.nz_return {
                    Some(r) => format!("{:>8.2}", r),
                    None => format!("{:>8}", "?"),
                },
                format!("{:>5.0}%", ep.n_alive_policy_frames as f64 / n_ts_per_e.max(1) as f64 * 100.0),
            ])
            .style(style)
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(4),  // #
            Constraint::Length(5),  // Steps
            Constraint::Length(9),  // NReturn
            Constraint::Length(8),  // Survival
        ],
    )
    .header(
        Row::new(vec!["#", "Steps", "NReturn", "Survival"])
            .style(Style::default().bold())
            .bottom_margin(0),
    )
    .block(block);

    let total_episodes = episodes.len();
    frame.render_widget(table, area);
    render_scrollbar(frame, area, total_episodes, visible_rows, app.episodes_scroll, focused, ui);
}

/// Draw the highway panel (right) - split into Scene and Dashboard panes.
pub(crate) fn draw_highway_panel(frame: &mut Frame, app: &App, area: Rect) {
    let db_height = dashboard_height(app);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(db_height)])
        .split(area);

    draw_scene_pane(frame, app, chunks[0]);
    let ui = &app.config.octane.colors.ui;
    draw_dashboard(frame, app, chunks[1], DashboardContext::Runs, ui);
}

/// Draw the Scene pane containing only the highway maize visualization.
fn draw_scene_pane(frame: &mut Frame, app: &App, area: Rect) {
    let total_start = Instant::now();
    let ui = &app.config.octane.colors.ui;
    let focused = app.focus == Focus::Highway;
    let block = make_block("scene", focused, Some('s'), ui);

    // Calculate inner dimensions (area minus borders)
    let inner_width = area.width.saturating_sub(2) as u32;
    let inner_height = area.height.saturating_sub(2) as u32;

    // Render the highway visualization using SvgEpisode
    let render_start = Instant::now();

    let highway_viz = if app.trek.env_type.is_none() {
        String::from("Unknown environment (no Octane support)")
    } else if let Some(ref svg_episode) = app.svg_episode {
        let fallback = String::from("[Mango render failed]");
        if let Some(render_config) = app.scene_render_config(inner_width, inner_height) {
            render_frame_with_svg_episode_or_fallback(
                svg_episode,
                app.playback.scene_time,
                &render_config,
                app.use_sextants,
                app.use_octants,
                &fallback,
            )
        } else {
            fallback
        }
    } else if let Some(ref err) = app.trek.load_error {
        format!("Failed to load episode data: {}", err)
    } else {
        String::from("No episode data")
    };
    let render_elapsed = render_start.elapsed();

    // Convert ANSI escape codes to ratatui styled Text
    let styled_text = highway_viz.into_text().unwrap_or_else(|_| Text::raw(&highway_viz));

    let para = Paragraph::new(styled_text).block(block);
    frame.render_widget(para, area);

    let total_elapsed = total_start.elapsed();
    if app.playback.playing {
        // Get vehicle heading for debug
        let heading = app.current_frame()
            .and_then(|f| f.vehicle_state.as_ref())
            .map(|s| s.ego.heading)
            .unwrap_or(0.0);
        trace!(
            "ANIM: draw_scene scene_time={:.4} heading={:.3} render_ms={:.1} total_ms={:.1}",
            app.playback.scene_time, heading,
            render_elapsed.as_secs_f64() * 1000.0,
            total_elapsed.as_secs_f64() * 1000.0
        );
    }
}

/// Draw the status bar at the bottom (lobby-style: amber keys, light gray text).
pub(crate) fn draw_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let ui = &app.config.octane.colors.ui;
    let bg = hex_to_ratatui(&ui.status_bar_bg);
    let key_style = Style::default().fg(hex_to_ratatui(&ui.status_bar_key)).bg(bg).bold();
    let text_style = Style::default().fg(hex_to_ratatui(&ui.status_bar_text)).bg(bg);

    let spans = vec![
        Span::styled(" ", text_style),
        Span::styled("j", key_style), Span::styled("/", text_style),
        Span::styled("k", key_style), Span::styled(" Timestep  ", text_style),
        Span::styled("h", key_style), Span::styled("/", text_style),
        Span::styled("l", key_style), Span::styled(" First/Last  ", text_style),
        Span::styled("p", key_style), Span::styled(" Play/Pause  ", text_style),
        Span::styled("g", key_style), Span::styled("/", text_style),
        Span::styled("G", key_style), Span::styled(" Jump  ", text_style),
        Span::styled("r", key_style), Span::styled(" Gfx  ", text_style),
        Span::styled("b", key_style), Span::styled(" Capture  ", text_style),
        Span::styled("2", key_style), Span::styled(" Behaviors  ", text_style),
        Span::styled("d", key_style), Span::styled(" Debug  ", text_style),
        Span::styled("C", key_style), Span::styled(" CopyCmd  ", text_style),
        Span::styled("D", key_style), Span::styled(" CopyDbg  ", text_style),
        Span::styled("^L", key_style), Span::styled(" Log  ", text_style),
        Span::styled("W", key_style), Span::styled(" Width  ", text_style),
        Span::styled("?", key_style), Span::styled(" Help  ", text_style),
        Span::styled("q", key_style), Span::styled(" Quit", text_style),
    ];

    let para = Paragraph::new(Line::from(spans)).style(Style::default().bg(bg));
    frame.render_widget(para, area);
}
