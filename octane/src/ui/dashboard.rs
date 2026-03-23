//! Dashboard pane: unified fixed-layout info panel used across all tabs.
//!
//! Each data point is a "gauge" at a fixed position. Disabled gauges show
//! blank but still occupy space, keeping the layout identical across screens.
//!
//! Rows:
//!   0  Seeker bar (Runs only)
//!   1  Playback: play indicator, timestep, reward, crash, V, return, logP, adv, nadv (Runs only)
//!   2  Vehicle: velocity bar, action, heading, npc count (all screens)
//!   3  Location: epoch, e, t (all screens)

use ratatui::prelude::*;
use ratatui::widgets::Paragraph;

use crate::app::App;
use crate::config::UiColorConfig;
use crate::envs::EnvType;

use super::styles::{make_block, unfocused_style};
use super::widgets::{build_seeker_line, build_velocity_bar};

/// Style for disabled gauges: dim dark gray text.
fn disabled_style() -> Style {
    Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM)
}

/// Disabled placeholder for the vehicle line (row 2).
fn disabled_vehicle_line() -> Line<'static> {
    Line::styled(
        "  -------- ------  Action: -------  \u{279C}  ---.-\u{00B0}  -npc",
        disabled_style(),
    )
}

/// Which tab is hosting the dashboard, determining which gauges are active.
pub enum DashboardContext {
    Runs,
    Explorer,
}

/// Height of the dashboard pane in rows (including borders).
pub fn dashboard_height(_app: &App) -> u16 {
    6
}

/// Draw the dashboard pane.
pub(crate) fn draw_dashboard(
    frame: &mut Frame,
    app: &App,
    area: Rect,
    context: DashboardContext,
    ui: &UiColorConfig,
) {
    let block = make_block("dashboard", false, None, ui);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height == 0 || inner.width == 0 {
        return;
    }

    let inner_width = inner.width as usize;
    let mut lines: Vec<Line> = Vec::new();

    let dim = disabled_style();

    // Row 0: Seeker
    lines.push(match context {
        DashboardContext::Runs => {
            let frame_count = app.trek.epochs
                .get(app.selected_epoch)
                .and_then(|e| e.episodes.get(app.selected_episode))
                .map(|ep| ep.n_frames)
                .unwrap_or(0);
            build_seeker_line(inner_width, app.frame_index, frame_count, ui)
        }
        _ => Line::styled(
            format!(" {}", "\u{2500}".repeat(inner_width.saturating_sub(2))),
            dim,
        ),
    });

    // Row 1: Playback info
    lines.push(match context {
        DashboardContext::Runs => build_playback_line(app),
        _ => Line::styled(
            " \u{25A0}  Timestep -------/----  Reward:-------  Crash:-------  V:---------  Return:---------  LogP:-------  Adv:--------  NAdv:--------",
            dim,
        ),
    });

    // Row 2: Vehicle info
    lines.push(build_vehicle_line(app, &context));

    // Row 3: Location
    lines.push(build_location_line(app, &context, ui));

    lines.truncate(inner.height as usize);
    let paragraph = Paragraph::new(lines);
    frame.render_widget(paragraph, inner);
}

/// Row 1: playback gauges (Runs only).
fn build_playback_line(app: &App) -> Line<'static> {
    let playing_indicator = if app.playback.playing { "\u{25B6}\u{FE0E}" } else { "\u{25A0}" };

    let n_sub = app.effective_n_sub;
    let format_frame = |idx: usize| -> String {
        if n_sub > 1 {
            let policy_t = idx / n_sub;
            let fraction = (idx % n_sub) as f64 / n_sub as f64;
            format!("{}{}", policy_t, &format!("{:.2}", fraction)[1..])
        } else {
            format!("{}", idx)
        }
    };

    let frame_count = app.trek.epochs
        .get(app.selected_epoch)
        .and_then(|e| e.episodes.get(app.selected_episode))
        .map(|ep| ep.n_frames)
        .unwrap_or(0);
    let timestep_str = format_frame(app.frame_index);
    let max_str = format_frame(frame_count.saturating_sub(1));

    if let Some(current_frame) = app.current_frame() {
        let crash_str = current_frame.crash_reward
            .map(|cr| format!("  Crash:{:>7.1}", cr))
            .unwrap_or_default();
        let v_str = current_frame.v
            .map(|v| format!("  V:{:>9.4}", v))
            .unwrap_or_default();
        let return_str = current_frame.return_value
            .map(|r| format!("  Return:{:>9.2}", r))
            .unwrap_or_default();
        let logp_str = current_frame.tendency
            .map(|t| format!("  LogP:{:>7.3}", t))
            .unwrap_or_default();
        let adv_str = current_frame.advantage
            .map(|a| format!("  Adv:{:>8.3}", a))
            .unwrap_or_default();
        let nadv_str = current_frame.nz_advantage
            .map(|a| format!("  NAdv:{:>8.3}", a))
            .unwrap_or_default();
        Line::from(format!(
            " {}  Timestep {:>7}/{:<4}  Reward:{:>7.2}{}{}{}{}{}{}",
            playing_indicator, timestep_str, max_str,
            current_frame.reward,
            crash_str, v_str, return_str, logp_str, adv_str, nadv_str,
        ))
    } else {
        Line::from(format!(
            " {}  Timestep {:>7}/{:<4}",
            playing_indicator, timestep_str, max_str,
        ))
    }
}

/// Row 2: vehicle gauges (all screens, data source varies by context).
fn build_vehicle_line(app: &App, context: &DashboardContext) -> Line<'static> {
    let (ego_speed, speed_min, speed_max, action_str, heading_rad, n_npcs) = match context {
        DashboardContext::Runs => {
            let ego_speed = app.current_ego_speed().unwrap_or(0.0);
            let (speed_min, speed_max) = app.trek.ego_speed_range.unwrap_or((0.0, 30.0));
            let env_type = app.trek.env_type.unwrap_or(EnvType::Highway);
            let action = app.current_frame()
                .map(|f| f.action_name.as_deref()
                    .unwrap_or_else(|| env_type.action_name(f.action))
                    .to_string())
                .unwrap_or_else(|| "\u{2014}".to_string());
            let heading_rad = app.current_ego_heading().unwrap_or(0.0);
            let n_npcs = app.current_frame()
                .and_then(|f| f.vehicle_state.as_ref())
                .map(|vs| vs.npcs.len());
            (ego_speed, speed_min, speed_max, Some(action), heading_rad, n_npcs)
        }
        DashboardContext::Explorer => {
            use crate::app::EditorFieldKind;
            let (ego_speed, heading_rad, n_npcs) = if app.explorer.mode == crate::app::ExplorerMode::Edit {
                let speed = app.explorer.editor_fields.iter()
                    .find(|f| matches!(f.kind, EditorFieldKind::EgoSpeed))
                    .map(|f| f.value)
                    .unwrap_or(0.0);
                let npcs = app.explorer.editor_fields.iter()
                    .filter(|f| matches!(f.kind, EditorFieldKind::NpcRelX(_)))
                    .count();
                (speed, 0.0, npcs)
            } else {
                match app.explorer.scenarios.get(app.explorer.selected_scenario) {
                    Some(s) if s.has_state => (s.ego_speed, s.ego_heading, s.n_npcs),
                    _ => return disabled_vehicle_line(),
                }
            };
            let (speed_min, speed_max) = app.trek.ego_speed_range.unwrap_or((0.0, 30.0));
            (ego_speed, speed_min, speed_max, None, heading_rad, Some(n_npcs))
        }
    };

    let display_deg = -heading_rad.to_degrees();
    let (display_deg, arrow) = if display_deg > 0.1 {
        (display_deg, "\u{2197}")       // ↗
    } else if display_deg < -0.1 {
        (display_deg, "\u{2198}")       // ↘
    } else {
        (0.0, "\u{279C}")              // ➜
    };

    let mut spans = build_velocity_bar(ego_speed, speed_min, speed_max);
    if let Some(action) = action_str {
        spans.push(Span::raw(format!("  Action: {:<7}  ", action)));
    } else {
        spans.push(Span::raw("                    "));
    }
    spans.push(Span::styled(arrow.to_string(), Style::default().bold()));
    spans.push(Span::raw(format!("{:>6.1}\u{00B0}", display_deg)));
    if let Some(n) = n_npcs {
        spans.push(Span::raw(format!("  {}npc", n)));
    }
    Line::from(spans)
}

/// Row 3: location gauges (epoch/e/t for all screens).
fn build_location_line(app: &App, context: &DashboardContext, ui: &UiColorConfig) -> Line<'static> {
    let bright = Style::default().fg(Color::White);
    let normal = Style::default().fg(unfocused_style(ui).fg.unwrap_or(Color::Gray));
    let dim = disabled_style();

    match context {
        DashboardContext::Runs => {
            let epoch_num = app.trek.epochs
                .get(app.selected_epoch)
                .map(|e| e.epoch_number);
            let epoch_str = match epoch_num {
                Some(n) => format!("{:>3}", n),
                None => format!("{:>3}", "-"),
            };
            let policy_t = app.frame_index / app.effective_n_sub;
            Line::from(vec![
                Span::styled(format!(" Epoch {} | ", epoch_str), bright),
                Span::styled(
                    format!("e={:>3} t={:>3}", app.selected_episode, policy_t),
                    normal,
                ),
                Span::styled(" | Agent - | Rank - | KLD --------", dim),
            ])
        }
        DashboardContext::Explorer => {
            let scenario = app.explorer.scenarios.get(app.explorer.selected_scenario);
            let epoch_str = scenario
                .and_then(|s| s.source_epoch)
                .map(|n| format!("{:>3}", n))
                .unwrap_or_else(|| format!("{:>3}", "-"));
            let e_str = scenario
                .and_then(|s| s.source_episode)
                .map(|n| format!("{:>3}", n))
                .unwrap_or_else(|| format!("{:>3}", "-"));
            let t_str = scenario
                .and_then(|s| s.source_t)
                .map(|t| format!("{:>3}", t as i64))
                .unwrap_or_else(|| format!("{:>3}", "-"));
            Line::from(vec![
                Span::styled(format!(" Epoch {} | ", epoch_str), bright),
                Span::styled(format!("e={} t={}", e_str, t_str), normal),
                Span::styled(" | Agent - | Rank - | KLD --------", dim),
            ])
        }
    }
}
