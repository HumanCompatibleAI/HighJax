//! Small UI widgets: toast notification, debug overlay, seeker bar, velocity bar.

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, Paragraph};

use crate::app::{App, Toast};
use crate::config::UiColorConfig;
use crate::render::color::{hex_to_ratatui, speed_color_rgb};

/// Draw all active toast notifications, stacked from bottom up.
pub(crate) fn draw_toasts(frame: &mut Frame, area: Rect, toasts: &std::collections::VecDeque<Toast>, ui: &UiColorConfig) {
    let prefix_style = Style::default().fg(hex_to_ratatui(&ui.toast_prefix));
    let msg_style = Style::default();
    let border_style = Style::default().fg(hex_to_ratatui(&ui.toast_border));
    let bg_style = Style::default().bg(hex_to_ratatui(&ui.modal_bg));

    // Draw from bottom up, most recent at bottom
    let mut y_offset = 0u16;
    for toast in toasts.iter().rev() {
        let ellipsis_extra = if toast.ellipsis_at.is_some() { 1 } else { 0 };
        let display_len = toast.prefix.as_ref().map_or(0, |p| p.len())
            + toast.message.len() + ellipsis_extra;
        let width = (display_len as u16 + 4).min(area.width);
        let x = (area.width.saturating_sub(width)) / 2;
        let y = area.height.saturating_sub(3 + y_offset);
        let toast_area = Rect::new(x, y, width, 3);

        frame.render_widget(Clear, toast_area);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(border_style)
            .style(bg_style);

        let mut spans = vec![Span::raw(" ")];
        if let Some(ref prefix) = toast.prefix {
            spans.push(Span::styled(prefix.clone(), prefix_style));
        }
        if let Some(pos) = toast.ellipsis_at {
            let left = &toast.message[..pos];
            let right = &toast.message[pos..];
            spans.push(Span::styled(left.to_string(), msg_style));
            spans.push(Span::styled("\u{2026}", prefix_style)); // …
            spans.push(Span::styled(right.to_string(), msg_style));
        } else {
            spans.push(Span::styled(toast.message.clone(), msg_style));
        }

        let para = Paragraph::new(Line::from(spans))
            .block(block)
            .alignment(ratatui::layout::Alignment::Center);
        frame.render_widget(para, toast_area);

        y_offset += 3;
        if y_offset + 3 > area.height {
            break;
        }
    }
}

/// Draw the debug overlay.
pub(crate) fn draw_debug_overlay(frame: &mut Frame, area: Rect, app: &App, ui: &UiColorConfig) {
    let debug_width = 40;
    let debug_height = 13;
    // Position in top-right corner
    let x = area.width.saturating_sub(debug_width + 2);
    let y = 1;
    let debug_area = Rect::new(x, y, debug_width.min(area.width), debug_height.min(area.height));

    frame.render_widget(Clear, debug_area);

    // Indent each line for padding inside the border
    let debug_text: String = std::iter::once(String::new())
        .chain(app.debug_info_text().lines().map(|l| format!("  {l}")))
        .chain(std::iter::once(String::new()))
        .collect::<Vec<_>>()
        .join("\n");

    let block = Block::default()
        .title(" Debug ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(hex_to_ratatui(&ui.debug_border)))
        .style(Style::default().bg(hex_to_ratatui(&ui.modal_bg)));

    let para = Paragraph::new(debug_text).block(block);
    frame.render_widget(para, debug_area);
}

/// Build the seeker bar line as styled spans using sextant middle-row characters.
pub(crate) fn build_seeker_line(width: usize, frame_index: usize, frame_count: usize, ui: &UiColorConfig) -> Line<'static> {
    if width < 2 || frame_count == 0 {
        return Line::from("");
    }

    let max_index = frame_count.saturating_sub(1);
    // Number of played cells (including current position)
    let played = if max_index == 0 {
        width
    } else {
        let progress = frame_index as f64 / max_index as f64;
        ((progress * width as f64).round() as usize).clamp(1, width)
    };

    // Sextant middle row (positions 3+4 filled) = U+1FB0B
    let bar_char = '\u{1FB0B}';
    let played_style = Style::default().fg(hex_to_ratatui(&ui.seeker_played));
    let remaining_style = Style::default().fg(hex_to_ratatui(&ui.seeker_remaining));

    let mut spans = Vec::with_capacity(2);
    spans.push(Span::styled(
        bar_char.to_string().repeat(played),
        played_style,
    ));
    let remaining = width.saturating_sub(played);
    if remaining > 0 {
        spans.push(Span::styled(
            bar_char.to_string().repeat(remaining),
            remaining_style,
        ));
    }

    Line::from(spans)
}

/// Build a 10-character velocity bar using Unicode block characters.
///
/// Uses full block (█), left half (▌), left-quarter (▎), and left-3/4 (▊)
/// for sub-char resolution. Each character has 4 sub-positions, so 10 chars
/// give 40 steps of resolution.
///
/// Color ramps green → yellow → red where green=speed_min, yellow=midpoint,
/// red=speed_max. The bar fill maps speed_min..speed_max to 0..100%.
pub(crate) fn build_velocity_bar(speed: f64, speed_min: f64, speed_max: f64) -> Vec<Span<'static>> {
    const BAR_WIDTH: usize = 10;
    const STEPS_PER_CHAR: usize = 4;
    const TOTAL_STEPS: usize = BAR_WIDTH * STEPS_PER_CHAR;

    let range = speed_max - speed_min;
    let fraction = if range > 0.0 {
        ((speed - speed_min) / range).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let filled_steps = (fraction * TOTAL_STEPS as f64).round() as usize;
    let full_chars = filled_steps / STEPS_PER_CHAR;
    let remainder = filled_steps % STEPS_PER_CHAR;

    let mut bar = String::with_capacity(BAR_WIDTH * 4);

    // Full blocks
    for _ in 0..full_chars {
        bar.push('\u{2588}'); // █
    }

    // Fractional character
    let chars_used = if full_chars < BAR_WIDTH && remainder > 0 {
        match remainder {
            1 => bar.push('\u{258E}'), // ▎ (left 1/4)
            2 => bar.push('\u{258C}'), // ▌ (left 1/2)
            3 => bar.push('\u{258A}'), // ▊ (left 3/4)
            _ => {}
        }
        full_chars + 1
    } else {
        full_chars
    };

    // Pad with spaces
    for _ in chars_used..BAR_WIDTH {
        bar.push(' ');
    }

    let (r, g, b) = speed_color_rgb(fraction);

    let bg = Color::Rgb(0x22, 0x22, 0x22);
    let bar_color = Color::Rgb(r, g, b);

    let label = format!(" {:>5.1} m/s", speed);
    let bar_style = Style::default().fg(bar_color).bg(bg);
    let empty_style = Style::default().bg(bg);
    let label_style = Style::default().fg(bar_color);

    // Split bar into filled and empty parts for proper bg on empty portion
    let filled_len = chars_used;
    let empty_len = BAR_WIDTH - chars_used;
    let filled_str: String = bar.chars().take(filled_len).collect();
    let empty_str: String = " ".repeat(empty_len);

    vec![
        Span::styled(" ".to_string(), Style::default()),
        Span::styled(filled_str, bar_style),
        Span::styled(empty_str, empty_style),
        Span::styled(label, label_style),
    ]
}
