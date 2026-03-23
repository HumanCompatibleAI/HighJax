//! Shared UI styles: focused/unfocused borders, selected rows, block builders, scrollbar.

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Scrollbar, ScrollbarOrientation, ScrollbarState, Tabs};

use crate::app::Screen;
use crate::config::UiColorConfig;
use crate::render::color::{hex_to_ratatui, scale_color};

/// Style for focused pane border.
pub(crate) fn focused_style(ui: &UiColorConfig) -> Style {
    Style::default().fg(hex_to_ratatui(&ui.focused_border)).bold()
}

/// Style for unfocused pane border.
pub(crate) fn unfocused_style(ui: &UiColorConfig) -> Style {
    Style::default().fg(hex_to_ratatui(&ui.unfocused_border))
}

/// Style for selected row.
pub(crate) fn selected_style(ui: &UiColorConfig) -> Style {
    Style::default().bg(hex_to_ratatui(&ui.selected_bg)).fg(hex_to_ratatui(&ui.selected_fg))
}

/// Create a block with appropriate focus styling and an optional mnemonic letter.
pub(crate) fn make_block<'a>(title: &'a str, focused: bool, mnemonic: Option<char>, ui: &UiColorConfig) -> Block<'a> {
    let border = if focused {
        focused_style(ui)
    } else {
        unfocused_style(ui)
    };

    let title_line = if let Some(ch) = mnemonic {
        let gold = Style::default()
            .fg(hex_to_ratatui(&ui.mnemonic))
            .add_modifier(Modifier::BOLD);
        if let Some(pos) = title.find(ch) {
            let before = &title[..pos];
            let letter = &title[pos..pos + ch.len_utf8()];
            let after = &title[pos + ch.len_utf8()..];
            Line::from(vec![
                Span::styled(before.to_string(), border),
                Span::styled(letter.to_string(), gold),
                Span::styled(after.to_string(), border),
            ])
        } else {
            Line::styled(title.to_string(), border)
        }
    } else {
        Line::styled(title.to_string(), border)
    };

    Block::default()
        .title(title_line)
        .borders(Borders::ALL)
        .border_style(border)
}

/// Render a vertical scrollbar on the right edge of `area` if content overflows.
///
/// `total_items` is the full list length, `visible_rows` is how many fit on screen,
/// `scroll_offset` is the current scroll position.
pub(crate) fn render_scrollbar(frame: &mut Frame, area: Rect, total_items: usize, visible_rows: usize, scroll_offset: usize, focused: bool, ui: &UiColorConfig) {
    if total_items <= visible_rows {
        return;
    }
    let border_color = if focused { &ui.focused_border } else { &ui.unfocused_border };
    let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
        .begin_symbol(None)
        .end_symbol(None)
        .track_symbol(Some("│"))
        .thumb_symbol("█")
        .track_style(Style::default().fg(scale_color(border_color, ui.scrollbar_track_multiplier)))
        .thumb_style(Style::default().fg(scale_color(border_color, ui.scrollbar_thumb_multiplier)));
    let mut state = ScrollbarState::new(total_items.saturating_sub(visible_rows))
        .position(scroll_offset);
    frame.render_stateful_widget(
        scrollbar,
        area.inner(Margin { vertical: 1, horizontal: 0 }),
        &mut state,
    );
}

/// Build the tab bar widget using ratatui's `Tabs`.
pub(crate) fn make_tab_bar<'a>(active: Screen, ui: &UiColorConfig) -> Tabs<'a> {
    let bg = hex_to_ratatui(&ui.title_bar_bg);
    let active_fg = hex_to_ratatui(&ui.title_bar_fg);
    let dim_fg = hex_to_ratatui(&ui.unfocused_border);
    let highlight_bg = hex_to_ratatui(&ui.selected_bg);

    let tab_titles = vec![
        Line::from(" 1 Runs "),
        Line::from(" 2 Behaviors "),
    ];

    let selected = match active {
        Screen::Browse => 0,
        Screen::BehaviorExplorer => 1,
    };

    Tabs::new(tab_titles)
        .select(selected)
        .style(Style::default().fg(dim_fg).bg(bg))
        .highlight_style(
            Style::default()
                .fg(active_fg)
                .bg(highlight_bg)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        )
        .divider("│")
        .padding("", "")
}
