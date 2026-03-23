//! Main application state and event loop.

mod input;
mod navigation;
mod playback;
mod trek;
mod clipboard;
mod episode;
mod explorer;
mod explorer_editing;
mod modal_behavior;
mod modal_graphics;
mod zoltar;
mod modal_jump;
mod state;
#[cfg(test)]
mod tests;

pub use state::*;

use anyhow::Result;
use crossterm::event::{self, Event, KeyEventKind};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::collections::{HashMap, VecDeque};
use std::io::Stdout;
use std::path::PathBuf;
use std::time::Duration;
use tracing::{debug, info, trace};

use crate::config::Config;
use crate::data::{Episode, ParquetSource, Trek};
use crate::render::RenderConfig;
use crate::worlds::{SvgEpisode, ViewportEpisode};
use std::sync::mpsc::Receiver;

/// Main application state.
pub struct App {
    /// Which screen is active (Browse or BehaviorExplorer).
    pub screen: Screen,
    /// Central configuration (hot-reloadable).
    pub config: Config,
    /// Receiver for config hot-reload updates.
    pub config_rx: Option<Receiver<Config>>,
    /// The loaded trek data.
    pub trek: Trek,
    /// All available treks from ~/.highjax/t/.
    pub trek_entries: Vec<crate::data::TrekEntry>,
    /// Selected trek index in the treks list.
    pub selected_trek: usize,
    /// Scroll offset for treks list.
    pub treks_scroll: usize,
    /// Currently focused pane.
    pub focus: Focus,
    /// All discovered parquet sources for the current trek.
    pub parquet_sources: Vec<ParquetSource>,
    /// Selected parquet source index.
    pub selected_parquet: usize,
    /// Scroll offset for parquets list.
    pub parquets_scroll: usize,
    /// Selected epoch index.
    pub selected_epoch: usize,
    /// Selected episode index within the epoch.
    pub selected_episode: usize,
    /// Current frame index within the episode.
    pub frame_index: usize,
    /// Remembered frame positions per (epoch_index, episode_index).
    episode_frame_positions: HashMap<(usize, usize), usize>,
    /// Whether the app should quit.
    pub should_quit: bool,
    /// Playback and animation timing state.
    pub playback: PlaybackState,
    /// Scroll offset for epochs list.
    pub epochs_scroll: usize,
    /// Scroll offset for episodes list.
    pub episodes_scroll: usize,
    /// Visible rows in treks panel (set during rendering).
    treks_visible_rows: Option<usize>,
    /// Visible rows in parquets panel (set during rendering).
    parquets_visible_rows: Option<usize>,
    /// Visible rows in epochs panel (set during rendering).
    epochs_visible_rows: Option<usize>,
    /// Visible rows in episodes panel (set during rendering).
    episodes_visible_rows: Option<usize>,
    /// Currently loaded episode (cached for performance).
    pub current_episode: Option<Episode>,
    /// Epoch/episode indices for the cached episode.
    cached_episode_key: (usize, usize),
    /// Mango rendering configuration.
    pub render_config: RenderConfig,
    /// Active modal dialog, if any.
    pub active_modal: Option<Modal>,
    /// Zoom level (magnification, 1.0 = default).
    pub zoom: f64,
    /// Input buffer for jump dialogs.
    pub jump_input: String,
    /// Last successful jump value (for repeat).
    pub last_jump: Option<usize>,
    /// Whether to show debug overlay.
    pub show_debug: bool,
    /// Whether to use sextant characters (higher resolution but less compatible).
    pub use_sextants: bool,
    /// Whether to use octant characters (highest resolution, requires Unicode 16.0).
    pub use_octants: bool,
    /// Scene color theme (dark or light), interpreted by each env.
    pub scene_theme: crate::config::SceneTheme,
    /// Per-env preferences for Highway (preserved across trek switches).
    pub highway_prefs: crate::envs::highway::HighwayPrefs,
    /// Selected control in graphics dialog.
    pub graphics_selection: GraphicsControl,
    /// Viewport smoothing omega (spring constant).
    pub omega: f64,
    /// Pre-computed viewport episode (contains trajectory and scene data).
    pub viewport_episode: Option<ViewportEpisode>,
    /// Pre-computed SVG episode for rendering.
    pub svg_episode: Option<SvgEpisode>,
    /// Current scene dimensions (cols, rows) for SVG episode.
    scene_dims: (u32, u32),
    /// Toast message queue (rendered bottom-up, oldest at bottom).
    pub toasts: VecDeque<Toast>,
    /// Display text of the last toast pushed (survives expiry for 'c' to copy).
    pub last_toast_text: Option<String>,
    /// Sidebar width in columns (toggled with Shift+W).
    pub sidebar_width: u16,
    /// Saved per-trek selection state (epoch, episode, scroll positions).
    trek_view_states: HashMap<PathBuf, TrekViewState>,
    /// Behavior scenario capture modal state.
    pub behavior_modal: BehaviorModalState,
    /// Behavior Explorer tab state.
    pub explorer: ExplorerState,
    // -- Per-episode effective timing (derived from data, not meta.yaml) --

    /// Seconds per frame for the current episode.
    /// Equals `seconds_per_sub_t` when the parquet has sub-step rows,
    /// or `seconds_per_t` when it has only policy-step rows (lerp fills in).
    pub effective_seconds_per_frame: f64,
    /// Effective sub-steps-per-policy-step for the current episode.
    /// 1 when the parquet has only policy-step rows.
    pub effective_n_sub: usize,

    /// Deferred timestep from `--timestep` CLI or `navigate_to_scenario_source()`.
    /// Applied after episode load via binary search on frame `t` values.
    pub pending_timestep: Option<f64>,

    /// Zoltar IPC channels (None if disabled).
    pub zoltar_channels: Option<crate::zoltar::ZoltarChannels>,
}

impl App {
    /// Create a new App with the given trek (used in tests).
    #[cfg(test)]
    pub fn new(trek: Trek) -> Self {
        Self::with_config(trek, Config::default(), RenderConfig::default(), None)
    }

    /// Create a new App with custom render and central configuration.
    pub fn with_config(
        trek: Trek,
        config: Config,
        render_config: RenderConfig,
        config_rx: Option<Receiver<Config>>,
    ) -> Self {
        let playback_speed = config.octane.rendering.playback_speed;
        let fps = config.octane.rendering.fps;
        let use_sextants = config.octane.rendering.use_sextants;
        let use_octants = config.octane.rendering.use_octants;
        let scene_theme = config.octane.rendering.theme;
        let omega = config.octane.podium.omega;
        let highway_prefs = crate::envs::highway::HighwayPrefs::from_config(&config, scene_theme);

        // Discover all treks and find which one matches the loaded trek
        info!("Discovering available treks");
        let mut trek_entries = crate::data::discover_treks();
        let mut selected_trek = trek_entries
            .iter()
            .position(|e| e.path == trek.path);
        // If the loaded trek isn't in the discovered list (e.g. given via -t from
        // an external path), add it so it still appears in the treks pane.
        if selected_trek.is_none() {
            let display = crate::util::posh_path(&trek.path);
            trek_entries.push(crate::data::TrekEntry {
                path: trek.path.clone(),
                display,
            });
            selected_trek = Some(trek_entries.len() - 1);
        }
        let selected_trek = selected_trek.unwrap();
        let treks_scroll = selected_trek.saturating_sub(5);

        let parquet_sources = crate::data::discover_parquet_sources(&trek.path);
        info!("Discovered {} parquet sources", parquet_sources.len());

        let initial_epoch = trek.epochs.len().saturating_sub(1);

        let mut app = Self {
            screen: Screen::Browse,
            config,
            config_rx,
            trek,
            trek_entries,
            selected_trek,
            treks_scroll,
            focus: Focus::Epochs,
            parquet_sources,
            selected_parquet: 0,
            parquets_scroll: 0,
            selected_epoch: initial_epoch,
            selected_episode: 0,
            frame_index: 0,
            episode_frame_positions: HashMap::new(),
            should_quit: false,
            playback: PlaybackState {
                playing: false,
                fps,
                speed: playback_speed,
                start: None,
                start_scene_time: 0.0,
                scene_time: 0.0,
            },
            epochs_scroll: 0,
            episodes_scroll: 0,
            treks_visible_rows: None,
            parquets_visible_rows: None,
            epochs_visible_rows: None,
            episodes_visible_rows: None,
            current_episode: None,
            cached_episode_key: (usize::MAX, usize::MAX),
            render_config,
            active_modal: None,
            zoom: 1.0,
            jump_input: String::new(),
            last_jump: None,
            show_debug: false,
            use_sextants,
            use_octants,
            scene_theme,
            highway_prefs,
            graphics_selection: GraphicsControl::Fps,
            omega,
            viewport_episode: None,
            svg_episode: None,
            scene_dims: (0, 0),
            toasts: VecDeque::new(),
            last_toast_text: None,
            sidebar_width: 46,
            trek_view_states: HashMap::new(),
            behavior_modal: BehaviorModalState {
                action_weights: [0.0; 5],
                action_input: String::new(),
                name: String::new(),
                focus: BehaviorScenarioField::Actions,
                action_cursor: 0,
                suggestions: Vec::new(),
                suggestion_cursor: None,
            },
            explorer: ExplorerState {
                behaviors: Vec::new(),
                selected_behavior: 0,
                behaviors_scroll: 0,
                behaviors_visible_rows: None,
                pane_focus: ExplorerPane::Behaviors,
                scenarios: Vec::new(),
                selected_scenario: 0,
                scenarios_scroll: 0,
                scenarios_visible_rows: None,
                preview_cache: None,
                preview_dims: (0, 0),
                mode: ExplorerMode::Browse,
                editor_fields: Vec::new(),
                editor_selected_field: 0,
                editor_dirty: false,
                delete_pending: false,
                npc_remove_pending: false,
                new_behavior_name: String::new(),
                load_rx: None,
                loaded: false,
                action_probs_cache: std::collections::HashMap::new(),
                action_probs_rx: None,
            },
            effective_seconds_per_frame: 0.0, // set below
            effective_n_sub: 0,               // set below
            pending_timestep: None,
            zoltar_channels: None,
        };
        app.reset_effective_timing();
        app.adjust_epochs_scroll();
        // Try to load initial episode
        app.ensure_episode_loaded();
        // Start background behavior loading immediately
        app.spawn_behavior_load();
        app
    }
}

/// Initialize the terminal for TUI mode.
pub fn init_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(
        stdout,
        crossterm::terminal::EnterAlternateScreen,
        crossterm::event::EnableMouseCapture,
    )?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;
    let (cols, rows) = crossterm::terminal::size().unwrap_or((0, 0));
    info!("Terminal initialized: {}x{}", cols, rows);
    Ok(terminal)
}

/// Restore the terminal to normal mode.
pub fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    crossterm::terminal::disable_raw_mode()?;
    crossterm::execute!(
        terminal.backend_mut(),
        crossterm::terminal::LeaveAlternateScreen,
        crossterm::event::DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    info!("Terminal restored");
    Ok(())
}

/// Run the main event loop.
pub fn run(terminal: &mut Terminal<CrosstermBackend<Stdout>>, app: &mut App) -> Result<()> {
    use std::time::Instant;

    let mut key_press_time: Option<Instant> = None;
    let mut last_dims: (u32, u32) = (0, 0);
    let mut loop_iteration: u64 = 0;
    let mut last_loop_start = Instant::now();

    loop {
        let loop_start = Instant::now();
        let loop_delta = loop_start.duration_since(last_loop_start);
        loop_iteration += 1;

        // Recalculate tick_rate each iteration so FPS changes take effect immediately
        let tick_rate = Duration::from_millis(1000 / app.playback.fps as u64);

        if app.playback.playing {
            trace!(
                "ANIM: loop_start iter={} delta_ms={:.1} tick_rate_ms={}",
                loop_iteration, loop_delta.as_secs_f64() * 1000.0, tick_rate.as_millis()
            );
        }

        // Check for config hot-reload
        app.check_config_updates();

        // Lazy load episode data only when needed for rendering (Runs tab only)
        let load_start = Instant::now();
        let old_episode_key = app.cached_episode_key;
        if app.screen == Screen::Browse {
            app.ensure_episode_loaded();
        }
        let episode_changed = app.cached_episode_key != old_episode_key;
        let load_elapsed = load_start.elapsed();

        // Get render dimensions from terminal (exact scene pane inner size)
        // Layout: sidebar + scene borders (2) cols subtracted
        //         status bar (1) + info pane (2) + scene borders (2) = 5 rows
        let (cols, rows) = crossterm::terminal::size().unwrap_or((80, 24));
        let sidebar_overhead = app.sidebar_width + 2; // sidebar + scene borders
        let scene_cols = cols.saturating_sub(sidebar_overhead).max(20) as u32;
        let scene_rows = rows.saturating_sub(5).max(10) as u32;
        let current_dims = (scene_cols, scene_rows);
        let dims_changed = current_dims != last_dims;

        // Compute viewport and SVG episodes on dimension, episode, or zoom change
        let viewport_invalidated = app.svg_episode.is_none();
        if dims_changed || episode_changed || viewport_invalidated {
            app.compute_episode_viewport_with_dims(scene_cols, scene_rows);
            last_dims = current_dims;
        }

        // Advance frame if playing
        let tick_start = Instant::now();
        app.tick();
        let tick_elapsed = tick_start.elapsed();

        // Sync scene_time to exact current instant right before draw
        let sync_start = Instant::now();
        app.sync_scene_time();
        let sync_elapsed = sync_start.elapsed();

        // Draw UI
        let draw_start = Instant::now();
        terminal.draw(|frame| {
            crate::ui::draw(frame, &mut *app);
        })?;
        let draw_elapsed = draw_start.elapsed();

        // Check if background behavior load has completed.
        app.check_behavior_load();

        // Check if background action probs fetch has completed.
        app.check_action_probs_load();

        // Process zoltar queries (non-blocking).
        // Take channels temporarily to avoid borrow conflict with &mut self handlers.
        if let Some(channels) = app.zoltar_channels.take() {
            while let Ok(envelope) = channels.request_rx.try_recv() {
                let response = app.handle_zoltar_request(&envelope.request);
                let _ = envelope.response_tx.send(response);
            }
            app.zoltar_channels = Some(channels);
        }

        if app.playback.playing {
            trace!(
                "ANIM: loop_work iter={} load_ms={:.1} tick_ms={:.1} sync_ms={:.1} draw_ms={:.1}",
                loop_iteration,
                load_elapsed.as_secs_f64() * 1000.0,
                tick_elapsed.as_secs_f64() * 1000.0,
                sync_elapsed.as_secs_f64() * 1000.0,
                draw_elapsed.as_secs_f64() * 1000.0,
            );
        }

        // Log timing if we just processed a keypress
        if let Some(key_time) = key_press_time.take() {
            let total = key_time.elapsed();
            debug!(
                "Keypress->render: total={:?} (load={:?}, draw={:?})",
                total, load_elapsed, draw_elapsed
            );
        }

        // Handle events with timeout for playback
        // Compensate for work time: wait only remaining time to hit tick_rate
        let work_elapsed = loop_start.elapsed();
        let poll_timeout = tick_rate.saturating_sub(work_elapsed);
        let event_poll_start = Instant::now();
        let had_event = event::poll(poll_timeout)?;
        let event_poll_elapsed = event_poll_start.elapsed();

        if app.playback.playing {
            trace!(
                "ANIM: event_poll iter={} work_ms={:.1} poll_timeout_ms={:.1} poll_actual_ms={:.1} had_event={}",
                loop_iteration,
                work_elapsed.as_secs_f64() * 1000.0,
                poll_timeout.as_secs_f64() * 1000.0,
                event_poll_elapsed.as_secs_f64() * 1000.0,
                had_event
            );
        }

        if had_event {
            if let Event::Key(key) = event::read()? {
                // Only handle key press events, ignore repeats and releases
                if key.kind == KeyEventKind::Press {
                    key_press_time = Some(Instant::now());
                    app.handle_key(key);
                }
            }
        }

        last_loop_start = loop_start;

        if app.should_quit {
            break;
        }
    }

    Ok(())
}
