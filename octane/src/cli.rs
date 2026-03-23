//! CLI entry point, argument parsing, and command dispatch.

mod args;
mod commands;

use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use std::path::PathBuf;
use tracing::{info, warn};

use args::{Cli, Commands, ConfigCommands, MangoCommands};
use commands::{run_animate, run_draw};
use commands::run_draw_behavior;
use commands::run_zoltar;

use crate::app::App;
use crate::config::Config;
use crate::data::Trek;
use crate::render::RenderConfig;
use crate::util::{find_parquet_file, posh_path};

/// Find the most recent trek in ~/.highjax/t/.
/// Treks are directories with timestamp names (e.g., 2026-01-31-02-13-44-322216).
/// Selects the newest by directory name.
fn find_latest_trek() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let home = PathBuf::from(home);

    let highjax_dir = if let Ok(highjax_home) = std::env::var("HIGHJAX_HOME") {
        PathBuf::from(highjax_home).join("t")
    } else {
        home.join(".highjax").join("t")
    };

    // Filter to actual trek directories. Check name first (no I/O) to avoid
    // slow stat calls over network mounts.
    let is_trek_dir = |entry: &fs::DirEntry| -> bool {
        // Use file_type() which avoids extra stat on Linux (uses d_type from readdir)
        if !entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
            return false;
        }
        // Name starts with digit (timestamp format like 2024-09-17-...) — no I/O needed
        if entry.file_name()
            .to_str()
            .map(|s| s.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false))
            .unwrap_or(false)
        {
            return true;
        }
        // Fall back to checking for trek markers (requires stat calls)
        let path = entry.path();
        path.join("meta.yaml").exists() || find_parquet_file(&path, "sample_es").is_some()
    };

    let mut candidates: Vec<_> = Vec::new();
    for dir in [&highjax_dir] {
        if dir.is_dir() {
            if let Ok(entries) = fs::read_dir(dir) {
                candidates.extend(
                    entries
                        .filter_map(|e| e.ok())
                        .filter(|e| is_trek_dir(e))
                );
            }
        }
    }

    candidates
        .into_iter()
        .max_by(|a, b| a.file_name().cmp(&b.file_name()))
        .map(|e| e.path())
}

/// Run mango subcommands.
fn run_mango_command(command: MangoCommands, use_sextants: bool, use_octants: bool) -> Result<()> {
    use crate::mango;

    match command {
        MangoCommands::Benchmark { iterations } => {
            mango::run_benchmark(iterations, use_sextants, use_octants);
            Ok(())
        }
        MangoCommands::Svg { path, cols, rows } => {
            let (term_cols, term_rows) = mango::get_terminal_size();
            let cols = cols.unwrap_or(term_cols);
            let rows = rows.unwrap_or(term_rows.saturating_sub(1)); // Leave room for prompt
            mango::render_file(&path, cols, rows, use_sextants, use_octants)
                .map_err(|e| anyhow::anyhow!("{}", e))
        }
        MangoCommands::Image { path, cols, rows } => {
            let (term_cols, term_rows) = mango::get_terminal_size();
            let cols = cols.unwrap_or(term_cols);
            let rows = rows.unwrap_or(term_rows.saturating_sub(1));
            mango::render_file(&path, cols, rows, use_sextants, use_octants)
                .map_err(|e| anyhow::anyhow!("{}", e))
        }
        MangoCommands::Chars => {
            mango::show_chars();
            Ok(())
        }
    }
}

/// Run config subcommands.
fn run_config_command(command: ConfigCommands) -> Result<()> {
    match command {
        ConfigCommands::FillDefaults => {
            Config::fill_defaults()?;
            Ok(())
        }
        ConfigCommands::Path => {
            println!("{}", Config::path().display());
            Ok(())
        }
    }
}

/// Given a path to a parquet file (.parquet/.pq), walk up directories until
/// we find one containing meta.yaml (a trek directory). Returns the trek path.
fn find_trek_from_parquet(parquet_path: &std::path::Path) -> Result<PathBuf> {
    let canonical = parquet_path.canonicalize()
        .with_context(|| format!("Cannot resolve path: {}", parquet_path.display()))?;
    let mut dir = canonical.parent();
    while let Some(d) = dir {
        if d.join("meta.yaml").exists() {
            return Ok(d.to_path_buf());
        }
        dir = d.parent();
    }
    anyhow::bail!(
        "Could not find trek (no meta.yaml in any parent of {})",
        posh_path(&canonical),
    )
}

/// Resolve a parquet file path to a parquet source index.
///
/// Tries two strategies in order:
/// 1. Full path: canonicalize the path and match against known sources.
/// 2. Substring match: case-insensitive substring against each source's relative path.
fn resolve_parquet_source(
    query: &str,
    sources: &[crate::data::ParquetSource],
    trek_path: &std::path::Path,
) -> std::result::Result<usize, String> {
    // Strategy 1: full path resolution
    let query_path = PathBuf::from(query);
    if let Ok(canonical) = query_path.canonicalize() {
        if let Ok(trek_canonical) = trek_path.canonicalize() {
            if canonical.starts_with(&trek_canonical) {
                if let Some(idx) = sources.iter().position(|s| {
                    s.path.canonicalize().ok().as_ref() == Some(&canonical)
                }) {
                    return Ok(idx);
                }
                // Path is inside trek but not a known source — try matching parent dir
                // (user might pass the run folder instead of the es.parquet inside it)
                let with_es = canonical.join("es.parquet");
                if let Some(idx) = sources.iter().position(|s| {
                    s.path.canonicalize().ok().as_ref() == Some(&with_es)
                }) {
                    return Ok(idx);
                }
                // Backward compat: try .pq extension too
                let with_es_pq = canonical.join("es.pq");
                if let Some(idx) = sources.iter().position(|s| {
                    s.path.canonicalize().ok().as_ref() == Some(&with_es_pq)
                }) {
                    return Ok(idx);
                }
                return Err(format!(
                    "Path '{}' is inside the trek but doesn't match any parquet source",
                    query,
                ));
            }
        }
    }

    // Strategy 2: case-insensitive substring against relative_path
    let lower = query.to_lowercase();
    let matches: Vec<usize> = sources
        .iter()
        .enumerate()
        .filter(|(_, s)| s.relative_path.to_lowercase().contains(&lower))
        .map(|(i, _)| i)
        .collect();

    match matches.len() {
        1 => Ok(matches[0]),
        0 => Err(format!(
            "No parquet source matching '{}' (have: {})",
            query,
            sources.iter().map(|s| s.display.as_str()).collect::<Vec<_>>().join(", "),
        )),
        _ => Err(format!(
            "Ambiguous parquet source '{}' — matches {} sources: {}",
            query,
            matches.len(),
            matches.iter()
                .map(|&i| sources[i].display.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        )),
    }
}

/// Resolve --epoch (number) to epoch array index. Defaults to first epoch.
fn resolve_epoch_idx(trek: &Trek, epoch_num: Option<i64>) -> Result<usize> {
    match epoch_num {
        None => Ok(0),
        Some(num) => trek.epochs.iter()
            .position(|e| e.epoch_number == num)
            .ok_or_else(|| anyhow::anyhow!("Epoch {} not found in trek", num)),
    }
}

/// Resolve --episode (es_episode number) to episode array index within epoch. Defaults to 0.
fn resolve_episode_idx(trek: &Trek, epoch_idx: usize, episode_num: Option<usize>) -> Result<usize> {
    let epoch = trek.epochs.get(epoch_idx)
        .ok_or_else(|| anyhow::anyhow!("Epoch index {} out of range", epoch_idx))?;
    match episode_num {
        None => Ok(0),
        Some(num) => epoch.episodes.iter()
            .position(|em| em.es_episode == Some(num as i64))
            .ok_or_else(|| anyhow::anyhow!("Episode {} not found in epoch {}", num, epoch.epoch_number)),
    }
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();

    // Check for log level env var
    let verbose = cli.verbose || std::env::var("ASPHALT_TUI_LOG").is_ok();

    // Initialize logging
    let command_name = match &cli.command {
        Some(Commands::Mango { .. }) => "mango",
        Some(Commands::Config { .. }) => "config",
        Some(Commands::Zoltar { .. }) => "zoltar",
        Some(Commands::Draw { .. }) => "draw",
        Some(Commands::Animate { .. }) => "animate",
        None => "browse",
    };
    let log_config = leona::LogConfig::new("octane")
        .verbose(verbose)
        .command(command_name)
        .version(format!(
            "{} (built {})",
            env!("OCTANE_GIT_HASH"),
            env!("OCTANE_BUILD_TIME"),
        ));
    let log_path = leona::init(log_config)?;
    leona::install_panic_hook();

    info!("Starting octane");
    info!("Log file: {}", posh_path(&log_path));

    // Handle mango commands early (don't need trek)
    if let Some(Commands::Mango { command }) = cli.command {
        // Load config for defaults; CLI flags override
        let config = Config::load();
        let use_sextants = if cli.sextants { true } else if cli.no_sextants { false } else { config.octane.rendering.use_sextants };
        let use_octants = if cli.octants { true } else if cli.no_octants { false } else { config.octane.rendering.use_octants };
        return run_mango_command(command, use_sextants, use_octants);
    }

    // Handle config commands early (don't need trek)
    if let Some(Commands::Config { command }) = cli.command {
        return run_config_command(command);
    }

    // Handle zoltar client commands early (connects to running instance, no trek needed)
    if let Some(Commands::Zoltar { pid, command }) = cli.command {
        return run_zoltar(pid, command);
    }

    // Handle behavior draw early (doesn't need a trek)
    if let Some(Commands::Draw {
        behavior: Some(ref behavior_name),
        scenario, svg, png, cols, rows, ref hardcoded_action, ..
    }) = cli.command {
        let mut config = Config::load();
        if let Some(theme) = cli.theme {
            config.octane.rendering.theme = theme;
        }
        let zoom = cli.zoom.unwrap_or(1.0);
        let use_sextants = if cli.sextants { true } else if cli.no_sextants { false } else { config.octane.rendering.use_sextants };
        let use_octants = if cli.octants { true } else if cli.no_octants { false } else { config.octane.rendering.use_octants };
        return run_draw_behavior(
            &config, behavior_name, scenario,
            cols, rows, svg, png, zoom, use_sextants, use_octants,
            cli.prefs.as_deref(), hardcoded_action.as_deref(),
        );
    }

    // Load central config from ~/.highjax/config.json
    let mut config = Config::load();

    // Apply --theme CLI override to config (used by draw/animate before App exists)
    if let Some(theme) = cli.theme {
        config.octane.rendering.theme = theme;
    }

    // Determine trek path and optional parquet override from --target
    let (trek_path, parquet_override) = if let Some(ref target) = cli.target {
        let target_path = PathBuf::from(target);
        if target.ends_with(".parquet")
            || target.ends_with(".pq")
            || (target_path.exists() && target_path.is_file())
        {
            // Target is a parquet file — find containing trek
            let trek = find_trek_from_parquet(&target_path)?;
            info!("Target is parquet file, trek: {}", posh_path(&trek));
            let canonical = target_path.canonicalize()
                .with_context(|| format!("Cannot resolve path: {}", target_path.display()))?;
            (trek.to_string_lossy().to_string(), Some(canonical))
        } else {
            // Target is a trek directory
            (target.clone(), None)
        }
    } else {
        match find_latest_trek() {
            Some(p) => {
                info!("No target specified, using latest: {}", posh_path(&p));
                (p.to_string_lossy().to_string(), None)
            }
            None => {
                eprintln!("No target provided and no treks found in ~/.highjax/t/");
                eprintln!("Use -t <path> to specify a trek directory or parquet file.");
                std::process::exit(1);
            }
        }
    };

    info!("Trek path: {}", posh_path(&PathBuf::from(&trek_path)));
    let trek = Trek::load(PathBuf::from(&trek_path))?;
    info!(
        "Loaded {} epochs, {} total episodes",
        trek.epoch_count(),
        trek.episode_count()
    );

    // Handle subcommands
    match cli.command {
        Some(Commands::Mango { .. }) | Some(Commands::Config { .. }) | Some(Commands::Zoltar { .. }) => {
            unreachable!("Handled above")
        }
        Some(Commands::Draw { svg, png, cols, rows, .. }) => {
            let epoch_idx = resolve_epoch_idx(&trek, cli.epoch)?;
            let episode_idx = resolve_episode_idx(&trek, epoch_idx, cli.episode)?;
            let zoom = cli.zoom.unwrap_or(1.0);
            let use_sextants = if cli.sextants { true } else if cli.no_sextants { false } else { config.octane.rendering.use_sextants };
            let use_octants = if cli.octants { true } else if cli.no_octants { false } else { config.octane.rendering.use_octants };
            run_draw(&trek, &config, epoch_idx, episode_idx, cli.timestep, cols, rows,
                     svg, png, zoom, use_sextants, use_octants, cli.prefs.as_deref())
        }
        Some(Commands::Animate { output, start, end, width, height }) => {
            let epoch_idx = resolve_epoch_idx(&trek, cli.epoch)?;
            let episode_idx = resolve_episode_idx(&trek, epoch_idx, cli.episode)?;
            let omega = cli.omega.unwrap_or(config.octane.podium.omega);
            let speed = cli.speed.unwrap_or(1.0);
            let fps = cli.fps.unwrap_or(30);
            let zoom = cli.zoom.unwrap_or(1.0);
            run_animate(&trek, &config, epoch_idx, episode_idx, start, end, speed, fps,
                        width, height, omega, output, zoom, cli.prefs.as_deref())
        }
        None => {
            // Default: run TUI or list epochs
            if cli.list_epochs {
                println!("Trek: {}", trek_path);
                println!("Epochs: {}", trek.epoch_count());
                println!();
                println!("{:>5}  {:>6}  {:>10}  {:>8}", "Epoch", "Eps", "NReturn", "Survival");
                println!("{}", "-".repeat(35));
                for epoch in &trek.epochs {
                    println!(
                        "{:>5}  {:>6}  {:>10}  {:>8}",
                        epoch.epoch_number,
                        epoch.episode_count(),
                        match epoch.mean_nreturn() {
                            Some(r) => format!("{:.2}", r),
                            None => "?".to_string(),
                        },
                        match epoch.epochia_alive_fraction {
                            Some(f) => format!("{:.0}%", f * 100.0),
                            None => "?".to_string(),
                        }
                    );
                }
                return Ok(());
            }

            // Initialize terminal
            let mut terminal = crate::app::init_terminal()?;

            // Set up panic hook to restore terminal
            let original_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |panic_info| {
                let _ = crossterm::terminal::disable_raw_mode();
                let _ = crossterm::execute!(
                    std::io::stdout(),
                    crossterm::terminal::LeaveAlternateScreen,
                    crossterm::event::DisableMouseCapture
                );
                original_hook(panic_info);
            }));

            // Create app with central config
            let render_config = if cli.no_mango {
                info!("Mango rendering disabled via CLI");
                RenderConfig::disabled()
            } else {
                RenderConfig::default()
            };

            // Start config file watcher for hot-reload
            let config_rx = crate::config::start_config_watcher();

            let mut app = App::with_config(trek, config, render_config, config_rx);

            // Start zoltar IPC listener (unless --no-zoltar)
            app.zoltar_channels = crate::zoltar::start(cli.no_zoltar, cli.zoltar_file_ipc);

            // CLI overrides (take priority over config file)
            if let Some(fps) = cli.fps {
                app.playback.fps = fps;
                info!("FPS overridden to {} from CLI", fps);
            }
            if let Some(omega) = cli.omega {
                app.omega = omega;
                info!("Omega overridden to {} from CLI", omega);
            }
            if cli.sextants {
                app.use_sextants = true;
                info!("Sextants enabled from CLI");
            } else if cli.no_sextants {
                app.use_sextants = false;
                info!("Sextants disabled from CLI");
            }
            if cli.octants {
                app.use_octants = true;
                info!("Octants enabled from CLI");
            } else if cli.no_octants {
                app.use_octants = false;
                info!("Octants disabled from CLI");
            }
            if let Some(theme) = cli.theme {
                app.scene_theme = theme;
                info!("Scene theme set to {} from CLI", theme.label());
            }
            if let Some(speed) = cli.speed {
                app.playback.speed = speed;
                info!("Playback speed set to {} from CLI", speed);
            }
            if let Some(zoom) = cli.zoom {
                app.zoom = zoom;
                info!("Zoom set to {} from CLI", zoom);
            }
            if let Some(w) = cli.sidebar_width {
                app.sidebar_width = w;
                info!("Sidebar width set to {} from CLI", w);
            }
            if let Some(ref prefs_str) = cli.prefs {
                match app.highway_prefs.apply_overrides(prefs_str) {
                    Ok(()) => info!("Applied prefs overrides from CLI: {}", prefs_str),
                    Err(e) => {
                        warn!("Bad --prefs: {}", e);
                        app.show_toast(format!("Bad --prefs: {}", e), std::time::Duration::from_secs(5));
                    }
                }
            }
            if let Some(ref parquet_path) = parquet_override {
                let parquet_str = parquet_path.to_string_lossy();
                let resolved = resolve_parquet_source(
                    &parquet_str,
                    &app.parquet_sources,
                    &app.trek.path,
                );
                match resolved {
                    Ok(idx) => {
                        app.selected_parquet = idx;
                        app.switch_parquet_source();
                        info!("Parquet source set to '{}' from target", app.parquet_sources[idx].display);
                    }
                    Err(msg) => {
                        warn!("{}", msg);
                        app.show_toast(msg, std::time::Duration::from_secs(5));
                    }
                }
            }
            if let Some(epoch_num) = cli.epoch {
                if let Some(idx) = app.trek.epochs.iter()
                    .position(|e| e.epoch_number == epoch_num)
                {
                    app.selected_epoch = idx;
                    app.selected_episode = 0;
                    app.frame_index = 0;
                    app.adjust_epochs_scroll();
                    info!("Epoch set to {} (index {}) from CLI", epoch_num, idx);
                } else {
                    let msg = format!("Epoch {} not found in trek", epoch_num);
                    warn!("{}", msg);
                    app.show_toast(msg, std::time::Duration::from_secs(5));
                }
            }
            if let Some(ep) = cli.episode {
                // Look up by es_episode number (the `e` column in parquet)
                let epoch_num = app.trek.epochs.get(app.selected_epoch)
                    .map(|e| e.epoch_number).unwrap_or(-1);
                let found = app.trek.epochs.get(app.selected_epoch)
                    .and_then(|e| e.episodes.iter()
                        .position(|em| em.es_episode == Some(ep as i64)));
                if let Some(idx) = found {
                    app.selected_episode = idx;
                    info!(
                        "Episode set to es_episode={} (index {} in epoch {}) from CLI",
                        ep, idx, epoch_num
                    );
                } else {
                    let n_episodes = app.trek.epochs.get(app.selected_epoch)
                        .map(|e| e.episodes.len())
                        .unwrap_or(0);
                    let es_range = app.trek.epochs.get(app.selected_epoch)
                        .and_then(|e| {
                            let first = e.episodes.first()?.es_episode?;
                            let last = e.episodes.last()?.es_episode?;
                            Some((first, last))
                        });
                    let range_str = match es_range {
                        Some((first, last)) => format!(", es_episode range {}..{}", first, last),
                        None => String::new(),
                    };
                    let msg = format!(
                        "Episode {} not found in epoch {} ({} episode{}{})",
                        ep, epoch_num, n_episodes,
                        if n_episodes == 1 { "" } else { "s" },
                        range_str,
                    );
                    warn!("{}", msg);
                    app.show_toast(msg, std::time::Duration::from_secs(5));
                    app.selected_episode = n_episodes.saturating_sub(1);
                }
            }
            if let Some(ts) = cli.timestep {
                app.pending_timestep = Some(ts);
                info!("Timestep deferred to {} from CLI", ts);
            }
            let result = crate::app::run(&mut terminal, &mut app);

            // Clean up zoltar socket
            crate::zoltar::cleanup();

            // Restore terminal
            crate::app::restore_terminal(&mut terminal)?;

            result
        }
    }
}
