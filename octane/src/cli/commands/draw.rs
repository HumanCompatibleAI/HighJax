//! Single-frame rendering command.

use anyhow::{Context, Result};
use std::path::PathBuf;

use crate::config::Config;
use crate::data::behaviors::{build_frame_from_state, find_behavior_json};
use crate::data::Trek;
use crate::mango::{render_svg_to_ansi, MangoConfig};
use crate::render::SceneRenderConfig;
use crate::util::posh_path;
use crate::worlds::{SceneEpisode, SvgConfig, SvgEpisode, ViewportConfig};

/// Run the draw command - render a single frame.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_draw(
    trek: &Trek,
    config: &Config,
    epoch_idx: usize,
    episode_idx: usize,
    timestep: Option<f64>,
    cols: u32,
    rows: u32,
    svg_path: Option<PathBuf>,
    png_path: Option<PathBuf>,
    zoom: f64,
    use_sextants: bool,
    use_octants: bool,
    prefs: Option<&str>,
) -> Result<()> {
    let env_type = trek.env_type
        .ok_or_else(|| anyhow::anyhow!("Unknown environment type (check meta.yaml)"))?;

    // Validate indices
    if epoch_idx >= trek.epochs.len() {
        anyhow::bail!("Epoch {} not found (have {} epochs)", epoch_idx, trek.epochs.len());
    }
    let epoch = &trek.epochs[epoch_idx];

    if episode_idx >= epoch.episodes.len() {
        anyhow::bail!("Episode {} not found in epoch {} (have {} episodes)",
            episode_idx, epoch_idx, epoch.episodes.len());
    }

    // Load frame state - use parquet for highjax treks, otherwise episode JSON
    let ep_meta = &epoch.episodes[episode_idx];
    let es_keys = match (ep_meta.es_epoch, ep_meta.es_episode) {
        (Some(e), Some(ep)) => Some((e, ep)),
        _ => None,
    };
    let state = if let (Some(ref pq_index), Some((es_epoch, es_episode))) = (&trek.es_parquet_index, es_keys) {
        let frames = pq_index.load_episode(es_epoch, es_episode, env_type)
            .context("Failed to load episode from parquet")?;
        if frames.is_empty() {
            anyhow::bail!("Episode has no frames");
        }
        let frame_idx = resolve_timestep_to_frame(&frames, timestep);
        frames[frame_idx].state.clone()
    } else {
        // Standard format - load episode JSON
        let episode = epoch.episodes[episode_idx].load()?;
        let frame_idx = timestep
            .map(|_| 0usize) // JSON episodes don't have t values; just use 0
            .unwrap_or(0);
        if frame_idx >= episode.frames.len() {
            anyhow::bail!("Frame {} not found in episode {} (have {} frames)",
                frame_idx, episode_idx, episode.frames.len());
        }
        let frame = &episode.frames[frame_idx];
        match &frame.vehicle_state {
            Some(s) => s.clone(),
            None => anyhow::bail!("Frame {} has no vehicle_state data", frame_idx),
        }
    };

    let scene = SceneEpisode::from_frames_with_timestep(vec![state], trek.seconds_per_sub_t);

    // Build viewport via env adapter
    let vp_config = ViewportConfig {
        zoom,
        corn_aspro: config.octane.rendering.corn_aspro,
        ..ViewportConfig::default()
    };
    let viewport = env_type.build_viewport(scene, vp_config, trek, config);
    let svg_config = SvgConfig::new(cols, rows, config.octane.rendering.corn_aspro);
    let svg_episode = SvgEpisode::new(viewport, svg_config);

    // Build env-appropriate render config via centralized dispatch
    let theme = config.octane.rendering.theme;
    let mut render_config = env_type.scene_render_config(config, trek, cols, rows, theme);

    // Apply --prefs overrides
    if let Some(prefs_str) = prefs {
        apply_prefs_to_render_config(&mut render_config, prefs_str)?;
    }

    render_svg_output(&svg_episode, &render_config, svg_path, png_path, use_sextants, use_octants)
}

/// Resolve a timestep value to a frame index via binary search on `t` values.
fn resolve_timestep_to_frame(frames: &[crate::data::jsonla::EsFrame], timestep: Option<f64>) -> usize {
    let Some(ts) = timestep else {
        return 0;
    };
    match frames.binary_search_by(|f| f.t.partial_cmp(&ts).unwrap()) {
        Ok(i) => i,
        Err(i) => {
            // Nearest neighbor
            if i == 0 { return 0; }
            if i >= frames.len() { return frames.len() - 1; }
            let before = (frames[i - 1].t - ts).abs();
            let after = (frames[i].t - ts).abs();
            if before <= after { i - 1 } else { i }
        }
    }
}

/// Apply --prefs string to a SceneRenderConfig (dispatches to env-specific apply).
pub(crate) fn apply_prefs_to_render_config(render_config: &mut SceneRenderConfig, prefs_str: &str) -> Result<()> {
    match render_config {
        SceneRenderConfig::Highway(ref mut hrc) => {
            // Parse prefs and apply to the highway render config fields directly
            for pair in prefs_str.split(',') {
                let pair = pair.trim();
                if pair.is_empty() { continue; }
                let (key, val) = pair.split_once('=')
                    .ok_or_else(|| anyhow::anyhow!("invalid pref '{}' (expected key=value)", pair))?;
                let key = key.trim();
                let val = val.trim();
                match key {
                    "podium_marker" => hrc.show_podium_marker = parse_bool(val, key)?,
                    "scala" => hrc.show_scala = parse_bool(val, key)?,
                    "attention" => hrc.show_attention = parse_bool(val, key)?,
                    "debug_eye" => hrc.debug_eye = parse_bool(val, key)?,
                    "light_blend" => hrc.light_blend_mode = val.to_string(),
                    "velocity_arrows" => hrc.show_velocity_arrows = parse_bool_from_mode(val)?,
                    "action_distribution" => hrc.show_action_distribution = parse_bool_from_mode(val)?,
                    "action_distribution_text" => hrc.show_action_distribution_text = parse_bool_from_mode(val)?,
                    "npc_text" => hrc.show_npc_text = parse_bool_from_mode(val)?,
                    _ => anyhow::bail!("unknown Highway pref '{}'", key),
                }
            }
        }
    }
    Ok(())
}

fn parse_bool(val: &str, key: &str) -> Result<bool> {
    match val {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => anyhow::bail!("invalid bool for '{}': '{}'", key, val),
    }
}

/// For DisplayMode prefs applied to a render config (which has show_X bools, not DisplayMode),
/// "always" → true, "never" → false, "on-pause" → true (draw is always "paused").
fn parse_bool_from_mode(val: &str) -> Result<bool> {
    match val {
        "always" | "on-pause" | "on_pause" | "onpause" | "true" => Ok(true),
        "never" | "false" => Ok(false),
        _ => anyhow::bail!("invalid display mode: '{}' (expected always/on-pause/never)", val),
    }
}

/// Run the draw command for a behavior scenario (no trek needed).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_draw_behavior(
    config: &Config,
    behavior_name: &str,
    scenario_idx: usize,
    cols: u32,
    rows: u32,
    svg_path: Option<PathBuf>,
    png_path: Option<PathBuf>,
    zoom: f64,
    use_sextants: bool,
    use_octants: bool,
    prefs: Option<&str>,
    hardcoded_action: Option<&str>,
) -> Result<()> {
    let lane_width = config.octane.road.lane_width;

    // Find behavior JSON
    // No trek available, so search all env types
    let path = find_behavior_json(behavior_name, None)
        .ok_or_else(|| anyhow::anyhow!(
            "Behavior '{}' not found in user or preset behaviors",
            behavior_name,
        ))?;

    let text = std::fs::read_to_string(&path)
        .context("Failed to read behavior JSON")?;
    let data: serde_json::Value = serde_json::from_str(&text)
        .context("Failed to parse behavior JSON")?;

    let scenarios = data["scenarios"].as_array()
        .ok_or_else(|| anyhow::anyhow!("No scenarios array in behavior"))?;
    if scenario_idx >= scenarios.len() {
        anyhow::bail!(
            "Scenario {} not found (behavior '{}' has {} scenarios)",
            scenario_idx, behavior_name, scenarios.len(),
        );
    }
    let scenario = &scenarios[scenario_idx];

    let state = scenario.get("state")
        .ok_or_else(|| anyhow::anyhow!(
            "Scenario {} has no 'state' field (only state-based scenarios can be drawn)",
            scenario_idx,
        ))?;

    let n_lanes = config.octane.road.n_lanes;
    let frame = build_frame_from_state(state, n_lanes, lane_width);

    let scene = SceneEpisode::from_frames(vec![frame]);
    // Behaviors don't carry env type; default to Highway road geometry
    let n_lanes = config.octane.road.n_lanes;
    let road_center_y = (n_lanes - 1) as f64 * lane_width / 2.0;
    let vp_config = ViewportConfig {
        zoom,
        corn_aspro: config.octane.rendering.corn_aspro,
        ..ViewportConfig::default()
    };
    let viewport = crate::worlds::ViewportEpisode::new(scene, vp_config, road_center_y);
    let svg_config = SvgConfig::new(cols, rows, config.octane.rendering.corn_aspro);
    let svg_episode = SvgEpisode::new(viewport, svg_config);

    // Use Highway render config (behaviors are currently Highway-only)
    let theme = config.octane.rendering.theme;
    let mut render_config = crate::envs::highway::scene_render_config(config, &Trek::empty(), cols, rows, theme);

    // --hardcoded-action: set on render config for filled-polygon arrow
    if let Some(action_name) = hardcoded_action {
        match action_name {
            "left" | "idle" | "right" | "faster" | "slower" => {}
            _ => anyhow::bail!("Unknown action '{}' (expected: left, idle, right, faster, slower)", action_name),
        }
        let SceneRenderConfig::Highway(ref mut hrc) = render_config;
        hrc.hardcoded_action = Some(action_name.to_string());
    }

    // Apply --prefs overrides
    if let Some(prefs_str) = prefs {
        apply_prefs_to_render_config(&mut render_config, prefs_str)?;
    }

    render_svg_output(&svg_episode, &render_config, svg_path, png_path, use_sextants, use_octants)
}

/// Generate SVG from episode and output to file or terminal.
fn render_svg_output(
    svg_episode: &SvgEpisode,
    render_config: &SceneRenderConfig,
    svg_path: Option<PathBuf>,
    png_path: Option<PathBuf>,
    use_sextants: bool,
    use_octants: bool,
) -> Result<()> {
    let svg = render_config.render_svg(svg_episode, 0.0)
        .ok_or_else(|| anyhow::anyhow!("Failed to render frame"))?;

    if let Some(path) = svg_path {
        std::fs::write(&path, &svg)?;
        println!("SVG written to: {}", posh_path(&path));
        return Ok(());
    }

    if let Some(path) = png_path {
        let mut opt = resvg::usvg::Options::default();
        opt.fontdb_mut().load_system_fonts();
        opt.fontdb_mut().set_monospace_family("DejaVu Sans Mono");
        opt.fontdb_mut().set_sans_serif_family("DejaVu Sans");
        opt.fontdb_mut().set_serif_family("DejaVu Serif");
        let tree = resvg::usvg::Tree::from_str(&svg, &opt)
            .context("Failed to parse SVG")?;
        let size = tree.size();
        let width = size.width() as u32;
        let height = size.height() as u32;
        let mut pixmap = resvg::tiny_skia::Pixmap::new(width, height)
            .context("Failed to create pixmap")?;
        resvg::render(&tree, resvg::tiny_skia::Transform::default(), &mut pixmap.as_mut());
        pixmap.save_png(&path)
            .context("Failed to save PNG")?;
        println!("PNG written to: {}", posh_path(&path));
        return Ok(());
    }

    // Default: render to terminal using mango
    let mango_config = MangoConfig {
        n_cols: render_config.n_cols(),
        n_rows: render_config.n_rows(),
        use_sextants,
        use_octants,
    };
    let ansi_output = render_svg_to_ansi(&svg, &mango_config)
        .map_err(|e| anyhow::anyhow!("Mango render failed: {}", e))?;
    print!("{}", ansi_output);
    Ok(())
}
