//! Video animation command.

use anyhow::{Context, Result};
use std::path::PathBuf;

use crate::config::Config;
use crate::data::Trek;
use crate::util::posh_path;
use crate::worlds::{SceneEpisode, SvgConfig, SvgEpisode, ViewportConfig};

/// Run the animate command - render episode to video with interpolation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_animate(
    trek: &Trek,
    config: &Config,
    epoch_idx: usize,
    episode_idx: usize,
    start_frame: usize,
    end_frame: Option<usize>,
    speed: f64,
    fps: u32,
    width: u32,
    height: u32,
    omega: f64,
    output_path: PathBuf,
    zoom: f64,
    prefs: Option<&str>,
) -> Result<()> {
    use rayon::prelude::*;
    use std::process::Command;
    use std::sync::atomic::{AtomicU32, Ordering};

    // Validate indices
    if epoch_idx >= trek.epochs.len() {
        anyhow::bail!("Epoch {} not found (have {} epochs)", epoch_idx, trek.epochs.len());
    }
    let epoch = &trek.epochs[epoch_idx];

    if episode_idx >= epoch.episodes.len() {
        anyhow::bail!("Episode {} not found in epoch {} (have {} episodes)",
            episode_idx, epoch_idx, epoch.episodes.len());
    }

    let env_type = trek.env_type
        .ok_or_else(|| anyhow::anyhow!("Unknown environment type (check meta.yaml)"))?;

    // Load all frames
    let ep_meta = &epoch.episodes[episode_idx];
    let es_keys = match (ep_meta.es_epoch, ep_meta.es_episode) {
        (Some(e), Some(ep)) => Some((e, ep)),
        _ => None,
    };
    let frames = if let (Some(ref pq_index), Some((es_epoch, es_episode))) = (&trek.es_parquet_index, es_keys) {
        let es_frames = pq_index.load_episode(es_epoch, es_episode, env_type)
            .context("Failed to load episode from parquet")?;
        es_frames.into_iter().map(|f| f.state).collect::<Vec<_>>()
    } else {
        let episode = epoch.episodes[episode_idx].load()?;
        episode.frames.into_iter()
            .filter_map(|f| f.vehicle_state)
            .collect::<Vec<_>>()
    };

    let n_frames = frames.len();
    if n_frames == 0 {
        anyhow::bail!("Episode has no frames with vehicle state");
    }

    // Find crash frame (stop there if no explicit end)
    let crash_frame = frames.iter().position(|f| f.crashed);
    let default_end = crash_frame.unwrap_or(n_frames - 1);
    let end_frame = end_frame.map(|e| e.min(default_end)).unwrap_or(default_end);

    if start_frame > end_frame {
        anyhow::bail!("Start frame {} > end frame {}", start_frame, end_frame);
    }

    // SVG render config - calculate cols/rows from pixel dimensions
    let cols = (width / 21).max(40);
    let rows = (height / 42).max(20);

    // Build full episode stack: Scene -> Viewport -> SVG (env-modular)
    let dt = trek.seconds_per_sub_t;
    let mut scene = SceneEpisode::from_frames_with_timestep(frames, dt);
    scene.set_acceleration_lookback_seconds(
        config.octane.rendering.brakelight_deceleration_lookback_seconds,
    );
    let viewport_config = ViewportConfig {
        zoom,
        omega,
        corn_aspro: config.octane.rendering.corn_aspro,
        podium_fraction: config.octane.podium.offset,
        damping_ratio: config.octane.podium.damping_ratio,
    };
    let viewport = env_type.build_viewport(scene, viewport_config, trek, config);
    let svg_config = SvgConfig::new(cols, rows, config.octane.rendering.corn_aspro);
    let svg_episode = SvgEpisode::new(viewport, svg_config);

    // Build env-appropriate render config via centralized dispatch
    let theme = config.octane.rendering.theme;
    let mut render_config = env_type.scene_render_config(config, trek, cols, rows, theme);

    // Apply --prefs overrides
    if let Some(prefs_str) = prefs {
        super::draw::apply_prefs_to_render_config(&mut render_config, prefs_str)?;
    }

    // Calculate subdivisions from speed
    let subdivisions = ((dt * fps as f64) / speed).round().max(1.0) as usize;

    println!("Animating frames {}-{} ({} states){}",
        start_frame, end_frame, end_frame - start_frame + 1,
        if crash_frame.is_some() && end_frame == crash_frame.unwrap() { " [stops at crash]" } else { "" });
    println!("Speed: {}x, FPS: {}, Subdivisions: {}, Resolution: {}x{}", speed, fps, subdivisions, width, height);

    // Create temp directory for PNGs
    let temp_dir = tempfile::tempdir().context("Failed to create temp directory")?;
    let temp_path = temp_dir.path();

    // Pre-compute all work items: (frame_number, scene_time)
    let total_video_frames = (end_frame - start_frame) * subdivisions + 1;
    let mut work_items: Vec<(u32, f64)> = Vec::with_capacity(total_video_frames);
    let mut video_frame_num = 0u32;

    for state_idx in start_frame..=end_frame {
        let n_subs = if state_idx < end_frame { subdivisions } else { 1 };

        for sub_idx in 0..n_subs {
            let lerp_t = sub_idx as f64 / subdivisions as f64;
            let scene_time = (state_idx as f64 + lerp_t) * dt;
            work_items.push((video_frame_num, scene_time));
            video_frame_num += 1;
        }
    }

    // Render frames in parallel with 10 threads
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(10)
        .build()
        .context("Failed to create thread pool")?;

    let progress = AtomicU32::new(0);
    let total = total_video_frames as u32;

    // Load system fonts once before parallel rendering
    let mut font_opt = resvg::usvg::Options::default();
    font_opt.fontdb_mut().load_system_fonts();
    font_opt.fontdb_mut().set_monospace_family("DejaVu Sans Mono");
    font_opt.fontdb_mut().set_sans_serif_family("DejaVu Sans");
    font_opt.fontdb_mut().set_serif_family("DejaVu Serif");

    pool.install(|| {
        work_items.par_iter().try_for_each(|(frame_num, scene_time)| -> Result<()> {
            let svg = render_config.render_svg(&svg_episode, *scene_time)
                .ok_or_else(|| anyhow::anyhow!("Failed to render frame at scene_time={}", scene_time))?;

            let tree = resvg::usvg::Tree::from_str(&svg, &font_opt)
                .context("Failed to parse SVG")?;

            let svg_size = tree.size();
            let scale_x = width as f32 / svg_size.width();
            let scale_y = height as f32 / svg_size.height();

            let mut pixmap = resvg::tiny_skia::Pixmap::new(width, height)
                .context("Failed to create pixmap")?;
            let transform = resvg::tiny_skia::Transform::from_scale(scale_x, scale_y);
            resvg::render(&tree, transform, &mut pixmap.as_mut());

            let png_path = temp_path.join(format!("frame_{:06}.png", frame_num));
            pixmap.save_png(&png_path)
                .context("Failed to save PNG")?;

            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if done.is_multiple_of(10) || done == total {
                print!("\rRendering: {}/{} frames", done, total);
                let _ = std::io::Write::flush(&mut std::io::stdout());
            }
            Ok(())
        })
    })?;
    println!();

    // Run ffmpeg to create video
    println!("Encoding video with ffmpeg...");
    let ffmpeg_status = Command::new("ffmpeg")
        .arg("-y")  // Overwrite output
        .arg("-framerate").arg(fps.to_string())
        .arg("-i").arg(temp_path.join("frame_%06d.png"))
        .arg("-c:v").arg("libx264")
        .arg("-pix_fmt").arg("yuv420p")
        .arg("-crf").arg("18")  // High quality
        .arg(&output_path)
        .status()
        .context("Failed to run ffmpeg")?;

    if !ffmpeg_status.success() {
        anyhow::bail!("ffmpeg failed with status: {}", ffmpeg_status);
    }

    println!("Video written to: {}", posh_path(&output_path));
    Ok(())
}
