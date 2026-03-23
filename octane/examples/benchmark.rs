//! Benchmark for the rendering pipeline.

use std::time::{Duration, Instant};

use octane::data::{FrameState, VehicleState};
use octane::mango::{render_svg_to_ansi, MangoConfig};
use octane::render::highway_svg::{render_highway_svg_from_episode, HighwayRenderConfig};
use octane::worlds::{
    SceneEpisode, ScenePoint, SvgConfig, SvgEpisode, ViewportConfig, ViewportEpisode,
    DEFAULT_CORN_ASPRO,
};

fn make_test_state(frame_idx: usize) -> FrameState {
    // Simulate movement over time
    let ego_x = 50.0 + (frame_idx as f64) * 0.5;
    FrameState {
        crashed: false,
        ego: VehicleState {
            x: ego_x,
            y: 4.0,
            heading: 0.0,
            speed: 0.0,
            acceleration: 0.0,
            attention: None,
        },
        npcs: vec![
            VehicleState {
                x: ego_x + 20.0,
                y: 0.0,
                heading: 0.0,
                speed: 0.0,
                acceleration: 0.0,
                attention: None,
            },
            VehicleState {
                x: ego_x + 30.0,
                y: 8.0,
                heading: 0.0,
                speed: 0.0,
                acceleration: 0.0,
                attention: None,
            },
            VehicleState {
                x: ego_x - 10.0,
                y: 12.0,
                heading: 0.0,
                speed: 0.0,
                acceleration: 0.0,
                attention: None,
            },
        ],
        ..Default::default()
    }
}

fn main() {
    let n_frames_short = 30;
    let n_frames_long = 1000;
    let config = HighwayRenderConfig {
        n_cols: 120,
        n_rows: 40,
        ..Default::default()
    };

    println!(
        "=== Octane Rendering Benchmarks (debug build) ===\n\
         Terminal size: {}x{}\n",
        config.n_cols, config.n_rows
    );

    // === Benchmark 1: ViewportEpisode construction ===
    println!("=== ViewportEpisode Construction ===");
    let frames_1k: Vec<FrameState> = (0..n_frames_long).map(make_test_state).collect();

    let start = Instant::now();
    let scene = SceneEpisode::from_frames(frames_1k.clone());
    let scene_time = start.elapsed();

    let start = Instant::now();
    let viewport = ViewportEpisode::new_default(scene.clone(), ViewportConfig::default());
    let viewport_time = start.elapsed();

    let start = Instant::now();
    let svg_episode = SvgEpisode::new(
        viewport,
        SvgConfig::new(config.n_cols, config.n_rows, DEFAULT_CORN_ASPRO),
    );
    let svg_episode_time = start.elapsed();

    println!("  {} frames:", n_frames_long);
    println!("    SceneEpisode::from_frames: {:?}", scene_time);
    println!("    ViewportEpisode::new_default: {:?}", viewport_time);
    println!("    SvgEpisode::new: {:?}", svg_episode_time);
    println!(
        "    Total episode stack: {:?}",
        scene_time + viewport_time + svg_episode_time
    );
    println!();

    // === Benchmark 2: scene_to_svg transforms ===
    println!("=== Coordinate Transforms (scene_to_svg) ===");
    let n_points = 1000;
    let points: Vec<ScenePoint> = (0..n_points)
        .map(|i| ScenePoint::new(i as f64 * 0.1, (i % 16) as f64))
        .collect();

    let start = Instant::now();
    for point in &points {
        let _ = svg_episode.scene_to_svg(*point, 0.0);
    }
    let transform_time = start.elapsed();

    println!("  {} points at scene_time=0.0:", n_points);
    println!("    Total time: {:?}", transform_time);
    println!(
        "    Per-point: {:?}",
        transform_time / n_points as u32
    );
    println!(
        "    Throughput: {:.0} points/sec",
        n_points as f64 / transform_time.as_secs_f64()
    );
    println!();

    // === Benchmark 3: SVG generation (short episode) ===
    println!("=== SVG Generation ({} frames) ===", n_frames_short);
    let frames_short: Vec<FrameState> = (0..n_frames_short).map(make_test_state).collect();
    let scene_short = SceneEpisode::from_frames(frames_short);
    let viewport_short = ViewportEpisode::new_default(scene_short, ViewportConfig::default());
    let svg_episode_short = SvgEpisode::new(
        viewport_short,
        SvgConfig::new(config.n_cols, config.n_rows, DEFAULT_CORN_ASPRO),
    );

    // Warm up and save sample
    let sample_svg =
        render_highway_svg_from_episode(&svg_episode_short, 0.0, &config).expect("Render failed");
    let svg_path = std::env::temp_dir().join("octane-benchmark.svg");
    std::fs::write(&svg_path, &sample_svg).expect("Failed to write sample SVG");
    println!("  Sample SVG: {} ({} bytes)", svg_path.display(), sample_svg.len());

    let mut svg_times: Vec<Duration> = Vec::new();
    let mut svg_sizes: Vec<usize> = Vec::new();

    for i in 0..n_frames_short {
        let scene_time = i as f64 * 0.1;
        let start = Instant::now();
        let svg = render_highway_svg_from_episode(&svg_episode_short, scene_time, &config)
            .expect("Render failed");
        svg_times.push(start.elapsed());
        svg_sizes.push(svg.len());
    }

    let avg_svg_time: Duration = svg_times.iter().sum::<Duration>() / n_frames_short as u32;
    let avg_svg_size: usize = svg_sizes.iter().sum::<usize>() / n_frames_short;

    println!("  Average time: {:?}", avg_svg_time);
    println!("  Average size: {} bytes", avg_svg_size);
    println!(
        "  Max throughput: {:.1} fps",
        1.0 / avg_svg_time.as_secs_f64()
    );
    println!();

    // === Benchmark 4: Mango rendering ===
    println!("=== Mango Rendering ===");
    let mut mango_times: Vec<Duration> = Vec::new();
    let mango_config = MangoConfig {
        n_cols: config.n_cols,
        n_rows: config.n_rows,
        use_sextants: true,
        use_octants: true,
    };

    for i in 0..n_frames_short {
        let scene_time = i as f64 * 0.1;
        let svg = render_highway_svg_from_episode(&svg_episode_short, scene_time, &config)
            .expect("Render failed");

        let start = Instant::now();
        let _output = render_svg_to_ansi(&svg, &mango_config).expect("Mango render failed");
        mango_times.push(start.elapsed());
    }

    let avg_mango_time: Duration =
        mango_times.iter().sum::<Duration>() / mango_times.len() as u32;
    println!("  Average time: {:?}", avg_mango_time);
    println!(
        "  Max throughput: {:.2} fps",
        1.0 / avg_mango_time.as_secs_f64()
    );
    println!();

    // === Summary ===
    let total_time = avg_svg_time + avg_mango_time;
    println!("=== Total Pipeline ===");
    println!("  SVG gen + Mango: {:?}", total_time);
    println!(
        "  Max throughput: {:.2} fps",
        1.0 / total_time.as_secs_f64()
    );

    let target_frame_time = Duration::from_secs_f64(1.0 / 30.0);
    if total_time < target_frame_time {
        println!("  30 FPS: ACHIEVABLE ✓");
    } else {
        println!(
            "  30 FPS: NOT achievable (need {:?}, have {:?})",
            target_frame_time, total_time
        );
        println!(
            "  Need {:.1}x speedup",
            total_time.as_secs_f64() / target_frame_time.as_secs_f64()
        );
    }
}
