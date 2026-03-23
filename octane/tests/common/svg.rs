//! SVG test helpers: state builders, rendering, and parsing.

use octane::data::{FrameState, VehicleState};
use octane::render::highway_svg::{
    render_highway_svg_from_episode, HighwayRenderConfig,
};
use octane::worlds::{
    SceneEpisode, SvgConfig, SvgEpisode, ViewportConfig, ViewportEpisode, DEFAULT_CORN_ASPRO,
};
use octane::config::SceneColorConfig;

fn default_ego() -> VehicleState {
    VehicleState {
        x: 50.0,
        y: 4.0,
        heading: 0.0,
        speed: 0.0,
        acceleration: 0.0,
        attention: None,
    }
}

pub fn make_simple_state() -> FrameState {
    FrameState {
        crashed: false,
        ego: default_ego(),
        npcs: vec![],
        ..Default::default()
    }
}

pub fn make_state_with_npcs(n_npcs: usize) -> FrameState {
    let mut npcs = Vec::with_capacity(n_npcs);
    for i in 0..n_npcs {
        npcs.push(VehicleState {
            x: 60.0 + (i as f64) * 15.0,
            y: (i % 4) as f64 * 4.0,
            heading: 0.0,
            speed: 0.0,
            acceleration: 0.0,
            attention: None,
        });
    }

    FrameState {
        crashed: false,
        ego: default_ego(),
        npcs,
        ..Default::default()
    }
}

pub fn make_crashed_state() -> FrameState {
    FrameState {
        crashed: true,
        ego: default_ego(),
        npcs: vec![],
        ..Default::default()
    }
}

/// Create an SvgEpisode from a single FrameState for testing.
pub fn make_svg_episode(state: FrameState, cols: u32, rows: u32) -> SvgEpisode {
    make_svg_episode_with_zoom(state, cols, rows, 1.0)
}

pub fn svg_default_config() -> HighwayRenderConfig {
    HighwayRenderConfig::default()
}

pub fn svg_config_with_size(cols: u32, rows: u32) -> HighwayRenderConfig {
    HighwayRenderConfig {
        n_cols: cols,
        n_rows: rows,
        ..Default::default()
    }
}

/// Render helper that creates episode and renders in one call.
pub fn render_state(state: &FrameState, config: &HighwayRenderConfig) -> String {
    let episode = make_svg_episode(state.clone(), config.n_cols, config.n_rows);
    render_highway_svg_from_episode(&episode, 0.0, config).unwrap()
}

/// Extract the center point of the first vehicle group from SVG.
pub fn extract_first_vehicle_center(svg: &str) -> Option<(f64, f64)> {
    let g_start = svg.find("<g transform=\"translate(")?;
    let translate_start = g_start + "<g transform=\"translate(".len();
    let translate_end = svg[translate_start..].find(')')? + translate_start;
    let translate_str = &svg[translate_start..translate_end];

    let mut parts = translate_str.split(',');
    let x: f64 = parts.next()?.parse().ok()?;
    let y: f64 = parts.next()?.parse().ok()?;
    Some((x, y))
}

/// Create an SvgEpisode with a specific zoom level.
pub fn make_svg_episode_with_zoom(state: FrameState, cols: u32, rows: u32, zoom: f64) -> SvgEpisode {
    let scene = SceneEpisode::from_frames(vec![state]);
    let viewport_config = ViewportConfig::with_omega(3.0).zoom(zoom);
    let viewport = ViewportEpisode::new_default(scene, viewport_config);
    SvgEpisode::new(viewport, SvgConfig::new(cols, rows, DEFAULT_CORN_ASPRO))
}

/// Parse an SVG path d="" attribute and return (width, height) of its bounding box.
pub fn path_bounding_box(path_data: &str) -> Option<(f64, f64)> {
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;
    let mut found_any = false;

    let nums: Vec<f64> = path_data
        .replace('M', " ")
        .replace('Q', " ")
        .replace('Z', "")
        .replace(',', " ")
        .split_whitespace()
        .filter_map(|s| s.parse::<f64>().ok())
        .collect();

    for chunk in nums.chunks(2) {
        if chunk.len() == 2 {
            let (x, y) = (chunk[0], chunk[1]);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            found_any = true;
        }
    }

    if found_any {
        Some((max_x - min_x, max_y - min_y))
    } else {
        None
    }
}

/// Extract bounding box sizes of all terrain blob paths from SVG.
pub fn extract_terrain_blob_sizes(svg: &str) -> Vec<(f64, f64)> {
    let mut sizes = Vec::new();
    let terrain_marker_string = format!("fill=\"{}\"", SceneColorConfig::default().terrain);
    let terrain_marker = terrain_marker_string.as_str();

    let mut search_from = 0;
    while let Some(path_pos) = svg[search_from..].find("<path d=\"") {
        let abs_pos = search_from + path_pos;
        let d_start = abs_pos + 9;

        if let Some(d_end_rel) = svg[d_start..].find('"') {
            let d_end = d_start + d_end_rel;
            let path_data = &svg[d_start..d_end];

            let remainder = &svg[d_end..d_end + 50.min(svg.len() - d_end)];
            if remainder.contains(terrain_marker) {
                if let Some(bbox) = path_bounding_box(path_data) {
                    sizes.push(bbox);
                }
            }

            search_from = d_end;
        } else {
            break;
        }
    }

    sizes
}
