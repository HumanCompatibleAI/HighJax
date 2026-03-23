//! SVG coordinate transform and terrain zoom tests.

mod common;

use common::{
    extract_first_vehicle_center, extract_terrain_blob_sizes,
    make_simple_state, make_svg_episode_with_zoom, render_state, svg_default_config,
};
use octane::config::SceneColorConfig;
use octane::data::{FrameState, VehicleState};
use octane::render::highway_svg::{render_highway_svg_from_episode, HighwayRenderConfig};
use octane::worlds::DEFAULT_CORN_ASPRO;

fn compute_scale(zoom: f64, corn_aspro: f64) -> f64 {
    use octane::worlds::UNZOOMED_CANVAS_DIAGONAL_IN_METERS;
    let canvas_diagonal = (1.0_f64 + corn_aspro * corn_aspro).sqrt();
    let scene_diagonal = UNZOOMED_CANVAS_DIAGONAL_IN_METERS / zoom;
    canvas_diagonal / scene_diagonal
}

fn compute_visible_width(scale: f64) -> f64 {
    1.0 / scale
}

// =============================================================================
// Coordinate Transform Tests
// =============================================================================

#[test]
fn test_scale_calculation_at_zoom_1() {
    let scale = compute_scale(1.0, DEFAULT_CORN_ASPRO);
    let visible_width = compute_visible_width(scale);

    assert!(
        visible_width > 80.0 && visible_width < 90.0,
        "At zoom=1.0, visible width should be ~85m, got {:.1}m",
        visible_width
    );
}

#[test]
fn test_scale_calculation_at_zoom_2_halves_visible_width() {
    let scale_1 = compute_scale(1.0, DEFAULT_CORN_ASPRO);
    let scale_2 = compute_scale(2.0, DEFAULT_CORN_ASPRO);

    let width_1 = compute_visible_width(scale_1);
    let width_2 = compute_visible_width(scale_2);

    let ratio = width_2 / width_1;
    assert!(
        (ratio - 0.5).abs() < 0.01,
        "Zoom=2.0 should halve visible width, got ratio {:.3}",
        ratio
    );
}

#[test]
fn test_scale_calculation_at_zoom_half_doubles_visible_width() {
    let scale_1 = compute_scale(1.0, DEFAULT_CORN_ASPRO);
    let scale_half = compute_scale(0.5, DEFAULT_CORN_ASPRO);

    let width_1 = compute_visible_width(scale_1);
    let width_half = compute_visible_width(scale_half);

    let ratio = width_half / width_1;
    assert!(
        (ratio - 2.0).abs() < 0.01,
        "Zoom=0.5 should double visible width, got ratio {:.3}",
        ratio
    );
}

#[test]
fn test_podium_position_ego_at_20_percent_from_left() {
    let state = make_simple_state();
    let config = svg_default_config();
    let svg = render_state(&state, &config);

    let center = extract_first_vehicle_center(&svg)
        .expect("Should find ego polygon");

    assert!(
        center.0 > 0.15 && center.0 < 0.25,
        "Ego should be at ~20% from left (x≈0.2), got x={:.3}",
        center.0
    );
}

#[test]
fn test_different_terminal_dimensions_same_world_mapping() {
    let state = make_simple_state();

    let svg_small = render_state(&state, &HighwayRenderConfig { n_cols: 40, n_rows: 10, ..Default::default() });
    let svg_large = render_state(&state, &HighwayRenderConfig { n_cols: 160, n_rows: 40, ..Default::default() });

    let center_small = extract_first_vehicle_center(&svg_small).unwrap();
    let center_large = extract_first_vehicle_center(&svg_large).unwrap();

    assert!(
        (center_small.0 - center_large.0).abs() < 0.02,
        "Different terminal sizes should have same ego x position: small={:.3}, large={:.3}",
        center_small.0, center_large.0
    );
}

// =============================================================================
// Transform Composition Tests
// =============================================================================

#[test]
fn test_mango_receives_correct_pixel_dimensions() {
    let state = make_simple_state();

    for (n_cols, n_rows) in [(40, 10), (80, 20), (120, 30)] {
        let config = HighwayRenderConfig {
            n_cols,
            n_rows,
            ..Default::default()
        };
        let svg = render_state(&state, &config);

        let width_start = svg.find("width=\"").unwrap() + 7;
        let width_end = svg[width_start..].find('"').unwrap() + width_start;
        let width: u32 = svg[width_start..width_end].parse().unwrap();

        let height_start = svg.find("height=\"").unwrap() + 8;
        let height_end = svg[height_start..].find('"').unwrap() + height_start;
        let height: u32 = svg[height_start..height_end].parse().unwrap();

        assert!(
            width >= n_cols * 8 && width <= n_cols * 16,
            "n_cols={}: width={} should be in reasonable range",
            n_cols, width
        );
        assert!(
            height >= n_rows * 8 && height <= n_rows * 30,
            "n_rows={}: height={} should be in reasonable range",
            n_rows, height
        );
    }
}

#[test]
fn test_viewbox_matches_normalized_svg_coords() {
    let state = make_simple_state();
    let config = HighwayRenderConfig {
        n_cols: 80,
        n_rows: 20,
        ..Default::default()
    };
    let svg = render_state(&state, &config);

    let vb_start = svg.find("viewBox=\"").unwrap() + 9;
    let vb_end = svg[vb_start..].find('"').unwrap() + vb_start;
    let viewbox = &svg[vb_start..vb_end];

    assert!(
        viewbox.starts_with("0 0 1 "),
        "viewBox should start with '0 0 1 ', got '{}'",
        viewbox
    );

    let parts: Vec<&str> = viewbox.split_whitespace().collect();
    assert_eq!(parts.len(), 4);
    let vb_height: f64 = parts[3].parse().unwrap();

    let expected_height = (20.0 / 80.0) * DEFAULT_CORN_ASPRO;
    assert!(
        (vb_height - expected_height).abs() < 0.01,
        "viewBox height={}, expected={}",
        vb_height, expected_height
    );
}

#[test]
fn test_ego_position_stable_across_frames() {
    let config = svg_default_config();

    let positions_to_test = [0.0, 50.0, 100.0, 500.0, 1000.0];
    let mut ego_svg_positions = Vec::new();

    for ego_x in positions_to_test {
        let state = FrameState {
            crashed: false,
            ego: VehicleState { x: ego_x, y: 4.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None },
            npcs: vec![],
            ..Default::default()
        };
        let svg = render_state(&state, &config);
        let center = extract_first_vehicle_center(&svg).unwrap();
        ego_svg_positions.push(center.0);
    }

    let first = ego_svg_positions[0];
    for (i, &pos) in ego_svg_positions.iter().enumerate() {
        assert!(
            (pos - first).abs() < 0.01,
            "Ego at world x={} has SVG x={}, expected {}",
            positions_to_test[i], pos, first
        );
    }
}

#[test]
fn test_relative_npc_positions_preserved() {
    let ego_x = 50.0;

    let state = FrameState {
        crashed: false,
        ego: VehicleState { x: ego_x, y: 4.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None },
        npcs: vec![VehicleState { x: ego_x + 10.0, y: 4.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None }],
        ..Default::default()
    };

    let config = svg_default_config();
    let svg = render_state(&state, &config);

    // Find the first and second vehicle groups via <g transform="translate(...)
    let search_pattern = "<g transform=\"translate(";
    let first_pos = svg.find(search_pattern).unwrap();
    let first_end = first_pos + search_pattern.len();
    let second_start = svg[first_end..].find(search_pattern);

    if let Some(offset) = second_start {
        let ego_center = extract_first_vehicle_center(&svg).unwrap();
        let npc_slice = &svg[first_end + offset..];
        let npc_center = extract_first_vehicle_center(npc_slice).unwrap();

        assert!(
            npc_center.0 > ego_center.0,
            "NPC ahead of ego should be right in SVG: npc={:.3}, ego={:.3}",
            npc_center.0, ego_center.0
        );
    }
}

// =============================================================================
// Terrain Zoom Scaling Tests
// =============================================================================

#[test]
fn test_terrain_blob_sizes_scale_with_zoom() {
    let state = make_simple_state();
    let cols = 80;
    let rows = 14;

    let episode_z1 = make_svg_episode_with_zoom(state.clone(), cols, rows, 1.0);
    let episode_z2 = make_svg_episode_with_zoom(state.clone(), cols, rows, 2.0);

    let config = HighwayRenderConfig {
        n_cols: cols,
        n_rows: rows,
        ..Default::default()
    };

    let svg_z1 = render_highway_svg_from_episode(&episode_z1, 0.0, &config).unwrap();
    let svg_z2 = render_highway_svg_from_episode(&episode_z2, 0.0, &config).unwrap();

    let sizes_z1 = extract_terrain_blob_sizes(&svg_z1);
    let sizes_z2 = extract_terrain_blob_sizes(&svg_z2);

    assert!(!sizes_z1.is_empty(), "Should have terrain blobs at zoom=1");
    assert!(!sizes_z2.is_empty(), "Should have terrain blobs at zoom=2");

    let avg_w_z1: f64 = sizes_z1.iter().map(|(w, _)| w).sum::<f64>() / sizes_z1.len() as f64;
    let avg_w_z2: f64 = sizes_z2.iter().map(|(w, _)| w).sum::<f64>() / sizes_z2.len() as f64;

    // At zoom=2, blobs should be ~2x bigger in SVG units
    let ratio = avg_w_z2 / avg_w_z1;
    assert!(
        ratio > 1.5 && ratio < 2.5,
        "Zoom=2 blob sizes should be ~2x zoom=1: avg_z1={:.4}, avg_z2={:.4}, ratio={:.2}",
        avg_w_z1, avg_w_z2, ratio
    );
}

#[test]
fn test_terrain_blob_sizes_shrink_when_zooming_out() {
    let state = make_simple_state();
    let cols = 80;
    let rows = 14;

    let episode_z1 = make_svg_episode_with_zoom(state.clone(), cols, rows, 1.0);
    let episode_zhalf = make_svg_episode_with_zoom(state.clone(), cols, rows, 0.5);

    let config = HighwayRenderConfig {
        n_cols: cols,
        n_rows: rows,
        ..Default::default()
    };

    let svg_z1 = render_highway_svg_from_episode(&episode_z1, 0.0, &config).unwrap();
    let svg_zhalf = render_highway_svg_from_episode(&episode_zhalf, 0.0, &config).unwrap();

    let sizes_z1 = extract_terrain_blob_sizes(&svg_z1);
    let sizes_zhalf = extract_terrain_blob_sizes(&svg_zhalf);

    assert!(!sizes_z1.is_empty());
    assert!(!sizes_zhalf.is_empty());

    let avg_w_z1: f64 = sizes_z1.iter().map(|(w, _)| w).sum::<f64>() / sizes_z1.len() as f64;
    let avg_w_zhalf: f64 =
        sizes_zhalf.iter().map(|(w, _)| w).sum::<f64>() / sizes_zhalf.len() as f64;

    // At zoom=0.5, blobs should be ~0.5x
    let ratio = avg_w_zhalf / avg_w_z1;
    assert!(
        ratio > 0.3 && ratio < 0.7,
        "Zoom=0.5 blobs should be ~0.5x zoom=1: avg_z1={:.4}, avg_zhalf={:.4}, ratio={:.2}",
        avg_w_z1, avg_w_zhalf, ratio
    );
}

#[test]
fn test_terrain_blobs_are_roughly_circular() {
    let state = make_simple_state();
    let episode = make_svg_episode_with_zoom(state, 80, 14, 1.0);
    let config = svg_default_config();
    let svg = render_highway_svg_from_episode(&episode, 0.0, &config).unwrap();

    let sizes = extract_terrain_blob_sizes(&svg);
    assert!(!sizes.is_empty());

    // Since rx == ry, blobs should be roughly circular (aspect ratio near 1)
    let avg_aspect: f64 = sizes
        .iter()
        .map(|(w, h)| if *w > *h { w / h } else { h / w })
        .sum::<f64>()
        / sizes.len() as f64;

    assert!(
        avg_aspect < 2.0,
        "Blobs should be roughly circular (avg aspect < 2.0), got {:.2}",
        avg_aspect
    );
}

#[test]
fn test_terrain_blobs_present_at_high_zoom() {
    // At high zoom, blobs are large in SVG units. The cull margin must be
    // large enough that blobs near the edge aren't incorrectly removed.
    let state = make_simple_state();
    let episode = make_svg_episode_with_zoom(state, 80, 14, 4.0);
    let config = svg_default_config();

    let svg = render_highway_svg_from_episode(&episode, 0.0, &config).unwrap();
    let terrain_fill = format!("fill=\"{}\"", SceneColorConfig::default().terrain);
    let blob_count = svg.matches(terrain_fill.as_str()).count();

    assert!(
        blob_count >= 1,
        "Should have terrain blobs even at zoom=4, got {}",
        blob_count
    );
}
