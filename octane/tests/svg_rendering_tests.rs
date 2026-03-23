//! SVG rendering tests: vehicles, terrain, determinism, config, edge cases.

mod common;

use common::{
    make_crashed_state, make_simple_state, make_state_with_npcs, render_state, svg_config_with_size,
    svg_default_config,
};
use octane::config::SceneColorConfig;
use octane::data::{FrameState, VehicleState};
use octane::render::highway_svg::HighwayRenderConfig;

// =============================================================================
// Vehicle Rendering Tests
// =============================================================================

#[test]
fn test_svg_contains_ego_vehicle() {
    let state = make_simple_state();
    let svg = render_state(&state, &svg_default_config());
    assert!(svg.contains("fill=\"#b4b4b4\""), "Should have ego vehicle color");
}

#[test]
fn test_svg_crashed_ego_is_red() {
    let state = make_crashed_state();
    let svg = render_state(&state, &svg_default_config());
    let red_count = svg.matches("fill=\"#ff6464\"").count();
    assert!(red_count >= 1, "Crashed ego should be red");
}

#[test]
fn test_svg_contains_npc_vehicles() {
    let state = make_state_with_npcs(3);
    let svg = render_state(&state, &svg_default_config());

    // With OKLCH golden angle colors, each NPC gets a distinct hue
    let npc_colors: Vec<String> = (0..3)
        .map(|i| {
            let hex = octane::render::color::vehicle_color_hex(i, 0.75, 0.15);
            format!("fill=\"{}\"", hex)
        })
        .collect();

    for color in &npc_colors {
        assert!(svg.contains(color), "Should contain NPC color {}", color);
    }
}

#[test]
fn test_svg_vehicle_count_matches_npcs() {
    for n in 0..6 {
        let state = make_state_with_npcs(n);
        let svg = render_state(&state, &svg_default_config());

        // Each vehicle has a unique gradient def with id="cg_v{N}"
        let vehicle_count = (0..=n)
            .filter(|&i| svg.contains(&format!("id=\"cg_v{}\"", i)))
            .count();
        assert_eq!(vehicle_count, n + 1,
            "With {} NPCs, should have {} vehicle gradient defs (cg_v0..cg_v{})", n, n + 1, n);
    }
}

#[test]
fn test_svg_vehicles_are_car_groups() {
    let state = make_state_with_npcs(2);
    let svg = render_state(&state, &svg_default_config());
    assert!(svg.contains("<g transform="),
        "Vehicles should be rendered as <g transform=...> groups");
    // Each vehicle should have a gradient def
    assert!(svg.contains("id=\"cg_v0\""), "Should have ego gradient def cg_v0");
    assert!(svg.contains("id=\"cg_v1\""), "Should have NPC 0 gradient def cg_v1");
    assert!(svg.contains("id=\"cg_v2\""), "Should have NPC 1 gradient def cg_v2");
}

// =============================================================================
// Terrain Tests
// =============================================================================

#[test]
fn test_svg_contains_terrain_paths() {
    let state = make_simple_state();
    let svg = render_state(&state, &svg_default_config());
    assert!(svg.contains("<path"), "Should contain terrain path elements");
}

#[test]
fn test_svg_terrain_uses_bezier_curves() {
    let state = make_simple_state();
    let svg = render_state(&state, &svg_default_config());
    assert!(svg.contains("Q"), "Terrain should use quadratic bezier curves");
}

#[test]
fn test_svg_terrain_has_dark_fill() {
    let state = make_simple_state();
    let svg = render_state(&state, &svg_default_config());
    let terrain = format!("fill=\"{}\"", SceneColorConfig::default().terrain);
    assert!(svg.contains(&terrain), "Terrain blobs should have dark green fill");
}

#[test]
fn test_svg_terrain_paths_are_closed() {
    let state = make_simple_state();
    let svg = render_state(&state, &svg_default_config());
    let terrain_close = format!("Z\" fill=\"{}\"", SceneColorConfig::default().terrain);
    let path_with_z = svg.matches(&terrain_close.as_str()).count();
    assert!(path_with_z > 0, "Terrain paths should be closed (end with Z)");
}

#[test]
fn test_svg_no_pixel_rects_for_terrain() {
    let state = make_simple_state();
    let config = svg_config_with_size(200, 50);
    let svg = render_state(&state, &config);

    // 1 ego vehicle * 4 wheel rects + background + road = 6 rects expected.
    // Should not have hundreds of pixel rects for terrain.
    let rect_count = svg.matches("<rect").count();
    assert!(rect_count < 20, "Should not have pixel rects for terrain: {} rects", rect_count);
}

// =============================================================================
// Determinism Tests
// =============================================================================

#[test]
fn test_svg_is_deterministic() {
    let state = make_simple_state();
    let config = svg_default_config();

    let svg1 = render_state(&state, &config);
    let svg2 = render_state(&state, &config);

    assert_eq!(svg1, svg2, "Same input should produce identical SVG");
}

#[test]
fn test_svg_terrain_deterministic_with_position() {
    let state1 = FrameState {
        crashed: false,
        ego: VehicleState { x: 100.0, y: 4.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None },
        npcs: vec![],
        ..Default::default()
    };

    let state2 = FrameState {
        crashed: false,
        ego: VehicleState { x: 100.0, y: 4.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None },
        npcs: vec![],
        ..Default::default()
    };

    let config = svg_default_config();
    let svg1 = render_state(&state1, &config);
    let svg2 = render_state(&state2, &config);

    assert_eq!(svg1, svg2, "Same position should produce same terrain");
}

#[test]
fn test_svg_different_positions_different_terrain() {
    let state1 = FrameState {
        crashed: false,
        ego: VehicleState { x: 0.0, y: 4.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None },
        npcs: vec![],
        ..Default::default()
    };

    let state2 = FrameState {
        crashed: false,
        ego: VehicleState { x: 1000.0, y: 4.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None },
        npcs: vec![],
        ..Default::default()
    };

    let config = svg_default_config();
    let svg1 = render_state(&state1, &config);
    let svg2 = render_state(&state2, &config);

    assert_ne!(svg1, svg2, "Different positions should produce different SVGs");
}

// =============================================================================
// Config Tests
// =============================================================================

#[test]
fn test_svg_config_default_values() {
    let config = HighwayRenderConfig::default();
    assert_eq!(config.n_cols, 80);
    assert_eq!(config.n_rows, 14);
    assert_eq!(config.n_lanes, 4);
}

#[test]
fn test_svg_different_lane_count() {
    let state = make_simple_state();

    let config_4lanes = HighwayRenderConfig { n_lanes: 4, ..Default::default() };
    let config_6lanes = HighwayRenderConfig { n_lanes: 6, ..Default::default() };

    let svg4 = render_state(&state, &config_4lanes);
    let svg6 = render_state(&state, &config_6lanes);

    let lines4 = svg4.matches("<line").count();
    let lines6 = svg6.matches("<line").count();

    assert!(lines6 > lines4,
        "6 lanes should have more lines than 4 lanes (got {} vs {})", lines6, lines4);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_svg_zero_npcs() {
    let state = make_state_with_npcs(0);
    let svg = render_state(&state, &svg_default_config());

    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("</svg>"));

    // Only ego vehicle: cg_v0 should be present, cg_v1 should not
    assert!(svg.contains("id=\"cg_v0\""), "Should have ego gradient def");
    assert!(!svg.contains("id=\"cg_v1\""), "Should not have NPC gradient defs");
}

#[test]
fn test_svg_many_npcs() {
    let state = make_state_with_npcs(20);
    let svg = render_state(&state, &svg_default_config());

    assert!(svg.starts_with("<svg"));

    // Each vehicle has a unique gradient def: cg_v0 (ego) through cg_v20 (NPC 19)
    let vehicle_count = (0..=20)
        .filter(|&i| svg.contains(&format!("id=\"cg_v{}\"", i)))
        .count();
    assert_eq!(vehicle_count, 21, "Should have ego + 20 NPCs");
}

#[test]
fn test_svg_vehicle_at_origin() {
    let state = FrameState {
        crashed: false,
        ego: VehicleState { x: 0.0, y: 0.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None },
        npcs: vec![],
        ..Default::default()
    };

    let svg = render_state(&state, &svg_default_config());
    assert!(svg.starts_with("<svg"), "Should handle vehicle at origin");
}

#[test]
fn test_svg_vehicle_far_away() {
    let state = FrameState {
        crashed: false,
        ego: VehicleState { x: 10000.0, y: 4.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None },
        npcs: vec![],
        ..Default::default()
    };

    let svg = render_state(&state, &svg_default_config());
    assert!(svg.starts_with("<svg"), "Should handle vehicle far away");
}

#[test]
fn test_svg_vehicle_with_heading() {
    let state = FrameState {
        crashed: false,
        ego: VehicleState { x: 50.0, y: 4.0, heading: 0.5, speed: 0.0, acceleration: 0.0, attention: None },
        npcs: vec![],
        ..Default::default()
    };

    let svg = render_state(&state, &svg_default_config());
    assert!(svg.contains("<g transform="), "Vehicle with heading should be a <g> group");
    assert!(svg.contains("rotate("), "Vehicle with heading should have a rotate transform");
}

#[test]
fn test_svg_minimum_dimensions() {
    let state = make_simple_state();
    let config = svg_config_with_size(10, 5);

    let svg = render_state(&state, &config);
    assert!(svg.starts_with("<svg"), "Should handle minimum dimensions");
}

#[test]
fn test_svg_large_dimensions() {
    let state = make_simple_state();
    let config = svg_config_with_size(500, 200);

    let svg = render_state(&state, &config);
    assert!(svg.starts_with("<svg"), "Should handle large dimensions");

    let size_kb = svg.len() / 1024;
    assert!(size_kb < 500, "Large dimensions should still be under 500KB, got {}KB", size_kb);
}
