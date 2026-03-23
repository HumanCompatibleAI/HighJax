//! SVG structure tests: basic tags, road elements, element counts, sizes.

mod common;

use common::{
    make_simple_state, make_state_with_npcs, render_state, svg_config_with_size, svg_default_config,
};
use octane::config::SceneColorConfig;

// =============================================================================
// Basic SVG Structure Tests
// =============================================================================

#[test]
fn test_svg_starts_with_svg_tag() {
    let state = make_simple_state();
    let svg = render_state(&state, &svg_default_config());
    assert!(svg.starts_with("<svg"), "SVG should start with <svg tag");
}

#[test]
fn test_svg_ends_with_svg_tag() {
    let state = make_simple_state();
    let svg = render_state(&state, &svg_default_config());
    assert!(svg.trim().ends_with("</svg>"), "SVG should end with </svg> tag");
}

#[test]
fn test_svg_has_xmlns() {
    let state = make_simple_state();
    let svg = render_state(&state, &svg_default_config());
    assert!(svg.contains("xmlns=\"http://www.w3.org/2000/svg\""),
        "SVG should have proper xmlns attribute");
}

#[test]
fn test_svg_has_viewbox() {
    let state = make_simple_state();
    let svg = render_state(&state, &svg_default_config());
    assert!(svg.contains("viewBox"), "SVG should have viewBox attribute");
}

#[test]
fn test_svg_has_width_and_height() {
    let state = make_simple_state();
    let svg = render_state(&state, &svg_default_config());
    assert!(svg.contains("width="), "SVG should have width attribute");
    assert!(svg.contains("height="), "SVG should have height attribute");
}

#[test]
fn test_svg_has_preserve_aspect_ratio() {
    let state = make_simple_state();
    let svg = render_state(&state, &svg_default_config());
    assert!(svg.contains("preserveAspectRatio=\"none\""),
        "SVG should have preserveAspectRatio=none for proper scaling");
}

// =============================================================================
// Road Element Tests
// =============================================================================

#[test]
fn test_svg_contains_background_rect() {
    let state = make_simple_state();
    let svg = render_state(&state, &svg_default_config());
    let bg = format!("fill=\"{}\"", SceneColorConfig::default().background);
    assert!(svg.contains(&bg), "Should have dark brown background");
}

#[test]
fn test_svg_contains_road_surface() {
    let state = make_simple_state();
    let svg = render_state(&state, &svg_default_config());
    let road = format!("fill=\"{}\"", SceneColorConfig::default().road_surface);
    assert!(svg.contains(&road), "Should have road surface color");
}

#[test]
fn test_svg_contains_road_edges() {
    let state = make_simple_state();
    let svg = render_state(&state, &svg_default_config());
    let lane = format!("stroke=\"{}\"", SceneColorConfig::default().lane_divider);
    assert!(svg.contains(&lane), "Should have lane divider color");
}

#[test]
fn test_svg_contains_lane_dividers() {
    let state = make_simple_state();
    let svg = render_state(&state, &svg_default_config());
    assert!(svg.contains("<line"), "Should have lane divider lines");
}

#[test]
fn test_svg_lane_dividers_are_lines() {
    let state = make_simple_state();
    let svg = render_state(&state, &svg_default_config());
    let line_count = svg.matches("<line").count();
    assert!(line_count >= 2, "Should have line elements for road markings");
}

// =============================================================================
// Element Count Tests
// =============================================================================

#[test]
fn test_svg_element_counts_reasonable() {
    let state = make_state_with_npcs(5);
    let config = svg_config_with_size(120, 40);
    let svg = render_state(&state, &config);

    let rect_count = svg.matches("<rect").count();
    let path_count = svg.matches("<path").count();
    let line_count = svg.matches("<line").count();

    // 6 vehicles * (4 wheel + 2 headlight) rects each = 36, plus 1 background + 1 road = 38
    assert!(rect_count >= 20 && rect_count <= 40,
        "Rect count should be 20-40 (car rects + background + road), got {}", rect_count);
    // Each car has ~20 paths plus terrain blobs
    assert!(path_count > 50 && path_count < 300,
        "Path count should be 50-300 (car paths + terrain), got {}", path_count);
    // Lines include road edges, lane dividers, and linearGradient defs (headlights + brakelights)
    assert!(line_count >= 1 && line_count <= 120,
        "Line count should be 1-120, got {}", line_count);
}

// =============================================================================
// Size Tests
// =============================================================================

#[test]
fn test_svg_size_reasonable_across_dimensions() {
    let state = make_simple_state();

    let small = render_state(&state, &svg_config_with_size(40, 20));
    let large = render_state(&state, &svg_config_with_size(200, 80));

    let small_kb = small.len() / 1024;
    let large_kb = large.len() / 1024;

    assert!(small_kb < 500, "Small config SVG should be under 500KB, got {}KB", small_kb);
    assert!(large_kb < 500, "Large config SVG should be under 500KB, got {}KB", large_kb);

    let ratio = (large.len() as f64) / (small.len() as f64);
    assert!(ratio > 0.2 && ratio < 5.0,
        "SVG sizes should be in reasonable ratio (same scene), got ratio {:.2}", ratio);
}

#[test]
fn test_svg_size_reasonable() {
    let state = make_state_with_npcs(10);
    let config = svg_config_with_size(200, 50);
    let svg = render_state(&state, &config);

    let size_kb = svg.len() / 1024;
    assert!(size_kb < 500, "SVG should be under 500KB, got {}KB", size_kb);
}
