//! Highway SVG generation with proper vehicle rendering.
//!
//! Ports the Python `render_highway_to_svg()` from highjax for high-quality
//! highway visualization with vehicles, lanes, and terrain.
//!
//! Uses `render_highway_svg_from_episode(SvgEpisode, scene_time, config)` for episode-based
//! rendering with consistent coordinate transforms via the worlds architecture.
//! See docs/Octane/ for coordinate system design details.

use super::color::{color_to_hex, parse_hex_color, vehicle_color_hex};
use crate::worlds::{ScenePoint, SvgEpisode};

/// Configuration for highway rendering.
#[derive(Debug, Clone)]
pub struct HighwayRenderConfig {
    pub n_cols: u32,
    pub n_rows: u32,
    pub n_lanes: usize,
    pub lane_width: f64,
    pub corn_aspro: f64,
    pub pixels_per_corn_diagonal: f64,
    /// Terrain grid scale in meters (controls blob spacing).
    pub terrain_scale: f64,
    /// Whether to show the podium marker (red line at ego target position).
    pub show_podium_marker: bool,
    /// Whether to show velocity arrows on vehicles.
    pub show_velocity_arrows: bool,
    /// Velocity arrow visual config.
    pub velocity_arrows: crate::config::VelocityArrowsConfig,
    /// Background color (hex string).
    pub color_background: String,
    /// Terrain blob fill color (hex string).
    pub color_terrain: String,
    /// Road surface fill color (hex string).
    pub color_road_surface: String,
    /// Road edge stroke color (hex string).
    pub color_road_edge: String,
    /// Lane divider stroke color (hex string).
    pub color_lane_divider: String,
    /// Ego vehicle color (hex string).
    pub ego_color: String,
    /// Ego vehicle color when crashed (hex string).
    pub ego_crashed_color: String,
    /// NPC vehicle OKLCH lightness.
    pub npc_lightness: f64,
    /// NPC vehicle OKLCH chroma.
    pub npc_chroma: f64,
    /// Vehicle length in meters.
    pub vehicle_length: f64,
    /// Vehicle width in meters.
    pub vehicle_width: f64,
    /// Road edge stroke width.
    pub stroke_edge: f64,
    /// Lane divider stroke width.
    pub stroke_lane: f64,
    /// Whether lane dividers are dashed (light theme).
    pub lane_divider_dashed: bool,
    /// Window color for car SVG template (hex string).
    pub window_color: String,
    /// Multiplier for road edge border stroke.
    pub stroke_edge_border_multiplier: f64,
    /// Podium marker stroke width.
    pub stroke_podium_marker: f64,
    /// Terrain blob density threshold (0-1).
    pub terrain_density: f64,
    /// Terrain blob minimum size fraction.
    pub terrain_blob_size_min: f64,
    /// Terrain blob size variation range fraction.
    pub terrain_blob_size_range: f64,
    /// Whether to show attention overlay.
    pub show_attention: bool,
    /// Debug eye mode: fixed opacities instead of real attention weights.
    pub debug_eye: bool,
    /// Hardcoded action arrow to render (e.g. "left", "right", "faster", "slower").
    pub hardcoded_action: Option<String>,
    /// Hardcoded arrow color (from theme).
    pub hardcoded_arrow_color: String,
    /// Hardcoded arrow geometry config.
    pub hardcoded_arrow: crate::config::HardcodedArrowConfig,
    /// Whether headlight cones are enabled.
    pub headlights: bool,
    /// Headlight cone base opacity (0.0-1.0).
    pub headlight_opacity: f64,
    /// Brakelight cone base opacity (0.0-1.0).
    pub brakelight_opacity: f64,
    /// SVG blend mode for headlight and brakelight cones.
    pub light_blend_mode: String,
    /// Deceleration threshold (m/s², positive) for brakelight activation.
    pub brakelight_deceleration_threshold_m_s2: f64,
    /// Whether to show action distribution on ego vehicle.
    pub show_action_distribution: bool,
    /// Whether to show action distribution percentage text.
    pub show_action_distribution_text: bool,
    /// Whether playback is paused (for text overlay).
    pub is_paused: bool,
    /// Action distribution visual config.
    pub action_distribution: crate::config::ActionDistributionConfig,
    /// Whether to show NPC text labels on vehicles.
    pub show_npc_text: bool,
    /// NPC text label visual config.
    pub npc_text: crate::config::NpcTextConfig,
    /// Ego speed range (min, max) in m/s for speed coloring.
    pub ego_speed_range: (f64, f64),
    /// Whether to show the scale bar in the bottom-right corner.
    pub show_scala: bool,
    /// Scale bar foreground color (hex string).
    pub color_scale_bar_fg: String,
    /// Scale bar background stroke color (hex string).
    pub color_scale_bar_bg: String,
    /// Headlight cone color (hex string).
    pub color_headlight: String,
    /// Brakelight cone color (hex string).
    pub color_brakelight: String,
}

impl Default for HighwayRenderConfig {
    fn default() -> Self {
        Self::from_config(&crate::config::Config::default(), crate::config::SceneTheme::Dark)
    }
}

impl HighwayRenderConfig {
    /// Build a render config from the central Config, setting only the config-derived fields.
    /// n_cols and n_rows are set separately (they depend on terminal size).
    /// Both dark and light scene colors come from the config file (user-customizable).
    pub fn from_config(config: &crate::config::Config, theme: crate::config::SceneTheme) -> Self {
        let scene = match theme {
            crate::config::SceneTheme::Dark => &config.octane.colors.scene_themes.dark,
            crate::config::SceneTheme::Light => &config.octane.colors.scene_themes.light,
        };
        Self {
            n_lanes: config.octane.road.n_lanes,
            lane_width: config.octane.road.lane_width,
            corn_aspro: config.octane.rendering.corn_aspro,
            pixels_per_corn_diagonal: config.octane.rendering.pixels_per_corn_diagonal,
            terrain_scale: config.octane.terrain.scale,
            show_podium_marker: config.octane.podium.show_marker,
            show_velocity_arrows: config.octane.rendering.velocity_arrows != crate::config::DisplayMode::Never,
            velocity_arrows: config.octane.velocity_arrows.clone(),
            color_background: color_to_hex(&scene.background),
            color_terrain: color_to_hex(&scene.terrain),
            color_road_surface: color_to_hex(&scene.road_surface),
            color_road_edge: color_to_hex(&scene.road_edge),
            color_lane_divider: color_to_hex(&scene.lane_divider),
            ego_color: color_to_hex(&scene.ego),
            ego_crashed_color: color_to_hex(&scene.ego_crashed),
            npc_lightness: scene.npc_lightness,
            npc_chroma: scene.npc_chroma,
            vehicle_length: config.octane.road.vehicle_length,
            vehicle_width: config.octane.road.vehicle_width,
            stroke_edge: config.octane.road.edge_stroke,
            stroke_lane: config.octane.road.lane_stroke,
            lane_divider_dashed: scene.lane_divider_dashed,
            window_color: color_to_hex(&scene.window),
            stroke_edge_border_multiplier: config.octane.road.edge_border_multiplier,
            stroke_podium_marker: config.octane.podium.marker_stroke,
            terrain_density: config.octane.terrain.density,
            terrain_blob_size_min: config.octane.terrain.blob_size_min,
            terrain_blob_size_range: config.octane.terrain.blob_size_range,
            show_attention: config.octane.attention.show,
            hardcoded_action: None,
            hardcoded_arrow_color: scene.hardcoded_arrow_color.clone(),
            hardcoded_arrow: config.octane.hardcoded_arrow.clone(),
            headlights: scene.headlights,
            headlight_opacity: config.octane.rendering.headlight_opacity,
            brakelight_opacity: config.octane.rendering.brakelight_opacity,
            light_blend_mode: scene.light_blend_mode.clone(),
            brakelight_deceleration_threshold_m_s2: config.octane.rendering.brakelight_deceleration_threshold_m_s2,
            n_cols: 80,
            n_rows: 14,
            debug_eye: false,
            show_action_distribution: config.octane.rendering.action_distribution != crate::config::DisplayMode::Never,
            show_action_distribution_text: config.octane.rendering.action_distribution_text != crate::config::DisplayMode::Never,
            is_paused: true,
            action_distribution: {
                let mut ad = config.octane.action_distribution.clone();
                ad.color = scene.action_color.clone();
                ad.chosen_color = scene.action_chosen_color.clone();
                ad
            },
            show_npc_text: config.octane.rendering.npc_text != crate::config::DisplayMode::Never,
            npc_text: config.octane.npc_text.clone(),
            ego_speed_range: (0.0, 30.0),
            show_scala: config.octane.rendering.show_scala,
            color_scale_bar_fg: color_to_hex(&scene.scale_bar_fg),
            color_scale_bar_bg: color_to_hex(&scene.scale_bar_bg),
            color_headlight: color_to_hex(&scene.headlight),
            color_brakelight: color_to_hex(&scene.brakelight),
        }
    }
}

use super::brakelights::render_brakelight_cones;
use super::terrain::generate_terrain_blobs_from_bounds;
use super::headlights::render_headlight_cones;
use super::velocity_arrows::render_velocity_arrow_svg;
use super::action_distribution::{render_action_distribution_svg, render_action_delta_svg};


/// Render a full-width horizontal SVG line at the given y coordinate.
fn horizontal_line(y: f64, width: f64, color: &str, stroke_width: f64) -> String {
    format!(
        r#"<line x1="0" y1="{:.6}" x2="{}" y2="{:.6}" stroke="{}" stroke-width="{:.6}"/>"#,
        y, width, y, color, stroke_width
    )
}

/// Pick the largest "nice" number of meters (1, 2, 5 × 10^k) that fits within
/// `max_svg_width` SVG units at the given scale (SVG units per meter).
fn nice_scale_meters(max_svg_width: f64, scale: f64) -> f64 {
    let max_meters = max_svg_width / scale;
    if max_meters <= 0.0 {
        return 1.0;
    }
    let pow = 10.0_f64.powf(max_meters.log10().floor());
    let mantissa = max_meters / pow;
    let nice = if mantissa >= 5.0 {
        5.0
    } else if mantissa >= 2.0 {
        2.0
    } else {
        1.0
    };
    nice * pow
}

/// Render attention weight bars on NPC vehicles.
fn render_attention_bars_svg(
    elements: &mut Vec<String>,
    npcs: &[crate::data::jsonla::VehicleState],
    vehicle_length: f64,
    vehicle_width: f64,
    debug_eye: bool,
    world_to_svg: &impl Fn(f64, f64) -> (f64, f64),
) {
    let debug_levels = [1.0_f64, 0.6, 0.3];
    let bar_w_frac = 0.60;
    let bar_h_frac = 0.432;
    let bar_w = vehicle_length * bar_w_frac;
    let bar_h = vehicle_width * bar_h_frac;

    let render_bar = |npc: &crate::data::jsonla::VehicleState, fill: f64, elements: &mut Vec<String>| {
        let (npc_sx, npc_sy) = world_to_svg(npc.x, npc.y);
        let (px, py) = world_to_svg(npc.x + 1.0, npc.y);
        let svg_per_meter = ((px - npc_sx).powi(2) + (py - npc_sy).powi(2)).sqrt();
        if svg_per_meter < 1e-6 { return; }
        let angle = npc.heading.to_degrees();
        let sw = bar_w * svg_per_meter;
        let sh = bar_h * svg_per_meter;
        let fill_w = sw * fill.clamp(0.0, 1.0);
        let border = 0.15 * svg_per_meter;
        let hw = sw / 2.0;
        let hh = sh / 2.0;

        let fill_rect = if fill_w > 1e-6 {
            format!(r##"<rect x="{:.6}" y="{:.6}" width="{:.6}" height="{:.6}" fill="#ffd700"/>"##, -hw, -hh, fill_w, sh)
        } else {
            String::new()
        };
        elements.push(format!(
            concat!(
                r##"<g transform="translate({cx:.6},{cy:.6}) rotate({a:.2})" opacity="0.8">"##,
                r##"<rect x="{x:.6}" y="{y:.6}" width="{w:.6}" height="{h:.6}" fill="#fff"/>"##,
                "{fill_rect}",
                r##"<rect x="{x:.6}" y="{y:.6}" width="{w:.6}" height="{h:.6}" fill="none" stroke="#000" stroke-width="{bw:.6}"/>"##,
                r##"</g>"##,
            ),
            cx = npc_sx, cy = npc_sy, a = angle,
            x = -hw, y = -hh, w = sw, h = sh, bw = border, fill_rect = fill_rect,
        ));
    };

    if debug_eye {
        for (i, npc) in npcs.iter().enumerate() {
            render_bar(npc, debug_levels[i % 3], elements);
        }
    } else {
        for npc in npcs {
            if let Some(weight) = npc.attention {
                render_bar(npc, weight, elements);
            }
        }
    }
}

/// Render a scale bar in the bottom-right corner of the SVG.
fn render_scale_bar_svg(
    elements: &mut Vec<String>,
    svg_width: f64,
    svg_height: f64,
    scale: f64,
    font_family: &str,
    fg_color: &str,
    bg_color: &str,
) {
    let margin = svg_width * 0.02;
    let max_bar_svg = svg_width * 0.25;
    let bar_meters = nice_scale_meters(max_bar_svg, scale);
    let bar_svg = bar_meters * scale;
    let tick_h = svg_height * 0.015;
    let stroke_w = svg_width * 0.003;
    let font_sz = svg_height * 0.03;

    let x_right = svg_width - margin;
    let x_left = x_right - bar_svg;
    let y_bar = svg_height - margin;
    let y_tick_top = y_bar - tick_h;
    let y_label = y_bar - tick_h - font_sz * 0.3;

    let label = if bar_meters >= 1.0 {
        format!("{}m", bar_meters as u64)
    } else {
        format!("{:.1}m", bar_meters)
    };

    let fg = fg_color;
    let bg = bg_color;
    let bg_stroke = format!(
        r#" stroke="{}" stroke-width="{:.6}" paint-order="stroke" stroke-linejoin="round""#,
        bg, stroke_w * 3.0
    );

    // Bar line
    elements.push(format!(
        r#"<line x1="{:.6}" y1="{:.6}" x2="{:.6}" y2="{:.6}" stroke="{}" stroke-width="{:.6}" opacity="0.5"/>"#,
        x_left, y_bar, x_right, y_bar, fg, stroke_w
    ));
    // Left tick
    elements.push(format!(
        r#"<line x1="{:.6}" y1="{:.6}" x2="{:.6}" y2="{:.6}" stroke="{}" stroke-width="{:.6}" opacity="0.5"/>"#,
        x_left, y_tick_top, x_left, y_bar, fg, stroke_w
    ));
    // Right tick
    elements.push(format!(
        r#"<line x1="{:.6}" y1="{:.6}" x2="{:.6}" y2="{:.6}" stroke="{}" stroke-width="{:.6}" opacity="0.5"/>"#,
        x_right, y_tick_top, x_right, y_bar, fg, stroke_w
    ));
    // Label
    let x_center = (x_left + x_right) / 2.0;
    elements.push(format!(
        r#"<text x="{:.6}" y="{:.6}" fill="{}"{} font-size="{:.6}" font-family="{}" font-weight="bold" text-anchor="middle" dominant-baseline="auto" data-mango-fg="{}" data-mango-bg="{}">{}</text>"#,
        x_center, y_label, fg, bg_stroke, font_sz, font_family, fg, bg, label
    ));
}

/// Render highway scene using the worlds coordinate system.
///
/// Uses SvgEpisode for coordinate transforms with viewport tracking.
///
/// # Arguments
/// * `svg_episode` - The SvgEpisode containing viewport and scene data
/// * `scene_time` - Time in the episode to render
/// * `config` - Rendering configuration (colors, road params)
pub fn render_highway_svg_from_episode(
    svg_episode: &SvgEpisode,
    scene_time: f64,
    config: &HighwayRenderConfig,
) -> Option<String> {
    let state = svg_episode.state_at(scene_time)?;

    // Snapped state: round to nearest discrete frame for stable TTC computation.
    // Interpolated headings between frames can create near-miss flicker.
    let dt = svg_episode.seconds_per_timestep();
    let snapped_time = (scene_time / dt).round() * dt;
    let snapped_state = svg_episode.state_at(snapped_time);

    let n_cols = config.n_cols;
    let n_rows = config.n_rows;
    let n_lanes = config.n_lanes;
    let lane_width = config.lane_width;
    let corn_aspro = config.corn_aspro;
    let pixels_per_diag = config.pixels_per_corn_diagonal;

    // Get SVG dimensions from episode
    let (svg_width, svg_height) = svg_episode.bounds();

    // Get viewport parameters from episode
    let viewport = svg_episode.viewport();
    let scale = viewport.scale();
    let view_x = viewport.view_x_at(scene_time)?;
    let view_y = viewport.view_y_at(scene_time)?;

    // Visible world bounds from episode
    let bounds = svg_episode.visible_bounds_at(scene_time)?;

    // Output pixel dimensions
    let aspro_factor = (1.0 + corn_aspro * corn_aspro).sqrt();
    let cell_pixel_w = pixels_per_diag / aspro_factor;
    let cell_pixel_h = pixels_per_diag * corn_aspro / aspro_factor;
    let pixel_width = (n_cols as f64 * cell_pixel_w) as u32;
    let pixel_height = (n_rows as f64 * cell_pixel_h) as u32;

    // Coordinate transform using SvgEpisode
    let world_to_svg = |wx: f64, wy: f64| -> (f64, f64) {
        let point = ScenePoint::new(wx, wy);
        if let Some(svg_pt) = svg_episode.scene_to_svg(point, scene_time) {
            (svg_pt.x, svg_pt.y)
        } else {
            // Fallback (shouldn't happen if scene_time is valid)
            let sx = (wx - view_x) * scale + svg_width / 2.0;
            let sy = (wy - view_y) * scale + svg_height / 2.0;
            (sx, sy)
        }
    };

    // Stroke widths in normalized units
    let stroke_scale = 1.0 / pixel_width as f64;
    let edge_stroke = config.stroke_edge * stroke_scale;
    let lane_stroke = config.stroke_lane * stroke_scale;

    // Road boundaries
    let road_min_y = -lane_width / 2.0;
    let road_max_y = (n_lanes as f64 - 0.5) * lane_width;

    let mut elements = Vec::new();

    // Background
    elements.push(format!(
        r##"<rect x="0" y="0" width="{}" height="{}" fill="{}"/>"##,
        svg_width, svg_height, &config.color_background
    ));

    // Terrain rendering using visible bounds from episode
    let terrain_blobs = generate_terrain_blobs_from_bounds(
        &bounds,
        svg_width,
        svg_height,
        scale,
        view_x,
        view_y,
        road_min_y,
        road_max_y,
        config.terrain_scale,
        config.terrain_density,
        config.terrain_blob_size_min,
        config.terrain_blob_size_range,
        &config.color_terrain,
    );
    elements.extend(terrain_blobs);

    // Road boundaries in SVG coords
    let (_, road_top_svg) = world_to_svg(0.0, road_min_y);
    let (_, road_bottom_svg) = world_to_svg(0.0, road_max_y);

    // Road surface
    let road_height = road_bottom_svg - road_top_svg;
    if road_height > 0.0 {
        elements.push(format!(
            r##"<rect x="0" y="{:.6}" width="{}" height="{:.6}" fill="{}"/>"##,
            road_top_svg, svg_width, road_height, &config.color_road_surface
        ));
    }

    // Road edges
    let edge_border_stroke = edge_stroke * config.stroke_edge_border_multiplier;
    let (_, edge_top_y) = world_to_svg(0.0, road_min_y);
    let (_, edge_bottom_y) = world_to_svg(0.0, road_max_y);
    elements.push(horizontal_line(edge_top_y, svg_width, &config.color_road_edge, edge_border_stroke));
    elements.push(horizontal_line(edge_bottom_y, svg_width, &config.color_road_edge, edge_border_stroke));

    // Lane dividers
    for lane_idx in 1..n_lanes {
        let lane_y = lane_idx as f64 * lane_width - lane_width / 2.0;
        let (_, svg_y) = world_to_svg(0.0, lane_y);
        if config.lane_divider_dashed {
            elements.push(format!(
                r#"<line x1="0" y1="{:.6}" x2="{}" y2="{:.6}" stroke="{}" stroke-width="{:.6}" stroke-dasharray="{:.2} {:.2}"/>"#,
                svg_y, svg_width, svg_y, &config.color_lane_divider, lane_stroke,
                lane_stroke * 12.0, lane_stroke * 12.0,
            ));
        } else {
            elements.push(horizontal_line(svg_y, svg_width, &config.color_lane_divider, lane_stroke));
        }
    }

    // Headlight cones with shadow occlusion
    let all_vehicles: Vec<(f64, f64, f64)> = std::iter::once(
        (state.ego.x, state.ego.y, state.ego.heading)
    ).chain(
        state.npcs.iter().map(|n| (n.x, n.y, n.heading))
    ).collect();
    let (headlight_defs, headlight_elements) = if config.headlights {
        render_headlight_cones(
            config, &all_vehicles, svg_width, svg_height, &world_to_svg,
        )
    } else {
        (Vec::new(), Vec::new())
    };

    // Brakelight cones: only for vehicles that are braking (decel below threshold)
    let threshold = -config.brakelight_deceleration_threshold_m_s2;
    let braking_vehicles: Vec<(f64, f64, f64)> = std::iter::once(&state.ego)
        .chain(state.npcs.iter())
        .filter(|v| v.acceleration < threshold)
        .map(|v| (v.x, v.y, v.heading))
        .collect();
    let (brakelight_defs, brakelight_elements) = render_brakelight_cones(
        config, &braking_vehicles, svg_width, svg_height, &world_to_svg,
    );

    // Wrap all light cones in a group with the configured blend mode
    let blend = &config.light_blend_mode;
    if blend != "normal" {
        elements.push(format!(r#"<g style="mix-blend-mode: {}">"#, blend));
        elements.extend(headlight_elements);
        elements.extend(brakelight_elements);
        elements.push("</g>".into());
    } else {
        elements.extend(headlight_elements);
        elements.extend(brakelight_elements);
    }

    // Render ego vehicle using car SVG template
    let ego_hex = if state.crashed {
        &config.ego_crashed_color
    } else {
        &config.ego_color
    };
    let (ego_r, ego_g, ego_b) = parse_hex_color(ego_hex);
    let mut svg_defs = Vec::new();
    svg_defs.extend(headlight_defs);
    svg_defs.extend(brakelight_defs);

    let (ego_defs, ego_elem) = crate::render::car_template::render_car(
        0, state.ego.x, state.ego.y, state.ego.heading,
        config.vehicle_length, config.vehicle_width,
        ego_r, ego_g, ego_b, &config.window_color, &world_to_svg,
    );
    svg_defs.push(ego_defs);
    elements.push(ego_elem);

    // Render NPC vehicles (colors computed from OKLCH with golden angle hue spacing)
    for (i, npc) in state.npcs.iter().enumerate() {
        let npc_hex = vehicle_color_hex(
            i, config.npc_lightness, config.npc_chroma,
        );
        let (r, g, b) = parse_hex_color(&npc_hex);
        let (npc_defs, npc_elem) = crate::render::car_template::render_car(
            i + 1, npc.x, npc.y, npc.heading,
            config.vehicle_length, config.vehicle_width,
            r, g, b, &config.window_color, &world_to_svg,
        );
        svg_defs.push(npc_defs);
        elements.push(npc_elem);
    }

    // Velocity arrows
    if config.show_velocity_arrows {
        let va = &config.velocity_arrows;
        let pw = pixel_width as f64;

        let make_arrow = |elements: &mut Vec<String>, vx: f64, vy: f64,
                          heading: f64, speed: f64, color: &str| {
            let arrow_length = speed * va.length_scale;
            // Stroke: scale by speed, clamp
            let stroke_px = (va.stroke_width * (1.0 + speed * va.stroke_speed_factor))
                .clamp(va.stroke_min, va.stroke_max);
            let stroke_svg = stroke_px / pw;
            // Head: scale fraction by speed, convert to meters, clamp
            let head_frac = va.head_size * (1.0 + speed * va.head_speed_factor);
            let head_meters = (arrow_length * head_frac)
                .clamp(va.head_min_meters, va.head_max_meters);
            render_velocity_arrow_svg(
                elements, vx, vy, heading, arrow_length,
                stroke_svg, color, va.opacity, head_meters, va.head_angle,
                &world_to_svg,
            );
        };

        // Ego arrow (same color as ego vehicle)
        if state.ego.speed.abs() > va.min_speed {
            make_arrow(&mut elements, state.ego.x, state.ego.y,
                       state.ego.heading, state.ego.speed, ego_hex);
        }
        // NPC arrows (same color as each NPC's vehicle)
        for (i, npc) in state.npcs.iter().enumerate() {
            if npc.speed.abs() > va.min_speed {
                let npc_color = vehicle_color_hex(
                    i, config.npc_lightness, config.npc_chroma,
                );
                make_arrow(&mut elements, npc.x, npc.y,
                           npc.heading, npc.speed, &npc_color);
            }
        }
    }

    // Action distribution overlay on ego vehicle
    if let Some(ref old_dist) = state.old_action_distribution {
        // Delta overlay (always shown when old_action_distribution is set)
        if let Some(ref dist) = state.action_distribution {
            let ad = &config.action_distribution;
            render_action_delta_svg(
                &mut elements,
                state.ego.x,
                state.ego.y,
                state.ego.heading,
                config.vehicle_length,
                config.vehicle_width,
                &old_dist.probs,
                &dist.probs,
                ad,
                pixel_width as f64,
                &world_to_svg,
            );
        }
    } else if config.show_action_distribution {
        if let Some(ref dist) = state.action_distribution {
            let ad = &config.action_distribution;
            let show_text = config.show_action_distribution_text;
            render_action_distribution_svg(
                &mut elements,
                state.ego.x,
                state.ego.y,
                state.ego.heading,
                config.vehicle_length,
                config.vehicle_width,
                dist,
                ad,
                state.chosen_action,
                show_text,
                pixel_width as f64,
                &world_to_svg,
            );
        }
    }

    // Hardcoded action arrow (filled polygon, separate from action distribution)
    if let Some(ref action_name) = config.hardcoded_action {
        render_hardcoded_arrow(
            &mut elements,
            state.ego.x, state.ego.y,
            config.vehicle_length, config.vehicle_width,
            action_name,
            &config.hardcoded_arrow_color,
            &config.hardcoded_arrow,
            &world_to_svg,
        );
    }

    // Attention overlay: progress bar on each NPC car
    if config.show_attention {
        render_attention_bars_svg(
            &mut elements, &state.npcs, config.vehicle_length, config.vehicle_width,
            config.debug_eye, &world_to_svg,
        );
    }

    // NPC text labels
    if config.show_npc_text {
        let snap = snapped_state.as_ref().unwrap_or(&state);
        super::npc_text::render_npc_text_svg(
            &mut elements,
            &state.ego,
            &state.npcs,
            &snap.ego,
            &snap.npcs,
            config.npc_lightness,
            config.npc_chroma,
            &config.npc_text,
            config.ego_speed_range,
            config.vehicle_length,
            config.vehicle_width,
            pixel_width as f64,
            &world_to_svg,
        );
    }

    // Draw podium marker (red line at 30% from left) if enabled
    if config.show_podium_marker {
        let ideal_svg_x = svg_width * 0.2;
        let line_stroke = config.stroke_podium_marker / pixel_width as f64;
        elements.push(format!(
            r#"<line x1="{:.6}" y1="0" x2="{:.6}" y2="{}" stroke="red" stroke-width="{:.6}" opacity="0.7"/>"#,
            ideal_svg_x, ideal_svg_x, svg_height, line_stroke
        ));
    }

    // Scale bar in bottom-right corner
    if config.show_scala {
        render_scale_bar_svg(
            &mut elements, svg_width, svg_height, scale,
            &config.npc_text.font_family,
            &config.color_scale_bar_fg,
            &config.color_scale_bar_bg,
        );
    }

    // Build final SVG with defs block for vehicle gradients
    let defs_block = if svg_defs.is_empty() {
        String::new()
    } else {
        format!(
            "  <defs>\n{}\n  </defs>",
            svg_defs.iter().map(|d| format!("    {}", d.replace('\n', "\n    "))).collect::<Vec<_>>().join("\n")
        )
    };

    Some(format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{}" height="{}" viewBox="0 0 {} {}" preserveAspectRatio="none">
{}
{}
</svg>"#,
        pixel_width,
        pixel_height,
        svg_width,
        svg_height,
        defs_block,
        elements.iter().map(|e| format!("  {}", e)).collect::<Vec<_>>().join("\n")
    ))
}

/// Render a filled-polygon arrow for a hardcoded action.
///
/// Draws a solid arrow (shaft rectangle + triangular head) as a single polygon,
/// originating from the ego vehicle edge in the action direction.
#[allow(clippy::too_many_arguments)]
fn render_hardcoded_arrow<F>(
    elements: &mut Vec<String>,
    ego_x: f64,
    ego_y: f64,
    vehicle_length: f64,
    vehicle_width: f64,
    action_name: &str,
    color: &str,
    cfg: &crate::config::HardcodedArrowConfig,
    world_to_svg: &F,
) where
    F: Fn(f64, f64) -> (f64, f64),
{
    // Direction and base point for each action (road-aligned, not heading-rotated)
    let half_l = vehicle_length / 2.0;
    let half_w = vehicle_width / 2.0;
    let diag = std::f64::consts::FRAC_1_SQRT_2;
    let (dx, dy, bx, by): (f64, f64, f64, f64) = match action_name {
        "faster" => (1.0, 0.0, ego_x + half_l + 0.15, ego_y),
        "slower" => (-1.0, 0.0, ego_x - half_l, ego_y),
        "left" => (diag, -diag, ego_x + half_l, ego_y - half_w),   // 45° forward-left from front-left corner
        "right" => (diag, diag, ego_x + half_l, ego_y + half_w),   // 45° forward-right from front-right corner
        _ => return,
    };

    let length = cfg.length;
    let shaft_hw = cfg.shaft_width / 2.0;
    let head_hw = cfg.head_width / 2.0;
    let head_len = cfg.head_length;
    let shaft_len = length - head_len;

    // Perpendicular direction (90° CCW from arrow direction)
    let (px, py) = (-dy, dx);

    // 7-point polygon: shaft base → shaft sides → head base (wide) → tip
    let points_world: [(f64, f64); 7] = [
        (bx + px * shaft_hw, by + py * shaft_hw),                                           // shaft left base
        (bx + dx * shaft_len + px * shaft_hw, by + dy * shaft_len + py * shaft_hw),         // shaft left top
        (bx + dx * shaft_len + px * head_hw, by + dy * shaft_len + py * head_hw),           // head left
        (bx + dx * length, by + dy * length),                                               // tip
        (bx + dx * shaft_len - px * head_hw, by + dy * shaft_len - py * head_hw),           // head right
        (bx + dx * shaft_len - px * shaft_hw, by + dy * shaft_len - py * shaft_hw),         // shaft right top
        (bx - px * shaft_hw, by - py * shaft_hw),                                           // shaft right base
    ];

    let svg_points: String = points_world.iter()
        .map(|&(wx, wy)| {
            let (sx, sy) = world_to_svg(wx, wy);
            format!("{:.4},{:.4}", sx, sy)
        })
        .collect::<Vec<_>>()
        .join(" ");

    elements.push(format!(
        r#"<polygon points="{}" fill="{}" opacity="{:.2}"/>"#,
        svg_points, color, cfg.opacity,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{FrameState, VehicleState};
    use crate::worlds::{
        SceneEpisode, SvgConfig, SvgEpisode, ViewportConfig, ViewportEpisode,
        DEFAULT_CORN_ASPRO,
    };

    fn make_test_state() -> FrameState {
        FrameState {
            crashed: false,
            ego: VehicleState {
                x: 50.0,
                y: 4.0,
                heading: 0.0,
                speed: 0.0,
                acceleration: 0.0,
                attention: None,
            },
            npcs: vec![
                VehicleState {
                    x: 70.0,
                    y: 0.0,
                    heading: 0.0,
                    speed: 0.0,
                    acceleration: 0.0,
                    attention: None,
                },
                VehicleState {
                    x: 80.0,
                    y: 8.0,
                    heading: 0.0,
                    speed: 0.0,
                    acceleration: 0.0,
                    attention: None,
                },
            ],
            ..Default::default()
        }
    }

    fn make_svg_episode(frames: Vec<FrameState>) -> SvgEpisode {
        let scene = SceneEpisode::from_frames(frames);
        let viewport = ViewportEpisode::new_default(scene, ViewportConfig::with_omega(3.0));
        SvgEpisode::new(viewport, SvgConfig::new(80, 14, DEFAULT_CORN_ASPRO))
    }

    #[test]
    fn test_render_highway_svg_basic() {
        let frames = vec![make_test_state()];
        let svg_episode = make_svg_episode(frames);
        let config = HighwayRenderConfig::default();

        let svg = render_highway_svg_from_episode(&svg_episode, 0.0, &config).unwrap();

        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("viewBox"));
    }

    #[test]
    fn test_svg_contains_road() {
        let frames = vec![make_test_state()];
        let svg_episode = make_svg_episode(frames);
        let config = HighwayRenderConfig::default();

        let svg = render_highway_svg_from_episode(&svg_episode, 0.0, &config).unwrap();

        // Should contain road surface color (from SceneColorConfig default)
        let colors = crate::config::SceneColorConfig::default();
        assert!(svg.contains(&colors.road_surface));
        // Should contain lane divider color
        assert!(svg.contains(&colors.lane_divider));
    }

    #[test]
    fn test_svg_contains_vehicles() {
        let frames = vec![make_test_state()];
        let svg_episode = make_svg_episode(frames);
        let config = HighwayRenderConfig::default();

        let svg = render_highway_svg_from_episode(&svg_episode, 0.0, &config).unwrap();

        // Ego vehicle (vid=0) car SVG group and gradient defs
        assert!(svg.contains("cg_v0"), "Should contain ego gradient defs");
        assert!(svg.contains(r##"fill="#b4b4b4""##), "Should contain ego body fill color");
        // NPC vehicles have car SVG groups too
        assert!(svg.contains("cg_v1"), "Should contain NPC 0 gradient defs");
        assert!(svg.contains("cg_v2"), "Should contain NPC 1 gradient defs");
    }

    #[test]
    fn test_crashed_ego_color() {
        let mut state = make_test_state();
        state.crashed = true;
        let frames = vec![state];
        let svg_episode = make_svg_episode(frames);
        let config = HighwayRenderConfig::default();

        let svg = render_highway_svg_from_episode(&svg_episode, 0.0, &config).unwrap();

        // Crashed ego should use crashed color in body fill
        assert!(svg.contains(r##"fill="#ff6464""##));
    }

    #[test]
    fn test_render_from_episode_empty_returns_none() {
        let svg_episode = make_svg_episode(vec![]);
        let config = HighwayRenderConfig::default();

        let result = render_highway_svg_from_episode(&svg_episode, 0.0, &config);
        assert!(result.is_none());
    }

    #[test]
    fn test_render_from_episode_interpolated_time() {
        // Create episode with 3 frames
        let frames = vec![
            FrameState {
                crashed: false,
                ego: VehicleState { x: 0.0, y: 4.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None },
                npcs: vec![],
                ..Default::default()
            },
            FrameState {
                crashed: false,
                ego: VehicleState { x: 10.0, y: 4.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None },
                npcs: vec![],
                ..Default::default()
            },
            FrameState {
                crashed: false,
                ego: VehicleState { x: 20.0, y: 4.0, heading: 0.0, speed: 0.0, acceleration: 0.0, attention: None },
                npcs: vec![],
                ..Default::default()
            },
        ];
        let svg_episode = make_svg_episode(frames);
        let config = HighwayRenderConfig::default();

        // Render at interpolated time (midpoint)
        let svg = render_highway_svg_from_episode(&svg_episode, 0.05, &config);
        assert!(svg.is_some());
    }

    #[test]
    fn test_light_theme_uses_light_colors() {
        let config = HighwayRenderConfig::from_config(
            &crate::config::Config::default(),
            crate::config::SceneTheme::Light,
        );
        // Light theme should have a light background
        assert_eq!(config.color_background, "#f0ece0");
        assert_ne!(config.color_background, "#030100");
        // Scale bar should have dark text
        assert_eq!(config.color_scale_bar_fg, "#333333");
    }

    #[test]
    fn test_dark_theme_uses_config_colors() {
        let mut cfg = crate::config::Config::default();
        cfg.octane.colors.scene_themes.dark.background = "#112233".into();
        let config = HighwayRenderConfig::from_config(&cfg, crate::config::SceneTheme::Dark);
        // Dark theme should use the config value
        assert_eq!(config.color_background, "#112233");
    }

    #[test]
    fn test_light_theme_renders_svg() {
        let frames = vec![make_test_state()];
        let svg_episode = make_svg_episode(frames);
        let config = HighwayRenderConfig::from_config(
            &crate::config::Config::default(),
            crate::config::SceneTheme::Light,
        );
        let svg = render_highway_svg_from_episode(&svg_episode, 0.0, &config);
        assert!(svg.is_some());
        let svg = svg.unwrap();
        // Light background should be in the SVG
        assert!(svg.contains("#f0ece0"));
    }

}
