//! Brakelight cone rendering.
//!
//! Thin wrapper over the shared light cone engine in headlights.rs.

use super::headlights::{LightConeParams, render_light_cones};
use super::highway_svg::HighwayRenderConfig;

/// Render brakelight cones for all vehicles with shadow occlusion.
pub fn render_brakelight_cones<F>(
    config: &HighwayRenderConfig,
    all_vehicles: &[(f64, f64, f64)],
    svg_width: f64,
    svg_height: f64,
    world_to_svg: &F,
) -> (Vec<String>, Vec<String>)
where
    F: Fn(f64, f64) -> (f64, f64),
{
    let params = LightConeParams {
        cone_length: config.vehicle_length * 0.5,
        cone_half_spread: config.vehicle_width * 1.2,
        lamp_half_width: config.vehicle_width * 0.1,
        layers: &[
            (1.0, 0.35),
            (0.5, 0.35),
            (0.2, 0.30),
        ],
        base_opacity: config.brakelight_opacity,
        color: &config.color_brakelight,
        id_prefix: "bl",
        direction: -1.0,
    };
    render_light_cones(&params, config, all_vehicles, svg_width, svg_height, world_to_svg)
}
