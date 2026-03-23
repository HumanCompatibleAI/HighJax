//! NPC text labels: show spawn-order index (e.g. "npc4"), speed, and collision warning.

use super::color::{vehicle_color_hex, speed_color_hex, ttc_color_hex};
use crate::config::NpcTextConfig;
use crate::data::jsonla::VehicleState;

/// Maximum time-to-collision (seconds) to display.
const MAX_TTC_SECONDS: f64 = 30.0;

/// Compute time-to-collision between ego and an NPC assuming constant velocity
/// and constant heading, using exact oriented-bounding-box (OBB) collision.
///
/// Both vehicles are modeled as axis-aligned rectangles (length × width) rotated
/// by their heading. Uses the separating axis theorem (SAT) with 4 axes (2 edge
/// normals per rectangle). Since headings are constant, the rectangles translate
/// without rotating, giving exact linear overlap intervals on each axis.
///
/// Returns `Some(ttc)` if they will collide within `MAX_TTC_SECONDS`, else `None`.
fn time_to_collision(
    ego: &VehicleState,
    npc: &VehicleState,
    vehicle_length: f64,
    vehicle_width: f64,
) -> Option<f64> {
    let hl = vehicle_length / 2.0;
    let hw = vehicle_width / 2.0;

    // Velocity components from heading + speed
    let evx = ego.speed * ego.heading.cos();
    let evy = ego.speed * ego.heading.sin();
    let nvx = npc.speed * npc.heading.cos();
    let nvy = npc.speed * npc.heading.sin();

    // Relative position and velocity (npc relative to ego)
    let dx = npc.x - ego.x;
    let dy = npc.y - ego.y;
    let dvx = nvx - evx;
    let dvy = nvy - evy;

    // Precompute trig for both vehicles
    let (eco, esi) = (ego.heading.cos(), ego.heading.sin());
    let (nco, nsi) = (npc.heading.cos(), npc.heading.sin());

    // 4 separating axes: 2 edge normals from each rectangle
    let axes = [
        (eco, esi),   // ego forward
        (-esi, eco),  // ego lateral
        (nco, nsi),   // npc forward
        (-nsi, nco),  // npc lateral
    ];

    let mut t_enter = 0.0_f64;
    let mut t_exit = MAX_TTC_SECONDS;

    for &(nx, ny) in &axes {
        // Half-extent of each rectangle projected onto this axis.
        // For a rectangle with forward dir (cos θ, sin θ) and lateral (-sin θ, cos θ),
        // the projection extent = hl * |forward · axis| + hw * |lateral · axis|.
        let ego_ext = hl * (eco * nx + esi * ny).abs()
                    + hw * (-esi * nx + eco * ny).abs();
        let npc_ext = hl * (nco * nx + nsi * ny).abs()
                    + hw * (-nsi * nx + nco * ny).abs();
        let sum_ext = ego_ext + npc_ext;

        // Relative displacement and velocity along this axis
        let d = dx * nx + dy * ny;
        let v = dvx * nx + dvy * ny;

        if v.abs() < 1e-12 {
            // No relative motion on this axis
            if d.abs() >= sum_ext {
                return None; // Permanently separated
            }
            continue; // Overlapping on this axis — no time constraint
        }

        // Solve: -sum_ext < d + v*t < sum_ext
        let t1 = (-sum_ext - d) / v;
        let t2 = (sum_ext - d) / v;
        let (t_in, t_out) = if t1 < t2 { (t1, t2) } else { (t2, t1) };

        t_enter = t_enter.max(t_in);
        t_exit = t_exit.min(t_out);

        if t_enter > t_exit {
            return None;
        }
    }

    if t_enter >= 0.0 && t_enter <= t_exit {
        Some(t_enter)
    } else if t_enter < 0.0 && t_exit > 0.0 {
        Some(0.0) // Currently overlapping
    } else {
        None
    }
}

/// Render NPC text labels on top of each NPC vehicle.
///
/// Each label shows up to three lines centered on the vehicle:
/// - "npc{i}" in the NPC's vehicle color
/// - "{speed}㎧" colored by the speed ramp (green→yellow→red)
/// - "!{ttc}s" collision warning colored by TTC ramp (only if on collision course)
#[allow(clippy::too_many_arguments)]
pub fn render_npc_text_svg<F>(
    elements: &mut Vec<String>,
    _ego: &VehicleState,
    npcs: &[VehicleState],
    ttc_ego: &VehicleState,
    ttc_npcs: &[VehicleState],
    npc_lightness: f64,
    npc_chroma: f64,
    nc: &NpcTextConfig,
    ego_speed_range: (f64, f64),
    vehicle_length: f64,
    vehicle_width: f64,
    _pixel_width: f64,
    world_to_svg: &F,
) where
    F: Fn(f64, f64) -> (f64, f64),
{
    // Compute SVG-units-per-meter for font sizing.
    let (sx0, _sy0) = world_to_svg(0.0, 0.0);
    let (sx1, _sy1) = world_to_svg(1.0, 0.0);
    let svg_per_meter_x = (sx1 - sx0).abs();
    if svg_per_meter_x < 1e-9 {
        return;
    }

    let font_size = nc.font_size_meters * svg_per_meter_x;
    let stroke_width = font_size * 0.45;
    let line_spacing = font_size * 0.9;

    for (i, npc) in npcs.iter().enumerate() {
        let (sx, sy) = world_to_svg(npc.x, npc.y);
        let fg = vehicle_color_hex(i, npc_lightness, npc_chroma);
        let label = format!("npc{}", i);

        // Use snapped (discrete frame) states for TTC to avoid flicker from interpolation
        let ttc = ttc_npcs.get(i).and_then(|tnpc| {
            time_to_collision(ttc_ego, tnpc, vehicle_length, vehicle_width)
        });
        let n_lines = if ttc.is_some() { 3 } else { 2 };
        let block_top = sy - (n_lines as f64 - 1.0) * line_spacing * 0.5;

        // Line 1: npc label
        let y_name = block_top;
        elements.push(format!(
            r#"<text x="{:.6}" y="{:.6}" fill="{}"{} font-size="{:.6}" font-family="{}" font-weight="bold" text-anchor="middle" dominant-baseline="central" data-mango-fg="{}" data-mango-bg="{}">{}</text>"#,
            sx, y_name, fg,
            format_args!(r#" stroke="{}" stroke-width="{:.6}" paint-order="stroke" stroke-linejoin="round""#,
                nc.bg_color, stroke_width),
            font_size, nc.font_family, fg, nc.bg_color, label
        ));

        // Line 2: speed label
        let y_speed = block_top + line_spacing;
        let speed_fg = speed_color_hex(npc.speed, ego_speed_range.0, ego_speed_range.1);
        let speed_label = format!("{:.0}m/s", npc.speed);
        elements.push(format!(
            r#"<text x="{:.6}" y="{:.6}" fill="{}"{} font-size="{:.6}" font-family="{}" font-weight="bold" text-anchor="middle" dominant-baseline="central" data-mango-fg="{}" data-mango-bg="{}">{}</text>"#,
            sx, y_speed, speed_fg,
            format_args!(r#" stroke="{}" stroke-width="{:.6}" paint-order="stroke" stroke-linejoin="round""#,
                nc.bg_color, stroke_width),
            font_size, nc.font_family, speed_fg, nc.bg_color, speed_label
        ));

        // Line 3: collision warning (only if on collision course)
        if let Some(t) = ttc {
            let y_ttc = block_top + line_spacing * 2.0;
            let warn_fg = ttc_color_hex(t);
            let ttc_label = format!("!{:.1}s", t);
            elements.push(format!(
                r#"<text x="{:.6}" y="{:.6}" fill="{}"{} font-size="{:.6}" font-family="{}" font-weight="bold" text-anchor="middle" dominant-baseline="central" data-mango-fg="{}" data-mango-bg="{}">{}</text>"#,
                sx, y_ttc, warn_fg,
                format_args!(r#" stroke="{}" stroke-width="{:.6}" paint-order="stroke" stroke-linejoin="round""#,
                    nc.bg_color, stroke_width),
                font_size, nc.font_family, warn_fg, nc.bg_color, ttc_label
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const VL: f64 = 5.0; // vehicle length
    const VW: f64 = 2.0; // vehicle width

    fn vehicle(x: f64, y: f64, heading: f64, speed: f64) -> VehicleState {
        VehicleState {
            x,
            y,
            heading,
            speed,
            acceleration: 0.0,
            attention: None,
        }
    }

    // =========================================================================
    // Head-on collision
    // =========================================================================

    #[test]
    fn test_head_on_collision() {
        // Ego heading east, npc heading west, on same line, closing at 20 m/s
        let ego = vehicle(0.0, 0.0, 0.0, 10.0);
        let npc = vehicle(100.0, 0.0, PI, 10.0);
        let ttc = time_to_collision(&ego, &npc, VL, VW).unwrap();
        // Distance 100m, closing 20m/s, they touch when center gap = vehicle_length = 5m
        // t = (100 - 5) / 20 = 4.75
        assert!((ttc - 4.75).abs() < 0.01, "head-on TTC: {}", ttc);
    }

    // =========================================================================
    // Rear-end (faster car catching slower car)
    // =========================================================================

    #[test]
    fn test_rear_end_collision() {
        // Ego faster, behind npc, both heading east
        let ego = vehicle(0.0, 0.0, 0.0, 30.0);
        let npc = vehicle(50.0, 0.0, 0.0, 20.0);
        let ttc = time_to_collision(&ego, &npc, VL, VW).unwrap();
        // Closing at 10 m/s, gap 50m, touch at center dist = 5m
        // t = (50 - 5) / 10 = 4.5
        assert!((ttc - 4.5).abs() < 0.01, "rear-end TTC: {}", ttc);
    }

    // =========================================================================
    // T-bone collision (perpendicular)
    // =========================================================================

    #[test]
    fn test_t_bone_collision() {
        // Ego heading east, npc heading north, paths cross
        let ego = vehicle(0.0, 0.0, 0.0, 20.0);
        let npc = vehicle(50.0, -50.0, PI / 2.0, 20.0);
        let ttc = time_to_collision(&ego, &npc, VL, VW).unwrap();
        // Both reach crossing point at t≈2.5s
        // With OBB: on ego-forward axis (1,0), sum_ext = hl + hw = 2.5 + 1.0 = 3.5
        //   d = 50, v = -20, t_in = (50-3.5)/20 = 2.325
        assert!((ttc - 2.325).abs() < 0.01, "t-bone TTC: {}", ttc);
    }

    // =========================================================================
    // Parallel same lane, same speed → no collision
    // =========================================================================

    #[test]
    fn test_parallel_same_speed_no_collision() {
        let ego = vehicle(0.0, 0.0, 0.0, 20.0);
        let npc = vehicle(50.0, 0.0, 0.0, 20.0);
        assert!(time_to_collision(&ego, &npc, VL, VW).is_none());
    }

    // =========================================================================
    // Adjacent lanes, opposite directions → no collision (lateral gap > width)
    // =========================================================================

    #[test]
    fn test_adjacent_lane_passing_no_collision() {
        // Ego in lane at y=0, npc in lane at y=4, heading opposite
        let ego = vehicle(0.0, 0.0, 0.0, 20.0);
        let npc = vehicle(100.0, 4.0, PI, 20.0);
        // Lateral gap = 4m, combined lateral extent on ego-lateral axis = 1+1 = 2m
        // 4 > 2 → permanently separated on lateral axis
        assert!(time_to_collision(&ego, &npc, VL, VW).is_none());
    }

    // =========================================================================
    // Adjacent lane, barely touching laterally → collision
    // =========================================================================

    #[test]
    fn test_adjacent_lane_barely_touching() {
        // Both heading east, lateral offset exactly equal to vehicle width (2m)
        // On ego-lateral axis: sum_ext = hw + hw = 2.0, d = 2.0 → barely separated
        let ego = vehicle(0.0, 0.0, 0.0, 30.0);
        let npc = vehicle(50.0, 2.0, 0.0, 20.0);
        assert!(time_to_collision(&ego, &npc, VL, VW).is_none());
    }

    #[test]
    fn test_adjacent_lane_overlapping_laterally() {
        // Both heading east, lateral offset 1.9m < vehicle width
        let ego = vehicle(0.0, 0.0, 0.0, 30.0);
        let npc = vehicle(50.0, 1.9, 0.0, 20.0);
        let ttc = time_to_collision(&ego, &npc, VL, VW);
        assert!(ttc.is_some(), "should collide when lanes partially overlap");
    }

    // =========================================================================
    // Already overlapping → TTC = 0
    // =========================================================================

    #[test]
    fn test_already_overlapping() {
        let ego = vehicle(0.0, 0.0, 0.0, 10.0);
        let npc = vehicle(1.0, 0.0, 0.0, 10.0);
        let ttc = time_to_collision(&ego, &npc, VL, VW).unwrap();
        assert!((ttc - 0.0).abs() < 0.01, "overlapping TTC: {}", ttc);
    }

    // =========================================================================
    // Diverging vehicles → no collision
    // =========================================================================

    #[test]
    fn test_diverging_no_collision() {
        // Ego heading east, npc heading west, but npc is behind ego
        let ego = vehicle(0.0, 0.0, 0.0, 10.0);
        let npc = vehicle(-50.0, 0.0, PI, 10.0);
        assert!(time_to_collision(&ego, &npc, VL, VW).is_none());
    }

    // =========================================================================
    // Beyond MAX_TTC_SECONDS → None
    // =========================================================================

    #[test]
    fn test_beyond_max_ttc() {
        // Very far apart, slow closing → TTC > 30s
        let ego = vehicle(0.0, 0.0, 0.0, 10.0);
        let npc = vehicle(1000.0, 0.0, PI, 10.0);
        // Distance 1000m, closing 20 m/s, TTC ≈ (1000-5)/20 = 49.75s > 30
        assert!(time_to_collision(&ego, &npc, VL, VW).is_none());
    }

    // =========================================================================
    // Both stationary, separated → None
    // =========================================================================

    #[test]
    fn test_stationary_separated() {
        let ego = vehicle(0.0, 0.0, 0.0, 0.0);
        let npc = vehicle(50.0, 0.0, 0.0, 0.0);
        assert!(time_to_collision(&ego, &npc, VL, VW).is_none());
    }

    // =========================================================================
    // Both stationary, overlapping → TTC = 0
    // =========================================================================

    #[test]
    fn test_stationary_overlapping() {
        let ego = vehicle(0.0, 0.0, 0.0, 0.0);
        let npc = vehicle(2.0, 0.0, 0.0, 0.0);
        let ttc = time_to_collision(&ego, &npc, VL, VW).unwrap();
        assert!((ttc - 0.0).abs() < 0.01);
    }

    // =========================================================================
    // Angled approach — vehicles at 45°
    // =========================================================================

    #[test]
    fn test_angled_approach() {
        // Ego heading east, npc heading southwest at 45°
        let ego = vehicle(0.0, 0.0, 0.0, 10.0);
        let npc = vehicle(80.0, 40.0, PI + PI / 4.0, 14.142); // ~10√2 m/s
        let ttc = time_to_collision(&ego, &npc, VL, VW);
        // Npc moves at (-10, -10) m/s, ego at (10, 0) m/s
        // Relative velocity: (-20, -10). They should converge.
        assert!(ttc.is_some(), "angled approach should produce TTC");
        assert!(ttc.unwrap() > 0.0 && ttc.unwrap() < MAX_TTC_SECONDS);
    }

    // =========================================================================
    // Near-miss: paths cross but vehicles pass at different times
    // =========================================================================

    #[test]
    fn test_near_miss_timing() {
        // Ego heading east at (0, 0), speed 20
        // NPC heading north at (100, -20), speed 20
        // NPC reaches y=0 at t=1s, ego reaches x=100 at t=5s — timing mismatch
        let ego = vehicle(0.0, 0.0, 0.0, 20.0);
        let npc = vehicle(100.0, -20.0, PI / 2.0, 20.0);
        assert!(
            time_to_collision(&ego, &npc, VL, VW).is_none(),
            "paths cross but timing means they miss"
        );
    }

    // =========================================================================
    // Symmetry: TTC(ego, npc) == TTC(npc, ego)
    // =========================================================================

    #[test]
    fn test_symmetry() {
        let ego = vehicle(0.0, 0.0, 0.0, 15.0);
        let npc = vehicle(60.0, 3.0, PI + 0.1, 12.0);
        let ttc_a = time_to_collision(&ego, &npc, VL, VW);
        let ttc_b = time_to_collision(&npc, &ego, VL, VW);
        match (ttc_a, ttc_b) {
            (Some(a), Some(b)) => assert!((a - b).abs() < 0.01, "asymmetric: {} vs {}", a, b),
            (None, None) => {}
            _ => panic!("one is Some, other is None: {:?} vs {:?}", ttc_a, ttc_b),
        }
    }

    // =========================================================================
    // Verify rectangle geometry: lateral gap exactly at boundary
    // =========================================================================

    #[test]
    fn test_exact_lateral_boundary() {
        // Two vehicles heading east, lateral offset = exactly vehicle_width (2.0)
        // On lateral axis, sum_ext = hw + hw = 1.0 + 1.0 = 2.0
        // |d| = 2.0 >= sum_ext → separated (no collision)
        let ego = vehicle(0.0, 0.0, 0.0, 30.0);
        let npc = vehicle(50.0, VW, 0.0, 20.0);
        assert!(time_to_collision(&ego, &npc, VL, VW).is_none());

        // Offset by 1.99m → just within collision zone
        let npc2 = vehicle(50.0, VW - 0.01, 0.0, 20.0);
        assert!(time_to_collision(&ego, &npc2, VL, VW).is_some());
    }

    // =========================================================================
    // Verify rectangle geometry: longitudinal gap exactly at boundary
    // =========================================================================

    #[test]
    fn test_exact_longitudinal_boundary() {
        // Two vehicles heading east, one behind the other
        // On forward axis, sum_ext = hl + hl = 2.5 + 2.5 = 5.0
        // Ego at origin, npc at x=5.0 → center gap = 5.0 = sum_ext → boundary
        let ego = vehicle(0.0, 0.0, 0.0, 20.0);
        let npc = vehicle(VL, 0.0, 0.0, 20.0);
        // Same speed → no relative motion → check static: |d| = 5.0 >= 5.0 → separated
        assert!(time_to_collision(&ego, &npc, VL, VW).is_none());

        // Center gap = 4.99 → just overlapping
        let npc2 = vehicle(VL - 0.01, 0.0, 0.0, 20.0);
        // Still same speed → no relative motion → |d| = 4.99 < 5.0 → overlapping → TTC = 0
        let ttc = time_to_collision(&ego, &npc2, VL, VW).unwrap();
        assert!((ttc - 0.0).abs() < 0.01);
    }
}
