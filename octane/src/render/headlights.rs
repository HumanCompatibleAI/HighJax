//! Light cone rendering with shadow occlusion.
//!
//! Shared engine for headlights and brakelights. Each vehicle casts two cones
//! (left/right lamps). Each cone is a trapezoid with opacity layers. Blocking
//! cars cast shadows via intersection-based polygon clipping.

use super::geometry::{Pt, clip_polygon_half, orient_line_towards, convex_hull, subtract_convex};
use super::color::{parse_hex_color, vehicle_color_hex};
use super::highway_svg::HighwayRenderConfig;

// =============================================================================
// Silhouette: tighter-fitting car polygon (14 vertices)
// =============================================================================

/// Approximate car outline as fractions of (half_l, half_w).
/// Traced from the car SVG body path, symmetrized.
/// along = forward (heading), perp = rightward.
pub const SILHOUETTE_SHAPE: [(f64, f64); 14] = [
    ( 0.97,  0.00),   // front tip (nose)
    ( 0.95,  0.82),   // front-right nose inner
    ( 0.88,  0.90),   // front-right nose outer
    ( 0.74,  0.94),   // front-right corner
    ( 0.06,  0.87),   // right side, mid
    (-0.50,  0.84),   // right side, rear-mid
    (-0.93,  0.82),   // rear-right corner
    (-0.98,  0.00),   // rear center
    (-0.93, -0.82),   // rear-left corner
    (-0.50, -0.84),   // left side, rear-mid
    ( 0.06, -0.87),   // left side, mid
    ( 0.74, -0.94),   // front-left corner
    ( 0.88, -0.90),   // front-left nose outer
    ( 0.95, -0.82),   // front-left nose inner
];

/// Build silhouette corners in SVG space for one vehicle.
pub fn silhouette_corners<F>(
    vx: f64, vy: f64, vh: f64,
    half_l: f64, half_w: f64,
    world_to_svg: &F,
) -> Vec<Pt>
where
    F: Fn(f64, f64) -> (f64, f64),
{
    let c = vh.cos();
    let s = vh.sin();
    SILHOUETTE_SHAPE.iter().map(|&(al, ap)| {
        let dx = al * half_l;
        let dy = ap * half_w;
        world_to_svg(vx + c * dx - s * dy, vy + s * dx + c * dy)
    }).collect()
}

// =============================================================================
// Debug SVG builder for step-by-step debug images
// =============================================================================

#[derive(Clone)]
struct DbgSvg {
    defs: String,
    body: String,
    w: f64,
    h: f64,
}
#[allow(dead_code)]
impl DbgSvg {
    fn new(w: f64, h: f64) -> Self {
        Self {
            defs: String::new(),
            body: format!(
                r##"<rect width="{w}" height="{h}" fill="#2a2a2a"/>"##,
            ),
            w,
            h,
        }
    }
    fn pts(p: &[Pt]) -> String {
        p.iter().map(|v| format!("{:.4},{:.4}", v.0, v.1)).collect::<Vec<_>>().join(" ")
    }
    fn poly(&mut self, p: &[Pt], fill: &str, op: f64) {
        if p.len() < 3 { return; }
        let s = Self::pts(p);
        self.body.push_str(&format!(
            r#"<polygon points="{s}" fill="{fill}" opacity="{op:.2}"/><polygon points="{s}" fill="none" stroke="{fill}" stroke-width="0.001"/>"#,
        ));
    }
    fn line(&mut self, a: Pt, b: Pt, color: &str) {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        self.body.push_str(&format!(
            r#"<line x1="{:.4}" y1="{:.4}" x2="{:.4}" y2="{:.4}" stroke="{}" stroke-width="0.001" opacity="0.7"/>"#,
            a.0 - dx * 10.0, a.1 - dy * 10.0, a.0 + dx * 10.0, a.1 + dy * 10.0, color,
        ));
    }
    fn ray(&mut self, origin: Pt, dir: Pt, color: &str) {
        self.body.push_str(&format!(
            r#"<line x1="{:.4}" y1="{:.4}" x2="{:.4}" y2="{:.4}" stroke="{}" stroke-width="0.0015" opacity="0.8"/>"#,
            origin.0, origin.1, origin.0 + dir.0 * 10.0, origin.1 + dir.1 * 10.0, color,
        ));
    }
    fn dot(&mut self, p: Pt, color: &str) {
        self.body.push_str(&format!(
            r#"<circle cx="{:.4}" cy="{:.4}" r="0.003" fill="{}"/>"#, p.0, p.1, color,
        ));
    }
    fn label(&mut self, text: &str) {
        self.body.push_str(&format!(
            r#"<text x="0.01" y="0.03" font-size="0.015" fill="white" font-family="monospace">{}</text>"#, text,
        ));
    }
    fn add_defs(&mut self, d: &str) {
        self.defs.push_str(d);
    }
    fn raw(&mut self, s: &str) {
        self.body.push_str(s);
    }
    fn save(self, svg_path: &str, png_path: &str) {
        let svg = format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1600" height="1000" viewBox="0 0 {w} {h}"><defs>{defs}</defs>{body}</svg>"##,
            w = self.w, h = self.h, defs = self.defs, body = self.body,
        );
        let _ = std::fs::write(svg_path, &svg);
        // Convert to PNG via resvg
        let opt = resvg::usvg::Options::default();
        if let Ok(tree) = resvg::usvg::Tree::from_str(&svg, &opt) {
            if let Some(mut pixmap) = resvg::tiny_skia::Pixmap::new(1600, 1000) {
                resvg::render(&tree, resvg::tiny_skia::Transform::default(), &mut pixmap.as_mut());
                let _ = pixmap.save_png(png_path);
            }
        }
    }
}

// =============================================================================
// Shared light cone parameters and rendering
// =============================================================================

/// Parameters that differ between headlights and brakelights.
pub struct LightConeParams<'a> {
    /// Cone length in meters.
    pub cone_length: f64,
    /// Half-spread at far end of cone in meters.
    pub cone_half_spread: f64,
    /// Half-width at lamp (near end) in meters.
    pub lamp_half_width: f64,
    /// Opacity layers: (spread_fraction, opacity_fraction).
    pub layers: &'a [(f64, f64)],
    /// Base opacity multiplier.
    pub base_opacity: f64,
    /// Gradient stop color (hex, e.g. "#ffffcc").
    pub color: &'a str,
    /// Gradient ID prefix (e.g. "hl" or "bl").
    pub id_prefix: &'a str,
    /// +1.0 for forward (headlights), -1.0 for backward (brakelights).
    pub direction: f64,
}

/// Render light cones for all vehicles with shadow occlusion.
///
/// Returns `(defs, elements)` where defs are SVG gradient definitions and
/// elements are the visible cone polygon fragments.
pub fn render_light_cones<F>(
    params: &LightConeParams,
    config: &HighwayRenderConfig,
    all_vehicles: &[(f64, f64, f64)],
    svg_width: f64,
    svg_height: f64,
    world_to_svg: &F,
) -> (Vec<String>, Vec<String>)
where
    F: Fn(f64, f64) -> (f64, f64),
{
    let cone_length = params.cone_length;
    let cone_half_spread = params.cone_half_spread;
    let lamp_half_width = params.lamp_half_width;
    let dir = params.direction;
    let mut cone_id = 0_usize;
    let mut cone_defs: Vec<String> = Vec::new();
    let mut cone_elements: Vec<String> = Vec::new();

    // Precompute blocker corners in SVG space
    let half_l = config.vehicle_length / 2.0;
    let half_w = config.vehicle_width / 2.0 * 1.134;
    let blocker_corners_svg: Vec<[Pt; 4]> = all_vehicles.iter().map(|&(bx, by, bh)| {
        let bc = bh.cos();
        let bs = bh.sin();
        [
            world_to_svg(bx + bc * half_l - bs * half_w, by + bs * half_l + bc * half_w),
            world_to_svg(bx + bc * half_l + bs * half_w, by + bs * half_l - bc * half_w),
            world_to_svg(bx - bc * half_l + bs * half_w, by - bs * half_l - bc * half_w),
            world_to_svg(bx - bc * half_l - bs * half_w, by - bs * half_l + bc * half_w),
        ]
    }).collect();

    // Precompute silhouettes (tighter car outlines) in SVG space
    let silhouette_svg: Vec<Vec<Pt>> = all_vehicles.iter().map(|&(bx, by, bh)| {
        silhouette_corners(bx, by, bh, half_l, half_w, world_to_svg)
    }).collect();

    for (src_idx, &(src_x, src_y, src_h)) in all_vehicles.iter().enumerate() {
        let cos_h = src_h.cos();
        let sin_h = src_h.sin();
        let front_offset = config.vehicle_length * 0.45 * dir;
        let lamp_offset = config.vehicle_width * 0.28;

        for side in [-1.0_f64, 1.0] {
            let lx = src_x + cos_h * front_offset - sin_h * lamp_offset * side;
            let ly = src_y + sin_h * front_offset + cos_h * lamp_offset * side;
            let ex = lx + cos_h * cone_length * dir;
            let ey = ly + sin_h * cone_length * dir;
            let (gx1, gy1) = world_to_svg(lx, ly);
            let (gx2, gy2) = world_to_svg(ex, ey);
            let lamp_svg = (gx1, gy1);

            for &(spread_frac, opacity_frac) in params.layers {
                let layer_spread = cone_half_spread * spread_frac;
                let layer_lamp = lamp_half_width * spread_frac.max(0.3);
                let layer_opacity = params.base_opacity * opacity_frac;

                let n1: Pt = world_to_svg(
                    lx - sin_h * layer_lamp, ly + cos_h * layer_lamp,
                );
                let n2: Pt = world_to_svg(
                    lx + sin_h * layer_lamp, ly - cos_h * layer_lamp,
                );
                let f1: Pt = world_to_svg(
                    ex - sin_h * layer_spread, ey + cos_h * layer_spread,
                );
                let f2: Pt = world_to_svg(
                    ex + sin_h * layer_spread, ey - cos_h * layer_spread,
                );

                // Gradient def
                let gid = format!("{}{}", params.id_prefix, cone_id);
                cone_id += 1;
                cone_defs.push(format!(
                    concat!(
                        r##"<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" "##,
                        r##"x1="{x1:.4}" y1="{y1:.4}" x2="{x2:.4}" y2="{y2:.4}">"##,
                        r##"<stop offset="0" stop-color="{color}" stop-opacity="{o60:.3}"/>"##,
                        r##"<stop offset="0.15" stop-color="{color}" stop-opacity="{o100:.3}"/>"##,
                        r##"<stop offset="0.5" stop-color="{color}" stop-opacity="{o20:.3}"/>"##,
                        r##"<stop offset="1" stop-color="{color}" stop-opacity="0"/>"##,
                        r##"</linearGradient>"##,
                    ),
                    gid = gid,
                    color = params.color,
                    x1 = gx1, y1 = gy1, x2 = gx2, y2 = gy2,
                    o60 = layer_opacity * 0.6,
                    o100 = layer_opacity,
                    o20 = layer_opacity * 0.2,
                ));

                // Start with full trapezoid: n1, n2, f2, f1
                let mut visible: Vec<Vec<Pt>> = vec![vec![n1, n2, f2, f1]];
                let mut new_visible: Vec<Vec<Pt>> = Vec::new();

                // Shadow directions from this layer's edges
                let top_dir = (f2.0 - n2.0, f2.1 - n2.1);
                let bot_dir = (f1.0 - n1.0, f1.1 - n1.1);

                let is_dbg = false;
                let jaj_dir: Option<String> = if is_dbg {
                    let ts = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default().as_secs();
                    let dir = format!("{}/Desktop/jaj/{}",
                        std::env::var("HOME").unwrap_or_default(), ts);
                    std::fs::create_dir_all(format!("{}/svg", dir)).ok();
                    let mut d = DbgSvg::new(svg_width, svg_height);
                    d.poly(&[n1, n2, f2, f1], "yellow", 0.4);
                    for (bi, &(bx, by, bh)) in all_vehicles.iter().enumerate() {
                        if bi == src_idx { continue; }
                        let (r, g, b) = if bi == 0 {
                            parse_hex_color(&config.ego_color)
                        } else {
                            let hex = vehicle_color_hex(
                                bi - 1, config.npc_lightness, config.npc_chroma,
                            );
                            parse_hex_color(&hex)
                        };
                        let (car_defs, car_elem) = crate::render::car_template::render_car(
                            bi, bx, by, bh,
                            config.vehicle_length, config.vehicle_width,
                            r, g, b, &config.window_color, world_to_svg,
                        );
                        d.add_defs(&car_defs);
                        d.raw(&car_elem);
                        d.poly(blocker_corners_svg[bi].as_slice(), "red", 0.0);
                        d.poly(&silhouette_svg[bi], "#00ffff", 0.0);
                    }
                    d.dot(lamp_svg, "white");
                    d.label("00: cars + bbox(red) + silhouette(cyan)");
                    d.save(&format!("{}/svg/00_initial.svg", dir),
                           &format!("{}/00_initial.png", dir));
                    Some(dir)
                } else { None };
                let mut dbg_step = 1_usize;

                // Trapezoid vertices and centroid for clipping
                let trap = [n1, n2, f2, f1];
                let trap_cx = (n1.0 + n2.0 + f1.0 + f2.0) / 4.0;
                let trap_cy = (n1.1 + n2.1 + f1.1 + f2.1) / 4.0;
                let trap_centroid = (trap_cx, trap_cy);

                // Clip by each blocking car using intersection-based shadows
                for (blk_idx, &(bx, by, _bh)) in all_vehicles.iter().enumerate() {
                    if blk_idx == src_idx { continue; }

                    // Quick reject: distance-based
                    let dx = bx - lx;
                    let dy = by - ly;
                    let ahead = (dx * cos_h + dy * sin_h) * dir;
                    if ahead < -half_l || ahead > cone_length + half_l {
                        continue;
                    }
                    let perp_world = (-dx * sin_h + dy * cos_h).abs();
                    let t = (ahead / cone_length).clamp(0.0, 1.0);
                    let spread_at = layer_lamp + (layer_spread - layer_lamp) * t;
                    if perp_world > spread_at + half_w + 0.5 {
                        continue;
                    }

                    // Step 1: Compute car-trapezoid intersection polygon
                    let car = &silhouette_svg[blk_idx];
                    let mut intersection: Vec<Pt> = car.clone();
                    for i in 0..4 {
                        let e1 = trap[i];
                        let e2 = trap[(i + 1) % 4];
                        let (p1, p2) = orient_line_towards(e1, e2, trap_centroid);
                        intersection = clip_polygon_half(&intersection, p1, p2);
                    }
                    if intersection.len() < 3 { continue; }

                    // Step 2: Build shadow points from each intersection vertex
                    let n_isect = intersection.len();
                    let mut shadow_pts: Vec<Pt> = Vec::with_capacity(n_isect * 3);
                    shadow_pts.extend_from_slice(&intersection);
                    for &v in &intersection {
                        shadow_pts.push((v.0 + top_dir.0 * 10.0, v.1 + top_dir.1 * 10.0));
                        shadow_pts.push((v.0 + bot_dir.0 * 10.0, v.1 + bot_dir.1 * 10.0));
                    }

                    // Step 3: Convex hull of all shadow points
                    let shadow_hull = convex_hull(shadow_pts);

                    // Step 4: Clip shadow hull to the trapezoid
                    let mut shadow_mask = shadow_hull;
                    for i in 0..4 {
                        let e1 = trap[i];
                        let e2 = trap[(i + 1) % 4];
                        let (p1, p2) = orient_line_towards(e1, e2, trap_centroid);
                        shadow_mask = clip_polygon_half(&shadow_mask, p1, p2);
                    }
                    if shadow_mask.len() < 3 { continue; }

                    // Debug steps
                    if let Some(ref dir) = jaj_dir {
                        let mut d = DbgSvg::new(svg_width, svg_height);
                        d.poly(&[n1, n2, f2, f1], "yellow", 0.15);
                        for p in &visible { d.poly(p, "#00ff00", 0.4); }
                        for (bi, s) in silhouette_svg.iter().enumerate() {
                            if bi != src_idx {
                                d.poly(s, "red", if bi == blk_idx { 0.4 } else { 0.1 });
                            }
                        }
                        d.poly(&intersection, "#00ffff", 0.5);
                        for &v in &intersection { d.dot(v, "#ffff00"); }
                        d.dot(lamp_svg, "white");
                        d.label(&format!("{:02}: blk{} intersection ({} verts)", dbg_step, blk_idx, intersection.len()));
                        d.save(&format!("{}/svg/{:02}_blk{}_isect.svg", dir, dbg_step, blk_idx),
                               &format!("{}/{:02}_blk{}_isect.png", dir, dbg_step, blk_idx));
                        dbg_step += 1;

                        let mut d = DbgSvg::new(svg_width, svg_height);
                        d.poly(&[n1, n2, f2, f1], "yellow", 0.15);
                        d.poly(&intersection, "#00ffff", 0.3);
                        for &v in &intersection {
                            d.ray(v, top_dir, "#ff00ff");
                            d.ray(v, bot_dir, "#ff8800");
                            d.dot(v, "#ffff00");
                        }
                        d.dot(lamp_svg, "white");
                        d.label(&format!("{:02}: blk{} shadow rays: magenta=top_dir, orange=bot_dir", dbg_step, blk_idx));
                        d.save(&format!("{}/svg/{:02}_blk{}_rays.svg", dir, dbg_step, blk_idx),
                               &format!("{}/{:02}_blk{}_rays.png", dir, dbg_step, blk_idx));
                        dbg_step += 1;

                        let mut d = DbgSvg::new(svg_width, svg_height);
                        d.poly(&[n1, n2, f2, f1], "yellow", 0.15);
                        for p in &visible { d.poly(p, "#00ff00", 0.4); }
                        d.poly(&shadow_mask, "#ff0000", 0.4);
                        d.poly(&intersection, "#00ffff", 0.3);
                        d.dot(lamp_svg, "white");
                        d.label(&format!("{:02}: blk{} shadow mask (red) clipped to trapezoid", dbg_step, blk_idx));
                        d.save(&format!("{}/svg/{:02}_blk{}_mask.svg", dir, dbg_step, blk_idx),
                               &format!("{}/{:02}_blk{}_mask.png", dir, dbg_step, blk_idx));
                        dbg_step += 1;
                    }

                    // Step 5: Subtract shadow mask from all visible pieces
                    new_visible.clear();
                    for poly in &visible {
                        let pieces = subtract_convex(poly, &shadow_mask);
                        new_visible.extend(pieces);
                    }
                    std::mem::swap(&mut visible, &mut new_visible);
                }

                // Debug: final step
                if let Some(ref dir) = jaj_dir {
                    let mut d = DbgSvg::new(svg_width, svg_height);
                    d.poly(&[n1, n2, f2, f1], "yellow", 0.15);
                    let colors = ["#00ff00", "#00cc44", "#0088ff", "#ff8800"];
                    for (i, p) in visible.iter().enumerate() {
                        d.poly(p, colors[i % colors.len()], 0.5);
                    }
                    for (bi, s) in silhouette_svg.iter().enumerate() {
                        if bi != src_idx { d.poly(s, "red", 0.15); }
                    }
                    d.dot(lamp_svg, "white");
                    d.label(&format!("{:02}: final - {} visible pieces", dbg_step, visible.len()));
                    d.save(&format!("{}/svg/{:02}_final.svg", dir, dbg_step),
                           &format!("{}/{:02}_final.png", dir, dbg_step));
                }

                // Render visible polygon pieces
                for poly in &visible {
                    if poly.len() < 3 { continue; }
                    use std::fmt::Write;
                    let mut buf = String::with_capacity(poly.len() * 20 + 50);
                    buf.push_str(r#"<polygon points=""#);
                    for (j, (x, y)) in poly.iter().enumerate() {
                        if j > 0 { buf.push(' '); }
                        let _ = write!(buf, "{:.4},{:.4}", x, y);
                    }
                    let _ = write!(buf, r##"" fill="url(#{gid})"/>"##, gid = gid);
                    cone_elements.push(buf);
                }
            }
        }
    }

    (cone_defs, cone_elements)
}

// =============================================================================
// Headlight-specific entry point
// =============================================================================

/// Render headlight cones for all vehicles with shadow occlusion.
pub fn render_headlight_cones<F>(
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
        cone_length: config.vehicle_length * 3.0,
        cone_half_spread: config.vehicle_width * 1.2,
        lamp_half_width: config.vehicle_width * 0.1,
        layers: &[
            (1.0, 0.30),
            (0.6, 0.35),
            (0.25, 0.35),
        ],
        base_opacity: config.headlight_opacity,
        color: &config.color_headlight,
        id_prefix: "hl",
        direction: 1.0,
    };
    render_light_cones(&params, config, all_vehicles, svg_width, svg_height, world_to_svg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silhouette_shape_is_valid() {
        // All coordinates within [-1, 1]
        for &(al, ap) in &SILHOUETTE_SHAPE {
            assert!(al.abs() <= 1.0, "along {} out of range", al);
            assert!(ap.abs() <= 1.0, "perp {} out of range", ap);
        }
        // Symmetric: for each (al, ap) with ap != 0, there should be (al, -ap)
        for &(al, ap) in &SILHOUETTE_SHAPE {
            if ap.abs() < 1e-9 { continue; }
            assert!(
                SILHOUETTE_SHAPE.iter().any(|&(a, p)| (a - al).abs() < 1e-9 && (p + ap).abs() < 1e-9),
                "Missing mirror for ({}, {})", al, ap,
            );
        }
    }

    #[test]
    fn test_silhouette_corners_count() {
        let identity = |x: f64, y: f64| (x, y);
        let pts = silhouette_corners(0.0, 0.0, 0.0, 2.5, 1.0, &identity);
        assert_eq!(pts.len(), SILHOUETTE_SHAPE.len());
    }

    #[test]
    fn test_silhouette_corners_inside_bbox() {
        let identity = |x: f64, y: f64| (x, y);
        let half_l = 2.5;
        let half_w = 1.134;
        let pts = silhouette_corners(10.0, 5.0, 0.0, half_l, half_w, &identity);
        for &(x, y) in &pts {
            assert!(x >= 10.0 - half_l - 0.01 && x <= 10.0 + half_l + 0.01,
                "x={} out of bbox", x);
            assert!(y >= 5.0 - half_w - 0.01 && y <= 5.0 + half_w + 0.01,
                "y={} out of bbox", y);
        }
    }
}
