//! Car SVG template for vehicle rendering.
//!
//! Embeds a top-view car SVG (originally pointing up) with per-vehicle
//! gradient IDs and hue-adjusted body colors.

/// Car SVG viewBox dimensions (pointing up, nose at y=0).
pub const CAR_SVG_WIDTH: f64 = 358.85;
pub const CAR_SVG_HEIGHT: f64 = 789.36;

/// Brighten or darken an RGB color by a multiplicative factor.
fn scale_color(r: u8, g: u8, b: u8, factor: f64) -> (u8, u8, u8) {
    (
        (r as f64 * factor).min(255.0) as u8,
        (g as f64 * factor).min(255.0) as u8,
        (b as f64 * factor).min(255.0) as u8,
    )
}

/// Generate gradient definitions for one vehicle.
///
/// Each gradient ID is suffixed with `_v{vid}` to avoid conflicts when
/// multiple cars are embedded in the same SVG.
pub fn car_defs(vid: usize, r: u8, g: u8, b: u8) -> String {
    let (lr, lg, lb) = scale_color(r, g, b, 1.15);
    let (dr, dg, db) = scale_color(r, g, b, 0.72);
    format!(
        r##"<linearGradient id="cg_v{vid}">
  <stop stop-color="#{lr:02x}{lg:02x}{lb:02x}" offset="0"/>
  <stop stop-color="#{dr:02x}{dg:02x}{db:02x}" offset="1"/>
</linearGradient>
<radialGradient id="cr1_v{vid}" xlink:href="#cg_v{vid}" gradientUnits="userSpaceOnUse" cy="819.9" cx="378.79" gradientTransform="matrix(1.1693 0 0 .93776 -273.1 -147.55)" r="115.39"/>
<radialGradient id="cr2_v{vid}" xlink:href="#cg_v{vid}" gradientUnits="userSpaceOnUse" cy="598.63" cx="294.43" gradientTransform="matrix(1.5041 0 0 .74199 -264.04 -146.71)" r="91.234"/>
<radialGradient id="cr3_v{vid}" xlink:href="#cg_v{vid}" gradientUnits="userSpaceOnUse" cy="153.65" cx="897.24" gradientTransform="matrix(-.39086 0 0 3.6334 703.38 -167.47)" r="15.849"/>
<radialGradient id="cr4_v{vid}" xlink:href="#cg_v{vid}" gradientUnits="userSpaceOnUse" cy="153.65" cx="897.24" gradientTransform="matrix(.39086 0 0 3.6334 -343 -162.47)" r="15.849"/>
<radialGradient id="cr5_v{vid}" xlink:href="#cg_v{vid}" gradientUnits="userSpaceOnUse" cy="319.37" cx="657.76" gradientTransform="matrix(.65803 0 0 1.5197 -252.14 -111.82)" r="1269.3"/>"##,
        vid = vid,
        lr = lr, lg = lg, lb = lb,
        dr = dr, dg = dg, db = db,
    )
}

/// Generate car SVG body elements referencing vehicle-specific gradient IDs.
///
/// Non-gradient body fills (`#f7ce00`) are replaced with the vehicle's color.
/// Non-body colors (wheels, headlights, outlines) are kept as-is.
pub fn car_body(vid: usize, r: u8, g: u8, b: u8, window_color: &str) -> String {
    let fill = format!("#{:02x}{:02x}{:02x}", r, g, b);
    format!(
        r##"<rect fill-rule="evenodd" rx="8.5849" ry="8.5849" height="78.696" width="27.775" y="623.04" x="16.287" fill="#60585a"/>
<rect fill-rule="evenodd" rx="8.5849" ry="8.5849" height="78.696" width="27.775" y="613.04" x="311.29" fill="#60585a"/>
<rect fill-rule="evenodd" rx="8.5849" ry="8.5849" height="78.696" width="27.775" y="98.038" x="318.79" fill="#60585a"/>
<rect fill-rule="evenodd" rx="8.5849" ry="8.5849" height="78.696" width="27.775" y="101.12" x="8.6333" fill="#60585a"/>
<path d="m178.73 782.98c-113.07 2.362-130.4-17.92-147.11-21.261-16.705-38.776-19.877-365.73-9.855-392.46 7.493-60.54-4.936-70.565-8.687-143.53-7.14-85.213 9.815-37.829-4.439-124.48 21.658-90.216-19.136-92.053 168.52-100.63 172.21 2.401 147.96 10.415 169.61 100.63-14.254 86.652 2.701 39.268-4.439 124.48-3.751 72.961-16.18 82.986-8.687 143.53 10.022 26.727 6.85 353.68-9.855 392.46-26.153 15.153-95.459 21.261-145.07 21.261z" fill-rule="evenodd" stroke="#000" stroke-width="1pt" fill="url(#cr5_v{vid})"/>
<path d="m41.537 281.88s21.534 82.442 21.534 82.442 0 154.58-3.0758 157.15c-3.0771 2.5759-27.686 46.372-27.686 46.372s6.1517-280.81 9.2288-285.97z" fill-rule="evenodd" fill="url(#cr4_v{vid})"/>
<path d="m318.84 276.88s-21.534 82.442-21.534 82.442 0 154.58 3.0758 157.15c3.0771 2.5759 27.686 46.372 27.686 46.372s-6.1517-280.81-9.2288-285.97z" fill-rule="evenodd" fill="url(#cr3_v{vid})"/>
<path d="m37.198 44.521c-11.667 18.667-10.816 196.22 7.851 210.22 18.03-14.851 122.48-28.646 142.34-27.364 20.288-1.492 99.694 8.055 124.09 22.697 6.577 2.13 19.727-205.55 1.059-219.55-58.34-16.344-252.01-16.344-275.34 13.99z" stroke-opacity=".45912" fill-rule="evenodd" stroke="#000" stroke-width="1pt" fill="{fill}"/>
<path d="m41.764 257.39s65.53-26.897 138.64-27.585c65.833-0.64088 95.565 9.623 135.61 25.019-4.8534 48.756-9.7078 94.943-21.843 110.34-111.64-28.23-118.92-28.23-230.56 0-2.43-15.39-24.273-105.21-21.846-107.77z" fill-rule="evenodd" fill="url(#cr2_v{vid})"/>
<path d="m45.51 260.2s63.783-24.762 134.95-25.395c64.078-0.59 93.017 8.859 132 23.033-4.724 44.885-9.449 87.405-21.261 101.58-108.67-25.986-115.75-25.985-224.42 0.001-2.362-14.174-23.623-96.856-21.261-99.218z" fill-rule="evenodd" fill="{window}" stroke="#000" stroke-width="1pt"/>
<path d="m75.697 531.43c-12.369 59.368-22.263 173.16-22.263 173.16 19.79 17.316 121.21 24.737 123.69 24.737 7.4212 0 108.84-7.4212 126.16-32.158 0-14.842-9.8956-121.21-19.79-168.21-86.579 12.368-202.84 7.4212-207.79 2.4734z" fill-rule="evenodd" fill="url(#cr1_v{vid})"/>
<path d="m80.945 536.59c-11.812 56.695-21.261 165.36-21.261 165.36 18.899 16.536 115.75 23.623 118.12 23.623 7.087 0 103.94-7.087 120.48-30.71 0-14.174-9.45-115.75-18.899-160.64-82.681 11.811-193.71 7.087-198.44 2.362z" fill-rule="evenodd" fill="{window}" stroke="#000" stroke-width="1pt"/>
<path d="m321.9 279.09l28.348 2.362s14.174 14.174 4.725 21.261c-9.45 7.087-33.073-2.362-33.073-2.362v-21.261z" stroke-opacity=".55346" fill-rule="evenodd" stroke="#000" stroke-width="1.25" fill="{fill}"/>
<path d="m36.946 283.82l-28.348 2.362s-14.174 14.174-4.725 21.261c9.45 7.087 33.073-2.362 33.073-2.362v-21.261z" stroke-opacity=".54717" fill-rule="evenodd" stroke="#000" stroke-width="1pt" fill="{fill}"/>
<path d="m52.582 17.025c-9.023 1.573-19.32 18.998-14.584 21.902 11.281-5.689 33.327-13.506 54.984-15.684 4.286-2.3 10.138-9.356 10.358-10.929-14.886-0.847-40.915 2.775-50.758 4.711z" stroke-opacity=".25157" fill-rule="evenodd" stroke="#000" stroke-width="1.082pt" fill="#ffffff"/>
<path d="m298.7 12.115c9.0231 1.5731 19.32 14.633 14.584 17.537-11.28-5.6886-33.327-7.5051-54.984-9.6837-4.2855-2.2999-10.134-10.992-10.36-12.565 14.888-0.84719 40.917 2.7754 50.76 4.7115z" stroke-opacity=".25157" fill-rule="evenodd" stroke="#000" stroke-width="1.082pt" fill="#ffffff"/>
<path d="m112.29 10.38c37.098-6.547 99.837-6.002 126.57-2.729" stroke-opacity=".37107" stroke="#000" stroke-width="1pt" fill="none"/>
<path d="m112.29 13.108c37.098-6.547 99.837-6.002 126.57-2.729" stroke-opacity=".37107" stroke="#000" stroke-width="1pt" fill="none"/>
<path d="m112.29 16.381c37.098-6.547 99.837-6.002 126.57-2.729" stroke-opacity=".37107" stroke="#000" stroke-width="1pt" fill="none"/>
<path d="m112.29 20.2c37.098-6.001 98.2-4.911 124.39-3.275" stroke-opacity=".37107" stroke="#000" stroke-width="1pt" fill="none"/>
<path d="m43.148 295.63s16.536 75.595 16.536 75.595 0 141.74-2.362 144.1c-2.363 2.362-21.261 42.521-21.261 42.521s4.724-257.49 7.087-262.22z" fill-rule="evenodd" fill="{window}" stroke="#000" stroke-width="1pt"/>
<path d="m36.35 310.53c-2.333 25.667-14.991 443.45 4.666 443.34 69.856 26.467 221.62 23.584 278.52-7.0003 17.53-0.0711 6.999-417.68 4.666-443.34" stroke="#000" stroke-width="1pt" fill="none"/>
<path d="m317.18 290.91s-16.536 75.595-16.536 75.595 0 141.74 2.362 144.1c2.363 2.362 21.261 42.521 21.261 42.521s-4.724-257.49-7.087-262.22z" fill-rule="evenodd" fill="{window}" stroke="#000" stroke-width="1pt"/>
"##,
        vid = vid,
        fill = fill,
        window = window_color,
    )
}

/// Render a car as SVG elements with proper positioning and rotation.
///
/// Returns (defs, element) where defs should go in the SVG `<defs>` block
/// and element is the positioned `<g>` group.
#[allow(clippy::too_many_arguments)]
pub fn render_car<F>(
    vid: usize,
    x: f64,
    y: f64,
    heading: f64,
    length: f64,
    width: f64,
    r: u8,
    g: u8,
    b: u8,
    window_color: &str,
    world_to_svg: &F,
) -> (String, String)
where
    F: Fn(f64, f64) -> (f64, f64),
{
    // Vehicle center in SVG coordinates
    let (cx, cy) = world_to_svg(x, y);

    // Compute SVG scale (SVG units per world meter)
    let (px, py) = world_to_svg(x + 1.0, y);
    let svg_per_meter = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();

    // Scale factors: map car template dimensions to vehicle world dimensions
    // Car SVG points up: x-axis = width (358.85), y-axis = length (789.36)
    // scale(sx,sy) applies before rotation, so sx→template x, sy→template y
    let sx = width * svg_per_meter / CAR_SVG_WIDTH * 1.134;
    let sy = length * svg_per_meter / CAR_SVG_HEIGHT;

    // Rotation: car SVG points up (-Y in SVG), vehicle heading=0 points right (+X)
    // Rotate 90° CW to align up→right, then add heading
    let angle_deg = 90.0 + heading.to_degrees();

    // Center of car template
    let car_cx = CAR_SVG_WIDTH / 2.0;
    let car_cy = CAR_SVG_HEIGHT / 2.0;

    let defs = car_defs(vid, r, g, b);
    let body = car_body(vid, r, g, b, window_color);

    let element = format!(
        r#"<g transform="translate({cx:.6},{cy:.6}) rotate({angle:.2}) scale({sx:.8},{sy:.8}) translate({ncx:.2},{ncy:.2})">{body}</g>"#,
        cx = cx,
        cy = cy,
        angle = angle_deg,
        sx = sx,
        sy = sy,
        ncx = -car_cx,
        ncy = -car_cy,
        body = body,
    );

    (defs, element)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_color_brighten() {
        let (r, g, b) = scale_color(100, 200, 50, 1.2);
        assert_eq!(r, 120);
        assert_eq!(g, 240);
        assert_eq!(b, 60);
    }

    #[test]
    fn test_scale_color_clamps_at_255() {
        let (r, _, _) = scale_color(250, 100, 100, 1.2);
        assert_eq!(r, 255); // 250 * 1.2 = 300, clamped
    }

    #[test]
    fn test_scale_color_darken() {
        let (r, g, b) = scale_color(100, 200, 100, 0.5);
        assert_eq!(r, 50);
        assert_eq!(g, 100);
        assert_eq!(b, 50);
    }

    #[test]
    fn test_car_defs_unique_ids() {
        let d0 = car_defs(0, 200, 100, 50);
        let d1 = car_defs(1, 200, 100, 50);
        assert!(d0.contains("cg_v0"));
        assert!(d0.contains("cr5_v0"));
        assert!(d1.contains("cg_v1"));
        assert!(d1.contains("cr5_v1"));
        assert!(!d0.contains("_v1"));
    }

    #[test]
    fn test_car_body_uses_fill_color() {
        let body = car_body(3, 0xAA, 0xBB, 0xCC, "#1a1a2e");
        assert!(body.contains(r##"fill="#aabbcc""##));
        assert!(body.contains("url(#cr5_v3)"));
    }

    #[test]
    fn test_render_car_produces_group() {
        let identity = |x: f64, y: f64| (x, y);
        let (defs, elem) = render_car(0, 10.0, 5.0, 0.0, 5.0, 2.0, 180, 180, 180, "#1a1a2e", &identity);
        assert!(defs.contains("linearGradient"));
        assert!(elem.starts_with("<g transform="));
        assert!(elem.contains("rotate(90"));
        assert!(elem.ends_with("</g>"));
    }

    #[test]
    fn test_render_car_heading_rotation() {
        let identity = |x: f64, y: f64| (x, y);
        // heading = PI/2 → angle = 90 + 90 = 180
        let (_, elem) = render_car(
            0, 0.0, 0.0, std::f64::consts::FRAC_PI_2, 5.0, 2.0, 100, 100, 100, "#1a1a2e", &identity,
        );
        assert!(elem.contains("rotate(180"));
    }
}
