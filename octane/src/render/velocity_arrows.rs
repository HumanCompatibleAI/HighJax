//! Velocity arrow rendering for vehicles.

/// Render a velocity arrow from a vehicle's center in the direction of travel.
///
/// * `arrow_length` - shaft length in meters (pre-computed from speed * length_scale)
/// * `stroke_width` - SVG stroke width (pre-scaled from pixels)
/// * `head_meters` - arrowhead length in meters (pre-computed and clamped)
/// * `head_angle` - arrowhead half-angle in radians
#[allow(clippy::too_many_arguments)]
pub fn render_velocity_arrow_svg<F>(
    elements: &mut Vec<String>,
    x: f64,
    y: f64,
    heading: f64,
    arrow_length: f64,
    stroke_width: f64,
    color: &str,
    opacity: f64,
    head_meters: f64,
    head_angle: f64,
    world_to_svg: &F,
) where
    F: Fn(f64, f64) -> (f64, f64),
{
    let cos_h = heading.cos();
    let sin_h = heading.sin();

    // Arrow tip
    let tip_x = x + cos_h * arrow_length;
    let tip_y = y + sin_h * arrow_length;

    let (sx1, sy1) = world_to_svg(x, y);
    let (sx2, sy2) = world_to_svg(tip_x, tip_y);

    // Shaft
    elements.push(format!(
        r#"<line x1="{:.6}" y1="{:.6}" x2="{:.6}" y2="{:.6}" stroke="{}" stroke-width="{:.6}" opacity="{:.2}" stroke-linecap="round"/>"#,
        sx1, sy1, sx2, sy2, color, stroke_width, opacity
    ));

    // Arrowhead (two lines from tip, angled back)
    for sign in [-1.0_f64, 1.0] {
        let angle = heading + std::f64::consts::PI + sign * head_angle;
        let hx = tip_x + angle.cos() * head_meters;
        let hy = tip_y + angle.sin() * head_meters;
        let (shx, shy) = world_to_svg(hx, hy);
        elements.push(format!(
            r#"<line x1="{:.6}" y1="{:.6}" x2="{:.6}" y2="{:.6}" stroke="{}" stroke-width="{:.6}" opacity="{:.2}" stroke-linecap="round"/>"#,
            sx2, sy2, shx, shy, color, stroke_width, opacity
        ));
    }
}
