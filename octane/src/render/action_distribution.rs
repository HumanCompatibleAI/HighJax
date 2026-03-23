//! Action distribution overlay: directional arrows + idle circle on ego vehicle.

use crate::config::ActionDistributionConfig;
use crate::data::jsonla::ActionDistribution;

/// Render action distribution overlay on the ego vehicle.
///
/// Draws arrows for directional actions and an idle body tint,
/// using probabilities looked up by deed name from the dynamic distribution.
/// When `show_text` is true, overlays percentage labels on each element.
///
/// Arrow directions are road-aligned (not rotated by ego heading):
/// forward = +X (highway direction), left = -Y, right = +Y.
#[allow(clippy::too_many_arguments)]
pub fn render_action_distribution_svg<F>(
    elements: &mut Vec<String>,
    ego_x: f64,
    ego_y: f64,
    ego_heading: f64,
    vehicle_length: f64,
    vehicle_width: f64,
    dist: &ActionDistribution,
    ad: &ActionDistributionConfig,
    chosen_action: Option<u8>,
    show_text: bool,
    pixel_width: f64,
    world_to_svg: &F,
) where
    F: Fn(f64, f64) -> (f64, f64),
{
    // Stroke width: convert from pixels to SVG viewBox units.
    let svg_stroke = ad.stroke_width / pixel_width;

    // Compute SVG-units-per-meter for circle radius and font sizing.
    let (sx0, sy0) = world_to_svg(ego_x, ego_y);
    let (sx1, _sy1) = world_to_svg(ego_x + 1.0, ego_y);
    let svg_per_meter_x = (sx1 - sx0).abs();
    if svg_per_meter_x < 1e-9 {
        return;
    }

    let font_size = ad.font_size_meters * svg_per_meter_x;

    // Directional arrows: (deed_name, local_dx, local_dy, edge_offset, action_index)
    let half_len = vehicle_length / 2.0;
    let half_wid = vehicle_width / 2.0;
    let arrows: [(&str, f64, f64, f64, u8); 4] = [
        ("faster", 1.0, 0.0, half_len, 3),   // forward: from front edge
        ("slower", -1.0, 0.0, half_len, 4),   // backward: from rear edge
        ("left", 0.0, -1.0, half_wid, 0),     // left: from left edge
        ("right", 0.0, 1.0, half_wid, 2),     // right: from right edge
    ];
    for &(deed_name, dx, dy, edge_offset, action_idx) in arrows.iter() {
        let prob = dist.get(deed_name).unwrap_or(0.0);
        let is_chosen = chosen_action == Some(action_idx);
        let color = if is_chosen { &ad.chosen_color } else { &ad.color };
        let opacity = ad.min_opacity + (ad.max_opacity - ad.min_opacity) * prob;
        let arrow_length = ad.max_arrow_length * prob;
        // Arrow starts from bounding box edge, not center (road-aligned)
        let base_x = ego_x + dx * edge_offset;
        let base_y = ego_y + dy * edge_offset;
        let tip_x = base_x + dx * arrow_length;
        let tip_y = base_y + dy * arrow_length;

        let (s_base_x, s_base_y) = world_to_svg(base_x, base_y);
        let (s_tip_x, s_tip_y) = world_to_svg(tip_x, tip_y);

        // Shaft
        elements.push(format!(
            r#"<line x1="{:.6}" y1="{:.6}" x2="{:.6}" y2="{:.6}" stroke="{}" stroke-width="{:.6}" opacity="{:.2}" stroke-linecap="round"/>"#,
            s_base_x, s_base_y, s_tip_x, s_tip_y, color, svg_stroke, opacity
        ));

        // Arrowhead
        let head_meters = arrow_length * ad.head_size;
        if head_meters > 0.01 {
            let arrow_angle = dy.atan2(dx);
            for sign in [-1.0_f64, 1.0] {
                let ha = arrow_angle + std::f64::consts::PI + sign * ad.head_angle;
                let hx = tip_x + ha.cos() * head_meters;
                let hy = tip_y + ha.sin() * head_meters;
                let (shx, shy) = world_to_svg(hx, hy);
                elements.push(format!(
                    r#"<line x1="{:.6}" y1="{:.6}" x2="{:.6}" y2="{:.6}" stroke="{}" stroke-width="{:.6}" opacity="{:.2}" stroke-linecap="round"/>"#,
                    s_tip_x, s_tip_y, shx, shy, color, svg_stroke, opacity
                ));
            }
        }

        // Text label when paused
        if show_text {
            let pct = prob * 100.0;
            let label_x = ego_x + dx * (edge_offset + ad.text_offset_meters);
            let label_y = ego_y + dy * (edge_offset + ad.text_offset_meters);
            let (slx, sly) = world_to_svg(label_x, label_y);
            let text_fill = if is_chosen && !ad.chosen_text_color.is_empty() {
                &ad.chosen_text_color
            } else if is_chosen {
                color
            } else if !ad.text_color.is_empty() {
                &ad.text_color
            } else {
                color
            };
            let label = format!("{:>5.1}%", pct);
            push_label_with_bg(elements, slx, sly, text_fill, &ad.text_bg, font_size, &ad.font_family, &label);
        }
    }

    // Idle: tint the entire car body
    let idle_prob = dist.get("idle").unwrap_or(0.0);
    let is_idle_chosen = chosen_action == Some(1);
    let idle_color = if is_idle_chosen { &ad.chosen_color } else { &ad.circle_color };
    let idle_opacity = ad.circle_min_opacity + (ad.circle_max_opacity - ad.circle_min_opacity) * idle_prob;
    let cos_h = ego_heading.cos();
    let sin_h = ego_heading.sin();
    let corners = [
        ( half_len,  half_wid),
        ( half_len, -half_wid),
        (-half_len, -half_wid),
        (-half_len,  half_wid),
    ];
    let svg_corners: Vec<(f64, f64)> = corners.iter().map(|&(lx, ly)| {
        let wx = ego_x + lx * cos_h - ly * sin_h;
        let wy = ego_y + lx * sin_h + ly * cos_h;
        world_to_svg(wx, wy)
    }).collect();
    let points: String = svg_corners.iter()
        .map(|(x, y)| format!("{:.6},{:.6}", x, y))
        .collect::<Vec<_>>()
        .join(" ");
    elements.push(format!(
        r#"<polygon points="{}" fill="{}" opacity="{:.2}"/>"#,
        points, idle_color, idle_opacity
    ));

    if show_text {
        let pct = idle_prob * 100.0;
        let text_fill = if is_idle_chosen && !ad.chosen_text_color.is_empty() {
            &ad.chosen_text_color
        } else if is_idle_chosen {
            &ad.chosen_color
        } else if !ad.text_color.is_empty() {
            &ad.text_color
        } else {
            &ad.color
        };
        let label = format!("{:>5.1}%", pct);
        push_label_with_bg(elements, sx0, sy0, text_fill, &ad.text_bg, font_size, &ad.font_family, &label);
    }
}

/// Push a text label with an optional background via thick stroke.
fn push_label_with_bg(
    elements: &mut Vec<String>,
    x: f64, y: f64,
    fill: &str, bg: &str,
    font_size: f64, font_family: &str,
    label: &str,
) {
    // paint-order="stroke" draws stroke behind fill, so a thick bg-colored stroke
    // acts as a tight background that hugs the glyphs.
    let stroke_attr = if !bg.is_empty() {
        format!(r#" stroke="{}" stroke-width="{:.6}" paint-order="stroke" stroke-linejoin="round""#,
            bg, font_size * 0.45)
    } else {
        String::new()
    };
    elements.push(format!(
        r#"<text x="{:.6}" y="{:.6}" fill="{}"{} font-size="{:.6}" font-family="{}" font-weight="bold" text-anchor="middle" dominant-baseline="central" data-mango-fg="{}" data-mango-bg="{}">{}</text>"#,
        x, y, fill, stroke_attr, font_size, font_family, fill, if bg.is_empty() { "#000000" } else { bg }, label
    ));
}

/// Render action **delta** overlay on the ego vehicle.
///
/// Arrows are sized by the **new** probability (like the standard overlay),
/// but colored green/red based on the delta direction. Text labels show
/// the signed delta percentage (+X.X% or -X.X%).
#[allow(clippy::too_many_arguments)]
pub fn render_action_delta_svg<F>(
    elements: &mut Vec<String>,
    ego_x: f64,
    ego_y: f64,
    ego_heading: f64,
    vehicle_length: f64,
    vehicle_width: f64,
    old_probs: &[(String, f64)],
    new_probs: &[(String, f64)],
    ad: &ActionDistributionConfig,
    pixel_width: f64,
    world_to_svg: &F,
) where
    F: Fn(f64, f64) -> (f64, f64),
{
    let svg_stroke = ad.stroke_width / pixel_width;

    let (sx0, sy0) = world_to_svg(ego_x, ego_y);
    let (sx1, _) = world_to_svg(ego_x + 1.0, ego_y);
    let svg_per_meter_x = (sx1 - sx0).abs();
    if svg_per_meter_x < 1e-9 {
        return;
    }
    let font_size = ad.font_size_meters * svg_per_meter_x;

    let get_prob = |probs: &[(String, f64)], name: &str| -> f64 {
        probs.iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, p)| *p)
            .unwrap_or(0.0)
    };

    let increase_color = "#00ff88"; // green
    let decrease_color = "#ff4444"; // red

    let half_len = vehicle_length / 2.0;
    let half_wid = vehicle_width / 2.0;
    let arrows: [(&str, f64, f64, f64); 4] = [
        ("faster", 1.0, 0.0, half_len),
        ("slower", -1.0, 0.0, half_len),
        ("left", 0.0, -1.0, half_wid),
        ("right", 0.0, 1.0, half_wid),
    ];

    for &(deed_name, dx, dy, edge_offset) in arrows.iter() {
        let old_p = get_prob(old_probs, deed_name);
        let new_p = get_prob(new_probs, deed_name);
        let delta = new_p - old_p;
        let color = if delta >= 0.0 { increase_color } else { decrease_color };
        // Arrow sized by new probability (like standard overlay)
        let opacity = ad.min_opacity + (ad.max_opacity - ad.min_opacity) * new_p;
        let arrow_length = ad.max_arrow_length * new_p;

        let base_x = ego_x + dx * edge_offset;
        let base_y = ego_y + dy * edge_offset;
        let tip_x = base_x + dx * arrow_length;
        let tip_y = base_y + dy * arrow_length;

        let (s_base_x, s_base_y) = world_to_svg(base_x, base_y);
        let (s_tip_x, s_tip_y) = world_to_svg(tip_x, tip_y);

        // Shaft
        elements.push(format!(
            r#"<line x1="{:.6}" y1="{:.6}" x2="{:.6}" y2="{:.6}" stroke="{}" stroke-width="{:.6}" opacity="{:.2}" stroke-linecap="round"/>"#,
            s_base_x, s_base_y, s_tip_x, s_tip_y, color, svg_stroke, opacity
        ));

        // Arrowhead
        let head_meters = arrow_length * ad.head_size;
        if head_meters > 0.01 {
            let arrow_angle = dy.atan2(dx);
            for sign in [-1.0_f64, 1.0] {
                let ha = arrow_angle + std::f64::consts::PI + sign * ad.head_angle;
                let hx = tip_x + ha.cos() * head_meters;
                let hy = tip_y + ha.sin() * head_meters;
                let (shx, shy) = world_to_svg(hx, hy);
                elements.push(format!(
                    r#"<line x1="{:.6}" y1="{:.6}" x2="{:.6}" y2="{:.6}" stroke="{}" stroke-width="{:.6}" opacity="{:.2}" stroke-linecap="round"/>"#,
                    s_tip_x, s_tip_y, shx, shy, color, svg_stroke, opacity
                ));
            }
        }

        // Delta text label
        let pct = delta * 100.0;
        let label = if delta >= 0.0 {
            format!("+{:.1}%", pct)
        } else {
            format!("{:.1}%", pct)
        };
        let label_x = ego_x + dx * (edge_offset + ad.text_offset_meters);
        let label_y = ego_y + dy * (edge_offset + ad.text_offset_meters);
        let (slx, sly) = world_to_svg(label_x, label_y);
        push_label_with_bg(elements, slx, sly, color, &ad.text_bg, font_size, &ad.font_family, &label);
    }

    // Idle delta: tint car body by new probability, color by delta direction
    let old_idle = get_prob(old_probs, "idle");
    let new_idle = get_prob(new_probs, "idle");
    let idle_delta = new_idle - old_idle;
    let idle_color = if idle_delta >= 0.0 { increase_color } else { decrease_color };
    let idle_opacity = ad.circle_min_opacity + (ad.circle_max_opacity - ad.circle_min_opacity) * new_idle;

    let cos_h = ego_heading.cos();
    let sin_h = ego_heading.sin();
    let corners = [
        ( half_len,  half_wid),
        ( half_len, -half_wid),
        (-half_len, -half_wid),
        (-half_len,  half_wid),
    ];
    let svg_corners: Vec<(f64, f64)> = corners.iter().map(|&(lx, ly)| {
        let wx = ego_x + lx * cos_h - ly * sin_h;
        let wy = ego_y + lx * sin_h + ly * cos_h;
        world_to_svg(wx, wy)
    }).collect();
    let points: String = svg_corners.iter()
        .map(|(x, y)| format!("{:.6},{:.6}", x, y))
        .collect::<Vec<_>>()
        .join(" ");
    elements.push(format!(
        r#"<polygon points="{}" fill="{}" opacity="{:.2}"/>"#,
        points, idle_color, idle_opacity
    ));

    let pct = idle_delta * 100.0;
    let label = if idle_delta >= 0.0 {
        format!("+{:.1}%", pct)
    } else {
        format!("{:.1}%", pct)
    };
    push_label_with_bg(elements, sx0, sy0, idle_color, &ad.text_bg, font_size, &ad.font_family, &label);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_highway_dist(left: f64, idle: f64, right: f64, faster: f64, slower: f64) -> ActionDistribution {
        ActionDistribution {
            probs: vec![
                ("left".into(), left), ("idle".into(), idle), ("right".into(), right),
                ("faster".into(), faster), ("slower".into(), slower),
            ],
        }
    }

    #[test]
    fn test_render_produces_reasonable_svg() {
        let dist = make_highway_dist(0.1, 0.5, 0.1, 0.2, 0.1);

        let mut elements = Vec::new();
        let scale = 0.005;
        let offset_x = 0.5;
        let offset_y = 0.3;
        let world_to_svg = |wx: f64, wy: f64| -> (f64, f64) {
            (offset_x + (wx - 50.0) * scale, offset_y + (wy - 4.0) * scale)
        };
        let ad = ActionDistributionConfig {
            stroke_width: 3.0,
            max_arrow_length: 4.0,
            ..Default::default()
        };

        render_action_distribution_svg(
            &mut elements,
            50.0, 4.0, 0.0,
            5.0, 2.0,
            &dist,
            &ad,
            Some(3), // chosen: FASTER
            true,
            960.0,
            &world_to_svg,
        );

        assert!(!elements.is_empty(), "Should produce SVG elements");

        // Check stroke-width is reasonable (should be 3/960 ~ 0.003125)
        let expected_stroke = 3.0 / 960.0;
        for elem in &elements {
            if let Some(pos) = elem.find("stroke-width=\"") {
                let start = pos + "stroke-width=\"".len();
                let end = elem[start..].find('"').unwrap() + start;
                let sw: f64 = elem[start..end].parse().unwrap();
                assert!(sw < 0.01, "Stroke-width {} too large (expected ~{:.6})", sw, expected_stroke);
                assert!(sw > 0.001, "Stroke-width {} too small (expected ~{:.6})", sw, expected_stroke);
            }
        }

        // Check idle polygon exists (car body overlay)
        let polygon_count = elements.iter().filter(|e| e.contains("<polygon")).count();
        assert_eq!(polygon_count, 1, "Should have exactly one idle polygon overlay");

        // Check font-size is reasonable
        for elem in &elements {
            if let Some(pos) = elem.find("font-size=\"") {
                let start = pos + "font-size=\"".len();
                let end = elem[start..].find('"').unwrap() + start;
                let fs: f64 = elem[start..end].parse().unwrap();
                assert!(fs < 0.02, "Font-size {} SVG units too large", fs);
                assert!(fs > 0.001, "Font-size {} SVG units too small", fs);
            }
        }
    }

    #[test]
    fn test_zero_probs_still_render() {
        let dist = make_highway_dist(0.0, 0.0, 0.0, 0.0, 0.0);
        let mut elements = Vec::new();
        let world_to_svg = |wx: f64, wy: f64| (wx * 0.005, wy * 0.005);
        let ad = ActionDistributionConfig::default();

        render_action_distribution_svg(
            &mut elements, 50.0, 4.0, 0.0, 5.0, 2.0, &dist,
            &ad, None, true, 960.0, &world_to_svg,
        );

        // Even with zero probs, elements should be rendered (arrows + text + idle circle + text)
        assert!(!elements.is_empty(), "Zero probs should still produce elements");
        // Should have text labels for all 5 actions
        let text_count = elements.iter().filter(|e| e.contains("<text")).count();
        assert_eq!(text_count, 5, "Should have text labels for all 5 actions");
    }

    #[test]
    fn test_empty_distribution_renders_nothing() {
        let dist = ActionDistribution { probs: vec![] };
        let mut elements = Vec::new();
        let world_to_svg = |wx: f64, wy: f64| (wx * 0.005, wy * 0.005);
        let ad = ActionDistributionConfig::default();

        render_action_distribution_svg(
            &mut elements, 50.0, 4.0, 0.0, 5.0, 2.0, &dist,
            &ad, None, false, 960.0, &world_to_svg,
        );

        // Should still render arrows (with 0 prob) and idle polygon
        let line_count = elements.iter().filter(|e| e.contains("<line")).count();
        assert_eq!(line_count, 4, "Should have 4 arrow shafts even with empty dist");
    }

    fn make_prob_pairs(left: f64, idle: f64, right: f64, faster: f64, slower: f64) -> Vec<(String, f64)> {
        vec![
            ("left".into(), left), ("idle".into(), idle), ("right".into(), right),
            ("faster".into(), faster), ("slower".into(), slower),
        ]
    }

    #[test]
    fn test_delta_render_produces_elements() {
        let old_probs = make_prob_pairs(0.2, 0.3, 0.2, 0.2, 0.1);
        let new_probs = make_prob_pairs(0.1, 0.2, 0.3, 0.3, 0.1);
        let mut elements = Vec::new();
        let world_to_svg = |wx: f64, wy: f64| (wx * 0.005, wy * 0.005);
        let ad = ActionDistributionConfig::default();

        render_action_delta_svg(
            &mut elements, 50.0, 4.0, 0.0, 5.0, 2.0,
            &old_probs, &new_probs, &ad, 960.0, &world_to_svg,
        );

        assert!(!elements.is_empty());
        // 5 text labels (4 directional + 1 idle)
        let text_count = elements.iter().filter(|e| e.contains("<text")).count();
        assert_eq!(text_count, 5);
        // 1 idle polygon
        let polygon_count = elements.iter().filter(|e| e.contains("<polygon")).count();
        assert_eq!(polygon_count, 1);
    }

    #[test]
    fn test_delta_colors_green_red() {
        let old_probs = make_prob_pairs(0.2, 0.3, 0.2, 0.2, 0.1);
        let new_probs = make_prob_pairs(0.1, 0.2, 0.3, 0.3, 0.1); // left/idle decreased, right/faster increased
        let mut elements = Vec::new();
        let world_to_svg = |wx: f64, wy: f64| (wx * 0.005, wy * 0.005);
        let ad = ActionDistributionConfig::default();

        render_action_delta_svg(
            &mut elements, 50.0, 4.0, 0.0, 5.0, 2.0,
            &old_probs, &new_probs, &ad, 960.0, &world_to_svg,
        );

        // Check that green (#00ff88) and red (#ff4444) colors appear
        let has_green = elements.iter().any(|e| e.contains("#00ff88"));
        let has_red = elements.iter().any(|e| e.contains("#ff4444"));
        assert!(has_green, "Should have green elements for increases");
        assert!(has_red, "Should have red elements for decreases");
    }

    #[test]
    fn test_delta_labels_show_signed_percent() {
        let old_probs = make_prob_pairs(0.2, 0.3, 0.2, 0.2, 0.1);
        let new_probs = make_prob_pairs(0.1, 0.2, 0.3, 0.3, 0.1);
        let mut elements = Vec::new();
        let world_to_svg = |wx: f64, wy: f64| (wx * 0.005, wy * 0.005);
        let ad = ActionDistributionConfig::default();

        render_action_delta_svg(
            &mut elements, 50.0, 4.0, 0.0, 5.0, 2.0,
            &old_probs, &new_probs, &ad, 960.0, &world_to_svg,
        );

        let texts: Vec<&String> = elements.iter().filter(|e| e.contains("<text")).collect();

        // Should have signed labels: + for increases, - for decreases
        let has_plus = texts.iter().any(|t| t.contains(">+"));
        let has_minus = texts.iter().any(|t| t.contains(">-"));
        assert!(has_plus, "Should have positive delta labels");
        assert!(has_minus, "Should have negative delta labels");
    }

    #[test]
    fn test_delta_zero_change_still_renders() {
        let probs = make_prob_pairs(0.2, 0.3, 0.2, 0.2, 0.1);
        let mut elements = Vec::new();
        let world_to_svg = |wx: f64, wy: f64| (wx * 0.005, wy * 0.005);
        let ad = ActionDistributionConfig::default();

        render_action_delta_svg(
            &mut elements, 50.0, 4.0, 0.0, 5.0, 2.0,
            &probs, &probs, &ad, 960.0, &world_to_svg,
        );

        assert!(!elements.is_empty(), "Should produce elements even with zero deltas");
    }
}
