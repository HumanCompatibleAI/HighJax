//! Terrain noise and blob generation for highway backgrounds.
//!
//! Deterministic cloud-like noise produces organic terrain blobs
//! that tile seamlessly as the viewport scrolls.

/// Simple hash-based noise function returning 0-1.
fn noise_hash(x: i32, y: i32) -> f64 {
    let mut h = ((x as u32).wrapping_mul(374761393))
        .wrapping_add((y as u32).wrapping_mul(668265263));
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    ((h & 0xFFFF) as f64) / 65535.0
}

/// Smooth interpolation function.
fn smoothstep(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

/// Linear interpolation.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Generate smooth cloud-like noise for terrain (deterministic based on world coords).
fn terrain_noise(world_x: f64, world_y: f64, scale: f64) -> f64 {
    let x = world_x / scale;
    let y = world_y / scale;

    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let sx = smoothstep(fx);
    let sy = smoothstep(fy);

    let n00 = noise_hash(x0, y0);
    let n10 = noise_hash(x1, y0);
    let n01 = noise_hash(x0, y1);
    let n11 = noise_hash(x1, y1);

    let nx0 = lerp(n00, n10, sx);
    let nx1 = lerp(n01, n11, sx);
    lerp(nx0, nx1, sy)
}

/// Generate an organic blob path using bezier curves.
/// Returns SVG path data for a cloud-like shape.
fn generate_blob_path(cx: f64, cy: f64, base_rx: f64, base_ry: f64, seed_x: f64, seed_y: f64) -> String {
    let n_points = 8; // Number of control points around the blob
    let mut path = String::new();

    // Generate points around the blob with noise-perturbed radii
    let mut points: Vec<(f64, f64)> = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let angle = (i as f64 / n_points as f64) * std::f64::consts::TAU;

        // Perturb radius using noise based on angle and seed
        let noise_input_x = seed_x + angle.cos() * 5.0;
        let noise_input_y = seed_y + angle.sin() * 5.0;
        let radius_noise = noise_hash(
            (noise_input_x * 100.0) as i32,
            (noise_input_y * 100.0) as i32
        );

        let r_factor = 0.3 + radius_noise * 1.4; // 0.3 to 1.7 - more variation
        let rx = base_rx * r_factor;
        let ry = base_ry * r_factor;

        let px = cx + angle.cos() * rx;
        let py = cy + angle.sin() * ry;
        points.push((px, py));
    }

    // Build smooth closed path using quadratic beziers
    // Move to midpoint between last and first point
    let (x0, y0) = points[0];
    let (xlast, ylast) = points[n_points - 1];
    let start_x = (xlast + x0) / 2.0;
    let start_y = (ylast + y0) / 2.0;

    path.push_str(&format!("M{:.4},{:.4}", start_x, start_y));

    // Quadratic bezier through each point, with control point at the point
    // and end point at midpoint to next point
    for i in 0..n_points {
        let (px, py) = points[i];
        let (nx, ny) = points[(i + 1) % n_points];
        let mid_x = (px + nx) / 2.0;
        let mid_y = (py + ny) / 2.0;
        path.push_str(&format!("Q{:.4},{:.4},{:.4},{:.4}", px, py, mid_x, mid_y));
    }

    path.push('Z');
    path
}

/// Generate terrain blobs using pre-computed visible bounds from episode.
#[allow(clippy::too_many_arguments)]
pub fn generate_terrain_blobs_from_bounds(
    bounds: &crate::worlds::SceneBounds,
    svg_width: f64,
    svg_height: f64,
    scale: f64,
    view_x: f64,
    view_y: f64,
    road_min_y: f64,
    road_max_y: f64,
    terrain_scale: f64,
    terrain_density: f64,
    terrain_blob_size_min: f64,
    terrain_blob_size_range: f64,
    color_terrain: &str,
) -> Vec<String> {
    let mut elements = Vec::new();

    // Use bounds from episode
    let world_left = bounds.min_x;
    let world_right = bounds.max_x;
    let world_top = bounds.min_y;
    let world_bottom = bounds.max_y;

    // Snap to grid
    let grid_left = (world_left / terrain_scale).floor() as i32 - 1;
    let grid_right = (world_right / terrain_scale).ceil() as i32 + 1;
    let grid_top = (world_top / terrain_scale).floor() as i32 - 1;
    let grid_bottom = (world_bottom / terrain_scale).ceil() as i32 + 1;

    for gy in grid_top..=grid_bottom {
        for gx in grid_left..=grid_right {
            let world_cx = (gx as f64 + 0.5) * terrain_scale;
            let world_cy = (gy as f64 + 0.5) * terrain_scale;

            // Skip if in road area
            if world_cy >= road_min_y && world_cy <= road_max_y {
                continue;
            }

            // Use noise to determine if blob exists
            let noise_val = terrain_noise(world_cx, world_cy, terrain_scale * 2.0);
            if noise_val < terrain_density {
                continue;
            }

            // Convert to SVG coords
            let svg_cx = (world_cx - view_x) * scale + svg_width / 2.0;
            let svg_cy = (world_cy - view_y) * scale + svg_height / 2.0;

            // Blob size in world meters, scaled to SVG via scale factor
            let size_noise = terrain_noise(world_cx * 1.7, world_cy * 1.3, terrain_scale);
            let r_meters = terrain_scale * (terrain_blob_size_min + size_noise * terrain_blob_size_range);
            let rx = r_meters * scale;
            let ry = rx;

            // Skip if outside visible area (margin accounts for max blob extent)
            let margin = rx * 1.7;
            if svg_cx < -margin || svg_cx > svg_width + margin
               || svg_cy < -margin || svg_cy > svg_height + margin {
                continue;
            }

            // Generate organic blob path
            let path_data = generate_blob_path(svg_cx, svg_cy, rx, ry, world_cx, world_cy);
            elements.push(format!(
                r##"<path d="{}" fill="{}"/>"##,
                path_data, color_terrain
            ));
        }
    }

    elements
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terrain_noise() {
        let noise = terrain_noise(0.0, 0.0, 8.0);
        assert!(noise >= 0.0 && noise <= 1.0);

        let noise2 = terrain_noise(100.0, 100.0, 8.0);
        assert!(noise2 >= 0.0 && noise2 <= 1.0);
    }
}
