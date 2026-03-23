//! Pure polygon math for headlight shadow occlusion.
//!
//! Provides Sutherland-Hodgman clipping, convex hull (Andrew's monotone chain),
//! and convex polygon subtraction via edge peeling.

/// A 2D point in SVG coordinate space.
pub type Pt = (f64, f64);

/// Sutherland-Hodgman: clip polygon to the half-plane where
/// cross(lp1->lp2, lp1->point) >= 0.
pub fn clip_polygon_half(poly: &[Pt], lp1: Pt, lp2: Pt) -> Vec<Pt> {
    if poly.len() < 3 {
        return vec![];
    }
    let cross =
        |p: Pt| (lp2.0 - lp1.0) * (p.1 - lp1.1) - (lp2.1 - lp1.1) * (p.0 - lp1.0);
    let mut result = Vec::new();
    let n = poly.len();
    for i in 0..n {
        let curr = poly[i];
        let next = poly[(i + 1) % n];
        let cc = cross(curr);
        let nc = cross(next);
        if cc >= 0.0 {
            result.push(curr);
        }
        if (cc > 0.0 && nc < 0.0) || (cc < 0.0 && nc > 0.0) {
            let t = cc / (cc - nc);
            result.push((
                curr.0 + t * (next.0 - curr.0),
                curr.1 + t * (next.1 - curr.1),
            ));
        }
    }
    result
}

/// Orient line p1->p2 so that `ref_pt` is on the kept (>= 0) side.
pub fn orient_line_towards(p1: Pt, p2: Pt, ref_pt: Pt) -> (Pt, Pt) {
    let cross =
        (p2.0 - p1.0) * (ref_pt.1 - p1.1) - (p2.1 - p1.1) * (ref_pt.0 - p1.0);
    if cross >= 0.0 {
        (p1, p2)
    } else {
        (p2, p1)
    }
}

/// Convex hull via Andrew's monotone chain (returns CCW polygon).
/// Takes ownership to sort in-place (avoids cloning).
pub fn convex_hull(mut pts: Vec<Pt>) -> Vec<Pt> {
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap())
    });
    pts.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12);
    if pts.len() <= 2 { return pts; }
    let cross = |o: Pt, a: Pt, b: Pt| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut hull = Vec::new();
    for &p in &pts {
        while hull.len() >= 2 && cross(hull[hull.len()-2], hull[hull.len()-1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower_len = hull.len();
    for &p in pts.iter().rev().skip(1) {
        while hull.len() > lower_len && cross(hull[hull.len()-2], hull[hull.len()-1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}

/// Subtract a convex shadow polygon from a convex visible piece.
/// Returns the visible (non-shadow) fragments using edge peeling.
pub fn subtract_convex(poly: &[Pt], shadow: &[Pt]) -> Vec<Vec<Pt>> {
    if shadow.len() < 3 { return vec![poly.to_vec()]; }
    let mut results = Vec::new();
    let mut remainder = poly.to_vec();
    // Shadow centroid for orienting edges
    let cx = shadow.iter().map(|p| p.0).sum::<f64>() / shadow.len() as f64;
    let cy = shadow.iter().map(|p| p.1).sum::<f64>() / shadow.len() as f64;
    let centroid = (cx, cy);
    let n = shadow.len();
    for i in 0..n {
        let e1 = shadow[i];
        let e2 = shadow[(i + 1) % n];
        let (p1, p2) = orient_line_towards(e1, e2, centroid);
        // Outside this edge (away from centroid) = visible piece
        let outside = clip_polygon_half(&remainder, p2, p1);
        if outside.len() >= 3 {
            results.push(outside);
        }
        // Inside this edge (toward centroid) = continue peeling
        let inside = clip_polygon_half(&remainder, p1, p2);
        if inside.len() < 3 { return results; }
        remainder = inside;
    }
    // remainder is fully inside shadow -> discarded
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convex_hull_triangle() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)];
        let hull = convex_hull(pts);
        assert_eq!(hull.len(), 3);
    }

    #[test]
    fn test_convex_hull_with_interior_points() {
        let pts = vec![
            (0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0),
            (1.0, 1.0), (0.5, 0.5), // interior points
        ];
        let hull = convex_hull(pts);
        assert_eq!(hull.len(), 4); // only the 4 corners
    }

    #[test]
    fn test_convex_hull_collinear() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)];
        let hull = convex_hull(pts);
        assert!(hull.len() <= 3);
    }

    #[test]
    fn test_subtract_convex_no_overlap() {
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let shadow = vec![(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0)];
        let pieces = subtract_convex(&poly, &shadow);
        let total_area: f64 = pieces.iter().map(|p| polygon_area(p)).sum();
        assert!((total_area - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_subtract_convex_full_overlap() {
        let poly = vec![(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)];
        let shadow = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let pieces = subtract_convex(&poly, &shadow);
        let total_area: f64 = pieces.iter().map(|p| polygon_area(p)).sum();
        assert!(total_area < 0.01);
    }

    #[test]
    fn test_subtract_convex_partial_overlap() {
        let poly = vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
        let shadow = vec![(1.0, -1.0), (3.0, -1.0), (3.0, 3.0), (1.0, 3.0)];
        let pieces = subtract_convex(&poly, &shadow);
        let total_area: f64 = pieces.iter().map(|p| polygon_area(p)).sum();
        assert!((total_area - 2.0).abs() < 0.1);
    }

    /// Signed area of a polygon (positive for CCW).
    fn polygon_area(poly: &[Pt]) -> f64 {
        let n = poly.len();
        let mut area = 0.0;
        for i in 0..n {
            let (x1, y1) = poly[i];
            let (x2, y2) = poly[(i + 1) % n];
            area += x1 * y2 - x2 * y1;
        }
        (area / 2.0).abs()
    }
}
