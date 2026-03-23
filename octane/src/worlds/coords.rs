//! Coordinate types for Octane's coordinate system architecture.
//!
//! Each type represents a point in a specific coordinate system:
//! - [`ScenePoint`]: World/simulation coordinates in meters
//! - [`SvgPoint`]: Normalized SVG coordinates (width=1)

/// Point in scene coordinates (meters).
///
/// This is the simulation world coordinate system used by Highway.
/// Origin is typically at the start of the road segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScenePoint {
    /// X coordinate in meters (positive = forward along road).
    pub x: f64,
    /// Y coordinate in meters (positive = towards upper lanes).
    pub y: f64,
}

impl ScenePoint {
    /// Create a new scene point.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Create point at origin.
    pub fn origin() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    /// Compute distance to another point.
    pub fn distance_to(&self, other: &ScenePoint) -> f64 {
        let dx = other.x - self.x;
        let dy = other.y - self.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Linear interpolation between two points.
    pub fn lerp(&self, other: &ScenePoint, t: f64) -> ScenePoint {
        ScenePoint {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }
}

impl Default for ScenePoint {
    fn default() -> Self {
        Self::origin()
    }
}

/// Point in normalized SVG coordinates.
///
/// Width is always 1.0, height varies with aspect ratio.
/// Origin (0, 0) is top-left corner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SvgPoint {
    /// X coordinate (0.0 = left edge, 1.0 = right edge).
    pub x: f64,
    /// Y coordinate (0.0 = top edge, height = bottom edge).
    pub y: f64,
}

impl SvgPoint {
    /// Create a new SVG point.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Check if point is within the visible SVG bounds.
    pub fn is_visible(&self, svg_height: f64) -> bool {
        self.x >= 0.0 && self.x <= 1.0 && self.y >= 0.0 && self.y <= svg_height
    }
}

impl Default for SvgPoint {
    fn default() -> Self {
        Self { x: 0.0, y: 0.0 }
    }
}

/// Axis-aligned bounding box in scene coordinates.
///
/// Used to represent visible regions and collision bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneBounds {
    /// Minimum X coordinate (left edge).
    pub min_x: f64,
    /// Maximum X coordinate (right edge).
    pub max_x: f64,
    /// Minimum Y coordinate (bottom edge in world coords).
    pub min_y: f64,
    /// Maximum Y coordinate (top edge in world coords).
    pub max_y: f64,
}

impl SceneBounds {
    /// Create bounds from min/max coordinates.
    pub fn new(min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Self {
        Self {
            min_x,
            max_x,
            min_y,
            max_y,
        }
    }

    /// Create bounds centered at a point with given half-extents.
    pub fn centered_at(center: ScenePoint, half_width: f64, half_height: f64) -> Self {
        Self {
            min_x: center.x - half_width,
            max_x: center.x + half_width,
            min_y: center.y - half_height,
            max_y: center.y + half_height,
        }
    }

    /// Width of the bounds.
    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    /// Height of the bounds.
    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    /// Center point of the bounds.
    pub fn center(&self) -> ScenePoint {
        ScenePoint {
            x: (self.min_x + self.max_x) / 2.0,
            y: (self.min_y + self.max_y) / 2.0,
        }
    }

    /// Check if a point is contained within the bounds.
    pub fn contains(&self, point: &ScenePoint) -> bool {
        point.x >= self.min_x
            && point.x <= self.max_x
            && point.y >= self.min_y
            && point.y <= self.max_y
    }

    /// Check if two bounds intersect.
    pub fn intersects(&self, other: &SceneBounds) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }

    /// Expand bounds by a margin on all sides.
    pub fn expand(&self, margin: f64) -> SceneBounds {
        SceneBounds {
            min_x: self.min_x - margin,
            max_x: self.max_x + margin,
            min_y: self.min_y - margin,
            max_y: self.max_y + margin,
        }
    }
}

impl Default for SceneBounds {
    fn default() -> Self {
        Self {
            min_x: 0.0,
            max_x: 0.0,
            min_y: 0.0,
            max_y: 0.0,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // ScenePoint tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_scene_point_new() {
        let p = ScenePoint::new(10.0, 20.0);
        assert_eq!(p.x, 10.0);
        assert_eq!(p.y, 20.0);
    }

    #[test]
    fn test_scene_point_origin() {
        let p = ScenePoint::origin();
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 0.0);
    }

    #[test]
    fn test_scene_point_distance() {
        let p1 = ScenePoint::new(0.0, 0.0);
        let p2 = ScenePoint::new(3.0, 4.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_scene_point_lerp() {
        let p1 = ScenePoint::new(0.0, 0.0);
        let p2 = ScenePoint::new(10.0, 20.0);

        let mid = p1.lerp(&p2, 0.5);
        assert!((mid.x - 5.0).abs() < 1e-9);
        assert!((mid.y - 10.0).abs() < 1e-9);

        let start = p1.lerp(&p2, 0.0);
        assert!((start.x - 0.0).abs() < 1e-9);

        let end = p1.lerp(&p2, 1.0);
        assert!((end.x - 10.0).abs() < 1e-9);
    }

    // -------------------------------------------------------------------------
    // SvgPoint tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_svg_point_visible() {
        let svg_height = 0.5;

        let visible = SvgPoint::new(0.5, 0.25);
        assert!(visible.is_visible(svg_height));

        let off_left = SvgPoint::new(-0.1, 0.25);
        assert!(!off_left.is_visible(svg_height));

        let off_right = SvgPoint::new(1.1, 0.25);
        assert!(!off_right.is_visible(svg_height));

        let off_bottom = SvgPoint::new(0.5, 0.6);
        assert!(!off_bottom.is_visible(svg_height));
    }

    // -------------------------------------------------------------------------
    // SceneBounds tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_scene_bounds_new() {
        let b = SceneBounds::new(0.0, 100.0, -10.0, 10.0);
        assert_eq!(b.min_x, 0.0);
        assert_eq!(b.max_x, 100.0);
        assert_eq!(b.width(), 100.0);
        assert_eq!(b.height(), 20.0);
    }

    #[test]
    fn test_scene_bounds_centered_at() {
        let center = ScenePoint::new(50.0, 0.0);
        let b = SceneBounds::centered_at(center, 10.0, 5.0);
        assert_eq!(b.min_x, 40.0);
        assert_eq!(b.max_x, 60.0);
        assert_eq!(b.min_y, -5.0);
        assert_eq!(b.max_y, 5.0);
    }

    #[test]
    fn test_scene_bounds_center() {
        let b = SceneBounds::new(0.0, 100.0, -10.0, 10.0);
        let c = b.center();
        assert!((c.x - 50.0).abs() < 1e-9);
        assert!((c.y - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_scene_bounds_contains() {
        let b = SceneBounds::new(0.0, 100.0, 0.0, 50.0);

        assert!(b.contains(&ScenePoint::new(50.0, 25.0)));
        assert!(b.contains(&ScenePoint::new(0.0, 0.0)));
        assert!(b.contains(&ScenePoint::new(100.0, 50.0)));

        assert!(!b.contains(&ScenePoint::new(-1.0, 25.0)));
        assert!(!b.contains(&ScenePoint::new(50.0, 51.0)));
    }

    #[test]
    fn test_scene_bounds_intersects() {
        let b1 = SceneBounds::new(0.0, 10.0, 0.0, 10.0);
        let b2 = SceneBounds::new(5.0, 15.0, 5.0, 15.0);
        let b3 = SceneBounds::new(20.0, 30.0, 20.0, 30.0);

        assert!(b1.intersects(&b2));
        assert!(b2.intersects(&b1));
        assert!(!b1.intersects(&b3));
    }

    #[test]
    fn test_scene_bounds_expand() {
        let b = SceneBounds::new(10.0, 20.0, 10.0, 20.0);
        let expanded = b.expand(5.0);
        assert_eq!(expanded.min_x, 5.0);
        assert_eq!(expanded.max_x, 25.0);
        assert_eq!(expanded.min_y, 5.0);
        assert_eq!(expanded.max_y, 25.0);
    }
}
