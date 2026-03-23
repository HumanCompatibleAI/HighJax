//! Color conversion utilities for rendering.
//!
//! Supports multiple color formats:
//! - Hex: `#rrggbb`, `rrggbb`, `#rgb`, or `rgb`
//! - RGB: `rgb(r, g, b)` with 0-255 values
//! - OKLCH: `oklch(L, C, H)` with L in 0-1, C in 0-~0.4, H in degrees

/// Convert OKLCH (lightness, chroma, hue_degrees) to sRGB (r, g, b).
pub fn oklch_to_rgb(l: f64, c: f64, h_deg: f64) -> (u8, u8, u8) {
    let h = h_deg.to_radians();
    let a = c * h.cos();
    let b = c * h.sin();

    // OKLab → LMS (cube root space)
    let l_ = l + 0.3963377774 * a + 0.2158037573 * b;
    let m_ = l - 0.1055613458 * a - 0.0638541728 * b;
    let s_ = l - 0.0894841775 * a - 1.2914855480 * b;

    // Undo cube root
    let l3 = l_ * l_ * l_;
    let m3 = m_ * m_ * m_;
    let s3 = s_ * s_ * s_;

    // LMS → linear sRGB
    let rl = 4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3;
    let gl = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3;
    let bl = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3;

    // Linear sRGB → sRGB gamma
    fn gamma(x: f64) -> f64 {
        let x = x.clamp(0.0, 1.0);
        if x <= 0.0031308 {
            12.92 * x
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        }
    }

    (
        (gamma(rl) * 255.0).round() as u8,
        (gamma(gl) * 255.0).round() as u8,
        (gamma(bl) * 255.0).round() as u8,
    )
}

/// Convert OKLCH (lightness, chroma, hue_degrees) to sRGB hex string.
pub fn oklch_to_hex(l: f64, c: f64, h_deg: f64) -> String {
    let (r, g, b) = oklch_to_rgb(l, c, h_deg);
    format!("#{:02x}{:02x}{:02x}", r, g, b)
}

/// Generate a vehicle color for index `i` using the golden angle for maximum hue separation
/// between consecutive indices.
pub fn vehicle_color_hex(i: usize, lightness: f64, chroma: f64) -> String {
    // Golden angle ≈ 137.508° — ensures consecutive indices have maximally different hues
    const GOLDEN_ANGLE: f64 = 137.50776405003785;
    let hue = (i as f64 * GOLDEN_ANGLE) % 360.0;
    oklch_to_hex(lightness, chroma, hue)
}

/// Parse a hex color string like "#ff6464" or "#f64" into (r, g, b).
/// Supports both 6-digit and 3-digit hex (e.g. `#abc` → `#aabbcc`).
/// Falls back to (128, 128, 128) on parse failure.
pub fn parse_hex_color(hex: &str) -> (u8, u8, u8) {
    let hex = hex.trim_start_matches('#');
    if hex.len() == 6 {
        if let (Ok(r), Ok(g), Ok(b)) = (
            u8::from_str_radix(&hex[0..2], 16),
            u8::from_str_radix(&hex[2..4], 16),
            u8::from_str_radix(&hex[4..6], 16),
        ) {
            return (r, g, b);
        }
    } else if hex.len() == 3 {
        if let (Ok(r), Ok(g), Ok(b)) = (
            u8::from_str_radix(&hex[0..1], 16),
            u8::from_str_radix(&hex[1..2], 16),
            u8::from_str_radix(&hex[2..3], 16),
        ) {
            return (r * 17, g * 17, b * 17);
        }
    }
    (128, 128, 128)
}

/// Parse a color string in any supported format into (r, g, b).
///
/// Supported formats:
/// - `#rrggbb`, `rrggbb`, `#rgb`, or `rgb` (hex)
/// - `rgb(r, g, b)` with 0-255 integer values
/// - `oklch(L, C, H)` with L in 0.0-1.0, C in 0.0-~0.4, H in degrees
///
/// Falls back to (128, 128, 128) on parse failure.
pub fn parse_color(s: &str) -> (u8, u8, u8) {
    let s = s.trim();

    // oklch(L, C, H)
    if let Some(inner) = s.strip_prefix("oklch(").and_then(|s| s.strip_suffix(')')) {
        let parts: Vec<&str> = inner.split(',').collect();
        if parts.len() == 3 {
            if let (Ok(l), Ok(c), Ok(h)) = (
                parts[0].trim().parse::<f64>(),
                parts[1].trim().parse::<f64>(),
                parts[2].trim().parse::<f64>(),
            ) {
                return oklch_to_rgb(l, c, h);
            }
        }
        return (128, 128, 128);
    }

    // rgb(r, g, b)
    if let Some(inner) = s.strip_prefix("rgb(").and_then(|s| s.strip_suffix(')')) {
        let parts: Vec<&str> = inner.split(',').collect();
        if parts.len() == 3 {
            if let (Ok(r), Ok(g), Ok(b)) = (
                parts[0].trim().parse::<u8>(),
                parts[1].trim().parse::<u8>(),
                parts[2].trim().parse::<u8>(),
            ) {
                return (r, g, b);
            }
        }
        return (128, 128, 128);
    }

    // Hex (#rrggbb or rrggbb)
    parse_hex_color(s)
}

/// Convert a color string (any supported format) to a ratatui Color.
pub fn color_to_ratatui(s: &str) -> ratatui::prelude::Color {
    let (r, g, b) = parse_color(s);
    ratatui::prelude::Color::Rgb(r, g, b)
}

/// Convert a color string (any supported format) to a hex string.
pub fn color_to_hex(s: &str) -> String {
    let (r, g, b) = parse_color(s);
    format!("#{:02x}{:02x}{:02x}", r, g, b)
}

/// Scale a color string by a multiplier (0.0–1.0) and return a ratatui Color.
/// Each RGB channel is multiplied independently and clamped to 0–255.
pub fn scale_color(s: &str, multiplier: f64) -> ratatui::prelude::Color {
    let (r, g, b) = parse_color(s);
    let r = ((r as f64) * multiplier).round().clamp(0.0, 255.0) as u8;
    let g = ((g as f64) * multiplier).round().clamp(0.0, 255.0) as u8;
    let b = ((b as f64) * multiplier).round().clamp(0.0, 255.0) as u8;
    ratatui::prelude::Color::Rgb(r, g, b)
}

/// Speed color ramp: green → yellow → red.
///
/// `fraction` is 0.0 (slowest/green) to 1.0 (fastest/red).
/// Same ramp used by the velocity bar widget and NPC speed labels.
pub fn speed_color_rgb(fraction: f64) -> (u8, u8, u8) {
    let fraction = fraction.clamp(0.0, 1.0);
    if fraction < 0.5 {
        // Green (80,200,80) → Yellow (240,240,50)
        let t = fraction * 2.0;
        (
            (80.0 + t * 160.0) as u8,
            (200.0 + t * 40.0) as u8,
            (80.0 - t * 30.0) as u8,
        )
    } else {
        // Yellow (240,240,50) → Red (220,50,50)
        let t = (fraction - 0.5) * 2.0;
        (
            (240.0 - t * 20.0) as u8,
            (240.0 - t * 190.0) as u8,
            50,
        )
    }
}

/// Speed color as hex string for SVG use.
pub fn speed_color_hex(speed: f64, speed_min: f64, speed_max: f64) -> String {
    let range = speed_max - speed_min;
    let fraction = if range > 0.0 {
        ((speed - speed_min) / range).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (r, g, b) = speed_color_rgb(fraction);
    format!("#{:02x}{:02x}{:02x}", r, g, b)
}

/// TTC (time-to-collision) color as hex string for SVG use.
///
/// Uses the same green→yellow→red ramp as speed, but inverted:
/// - 0 s → red (imminent collision)
/// - 5 s → yellow (moderate danger)
/// - 10 s+ → green (distant)
///
/// Below 0 s is clamped to red, above 10 s is clamped to green.
pub fn ttc_color_hex(ttc_seconds: f64) -> String {
    let fraction = 1.0 - (ttc_seconds / 10.0).clamp(0.0, 1.0);
    let (r, g, b) = speed_color_rgb(fraction);
    format!("#{:02x}{:02x}{:02x}", r, g, b)
}

/// Legacy alias — use `color_to_ratatui` for new code.
pub fn hex_to_ratatui(hex: &str) -> ratatui::prelude::Color {
    color_to_ratatui(hex)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hex() {
        assert_eq!(parse_color("#ff6464"), (255, 100, 100));
        assert_eq!(parse_color("ff6464"), (255, 100, 100));
        assert_eq!(parse_color("#000000"), (0, 0, 0));
        // 3-digit hex: each digit is doubled (e.g. #abc → #aabbcc)
        assert_eq!(parse_color("#444"), (0x44, 0x44, 0x44));
        assert_eq!(parse_color("#fff"), (255, 255, 255));
        assert_eq!(parse_color("#000"), (0, 0, 0));
        assert_eq!(parse_color("abc"), (0xaa, 0xbb, 0xcc));
    }

    #[test]
    fn test_parse_rgb() {
        assert_eq!(parse_color("rgb(255, 100, 100)"), (255, 100, 100));
        assert_eq!(parse_color("rgb(0,0,0)"), (0, 0, 0));
    }

    #[test]
    fn test_parse_oklch() {
        // White-ish: L=1.0, C=0, H=0
        let (r, g, b) = parse_color("oklch(1.0, 0.0, 0)");
        assert!(r > 250 && g > 250 && b > 250);
        // Black: L=0, C=0, H=0
        assert_eq!(parse_color("oklch(0.0, 0.0, 0)"), (0, 0, 0));
    }

    #[test]
    fn test_parse_invalid_fallback() {
        assert_eq!(parse_color("garbage"), (128, 128, 128));
        assert_eq!(parse_color("rgb(999, 0, 0)"), (128, 128, 128));
        assert_eq!(parse_color("oklch(bad)"), (128, 128, 128));
    }

    #[test]
    fn test_color_to_hex() {
        assert_eq!(color_to_hex("#ff6464"), "#ff6464");
        assert_eq!(color_to_hex("rgb(255, 100, 100)"), "#ff6464");
    }

    #[test]
    fn test_oklch_roundtrip() {
        // Vehicle color should produce valid hex
        let hex = vehicle_color_hex(0, 0.75, 0.15);
        assert!(hex.starts_with('#'));
        assert_eq!(hex.len(), 7);
    }
}
