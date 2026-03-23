//! Mango rendering: SVG → ANSI terminal output.

use super::svg::{load_svg_exact, SvgError};
use super::templates::{
    MangoMode, MangoResult,
    mango_select_quadrants_only, mango_select_sextants, mango_select_octants, mango_select_all,
};
use image::GenericImageView;
use rayon::prelude::*;

/// Mango grid width: always 2 pixel columns per character.
const PISTON_WIDTH: usize = 2;

/// Configuration for mango rendering.
#[derive(Debug, Clone)]
pub struct MangoConfig {
    /// Output width in character columns.
    pub n_cols: u32,
    /// Output height in character rows.
    pub n_rows: u32,
    /// Whether to use sextant characters (2×3 grid).
    pub use_sextants: bool,
    /// Whether to use octant characters (2×4 grid).
    pub use_octants: bool,
}

impl Default for MangoConfig {
    fn default() -> Self {
        Self {
            n_cols: 80,
            n_rows: 24,
            use_sextants: false,
            use_octants: true,
        }
    }
}

impl MangoConfig {
    pub fn mode(&self) -> MangoMode {
        MangoMode::from_flags(self.use_sextants, self.use_octants)
    }
}

// Macro to generate a rendering branch for a given mode.
// Each branch uses its own PISTON_HEIGHT, pixel array size, and select function.
macro_rules! render_branch {
    ($piston_h:expr, $n_pixels:expr, $select_fn:ident,
     $img_buffer:expr, $corn_rows:expr, $corn_cols:expr, $width:expr, $height:expr) => {{
        (0..($corn_rows * $corn_cols))
            .into_par_iter()
            .map(|corn_idx| {
                let corn_x = corn_idx % $corn_cols;
                let corn_y = corn_idx / $corn_cols;
                let mut pixels = [(0u8, 0u8, 0u8); $n_pixels];
                for dy in 0..$piston_h {
                    for dx in 0..PISTON_WIDTH {
                        let px = corn_x * PISTON_WIDTH as u32 + dx as u32;
                        let py = corn_y * $piston_h as u32 + dy as u32;
                        let idx = dy * PISTON_WIDTH + dx;
                        if px < $width && py < $height {
                            let pixel = $img_buffer.get_pixel(px, py);
                            pixels[idx] = (pixel[0], pixel[1], pixel[2]);
                        }
                    }
                }
                $select_fn(&pixels)
            })
            .collect::<Vec<MangoResult>>()
    }};
}

/// A text overlay to render as actual terminal text on top of the mango raster.
#[derive(Debug, Clone)]
pub struct TextOverlay {
    /// Column in the terminal grid (0-based).
    pub col: u32,
    /// Row in the terminal grid (0-based).
    pub row: u32,
    /// Text string to render.
    pub text: String,
    /// Foreground color (r, g, b).
    pub fg: (u8, u8, u8),
    /// Background color (r, g, b).
    pub bg: (u8, u8, u8),
}

/// Parse hex color string like "#cccccc" to (r, g, b).
fn parse_hex_color(s: &str) -> Option<(u8, u8, u8)> {
    let s = s.strip_prefix('#')?;
    if s.len() == 6 {
        let r = u8::from_str_radix(&s[0..2], 16).ok()?;
        let g = u8::from_str_radix(&s[2..4], 16).ok()?;
        let b = u8::from_str_radix(&s[4..6], 16).ok()?;
        Some((r, g, b))
    } else if s.len() == 3 {
        let r = u8::from_str_radix(&s[0..1], 16).ok()? * 17;
        let g = u8::from_str_radix(&s[1..2], 16).ok()? * 17;
        let b = u8::from_str_radix(&s[2..3], 16).ok()? * 17;
        Some((r, g, b))
    } else {
        None
    }
}

/// Extract text overlays from SVG content by parsing `<text>` elements.
///
/// Maps SVG viewBox coordinates to terminal cell positions.
/// Each `<text>` element with a `data-mango-fg` attribute is treated as a text overlay.
/// The `data-mango-bg` attribute provides the background color.
pub fn extract_text_overlays(svg_content: &str, n_cols: u32, n_rows: u32) -> Vec<TextOverlay> {
    // Parse viewBox to get SVG coordinate space
    let (vb_w, vb_h) = parse_viewbox(svg_content).unwrap_or((1.0, 1.0));

    let mut overlays = Vec::new();

    // Simple regex-free parsing of <text> elements with data-mango-fg
    for text_start in svg_content.match_indices("<text ") {
        let rest = &svg_content[text_start.0..];
        // Only process text elements that have data-mango-fg
        let Some(tag_end) = rest.find('>') else { continue };
        let tag = &rest[..tag_end];
        if !tag.contains("data-mango-fg") {
            continue;
        }

        let fg = extract_attr(tag, "data-mango-fg")
            .and_then(|s| parse_hex_color(&s))
            .unwrap_or((204, 204, 204));
        let bg = extract_attr(tag, "data-mango-bg")
            .and_then(|s| parse_hex_color(&s))
            .unwrap_or((0, 0, 0));

        let x: f64 = extract_attr(tag, "x").and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let y: f64 = extract_attr(tag, "y").and_then(|s| s.parse().ok()).unwrap_or(0.0);

        // Get text content between > and </text>
        let content_start = text_start.0 + tag_end + 1;
        let Some(close_pos) = svg_content[content_start..].find("</text>") else { continue };
        let content = &svg_content[content_start..content_start + close_pos];

        // Map SVG coords to terminal cells
        // text-anchor="middle" means x is the center
        let frac_x = x / vb_w;
        let frac_y = y / vb_h;
        let center_col = (frac_x * n_cols as f64).round() as i32;
        let row = (frac_y * n_rows as f64).round() as i32;

        let char_len = content.chars().count() as i32;
        let start_col = center_col - char_len / 2;

        if row >= 0 && row < n_rows as i32 {
            let clip_chars = if start_col < 0 { (-start_col) as usize } else { 0 };
            if clip_chars < char_len as usize {
                let byte_offset = content.char_indices()
                    .nth(clip_chars)
                    .map(|(i, _)| i)
                    .unwrap_or(content.len());
                overlays.push(TextOverlay {
                    col: start_col.max(0) as u32,
                    row: row as u32,
                    text: content[byte_offset..].to_string(),
                    fg,
                    bg,
                });
            }
        }
    }

    overlays
}

fn parse_viewbox(svg: &str) -> Option<(f64, f64)> {
    let vb_start = svg.find("viewBox=\"")? + "viewBox=\"".len();
    let vb_end = svg[vb_start..].find('"')? + vb_start;
    let parts: Vec<f64> = svg[vb_start..vb_end]
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    if parts.len() == 4 {
        Some((parts[2], parts[3]))
    } else {
        None
    }
}

/// Strip all `<text ...>...</text>` elements from SVG content.
///
/// Used before rasterizing for the terminal: text is rendered as terminal character
/// overlays instead, so it must not also appear in the pixel raster.
fn strip_text_elements(svg: &str) -> String {
    let mut result = String::with_capacity(svg.len());
    let mut pos = 0;
    while let Some(start) = svg[pos..].find("<text ").or_else(|| svg[pos..].find("<text>")) {
        let abs_start = pos + start;
        result.push_str(&svg[pos..abs_start]);
        // Find matching </text>
        if let Some(end_offset) = svg[abs_start..].find("</text>") {
            pos = abs_start + end_offset + "</text>".len();
        } else {
            // Malformed — skip just the opening tag
            pos = abs_start + 1;
            result.push('<');
        }
    }
    result.push_str(&svg[pos..]);
    result
}

fn extract_attr(tag: &str, name: &str) -> Option<String> {
    let needle = format!("{}=\"", name);
    let start = tag.find(&needle)? + needle.len();
    let end = tag[start..].find('"')? + start;
    Some(tag[start..end].to_string())
}

/// Render SVG content to ANSI-escaped terminal output using mango.
pub fn render_svg_to_ansi(svg_content: &str, config: &MangoConfig) -> Result<String, RenderError> {
    let mode = config.mode();
    let piston_height = mode.piston_height();

    let pixel_width = config.n_cols * PISTON_WIDTH as u32;
    let pixel_height = config.n_rows * piston_height as u32;

    // Strip <text> elements before rasterizing: the mango overlay system renders
    // text as terminal characters separately, so we don't want resvg to also draw
    // them into the pixel buffer (which would cause double rendering).
    let raster_svg = strip_text_elements(svg_content);
    let img = load_svg_exact(raster_svg.as_bytes(), pixel_width, pixel_height)
        .map_err(RenderError::Svg)?;

    let (width, height) = img.dimensions();
    let img_buffer = img.to_rgba8();

    let corn_cols = width.div_ceil(PISTON_WIDTH as u32);
    let corn_rows = height.div_ceil(piston_height as u32);

    assert_eq!(
        corn_cols, config.n_cols,
        "Mango: corn_cols {} != requested n_cols {}, pixel_width={}, img_width={}",
        corn_cols, config.n_cols, pixel_width, width
    );
    assert_eq!(
        corn_rows, config.n_rows,
        "Mango: corn_rows {} != requested n_rows {}, pixel_height={}, img_height={}",
        corn_rows, config.n_rows, pixel_height, height
    );

    // Dispatch to mode-specific rendering — each branch is monomorphized
    let results: Vec<MangoResult> = match mode {
        MangoMode::QuadrantsOnly => {
            render_branch!(4, 8, mango_select_quadrants_only,
                img_buffer, corn_rows, corn_cols, width, height)
        }
        MangoMode::Sextants => {
            render_branch!(6, 12, mango_select_sextants,
                img_buffer, corn_rows, corn_cols, width, height)
        }
        MangoMode::Octants => {
            render_branch!(8, 16, mango_select_octants,
                img_buffer, corn_rows, corn_cols, width, height)
        }
        MangoMode::SextantsOctants => {
            render_branch!(12, 24, mango_select_all,
                img_buffer, corn_rows, corn_cols, width, height)
        }
    };

    // Extract text overlays from SVG before building output
    let overlays = extract_text_overlays(svg_content, config.n_cols, config.n_rows);

    // Build a lookup: (row, col) -> (char, fg, bg) for overlay cells
    let mut overlay_map: std::collections::HashMap<(u32, u32), (char, (u8, u8, u8), (u8, u8, u8))> =
        std::collections::HashMap::new();
    for ov in &overlays {
        for (i, ch) in ov.text.chars().enumerate() {
            let col = ov.col + i as u32;
            if col < config.n_cols {
                overlay_map.insert((ov.row, col), (ch, ov.fg, ov.bg));
            }
        }
    }

    // Build ANSI output string
    let mut output = String::with_capacity((corn_cols * corn_rows * 30) as usize);

    for corn_y in 0..corn_rows {
        for corn_x in 0..corn_cols {
            if let Some(&(ch, fg, bg)) = overlay_map.get(&(corn_y, corn_x)) {
                output.push_str(&format!(
                    "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m{}",
                    fg.0, fg.1, fg.2, bg.0, bg.1, bg.2, ch
                ));
            } else {
                let corn_idx = (corn_y * corn_cols + corn_x) as usize;
                let result = &results[corn_idx];
                output.push_str(&format!(
                    "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m{}",
                    result.fg.0, result.fg.1, result.fg.2,
                    result.bg.0, result.bg.1, result.bg.2,
                    result.ch
                ));
            }
        }

        if corn_y < corn_rows - 1 {
            output.push_str("\x1b[0m\n");
        }
    }

    output.push_str("\x1b[0m");

    let newline_count = output.matches('\n').count();
    assert_eq!(
        newline_count,
        (config.n_rows - 1) as usize,
        "Mango: output has {} newlines but expected {} (n_rows={})",
        newline_count,
        config.n_rows - 1,
        config.n_rows
    );

    Ok(output)
}

/// Errors that can occur during rendering.
#[derive(Debug)]
pub enum RenderError {
    Svg(SvgError),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderError::Svg(e) => write!(f, "SVG error: {}", e),
        }
    }
}

impl std::error::Error for RenderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RenderError::Svg(e) => Some(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_simple_svg() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
            <rect x="0" y="0" width="100" height="100" fill="red"/>
        </svg>"#;

        let config = MangoConfig {
            n_cols: 10,
            n_rows: 5,
            use_sextants: true,
            use_octants: true,
        };

        let result = render_svg_to_ansi(svg, &config).expect("Failed to render");
        assert!(result.contains("\x1b["));
        assert!(result.ends_with("\x1b[0m"));
    }

    #[test]
    fn test_render_produces_expected_dimensions() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
            <rect x="0" y="0" width="100" height="100" fill="blue"/>
        </svg>"#;

        let config = MangoConfig {
            n_cols: 20,
            n_rows: 10,
            use_sextants: true,
            use_octants: false,
        };

        let result = render_svg_to_ansi(svg, &config).expect("Failed to render");
        let newlines = result.matches('\n').count();
        assert_eq!(newlines, 9);
    }

    #[test]
    fn test_render_all_four_modes() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
            <rect x="0" y="0" width="100" height="100" fill="green"/>
        </svg>"#;

        for (s, o) in [(false, false), (true, false), (false, true), (true, true)] {
            let config = MangoConfig {
                n_cols: 10,
                n_rows: 5,
                use_sextants: s,
                use_octants: o,
            };
            let result = render_svg_to_ansi(svg, &config);
            assert!(result.is_ok(), "Failed for sextants={}, octants={}", s, o);
        }
    }

    #[test]
    fn test_render_gradient() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
            <defs>
                <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:red"/>
                    <stop offset="100%" style="stop-color:blue"/>
                </linearGradient>
            </defs>
            <rect x="0" y="0" width="100" height="100" fill="url(#grad)"/>
        </svg>"#;

        let config = MangoConfig {
            n_cols: 20,
            n_rows: 10,
            use_sextants: true,
            use_octants: true,
        };

        let result = render_svg_to_ansi(svg, &config).expect("Failed to render");
        assert!(result.contains("\x1b["));
    }
}
