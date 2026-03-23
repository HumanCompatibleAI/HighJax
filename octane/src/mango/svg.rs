//! SVG rasterization support using resvg.

use image::{DynamicImage, RgbaImage};
use std::sync::OnceLock;

/// Lazily-initialized usvg options with system fonts loaded.
/// Loading fonts is expensive (~50ms), so we do it once and reuse.
fn svg_options() -> &'static resvg::usvg::Options<'static> {
    static OPTIONS: OnceLock<resvg::usvg::Options<'static>> = OnceLock::new();
    OPTIONS.get_or_init(|| {
        let mut opt = resvg::usvg::Options::default();
        opt.fontdb_mut().load_system_fonts();
        opt.fontdb_mut().set_monospace_family("DejaVu Sans Mono");
        opt.fontdb_mut().set_sans_serif_family("DejaVu Sans");
        opt
    })
}

/// Check if data looks like SVG content by examining magic bytes.
#[allow(dead_code)]
pub fn is_svg_data(data: &[u8]) -> bool {
    let data = skip_whitespace(data);
    data.starts_with(b"<?xml") || data.starts_with(b"<svg") || data.starts_with(b"<SVG")
}

#[allow(dead_code)]
fn skip_whitespace(data: &[u8]) -> &[u8] {
    let mut i = 0;
    while i < data.len()
        && (data[i] == b' ' || data[i] == b'\t' || data[i] == b'\n' || data[i] == b'\r')
    {
        i += 1;
    }
    &data[i..]
}

/// Rasterize SVG data to a DynamicImage.
///
/// If both width and height are None, uses the SVG's natural size.
/// If one is specified, the other is computed to preserve aspect ratio.
/// If both are specified, the SVG is scaled to fit within the bounds.
#[allow(dead_code)]
pub fn load_svg(
    data: &[u8],
    target_width: Option<u32>,
    target_height: Option<u32>,
) -> Result<DynamicImage, SvgError> {
    let tree = resvg::usvg::Tree::from_data(data, svg_options())
        .map_err(|e| SvgError::Parse(e.to_string()))?;

    let svg_size = tree.size();
    let svg_width = svg_size.width();
    let svg_height = svg_size.height();

    let (render_width, render_height) = match (target_width, target_height) {
        (None, None) => (svg_width as u32, svg_height as u32),
        (Some(w), None) => {
            let scale = w as f32 / svg_width;
            (w, (svg_height * scale) as u32)
        }
        (None, Some(h)) => {
            let scale = h as f32 / svg_height;
            ((svg_width * scale) as u32, h)
        }
        (Some(w), Some(h)) => {
            let scale_w = w as f32 / svg_width;
            let scale_h = h as f32 / svg_height;
            let scale = scale_w.min(scale_h);
            ((svg_width * scale) as u32, (svg_height * scale) as u32)
        }
    };

    let render_width = render_width.max(1);
    let render_height = render_height.max(1);

    let mut pixmap = resvg::tiny_skia::Pixmap::new(render_width, render_height)
        .ok_or_else(|| SvgError::Render("Failed to create pixmap".to_string()))?;

    let scale_x = render_width as f32 / svg_width;
    let scale_y = render_height as f32 / svg_height;
    let transform = resvg::tiny_skia::Transform::from_scale(scale_x, scale_y);

    resvg::render(&tree, transform, &mut pixmap.as_mut());

    let data = pixmap.data();
    let img = RgbaImage::from_raw(render_width, render_height, data.to_vec())
        .ok_or_else(|| SvgError::Render("Failed to create image from pixmap".to_string()))?;

    Ok(DynamicImage::ImageRgba8(img))
}

/// Rasterize SVG data to exact dimensions (stretching, not preserving aspect ratio).
pub fn load_svg_exact(data: &[u8], width: u32, height: u32) -> Result<DynamicImage, SvgError> {
    let options = resvg::usvg::Options::default();
    let tree = resvg::usvg::Tree::from_data(data, &options)
        .map_err(|e| SvgError::Parse(e.to_string()))?;

    let svg_size = tree.size();
    let svg_width = svg_size.width();
    let svg_height = svg_size.height();

    let render_width = width.max(1);
    let render_height = height.max(1);

    let mut pixmap = resvg::tiny_skia::Pixmap::new(render_width, render_height)
        .ok_or_else(|| SvgError::Render("Failed to create pixmap".to_string()))?;

    let scale_x = render_width as f32 / svg_width;
    let scale_y = render_height as f32 / svg_height;
    let transform = resvg::tiny_skia::Transform::from_scale(scale_x, scale_y);

    resvg::render(&tree, transform, &mut pixmap.as_mut());

    let data = pixmap.data();
    let img = RgbaImage::from_raw(render_width, render_height, data.to_vec())
        .ok_or_else(|| SvgError::Render("Failed to create image from pixmap".to_string()))?;

    Ok(DynamicImage::ImageRgba8(img))
}

/// Errors that can occur when loading/rendering SVG.
#[derive(Debug)]
pub enum SvgError {
    /// Error parsing the SVG
    Parse(String),
    /// Error rendering the SVG
    Render(String),
}

impl std::fmt::Display for SvgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SvgError::Parse(e) => write!(f, "SVG parse error: {}", e),
            SvgError::Render(e) => write!(f, "SVG render error: {}", e),
        }
    }
}

impl std::error::Error for SvgError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_svg_data_xml_declaration() {
        assert!(is_svg_data(b"<?xml version=\"1.0\"?><svg></svg>"));
    }

    #[test]
    fn test_is_svg_data_svg_tag() {
        assert!(is_svg_data(
            b"<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>"
        ));
    }

    #[test]
    fn test_is_svg_data_with_whitespace() {
        assert!(is_svg_data(b"  \n\t<?xml version=\"1.0\"?>"));
        assert!(is_svg_data(b"\n<svg></svg>"));
    }

    #[test]
    fn test_is_svg_data_not_svg() {
        assert!(!is_svg_data(b"\x89PNG\r\n\x1a\n"));
        assert!(!is_svg_data(b"GIF89a"));
        assert!(!is_svg_data(b"hello world"));
    }

    #[test]
    fn test_load_simple_svg() {
        let svg = br#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
            <rect x="0" y="0" width="100" height="100" fill="red"/>
        </svg>"#;

        let img = load_svg(svg, None, None).expect("Failed to load SVG");
        assert_eq!(img.width(), 100);
        assert_eq!(img.height(), 100);
    }

    #[test]
    fn test_load_svg_with_target_width() {
        let svg = br#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50">
            <rect x="0" y="0" width="100" height="50" fill="blue"/>
        </svg>"#;

        let img = load_svg(svg, Some(200), None).expect("Failed to load SVG");
        assert_eq!(img.width(), 200);
        assert_eq!(img.height(), 100);
    }

    #[test]
    fn test_load_svg_exact() {
        let svg = br#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
            <rect x="0" y="0" width="100" height="100" fill="green"/>
        </svg>"#;

        let img = load_svg_exact(svg, 50, 200).expect("Failed to load SVG");
        assert_eq!(img.width(), 50);
        assert_eq!(img.height(), 200);
    }
}
