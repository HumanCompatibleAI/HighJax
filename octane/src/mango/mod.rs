//! Mango: SVG to ANSI terminal rendering using sextant/octant characters.
//!
//! Renders images using sextant/quadrant/octant Unicode characters with MSE-based
//! template matching. Eliminates subprocess overhead by embedding the renderer.

pub mod cli;
mod render;
mod svg;
mod templates;

#[allow(unused_imports)]
pub use cli::{get_terminal_size, load_file, render_file, render_image_to_ansi, run_benchmark, show_chars};
pub use render::{render_svg_to_ansi, MangoConfig};
