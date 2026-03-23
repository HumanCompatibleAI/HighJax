//! CLI functionality for mango renderer.

use super::render::MangoConfig;
use super::svg::{load_svg, load_svg_exact};
use super::templates::{
    MangoMode, MangoResult,
    mango_select_quadrants_only, mango_select_sextants, mango_select_octants, mango_select_all,
    octant_char,
};
use image::{DynamicImage, GenericImageView};
use std::path::Path;
use std::time::{Duration, Instant};

/// Mango grid width: always 2 pixel columns per character.
const PISTON_WIDTH: usize = 2;

/// Result of a benchmark run.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BenchmarkResult {
    pub name: String,
    pub cols: u32,
    pub rows: u32,
    pub total_time: Duration,
    pub load_time: Duration,
    pub render_time: Duration,
    pub chars_per_sec: f64,
    pub fps: f64,
}

/// Macro to generate a CLI rendering branch for a given mode.
macro_rules! cli_render_branch {
    ($piston_h:expr, $n_pixels:expr, $select_fn:ident,
     $img_buffer:expr, $corn_rows:expr, $corn_cols:expr, $width:expr, $height:expr) => {{
        let mut results = Vec::with_capacity(($corn_cols * $corn_rows) as usize);
        for corn_y in 0..$corn_rows {
            for corn_x in 0..$corn_cols {
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
                results.push($select_fn(&pixels));
            }
        }
        results
    }};
}

/// Render an image file to ANSI output.
pub fn render_image_to_ansi(
    img: &DynamicImage,
    config: &MangoConfig,
) -> String {
    let mode = config.mode();
    let piston_height = mode.piston_height();

    let pixel_width = config.n_cols * PISTON_WIDTH as u32;
    let pixel_height = config.n_rows * piston_height as u32;

    // Resize image to exact pixel dimensions
    let resized = img.resize_exact(
        pixel_width,
        pixel_height,
        image::imageops::FilterType::Lanczos3,
    );
    let img_buffer = resized.to_rgba8();
    let (width, height) = resized.dimensions();

    let corn_cols = config.n_cols;
    let corn_rows = config.n_rows;

    // Dispatch to mode-specific rendering
    let results: Vec<MangoResult> = match mode {
        MangoMode::QuadrantsOnly => {
            cli_render_branch!(4, 8, mango_select_quadrants_only,
                img_buffer, corn_rows, corn_cols, width, height)
        }
        MangoMode::Sextants => {
            cli_render_branch!(6, 12, mango_select_sextants,
                img_buffer, corn_rows, corn_cols, width, height)
        }
        MangoMode::Octants => {
            cli_render_branch!(8, 16, mango_select_octants,
                img_buffer, corn_rows, corn_cols, width, height)
        }
        MangoMode::SextantsOctants => {
            cli_render_branch!(12, 24, mango_select_all,
                img_buffer, corn_rows, corn_cols, width, height)
        }
    };

    // Build ANSI output string
    let mut output = String::with_capacity((corn_cols * corn_rows * 30) as usize);

    for corn_y in 0..corn_rows {
        for corn_x in 0..corn_cols {
            let corn_idx = (corn_y * corn_cols + corn_x) as usize;
            let result = &results[corn_idx];

            output.push_str(&format!(
                "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m{}",
                result.fg.0, result.fg.1, result.fg.2,
                result.bg.0, result.bg.1, result.bg.2,
                result.ch
            ));
        }

        if corn_y < corn_rows - 1 {
            output.push_str("\x1b[0m\n");
        }
    }

    output.push_str("\x1b[0m");
    output
}

/// Load an image or SVG file.
pub fn load_file(path: &Path) -> Result<DynamicImage, String> {
    let data = std::fs::read(path).map_err(|e| format!("Failed to read file: {}", e))?;

    // Check if it's SVG
    let is_svg = {
        let trimmed = data.iter().skip_while(|&&b| b.is_ascii_whitespace()).copied().collect::<Vec<_>>();
        trimmed.starts_with(b"<?xml") || trimmed.starts_with(b"<svg") || trimmed.starts_with(b"<SVG")
    };

    if is_svg {
        load_svg(&data, None, None).map_err(|e| format!("Failed to load SVG: {}", e))
    } else {
        image::load_from_memory(&data).map_err(|e| format!("Failed to load image: {}", e))
    }
}

/// Render a file (SVG or image) to ANSI and print to stdout.
pub fn render_file(path: &Path, cols: u32, rows: u32, use_sextants: bool, use_octants: bool) -> Result<(), String> {
    let img = load_file(path)?;
    let config = MangoConfig { n_cols: cols, n_rows: rows, use_sextants, use_octants };
    let output = render_image_to_ansi(&img, &config);
    println!("{}", output);
    Ok(())
}

/// Sample SVG patterns for benchmarking.
fn benchmark_samples() -> Vec<(&'static str, &'static str)> {
    vec![
        ("solid_red", r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
            <rect x="0" y="0" width="100" height="100" fill="red"/>
        </svg>"#),
        ("gradient_h", r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
            <defs>
                <linearGradient id="g1" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="red"/>
                    <stop offset="100%" stop-color="blue"/>
                </linearGradient>
            </defs>
            <rect x="0" y="0" width="100" height="100" fill="url(#g1)"/>
        </svg>"##),
        ("gradient_v", r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
            <defs>
                <linearGradient id="g2" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stop-color="green"/>
                    <stop offset="100%" stop-color="yellow"/>
                </linearGradient>
            </defs>
            <rect x="0" y="0" width="100" height="100" fill="url(#g2)"/>
        </svg>"##),
        ("circles", r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
            <rect x="0" y="0" width="100" height="100" fill="black"/>
            <circle cx="25" cy="25" r="20" fill="red"/>
            <circle cx="75" cy="25" r="20" fill="green"/>
            <circle cx="25" cy="75" r="20" fill="blue"/>
            <circle cx="75" cy="75" r="20" fill="yellow"/>
        </svg>"#),
        ("checkerboard", r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
            <defs>
                <pattern id="check" width="20" height="20" patternUnits="userSpaceOnUse">
                    <rect width="10" height="10" fill="white"/>
                    <rect x="10" width="10" height="10" fill="black"/>
                    <rect y="10" width="10" height="10" fill="black"/>
                    <rect x="10" y="10" width="10" height="10" fill="white"/>
                </pattern>
            </defs>
            <rect x="0" y="0" width="100" height="100" fill="url(#check)"/>
        </svg>"##),
        ("complex_scene", r##"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">
            <rect x="0" y="0" width="200" height="100" fill="#1a1a2e"/>
            <rect x="0" y="60" width="200" height="40" fill="#333"/>
            <line x1="0" y1="70" x2="200" y2="70" stroke="white" stroke-width="2" stroke-dasharray="10,10"/>
            <polygon points="50,50 55,65 45,65" fill="cyan"/>
            <rect x="100" y="45" width="15" height="20" fill="red"/>
            <rect x="150" y="40" width="20" height="25" fill="yellow"/>
        </svg>"##),
    ]
}

/// Run comprehensive benchmark.
pub fn run_benchmark(iterations: u32, use_sextants: bool, use_octants: bool) -> Vec<BenchmarkResult> {
    let samples = benchmark_samples();
    let dimensions = [
        (40, 12),   // Small
        (80, 24),   // Standard
        (120, 40),  // Large
        (160, 50),  // XL
    ];

    let mode = MangoMode::from_flags(use_sextants, use_octants);
    let piston_height = mode.piston_height();

    let mut results = Vec::new();

    println!("Mango Renderer Benchmark");
    println!("========================");
    println!("Mode: {:?} (piston_height={})", mode, piston_height);
    println!("Iterations per test: {}\n", iterations);

    for (cols, rows) in dimensions {
        println!("--- {}x{} ({} chars) ---", cols, rows, cols * rows);

        for (name, svg) in &samples {
            let config = MangoConfig { n_cols: cols, n_rows: rows, use_sextants, use_octants };
            let pixel_width = cols * PISTON_WIDTH as u32;
            let pixel_height = rows * piston_height as u32;

            // Warm up
            let _ = load_svg_exact(svg.as_bytes(), pixel_width, pixel_height);

            let mut total_load = Duration::ZERO;
            let mut total_render = Duration::ZERO;

            for _ in 0..iterations {
                // Time SVG loading/rasterization
                let load_start = Instant::now();
                let img = load_svg_exact(svg.as_bytes(), pixel_width, pixel_height).unwrap();
                total_load += load_start.elapsed();

                // Time mango rendering
                let render_start = Instant::now();
                let _ = render_image_to_ansi(&img, &config);
                total_render += render_start.elapsed();
            }

            let avg_load = total_load / iterations;
            let avg_render = total_render / iterations;
            let avg_total = avg_load + avg_render;

            let chars = (cols * rows) as f64;
            let chars_per_sec = chars / avg_render.as_secs_f64();
            let fps = 1.0 / avg_total.as_secs_f64();

            println!(
                "  {:15} load={:>8.2?}  render={:>8.2?}  total={:>8.2?}  {:.0} chars/s  {:.1} fps",
                name, avg_load, avg_render, avg_total, chars_per_sec, fps
            );

            results.push(BenchmarkResult {
                name: format!("{}@{}x{}", name, cols, rows),
                cols,
                rows,
                total_time: avg_total,
                load_time: avg_load,
                render_time: avg_render,
                chars_per_sec,
                fps,
            });
        }
        println!();
    }

    // Summary
    println!("=== Summary ===");
    let avg_fps: f64 = results.iter().map(|r| r.fps).sum::<f64>() / results.len() as f64;
    let min_fps = results.iter().map(|r| r.fps).fold(f64::INFINITY, f64::min);
    let max_fps = results.iter().map(|r| r.fps).fold(f64::NEG_INFINITY, f64::max);

    println!("FPS: avg={:.1}, min={:.1}, max={:.1}", avg_fps, min_fps, max_fps);
    println!("30 FPS target: {}", if min_fps >= 30.0 { "ACHIEVED" } else { "NOT achieved" });

    results
}

/// Get terminal dimensions, with fallback.
pub fn get_terminal_size() -> (u32, u32) {
    crossterm::terminal::size()
        .map(|(w, h)| (w as u32, h as u32))
        .unwrap_or((80, 24))
}

/// Display the mango character sets.
pub fn show_chars() {
    println!("Mango Character Sets");
    println!("====================\n");

    println!("Quadrants (16 chars, 2x2 grid):");
    println!("Pattern bits: [upper-left, upper-right, lower-left, lower-right]\n");
    let quadrants = [
        (' ', "0000", "empty"),
        ('\u{2598}', "0001", "upper-left"),
        ('\u{259D}', "0010", "upper-right"),
        ('\u{2580}', "0011", "upper half"),
        ('\u{2596}', "0100", "lower-left"),
        ('\u{258C}', "0101", "left half"),
        ('\u{259E}', "0110", "diagonal NE"),
        ('\u{259B}', "0111", "all but LR"),
        ('\u{2597}', "1000", "lower-right"),
        ('\u{259A}', "1001", "diagonal NW"),
        ('\u{2590}', "1010", "right half"),
        ('\u{259C}', "1011", "all but LL"),
        ('\u{2584}', "1100", "lower half"),
        ('\u{2599}', "1101", "all but UR"),
        ('\u{259F}', "1110", "all but UL"),
        ('\u{2588}', "1111", "full block"),
    ];

    for (ch, pattern, desc) in quadrants {
        println!("  {}  0b{}  {}", ch, pattern, desc);
    }

    println!("\n\nSextants (64 chars, 2x3 grid):");
    println!("Pattern bits: [UL, UR, ML, MR, LL, LR] (top to bottom, left to right)\n");

    let sextants: [char; 64] = [
        ' ', '\u{1FB00}', '\u{1FB01}', '\u{1FB02}', '\u{1FB03}', '\u{1FB04}', '\u{1FB05}', '\u{1FB06}',
        '\u{1FB07}', '\u{1FB08}', '\u{1FB09}', '\u{1FB0A}', '\u{1FB0B}', '\u{1FB0C}', '\u{1FB0D}', '\u{1FB0E}',
        '\u{1FB0F}', '\u{1FB10}', '\u{1FB11}', '\u{1FB12}', '\u{1FB13}', '\u{258C}', '\u{1FB14}', '\u{1FB15}',
        '\u{1FB16}', '\u{1FB17}', '\u{1FB18}', '\u{1FB19}', '\u{1FB1A}', '\u{1FB1B}', '\u{1FB1C}', '\u{1FB1D}',
        '\u{1FB1E}', '\u{1FB1F}', '\u{1FB20}', '\u{1FB21}', '\u{1FB22}', '\u{1FB23}', '\u{1FB24}', '\u{1FB25}',
        '\u{1FB26}', '\u{1FB27}', '\u{2590}', '\u{1FB28}', '\u{1FB29}', '\u{1FB2A}', '\u{1FB2B}', '\u{1FB2C}',
        '\u{1FB2D}', '\u{1FB2E}', '\u{1FB2F}', '\u{1FB30}', '\u{1FB31}', '\u{1FB32}', '\u{1FB33}', '\u{1FB34}',
        '\u{1FB35}', '\u{1FB36}', '\u{1FB37}', '\u{1FB38}', '\u{1FB39}', '\u{1FB3A}', '\u{1FB3B}', '\u{2588}',
    ];

    for row in 0..8 {
        print!("  ");
        for col in 0..8 {
            let idx = row * 8 + col;
            print!("{} {:02} ", sextants[idx], idx);
        }
        println!();
    }

    println!("\n\nOctants (256 chars, 2x4 grid):");
    println!("Pattern bits: [UL, UR, UML, UMR, LML, LMR, LL, LR] (top to bottom, left to right)\n");

    // Show first 64 and last 64 as representative sample
    println!("  Patterns 0-63:");
    for row in 0..8 {
        print!("  ");
        for col in 0..8 {
            let idx = row * 8 + col;
            print!("{} {:03} ", octant_char(idx as u8), idx);
        }
        println!();
    }
    println!("  ...");
    println!("  Patterns 192-255:");
    for row in 0..8 {
        print!("  ");
        for col in 0..8 {
            let idx = 192 + row * 8 + col;
            print!("{} {:03} ", octant_char(idx as u8), idx);
        }
        println!();
    }

    println!("\nUnicode ranges:");
    println!("  Quadrants: U+2580-U+259F (Block Elements)");
    println!("  Sextants:  U+1FB00-U+1FB3B (Symbols for Legacy Computing)");
    println!("  Octants:   U+1CD00-U+1CDE5 (Octants, Unicode 16.0)");
}
