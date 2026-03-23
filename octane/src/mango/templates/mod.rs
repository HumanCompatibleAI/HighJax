//! Character templates for mango rendering.
//!
//! Mango uses three character sets with MSE-based template matching:
//! - Quadrants (2×2 grid, 16 patterns, U+2580-U+259F)
//! - Sextants (2×3 grid, 64 patterns, U+1FB00-U+1FB3B)
//! - Octants (2×4 grid, 256 patterns, U+1CD00-U+1CDE5 + supplements)
//!
//! Each mode uses an optimal pixel grid size for efficiency:
//! - Quadrants only: 2×4 = 8 pixels per cell
//! - Sextants (+quadrants): 2×6 = 12 pixels per cell
//! - Octants (+quadrants): 2×8 = 16 pixels per cell
//! - All three: 2×12 = 24 pixels per cell

mod character_tables;
mod masks;
#[macro_use]
mod mse;

#[allow(unused_imports)]
pub use character_tables::{quadrant_char, sextant_char, octant_char};

use character_tables::{QUADRANT_CHARS, SEXTANT_CHARS, OCTANT_CHARS};
use masks::*;
use mse::{pixel_sq_total, compute_mse, pixel_totals, final_colors};

/// Result of mango character selection.
#[derive(Debug, Clone, Copy)]
pub struct MangoResult {
    /// The selected character.
    pub ch: char,
    /// Foreground color (RGB).
    pub fg: (u8, u8, u8),
    /// Background color (RGB).
    pub bg: (u8, u8, u8),
}

/// Rendering mode determined by which character sets are enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MangoMode {
    /// Quadrants only (2×2, 16 patterns). PISTON_HEIGHT=4, 8 pixels.
    QuadrantsOnly,
    /// Sextants + quadrants (2×3 + 2×2, 80 patterns). PISTON_HEIGHT=6, 12 pixels.
    Sextants,
    /// Octants + quadrants (2×4 + 2×2, 272 patterns). PISTON_HEIGHT=8, 16 pixels.
    Octants,
    /// All three (2×3 + 2×4 + 2×2, 336 patterns). PISTON_HEIGHT=12, 24 pixels.
    SextantsOctants,
}

impl MangoMode {
    pub fn from_flags(use_sextants: bool, use_octants: bool) -> Self {
        match (use_sextants, use_octants) {
            (false, false) => MangoMode::QuadrantsOnly,
            (true, false) => MangoMode::Sextants,
            (false, true) => MangoMode::Octants,
            (true, true) => MangoMode::SextantsOctants,
        }
    }

    pub fn piston_height(self) -> usize {
        match self {
            MangoMode::QuadrantsOnly => 4,
            MangoMode::Sextants => 6,
            MangoMode::Octants => 8,
            MangoMode::SextantsOctants => 12,
        }
    }
}

// =============================================================================
// Select functions — one per mode, each maximally efficient
// =============================================================================

/// QuadrantsOnly mode: 8 pixels, 8 patterns (half of 16, complement symmetry).
pub fn mango_select_quadrants_only(pixels: &[(u8, u8, u8); 8]) -> MangoResult {
    let (total_r, total_g, total_b) = pixel_totals(pixels);
    let total_sq = pixel_sq_total(pixels);
    let mut best_mse = u32::MAX;
    let mut best_ch = ' ';
    let mut best_fg_r = 0u32;
    let mut best_fg_g = 0u32;
    let mut best_fg_b = 0u32;
    let mut best_fg_n = 0u32;
    try_patterns!(pixels, QUADRANT_CHARS, QUAD8_MASKS, QUAD8_FG, 16,
        total_r, total_g, total_b, total_sq,
        best_mse, best_ch, best_fg_r, best_fg_g, best_fg_b, best_fg_n);
    let (fg, bg) = final_colors(best_fg_r, best_fg_g, best_fg_b, best_fg_n,
        total_r, total_g, total_b, 8);
    MangoResult { ch: best_ch, fg, bg }
}

/// Sextants mode: 12 pixels, 40 patterns (half of 80, complement symmetry).
pub fn mango_select_sextants(pixels: &[(u8, u8, u8); 12]) -> MangoResult {
    let (total_r, total_g, total_b) = pixel_totals(pixels);
    let total_sq = pixel_sq_total(pixels);
    let mut best_mse = u32::MAX;
    let mut best_ch = ' ';
    let mut best_fg_r = 0u32;
    let mut best_fg_g = 0u32;
    let mut best_fg_b = 0u32;
    let mut best_fg_n = 0u32;
    try_patterns!(pixels, SEXTANT_CHARS, SEXT12_MASKS, SEXT12_FG, 64,
        total_r, total_g, total_b, total_sq,
        best_mse, best_ch, best_fg_r, best_fg_g, best_fg_b, best_fg_n);
    try_patterns!(pixels, QUADRANT_CHARS, QUAD12_MASKS, QUAD12_FG, 16,
        total_r, total_g, total_b, total_sq,
        best_mse, best_ch, best_fg_r, best_fg_g, best_fg_b, best_fg_n);
    let (fg, bg) = final_colors(best_fg_r, best_fg_g, best_fg_b, best_fg_n,
        total_r, total_g, total_b, 12);
    MangoResult { ch: best_ch, fg, bg }
}

/// Octants mode: 16 pixels, 136 patterns (half of 272, complement symmetry).
/// Hand-optimized: block sums (8 blocks of 2 pixels) + algebraic MSE.
pub fn mango_select_octants(pixels: &[(u8, u8, u8); 16]) -> MangoResult {
    // Precompute totals and sum-of-squares in one pass
    let mut total_r = 0u32;
    let mut total_g = 0u32;
    let mut total_b = 0u32;
    let mut total_sq = 0u32;
    for &(r, g, b) in pixels.iter() {
        let (r, g, b) = (r as u32, g as u32, b as u32);
        total_r += r;
        total_g += g;
        total_b += b;
        total_sq += r * r + g * g + b * b;
    }

    // Precompute octant block sums (8 blocks of 2 pixels each).
    // Bit i (crow=i/2, col=i%2) controls pixels at p0=(i/2)*4+(i%2) and p1=p0+2.
    let mut blk_r = [0u32; 8];
    let mut blk_g = [0u32; 8];
    let mut blk_b = [0u32; 8];
    for i in 0..8usize {
        let p0 = (i / 2) * 4 + i % 2;
        let p1 = p0 + 2;
        blk_r[i] = pixels[p0].0 as u32 + pixels[p1].0 as u32;
        blk_g[i] = pixels[p0].1 as u32 + pixels[p1].1 as u32;
        blk_b[i] = pixels[p0].2 as u32 + pixels[p1].2 as u32;
    }

    let mut best_mse = u32::MAX;
    let mut best_ch = ' ';
    let mut best_fg_r = 0u32;
    let mut best_fg_g = 0u32;
    let mut best_fg_b = 0u32;
    let mut best_fg_n = 0u32;

    // Octant patterns: 128 of 256 (complement symmetry)
    for pattern in 0u32..128 {
        let fg_n = OCT16_FG[pattern as usize] as u32;
        let bg_n = 16 - fg_n;

        let mut fg_r = 0u32;
        let mut fg_g = 0u32;
        let mut fg_b = 0u32;
        for bit in 0..8u32 {
            if (pattern >> bit) & 1 == 1 {
                fg_r += blk_r[bit as usize];
                fg_g += blk_g[bit as usize];
                fg_b += blk_b[bit as usize];
            }
        }

        let bg_r = total_r - fg_r;
        let bg_g = total_g - fg_g;
        let bg_b = total_b - fg_b;
        let fg_ssq = fg_r * fg_r + fg_g * fg_g + fg_b * fg_b;
        let bg_ssq = bg_r * bg_r + bg_g * bg_g + bg_b * bg_b;
        let mse = total_sq
            - if fg_n > 0 { fg_ssq / fg_n } else { 0 }
            - if bg_n > 0 { bg_ssq / bg_n } else { 0 };

        if mse < best_mse {
            best_mse = mse;
            best_ch = OCTANT_CHARS[pattern as usize];
            best_fg_r = fg_r;
            best_fg_g = fg_g;
            best_fg_b = fg_b;
            best_fg_n = fg_n;
        }
    }

    // Quadrant patterns: 8 of 16 (complement symmetry).
    // Quad blocks composed from octant blocks.
    let qblk_r = [
        blk_r[0] + blk_r[2], blk_r[1] + blk_r[3],
        blk_r[4] + blk_r[6], blk_r[5] + blk_r[7],
    ];
    let qblk_g = [
        blk_g[0] + blk_g[2], blk_g[1] + blk_g[3],
        blk_g[4] + blk_g[6], blk_g[5] + blk_g[7],
    ];
    let qblk_b = [
        blk_b[0] + blk_b[2], blk_b[1] + blk_b[3],
        blk_b[4] + blk_b[6], blk_b[5] + blk_b[7],
    ];

    for pattern in 0u32..8 {
        let fg_n = QUAD16_FG[pattern as usize] as u32;
        let bg_n = 16 - fg_n;

        let mut fg_r = 0u32;
        let mut fg_g = 0u32;
        let mut fg_b = 0u32;
        for bit in 0..4u32 {
            if (pattern >> bit) & 1 == 1 {
                fg_r += qblk_r[bit as usize];
                fg_g += qblk_g[bit as usize];
                fg_b += qblk_b[bit as usize];
            }
        }

        let bg_r = total_r - fg_r;
        let bg_g = total_g - fg_g;
        let bg_b = total_b - fg_b;
        let fg_ssq = fg_r * fg_r + fg_g * fg_g + fg_b * fg_b;
        let bg_ssq = bg_r * bg_r + bg_g * bg_g + bg_b * bg_b;
        let mse = total_sq
            - if fg_n > 0 { fg_ssq / fg_n } else { 0 }
            - if bg_n > 0 { bg_ssq / bg_n } else { 0 };

        if mse < best_mse {
            best_mse = mse;
            best_ch = QUADRANT_CHARS[pattern as usize];
            best_fg_r = fg_r;
            best_fg_g = fg_g;
            best_fg_b = fg_b;
            best_fg_n = fg_n;
        }
    }

    let (fg, bg) = final_colors(best_fg_r, best_fg_g, best_fg_b, best_fg_n,
        total_r, total_g, total_b, 16);
    MangoResult { ch: best_ch, fg, bg }
}

/// SextantsOctants mode: 24 pixels, 168 patterns (half of 336, complement symmetry).
pub fn mango_select_all(pixels: &[(u8, u8, u8); 24]) -> MangoResult {
    let (total_r, total_g, total_b) = pixel_totals(pixels);
    let total_sq = pixel_sq_total(pixels);
    let mut best_mse = u32::MAX;
    let mut best_ch = ' ';
    let mut best_fg_r = 0u32;
    let mut best_fg_g = 0u32;
    let mut best_fg_b = 0u32;
    let mut best_fg_n = 0u32;
    try_patterns!(pixels, SEXTANT_CHARS, SEXT24_MASKS, SEXT24_FG, 64,
        total_r, total_g, total_b, total_sq,
        best_mse, best_ch, best_fg_r, best_fg_g, best_fg_b, best_fg_n);
    try_patterns!(pixels, OCTANT_CHARS, OCT24_MASKS, OCT24_FG, 256,
        total_r, total_g, total_b, total_sq,
        best_mse, best_ch, best_fg_r, best_fg_g, best_fg_b, best_fg_n);
    try_patterns!(pixels, QUADRANT_CHARS, QUAD24_MASKS, QUAD24_FG, 16,
        total_r, total_g, total_b, total_sq,
        best_mse, best_ch, best_fg_r, best_fg_g, best_fg_b, best_fg_n);
    let (fg, bg) = final_colors(best_fg_r, best_fg_g, best_fg_b, best_fg_n,
        total_r, total_g, total_b, 24);
    MangoResult { ch: best_ch, fg, bg }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadrant_empty() {
        assert_eq!(quadrant_char(0b0000), ' ');
    }

    #[test]
    fn test_quadrant_full() {
        assert_eq!(quadrant_char(0b1111), '█');
    }

    #[test]
    fn test_quadrant_upper_half() {
        assert_eq!(quadrant_char(0b0011), '▀');
    }

    #[test]
    fn test_quadrant_lower_half() {
        assert_eq!(quadrant_char(0b1100), '▄');
    }

    #[test]
    fn test_sextant_empty() {
        assert_eq!(sextant_char(0), ' ');
    }

    #[test]
    fn test_sextant_full() {
        assert_eq!(sextant_char(63), '█');
    }

    #[test]
    fn test_sextant_left_column() {
        assert_eq!(sextant_char(21), '▌');
    }

    #[test]
    fn test_sextant_right_column() {
        assert_eq!(sextant_char(42), '▐');
    }

    #[test]
    fn test_octant_empty() {
        assert_eq!(octant_char(0), ' ');
    }

    #[test]
    fn test_octant_full() {
        assert_eq!(octant_char(255), '█');
    }

    #[test]
    fn test_octant_left_half() {
        assert_eq!(octant_char(0b01010101), '▌');
    }

    #[test]
    fn test_octant_right_half() {
        assert_eq!(octant_char(0b10101010), '▐');
    }

    #[test]
    fn test_octant_upper_half() {
        assert_eq!(octant_char(0b00001111), '▀');
    }

    #[test]
    fn test_octant_lower_half() {
        assert_eq!(octant_char(0b11110000), '▄');
    }

    #[test]
    fn test_all_16_quadrant_patterns_unique() {
        let mut chars: Vec<char> = (0..16).map(quadrant_char).collect();
        chars.sort();
        chars.dedup();
        assert_eq!(chars.len(), 16);
    }

    #[test]
    fn test_all_64_sextant_patterns_unique() {
        let mut chars: Vec<char> = (0..64).map(sextant_char).collect();
        chars.sort();
        chars.dedup();
        assert_eq!(chars.len(), 64);
    }

    #[test]
    fn test_all_256_octant_patterns_unique() {
        let mut chars: Vec<char> = (0..=255u8).map(octant_char).collect();
        chars.sort();
        chars.dedup();
        assert_eq!(chars.len(), 256);
    }

    #[test]
    fn test_mango_select_sextants_uniform_white() {
        let pixels = [(255, 255, 255); 12];
        let result = mango_select_sextants(&pixels);
        assert_eq!(result.ch, ' ');
        assert_eq!(result.bg, (255, 255, 255));
    }

    #[test]
    fn test_mango_select_sextants_uniform_black() {
        let pixels = [(0, 0, 0); 12];
        let result = mango_select_sextants(&pixels);
        assert_eq!(result.ch, ' ');
        assert_eq!(result.bg, (0, 0, 0));
    }

    #[test]
    fn test_mango_select_sextants_half_split() {
        let mut pixels = [(0u8, 0u8, 0u8); 12];
        for i in 0..6 {
            pixels[i] = (255, 255, 255);
        }
        let result = mango_select_sextants(&pixels);
        assert_eq!(result.ch, '▀');
    }

    #[test]
    fn test_mango_select_quadrants_only_uniform() {
        let pixels = [(100, 100, 100); 8];
        let result = mango_select_quadrants_only(&pixels);
        assert_eq!(result.ch, ' ');
    }

    #[test]
    fn test_mango_select_octants_uniform() {
        let pixels = [(50, 50, 50); 16];
        let result = mango_select_octants(&pixels);
        assert_eq!(result.ch, ' ');
    }

    #[test]
    fn test_mango_select_all_uniform() {
        let pixels = [(200, 200, 200); 24];
        let result = mango_select_all(&pixels);
        assert_eq!(result.ch, ' ');
    }

    #[test]
    fn test_mango_select_octants_half_split() {
        // Top half white (8 pixels), bottom half black (8 pixels)
        let mut pixels = [(0u8, 0u8, 0u8); 16];
        for i in 0..8 {
            pixels[i] = (255, 255, 255);
        }
        let result = mango_select_octants(&pixels);
        assert_eq!(result.ch, '▀');
    }

    #[test]
    fn test_mode_from_flags() {
        assert_eq!(MangoMode::from_flags(false, false), MangoMode::QuadrantsOnly);
        assert_eq!(MangoMode::from_flags(true, false), MangoMode::Sextants);
        assert_eq!(MangoMode::from_flags(false, true), MangoMode::Octants);
        assert_eq!(MangoMode::from_flags(true, true), MangoMode::SextantsOctants);
    }

    #[test]
    fn test_piston_heights() {
        assert_eq!(MangoMode::QuadrantsOnly.piston_height(), 4);
        assert_eq!(MangoMode::Sextants.piston_height(), 6);
        assert_eq!(MangoMode::Octants.piston_height(), 8);
        assert_eq!(MangoMode::SextantsOctants.piston_height(), 12);
    }

}
