//! Mask generation macros and precomputed mask/fg-count constants for all mango modes.

// Generate mask table: for each pattern, which pixels are foreground.
// n_char_rows: 2 (quadrants), 3 (sextants), or 4 (octants)
// piston_height must be divisible by n_char_rows
macro_rules! make_masks {
    ($n_pixels:expr, $piston_height:expr, $n_char_rows:expr, $n_patterns:expr) => {{
        let rows_per_cell = $piston_height / $n_char_rows;
        let mut masks = [[false; $n_pixels]; $n_patterns];
        let mut pattern = 0usize;
        while pattern < $n_patterns {
            let mut crow = 0;
            while crow < $n_char_rows {
                let mut col = 0;
                while col < 2 {
                    let bit = crow * 2 + col;
                    let is_fg = (pattern >> bit) & 1 == 1;
                    let mut dy = 0;
                    while dy < rows_per_cell {
                        let pix_idx = (crow * rows_per_cell + dy) * 2 + col;
                        masks[pattern][pix_idx] = is_fg;
                        dy += 1;
                    }
                    col += 1;
                }
                crow += 1;
            }
            pattern += 1;
        }
        masks
    }};
}

// Generate fg_count table from a mask table.
macro_rules! make_fg_counts {
    ($masks:expr, $n_pixels:expr, $n_patterns:expr) => {{
        let mut counts = [0u8; $n_patterns];
        let mut i = 0;
        while i < $n_patterns {
            let mut count = 0u8;
            let mut j = 0;
            while j < $n_pixels {
                if $masks[i][j] { count += 1; }
                j += 1;
            }
            counts[i] = count;
            i += 1;
        }
        counts
    }};
}

// Mode: QuadrantsOnly (PISTON_HEIGHT=4, 8 pixels)
pub(super) const QUAD8_MASKS: [[bool; 8]; 16] = make_masks!(8, 4, 2, 16);
pub(super) const QUAD8_FG: [u8; 16] = make_fg_counts!(QUAD8_MASKS, 8, 16);

// Mode: Sextants (PISTON_HEIGHT=6, 12 pixels)
pub(super) const SEXT12_MASKS: [[bool; 12]; 64] = make_masks!(12, 6, 3, 64);
pub(super) const SEXT12_FG: [u8; 64] = make_fg_counts!(SEXT12_MASKS, 12, 64);
pub(super) const QUAD12_MASKS: [[bool; 12]; 16] = make_masks!(12, 6, 2, 16);
pub(super) const QUAD12_FG: [u8; 16] = make_fg_counts!(QUAD12_MASKS, 12, 16);

// Mode: Octants (PISTON_HEIGHT=8, 16 pixels)
pub(super) const OCT16_MASKS: [[bool; 16]; 256] = make_masks!(16, 8, 4, 256);
pub(super) const OCT16_FG: [u8; 256] = make_fg_counts!(OCT16_MASKS, 16, 256);
pub(super) const QUAD16_MASKS: [[bool; 16]; 16] = make_masks!(16, 8, 2, 16);
pub(super) const QUAD16_FG: [u8; 16] = make_fg_counts!(QUAD16_MASKS, 16, 16);

// Mode: SextantsOctants (PISTON_HEIGHT=12, 24 pixels)
pub(super) const SEXT24_MASKS: [[bool; 24]; 64] = make_masks!(24, 12, 3, 64);
pub(super) const SEXT24_FG: [u8; 64] = make_fg_counts!(SEXT24_MASKS, 24, 64);
pub(super) const OCT24_MASKS: [[bool; 24]; 256] = make_masks!(24, 12, 4, 256);
pub(super) const OCT24_FG: [u8; 256] = make_fg_counts!(OCT24_MASKS, 24, 256);
pub(super) const QUAD24_MASKS: [[bool; 24]; 16] = make_masks!(24, 12, 2, 16);
pub(super) const QUAD24_FG: [u8; 16] = make_fg_counts!(QUAD24_MASKS, 24, 16);
