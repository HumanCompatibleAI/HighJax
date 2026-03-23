//! MSE computation utilities for mango template matching.
//!
//! Uses algebraic identity: Σ(x-μ)² = Σx² - (Σx)²/n to eliminate a second pixel loop.

/// Precompute sum of squared pixel channels: Σ(r² + g² + b²).
#[inline(always)]
pub(super) fn pixel_sq_total<const N: usize>(pixels: &[(u8, u8, u8); N]) -> u32 {
    let mut sq = 0u32;
    for &(r, g, b) in pixels {
        let (r, g, b) = (r as u32, g as u32, b as u32);
        sq += r * r + g * g + b * b;
    }
    sq
}

/// Compute MSE using algebraic identity: Σ(x-μ)² = Σx² - (Σx)²/n.
/// Returns (mse, fg_r_sum, fg_g_sum, fg_b_sum) — colors deferred to winner.
#[inline(always)]
pub(super) fn compute_mse<const N: usize>(
    mask: &[bool; N],
    pixels: &[(u8, u8, u8); N],
    fg_n: u32,
    total_r: u32,
    total_g: u32,
    total_b: u32,
    total_sq: u32,
) -> (u32, u32, u32, u32) {
    let bg_n = N as u32 - fg_n;

    let mut fg_r = 0u32;
    let mut fg_g = 0u32;
    let mut fg_b = 0u32;
    for i in 0..N {
        if mask[i] {
            let (r, g, b) = pixels[i];
            fg_r += r as u32;
            fg_g += g as u32;
            fg_b += b as u32;
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

    (mse, fg_r, fg_g, fg_b)
}

/// Try half the patterns per character set (complement symmetry: MSE(P) == MSE(n-1-P)).
macro_rules! try_patterns {
    ($pixels:expr, $chars:expr, $masks:expr, $fg_counts:expr, $n:expr,
     $total_r:expr, $total_g:expr, $total_b:expr, $total_sq:expr,
     $best_mse:expr, $best_ch:expr,
     $best_fg_r:expr, $best_fg_g:expr, $best_fg_b:expr, $best_fg_n:expr) => {
        for pattern in 0..($n / 2) {
            let fg_n = $fg_counts[pattern] as u32;
            let (mse, fg_r, fg_g, fg_b) = compute_mse(
                &$masks[pattern], $pixels, fg_n,
                $total_r, $total_g, $total_b, $total_sq
            );
            if mse < $best_mse {
                $best_mse = mse;
                $best_ch = $chars[pattern];
                $best_fg_r = fg_r;
                $best_fg_g = fg_g;
                $best_fg_b = fg_b;
                $best_fg_n = fg_n;
            }
        }
    };
}

/// Precompute RGB totals.
#[inline(always)]
pub(super) fn pixel_totals<const N: usize>(pixels: &[(u8, u8, u8); N]) -> (u32, u32, u32) {
    let mut tr = 0u32;
    let mut tg = 0u32;
    let mut tb = 0u32;
    for &(r, g, b) in pixels {
        tr += r as u32;
        tg += g as u32;
        tb += b as u32;
    }
    (tr, tg, tb)
}

/// Compute final fg/bg colors from channel sums.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn final_colors(
    fg_r: u32, fg_g: u32, fg_b: u32, fg_n: u32,
    total_r: u32, total_g: u32, total_b: u32, n: u32,
) -> ((u8, u8, u8), (u8, u8, u8)) {
    let bg_n = n - fg_n;
    let fg = if fg_n > 0 {
        ((fg_r / fg_n) as u8, (fg_g / fg_n) as u8, (fg_b / fg_n) as u8)
    } else {
        (128, 128, 128)
    };
    let bg = if bg_n > 0 {
        (((total_r - fg_r) / bg_n) as u8,
         ((total_g - fg_g) / bg_n) as u8,
         ((total_b - fg_b) / bg_n) as u8)
    } else {
        (128, 128, 128)
    };
    (fg, bg)
}
