//! CLI command implementations (draw, animate, zoltar).

mod animate;
mod draw;
mod zoltar;

pub(super) use animate::run_animate;
pub(super) use draw::{run_draw, run_draw_behavior};
pub(super) use zoltar::run_zoltar;
