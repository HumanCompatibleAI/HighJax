//! Rendering modules for highway visualization.

pub mod color;
pub mod geometry;
pub mod highway_svg;
mod car_template;
mod frame_render;
mod brakelights;
mod headlights;
mod terrain;
mod velocity_arrows;
mod action_distribution;
mod npc_text;

pub use frame_render::{
    render_frame_with_svg_episode_or_fallback, RenderConfig, SceneRenderConfig,
};

/// Number of subdivisions per state for interpolation.
const N_SUBDIVISIONS: usize = 5;

/// Public access to subdivision count for playback timing.
pub const fn n_subdivisions() -> usize {
    N_SUBDIVISIONS
}
