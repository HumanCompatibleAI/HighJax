//! Coordinate system architecture for Octane.
//!
//! This module provides a layered Episode architecture for transforming vehicle
//! positions through multiple coordinate systems:
//!
//! - **Scene**: Simulation world (meters)
//! - **Viewport**: Camera-relative scene coordinates (meters, centered on viewport)
//! - **SVG**: Normalized drawing surface (width=1, height varies)

#[allow(dead_code)]
pub mod coords;
pub mod scene_episode;
#[allow(dead_code)]
pub mod svg_episode;
#[allow(dead_code)]
pub mod viewport_episode;

pub use coords::{SceneBounds, ScenePoint};
pub use scene_episode::SceneEpisode;
pub use svg_episode::{SvgConfig, SvgEpisode};
pub use viewport_episode::{ViewportConfig, ViewportEpisode};

// =============================================================================
// Constants
// =============================================================================

/// Scene distance (in meters) that the canvas diagonal spans at zoom=1.0×.
/// This is constant regardless of aspect ratio or output mode (TUI vs image).
/// At zoom=2.0×, the diagonal spans half this (objects appear twice as big).
pub const UNZOOMED_CANVAS_DIAGONAL_IN_METERS: f64 = 180.0;

/// Default corn aspect ratio (terminal character height / width).
/// Derived from typical terminal font metrics.
pub const DEFAULT_CORN_ASPRO: f64 = 1.875;

/// Default simulation timestep duration in seconds (for tests only).
/// Production code reads `seconds_per_t` from trek metadata.
pub const DEFAULT_SECONDS_PER_TIMESTEP: f64 = 0.1;
