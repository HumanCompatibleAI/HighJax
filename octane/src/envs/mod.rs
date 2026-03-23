//! Environment adapters for highway dispatch.

pub mod highway;

use crate::config::Config;
use crate::data::jsonla::VehicleState;
use crate::data::trek::Trek;
use crate::render::SceneRenderConfig;
use crate::worlds::{SceneEpisode, ViewportConfig, ViewportEpisode};

/// Highway-specific parquet columns, extracted from a record batch.
pub(crate) struct EnvParquetColumns(highway::HighwayParquetColumns);

impl EnvParquetColumns {
    /// Build frame state (ego, npcs) from highway columns at row.
    pub fn build_frame_state(
        &self,
        row: usize,
    ) -> (VehicleState, Vec<VehicleState>) {
        self.0.build_frame_state(row)
    }
}

/// Environment type discriminant (highway-only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvType {
    Highway,
}

impl EnvType {
    /// Detect env type from meta.yaml commands section.
    /// Returns None if env type cannot be determined.
    pub fn detect_from_meta_yaml(doc: &serde_yaml::Value) -> Option<Self> {
        if highway::detect_from_meta(doc) {
            return Some(Self::Highway);
        }
        None
    }

    /// Extract env-specific columns from a parquet record batch.
    pub(crate) fn extract_parquet_columns(
        &self,
        batch: &arrow::record_batch::RecordBatch,
        parquet_columns: &[String],
    ) -> EnvParquetColumns {
        match self {
            Self::Highway => EnvParquetColumns(
                highway::HighwayParquetColumns::extract(batch, parquet_columns),
            ),
        }
    }

    /// Build a ViewportEpisode for this env type.
    pub fn build_viewport(
        &self,
        scene: SceneEpisode,
        vp_config: ViewportConfig,
        trek: &Trek,
        app_config: &Config,
    ) -> ViewportEpisode {
        match self {
            Self::Highway => highway::build_viewport(scene, vp_config, trek, app_config),
        }
    }

    /// Human-readable action name for display.
    pub fn action_name(&self, action: u8) -> &'static str {
        match self {
            Self::Highway => match action {
                0 => "LEFT",
                1 => "IDLE",
                2 => "RIGHT",
                3 => "FASTER",
                4 => "SLOWER",
                _ => "???",
            },
        }
    }

    /// Parse action name string to index (case-insensitive).
    pub fn parse_action_name(&self, name: &str) -> u8 {
        match self {
            Self::Highway => match name.to_uppercase().as_str() {
                "LEFT" => 0,
                "IDLE" => 1,
                "RIGHT" => 2,
                "FASTER" => 3,
                "SLOWER" => 4,
                _ => 0,
            },
        }
    }

    /// Environment name matching Python's env_name (used for behavior subdirectories, etc).
    pub fn env_name(&self) -> &'static str {
        match self {
            Self::Highway => "highway",
        }
    }

    /// Subpath within the highjax package for this env's preset behaviors.
    pub fn behaviors_subpath(&self) -> &'static str {
        match self {
            Self::Highway => "behaviors",
        }
    }

    /// Build a SceneRenderConfig for this env type.
    pub fn scene_render_config(
        &self,
        config: &Config,
        trek: &Trek,
        n_cols: u32,
        n_rows: u32,
        theme: crate::config::SceneTheme,
    ) -> SceneRenderConfig {
        match self {
            Self::Highway => highway::scene_render_config(config, trek, n_cols, n_rows, theme),
        }
    }
}
