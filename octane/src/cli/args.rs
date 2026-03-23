//! CLI argument definitions (Clap).

use clap::{Parser, Subcommand};
use std::path::PathBuf;


#[derive(Parser)]
#[command(name = "octane")]
#[command(about = "Browse Highway training episodes")]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Trek directory or parquet file (.parquet/.pq; auto-finds trek by walking up to meta.yaml)
    #[arg(short = 't', long, global = true)]
    pub target: Option<String>,

    /// Enable verbose logging to console
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Disable mango rendering (use ASCII only)
    #[arg(long, global = true)]
    pub no_mango: bool,

    /// Enable sextant characters
    #[arg(long, global = true, conflicts_with = "no_sextants")]
    pub sextants: bool,

    /// Disable sextant characters (use quadrants only, better compatibility)
    #[arg(long, global = true, conflicts_with = "sextants")]
    pub no_sextants: bool,

    /// Enable octant characters
    #[arg(long, global = true, conflicts_with = "no_octants")]
    pub octants: bool,

    /// Disable octant characters (use sextants/quadrants only)
    #[arg(long, global = true, conflicts_with = "octants")]
    pub no_octants: bool,

    /// Scene color theme (dark or light)
    #[arg(long, value_enum, global = true)]
    pub theme: Option<crate::config::SceneTheme>,

    /// Initial FPS for playback (1-60, overrides config value)
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=60), global = true)]
    pub fps: Option<u32>,

    /// Viewport smoothing omega (lower = weaker spring, more drift)
    #[arg(long, global = true)]
    pub omega: Option<f64>,

    /// Playback speed multiplier (e.g. 2.0 = double speed)
    #[arg(long, global = true)]
    pub speed: Option<f64>,

    /// Zoom level (e.g. 1.0 = default, 2.0 = zoomed in)
    #[arg(long, global = true)]
    pub zoom: Option<f64>,

    /// Sidebar width in columns
    #[arg(long)]
    pub sidebar_width: Option<u16>,

    /// Env-specific rendering preferences as comma-separated key=value pairs.
    /// Highway keys: podium_marker, attention, debug_eye (bool: true/false),
    /// velocity_arrows, action_distribution, action_distribution_text, npc_text
    /// (mode: on-pause/always/never), light_blend (string).
    /// Example: --prefs "velocity_arrows=always,podium_marker=true"
    #[arg(long, global = true)]
    pub prefs: Option<String>,

    /// Epoch number
    #[arg(long, global = true)]
    pub epoch: Option<i64>,

    /// Episode number (from parquet `e` column)
    #[arg(short = 'e', long, global = true)]
    pub episode: Option<usize>,

    /// Timestep t-value (e.g. 48 for policy step 48, or 3.40 for sub-step)
    #[arg(long, global = true)]
    pub timestep: Option<f64>,

    /// List epochs and exit (non-interactive mode)
    #[arg(long)]
    pub list_epochs: bool,

    /// Disable zoltar IPC interface (default: on)
    #[arg(long)]
    pub no_zoltar: bool,

    /// Use file-based IPC for zoltar instead of Unix socket
    #[arg(long)]
    pub zoltar_file_ipc: bool,
}

#[derive(Subcommand)]
pub(crate) enum Commands {
    /// Mango renderer utilities (benchmark, render files)
    Mango {
        #[command(subcommand)]
        command: MangoCommands,
    },

    /// Configuration management
    Config {
        #[command(subcommand)]
        command: ConfigCommands,
    },

    /// Draw a single frame (default: output to terminal)
    Draw {
        /// Output as SVG file
        #[arg(long, value_name = "FILE")]
        svg: Option<PathBuf>,

        /// Output as PNG file
        #[arg(long, value_name = "FILE")]
        png: Option<PathBuf>,

        /// Terminal width in columns (default: 120)
        #[arg(long, default_value = "120")]
        cols: u32,

        /// Terminal height in rows (default: 40)
        #[arg(long, default_value = "40")]
        rows: u32,

        /// Behavior name (draws a behavior scenario instead of a trek frame)
        #[arg(long)]
        behavior: Option<String>,

        /// Scenario index within the behavior (default: 0)
        #[arg(long, default_value = "0")]
        scenario: usize,

        /// Hardcoded action to show as arrow overlay (e.g. left, right, faster, slower, idle)
        #[arg(long)]
        hardcoded_action: Option<String>,
    },

    /// Query a running Octane instance via zoltar IPC
    Zoltar {
        /// PID of the Octane process (auto-discovers if not specified)
        #[arg(long)]
        pid: Option<u32>,
        #[command(subcommand)]
        command: ZoltarCommands,
    },

    /// Animate an episode to video file (MP4)
    Animate {
        /// Output video file (MP4)
        #[arg(short, long, value_name = "FILE")]
        output: PathBuf,

        /// Start frame (default: 0)
        #[arg(long, default_value = "0")]
        start: usize,

        /// End frame (default: last frame, or crash)
        #[arg(long)]
        end: Option<usize>,

        /// Video width in pixels (default: 1920)
        #[arg(long, default_value = "1920")]
        width: u32,

        /// Video height in pixels (default: 1080)
        #[arg(long, default_value = "1080")]
        height: u32,
    },
}

/// Mango subcommands.
#[derive(Subcommand)]
pub(crate) enum MangoCommands {
    /// Run comprehensive mango renderer benchmark
    Benchmark {
        /// Number of iterations per test (default: 10)
        #[arg(short, long, default_value = "10")]
        iterations: u32,
    },

    /// Render an SVG file to terminal
    Svg {
        /// Path to SVG file
        path: PathBuf,

        /// Output width in columns (default: terminal width)
        #[arg(short, long)]
        cols: Option<u32>,

        /// Output height in rows (default: terminal height)
        #[arg(short, long)]
        rows: Option<u32>,
    },

    /// Render an image file to terminal (PNG, JPEG, etc.)
    Image {
        /// Path to image file
        path: PathBuf,

        /// Output width in columns (default: terminal width)
        #[arg(short, long)]
        cols: Option<u32>,

        /// Output height in rows (default: terminal height)
        #[arg(short, long)]
        rows: Option<u32>,
    },

    /// Show mango character set
    Chars,
}

/// Config subcommands.
#[derive(Subcommand)]
pub(crate) enum ConfigCommands {
    /// Create config file with defaults, or fill in missing fields in existing file.
    FillDefaults,
    /// Print the config file path.
    Path,
}

/// Zoltar subcommands: query a running Octane instance.
#[derive(Subcommand)]
pub(crate) enum ZoltarCommands {
    /// Get semantic state snapshot
    Genco,
    /// Inject keystrokes
    Press {
        /// Key names to inject (e.g., "g", "enter", "shift+b")
        keys: Vec<String>,
    },
    /// Jump to a specific position
    Navigate {
        /// Epoch number
        #[arg(long)]
        epoch: Option<u64>,
        /// Episode number
        #[arg(long)]
        episode: Option<u64>,
        /// Timestep t-value
        #[arg(long)]
        timestep: Option<u64>,
    },
    /// Query a pane for structured data
    Pane {
        /// Pane name (e.g., "metrics", "scene", "explorer.behaviors")
        pane: String,
        /// Query type (e.g., "data", "svg", "screenshot")
        query: String,
        /// Output path (for screenshot queries)
        #[arg(long)]
        path: Option<String>,
    },
    /// Ping the zoltar server
    Ping,
}
