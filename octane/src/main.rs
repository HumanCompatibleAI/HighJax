//! Highway TUI - Ratatui-based episode browser for Highway RL environment.
//!
//! This is a Rust reimplementation of the Python octane, providing a
//! terminal UI for browsing and visualizing Highway training episodes.

mod app;
mod cli;
mod config;
mod data;
mod envs;
mod mango;
mod render;
mod ui;
mod util;
mod worlds;
mod zoltar;

fn main() -> anyhow::Result<()> {
    let result = cli::run();
    if let Err(ref e) = result {
        tracing::error!("{:#}", e);
    }
    result
}
