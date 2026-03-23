//! Type definitions for application state: Focus, Modal, GraphicsControl, etc.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::time::Instant;

/// A toast notification with optional prefix styling and clipboard content.
pub struct Toast {
    /// Optional dim prefix (e.g. "Copied: ").
    pub prefix: Option<String>,
    /// Main message content.
    pub message: String,
    /// When the toast expires.
    pub expiry: Instant,
    /// Full underlying content (e.g. untruncated text for copy-generated toasts).
    pub clipboard_text: Option<String>,
    /// If Some(n), message was truncated: insert styled "…" at byte position n.
    pub ellipsis_at: Option<usize>,
}

impl Toast {
    /// Build the full display text as shown to the user.
    pub fn display_text(&self) -> String {
        let mut s = String::new();
        if let Some(ref prefix) = self.prefix {
            s.push_str(prefix);
        }
        if let Some(pos) = self.ellipsis_at {
            s.push_str(&self.message[..pos]);
            s.push('\u{2026}'); // …
            s.push_str(&self.message[pos..]);
        } else {
            s.push_str(&self.message);
        }
        s
    }
}

/// Which screen is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Browse,
    BehaviorExplorer,
}

/// Which pane has focus in the behavior explorer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplorerPane {
    Behaviors,
    Scenarios,
    Preview,
}

impl ExplorerPane {
    pub fn next(self) -> Self {
        match self {
            Self::Behaviors => Self::Scenarios,
            Self::Scenarios => Self::Preview,
            Self::Preview => Self::Behaviors,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::Behaviors => Self::Preview,
            Self::Scenarios => Self::Behaviors,
            Self::Preview => Self::Scenarios,
        }
    }
}

/// A behavior entry for the explorer list.
#[derive(Debug, Clone)]
pub struct BehaviorEntry {
    pub name: String,
    pub path: PathBuf,
    pub n_scenarios: usize,
    pub is_preset: bool,
}

/// A scenario entry for the explorer scenario list.
#[derive(Debug, Clone)]
pub struct ScenarioEntry {
    pub index: usize,
    pub action_weights: [f64; 5],
    pub has_state: bool,
    pub ego_speed: f64,
    pub ego_heading: f64,
    pub n_npcs: usize,
    /// Source target path: trek dir or parquet file (from scenario JSON "source.target").
    pub source_target: Option<std::path::PathBuf>,
    /// Source epoch number (from scenario JSON "source.epoch").
    pub source_epoch: Option<i64>,
    /// Source episode index (from scenario JSON "source.episode").
    pub source_episode: Option<usize>,
    /// Source timestep t-value (from scenario JSON "source.t").
    pub source_t: Option<f64>,
    /// Whether this scenario has been manually edited.
    pub edited: bool,
}

/// Explorer mode: browsing behaviors or editing a scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplorerMode {
    Browse,
    Edit,
    /// Text input mode for creating a new behavior.
    NewBehavior,
}

/// Kind of editor field, determines step sizes and labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorFieldKind {
    ActionWeight(usize),
    EgoSpeed,
    NpcRelX(usize),
    NpcRelY(usize),
    NpcSpeed(usize),
}

impl EditorFieldKind {
    pub fn label(self) -> String {
        match self {
            Self::ActionWeight(i) => {
                let names = ["left", "idle", "right", "faster", "slower"];
                format!("w:{}", names.get(i).unwrap_or(&"?"))
            }
            Self::EgoSpeed => "Ego speed".to_string(),
            Self::NpcRelX(i) => format!("NPC {} rel_x", i),
            Self::NpcRelY(i) => format!("NPC {} rel_y", i),
            Self::NpcSpeed(i) => format!("NPC {} speed", i),
        }
    }
}

/// An editable field in the scenario editor.
#[derive(Debug, Clone)]
pub struct EditorField {
    pub label: String,
    pub value: f64,
    pub kind: EditorFieldKind,
}

/// Raw behavior data returned by `list_all_behaviors()`.
pub type BehaviorLoadResult = Vec<(String, PathBuf, usize, String, bool)>;

/// State for the Behavior Explorer tab.
pub struct ExplorerState {
    pub behaviors: Vec<BehaviorEntry>,
    pub selected_behavior: usize,
    pub behaviors_scroll: usize,
    pub behaviors_visible_rows: Option<usize>,
    pub pane_focus: ExplorerPane,
    pub scenarios: Vec<ScenarioEntry>,
    pub selected_scenario: usize,
    pub scenarios_scroll: usize,
    pub scenarios_visible_rows: Option<usize>,
    pub preview_cache: Option<String>,
    pub preview_dims: (u32, u32),
    pub mode: ExplorerMode,
    pub delete_pending: bool,
    pub npc_remove_pending: bool,
    pub new_behavior_name: String,
    /// Receiver for background behavior loading.
    pub load_rx: Option<Receiver<BehaviorLoadResult>>,
    /// True once behaviors have been loaded at least once.
    pub loaded: bool,
    pub editor_fields: Vec<EditorField>,
    pub editor_selected_field: usize,
    pub editor_dirty: bool,
    /// Cached action probabilities keyed by (behavior_name, epoch).
    pub action_probs_cache: HashMap<(String, i64), Vec<crate::data::ActionDistribution>>,
    /// Receiver for background action probs fetch.
    pub action_probs_rx: Option<Receiver<ActionProbsResult>>,
}

/// Result from a background action probs fetch, tagged with request context.
pub struct ActionProbsResult {
    pub behavior_name: String,
    pub epoch: i64,
    pub probs: Option<Vec<crate::data::ActionDistribution>>,
}

/// State for the behavior scenario capture modal.
pub struct BehaviorModalState {
    /// Per-action weights [left, idle, right, faster, slower].
    pub action_weights: [f64; 5],
    /// Text input buffer for the currently-focused action weight.
    pub action_input: String,
    /// Behavior name input.
    pub name: String,
    /// Which field has focus.
    pub focus: BehaviorScenarioField,
    /// Cursor position in action list.
    pub action_cursor: usize,
    /// Cached behavior names from disk for autocomplete suggestions.
    pub suggestions: Vec<String>,
    /// Which suggestion is highlighted (None = no suggestion selected).
    pub suggestion_cursor: Option<usize>,
}

/// State for playback and animation timing.
pub struct PlaybackState {
    pub playing: bool,
    pub fps: u32,
    pub speed: f64,
    pub start: Option<Instant>,
    pub start_scene_time: f64,
    pub scene_time: f64,
}

/// The focus state for navigation between panes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Treks,
    Parquets,
    Epochs,
    Episodes,
    Highway,
}

impl Focus {
    /// Cycle to the next focus pane.
    pub fn next(self) -> Self {
        match self {
            Focus::Treks => Focus::Parquets,
            Focus::Parquets => Focus::Epochs,
            Focus::Epochs => Focus::Episodes,
            Focus::Episodes => Focus::Highway,
            Focus::Highway => Focus::Treks,
        }
    }

    /// Cycle to the previous focus pane.
    pub fn prev(self) -> Self {
        match self {
            Focus::Treks => Focus::Highway,
            Focus::Parquets => Focus::Treks,
            Focus::Epochs => Focus::Parquets,
            Focus::Episodes => Focus::Epochs,
            Focus::Highway => Focus::Episodes,
        }
    }
}

/// Saved selection state for a previously-visited trek.
#[derive(Debug, Clone)]
pub struct TrekViewState {
    pub selected_parquet: usize,
    pub selected_epoch: usize,
    pub selected_episode: usize,
    pub epochs_scroll: usize,
    pub episodes_scroll: usize,
}

/// Modal dialog types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modal {
    /// Help / keybindings.
    Help,
    /// Graphics settings dialog (FPS, mode, zoom, etc.).
    Graphics,
    /// Jump to epoch.
    JumpEpoch,
    /// Jump to episode.
    JumpEpisode,
    /// Jump to frame.
    JumpFrame,
    /// Behavior scenario capture (add scenario to a behavior).
    BehaviorScenario,
}

/// Which field has focus in the behavior scenario modal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BehaviorScenarioField {
    Actions,
    Name,
}

impl BehaviorScenarioField {
    pub fn next(self) -> Self {
        match self {
            Self::Actions => Self::Name,
            Self::Name => Self::Actions,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::Actions => Self::Name,
            Self::Name => Self::Actions,
        }
    }
}

pub const ACTION_NAMES: [&str; 5] = ["LEFT", "IDLE", "RIGHT", "FASTER", "SLOWER"];

/// Graphics dialog control items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphicsControl {
    Theme,
    Fps,
    PlaybackSpeed,
    Zoom,
    SidebarWidth,
    PodiumMarker,
    Sextants,
    Octants,
    VelocityArrows,
    ActionDistribution,
    ActionDistributionText,
    Attention,
    Scala,
    DebugEye,
    LightBlendMode,
    NpcText,
}

/// SVG blend modes available for light cones.
pub const LIGHT_BLEND_MODES: &[&str] = &[
    "normal", "screen", "lighten", "color-dodge",
    "multiply", "overlay", "hard-light", "soft-light",
    "difference", "exclusion",
];

impl GraphicsControl {
    pub(super) const ALL: [GraphicsControl; 16] = [
        GraphicsControl::Theme,
        GraphicsControl::Fps,
        GraphicsControl::PlaybackSpeed,
        GraphicsControl::Zoom,
        GraphicsControl::SidebarWidth,
        GraphicsControl::PodiumMarker,
        GraphicsControl::Scala,
        GraphicsControl::Sextants,
        GraphicsControl::Octants,
        GraphicsControl::VelocityArrows,
        GraphicsControl::ActionDistribution,
        GraphicsControl::ActionDistributionText,
        GraphicsControl::Attention,
        GraphicsControl::NpcText,
        GraphicsControl::DebugEye,
        GraphicsControl::LightBlendMode,
    ];

    /// Controls shown in the behavior explorer Graphics modal.
    pub(super) const EXPLORER: [GraphicsControl; 11] = [
        GraphicsControl::Theme,
        GraphicsControl::Zoom,
        GraphicsControl::PodiumMarker,
        GraphicsControl::Scala,
        GraphicsControl::Sextants,
        GraphicsControl::Octants,
        GraphicsControl::VelocityArrows,
        GraphicsControl::ActionDistribution,
        GraphicsControl::ActionDistributionText,
        GraphicsControl::NpcText,
        GraphicsControl::LightBlendMode,
    ];

    /// Navigate to next control within a filtered set.
    pub(super) fn next_in(self, set: &[GraphicsControl]) -> Self {
        let pos = set.iter().position(|&c| c == self).unwrap_or(0);
        set[(pos + 1) % set.len()]
    }

    /// Navigate to previous control within a filtered set.
    pub(super) fn prev_in(self, set: &[GraphicsControl]) -> Self {
        let pos = set.iter().position(|&c| c == self).unwrap_or(0);
        set[(pos + set.len() - 1) % set.len()]
    }
}
