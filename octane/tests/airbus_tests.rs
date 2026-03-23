//! Airbus-based integration tests for Octane TUI.
//!
//! These tests spawn octane in a real PTY via airbus, send keystrokes, and
//! assert on screen dumps.  They require airbus to be installed.
//! Linux only: ConPTY on Windows doesn't render TUI output correctly yet.
#![cfg(target_os = "linux")]

mod common;

use tempfile::TempDir;

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Create a unique session name from the test function name.
fn session(name: &str) -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let hash = RandomState::new().build_hasher().finish();
    format!("oct-{}-{:06x}", name, hash & 0xFFFFFF)
}

/// Guard that stops the session on drop.
struct SessionGuard(String);

impl SessionGuard {
    fn name(&self) -> &str { &self.0 }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        common::stop_session(&self.0);
    }
}

/// Spawn octane on a fixture trek, returning a guard that auto-cleans.
fn spawn_fixture(
    dir: &std::path::Path,
    test_name: &str,
    extra_args: &[&str],
) -> SessionGuard {
    let name = session(test_name);
    common::spawn_octane(dir, &name, extra_args);
    common::wait_for_text(&name, "epoch", 5);
    SessionGuard(name)
}

/// Spawn octane with a pre-created test behavior file, returning a guard.
fn spawn_fixture_with_behavior(
    dir: &std::path::Path,
    test_name: &str,
    extra_args: &[&str],
) -> SessionGuard {
    let name = session(test_name);
    common::write_test_behavior(&name);
    common::spawn_octane(dir, &name, extra_args);
    common::wait_for_text(&name, "epoch", 5);
    SessionGuard(name)
}

/// Spawn octane with a pre-created sourced behavior file, returning a guard.
fn spawn_fixture_with_sourced_behavior(
    dir: &std::path::Path,
    test_name: &str,
    extra_args: &[&str],
) -> SessionGuard {
    let name = session(test_name);
    common::write_test_behavior_with_source(&name);
    common::spawn_octane(dir, &name, extra_args);
    common::wait_for_text(&name, "epoch", 5);
    SessionGuard(name)
}

// ─── Phase 1: Smoke test ────────────────────────────────────────────────────

#[test]

fn airbus_smoke_simple_trek() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "smoke-simple", &[]);
    let screen = common::dump_text(guard.name());

    // Octane should have rendered *something* — screen should not be blank.
    let non_blank: usize = screen.chars().filter(|c| !c.is_whitespace()).count();
    assert!(non_blank > 50, "Screen looks blank:\n{}", screen);
}

#[test]

fn airbus_smoke_crash_trek() {
    let dir = TempDir::new().unwrap();
    common::generate_crash_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "smoke-crash", &[]);
    let screen = common::dump_text(guard.name());

    let non_blank: usize = screen.chars().filter(|c| !c.is_whitespace()).count();
    assert!(non_blank > 50, "Screen looks blank:\n{}", screen);
}

// ─── Phase 2: Startup and pane navigation ───────────────────────────────────

#[test]

fn airbus_startup_shows_status_bar() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "status-bar", &[]);
    let screen = common::dump_text(guard.name());

    // Status bar should mention epoch or episode info.
    let lower = screen.to_lowercase();
    assert!(
        lower.contains("epoch") || lower.contains("e:") || lower.contains("t:") || lower.contains("0/"),
        "Status bar not found in screen:\n{}", screen
    );
}

#[test]

fn airbus_tab_cycles_pane_focus() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "tab-focus", &[]);
    common::wait_for_text(guard.name(), "Timestep", 5);

    // Press Tab several times to cycle through all panes and back
    common::press_with_delay(guard.name(), &["tab", "tab", "tab", "tab"], 150);
    common::settle(); // Tab only changes highlighting, not text content

    let after = common::dump_text(guard.name());

    // After cycling through all panes, the UI should still be intact
    assert!(after.contains("Timestep"), "Screen should still show timestep info after Tab cycling");
    assert!(after.contains("epoch"), "Screen should still show epoch pane after Tab cycling");

    // The screen should remain stable (Tab doesn't break rendering)
    let non_blank: usize = after.chars().filter(|c| !c.is_whitespace()).count();
    assert!(non_blank > 50, "Screen should not be blank after Tab cycling");
}

#[test]

fn airbus_epoch_list_shows_epochs() {
    let dir = TempDir::new().unwrap();
    common::generate_crash_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "epoch-list", &[]);
    let epochs = common::zoltar_pane(guard.name(), "epochs", "data");
    let epoch_list = epochs["epochs"].as_array().unwrap();
    assert_eq!(epoch_list.len(), 2, "Crash trek should have 2 epochs: {}", epochs);
    assert_eq!(epoch_list[0]["number"], 0);
    assert_eq!(epoch_list[1]["number"], 1);
}

// ─── Phase 3: Playback and timestep controls ────────────────────────────────

#[test]

fn airbus_k_advances_timestep() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    // Start at t=2; k = forward 1 frame in octane
    let guard = spawn_fixture(dir.path(), "k-advance", &["--timestep", "2"]);
    common::wait_for_genco(guard.name(), "frame_index == 2", |z| z["frame_index"] == 2, 5);

    // Press k (forward) with delay between keys
    common::press_with_delay(guard.name(), &["k", "k", "k"], 100);
    common::wait_for_genco(guard.name(), "frame_index == 5",
        |z| z["frame_index"] == 5, 5);
}

#[test]

fn airbus_j_goes_back() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    // j = back 1 frame, k = forward 1 frame in octane
    let guard = spawn_fixture(dir.path(), "j-back", &["--timestep", "5"]);
    common::wait_for_genco(guard.name(), "frame_index == 5", |z| z["frame_index"] == 5, 5);

    // j goes back
    common::press(guard.name(), &["j", "j"]);
    common::wait_for_genco(guard.name(), "frame_index == 3", |z| z["frame_index"] == 3, 5);

    // k returns forward
    common::press(guard.name(), &["k", "k"]);
    common::wait_for_genco(guard.name(), "frame_index == 5", |z| z["frame_index"] == 5, 5);
}

#[test]

fn airbus_h_and_l_jump() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "hl-jump", &["--timestep", "5"]);
    common::wait_for_genco(guard.name(), "frame_index == 5", |z| z["frame_index"] == 5, 5);

    // h → first timestep
    common::press(guard.name(), &["h"]);
    common::wait_for_genco(guard.name(), "frame_index == 0", |z| z["frame_index"] == 0, 5);

    // l → last timestep
    common::press(guard.name(), &["l"]);
    common::wait_for_genco(guard.name(), "frame_index == 9", |z| z["frame_index"] == 9, 5);
}

#[test]

fn airbus_p_toggles_pause() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "p-pause", &[]);
    common::wait_for_genco(guard.name(), "frame_index == 0", |z| z["frame_index"] == 0, 5);

    // Toggle play with p, wait for playback to advance past frame 0
    common::press(guard.name(), &["p"]);
    common::wait_for_genco(guard.name(), "frame_index > 0",
        |z| z["frame_index"].as_u64().map_or(false, |f| f > 0), 5);

    // Pause again
    common::press(guard.name(), &["p"]);
}

// ─── Phase 4: Rendering correctness ─────────────────────────────────────────

#[test]

fn airbus_crash_trek_shows_crash() {
    let dir = TempDir::new().unwrap();
    common::generate_crash_trek(dir.path());

    // Navigate to epoch 0, episode 1, timestep 8 (after crash at t=7)
    let guard = spawn_fixture(dir.path(), "crash-render",
        &["--epoch", "0", "-e", "1", "--timestep", "8"]);
    let screen = common::dump_text(guard.name());

    // Navigate to epoch 0, episode 0 (no crash) for comparison
    let dir2 = TempDir::new().unwrap();
    common::generate_crash_trek(dir2.path());
    let guard2 = spawn_fixture(dir2.path(), "no-crash-render",
        &["--epoch", "0", "-e", "0", "--timestep", "8"]);
    let screen2 = common::dump_text(guard2.name());

    // The crashed and non-crashed scenes should look different.
    assert_ne!(screen, screen2,
        "Crashed and non-crashed episodes should render differently");
}

#[test]

fn airbus_resize_rerenders() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "resize", &[]);
    let before = common::dump_text(guard.name());

    // Resize via airbus
    let output = std::process::Command::new("airbus")
        .args(["-s", guard.name(), "resize", "160x50"])
        .output()
        .expect("airbus resize failed");
    assert!(output.status.success(), "resize command failed");
    common::settle();

    let after = common::dump_text(guard.name());

    // After resize, the screen dimensions change → different dump.
    assert_ne!(before, after, "Resize should produce different output");
}

// ─── Phase 5: Modals and edge cases ─────────────────────────────────────────

#[test]

fn airbus_help_modal() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "help-modal", &[]);
    common::wait_for_text(guard.name(), "Timestep", 5);

    // Open help with ?  (shift+/ sends ?)
    common::press(guard.name(), &["?"]);
    common::wait_for_text(guard.name(), "Quit", 5);
    let with_help = common::dump_text(guard.name());

    // Help modal should show keybinding info
    let lower = with_help.to_lowercase();
    assert!(
        lower.contains("help") || lower.contains("key") || lower.contains("quit")
            || lower.contains("pause") || lower.contains("timestep"),
        "Help modal should show keybinding info:\n{}", with_help
    );

    // Close help
    common::press(guard.name(), &["escape"]);
    common::wait_for_text(guard.name(), "Timestep", 3);
    let after_close = common::dump_text(guard.name());

    // After closing help, the normal UI should be back (status bar visible)
    assert!(after_close.contains("Timestep"),
        "After closing help, status bar should be visible:\n{}", after_close);
}

#[test]

fn airbus_npc_text() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    // --prefs npc_text=always enables NPC labels regardless of pause state
    let guard = spawn_fixture(dir.path(), "npc-nums", &["--prefs", "npc_text=always"]);
    common::wait_for_text(guard.name(), "Timestep", 5);

    let screen = common::dump_text(guard.name());

    // Simple trek has 2 NPCs; look for any NPC text evidence (label, speed, or collision warning).
    // Mango rasterization may partially occlude labels, and the viewport podium offset
    // can push NPCs off-screen, so we only require at least one NPC to be visible.
    let has_npc0 = screen.contains("npc0") || screen.contains("18m/s");
    let has_npc1 = screen.contains("npc1") || screen.contains("22m/s");
    assert!(has_npc0 || has_npc1,
        "At least one NPC text should be visible:\n{}", screen);
}

#[test]

fn airbus_reward_score_display() {
    let dir = TempDir::new().unwrap();
    common::generate_crash_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "reward-score", &[]);
    common::wait_for_text(guard.name(), "NReturn", 5);

    let screen = common::dump_text(guard.name());

    // Epochs pane should show NReturn and Survival columns
    assert!(screen.contains("NReturn"), "Epochs pane should have NReturn column:\n{}", screen);
    assert!(screen.contains("Survival"), "Epochs pane should have Survival column:\n{}", screen);

    // Crash trek has crashes, so survival should be < 100%
    // and NReturn values should be visible
    let lower = screen.to_lowercase();
    assert!(lower.contains("-") || lower.contains("0.") || lower.contains("%"),
        "NReturn/survival values should be visible (non-blank):\n{}", screen);
}

#[test]

fn airbus_q_exits_cleanly() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "q-exit", &[]);

    // q should cause octane to exit
    common::press(guard.name(), &["q"]);

    // Poll until the session is no longer [running] (may briefly appear as [dead]).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let running_pattern = format!("{} [running]", guard.name());
    loop {
        let list_output = std::process::Command::new("airbus")
            .arg("list")
            .output()
            .expect("airbus list failed");
        let list_text = String::from_utf8_lossy(&list_output.stdout);
        if !list_text.contains(&running_pattern) {
            break;
        }
        if std::time::Instant::now() >= deadline {
            panic!("Session '{}' still running 5s after q:\n{}",
                guard.name(), list_text);
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

// ─── Phase 6: Behavior Explorer ─────────────────────────────────────────────

#[test]

fn airbus_shift_b_opens_explorer() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "shift-b-open", &[]);
    common::wait_for_text(guard.name(), "Timestep", 5);

    // Shift-B should open the Behavior Explorer
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "scenarios", 5);

    let screen = common::dump_text(guard.name());

    // Explorer screen should show tab bar and pane headers
    assert!(screen.contains("Behaviors"),
        "Shift-B should open Behaviors tab:\n{}", screen);
    assert!(screen.contains("scenarios"),
        "Explorer should have scenarios pane:\n{}", screen);
    assert!(screen.contains("preview"),
        "Explorer should have preview pane:\n{}", screen);
}

#[test]

fn airbus_explorer_returns_with_q() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "explorer-q", &[]);
    common::wait_for_text(guard.name(), "Timestep", 5);

    // Open explorer
    common::type_text(guard.name(), "B");
    common::wait_for_genco(guard.name(), "screen contains Explorer",
        |z| z["screen"].as_str().map_or(false, |s| s.contains("Explorer")), 5);

    // q should return to browse mode
    common::press(guard.name(), &["q"]);
    common::wait_for_genco(guard.name(), "screen contains Browse",
        |z| z["screen"].as_str().map_or(false, |s| s.contains("Browse")), 3);
}

#[test]

fn airbus_explorer_tab_cycles_panes() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "explorer-tab", &[]);
    common::wait_for_text(guard.name(), "Timestep", 5);

    // Open explorer
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "scenarios", 5);

    // Tab should cycle through panes without breaking
    common::press_with_delay(guard.name(), &["tab", "tab", "tab"], 150);
    common::settle(); // Tab only changes highlighting

    let screen = common::dump_text(guard.name());
    assert!(screen.contains("scenarios"),
        "Tab cycling should not leave explorer:\n{}", screen);
    assert!(screen.contains("behaviors"),
        "Behaviors pane should still be visible:\n{}", screen);
}

#[test]

fn airbus_explorer_preview_renders() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture_with_behavior(dir.path(), "explorer-preview", &[]);

    // Open explorer — should show test-behavior
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "test-behavior", 5);

    let screen = common::dump_text(guard.name());
    assert!(screen.contains("test-behavior"),
        "Explorer should show test-behavior:\n{}", screen);

    // The preview pane should have non-whitespace content (rendered highway)
    // Count characters in the right portion of the screen (preview area)
    let lines: Vec<&str> = screen.lines().collect();
    let preview_chars: usize = lines.iter()
        .filter_map(|line| line.get(60..))  // Right side of screen
        .map(|s| s.chars().filter(|c| !c.is_whitespace()).count())
        .sum();
    assert!(preview_chars > 20,
        "Preview pane should have rendered content (got {} non-ws chars in right side):\n{}",
        preview_chars, screen);
}

#[test]

fn airbus_explorer_preview_changes_on_nav() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture_with_behavior(dir.path(), "explorer-preview-nav", &[]);

    // Open explorer
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "scenarios", 5);

    // Move focus to scenarios pane, capture first scenario's screen
    common::press(guard.name(), &["tab"]);
    common::settle(); // Tab only changes highlighting
    let screen1 = common::dump_text(guard.name());

    // Navigate to second scenario
    common::press(guard.name(), &["down"]);
    common::settle(); // Selection change, hard to detect specific text
    let screen2 = common::dump_text(guard.name());

    // The two screens should differ (different scenario = different preview)
    assert_ne!(screen1, screen2,
        "Navigating scenarios should change the preview");
}

#[test]

fn airbus_explorer_edit_mode() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture_with_behavior(dir.path(), "explorer-edit", &[]);

    // Open explorer, press e to enter edit mode
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "scenarios", 5);
    common::press(guard.name(), &["e"]);
    common::wait_for_genco(guard.name(), "explorer.mode contains Edit",
        |z| z["explorer"]["mode"].as_str().map_or(false, |s| s.contains("Edit")), 5);

    let editor = common::zoltar_pane(guard.name(), "explorer.editor", "data");
    let fields = editor["fields"].as_array().unwrap();
    assert!(fields.iter().any(|f| f["label"].as_str() == Some("Ego speed")),
        "Editor should have 'Ego speed' field: {}", editor);
}

#[test]

fn airbus_explorer_edit_cancel() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture_with_behavior(dir.path(), "explorer-edit-cancel", &[]);

    // Open explorer, enter edit mode, then cancel with Esc
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "scenarios", 5);
    common::press(guard.name(), &["e"]);
    common::wait_for_genco(guard.name(), "explorer.mode contains Edit",
        |z| z["explorer"]["mode"].as_str().map_or(false, |s| s.contains("Edit")), 5);

    common::press(guard.name(), &["escape"]);
    common::wait_for_genco(guard.name(), "explorer.mode not Edit",
        |z| z["explorer"]["mode"].as_str().map_or(true, |s| !s.contains("Edit")), 3);
}

#[test]

fn airbus_explorer_edit_value() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture_with_behavior(dir.path(), "explorer-edit-val", &[]);

    // Open explorer, enter edit mode
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "scenarios", 5);
    common::press(guard.name(), &["e"]);
    common::wait_for_genco(guard.name(), "explorer.mode contains Edit",
        |z| z["explorer"]["mode"].as_str().map_or(false, |s| s.contains("Edit")), 5);

    // Navigate past 5 action weight fields to Ego speed
    common::press(guard.name(), &["down", "down", "down", "down", "down"]);
    common::settle();

    let before = common::zoltar_pane(guard.name(), "explorer.editor", "data");
    let ego_field = before["fields"].as_array().unwrap().iter()
        .find(|f| f["label"].as_str() == Some("Ego speed")).unwrap();
    assert_eq!(ego_field["value"].as_f64().unwrap(), 20.0,
        "Initial ego speed should be 20.0");

    // Right arrow should increase ego speed by 1.0
    common::press(guard.name(), &["right"]);
    common::settle();

    let after = common::zoltar_pane(guard.name(), "explorer.editor", "data");
    let ego_field = after["fields"].as_array().unwrap().iter()
        .find(|f| f["label"].as_str() == Some("Ego speed")).unwrap();
    assert_eq!(ego_field["value"].as_f64().unwrap(), 21.0,
        "After right arrow, ego speed should be 21.0");
    assert!(after["dirty"].as_bool().unwrap(),
        "Should show modified (dirty) after edit");
}

#[test]

fn airbus_explorer_edit_save() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture_with_behavior(dir.path(), "explorer-edit-save", &[]);

    // Open explorer, enter edit, adjust value, save with Enter
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "scenarios", 5);
    common::press(guard.name(), &["e"]);
    common::wait_for_genco(guard.name(), "explorer.mode contains Edit",
        |z| z["explorer"]["mode"].as_str().map_or(false, |s| s.contains("Edit")), 5);
    // Navigate past 5 action weight fields to Ego speed
    common::press(guard.name(), &["down", "down", "down", "down", "down"]);
    common::press_with_delay(guard.name(), &["right", "right", "right"], 100);
    common::settle();
    common::press(guard.name(), &["enter"]);
    common::wait_for_genco(guard.name(), "explorer.mode not Edit",
        |z| z["explorer"]["mode"].as_str().map_or(true, |s| !s.contains("Edit")), 3);

    // Re-enter edit to verify persistence
    common::press(guard.name(), &["e"]);
    common::wait_for_genco(guard.name(), "explorer.mode contains Edit",
        |z| z["explorer"]["mode"].as_str().map_or(false, |s| s.contains("Edit")), 5);

    let re_edit = common::zoltar_pane(guard.name(), "explorer.editor", "data");
    let ego_field = re_edit["fields"].as_array().unwrap().iter()
        .find(|f| f["label"].as_str() == Some("Ego speed")).unwrap();
    assert_eq!(ego_field["value"].as_f64().unwrap(), 23.0,
        "Saved value (20+3=23) should persist");
}

#[test]

fn airbus_explorer_add_npc() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture_with_behavior(dir.path(), "explorer-add-npc", &[]);

    // Open explorer, enter edit mode
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "scenarios", 5);
    common::press(guard.name(), &["e"]);
    common::wait_for_genco(guard.name(), "explorer.mode contains Edit",
        |z| z["explorer"]["mode"].as_str().map_or(false, |s| s.contains("Edit")), 5);

    let before = common::zoltar_pane(guard.name(), "explorer.editor", "data");
    let before_npc_count = before["fields"].as_array().unwrap().iter()
        .filter(|f| f["label"].as_str().map_or(false, |l| l.contains("rel_x")))
        .count();
    assert_eq!(before_npc_count, 2, "Should have 2 NPCs before add");

    // Press 'a' to add NPC
    common::press(guard.name(), &["a"]);
    common::settle();

    let after = common::zoltar_pane(guard.name(), "explorer.editor", "data");
    let after_npc_count = after["fields"].as_array().unwrap().iter()
        .filter(|f| f["label"].as_str().map_or(false, |l| l.contains("rel_x")))
        .count();
    assert_eq!(after_npc_count, 3, "Should have 3 NPCs after add");
    assert!(after["dirty"].as_bool().unwrap(), "Should show modified after add");
}

#[test]

fn airbus_explorer_delete_scenario() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture_with_behavior(dir.path(), "explorer-del-scen", &[]);

    // Open explorer, focus scenarios pane
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "scenarios", 5);
    common::press(guard.name(), &["tab"]); // -> Scenarios
    common::settle();

    let before = common::zoltar_pane(guard.name(), "explorer.scenarios", "data");
    let before_count = before["scenarios"].as_array().unwrap().len();
    assert_eq!(before_count, 2, "Should start with 2 scenarios");

    // Press d twice to delete (first d prompts, second d confirms)
    common::press_with_delay(guard.name(), &["d", "d"], 200);
    common::wait_for_pane(guard.name(), "explorer.scenarios",
        |data| data["scenarios"].as_array().map_or(false, |s| s.len() == 1),
        "scenario count == 1", 5);
}

#[test]

fn airbus_explorer_new_scenario() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture_with_behavior(dir.path(), "explorer-new-scen", &[]);

    // Open explorer, focus scenarios pane
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "scenarios", 5);
    common::press(guard.name(), &["tab"]); // -> Scenarios
    common::settle();

    let before = common::zoltar_pane(guard.name(), "explorer.scenarios", "data");
    let before_count = before["scenarios"].as_array().unwrap().len();
    assert_eq!(before_count, 2, "Should start with 2 scenarios");

    // Press n to create new scenario
    common::press(guard.name(), &["n"]);
    common::settle();

    let after = common::zoltar_pane(guard.name(), "explorer.scenarios", "data");
    let after_count = after["scenarios"].as_array().unwrap().len();
    assert_eq!(after_count, 3, "Should have 3 scenarios after creating new one");
}

#[test]

fn airbus_explorer_edit_preview_updates() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture_with_behavior(dir.path(), "explorer-edit-preview", &[]);

    // Open explorer, enter edit
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "scenarios", 5);
    common::press(guard.name(), &["e"]);
    common::wait_for_text(guard.name(), "Ego speed", 5);

    let before = common::dump_text(guard.name());

    // Navigate to NPC 0 rel_x (field index 1) and adjust
    common::press(guard.name(), &["down"]);
    common::settle(); // Selection change
    common::press_with_delay(guard.name(), &["right", "right", "right"], 100);
    common::wait_for_text(guard.name(), "modified", 5);

    let after = common::dump_text(guard.name());
    // Preview should change after adjusting NPC position
    assert_ne!(before, after,
        "Preview should update after adjusting NPC rel_x");
}

#[test]

fn airbus_explorer_enter_behaviors_moves_to_scenarios() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture_with_behavior(dir.path(), "explorer-enter-beh", &[]);

    // Open explorer — focus starts on Behaviors pane
    common::type_text(guard.name(), "B");
    common::wait_for_genco(guard.name(), "screen contains Explorer",
        |z| z["screen"].as_str().map_or(false, |s| s.contains("Explorer")), 5);

    let before = common::zoltar_genco(guard.name());
    assert!(before["explorer"]["pane_focus"].as_str().unwrap().contains("Behaviors"),
        "Focus should start on Behaviors: {}", before["explorer"]["pane_focus"]);

    // Press Enter in Behaviors pane — should move focus to Scenarios
    common::press(guard.name(), &["enter"]);
    common::settle();

    let after = common::zoltar_genco(guard.name());
    assert!(after["explorer"]["pane_focus"].as_str().unwrap().contains("Scenarios"),
        "After Enter, focus should be on Scenarios: {}", after["explorer"]["pane_focus"]);
}

#[test]

fn airbus_explorer_scenarios_weight_columns() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture_with_behavior(dir.path(), "explorer-weight-cols", &[]);

    // Open explorer
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "scenarios", 5);

    let screen = common::dump_text(guard.name());
    // Verify column headers are present
    assert!(screen.contains("L") && screen.contains("I") && screen.contains("R")
        && screen.contains("F") && screen.contains("S"),
        "Scenarios table should show L I R F S column headers:\n{}", screen);

    // Verify the test behavior's action weights render (left:1 -> "1" in L column)
    // and zeros as dots
    assert!(screen.contains("·"),
        "Zero weights should render as dots:\n{}", screen);
}

#[test]

fn airbus_explorer_edit_action_weight() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture_with_behavior(dir.path(), "explorer-edit-aw", &[]);

    // Open explorer, enter edit mode
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "scenarios", 5);
    common::press(guard.name(), &["e"]);
    common::wait_for_genco(guard.name(), "explorer.mode contains Edit",
        |z| z["explorer"]["mode"].as_str().map_or(false, |s| s.contains("Edit")), 5);

    // Cursor starts on w:left (first action weight field)
    // Press right to increase it (step 0.5)
    common::press(guard.name(), &["right"]);
    common::settle();

    let editor = common::zoltar_pane(guard.name(), "explorer.editor", "data");
    let w_left = editor["fields"].as_array().unwrap().iter()
        .find(|f| f["label"].as_str() == Some("w:left")).unwrap();
    // w:left was 1.0 in the test behavior, after +0.5 should be 1.5
    assert_eq!(w_left["value"].as_f64().unwrap(), 1.5,
        "After right arrow on w:left, should be 1.5: {}", w_left);
    assert!(editor["dirty"].as_bool().unwrap(),
        "Should show modified (dirty) after edit");
}

#[test]

fn airbus_explorer_duplicate_scenario() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture_with_behavior(dir.path(), "explorer-dup", &[]);

    // Open explorer, focus scenarios, duplicate scenario 0
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "scenarios", 5);
    common::press(guard.name(), &["s"]);
    common::press(guard.name(), &["c"]);
    common::wait_for_toast(guard.name(), "duplicated", 5);

    let behaviors = common::zoltar_pane(guard.name(), "explorer.behaviors", "data");
    let selected = behaviors["selected"].as_u64().unwrap() as usize;
    let beh = &behaviors["behaviors"].as_array().unwrap()[selected];
    assert_eq!(beh["n_scenarios"].as_u64().unwrap(), 3,
        "After duplicate, behavior should have 3 scenarios: {}", beh);
}

#[test]

fn airbus_explorer_edit_sets_edited_flag() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture_with_behavior(dir.path(), "explorer-edited", &[]);

    // Open explorer — scenario 0 should NOT be marked as edited
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "scenarios", 5);

    let before = common::zoltar_pane(guard.name(), "explorer.scenarios", "data");
    let scenario0 = &before["scenarios"].as_array().unwrap()[0];
    assert!(!scenario0["edited"].as_bool().unwrap_or(true),
        "Before edit, scenario 0 should not be edited: {}", scenario0);

    // Edit and save scenario 0
    common::press(guard.name(), &["e"]);
    common::wait_for_text(guard.name(), "w:left", 5);
    common::press(guard.name(), &["enter"]);
    common::wait_for_toast(guard.name(), "Saved", 5);

    let after = common::zoltar_pane(guard.name(), "explorer.scenarios", "data");
    let scenario0 = &after["scenarios"].as_array().unwrap()[0];
    assert!(scenario0["edited"].as_bool().unwrap(),
        "After save, scenario 0 should be edited: {}", scenario0);
}

// ─── Phase 7: Behavior Scenario Capture Modal ────────────────────────────────

#[test]

fn airbus_b_opens_behavior_modal() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "b-modal-open", &[]);
    common::wait_for_text(guard.name(), "Timestep", 5);

    // Press b to open behavior scenario modal
    common::press(guard.name(), &["b"]);
    common::wait_for_text(guard.name(), "Add Behavior Scenario", 5);

    let screen = common::dump_text(guard.name());
    assert!(screen.contains("Add Behavior Scenario"),
        "b should open the behavior scenario modal:\n{}", screen);
    assert!(screen.contains("Actions"),
        "Modal should show Actions section:\n{}", screen);
    assert!(screen.contains("LEFT"),
        "Modal should show LEFT action:\n{}", screen);
    assert!(screen.contains("IDLE"),
        "Modal should show IDLE action:\n{}", screen);
}

#[test]

fn airbus_b_modal_shows_all_actions() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "b-modal-actions", &[]);
    common::wait_for_text(guard.name(), "Timestep", 5);

    common::press(guard.name(), &["b"]);
    common::wait_for_text(guard.name(), "Add Behavior Scenario", 5);

    let screen = common::dump_text(guard.name());
    for action in &["LEFT", "IDLE", "RIGHT", "FASTER", "SLOWER"] {
        assert!(screen.contains(action),
            "Modal should show {} action:\n{}", action, screen);
    }
}

#[test]

fn airbus_b_modal_esc_closes() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "b-modal-esc", &[]);
    common::wait_for_genco(guard.name(), "frame_index == 0", |z| z["frame_index"] == 0, 5);

    // Open modal
    common::press(guard.name(), &["b"]);
    common::wait_for_text(guard.name(), "Add Behavior Scenario", 5);

    // Close with Esc (settle first to ensure modal input handler is active)
    common::settle();
    common::press(guard.name(), &["escape"]);
    common::wait_for_text_absent(guard.name(), "Add Behavior Scenario", 3);

    // Verify normal UI is back via genco
    let genco = common::zoltar_genco(guard.name());
    assert!(genco["screen"].as_str().unwrap().contains("Browse"),
        "Should be back in Browse screen: {}", genco["screen"]);
}

#[test]

fn airbus_b_modal_space_toggles_action() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "b-modal-toggle", &[]);
    common::wait_for_text(guard.name(), "Timestep", 5);

    common::press(guard.name(), &["b"]);
    common::wait_for_text(guard.name(), "Add Behavior Scenario", 5);
    common::settle();

    // First Space toggles 0 → 1 (LEFT, since cursor starts at 0)
    common::press(guard.name(), &["space"]);
    common::wait_for_any_text(guard.name(), &["1]", "1_"], 5);

    // Second Space toggles 1 → 0
    common::press(guard.name(), &["space"]);
    // Wait for the weight to revert — cursor shows _ for empty field
    common::wait_for_any_text(guard.name(), &["[     ]", "[    _]"], 5);
}

#[test]

fn airbus_b_modal_tab_cycles_fields() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "b-modal-tab", &[]);
    common::wait_for_text(guard.name(), "Timestep", 5);

    common::press(guard.name(), &["b"]);
    common::wait_for_text(guard.name(), "Add Behavior Scenario", 5);

    // Initially focus is on Actions (cursor next to LEFT)
    let screen1 = common::dump_text(guard.name());

    // Tab moves to Name field — wait for the focus marker to appear on Behavior
    common::press(guard.name(), &["tab"]);
    common::wait_for_text(guard.name(), "\u{25BA} Behavior", 3);
    let screen2 = common::dump_text(guard.name());

    // Tab wraps back to Actions — wait for cursor to return to first action
    common::press(guard.name(), &["tab"]);
    common::wait_for_text(guard.name(), "_] LEFT", 5);
    let screen3 = common::dump_text(guard.name());

    // The screens should differ (focus indicator moves)
    assert_ne!(screen1, screen2, "Tab should change focus from Actions to Name");
    // After wrapping back, screen should look like screen1 again (cursor on Actions)
    assert_ne!(screen2, screen3, "Tab should change focus from Name back to Actions");
}

#[test]

fn airbus_b_modal_no_action_validation() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "b-no-action", &[]);
    common::wait_for_text(guard.name(), "Timestep", 5);

    // Open modal, Tab to Name, type name, Enter (no action selected)
    common::press(guard.name(), &["b"]);
    common::wait_for_text(guard.name(), "Add Behavior Scenario", 5);
    common::press(guard.name(), &["tab"]);
    common::settle();
    common::type_text(guard.name(), "test");
    common::settle();
    common::press(guard.name(), &["enter"]);

    // Should show validation toast
    common::wait_for_toast(guard.name(), "Select at least one action", 3);
}

#[test]

fn airbus_b_modal_no_name_validation() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "b-no-name", &[]);
    common::wait_for_text(guard.name(), "Timestep", 5);

    // Open modal, toggle an action but don't enter a name
    common::press(guard.name(), &["b"]);
    common::wait_for_text(guard.name(), "Add Behavior Scenario", 5);

    // Toggle LEFT action, then press Enter
    common::press(guard.name(), &["space"]);
    common::settle();
    common::press(guard.name(), &["enter"]);

    // Should show validation toast for missing name
    common::wait_for_toast(guard.name(), "name is required", 3);
}

#[test]

fn airbus_b_modal_save_creates_file() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let name = session("b-save");
    let home = common::isolated_home(&name);
    common::spawn_octane(dir.path(), &name, &[]);
    common::wait_for_text(&name, "epoch", 5);
    let guard = SessionGuard(name.clone());

    // Open modal
    common::press(guard.name(), &["b"]);
    common::wait_for_text(guard.name(), "Add Behavior Scenario", 5);

    // Toggle LEFT action (cycles to +1)
    common::press(guard.name(), &["space"]);
    common::settle();

    // Tab to Name field, type behavior name
    common::press(guard.name(), &["tab"]);
    common::settle();
    common::type_text(guard.name(), "mytest");
    common::settle();

    // Enter saves (from Name field with no suggestion selected)
    common::press(guard.name(), &["enter"]);
    common::wait_for_toast(guard.name(), "Scenario added", 5);

    // Verify the behavior JSON was created (read from sandbox overlay)
    let sandboxed_path = format!("{}/.highjax/behaviors/highway/mytest.json", home);
    let behavior_path = common::overlay_path(guard.name(), &sandboxed_path);
    assert!(behavior_path.exists(),
        "Behavior file should have been created at {:?}", behavior_path);

    // Read and verify JSON contents
    let content = std::fs::read_to_string(&behavior_path).unwrap();
    let data: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert_eq!(data["name"], "mytest");
    let scenarios = data["scenarios"].as_array().unwrap();
    assert_eq!(scenarios.len(), 1, "Should have 1 scenario");

    let scenario = &scenarios[0];
    // New format: action_weights object instead of actions list + weight scalar
    let aw = scenario["action_weights"].as_object().unwrap();
    assert_eq!(aw.len(), 1, "Should have 1 action weight entry");
    assert_eq!(aw["left"].as_i64().unwrap(), 1,
        "Selected action 'left' should have weight 1");

    // Verify state has flat format ego and NPC keys
    let state = &scenario["state"];
    assert!(state["ego_speed"].as_f64().is_some(), "Should have ego_speed");
    assert!(state["ego_x"].as_f64().is_some(), "Should have ego_x");
    assert!(state["ego_y"].as_f64().is_some(), "Should have ego_y");
    // Simple trek has 2 NPCs
    assert!(state["npc0_x"].as_f64().is_some(), "Should have npc0_x");
    assert!(state["npc0_y"].as_f64().is_some(), "Should have npc0_y");
    assert!(state["npc0_speed"].as_f64().is_some(), "Should have npc0_speed");
    assert!(state["npc1_x"].as_f64().is_some(), "Should have npc1_x");

    // Verify source info
    let source = &scenario["source"];
    assert_eq!(source["epoch"], 0);
    assert_eq!(source["episode"], 0);
    assert_eq!(source["t"], 0.0);
}

#[test]

fn airbus_b_modal_correct_npc_geometry() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let name = session("b-geometry");
    let home = common::isolated_home(&name);
    common::spawn_octane(dir.path(), &name, &[]);
    common::wait_for_text(&name, "epoch", 5);
    let guard = SessionGuard(name.clone());

    // Save a scenario at t=0
    common::press(guard.name(), &["b"]);
    common::wait_for_text(guard.name(), "Add Behavior Scenario", 5);

    // Toggle IDLE (move cursor down from LEFT to IDLE, then toggle)
    common::press(guard.name(), &["down"]);
    common::press(guard.name(), &["space"]);
    common::settle();

    // Tab to Name, type name, save
    common::press(guard.name(), &["tab"]);
    common::settle();
    common::type_text(guard.name(), "geom");
    common::settle();
    common::press(guard.name(), &["enter"]);
    common::wait_for_toast(guard.name(), "Scenario added", 5);

    // Read JSON and verify NPC geometry matches fixture data (from sandbox overlay)
    let sandboxed_path = format!("{}/.highjax/behaviors/highway/geom.json", home);
    let behavior_path = common::overlay_path(guard.name(), &sandboxed_path);
    let content = std::fs::read_to_string(&behavior_path).unwrap();
    let data: serde_json::Value = serde_json::from_str(&content).unwrap();

    let scenario = &data["scenarios"][0];
    let state = &scenario["state"];

    // At t=0: ego=(0,2,speed=20), NPC0=(30,2,speed=18), NPC1=(15,6,speed=22)
    assert_eq!(state["ego_speed"], 20.0);
    assert_eq!(state["ego_x"], 0.0);
    assert_eq!(state["ego_y"], 2.0);

    // NPC0: absolute (30, 2), speed 18
    assert_eq!(state["npc0_x"], 30.0);
    assert_eq!(state["npc0_y"], 2.0);
    assert_eq!(state["npc0_speed"], 18.0);
    // NPC1: absolute (15, 6), speed 22
    assert_eq!(state["npc1_x"], 15.0);
    assert_eq!(state["npc1_y"], 6.0);
    assert_eq!(state["npc1_speed"], 22.0);

    // Verify action uses new action_weights format (lowercase key)
    let aw = scenario["action_weights"].as_object().unwrap();
    assert_eq!(aw["idle"].as_i64().unwrap(), 1);
}

#[test]

fn airbus_b_modal_multiple_actions() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let name = session("b-multi-act");
    let home = common::isolated_home(&name);
    common::spawn_octane(dir.path(), &name, &[]);
    common::wait_for_text(&name, "epoch", 5);
    let guard = SessionGuard(name.clone());

    common::press(guard.name(), &["b"]);
    common::wait_for_text(guard.name(), "Add Behavior Scenario", 5);

    // Toggle LEFT (cursor starts here) — cycles to +1
    common::press(guard.name(), &["space"]);
    // Move to RIGHT (index 2) and toggle to +1
    common::press_with_delay(guard.name(), &["down", "down"], 100);
    // Wait for cursor to reach RIGHT (active cursor shows "_" input indicator)
    common::wait_for_text(guard.name(), "_] RIGHT", 5);
    common::press(guard.name(), &["space"]);

    // Tab to name, type, save
    common::press(guard.name(), &["tab"]);
    // Wait for focus to move to Name field
    common::wait_for_text(guard.name(), "\u{25BA} Behavior", 5);
    common::type_text(guard.name(), "multi");
    common::settle();
    common::press(guard.name(), &["enter"]);
    common::wait_for_toast(guard.name(), "Scenario added", 5);

    // Verify both actions saved in action_weights format (from sandbox overlay)
    let sandboxed_path = format!("{}/.highjax/behaviors/highway/multi.json", home);
    let behavior_path = common::overlay_path(guard.name(), &sandboxed_path);
    let content = std::fs::read_to_string(&behavior_path).unwrap();
    let data: serde_json::Value = serde_json::from_str(&content).unwrap();
    let aw = data["scenarios"][0]["action_weights"].as_object().unwrap();
    assert_eq!(aw.len(), 2, "Should have 2 entries in action_weights");
    assert_eq!(aw["left"].as_i64().unwrap(), 1, "left should have weight 1");
    assert_eq!(aw["right"].as_i64().unwrap(), 1, "right should have weight 1");
}

#[test]

fn airbus_b_modal_appends_scenario() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let name = session("b-append");
    let home = common::isolated_home(&name);
    common::spawn_octane(dir.path(), &name, &[]);
    common::wait_for_text(&name, "epoch", 5);
    let guard = SessionGuard(name.clone());

    // Save first scenario at t=0
    common::press(guard.name(), &["b"]);
    common::wait_for_text(guard.name(), "Add Behavior Scenario", 5);
    common::press(guard.name(), &["space"]); // toggle LEFT to +1
    common::press(guard.name(), &["tab"]);
    common::settle();
    common::type_text(guard.name(), "append");
    common::settle();
    common::press(guard.name(), &["enter"]);
    common::wait_for_toast(guard.name(), "Scenario added", 5);

    // Advance to t=1
    common::press(guard.name(), &["k"]);
    common::wait_for_text(guard.name(), "Timestep1/", 5);

    // Save second scenario (name "append" still remembered from last time)
    common::press(guard.name(), &["b"]);
    common::wait_for_text(guard.name(), "Add Behavior Scenario", 5);
    // Toggle IDLE (move down first)
    common::press(guard.name(), &["down"]);
    common::press(guard.name(), &["space"]);
    common::settle();
    // Tab past Name, Enter from Name field
    common::press(guard.name(), &["tab"]);
    common::settle();
    common::press(guard.name(), &["enter"]);
    common::wait_for_toast(guard.name(), "Scenario added", 5);

    // Verify both scenarios in file (from sandbox overlay)
    let sandboxed_path = format!("{}/.highjax/behaviors/highway/append.json", home);
    let behavior_path = common::overlay_path(guard.name(), &sandboxed_path);
    let content = std::fs::read_to_string(&behavior_path).unwrap();
    let data: serde_json::Value = serde_json::from_str(&content).unwrap();
    let scenarios = data["scenarios"].as_array().unwrap();
    assert_eq!(scenarios.len(), 2, "Should have 2 scenarios after appending");

    // First scenario was at t=0 with LEFT (+1)
    assert_eq!(scenarios[0]["source"]["t"], 0.0);
    assert_eq!(scenarios[0]["action_weights"]["left"].as_i64().unwrap(), 1);
    // Second scenario was at t=1 with IDLE (+1)
    assert_eq!(scenarios[1]["source"]["t"], 1.0);
    assert_eq!(scenarios[1]["action_weights"]["idle"].as_i64().unwrap(), 1);
}

#[test]

fn airbus_b_modal_negative_weight() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let name = session("b-negweight");
    let home = common::isolated_home(&name);
    common::spawn_octane(dir.path(), &name, &[]);
    common::wait_for_text(&name, "epoch", 5);
    let guard = SessionGuard(name.clone());

    common::press(guard.name(), &["b"]);
    common::wait_for_text(guard.name(), "Add Behavior Scenario", 5);

    // Type "-1" to set LEFT weight to -1 (use -- to prevent CLI flag parsing)
    common::type_text_raw(guard.name(), "-1");
    common::wait_for_text(guard.name(), "-1", 5);

    // Tab to Name, type name, save
    common::press(guard.name(), &["tab"]);
    common::settle();
    common::type_text(guard.name(), "negweighted");
    common::settle();
    common::press(guard.name(), &["enter"]);
    common::wait_for_toast(guard.name(), "Scenario added", 5);

    // Verify negative weight in JSON (from sandbox overlay)
    let sandboxed_path = format!("{}/.highjax/behaviors/highway/negweighted.json", home);
    let behavior_path = common::overlay_path(guard.name(), &sandboxed_path);
    let content = std::fs::read_to_string(&behavior_path).unwrap();
    let data: serde_json::Value = serde_json::from_str(&content).unwrap();
    let aw = data["scenarios"][0]["action_weights"].as_object().unwrap();
    assert_eq!(aw["left"].as_i64().unwrap(), -1,
        "left should have weight -1");
}

#[test]

fn airbus_b_modal_shows_source_info() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "b-source-info", &["--timestep", "3"]);
    common::wait_for_genco(guard.name(), "frame_index == 3", |z| z["frame_index"] == 3, 5);

    // Verify genco state matches expected position
    let genco = common::zoltar_genco(guard.name());
    assert_eq!(genco["epoch"], 0, "Should be at epoch 0");
    assert_eq!(genco["episode"], 0, "Should be at episode 0");
    assert_eq!(genco["frame_index"], 3, "Should be at frame 3");

    // Open modal and verify it shows the source info visually
    common::press(guard.name(), &["b"]);
    common::wait_for_text(guard.name(), "Add Behavior Scenario", 5);

    let screen = common::dump_text(guard.name());
    assert!(screen.contains("Epoch: 0"),
        "Modal should show current epoch:\n{}", screen);
    assert!(screen.contains("Episode: 0"),
        "Modal should show current episode:\n{}", screen);
    assert!(screen.contains("t: 3"),
        "Modal should show current t value:\n{}", screen);
}

#[test]

fn airbus_b_modal_suggestion_autocomplete() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    // Pre-create a behavior file so suggestions have something to show
    let name = session("b-suggest");
    let home = common::isolated_home(&name);
    let behaviors_dir = format!("{}/.highjax/behaviors/highway", home);
    std::fs::create_dir_all(&behaviors_dir).unwrap();
    std::fs::write(
        format!("{}/existing-beh.json", behaviors_dir),
        serde_json::json!({
            "name": "existing-beh",
            "description": "",
            "scenarios": []
        }).to_string(),
    ).unwrap();

    common::spawn_octane(dir.path(), &name, &[]);
    common::wait_for_text(&name, "epoch", 5);
    let guard = SessionGuard(name);

    // Open modal, Tab to Name field
    common::press(guard.name(), &["b"]);
    common::wait_for_text(guard.name(), "Add Behavior Scenario", 5);
    common::press(guard.name(), &["tab"]);
    // Wait for focus to move to Name field (shown as "► Behavior")
    common::wait_for_text(guard.name(), "\u{25BA} Behavior", 5);

    let screen = common::dump_text(guard.name());
    // With empty name and focus on Name field, all suggestions should show
    assert!(screen.contains("existing-beh"),
        "Suggestion autocomplete should show existing behavior name:\n{}", screen);
}

#[test]

fn airbus_b_modal_at_different_timestep() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let name = session("b-t5");
    let home = common::isolated_home(&name);
    common::spawn_octane(dir.path(), &name, &["--timestep", "5"]);
    common::wait_for_text(&name, "Timestep5/", 5);
    let guard = SessionGuard(name.clone());

    // Save scenario at t=5
    common::press(guard.name(), &["b"]);
    common::wait_for_text(guard.name(), "Add Behavior Scenario", 5);
    common::press(guard.name(), &["space"]); // toggle LEFT
    common::press(guard.name(), &["tab"]);
    common::settle();
    common::type_text(guard.name(), "tfive");
    common::settle();
    common::press(guard.name(), &["enter"]);
    common::wait_for_toast(guard.name(), "Scenario added", 5);

    // Verify source t=5 and NPC geometry at t=5 (from sandbox overlay)
    let sandboxed_path = format!("{}/.highjax/behaviors/highway/tfive.json", home);
    let behavior_path = common::overlay_path(guard.name(), &sandboxed_path);
    let content = std::fs::read_to_string(&behavior_path).unwrap();
    let data: serde_json::Value = serde_json::from_str(&content).unwrap();
    let scenario = &data["scenarios"][0];
    assert_eq!(scenario["source"]["t"], 5.0);

    // At t=5: ego_x = 5*2 = 10, ego_y = 2
    // NPC0: absolute (40, 2), speed=18
    // NPC1: absolute (25, 6), speed=22
    let state = &scenario["state"];
    assert_eq!(state["ego_x"], 10.0);
    assert_eq!(state["ego_y"], 2.0);
    assert_eq!(state["npc0_x"], 40.0);
    assert_eq!(state["npc0_y"], 2.0);
    assert_eq!(state["npc1_x"], 25.0);
    assert_eq!(state["npc1_y"], 6.0);
}

#[test]

fn airbus_b_modal_dismiss_preserves_ui() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "b-dismiss", &[]);
    common::wait_for_text(guard.name(), "Timestep", 5);

    // Open and dismiss modal (settle to ensure modal input handler is active)
    common::press(guard.name(), &["b"]);
    common::wait_for_text(guard.name(), "Add Behavior Scenario", 5);
    common::settle();
    common::press(guard.name(), &["escape"]);
    common::wait_for_text_absent(guard.name(), "Add Behavior Scenario", 3);

    let after = common::dump_text(guard.name());
    assert!(after.contains("Timestep"),
        "Timestep info should be visible:\n{}", after);
    assert!(after.contains("epoch"),
        "Epoch pane should be visible:\n{}", after);
}

#[test]

fn airbus_b_modal_new_file_has_action_names() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let name = session("b-action-names");
    let home = common::isolated_home(&name);
    common::spawn_octane(dir.path(), &name, &[]);
    common::wait_for_text(&name, "epoch", 5);
    let guard = SessionGuard(name.clone());

    // Save a scenario to create a new behavior file
    common::press(guard.name(), &["b"]);
    common::wait_for_text(guard.name(), "Add Behavior Scenario", 5);
    common::press(guard.name(), &["space"]); // toggle LEFT
    common::press(guard.name(), &["tab"]);
    common::settle();
    common::type_text(guard.name(), "actnames");
    common::settle();
    common::press(guard.name(), &["enter"]);
    common::wait_for_toast(guard.name(), "Scenario added", 5);

    // Verify the new file has action_names field (from sandbox overlay)
    let sandboxed_path = format!("{}/.highjax/behaviors/highway/actnames.json", home);
    let behavior_path = common::overlay_path(guard.name(), &sandboxed_path);
    let content = std::fs::read_to_string(&behavior_path).unwrap();
    let data: serde_json::Value = serde_json::from_str(&content).unwrap();
    let action_names = data["action_names"].as_array().unwrap();
    assert_eq!(action_names.len(), 5);
    assert_eq!(action_names[0], "left");
    assert_eq!(action_names[1], "idle");
    assert_eq!(action_names[2], "right");
    assert_eq!(action_names[3], "faster");
    assert_eq!(action_names[4], "slower");
}

// ─── Phase 8: Parquets Pane ─────────────────────────────────────────────────

#[test]

fn airbus_parquets_pane_shows_sample_es() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "parquets-sample", &[]);
    let parquets = common::zoltar_pane(guard.name(), "parquets", "data");
    let pq_list = parquets["parquets"].as_array().unwrap();
    assert!(pq_list.iter().any(|p| p["display"].as_str() == Some("sample_es")),
        "Parquets should include sample_es: {}", parquets);
}

#[test]

fn airbus_parquets_pane_lists_breakdown() {
    let dir = TempDir::new().unwrap();
    common::generate_breakdown_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "parquets-breakdown", &[]);
    let parquets = common::zoltar_pane(guard.name(), "parquets", "data");
    let pq_list = parquets["parquets"].as_array().unwrap();
    assert!(pq_list.iter().any(|p| p["display"].as_str() == Some("sample_es")),
        "sample_es should be listed: {}", parquets);
    assert!(pq_list.iter().any(|p| p["display"].as_str().map_or(false, |d| d.contains("test-run"))),
        "Breakdown run 'test-run' should be listed: {}", parquets);
}

#[test]

fn airbus_parquets_switch_updates_epochs() {
    let dir = TempDir::new().unwrap();
    common::generate_breakdown_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "parquets-switch", &[]);
    common::wait_for_text(guard.name(), "sample_es", 5);

    let before = common::zoltar_pane(guard.name(), "epochs", "data");
    let before_count = before["epochs"].as_array().unwrap().len();

    // Focus parquets pane, navigate down to select test-run, press Enter
    common::press(guard.name(), &["a"]);  // mnemonic for Parquets focus
    common::settle();
    common::press(guard.name(), &["down"]);
    common::settle();
    common::press(guard.name(), &["enter"]);
    common::settle();

    let after = common::zoltar_pane(guard.name(), "epochs", "data");
    let after_count = after["epochs"].as_array().unwrap().len();

    // Simple trek has 1 epoch, breakdown has 2 epochs
    assert_ne!(before_count, after_count,
        "Switching parquet source should change epoch count: before={}, after={}",
        before_count, after_count);
}

#[test]

fn airbus_parquets_mnemonic_a() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "parquets-mnemonic", &[]);
    common::wait_for_text(guard.name(), "Timestep", 5);

    // Press 'a' to focus parquets pane
    common::press(guard.name(), &["a"]);
    common::settle();

    let parquets = common::zoltar_pane(guard.name(), "parquets", "data");
    let pq_list = parquets["parquets"].as_array().unwrap();
    assert!(pq_list.iter().any(|p| p["display"].as_str() == Some("sample_es")),
        "sample_es should be accessible after 'a': {}", parquets);
}

#[test]

fn airbus_parquets_cli_flag() {
    let dir = TempDir::new().unwrap();
    common::generate_breakdown_trek(dir.path());

    // Use -t with the breakdown es.parquet path to pre-select the breakdown source
    let es_path = dir.path().join("breakdown/test-run/es.parquet");
    let guard = spawn_fixture(&es_path, "parquets-cli", &[]);
    common::wait_for_text(guard.name(), "Timestep", 5);

    let parquets = common::zoltar_pane(guard.name(), "parquets", "data");
    let selected = parquets["selected"].as_u64().unwrap() as usize;
    let pq_list = parquets["parquets"].as_array().unwrap();
    let selected_display = pq_list[selected]["display"].as_str().unwrap();
    assert!(selected_display.contains("test-run"),
        "Selected parquet should be test-run, got: {}", selected_display);
}

// ─── Phase 9: Scenario → Source Navigation ──────────────────────────────────

#[test]

fn airbus_explorer_enter_navigates_to_source() {
    let dir = TempDir::new().unwrap();
    common::generate_crash_trek(dir.path());

    // Spawn with a sourced behavior targeting the crash frame (epoch=0, ep=1, t=8)
    let name = session("explorer-enter-src");
    common::write_crash_source_behavior(&name);
    common::spawn_octane(dir.path(), &name, &["--prefs", "npc_text=always"]);
    common::wait_for_text(&name, "epoch", 5);
    let guard = SessionGuard(name);

    // Path to frame state JSON (written by 'y' key), resolved through sandbox overlay
    let state_sandboxed = format!("{}/octane-frame-state.json", dir.path().display());

    // Open explorer
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "sourced-behavior", 5);

    // Tab to Scenarios pane
    common::press(guard.name(), &["tab"]);
    common::settle();

    // Dump the behavior scenario's frame state via 'y' key
    common::type_text(guard.name(), "y");
    common::wait_for_toast(guard.name(), "state dumped", 3);
    let behavior_state: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(common::overlay_path(guard.name(), &state_sandboxed)).unwrap()
    ).unwrap();

    // Press Enter on scenario 0 (source: epoch=0, episode=1, t=8 — crashed)
    common::press(guard.name(), &["enter"]);
    // Should leave explorer and enter Runs tab
    common::wait_for_text_absent(guard.name(), "scenarios", 5);
    // Wait for the crash frame to render
    common::wait_for_text(guard.name(), "8/9", 5);

    // Dump the Runs tab frame state
    common::type_text(guard.name(), "y");
    common::wait_for_toast(guard.name(), "state dumped", 3);
    let runs_state: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(common::overlay_path(guard.name(), &state_sandboxed)).unwrap()
    ).unwrap();

    let screen = common::dump_text(guard.name());

    // Correct timestep
    assert!(screen.contains("8/9"),
        "Should navigate to source timestep 8:\n{}", screen);
    // Correct episode: only epoch=0, episode=1 has crash reward -50
    assert!(screen.contains("-50.0"),
        "Should show crash reward -50.0 (epoch=0, ep=1, t=8):\n{}", screen);
    // Explorer panes should be gone (back in browse mode)
    assert!(!screen.contains("scenarios"),
        "Should have left explorer:\n{}", screen);

    // Compare frame states: behavior scenario and Runs tab should show same vehicles
    assert_eq!(behavior_state["ego"]["speed"], runs_state["ego"]["speed"],
        "Ego speed should match between behavior and runs");
    assert_eq!(behavior_state["npcs"].as_array().unwrap().len(),
               runs_state["npcs"].as_array().unwrap().len(),
        "NPC count should match between behavior and runs");
    for (i, (b_npc, r_npc)) in behavior_state["npcs"].as_array().unwrap().iter()
        .zip(runs_state["npcs"].as_array().unwrap().iter()).enumerate()
    {
        assert_eq!(b_npc["x"], r_npc["x"],
            "NPC {} x should match", i);
        assert_eq!(b_npc["y"], r_npc["y"],
            "NPC {} y should match", i);
        assert_eq!(b_npc["speed"], r_npc["speed"],
            "NPC {} speed should match", i);
    }
}

#[test]

fn airbus_explorer_enter_no_source_shows_toast() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture_with_sourced_behavior(dir.path(), "explorer-enter-nosrc", &[]);

    // Open explorer
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "sourced-behavior", 5);

    // Tab to Scenarios, navigate to scenario 1 (no source info)
    common::press(guard.name(), &["tab"]);
    common::settle();
    common::press(guard.name(), &["down"]);
    common::settle();

    // Press Enter — should show toast, not navigate
    common::press(guard.name(), &["enter"]);
    common::wait_for_toast(guard.name(), "No source info", 3);

    let screen = common::dump_text(guard.name());
    // Should still be in explorer
    assert!(screen.contains("scenarios"),
        "Should remain in explorer after no-source Enter:\n{}", screen);
}

#[test]

fn airbus_explorer_enter_source_footer_hint() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture_with_sourced_behavior(dir.path(), "explorer-footer-src", &[]);

    // Open explorer
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "scenarios", 5);

    // Tab to Scenarios pane so the footer shows scenario-specific hints
    common::press(guard.name(), &["tab"]);
    common::settle();

    let screen = common::dump_text(guard.name());
    // Footer should show "Enter" and "Source" hints
    assert!(screen.contains("Source"),
        "Footer should show Source hint when Scenarios pane focused:\n{}", screen);
}

// ─── Phase 9b: source.target parsing and navigation ─────────────────────────

#[test]

fn airbus_explorer_source_target_parsed() {
    let dir = TempDir::new().unwrap();
    common::generate_crash_trek(dir.path());

    let name = session("src-target-parsed");
    common::write_target_source_behavior(&name, dir.path());
    common::spawn_octane(dir.path(), &name, &[]);
    common::wait_for_text(&name, "epoch", 5);
    let guard = SessionGuard(name);

    // Open explorer, tab to scenarios
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "sourced-behavior", 5);
    common::press(guard.name(), &["tab"]);
    common::settle();

    // Query zoltar for scenario data — source_target should be the trek path
    let data = common::zoltar_pane(guard.name(), "explorer.scenarios", "data");
    let scenario_0 = &data["scenarios"][0];
    assert!(scenario_0["source_target"].is_string(),
        "source_target should be parsed from 'target' key: {:?}", scenario_0);
    assert_eq!(scenario_0["source_epoch"], 0);
    assert_eq!(scenario_0["source_episode"], 1);
    assert_eq!(scenario_0["source_t"], 8.0);

    // Scenario 1 has no source
    let scenario_1 = &data["scenarios"][1];
    assert!(scenario_1["source_target"].is_null(),
        "Scenario without source should have null source_target: {:?}", scenario_1);
}

#[test]

fn airbus_explorer_source_legacy_trek_key() {
    let dir = TempDir::new().unwrap();
    common::generate_crash_trek(dir.path());

    let name = session("src-legacy-trek");
    common::write_legacy_trek_source_behavior(&name, dir.path());
    common::spawn_octane(dir.path(), &name, &[]);
    common::wait_for_text(&name, "epoch", 5);
    let guard = SessionGuard(name);

    // Open explorer, tab to scenarios
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "sourced-behavior", 5);
    common::press(guard.name(), &["tab"]);
    common::settle();

    // Query zoltar — source_target should be parsed from legacy "trek" key
    let data = common::zoltar_pane(guard.name(), "explorer.scenarios", "data");
    let scenario_0 = &data["scenarios"][0];
    assert!(scenario_0["source_target"].is_string(),
        "source_target should be parsed from legacy 'trek' key: {:?}", scenario_0);

    // Navigate should still work
    common::press(guard.name(), &["enter"]);
    common::wait_for_text_absent(guard.name(), "scenarios", 5);
    common::wait_for_text(guard.name(), "8/9", 5);

    let screen = common::dump_text(guard.name());
    assert!(screen.contains("-50.0"),
        "Should navigate to crash frame (reward -50.0):\n{}", screen);
}

#[test]

fn airbus_explorer_navigate_with_target_trek_dir() {
    let dir = TempDir::new().unwrap();
    common::generate_crash_trek(dir.path());

    let name = session("src-target-trek");
    common::write_target_source_behavior(&name, dir.path());
    common::spawn_octane(dir.path(), &name, &["--prefs", "npc_text=always"]);
    common::wait_for_text(&name, "epoch", 5);
    let guard = SessionGuard(name);

    // Open explorer, tab to scenarios
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "sourced-behavior", 5);
    common::press(guard.name(), &["tab"]);
    common::settle();

    // Navigate to source (epoch=0, episode=1, t=8 — crash frame)
    common::press(guard.name(), &["enter"]);
    common::wait_for_text_absent(guard.name(), "scenarios", 5);
    common::wait_for_text(guard.name(), "8/9", 5);

    let screen = common::dump_text(guard.name());
    assert!(screen.contains("-50.0"),
        "Should navigate to crash frame (reward -50.0):\n{}", screen);
    assert!(!screen.contains("scenarios"),
        "Should have left explorer:\n{}", screen);
}

#[test]

fn airbus_explorer_navigate_with_target_pq_file() {
    let dir = TempDir::new().unwrap();
    common::generate_crash_trek(dir.path());

    let name = session("src-target-pq");
    common::write_pq_target_source_behavior(&name, dir.path());
    common::spawn_octane(dir.path(), &name, &["--prefs", "npc_text=always"]);
    common::wait_for_text(&name, "epoch", 5);
    let guard = SessionGuard(name);

    // Open explorer, tab to scenarios
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "sourced-behavior", 5);
    common::press(guard.name(), &["tab"]);
    common::settle();

    // Navigate to source — target is a .parquet file, should resolve trek and navigate
    common::press(guard.name(), &["enter"]);
    common::wait_for_text_absent(guard.name(), "scenarios", 5);
    common::wait_for_text(guard.name(), "8/9", 5);

    let screen = common::dump_text(guard.name());
    assert!(screen.contains("-50.0"),
        "Should navigate to crash frame (reward -50.0):\n{}", screen);
}

// ─── Phase 10: Deferred timestep lookup ─────────────────────────────────────

#[test]

fn airbus_timestep_breakdown_parquet() {
    let dir = TempDir::new().unwrap();
    common::generate_breakdown_trek(dir.path());

    // Switch to breakdown parquet and navigate to t=4 (last frame of 5-frame episode)
    let es_path = dir.path().join("breakdown/test-run/es.parquet");
    let guard = spawn_fixture(
        &es_path,
        "ts-breakdown",
        &["--epoch", "0", "--episode", "0", "--timestep", "4"],
    );
    common::wait_for_genco(guard.name(), "frame_index == 4",
        |z| z["frame_index"] == 4, 5);
}

#[test]

fn airbus_timestep_out_of_range_toast() {
    let dir = TempDir::new().unwrap();
    common::generate_crash_trek(dir.path());

    // Request t=999 on a 10-frame episode (t=0..9) — should show warning toast
    // crash_trek default loads epoch 1; --epoch 0 forces a reload through binary search
    let guard = spawn_fixture(
        dir.path(),
        "ts-oor",
        &["--epoch", "0", "--timestep", "999"],
    );
    // Should clamp to last frame
    common::wait_for_genco(guard.name(), "frame_index == 9",
        |z| z["frame_index"] == 9, 5);
    // Toast should show "not found" warning (visual check — no zoltar query for toasts)
    common::wait_for_toast(guard.name(), "not found", 5);
}

#[test]

fn airbus_copycmd_uses_es_episode() {
    let dir = TempDir::new().unwrap();
    common::generate_sparse_episode_trek(dir.path());

    // Navigate to es_episode 20 (positional index 1), then press C to copy command
    let guard = spawn_fixture(
        dir.path(),
        "copycmd-ep",
        &["--episode", "20", "--timestep", "3"],
    );
    common::wait_for_text(guard.name(), "Timestep3/", 5);

    // Press C (CopyCmd) — toast shows the command that was copied
    common::type_text(guard.name(), "C");
    common::wait_for_toast(guard.name(), "Copied:", 5);

    let genco = common::zoltar_genco(guard.name());
    let toast = genco["last_toast"].as_str().unwrap_or("");
    // The toast should contain -e 20 (es_episode), not -e 1 (positional)
    assert!(
        toast.contains("-e 20"),
        "CopyCmd should use es_episode (20), not positional index (1):\nlast_toast: {}",
        toast
    );
}

// ─── Zoltar IPC tests ──────────────────────────────────────────────────────

#[test]
fn airbus_zoltar_ping() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "zoltar-ping", &[]);

    // Give octane a moment to start zoltar listener
    std::thread::sleep(std::time::Duration::from_millis(500));

    let (success, stdout, stderr) = common::zoltar(guard.name(), &["ping"]);
    assert!(
        success,
        "zoltar ping should succeed.\nstdout: {}\nstderr: {}",
        stdout, stderr
    );
    assert!(
        stdout.contains("pong"),
        "ping response should contain 'pong': {}",
        stdout
    );
}

#[test]
fn airbus_zoltar_genco() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "zoltar-genco", &[]);

    // Give octane a moment to start zoltar listener
    std::thread::sleep(std::time::Duration::from_millis(500));

    let (success, stdout, stderr) = common::zoltar(guard.name(), &["genco"]);
    assert!(
        success,
        "zoltar genco should succeed.\nstdout: {}\nstderr: {}",
        stdout, stderr
    );

    // Parse the JSON response
    let json: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("Failed to parse genco JSON: {}\nstdout: {}", e, stdout));

    // Check required fields exist
    assert!(json.get("screen").is_some(), "genco should have 'screen' field: {}", json);
    assert!(json.get("epoch").is_some(), "genco should have 'epoch' field: {}", json);
    assert!(json.get("n_epochs").is_some(), "genco should have 'n_epochs' field: {}", json);
    assert!(json.get("n_frames").is_some(), "genco should have 'n_frames' field: {}", json);
    assert!(json.get("playing").is_some(), "genco should have 'playing' field: {}", json);
    assert!(json.get("trek_path").is_some(), "genco should have 'trek_path' field: {}", json);

    // Check some values
    assert_eq!(json["playing"], false, "should not be playing initially");
    assert!(json["n_epochs"].as_u64().unwrap() > 0, "should have at least one epoch");
}

#[test]
fn airbus_zoltar_navigate() {
    let dir = TempDir::new().unwrap();
    common::generate_crash_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "zoltar-nav", &[]);
    std::thread::sleep(std::time::Duration::from_millis(500));

    // Navigate to epoch 1, episode 1
    let (success, stdout, stderr) = common::zoltar(
        guard.name(),
        &["navigate", "--epoch", "1", "--episode", "1"],
    );
    assert!(
        success,
        "zoltar navigate should succeed.\nstdout: {}\nstderr: {}",
        stdout, stderr
    );

    let json: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("Failed to parse navigate JSON: {}\nstdout: {}", e, stdout));

    assert_eq!(json["epoch"], 1, "should be at epoch 1: {}", json);
    assert_eq!(json["episode"], 1, "should be at episode 1: {}", json);
}

#[test]
fn airbus_zoltar_press() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "zoltar-press", &[]);
    std::thread::sleep(std::time::Duration::from_millis(500));

    // Tab twice (Epochs → Episodes → Highway), then right to advance frame
    let (success, stdout, stderr) = common::zoltar(
        guard.name(),
        &["press", "tab", "tab", "right"],
    );
    assert!(
        success,
        "zoltar press should succeed.\nstdout: {}\nstderr: {}",
        stdout, stderr
    );

    let json: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("Failed to parse press JSON: {}\nstdout: {}", e, stdout));

    // After pressing right, frame_index should have advanced from 0
    assert!(
        json["frame_index"].as_u64().unwrap() > 0,
        "frame should have advanced after pressing right: {}",
        json
    );
}

#[test]
fn airbus_zoltar_pane_epochs() {
    let dir = TempDir::new().unwrap();
    common::generate_crash_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "zoltar-pane-epochs", &[]);
    std::thread::sleep(std::time::Duration::from_millis(500));

    let (success, stdout, stderr) = common::zoltar(
        guard.name(),
        &["pane", "epochs", "data"],
    );
    assert!(success, "pane epochs data failed.\nstdout: {}\nstderr: {}", stdout, stderr);

    let json: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("Failed to parse: {}\nstdout: {}", e, stdout));

    assert!(json["selected"].is_number(), "should have 'selected': {}", json);
    let epochs = json["epochs"].as_array().expect("epochs should be array");
    assert_eq!(epochs.len(), 2, "crash trek has 2 epochs: {}", json);
    assert!(epochs[0]["number"].is_number(), "epoch should have 'number': {}", epochs[0]);
    assert!(epochs[0]["n_episodes"].is_number(), "epoch should have 'n_episodes': {}", epochs[0]);
}

#[test]
fn airbus_zoltar_pane_episodes() {
    let dir = TempDir::new().unwrap();
    common::generate_crash_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "zoltar-pane-episodes", &[]);
    std::thread::sleep(std::time::Duration::from_millis(500));

    let (success, stdout, stderr) = common::zoltar(
        guard.name(),
        &["pane", "episodes", "data"],
    );
    assert!(success, "pane episodes data failed.\nstdout: {}\nstderr: {}", stdout, stderr);

    let json: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("Failed to parse: {}\nstdout: {}", e, stdout));

    let episodes = json["episodes"].as_array().expect("episodes should be array");
    assert_eq!(episodes.len(), 2, "epoch 0 has 2 episodes: {}", json);
    assert!(episodes[0]["n_frames"].is_number(), "episode should have 'n_frames': {}", episodes[0]);
}

#[test]
fn airbus_zoltar_pane_metrics() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "zoltar-pane-metrics", &[]);
    std::thread::sleep(std::time::Duration::from_millis(500));

    let (success, stdout, stderr) = common::zoltar(
        guard.name(),
        &["pane", "metrics", "data"],
    );
    assert!(success, "pane metrics data failed.\nstdout: {}\nstderr: {}", stdout, stderr);

    let json: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("Failed to parse: {}\nstdout: {}", e, stdout));

    assert!(json.get("reward").is_some(), "metrics should have 'reward': {}", json);
    assert!(json.get("ego_speed").is_some(), "metrics should have 'ego_speed': {}", json);
}

#[test]
fn airbus_zoltar_pane_scene_svg() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "zoltar-pane-scene", &[]);
    std::thread::sleep(std::time::Duration::from_millis(500));

    let (success, stdout, stderr) = common::zoltar(
        guard.name(),
        &["pane", "scene", "svg"],
    );
    assert!(success, "pane scene svg failed.\nstdout: {}\nstderr: {}", stdout, stderr);

    let json: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("Failed to parse: {}\nstdout: {}", e, stdout));

    let svg = json["svg"].as_str().expect("should have 'svg' string field");
    assert!(svg.contains("<svg"), "SVG should start with <svg tag: {}...", &svg[..svg.len().min(100)]);
    assert!(svg.contains("</svg>"), "SVG should end with </svg>");
}

#[test]
fn airbus_zoltar_pane_unknown() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "zoltar-pane-unknown", &[]);
    std::thread::sleep(std::time::Duration::from_millis(500));

    let (success, stdout, _stderr) = common::zoltar(
        guard.name(),
        &["pane", "nonexistent", "data"],
    );
    // The command itself succeeds (airbus returns the JSON), but the response contains an error
    assert!(success, "zoltar pane should not crash");
    let json: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert!(json.get("error").is_some(), "unknown pane should return error: {}", json);
}

#[test]
fn airbus_zoltar_pane_explorer_behaviors() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "zoltar-pane-expl-beh", &[]);
    std::thread::sleep(std::time::Duration::from_millis(500));

    let (success, stdout, stderr) = common::zoltar(
        guard.name(),
        &["pane", "explorer.behaviors", "data"],
    );
    assert!(success, "pane explorer.behaviors failed.\nstdout: {}\nstderr: {}", stdout, stderr);

    let json: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("Failed to parse: {}\nstdout: {}", e, stdout));

    assert!(json["selected"].is_number(), "should have 'selected': {}", json);
    assert!(json["behaviors"].is_array(), "should have 'behaviors' array: {}", json);
}

#[test]
fn airbus_zoltar_pane_explorer_editor() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "zoltar-pane-expl-ed", &[]);
    std::thread::sleep(std::time::Duration::from_millis(500));

    let (success, stdout, stderr) = common::zoltar(
        guard.name(),
        &["pane", "explorer.editor", "data"],
    );
    assert!(success, "pane explorer.editor failed.\nstdout: {}\nstderr: {}", stdout, stderr);

    let json: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("Failed to parse: {}\nstdout: {}", e, stdout));

    assert!(json["fields"].is_array(), "should have 'fields' array: {}", json);
    assert!(json.get("dirty").is_some(), "should have 'dirty' field: {}", json);
    assert!(json.get("selected_field").is_some(), "should have 'selected_field': {}", json);
}

// ─── File IPC tests ─────────────────────────────────────────────────────────

#[test]
fn airbus_zoltar_file_ipc_ping() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "zoltar-fipc-ping", &["--zoltar-file-ipc"]);
    std::thread::sleep(std::time::Duration::from_millis(500));

    let (success, stdout, stderr) = common::zoltar(guard.name(), &["ping"]);
    assert!(
        success,
        "zoltar ping over file IPC should succeed.\nstdout: {}\nstderr: {}",
        stdout, stderr
    );
    assert!(
        stdout.contains("pong"),
        "ping response should contain 'pong': {}",
        stdout
    );
}

#[test]
fn airbus_zoltar_file_ipc_genco() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture(dir.path(), "zoltar-fipc-genco", &["--zoltar-file-ipc"]);
    common::wait_for_text(guard.name(), "epoch", 5);

    let (success, stdout, stderr) = common::zoltar(guard.name(), &["genco"]);
    assert!(
        success,
        "zoltar genco over file IPC should succeed.\nstdout: {}\nstderr: {}",
        stdout, stderr
    );

    let json: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("Failed to parse genco JSON: {}\nstdout: {}", e, stdout));
    assert!(json.get("screen").is_some(), "genco should have 'screen': {}", json);
    assert!(json.get("epoch").is_some(), "genco should have 'epoch': {}", json);
    assert!(json.get("n_frames").is_some(), "genco should have 'n_frames': {}", json);
    assert!(json.get("trek_path").is_some(), "genco should have 'trek_path': {}", json);
}

// ─── Phase 11: Explorer Action Probs ─────────────────────────────────────────

#[test]

fn airbus_explorer_action_probs_none_without_snapshot() {
    let dir = TempDir::new().unwrap();
    common::generate_simple_trek(dir.path());

    let guard = spawn_fixture_with_behavior(dir.path(), "explorer-action-probs", &[]);

    // Open explorer
    common::type_text(guard.name(), "B");
    common::wait_for_text(guard.name(), "test-behavior", 5);

    // Verify action_probs fields are exposed in genco and are null (no snapshot)
    let genco = common::zoltar_genco(guard.name());
    let explorer = &genco["explorer"];
    assert_eq!(explorer["has_action_probs"].as_bool(), Some(false),
        "No snapshot → has_action_probs should be false: {}", explorer);
    assert!(explorer["action_probs_epoch"].is_null(),
        "No snapshot → action_probs_epoch should be null: {}", explorer);
    assert!(explorer["action_probs_behavior"].is_null(),
        "No snapshot → action_probs_behavior should be null: {}", explorer);

    // Preview should still render fine (graceful degradation)
    let screen = common::dump_text(guard.name());
    let lines: Vec<&str> = screen.lines().collect();
    let preview_chars: usize = lines.iter()
        .filter_map(|line| line.get(60..))
        .map(|s| s.chars().filter(|c| !c.is_whitespace()).count())
        .sum();
    assert!(preview_chars > 20,
        "Preview should render even without action probs (got {} chars):\n{}",
        preview_chars, screen);
}

// ─── Phase 12: Corrupt Parquet Handling ──────────────────────────────────────

#[test]

fn airbus_corrupt_parquet_no_crash() {
    let dir = TempDir::new().unwrap();
    common::generate_corrupt_trek(dir.path());

    let name = session("corrupt-pq");
    common::spawn_octane(dir.path(), &name, &[]);
    let guard = SessionGuard(name);

    // Should start without crashing — wait for the UI to appear
    // With corrupt parquet, there are no epochs, so "epoch" text won't appear.
    // Wait for any rendered content (the TUI frame itself).
    std::thread::sleep(std::time::Duration::from_secs(1));
    let screen = common::dump_text(guard.name());

    // The scene pane should show the error message.
    // Note: airbus dump_text may collapse whitespace, so check for both forms.
    let has_error = screen.contains("Failed to load")
        || screen.contains("Failedtoload")
        || screen.contains("Parquet")
        || screen.contains("Corrupt");
    assert!(has_error,
        "Scene pane should show error for corrupt parquet:\n{}", screen);

    // App should still be running (not crashed) — verify via genco
    let genco = common::zoltar_genco(guard.name());
    assert!(genco.get("screen").is_some(),
        "App should still be running after corrupt parquet: {:?}", genco);
}

