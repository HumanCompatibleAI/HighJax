use super::*;
use crate::data::{Epoch, EpisodeMeta};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use std::path::PathBuf;

fn make_episode(index: usize, n_frames: usize, total_reward: f64) -> EpisodeMeta {
    EpisodeMeta {
        index,
        path: PathBuf::from(format!("/test/ep{}.json", index)),
        n_frames,
        n_policy_frames: n_frames,
        n_alive_policy_frames: n_frames,
        total_reward: Some(total_reward),
        nz_return: None,
        es_epoch: None,
        es_episode: None,
    }
}

fn make_trek_with_epochs(epochs: Vec<Epoch>) -> Trek {
    Trek {
        path: PathBuf::from("/test"),
        epochs,
        es_parquet_index: None,
        ego_speed_range: None,
        seconds_per_t: 0.1,
        seconds_per_sub_t: 0.1,
        n_sub_ts_per_t: 1,
        snapshot_epochs: std::collections::HashSet::new(),
        n_ts_per_e: 20,
        n_lanes: None,
        load_error: None,
        env_type: Some(crate::envs::EnvType::Highway),
    }
}

fn make_test_trek() -> Trek {
    make_trek_with_epochs(vec![
        Epoch::new(0, 0, PathBuf::from("/test/epoch_000"), vec![
            make_episode(0, 10, 5.0),
            make_episode(1, 20, 10.0),
        ]),
        Epoch::new(1, 1, PathBuf::from("/test/epoch_001"), vec![
            make_episode(0, 15, 7.5),
        ]),
    ])
}

#[test]
fn test_focus_cycle() {
    assert_eq!(Focus::Treks.next(), Focus::Parquets);
    assert_eq!(Focus::Parquets.next(), Focus::Epochs);
    assert_eq!(Focus::Epochs.next(), Focus::Episodes);
    assert_eq!(Focus::Episodes.next(), Focus::Highway);
    assert_eq!(Focus::Highway.next(), Focus::Treks);

    assert_eq!(Focus::Treks.prev(), Focus::Highway);
    assert_eq!(Focus::Parquets.prev(), Focus::Treks);
    assert_eq!(Focus::Epochs.prev(), Focus::Parquets);
    assert_eq!(Focus::Episodes.prev(), Focus::Epochs);
    assert_eq!(Focus::Highway.prev(), Focus::Episodes);
}

#[test]
fn test_app_initial_state() {
    let trek = make_test_trek();
    let app = App::new(trek);

    assert_eq!(app.focus, Focus::Epochs);
    // Initial selection is the last epoch (index 1 in 2-epoch trek)
    assert_eq!(app.selected_epoch, 1);
    assert_eq!(app.selected_episode, 0);
    assert_eq!(app.frame_index, 0);
    assert!(!app.should_quit);
    assert!(!app.playback.playing);
    assert_eq!(app.playback.fps, 30);
    assert_eq!(app.episodes_scroll, 0);
    assert!(app.render_config.enabled);
}

#[test]
fn test_help_modal() {
    let trek = make_test_trek();
    let mut app = App::new(trek);

    assert!(app.active_modal.is_none());
    app.handle_key(KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE));
    assert!(matches!(app.active_modal, Some(Modal::Help)));
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    assert!(app.active_modal.is_none());
}

#[test]
fn test_quit_on_q() {
    let trek = make_test_trek();
    let mut app = App::new(trek);

    app.handle_key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE));
    assert!(app.should_quit);
}

#[test]
fn test_quit_on_esc() {
    let trek = make_test_trek();
    let mut app = App::new(trek);

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    assert!(app.should_quit);
}

#[test]
fn test_focus_tab() {
    let trek = make_test_trek();
    let mut app = App::new(trek);

    assert_eq!(app.focus, Focus::Epochs);
    app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    assert_eq!(app.focus, Focus::Episodes);
    app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    assert_eq!(app.focus, Focus::Highway);
    app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    assert_eq!(app.focus, Focus::Treks);
    app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    assert_eq!(app.focus, Focus::Parquets);
    app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    assert_eq!(app.focus, Focus::Epochs);
}

#[test]
fn test_epoch_navigation() {
    let trek = make_test_trek();
    let mut app = App::new(trek);
    app.selected_epoch = 0; // Start at first epoch for this test

    assert_eq!(app.selected_epoch, 0);
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    assert_eq!(app.selected_epoch, 1);
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    assert_eq!(app.selected_epoch, 1); // Can't go past last
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
    assert_eq!(app.selected_epoch, 0);
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
    assert_eq!(app.selected_epoch, 0); // Can't go below 0
}

#[test]
fn test_episode_navigation() {
    let trek = make_test_trek();
    let mut app = App::new(trek);
    app.selected_epoch = 0; // Epoch 0 has 2 episodes
    app.focus = Focus::Episodes;

    assert_eq!(app.selected_episode, 0);
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    assert_eq!(app.selected_episode, 1);
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    assert_eq!(app.selected_episode, 1); // Can't go past last
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
    assert_eq!(app.selected_episode, 0);
}

#[test]
fn test_frame_navigation() {
    let trek = make_test_trek();
    let mut app = App::new(trek);
    app.focus = Focus::Highway;

    assert_eq!(app.frame_index, 0);
    app.handle_key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
    assert_eq!(app.frame_index, 1);
    app.handle_key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
    assert_eq!(app.frame_index, 0);
    app.handle_key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
    assert_eq!(app.frame_index, 0); // Can't go below 0
}

#[test]
fn test_play_pause() {
    let trek = make_test_trek();
    let mut app = App::new(trek);

    assert!(!app.playback.playing);
    app.handle_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE));
    assert!(app.playback.playing);
    app.handle_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE));
    assert!(!app.playback.playing);
}

#[test]
fn test_tick_advances_frame() {
    use std::time::Instant;

    let trek = make_test_trek();
    let mut app = App::new(trek);

    // Manually set up playback state at a time in the past so elapsed time is significant
    app.playback.playing = true;
    app.playback.speed = 1.0;
    // Set playback_start to 0.15 seconds ago (should advance to frame 1)
    app.playback.start = Some(Instant::now() - std::time::Duration::from_secs_f64(0.15));
    app.playback.start_scene_time = 0.0;

    assert_eq!(app.frame_index, 0);
    app.tick();
    // After 0.15s at speed 1x, scene_time ~= 0.15, frame_index = floor(0.15/0.1) = 1
    assert_eq!(app.frame_index, 1);
}

#[test]
fn test_tick_stops_at_end() {
    use std::time::Instant;

    let trek = make_test_trek();
    let mut app = App::new(trek);
    app.selected_epoch = 0; // Epoch 0, episode 0 has 10 frames

    // Set up playback starting at last frame (scene_time = 0.9 for frame 9)
    // max_scene_time = 9 * 0.1 = 0.9
    app.playback.playing = true;
    app.playback.start = Some(Instant::now());
    app.playback.start_scene_time = 9.0 * 0.1; // 0.9
    app.frame_index = 9;

    app.tick();
    // scene_time >= max_scene_time, so should stop
    assert_eq!(app.frame_index, 9); // Stays at last
    assert!(!app.playback.playing); // Stops playing
}

#[test]
fn test_home_end_keys() {
    let trek = make_test_trek();
    let mut app = App::new(trek);
    app.selected_epoch = 0; // Epoch 0 has 10-frame episode
    app.focus = Focus::Highway;
    app.frame_index = 5;

    app.handle_key(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE));
    assert_eq!(app.frame_index, 0);

    app.handle_key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE));
    assert_eq!(app.frame_index, 9); // 10 frames, 0-indexed
}

#[test]
fn test_epoch_change_resets_episode() {
    let trek = make_test_trek();
    let mut app = App::new(trek);
    app.selected_epoch = 0; // Start at epoch 0 so Down goes to 1
    app.selected_episode = 1;
    app.frame_index = 5;

    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    assert_eq!(app.selected_epoch, 1);
    assert_eq!(app.selected_episode, 0); // Reset
    assert_eq!(app.frame_index, 0); // Reset
}

#[test]
fn test_page_down_epochs() {
    let trek = make_test_trek();
    let mut app = App::new(trek);

    // With only 2 epochs, PageDown should go to last
    app.handle_key(KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE));
    assert_eq!(app.selected_epoch, 1);
}

#[test]
fn test_page_up_epochs() {
    let trek = make_test_trek();
    let mut app = App::new(trek);
    app.selected_epoch = 1;

    app.handle_key(KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE));
    assert_eq!(app.selected_epoch, 0);
}

#[test]
fn test_page_down_episodes() {
    let trek = make_test_trek();
    let mut app = App::new(trek);
    app.selected_epoch = 0; // Epoch 0 has 2 episodes
    app.focus = Focus::Episodes;

    // First epoch has 2 episodes, PageDown should go to last
    app.handle_key(KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE));
    assert_eq!(app.selected_episode, 1);
}

#[test]
fn test_page_down_frames() {
    let trek = make_test_trek();
    let mut app = App::new(trek);
    app.selected_epoch = 0; // Epoch 0, episode 0 has 10 frames
    app.focus = Focus::Highway;

    // First episode has 10 frames, PageDown jumps 80% of default page (8)
    app.handle_key(KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE));
    assert_eq!(app.frame_index, 8);
    // Another PageDown should clamp to max (9)
    app.handle_key(KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE));
    assert_eq!(app.frame_index, 9);
}

#[test]
fn test_vim_keys_navigation() {
    let trek = make_test_trek();
    let mut app = App::new(trek);
    app.selected_epoch = 0; // Epoch 0, episode 0 has 10 frames

    // Start in Highway focus for frame navigation
    app.focus = Focus::Highway;
    app.frame_index = 5;

    // j = previous frame
    app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
    assert_eq!(app.frame_index, 4);

    // k = next frame
    app.handle_key(KeyEvent::new(KeyCode::Char('k'), KeyModifiers::NONE));
    assert_eq!(app.frame_index, 5);

    // J (Shift+j) = 5 frames back
    app.handle_key(KeyEvent::new(KeyCode::Char('J'), KeyModifiers::SHIFT));
    assert_eq!(app.frame_index, 0);

    // K (Shift+k) = 5 frames forward
    app.handle_key(KeyEvent::new(KeyCode::Char('K'), KeyModifiers::SHIFT));
    assert_eq!(app.frame_index, 5);

    // h = go to first frame
    app.frame_index = 5;
    app.handle_key(KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE));
    assert_eq!(app.frame_index, 0);

    // l = go to last frame
    app.handle_key(KeyEvent::new(KeyCode::Char('l'), KeyModifiers::NONE));
    assert_eq!(app.frame_index, 9); // 10 frames, 0-indexed
}

#[test]
fn test_jump_epoch_modal_open() {
    let trek = make_test_trek();
    let mut app = App::new(trek);

    app.handle_key(KeyEvent::new(KeyCode::Char('g'), KeyModifiers::NONE));
    assert!(matches!(app.active_modal, Some(Modal::JumpEpoch)));
    assert!(app.jump_input.is_empty());
}

#[test]
fn test_jump_epoch_input_and_confirm() {
    let trek = make_test_trek();
    let mut app = App::new(trek);

    app.handle_key(KeyEvent::new(KeyCode::Char('g'), KeyModifiers::NONE));

    // Type "1"
    app.handle_key(KeyEvent::new(KeyCode::Char('1'), KeyModifiers::NONE));
    assert_eq!(app.jump_input, "1");

    // Confirm
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert_eq!(app.selected_epoch, 1);
    assert!(app.active_modal.is_none());
    assert_eq!(app.last_jump, Some(1));
}

#[test]
fn test_jump_epoch_bounds() {
    let trek = make_test_trek();
    let mut app = App::new(trek);

    app.handle_key(KeyEvent::new(KeyCode::Char('g'), KeyModifiers::NONE));

    // Type "99" (way beyond max)
    app.handle_key(KeyEvent::new(KeyCode::Char('9'), KeyModifiers::NONE));
    app.handle_key(KeyEvent::new(KeyCode::Char('9'), KeyModifiers::NONE));
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

    // Should be clamped to max epoch (1)
    assert_eq!(app.selected_epoch, 1);
}

#[test]
fn test_jump_episode_modal() {
    let trek = make_test_trek();
    let mut app = App::new(trek);

    app.handle_key(KeyEvent::new(KeyCode::Char('G'), KeyModifiers::NONE));
    assert!(matches!(app.active_modal, Some(Modal::JumpEpisode)));
}

#[test]
fn test_jump_frame_modal() {
    let trek = make_test_trek();
    let mut app = App::new(trek);

    app.handle_key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE));
    assert!(matches!(app.active_modal, Some(Modal::JumpFrame)));
}

#[test]
fn test_jump_input_backspace() {
    let trek = make_test_trek();
    let mut app = App::new(trek);

    app.handle_key(KeyEvent::new(KeyCode::Char('g'), KeyModifiers::NONE));
    app.handle_key(KeyEvent::new(KeyCode::Char('1'), KeyModifiers::NONE));
    app.handle_key(KeyEvent::new(KeyCode::Char('2'), KeyModifiers::NONE));
    assert_eq!(app.jump_input, "12");

    app.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
    assert_eq!(app.jump_input, "1");
}

#[test]
fn test_jump_repeat_with_n() {
    let trek = make_test_trek();
    let mut app = App::new(trek);

    // First jump to epoch 1
    app.handle_key(KeyEvent::new(KeyCode::Char('g'), KeyModifiers::NONE));
    app.handle_key(KeyEvent::new(KeyCode::Char('1'), KeyModifiers::NONE));
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert_eq!(app.selected_epoch, 1);

    // Go back to epoch 0
    app.selected_epoch = 0;

    // Use 'n' to repeat last jump
    app.handle_key(KeyEvent::new(KeyCode::Char('g'), KeyModifiers::NONE));
    app.handle_key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE));
    assert_eq!(app.selected_epoch, 1);
}

#[test]
fn test_debug_toggle() {
    let trek = make_test_trek();
    let mut app = App::new(trek);

    assert!(!app.show_debug);
    app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE));
    assert!(app.show_debug);
    app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE));
    assert!(!app.show_debug);
}

// Edge case tests

#[test]
fn test_single_frame_episode_playback() {
    use std::time::Instant;

    // Create a trek with a single-frame episode
    let trek = make_trek_with_epochs(vec![
        Epoch::new(0, 0, PathBuf::from("/test/epoch_000"), vec![make_episode(0, 1, 1.0)]),
    ]);
    let mut app = App::new(trek);

    // Set up playback properly
    app.playback.playing = true;
    app.playback.start = Some(Instant::now());
    app.playback.start_scene_time = 0.0;

    // Tick should immediately stop at single frame
    // For n_frames=1, max_scene_time = 0, so scene_time >= max_scene_time immediately
    app.tick();
    assert!(!app.playback.playing); // Should stop
    assert_eq!(app.frame_index, 0); // Should stay at 0
}

#[test]
fn test_empty_epochs_navigation() {
    // Trek with empty epochs
    let trek = make_trek_with_epochs(vec![]);
    let mut app = App::new(trek);

    // Navigation should not panic on empty epochs
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
    assert_eq!(app.selected_epoch, 0);
}

#[test]
fn test_epoch_with_no_episodes() {
    let trek = make_trek_with_epochs(vec![
        Epoch::new(0, 0, PathBuf::from("/test/epoch_000"), vec![]),
    ]);
    let mut app = App::new(trek);
    app.focus = Focus::Episodes;

    // Navigation should not panic
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    assert_eq!(app.selected_episode, 0);
}

#[test]
fn test_frame_bounds_on_episode_change() {
    let trek = make_test_trek();
    let mut app = App::new(trek);
    app.selected_epoch = 0; // Start at epoch 0 so Down goes to 1

    // Go to last frame of first episode (10 frames)
    app.frame_index = 9;

    // Change epoch - should reset frame index
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    assert_eq!(app.frame_index, 0);
}

#[test]
fn test_current_frame_with_no_episode() {
    let trek = make_trek_with_epochs(vec![
        Epoch::new(0, 0, PathBuf::from("/test/epoch_000"), vec![]),
    ]);
    let app = App::new(trek);

    // Should return None without panic
    assert!(app.current_frame().is_none());
}

#[test]
fn test_scene_theme_default_is_dark() {
    let trek = make_test_trek();
    let app = App::new(trek);
    assert_eq!(app.scene_theme, crate::config::SceneTheme::Dark);
}

#[test]
fn test_scene_theme_toggle_in_graphics_modal() {
    let trek = make_test_trek();
    let mut app = App::new(trek);
    assert_eq!(app.scene_theme, crate::config::SceneTheme::Dark);

    // Open graphics modal
    app.active_modal = Some(Modal::Graphics);
    app.graphics_selection = GraphicsControl::Theme;

    // Press right to cycle theme
    app.handle_key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
    assert_eq!(app.scene_theme, crate::config::SceneTheme::Light);

    // Press right again to cycle back
    app.handle_key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
    assert_eq!(app.scene_theme, crate::config::SceneTheme::Dark);
}

#[test]
fn test_scala_toggle() {
    let trek = make_test_trek();
    let mut app = App::new(trek);

    // Scala defaults to on
    assert!(app.highway_prefs.show_scala);

    // Open graphics modal, select Scala via mnemonic
    app.active_modal = Some(Modal::Graphics);
    app.last_toast_text = None; // Clear so 'c' reaches the modal
    app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE));
    assert_eq!(app.graphics_selection, GraphicsControl::Scala);

    // Toggle off
    app.handle_key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
    assert!(!app.highway_prefs.show_scala);

    // Toggle back on
    app.handle_key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
    assert!(app.highway_prefs.show_scala);
}
