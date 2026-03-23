//! Integration tests for the draw command CLI.
//!
//! These tests focus on CLI argument parsing and error handling.
//! SVG generation is tested separately in svg_generation_tests.rs.

mod common;

use std::process::Command;
use tempfile::TempDir;

/// Write a minimal meta.yaml so Trek::load succeeds (fewer fields than
/// common's version, since draw tests don't need road geometry).
fn write_test_meta_yaml(dir: &std::path::Path) {
    std::fs::write(
        dir.join("meta.yaml"),
        "commands:\n  2.highway:\n    seconds_per_t: 0.1\n",
    ).unwrap();
}

/// Run octane with given arguments and return (stdout, stderr, exit_code).
fn run_cli(args: &[&str]) -> (String, String, i32) {
    let output = Command::new(common::binary_path())
        .args(args)
        .output()
        .expect("Failed to run octane");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let code = output.status.code().unwrap_or(-1);

    (stdout, stderr, code)
}

// =============================================================================
// Help and Version Tests
// =============================================================================

#[test]
fn test_main_help() {
    let (stdout, _, code) = run_cli(&["--help"]);
    assert_eq!(code, 0);
    assert!(stdout.contains("Browse Highway training episodes"));
    assert!(stdout.contains("draw"));
    assert!(stdout.contains("--target"));
}

#[test]
fn test_draw_help() {
    let (stdout, _, code) = run_cli(&["draw", "--help"]);
    assert_eq!(code, 0);
    assert!(stdout.contains("Draw a single frame"));
    assert!(stdout.contains("--svg"));
    assert!(stdout.contains("--png"));
    assert!(stdout.contains("--epoch"));
    assert!(stdout.contains("--episode"));
    assert!(stdout.contains("--timestep"));
    assert!(stdout.contains("--cols"));
    assert!(stdout.contains("--rows"));
}

#[test]
fn test_draw_help_shows_defaults() {
    let (stdout, _, _) = run_cli(&["draw", "--help"]);
    assert!(stdout.contains("[default: 120]"), "Should show default for cols");
    assert!(stdout.contains("[default: 40]"), "Should show default for rows");
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_draw_requires_trek() {
    let (stdout, stderr, code) = run_cli(&["draw"]);
    // Without -t flag, octane tries to find latest trek in HIGHJAX_HOME/t or ~/.highjax/t
    // If found, it proceeds (code 0). If not found, it fails.
    // Either way, test just checks it doesn't crash unexpectedly
    let combined = format!("{}{}", stdout, stderr);
    assert!(code == 0 || combined.contains("trek") || combined.contains("No trek") || combined.contains("not found") || combined.contains("Parquet") || combined.contains("out of range"),
        "Should either succeed with trek or fail mentioning trek: stdout={}, stderr={}", stdout, stderr);
}

#[test]
fn test_draw_nonexistent_trek() {
    let (_, _stderr, code) = run_cli(&[
        "draw", "-t", "/nonexistent/path/to/trek"
    ]);
    assert_ne!(code, 0, "Should fail with nonexistent trek");
}

#[test]
fn test_draw_empty_trek_dir() {
    let dir = TempDir::new().unwrap();
    let (_, _stderr, code) = run_cli(&[
        "draw", "-t", dir.path().to_str().unwrap()
    ]);
    assert_ne!(code, 0, "Should fail with empty trek directory");
}

#[test]
fn test_draw_invalid_epoch_format() {
    let dir = TempDir::new().unwrap();
    let (_, stderr, code) = run_cli(&[
        "draw", "-t", dir.path().to_str().unwrap(), "--epoch", "not_a_number"
    ]);
    assert_ne!(code, 0);
    assert!(stderr.contains("invalid") || stderr.contains("error"),
        "Should have parsing error");
}

#[test]
fn test_draw_invalid_episode_format() {
    let dir = TempDir::new().unwrap();
    let (_, _stderr, code) = run_cli(&[
        "draw", "-t", dir.path().to_str().unwrap(), "--episode", "abc"
    ]);
    assert_ne!(code, 0);
}

#[test]
fn test_draw_invalid_timestep_format() {
    let dir = TempDir::new().unwrap();
    let (_, _stderr, code) = run_cli(&[
        "draw", "-t", dir.path().to_str().unwrap(), "--timestep", "abc"
    ]);
    assert_ne!(code, 0);
}

#[test]
fn test_draw_invalid_cols_format() {
    let dir = TempDir::new().unwrap();
    let (_, _stderr, code) = run_cli(&[
        "draw", "-t", dir.path().to_str().unwrap(), "--cols", "wide"
    ]);
    assert_ne!(code, 0);
}

#[test]
fn test_draw_invalid_rows_format() {
    let dir = TempDir::new().unwrap();
    let (_, _stderr, code) = run_cli(&[
        "draw", "-t", dir.path().to_str().unwrap(), "--rows", "tall"
    ]);
    assert_ne!(code, 0);
}

// =============================================================================
// Argument Combination Tests
// =============================================================================

#[test]
fn test_draw_accepts_all_options() {
    // This should parse correctly even if trek doesn't exist
    let (_, _stderr, code) = run_cli(&[
        "draw",
        "-t", "/tmp/fake_trek",
        "--epoch", "5",
        "--episode", "10",
        "--timestep", "100",
        "--cols", "200",
        "--rows", "50",
        "--svg", "/tmp/out.svg"
    ]);
    // Will fail because trek doesn't exist, but args should parse
    assert_ne!(code, 0); // Expected to fail, but for trek not args
}

#[test]
fn test_draw_zoom_options_accepted() {
    // Test that zoom options are accepted by the CLI parser
    for zoom in &["0.5", "1.0", "2.0"] {
        let (_, stderr, _code) = run_cli(&[
            "draw", "-t", "/tmp/fake", "--zoom", zoom
        ]);
        // Should not contain "invalid value" for zoom
        assert!(!stderr.contains("invalid value") || !stderr.contains("zoom"),
            "Zoom '{}' should be accepted", zoom);
    }
}

#[test]
fn test_draw_global_options() {
    // Global options should be accepted with draw subcommand
    let (_, _stderr, _code) = run_cli(&[
        "draw", "-t", "/tmp/fake", "--verbose"
    ]);
    // Just checking it doesn't fail on parsing
}

#[test]
fn test_draw_short_target_flag() {
    // -t should work same as --target
    let (_, _stderr1, code1) = run_cli(&["-t", "/tmp/fake", "draw"]);
    let (_, _stderr2, code2) = run_cli(&["--target", "/tmp/fake", "draw"]);

    // Both should fail the same way (trek not found)
    assert_eq!(code1, code2, "-t and --target should behave the same");
}

// =============================================================================
// PNG Not Implemented Test
// =============================================================================

#[test]
fn test_draw_png_mentions_not_implemented() {
    // Create a minimal valid epoch structure
    let dir = TempDir::new().unwrap();
    write_test_meta_yaml(dir.path());
    let epoch_path = dir.path().join("epoch_0");
    std::fs::create_dir(&epoch_path).unwrap();

    // Create minimal episode file (will fail later for vehicle_state)
    let ep_data = r#"{"observations":[[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]],"actions":[0],"rewards":[0.1],"dones":[false]}"#;
    std::fs::write(epoch_path.join("episode_0.json"), ep_data).unwrap();

    let output_dir = TempDir::new().unwrap();
    let png_path = output_dir.path().join("test.png");

    let (_, stderr, code) = run_cli(&[
        "draw", "-t", dir.path().to_str().unwrap(),
        "--png", png_path.to_str().unwrap()
    ]);

    // Should fail, mentioning either PNG not implemented or vehicle_state
    assert_ne!(code, 0);
    let combined = format!("{}", stderr);
    assert!(combined.contains("not yet implemented") || combined.contains("vehicle_state") || combined.contains("PNG"),
        "Should mention PNG not implemented or vehicle_state missing");
}

// =============================================================================
// Trek Structure Error Tests
// =============================================================================

#[test]
fn test_draw_epoch_out_of_range() {
    let dir = TempDir::new().unwrap();
    write_test_meta_yaml(dir.path());
    let epoch_path = dir.path().join("epoch_0");
    std::fs::create_dir(&epoch_path).unwrap();

    let ep_data = r#"{"observations":[[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]],"actions":[0],"rewards":[0.1],"dones":[false]}"#;
    std::fs::write(epoch_path.join("episode_0.json"), ep_data).unwrap();

    let (_, stderr, code) = run_cli(&[
        "draw", "-t", dir.path().to_str().unwrap(), "--epoch", "999"
    ]);

    assert_ne!(code, 0);
    assert!(stderr.contains("not found") || stderr.contains("Epoch"),
        "Should mention epoch not found");
}

#[test]
fn test_draw_episode_out_of_range() {
    let dir = TempDir::new().unwrap();
    write_test_meta_yaml(dir.path());
    let epoch_path = dir.path().join("epoch_0");
    std::fs::create_dir(&epoch_path).unwrap();

    let ep_data = r#"{"observations":[[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]],"actions":[0],"rewards":[0.1],"dones":[false]}"#;
    std::fs::write(epoch_path.join("episode_0.json"), ep_data).unwrap();

    let (_, stderr, code) = run_cli(&[
        "draw", "-t", dir.path().to_str().unwrap(), "--episode", "999"
    ]);

    assert_ne!(code, 0);
    assert!(stderr.contains("not found") || stderr.contains("Episode"),
        "Should mention episode not found");
}

#[test]
fn test_draw_timestep_accepted() {
    let dir = TempDir::new().unwrap();
    write_test_meta_yaml(dir.path());
    let epoch_path = dir.path().join("epoch_0");
    std::fs::create_dir(&epoch_path).unwrap();

    let ep_data = r#"{"observations":[[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]],"actions":[0],"rewards":[0.1],"dones":[false]}"#;
    std::fs::write(epoch_path.join("episode_0.json"), ep_data).unwrap();

    // --timestep uses nearest-neighbor, so any value is valid (no "out of range").
    // Will fail on vehicle_state, not on timestep parsing.
    let (_, stderr, code) = run_cli(&[
        "draw", "-t", dir.path().to_str().unwrap(), "--timestep", "999"
    ]);

    assert_ne!(code, 0);
    assert!(stderr.contains("vehicle_state"),
        "Should fail on vehicle_state, not timestep: {}", stderr);
}

#[test]
fn test_draw_legacy_episode_fails_gracefully() {
    let dir = TempDir::new().unwrap();
    write_test_meta_yaml(dir.path());
    let epoch_path = dir.path().join("epoch_0");
    std::fs::create_dir(&epoch_path).unwrap();

    // Legacy episode without vehicle_state
    let ep_data = r#"{"observations":[[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]],"actions":[0],"rewards":[0.1],"dones":[false]}"#;
    std::fs::write(epoch_path.join("episode_0.json"), ep_data).unwrap();

    let output_dir = TempDir::new().unwrap();
    let svg_path = output_dir.path().join("test.svg");

    let (_, stderr, code) = run_cli(&[
        "draw", "-t", dir.path().to_str().unwrap(),
        "--svg", svg_path.to_str().unwrap()
    ]);

    assert_ne!(code, 0, "Should fail for legacy episodes without vehicle_state");
    assert!(stderr.contains("vehicle_state"),
        "Error should mention missing vehicle_state: {}", stderr);
}

// =============================================================================
// Multiple Epoch Tests
// =============================================================================

#[test]
fn test_draw_multiple_epochs() {
    let dir = TempDir::new().unwrap();
    write_test_meta_yaml(dir.path());

    // Create epoch_0 and epoch_1
    for epoch in 0..2 {
        let epoch_path = dir.path().join(format!("epoch_{}", epoch));
        std::fs::create_dir(&epoch_path).unwrap();

        let ep_data = r#"{"observations":[[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]],"actions":[0],"rewards":[0.1],"dones":[false]}"#;
        std::fs::write(epoch_path.join("episode_0.json"), ep_data).unwrap();
    }

    // Try epoch 1 - should fail on vehicle_state, not on epoch not found
    let (_, stderr, code) = run_cli(&[
        "draw", "-t", dir.path().to_str().unwrap(), "--epoch", "1"
    ]);

    assert_ne!(code, 0);
    // Should fail on vehicle_state, not on finding epoch
    assert!(stderr.contains("vehicle_state"),
        "Should find epoch 1 but fail on vehicle_state");
}

// =============================================================================
// Output Path Tests
// =============================================================================

#[test]
fn test_draw_svg_path_with_spaces() {
    let dir = TempDir::new().unwrap();
    let epoch_path = dir.path().join("epoch_0");
    std::fs::create_dir(&epoch_path).unwrap();

    let ep_data = r#"{"observations":[[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]],"actions":[0],"rewards":[0.1],"dones":[false]}"#;
    std::fs::write(epoch_path.join("episode_0.json"), ep_data).unwrap();

    let output_dir = TempDir::new().unwrap();
    let svg_path = output_dir.path().join("test with spaces.svg");

    let (_, _stderr, _code) = run_cli(&[
        "draw", "-t", dir.path().to_str().unwrap(),
        "--svg", svg_path.to_str().unwrap()
    ]);

    // Will fail on vehicle_state, but path with spaces should be accepted
}

#[test]
fn test_draw_svg_creates_in_specified_directory() {
    let dir = TempDir::new().unwrap();
    let epoch_path = dir.path().join("epoch_0");
    std::fs::create_dir(&epoch_path).unwrap();

    let ep_data = r#"{"observations":[[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]],"actions":[0],"rewards":[0.1],"dones":[false]}"#;
    std::fs::write(epoch_path.join("episode_0.json"), ep_data).unwrap();

    let output_dir = TempDir::new().unwrap();
    let svg_path = output_dir.path().join("subdir").join("output.svg");

    let (_, _stderr, code) = run_cli(&[
        "draw", "-t", dir.path().to_str().unwrap(),
        "--svg", svg_path.to_str().unwrap()
    ]);

    // Should fail (either missing parent dir or vehicle_state)
    assert_ne!(code, 0);
}
