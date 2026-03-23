# Octane Configuration
Related: [[Octane overview]] | [[Octane graphics and rendering]] | [[Octane key bindings]]

## Overview

Octane reads its configuration from `~/.highjax/config.json`. The file is hot-reloaded on change, so you can edit it while Octane is running and see the effects immediately. All fields have defaults, so a partial config file (or even an empty `{}`) works fine.

The config code lives in `octane/src/config.rs`.

## Config File

Path: `~/.highjax/config.json`

To create or update the config with all defaults filled in:
```
octane config fill-defaults
```

To print the config file path:
```
octane config path
```

## Config Hierarchy

The JSON structure nests everything under an `octane` key:

```
Config
  octane: OctaneConfig
    colors: ColorsConfig
      scene_themes: SceneThemesConfig
        dark: SceneColorConfig
        light: SceneColorConfig
      ui: UiColorConfig
    podium: PodiumConfig
    terrain: TerrainConfig
    road: RoadConfig
    rendering: RenderingConfig
    velocity_arrows: VelocityArrowsConfig
    attention: AttentionConfig
    action_distribution: ActionDistributionConfig
    hardcoded_arrow: HardcodedArrowConfig
    npc_text: NpcTextConfig
```

Example partial config:
```json
{
  "octane": {
    "rendering": {
      "playback_speed": 3.0,
      "fps": 30,
      "use_octants": true
    },
    "podium": {
      "omega": 0.2
    }
  }
}
```

Any fields you omit get their defaults via serde's `#[serde(default)]`.

## Key Rendering Defaults

| Setting | Default | Notes |
|---|---|---|
| theme | dark | Scene color theme (dark or light) |
| playback_speed | 2.0 | Multiplier (2x = twice real-time) |
| fps | 30 | TUI refresh rate |
| use_sextants | true | Unicode 13.0 characters |
| use_octants | true | Unicode 16.0 characters (best quality) |
| show_scala | true | Scale bar in bottom-right corner |
| corn_aspro | 1.875 | Terminal character aspect ratio (height/width) |
| headlight_opacity | 0.12 | Headlight cone base opacity |
| brakelight_opacity | 0.4 | Brakelight cone base opacity |
| brakelight_deceleration_threshold_m_s2 | 5.0 | Deceleration (m/s^2) to trigger brakelights |
| brakelight_deceleration_lookback_seconds | 0.3 | Seconds to look back for deceleration |
| pixels_per_corn_diagonal | 25.3 | Rasterization resolution per character |
| unzoomed_canvas_diagonal_in_meters | 180.0 | Scene coverage at zoom=1.0 |
| velocity_arrows | Never | DisplayMode for velocity arrow overlays |
| action_distribution | Never | DisplayMode for action distribution overlay |
| action_distribution_text | Never | DisplayMode for action distribution text |
| npc_text | Never | DisplayMode for NPC text labels |

## Podium Defaults

| Setting | Default | Notes |
|---|---|---|
| offset | 0.3 | Ego at ~30% from left edge |
| show_marker | false | Red vertical line at podium position |
| marker_stroke | 2.0 | Marker stroke width |
| omega | 0.5 | Spring natural frequency (lower = more drift) |
| damping_ratio | 1.0 | 1.0 = critically damped (no oscillation) |

The omega controls how tightly the camera tracks the ego vehicle. Low values (0.01-0.05) give a smooth, drifting camera. High values (1.0+) make the camera snap to the ego. The default 0.5 is a moderate track.

## Terrain Defaults

| Setting | Default | Notes |
|---|---|---|
| scale | 15.0 | Grid size in meters for blob placement |
| density | 0.55 | Noise threshold (0-1, higher = fewer blobs) |
| blob_size_min | 0.45 | Min blob radius as fraction of scale |
| blob_size_range | 0.9 | Blob radius variation as fraction of scale |

## Road Defaults

| Setting | Default | Notes |
|---|---|---|
| n_lanes | 4 | Number of lanes |
| lane_width | 4.0 | Lane width in meters |
| vehicle_length | 5.0 | Vehicle length in meters |
| vehicle_width | 2.0 | Vehicle width in meters |
| edge_stroke | 5.0 | Road edge line width |
| lane_stroke | 2.5 | Lane divider line width |
| edge_border_multiplier | 1.5 | Multiplier for edge border width |

## Scene Colors (Dark Theme)

These are the defaults for `colors.scene_themes.dark`. The light theme has its own set of defaults.

| Setting | Default | Notes |
|---|---|---|
| background | #030100 | Near-black with warm tint |
| terrain | #001000 | Very dark green |
| road_surface | #2a2a2a | Dark gray |
| road_edge | #787878 | Medium gray |
| lane_divider | #646464 | Slightly darker gray |
| lane_divider_dashed | false | Whether lane dividers are dashed |
| ego | #b4b4b4 | Silver |
| ego_crashed | #ff6464 | Red |
| npc_lightness | 0.75 | OKLCH lightness for NPC colors |
| npc_chroma | 0.15 | OKLCH chroma for NPC colors |
| window | #1a1a2e | Vehicle window color |
| scale_bar_fg | #cccccc | Scale bar foreground |
| scale_bar_bg | #000000 | Scale bar background |
| headlights | true | Whether to render headlights |
| headlight | #ffffcc | Headlight color |
| brakelight | #ff3333 | Brakelight color |
| light_blend_mode | lighten | SVG blend mode for light cones |
| action_color | #00ffff | Action distribution arrow color |
| action_chosen_color | #ffd700 | Chosen action arrow color |
| hardcoded_arrow_color | #ffd700 | Hardcoded action arrow color |

NPC colors are generated via OKLCH with golden-angle hue spacing, so each NPC gets a distinct hue at the configured lightness and chroma.

## UI Colors

The `ui` section controls ratatui TUI elements:

| Setting | Default | Notes |
|---|---|---|
| focused_border | #b09000 | Gold for focused pane |
| unfocused_border | #808080 | Gray for unfocused panes |
| selected_bg | #444444 | Selected item background |
| selected_fg | #ffffff | Selected item foreground |
| mnemonic | #ffd700 | Gold for keyboard shortcuts in titles |
| title_bar_fg / bg | #ffffff / #14143c | Top title bar |
| scrollbar_track_multiplier | 0.25 | Track brightness multiplier |
| scrollbar_thumb_multiplier | 0.5 | Thumb brightness multiplier |
| toast_border | #ffff00 | Toast notification border |
| toast_prefix | #888888 | Toast notification prefix |
| modal_help_border | #00ffff | Help modal border |
| modal_border | #00ff00 | Modal border |
| debug_border | #ff00ff | Debug modal border |
| modal_bg | #000000 | Modal background |
| seeker_played / remaining | #ffff00 / #808080 | Playback progress bar |
| status_bar_bg / key / text | #111111 / #fea62b / #e0e0e0 | Bottom status bar |

## Velocity Arrows Defaults

| Setting | Default | Notes |
|---|---|---|
| opacity | 0.2 | Arrow opacity (0.0-1.0) |
| length_scale | 0.3 | Meters per m/s of speed |
| min_speed | 0.1 | Min speed (m/s) to show arrow |
| head_angle | 0.5 | Arrowhead half-angle in radians |
| stroke_width | 20.0 | Base stroke width in pixels |
| stroke_speed_factor | 0.05 | Additional stroke scaling per m/s |
| stroke_min | 1.0 | Minimum stroke width (pixels) |
| stroke_max | 30.0 | Maximum stroke width (pixels) |
| head_size | 0.25 | Head size as fraction of length |
| head_speed_factor | 0.03 | Additional head scaling per m/s |
| head_min_meters | 0.5 | Minimum head length (meters) |
| head_max_meters | 5.0 | Maximum head length (meters) |

## Attention Defaults

| Setting | Default | Notes |
|---|---|---|
| show | false | Whether to show attention overlay |

## Action Distribution Defaults

| Setting | Default | Notes |
|---|---|---|
| color | #00ffff | Arrow color |
| chosen_color | #ffd700 | Color for the chosen action |
| circle_color | #00ffff | Idle circle color |
| max_arrow_length | 10.0 | Max arrow length (meters, at p=1.0) |
| stroke_width | 20.0 | Arrow stroke width (pixels) |
| min_circle_radius | 1.0 | Min idle circle radius (meters) |
| max_circle_radius | 1.0 | Max idle circle radius (meters) |
| min_opacity | 0.1 | Min arrow opacity |
| max_opacity | 0.8 | Max arrow opacity |
| circle_min_opacity | 0.0 | Min idle circle opacity |
| circle_max_opacity | 1.0 | Max idle circle opacity |
| head_angle | 1.0 | Arrowhead half-angle (radians) |
| head_size | 0.5 | Head length as fraction of arrow |
| font_family | 'Cascadia Mono', monospace | Font for percentage labels |
| font_size_meters | 2.0 | Font size in world meters |
| text_color | #cccccc | Text color (empty = arrow color) |
| chosen_text_color | (empty) | Chosen text color (empty = chosen_color) |
| text_bg | #000000 | Text background (empty = none) |
| text_offset_meters | 1.0 | Distance from tip to label (meters) |

## Hardcoded Arrow Defaults

For `--hardcoded-action` in draw mode:

| Setting | Default | Notes |
|---|---|---|
| length | 5.0 | Arrow length in meters |
| shaft_width | 1.2 | Shaft width in meters |
| head_width | 3.0 | Arrowhead full width (meters) |
| head_length | 2.0 | Arrowhead length (meters) |
| opacity | 0.85 | Arrow opacity (0.0-1.0) |

## NPC Text Defaults

| Setting | Default | Notes |
|---|---|---|
| font_family | 'DejaVu Sans Mono', monospace | Font for NPC labels |
| font_size_meters | 1.2 | Font size in world meters |
| bg_color | #000000 | Label background color |

## Hot-Reloading

The config file is watched using the `notify` crate. On change:

1. Wait 100ms for the file to finish writing
2. Reload and parse the file
3. Send the new config through an mpsc channel
4. The main loop drains the channel, keeping only the latest config
5. `apply_config()` updates derived fields and invalidates the viewport cache

The watcher follows symlinks, so Dropbox-synced configs (via dotty) work correctly. It also handles atomic saves (write to tmp file, then rename) by watching the parent directory rather than the file itself.

## CLI Overrides

CLI flags take priority over the config file. Relevant flags:

```
--fps 60              # Override FPS
--omega 0.5           # Override viewport spring
--sextants            # Force sextants on
--no-sextants         # Force sextants off
--octants             # Force octants on
--no-octants          # Force octants off
--debug-eye           # Fixed attention opacities (100%, 60%, 30%)
```

Runtime changes via the Graphics modal (`r` key) override both config and CLI for the current session, but are not persisted.
