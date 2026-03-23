# Octane Architecture

Related: [[Octane overview]] | [[Octane graphics and rendering]] | [[Octane configuration]] | [[Octane navigation and playback]]

Developer-facing internals: source tree, coordinate system details, Mango renderer internals, and performance.

## Source tree

```
octane/
├── Cargo.toml
└── src/
    ├── leona/                  # Logging library (vendored from $DXP/mussel)
    │   ├── Cargo.toml
    │   └── src/
    │       ├── lib.rs
    │       ├── config.rs       # Log config
    │       ├── dedup.rs        # Deduplication
    │       ├── paths.rs        # Log file paths
    │       ├── pointers.rs     # Pointer utilities
    │       └── rotation.rs     # Log rotation
    ├── main.rs                 # Entry point, event loop
    ├── lib.rs                  # Library exports for tests
    ├── config.rs               # Config struct, hot-reload, CLI overrides
    ├── util.rs                 # Misc utilities
    ├── cli.rs                  # CLI module root
    ├── cli/
    │   ├── args.rs             # CLI argument definitions (clap)
    │   ├── commands.rs         # Command module root (re-exports run_draw, run_animate, run_zoltar)
    │   └── commands/
    │       ├── animate.rs      # Animate subcommand
    │       ├── draw.rs         # Draw subcommand
    │       └── zoltar.rs       # Zoltar CLI client (socket discovery, query)
    ├── app/                    # App state and logic
    │   ├── mod.rs              # App struct, construction
    │   ├── state.rs            # State types (Focus, Screen, ExplorerState, etc.)
    │   ├── input.rs            # Key handling dispatch
    │   ├── navigation.rs       # Pane navigation, frame position memory
    │   ├── playback.rs         # Playback timing, play/pause
    │   ├── trek.rs             # Trek loading, parquet discovery
    │   ├── episode.rs          # Episode loading and caching
    │   ├── clipboard.rs        # Clipboard operations
    │   ├── explorer.rs         # Behavior explorer logic
    │   ├── explorer_editing.rs # Scenario editor (add/remove NPC, save)
    │   ├── zoltar.rs           # Zoltar request handlers (zork, pane, press, navigate)
    │   ├── modal_behavior.rs   # Behavior capture modal
    │   ├── modal_graphics.rs   # Graphics settings modal
    │   ├── modal_jump.rs       # Jump-to-epoch/episode/frame modals
    │   └── tests.rs            # Unit tests
    ├── ui.rs                   # UI module root, screen dispatch
    ├── ui/
    │   ├── panels.rs           # Browse screen panes (treks, epochs, highway, metrics)
    │   ├── dashboard.rs        # Dashboard pane: unified fixed-layout info panel (gauges)
    │   ├── explorer.rs         # Behavior explorer screen rendering
    │   ├── modals.rs           # Modal dialog rendering
    │   ├── styles.rs           # Block styles, scrollbar, mnemonic highlighting
    │   └── widgets.rs          # Shared widgets (velocity bar, etc.)
    ├── data/
    │   ├── mod.rs
    │   ├── trek.rs             # Trek directory parsing, posh paths
    │   ├── episode.rs          # Episode/Frame data structures
    │   ├── es_parquet.rs       # Parquet index and episode loading
    │   ├── jsonla.rs           # Legacy JSONLA episode loading
    │   └── behaviors.rs        # Behavior JSON loading and preset discovery
    ├── envs/                   # Environment adapters for multi-env dispatch
    │   ├── mod.rs              # EnvType enum, EnvParquetColumns, dispatch logic
    │   └── highway.rs          # Highway-specific logic (parquet columns, viewport, render config)
    ├── worlds/                 # Coordinate system (see below)
    │   ├── mod.rs              # Re-exports, constants
    │   ├── coords.rs           # ScenePoint, SvgPoint, SceneBounds
    │   ├── scene_episode.rs    # Raw episode data with interpolation
    │   ├── viewport_episode.rs # Camera tracking with spring dynamics
    │   └── svg_episode.rs      # Scene -> SVG coordinate transforms
    ├── render/
    │   ├── mod.rs
    │   ├── highway_svg.rs      # SVG generation for highway scenes
    │   ├── frame_render.rs     # Frame rendering orchestration
    │   ├── car_template.rs     # Car body SVG path
    │   ├── color.rs            # OKLCH NPC color generation
    │   ├── headlights.rs       # Headlight cones with shadow occlusion
    │   ├── brakelights.rs      # Brake light rendering
    │   ├── terrain.rs          # Background terrain blobs
    │   ├── velocity_arrows.rs  # Speed/heading arrow overlays
    │   ├── action_distribution.rs # Action probability visualization
    │   ├── npc_text.rs         # NPC labels and time-to-collision
    │   └── geometry.rs         # Polygon clipping, convex hull, subtraction
    ├── zoltar/                 # IPC query interface (Unix socket + file-based)
    │   ├── mod.rs              # ZoltarRequest/Response types, channel setup
    │   ├── socket.rs           # Socket listener thread, discovery file
    │   ├── file_ipc.rs         # File-based IPC transport (cross-platform, default on Windows)
    │   └── protocol.rs         # JSON parsing and serialization
    └── mango/                  # Terminal graphics renderer
        ├── mod.rs
        ├── svg.rs              # SVG rasterization via resvg
        ├── render.rs           # Parallel ANSI output generation
        ├── cli.rs              # CLI command implementation
        └── templates/
            ├── mod.rs          # Template matching (MSE-based character selection)
            ├── character_tables.rs # Sextant/quadrant/octant pattern tables
            ├── masks.rs        # Pixel mask generation
            └── mse.rs          # MSE computation helpers
```

## Dependencies

```toml
[dependencies]
leona = { path = "src/leona" } # Logging
ratatui = "0.29"               # TUI framework
crossterm = "0.28"             # Terminal backend
serde = { version = "1", features = ["derive"] }
serde_json = "1"               # Episode parsing
tracing = "0.1"                # Logging macros
chrono = "0.4"                 # Timestamps
clap = { version = "4", features = ["derive"] }  # CLI
ansi-to-tui = "7"              # ANSI -> Ratatui conversion
resvg = "0.45"                 # SVG rasterization (Mango)
image = "0.25"                 # Image buffer handling
rayon = "1.10"                 # Parallel rendering
notify = "6"                   # Config file watching
rusqlite = "0.31"              # Snapshot database
parquet = "54"                 # Episode data (primary format)
arrow = "54"                   # Arrow arrays for parquet
anyhow = "1"                   # Error handling
tempfile = "3"                 # Temporary files
serde_yaml = "0.9"             # YAML config parsing
dirs = "5"                     # Platform-specific directories
```

## Coordinate system terminology

| Term | Definition |
|---|---|
| scene | Simulation world coordinates, in meters |
| SVG | Normalized canvas units (width=1, height=corn_aspro) |
| raster | Pixel buffer coordinates (internal to Mango renderer) |
| maize | ASCII picture made of corn (the character grid output) |
| corn | Terminal character bounding box -- the rectangular area one character occupies |
| `n_cols` | Number of character columns in the maize |
| `n_rows` | Number of character rows in the maize |
| `corn_aspro` | Visual aspro (h/w) of a corn as it appears on screen. Default 1.875 (15/8, typical terminal font 15px tall / 8px wide). |
| aspro | Aspect ratio, always expressed as height/width |
| `canvas_xy_center_in_meters` | Tuple (x, y) specifying viewport center in scene coordinates |
| `canvas_diagonals_per_meter` | Alternative to zoom. E.g., 0.01 means 1 canvas diagonal = 100m. |
| `unzoomed_canvas_diagonal_in_meters` | Scene distance (in meters) that the canvas diagonal spans at zoom=1.0x. Constant regardless of aspect ratio or output mode. |
| `zoom` | Magnification multiplier. At zoom=1.0x, the canvas diagonal spans `unzoomed_canvas_diagonal_in_meters`. At zoom=2.0x, it spans half that. |
| `pixels_per_corn_diagonal` | Raster resolution when generating pixel buffer. Default 25.3. |

## Episode stack

The coordinate system is a layered Episode stack in `worlds/`. Each layer wraps the previous one.

```
SceneEpisode        Raw simulation data (Vec<FrameState>)
      |                state_at(t) with interpolation
ViewportEpisode     Camera tracking with spring dynamics (ONLY stateful layer)
      |                view_x_at(t), view_center_at(t)
SvgEpisode          Scene -> SVG transform (normalized coords, width=1)
                       scene_to_svg(point, t)
```

Raster and maize transforms (SVG -> pixel -> character) are handled internally by Mango rather than as separate Episode layers. Rasterization is done in `mango/render.rs` (via `mango/svg.rs` for SVG-to-pixel) and character mapping in `mango/templates/mod.rs`.

Key design decisions:
1. **ViewportEpisode is the only stateful layer** -- it precomputes the entire camera trajectory on construction using critically damped spring dynamics.
2. **Transforms are time-indexed** -- all coordinate transforms take a `scene_time: f64` parameter, allowing queries at any point in the episode.
3. **Podium position** -- ego vehicle appears at ~30% from left (not center), achieved via `podium_offset = -visible_width * 0.3`.

### SceneEpisode

Raw episode data with on-demand interpolation.

- `from_frames(frames)` -- construct from Vec<FrameState>
- `n_frames()` -- number of discrete timesteps
- `duration()` -- total episode duration in seconds
- `state(index)` -- get state at discrete timestep
- `state_at(scene_time)` -- get interpolated state at arbitrary time (linear interpolation of x, y, heading between bracketing frames)
- `crash_frame()` -- find crash frame index, if any

### ViewportEpisode

Camera tracking with precomputed trajectory. The only layer with history-dependence.

- `new(scene, config)` -- construct and precompute trajectory
- `view_x(index)` -- viewport center at discrete timestep
- `view_x_at(scene_time)` -- interpolated viewport center
- `view_center_at(scene_time)` -- viewport center (x, y)
- `visible_bounds_at(scene_time)` -- scene bounds visible at time t
- `scale()` -- SVG units per meter

Spring dynamics (critically damped):
```
error = ego_x - viewport_x - podium_offset
v_error = ego_velocity - viewport_velocity
acceleration = omega^2 * error + 2 * omega * v_error
```

### SvgEpisode

Stateless transform from scene to SVG coordinates.

- `scene_to_svg(point, scene_time)` -- transform scene point to SVG
- `bounds()` -- SVG dimensions (width=1.0, height varies with terminal aspect ratio)

## Coordinate types

Defined in `worlds/coords.rs`:

| Type | Description |
|---|---|
| `ScenePoint` | Position in simulation world coordinates (meters) |
| `SvgPoint` | Position in normalized SVG canvas units |
| `SceneBounds` | Axis-aligned bounding box in scene coordinates |

## Zoom model

Aspect-ratio and resolution independent. The diagonal-based model ensures consistent behavior:

1. Define `UNZOOMED_CANVAS_DIAGONAL_IN_METERS = 180.0` (constant)
2. At zoom=1.0x, the diagonal from top-left to bottom-right spans exactly that distance
3. At zoom=2.0x, the diagonal spans half that (90.0m), objects appear twice as large
4. Works identically for TUI output (maize) and image/video output (raster)

Derivation, given canvas dimensions `(width, height)` in any units:
- Canvas diagonal length: `d = sqrt(width^2 + height^2)`
- Scene diagonal in meters: `scene_diag = unzoomed_canvas_diagonal_in_meters / zoom`
- Scale factor: `scale = d / scene_diag` (canvas units per meter)

For non-square pixels/characters (corn_aspro != 1), the diagonal accounts for the visual aspect ratio: `visual_height = height * corn_aspro`.

## Constants

Centralized in `worlds/mod.rs`:

| Constant | Value | Description |
|---|---|---|
| `UNZOOMED_CANVAS_DIAGONAL_IN_METERS` | 180.0 | Scene diagonal at zoom=1.0 |
| `DEFAULT_CORN_ASPRO` | 1.875 | Character height/width ratio |
| `DEFAULT_SECONDS_PER_TIMESTEP` | 0.1 | Simulation sub-step duration |

Mango-specific values are in `mango/templates/mod.rs`:

| Value | Description |
|---|---|
| `PISTON_WIDTH` | Always 2 pixels per character width |
| `MangoMode::piston_height()` | Pixels per character height, varies by mode (method on `MangoMode` enum) |

`piston_height()` returns: QuadrantsOnly=4, Sextants=6, Octants=8, SextantsOctants=12. See [[Octane graphics and rendering]] for all four character modes.

## Mango internals

### MSE template matching

For each character cell, Mango:

1. Extracts a pixel grid (piston-sized) from the rasterized SVG
2. For each candidate pattern across all enabled character sets:
   - Compute optimal foreground and background colors as mean RGB of covered/uncovered pixels
   - Calculate Mean Squared Error between pixel colors and pattern's fg/bg assignment
3. Select the pattern with lowest MSE
4. Output ANSI escape codes for fg/bg colors plus the character

The MSE computation uses an algebraic identity to avoid a second pixel loop: `sum((x-mu)^2) = sum(x^2) - (sum(x))^2/n`. Complement symmetry halves the search space (pattern P and pattern ~P have the same MSE with swapped fg/bg).

### MangoConfig

```rust
pub struct MangoConfig {
    pub n_cols: u32,         // Output width in characters
    pub n_rows: u32,         // Output height in characters
    pub use_sextants: bool,  // Sextant characters (2x3 grid)
    pub use_octants: bool,   // Octant characters (2x4 grid)
}
```

### Pixel grid layout (sextant mode)

```
  col 0   col 1
+-------+-------+
| pix 0 | pix 1 |  row 0  \
+-------+-------+          | sextant row 0
| pix 2 | pix 3 |  row 1  /
+-------+-------+
| pix 4 | pix 5 |  row 2  \
+-------+-------+          | sextant row 1
| pix 6 | pix 7 |  row 3  /
+-------+-------+
| pix 8 | pix 9 |  row 4  \
+-------+-------+          | sextant row 2
|pix 10 |pix 11 |  row 5  /
+-------+-------+
```

### ANSI output format

Each character is output with explicit 24-bit true color fg and bg:

```
\x1b[38;2;R;G;Bm\x1b[48;2;R;G;Bm{char}
```

Lines are separated by `\x1b[0m\n` (reset + newline), with a final `\x1b[0m` reset.

### API example

```rust
use octane::mango::{render_svg_to_ansi, MangoConfig};

let config = MangoConfig {
    n_cols: 40,
    n_rows: 20,
    use_sextants: true,
    use_octants: false,
};

let ansi_output = render_svg_to_ansi(svg, &config)?;
println!("{}", ansi_output);
```

## Performance

Benchmark baseline (debug build, 120x40 terminal):

| Stage | Time | Throughput |
|---|---|---|
| Episode construction | 248us | (1000 frames) |
| Coordinate transforms | 31ns/point | 31.7M points/s |
| SVG generation | 112us/frame | 8909 fps |
| Mango rendering | ~12ms/frame | 84 fps |
| **Total pipeline** | ~12ms/frame | **83 fps** |

The Mango SVG-to-ANSI rendering is the bottleneck. 30 FPS target is easily achievable.
