# Octane Glossary

Related: [[Octane docs]]

Quick reference for Octane-specific terms. See the linked docs for full details.

## Core concepts

**Octane**: Rust-based terminal UI for browsing HighJax training runs, episodes, and frames. Renders a highway scene using SVG-to-terminal conversion. See [[Octane overview]].

**Trek**: A training run directory. Top of the data hierarchy: Trek -> Epoch -> Episode -> Frame. Discovered from `~/.highjax/t/` by scanning for `meta.yaml` or `sample_es.parquet`. See [[Octane overview]].

**Frame**: One discrete sub-timestep of data within an episode. Contains vehicle positions, headings, speeds, actions, rewards, and optionally action distributions and attention weights. See [[Octane overview]].

**Scene time**: A continuous float representing seconds from episode start. Supports smooth interpolation between sub-steps during playback. Frame index derived as `floor(scene_time / seconds_per_sub_t)`. See [[Octane navigation and playback]].

**Trek discovery**: The startup process scanning configured directories for valid trek directories, sorted chronologically. See [[Octane overview]].

## UI layout

**Treks pane**: Top-left sidebar showing all discovered trek directories. Mnemonic key: `t`. See [[Octane overview]].

**Parquets pane**: Sidebar pane listing available parquet sources for the current trek (e.g. `sample_es`, breakdown parquets). Mnemonic key: `a`. See [[Octane overview]].

**Epochs pane**: Middle-left sidebar showing per-epoch statistics (episode count, mean reward, score, snapshot indicator). Mnemonic key: `o`. See [[Octane overview]].

**Episodes pane**: Bottom-left sidebar showing per-episode statistics for the selected epoch. Mnemonic key: `e`. See [[Octane overview]].

**Highway pane / Scene pane**: Main visualization area showing the rendered highway scene. Mnemonic key: `s`. See [[Octane overview]].

**Dashboard**: Below the Highway pane. Contains seeker bar, frame info (with additional RL metrics when available), and velocity bar. Non-focusable (no mnemonic key). See [[Octane overview]].

**Seeker bar**: Progress bar showing current position within episode playback. See [[Octane overview]].

**Velocity bar**: Color-ramped bar (green to red) indicating ego vehicle speed. See [[Octane overview]].

**Mnemonic key**: A highlighted letter in each pane title for direct pane navigation: `t`, `a`, `o`, `e`, `s`. See [[Octane key bindings]].

## Rendering

**Mango renderer**: The SVG-to-terminal converter. Takes a rasterized SVG pixel buffer, performs template matching against Unicode block character patterns, and outputs 24-bit true-color ANSI characters. See [[Octane graphics and rendering]].

**Piston**: The block of pixels that a single terminal character cell covers in Mango's template matching. Always 2 pixels wide; height varies by character mode (4, 6, 8, or 12 pixels). See [[Octane architecture]].

**Corn**: A terminal character bounding box -- the rectangular area one character occupies on screen. See [[Octane architecture]].

**Maize**: The character grid output of the Mango renderer -- the full Unicode picture made of corn cells. See [[Octane architecture]].

**Corn aspro (`corn_aspro`)**: Visual aspect ratio (height/width) of a single terminal character as it appears on screen. Default 1.875. See [[Octane configuration]].

**Aspro**: Octane's term for aspect ratio, always expressed as height/width. See [[Octane architecture]].

**Character modes**: Four rendering quality modes based on Unicode block character sets: Quadrants (2x2, 16 patterns), Sextants (2x3, 80 patterns), Octants (2x4, 272 patterns), Both (2x3+2x4, 336 patterns). See [[Octane graphics and rendering]].

**Quadrants**: Default fallback character mode using 2x2 block characters. Widest terminal support. See [[Octane graphics and rendering]].

**Sextants**: Character mode using Unicode 13.0 Symbols for Legacy Computing (U+1FB00). 2x3 grid, higher quality than quadrants. See [[Octane graphics and rendering]].

**Octants**: Character mode using Unicode 16.0 characters (U+1CD00). 2x4 grid, highest quality -- every possible bit pattern has a character. Default enabled mode. See [[Octane graphics and rendering]].

**MSE template matching**: Mango's algorithm: for each candidate Unicode pattern, compute optimal fg/bg colors and MSE against actual pixels, pick lowest MSE. See [[Octane graphics and rendering]].

**Complement symmetry**: Optimization in template matching: pattern P and its complement ~P have the same MSE with swapped fg/bg, halving the search space. See [[Octane graphics and rendering]].

## Camera and viewport

**Podium position**: Horizontal position where the ego vehicle is anchored -- approximately 30% from the left edge, giving more lookahead space. See [[Octane graphics and rendering]].

**Viewport**: The second coordinate layer, representing camera-relative meters. `ViewportEpisode` precomputes the camera trajectory using critically damped spring dynamics. See [[Octane architecture]].

**Critically damped spring**: The camera tracking mechanism. With damping ratio = 1.0, the camera follows ego with no oscillation. Controlled by `omega` (stiffness). See [[Octane configuration]].

**Omega**: Spring natural frequency controlling how tightly the camera tracks the ego. Low values (0.01-0.05) give a drifting camera; high values (1.0+) make it snap. See [[Octane configuration]].

**Episode stack**: The three-layer coordinate transformation pipeline: SceneEpisode -> ViewportEpisode -> SvgEpisode. Raster and maize transforms (SVG -> pixel -> character) are handled internally by the Mango renderer rather than as separate Episode layers. See [[Octane architecture]].

**`pixels_per_corn_diagonal`**: Raster resolution: how many pixels span the diagonal of one character cell during SVG rasterization. Default 25.3. See [[Octane configuration]].

**`unzoomed_canvas_diagonal_in_meters`**: Scene distance in meters that the canvas diagonal spans at zoom=1.0. Default 180.0m. See [[Octane configuration]].

## Overlays and effects

**Headlight cones**: Triangular SVG projections in front of vehicles simulating headlights. Configurable opacity and blend mode. See [[Octane graphics and rendering]].

**Brakelight cones**: Red triangular projections behind decelerating vehicles. Triggered when deceleration exceeds `brakelight_deceleration_threshold_m_s2` (default 5.0 m/s^2). See [[Octane graphics and rendering]].

**Action distribution overlay**: Optional overlay on ego showing policy action probabilities as arrows and/or percentages. Tri-state: never / on pause / always. See [[Octane graphics and rendering]].

**Attention bars**: Optional progress bars on NPC vehicles showing the policy's attention weights directed at each NPC. See [[Octane graphics and rendering]].

**Debug eye**: Mode overriding attention overlay opacities with fixed values (100%, 60%, 30%) for debugging. Activated via `--prefs "debug_eye=true"` or toggled in the graphics modal. See [[Octane key bindings]].

**Velocity arrows**: Optional arrows on vehicles showing speed and heading direction. Tri-state: never / on pause / always. See [[Octane graphics and rendering]].

**Terrain blobs**: Procedurally generated organic shapes (bezier-curve paths) in the background, avoiding the road area, for visual texture. See [[Octane graphics and rendering]].

**Light blend mode**: SVG compositing blend mode for headlight/brakelight groups. Default "lighten". See [[Octane configuration]].

**NPC colors (OKLCH)**: NPCs colored using OKLCH color space with golden-angle hue spacing for perceptually distinct hues. See [[Octane graphics and rendering]].

## Playback

**Wall-clock anchor system**: Playback mechanism: records wall time and scene_time on play; each frame computes `scene_time = start + elapsed * speed`. Frame-rate-independent. See [[Octane navigation and playback]].

**Playback speed**: Multiplier for how fast scene time advances vs wall clock. Range 0.25x-10.0x, default 2.0x. See [[Octane navigation and playback]].

**Frame position memory**: When switching episodes/epochs, saves and restores the current frame position per (epoch, episode) pair. See [[Octane navigation and playback]].

**Trek state memory**: When switching treks, saves and restores full selection state (epoch, episode, scroll positions). See [[Octane navigation and playback]].

## Modals

**Graphics modal**: `r`-key modal for adjusting rendering settings in real time: FPS, playback speed, zoom, character modes, overlays. See [[Octane key bindings]].

**Jump modals**: `g` (epoch), `G` (episode), `/` (frame) modals for navigating directly to a specific position by typing a number. See [[Octane key bindings]].

**Behavior Scenario modal**: `b`-key modal for capturing a scenario at the current frame. Select actions, name the behavior, set weight. See [[Octane behavior explorer]].

**Behaviors tab**: Full-screen tab (`2` or `B`) for browsing, editing, and creating behavior scenarios. Three-pane layout: behaviors list, scenarios list, live preview. Replaces the old Behavior Manager modal. See [[Octane behavior explorer]].

## CLI subcommands

**`octane draw`**: Renders a single frame to terminal, SVG, or PNG without launching the TUI. See [[Octane overview]].

**`octane animate`**: Renders an entire episode to MP4 via ffmpeg with parallel rayon rendering. See [[Octane overview]].

**`octane mango`**: Mango renderer utilities: benchmark, render SVG to terminal, render image to terminal, show character set. See [[Octane overview]].
