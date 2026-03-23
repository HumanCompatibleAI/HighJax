# Octane Graphics and Rendering
Related: [[Octane overview]] | [[Octane navigation and playback]] | [[Octane key bindings]] | [[Octane configuration]] | [[Octane architecture]]

## Overview

Octane renders highway driving episodes as animated terminal graphics. The rendering pipeline transforms raw simulation data through five coordinate layers, generates SVG scenes, rasterizes them, and converts to colored terminal characters using the Mango renderer. The whole thing runs at up to 60 FPS in a standard terminal.

The rendering code lives in `octane/src/render/` (SVG generation) and `octane/src/mango/` (SVG-to-terminal conversion).

## Rendering Pipeline

Five coordinate layers, each wrapping the previous one:

### 1. Scene (meters)

Raw simulation coordinates from Highway. `SceneEpisode` wraps the discrete frame data and provides `state_at(scene_time)` with linear interpolation between adjacent frames. Positions are in meters, time is in seconds. This is the base layer with no camera logic.

Defined in `octane/src/worlds/scene_episode.rs`.

### 2. Viewport (camera-relative meters)

Camera tracking system. `ViewportEpisode` is the only stateful layer - it precomputes the entire camera trajectory on construction using critically damped spring dynamics. The spring tracks the ego vehicle with configurable omega (stiffness) and damping ratio.

Podium position: ego vehicle is offset to ~30% from the left edge of the visible area (configurable via `podium.offset`). This gives more lookahead space in the driving direction.

Defined in `octane/src/worlds/viewport_episode.rs`.

### 3. SVG (normalized surface, width=1.0)

Converts scene coordinates to a normalized SVG coordinate space. Width is always 1.0, height varies with terminal aspect ratio. This layer is stateless - just applies the viewport transform.

Defined in `octane/src/worlds/svg_episode.rs`.

### 4. Raster (pixel buffer)

Pixel-level coordinates for SVG rasterization via resvg. Resolution is determined by terminal dimensions and `pixels_per_corn_diagonal` (default 25.3, derived from typical 8x15 pixel character cells).

Rasterization is implemented in `octane/src/mango/render.rs`.

### 5. Maize (character grid)

Terminal character coordinates. Each character cell covers a small block of pixels (the "piston"), whose size depends on the active character mode. This is what actually gets printed.

Character mapping is implemented in `octane/src/mango/templates/mod.rs`.

The coordinate types (`ScenePoint`, `SvgPoint`, `SceneBounds`) are all defined in `octane/src/worlds/coords.rs`.

## SVG Content

The function `render_highway_svg_from_episode()` in `octane/src/render/highway_svg.rs` generates the full scene SVG. Elements are drawn in this order:

1. **Background** - solid color fill (#030100 by default)
2. **Terrain blobs** - procedural organic shapes (bezier-curve paths) placed via noise threshold, avoiding the road area
3. **Road surface** - gray rectangle spanning all lanes
4. **Road edges** - white border strokes at top and bottom of the road
5. **Lane dividers** - lighter gray strokes between lanes
6. **Headlight cones** - triangular light projections with shadow occlusion from other vehicles
7. **Brakelight cones** - red triangular projections behind decelerating vehicles (deceleration below threshold triggers them)
8. **Car templates** - SVG car shapes with gradient fills. Ego is silver (#b4b4b4), red when crashed (#ff6464). NPCs get colors via OKLCH with golden-angle hue spacing.
9. **Velocity arrows** - optional arrows showing vehicle speed/heading
10. **Action distribution overlay** - optional arrows/circle showing the policy's action probabilities on the ego vehicle
11. **Hardcoded action arrow** - optional filled-polygon arrow for a single action direction (used in behavior scenario rendering)
12. **Attention bars** - optional progress bars on NPC vehicles showing attention weights
13. **NPC text labels** - optional labels on NPC vehicles showing spawn index, speed, and collision warning
14. **Podium marker** - optional red vertical line at the ego target position (~30% from left edge)
15. **Scale bar** - drawn in the bottom-right corner showing distance reference

Light cones (headlights and brakelights) are wrapped in a configurable SVG blend mode group (default: "lighten").

## Mango Renderer

Mango is the SVG-to-terminal converter. It lives in `octane/src/mango/`. The process:

1. **Rasterize**: resvg renders the SVG to a pixel buffer at the exact dimensions needed
2. **Template matching**: For each character cell, sample a block of pixels (the "piston"). Each enabled character set provides patterns (binary masks of which sub-cells are "foreground"). Mango tests all enabled patterns and picks the one that minimizes MSE between the 2-color approximation and the actual pixel colors.
3. **Text overlays**: SVG `<text>` elements with `data-mango-fg` and `data-mango-bg` attributes are parsed and rendered as actual terminal characters on top of the raster output. Used for action distribution percentages and NPC text labels.
4. **ANSI output**: Each character gets 24-bit true color codes (`\x1b[38;2;R;G;Bm` for foreground, `\x1b[48;2;R;G;Bm` for background).

The MSE computation uses an algebraic identity to avoid a second pixel loop: sum((x-mu)^2) = sum(x^2) - (sum(x))^2/n. Complement symmetry halves the search space (pattern P and pattern ~P have the same MSE with swapped fg/bg).

Rendering is parallelized across character cells using rayon.

## Character Modes

Four rendering modes, determined by which character sets are enabled:

| Mode | Grid | Patterns | Piston Height | Unicode |
|---|---|---|---|---|
| Quadrants only | 2x2 | 16 | 4 px | Basic block chars |
| Sextants | 2x3 | 64+16 = 80 | 6 px | Unicode 13.0 (U+1FB00) |
| Octants | 2x4 | 256+16 = 272 | 8 px | Unicode 16.0 (U+1CD00) |
| Both | 2x3 + 2x4 | 64+256+16 = 336 | 12 px (LCM of 6,8) | Both ranges |

Piston width is always 2 pixels. The piston height varies by mode and determines the vertical pixel resolution per character cell.

- **Quadrants** (default fallback): 2x2 grid, 16 patterns. Widest terminal support.
- **Sextants**: 2x3 grid, 64 patterns. Needs a terminal/font supporting Unicode 13.0 Symbols for Legacy Computing block.
- **Octants** (default): 2x4 grid, 256 patterns. Needs Unicode 16.0. Highest quality per-character-set, as every possible 2x4 bit pattern has a character.
- **Both**: Combines sextants and octants at 2x12 pixel pistons (LCM of 6 and 8). 336 total patterns. Best quality but tallest pixels.

Toggle these in the Graphics modal (see below) or via CLI flags (`--sextants`, `--octants`, `--no-sextants`, `--no-octants`).

## Graphics Modal Controls

Press `r` to open the Graphics modal. Each control has a mnemonic key to jump to it, and left/right arrows to adjust:

| Control | Key | Type | Range |
|---|---|---|---|
| Theme | e | enum | dark / light |
| FPS | f | int | 1-60 |
| Playback Speed | s | float | 0.25-10.0 (step 0.25) |
| Zoom | z | float | 0.1-10.0 (multiply by 1.1) |
| Sidebar Width | w | int | 30-80 (step 2) |
| Podium Marker | p | toggle | on/off |
| Scale Bar | c | toggle | on/off |
| Sextants | x | toggle | on/off |
| Octants | o | toggle | on/off |
| Velocity Arrows | v | tri-state | never / on pause / always |
| Action Distribution | i | tri-state | on pause / always / never |
| Action Dist Text | t | tri-state | on pause / always / never |
| Attention | a | toggle | on/off |
| NPC Text | n | tri-state | on pause / always / never |
| Debug Eye | d | toggle | on/off |
| Light Blend Mode | b | enum | 10 SVG blend modes |

The 10 blend modes are: normal, screen, lighten, color-dodge, multiply, overlay, hard-light, soft-light, difference, exclusion.

Press `r` or `Enter` to close the modal.

## Export Commands

Octane can export frames and videos from the command line:

**Single frame to SVG:**
```
octane draw -t /path/to/trek --svg out.svg --cols 120 --rows 40
```

**Single frame to PNG:**
```
octane draw -t /path/to/trek --png out.png --cols 120 --rows 40
```

**Episode to MP4 video:**
```
octane animate -t /path/to/trek -o video.mp4 --fps 30 --width 1920 --height 1080
```

The `animate` command renders all frames (or a range with `--start`/`--end`) as individual PNGs using parallel rayon threads, then stitches them with ffmpeg. It supports `--speed` for playback speed and `--omega` for camera smoothing.

Both `draw` and `animate` accept `--epoch` and `--episode` to select which episode to render.
