# Octane Overview

Related: [[Octane navigation and playback]] | [[Octane key bindings]] | [[Octane graphics and rendering]] | [[Octane behavior explorer]] | [[Octane configuration]] | [[Octane architecture]]

Octane is a Rust-based terminal UI for browsing [[Trainer training]] runs. It renders highway scenes from [[HighJax environment]] episodes using Unicode block characters (the Mango renderer). Built with ratatui for TUI, resvg for SVG rasterization, and rayon for parallel rendering.

## UI layout

The top of the screen has a tab bar with two tabs: **1 Runs** and **2 Behaviors** (press `1` or `2` to switch). The Runs tab layout is shown below; see [[Octane behavior explorer]] for the Behaviors tab.

```
+------------------------------------------+
| Tab Bar (1 Runs | 2 Behaviors)           |
+------------------------------------------+
| Title Bar (poshed trek path)             |
+--------+---------------------------------+
| treks  |                                 |
+--------+           scene                 |
|parquets|        (highway viz)            |
+--------+---------------------------------+
| epochs |          dashboard              |
+--------+ (seeker + info + velocity bar)  |
|episodes|                                 |
+--------+---------------------------------+
| Status Bar (key hints)                   |
+------------------------------------------+
```

The screen is split into a sidebar on the left (four stacked list panes) and the main visualization area on the right (scene on top, dashboard below). A title bar shows the current trek path (poshed), and a status bar at the bottom shows key hints.

## The five panes

Focus cycles through these panes with Tab / Shift-Tab:

1. **Treks** - All discovered trek directories, sorted chronologically. Selecting a trek loads it. Selection state is saved per trek, so switching back restores where you were.

2. **Parquets** - Available parquet sources for the current trek (e.g. `sample_es`, breakdown parquets). Selecting a source rebuilds the epoch/episode lists from that parquet.

3. **Epochs** - Table showing epoch number, episode count, NReturn (normalized discounted return), and Survival (alive fraction percentage).

4. **Episodes** - Table for the selected epoch: episode index, frame count, NReturn, Survival.

5. **Highway (scene)** - Main visualization. Renders the highway with ego vehicle, NPCs, road markings, terrain, headlights, brakelights, velocity arrows, and action distribution overlays.

Each pane title has a highlighted mnemonic letter (t, a, o, e, s) for direct focus jumps.

Below the scene is the **dashboard**, a non-focusable info area containing a seeker bar (progress through the episode), playback info (timestep, action, reward, and -- when present -- V, Return, LogP, Adv, NAdv), vehicle info (velocity bar, heading, NPC count), and location info (epoch, episode, timestep).

## Data model

The hierarchy is: **Trek -> Epoch -> Episode -> Frame**.

A trek is a training run directory. It contains epochs, each with multiple episodes. Each episode is a sequence of frames (one per sub-timestep). Frame data includes vehicle positions, headings, speeds, actions, rewards, and optionally action distributions and attention weights.

Data is loaded from `sample_es.parquet` (preferred, used by modern HighJax runs; `.p` and `.pq` are accepted as backward-compat fallbacks) or legacy `epoch_NNN/episode_NNN.json` directories. The parquet format uses an index for efficient random access.

## Trek discovery

On startup, Octane scans `~/.highjax/t/` (or `$HIGHJAX_HOME/t/`) for directories containing `meta.yaml` or `sample_es.parquet`. Results are sorted by directory name (timestamp format like `2026-01-31-02-13-44-322216`), so the most recent trek appears last.

Timing metadata comes from `meta.yaml`: `seconds_per_t` (policy timestep duration) and `seconds_per_sub_t` (sub-timestep duration, for environments with sub-stepping).

## Producing treks

Treks are produced by running `highjax-trainer train`, which writes outputs to `~/.highjax/t/` (or `$HIGHJAX_HOME/t/`). Each training run creates a timestamped directory (e.g. `2026-01-31-02-13-44-322216`) containing `meta.yaml`, epoch data, and optionally `sample_es.parquet` with sampled episodes.

## CLI usage

```bash
# Launch TUI, auto-find latest trek
octane

# Open a specific trek
octane -t ~/.highjax/t/2026-01-31-02-13-44-322216

# Open a breakdown parquet directly (auto-finds trek via meta.yaml)
octane -t ~/.highjax/t/2026-.../breakdown/2026-..._a0_e200-200_.../es.parquet

# Jump to a specific position on launch
octane --epoch 42 -e 3 --timestep 7.40

# List epochs non-interactively
octane --list-epochs

# Render a single frame to terminal, SVG, or PNG
octane draw                          # terminal output
octane draw --svg out.svg            # SVG file
octane draw --png out.png            # PNG file
octane draw --epoch 5 --episode 2 --timestep 100

# Render a behavior scenario
octane draw --behavior my_scenario --scenario 0
octane draw --behavior my_scenario --scenario 2 --hardcoded-action left

# Export an episode to video
octane animate -o video.mp4
octane animate -o video.mp4 --epoch 5 --episode 0 --speed 2.0 --fps 30

# Mango renderer utilities
octane mango benchmark               # run render benchmark
octane mango svg scene.svg            # render SVG to terminal
octane mango image photo.png          # render image to terminal
octane mango chars                    # show character set

# Query a running Octane instance via zoltar IPC
octane zoltar ping
octane zoltar genco
octane zoltar press g enter
octane zoltar navigate --epoch 10 --episode 3
octane zoltar pane metrics data

# Configuration
octane config fill-defaults           # create/update config file
octane config path                    # print config file path
```

## Key CLI arguments

| Argument | Description |
|---|---|
| `-t, --target PATH` | Trek directory or `.parquet` parquet file (auto-finds trek by walking up to `meta.yaml`) |
| `--epoch N` | Jump to epoch number N on launch |
| `-e, --episode N` | Jump to episode by `e` column number |
| `--timestep F` | Jump to timestep t-value (e.g. 48 or 3.40) |
| `--fps N` | Initial FPS (1-60) |
| `--speed F` | Playback speed multiplier (e.g. 2.0 = double speed) |
| `--zoom F` | Zoom level (e.g. 1.0 = default, 2.0 = zoomed in) |
| `--omega F` | Viewport smoothing spring constant |
| `--theme THEME` | Scene color theme (`dark` or `light`) |
| `--sidebar-width N` | Sidebar width in columns |
| `--prefs "K=V,..."` | Env-specific rendering preferences (e.g. `"debug_eye=true,velocity_arrows=always"`) |
| `--no-mango` | Disable Mango rendering (ASCII only) |
| `--sextants / --no-sextants` | Enable/disable sextant characters |
| `--octants / --no-octants` | Enable/disable octant characters |
| `--no-zoltar` | Disable zoltar IPC interface |
| `--zoltar-file-ipc` | Use file-based IPC for zoltar instead of Unix socket |
| `--list-epochs` | List epochs and exit |
| `-v, --verbose` | Enable verbose logging |

## Animate command

The `animate` subcommand renders an episode to an MP4 video using ffmpeg. It renders frames in parallel (10 threads via rayon), producing PNG intermediates that ffmpeg encodes. By default it stops at the crash frame if there is one.

```bash
octane animate -o episode.mp4 --epoch 10 --episode 0 --speed 1.0 --fps 30 --width 1920 --height 1080
```

## Configuration

Octane reads `~/.highjax/config.json` for all settings (colors, road geometry, rendering parameters, podium tracking, etc). The config supports hot-reload -- edit the file while Octane is running and changes apply immediately. See [[Octane configuration]] for details.
