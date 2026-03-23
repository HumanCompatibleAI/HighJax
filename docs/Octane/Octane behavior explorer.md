# Octane Behavior Explorer
Related: [[Octane overview]] | [[Octane key bindings]]

## Overview

Behavior crafting lets you capture specific moments from episodes as "scenarios" and assemble them into named behaviors. These behaviors define measurable properties of a policy, like "hesitating before lane changes" or "aggressive acceleration". Once defined, they can be measured across all training epochs.

The idea: you're watching an episode in Octane, you see the car doing something interesting, and you capture that moment. Do this a few times, give it a name, and you have a behavior metric you can track.

## Scenario Capture

Press `b` on any frame that has action distribution data (the frame must have been saved with policy outputs). This opens the Behavior Scenario modal.

Steps:
1. **Set action weights** — use up/down arrows to select an action. Space quick-toggles (0→1 or nonzero→0), and you can type numeric values directly (digits, minus, dot); Backspace deletes. The five actions are: LEFT, IDLE, RIGHT, FASTER, SLOWER. Positive weight means "want this action," negative means "don't want this action."
2. **Enter name** — type a behavior name. Autocomplete suggests existing behavior names (arrow down to select). If you pick an existing name, the scenario is added to that behavior.
3. **Press Enter** to confirm.

Tab cycles between the two fields (Actions, Name).

Each scenario captures:
- The traffic state (ego speed, NPC relative positions)
- Per-action weights (positive, negative, or zero for each action)
- Source info (epoch, episode, timestep) for reference

Scenarios are saved as JSON at `~/.highjax/behaviors/{env_name}/{name}.json`. Each behavior file can have multiple scenarios. See the behavior JSON files in ~/.highjax/behaviors/ for the format.

## Behavior Metric

For a behavior with N scenarios, the metric is: the weighted average of per-action values across all (scenario, action) pairs. For positive weights, the value is P(action|obs); for negative weights, the value is 1−P(action|obs). This keeps the metric in the range 0 to 1.

Higher metric = the policy aligns more with the specified action preferences across the captured scenarios. This is what gets tracked across training epochs in epochia.parquet.

## Behavior Explorer

The Behavior Explorer is a full-screen tab for browsing, editing, and creating behavior scenarios. Access it via the `2` key (tab switching) or `B` (Shift-B) from the Runs tab.

### Tab system

Octane has two tabs in the top bar: `1 Runs` and `2 Behaviors`. Press `1` or `2` to switch tabs. The Behaviors tab is the Behavior Explorer.

Behaviors load in the background at startup, so switching to the Behaviors tab is instant. After capturing a scenario via `b` in the Runs tab, the Behaviors tab auto-reloads.

### Layout

Two columns, three panes: left column stacks **behaviors** + **scenarios**, right column has **preview** with a dashboard strip below it.

- **behaviors pane** (`b`) — lists all behaviors (user and preset), shows name and scenario count. `*` prefix for presets.
- **scenarios pane** (`s`) — shows scenarios for the selected behavior: ego speed, NPC count, and action weights.
- **preview pane** (`p`) — live highway render of the selected scenario via the mango pipeline. Observation-only scenarios show "Preview unavailable". A dashboard strip below shows velocity bar, heading, NPC count, and source location.

Pane mnemonics `b`, `s`, `p` jump directly to a pane. Tab also cycles panes. Up/Down navigates within each pane.

### Browse mode keys

| Key | Pane | Action |
|-----|------|--------|
| `b` | Any | Jump to behaviors pane |
| `s` | Any | Jump to scenarios pane |
| `p` | Any | Jump to preview pane |
| `e` | Any | Enter edit mode for selected scenario |
| `n` | Scenarios | Create new empty scenario |
| `c` | Scenarios | Duplicate selected scenario |
| `d` (twice) | Scenarios | Delete selected scenario |
| `d` (twice) | Behaviors | Delete selected behavior |
| `N` | Behaviors | Create new behavior (text input) |
| `Enter` | Scenarios | Navigate to scenario source in Runs tab |
| `r` | Any | Open Graphics dialog |
| `1` | Any | Switch to Runs tab |
| `q` | Any | Return to Runs tab |
| `Tab` | Any | Cycle to next pane |

### Editor mode

Two columns: **editor fields** (left) | **live preview** (right)

Fields show the 5 action weight fields (w:left, w:idle, w:right, w:faster, w:slower), ego speed, and per-NPC rel_x, rel_y, speed values. Left/Right arrows adjust the selected field (Shift for fine adjustment). The preview re-renders in real time as values change.

| Key | Action |
|-----|--------|
| `↑↓` | Navigate fields |
| `←/→` | Adjust value (step depends on field type) |
| `Shift+←/→` | Fine adjust (0.1× step) |
| `a` | Add NPC at default position |
| `x` (twice) | Remove the NPC at cursor |
| `Enter` | Save to JSON and exit editor |
| `Esc` | Cancel and exit editor |
| `q` | Cancel and exit editor |

Step sizes: action weights ±0.5 (fine ±0.1), clamped to −10..10. Ego speed ±1.0 (fine ±0.1), rel_x ±5.0 (fine ±1.0), rel_y ±1.0 (fine ±0.1), NPC speed ±1.0 (fine ±0.1). Clamping: speed 0-50, rel_x ±200, rel_y ±20.

### Creating behaviors

Press `N` in the behaviors pane to enter text input mode. Type a name (letters, digits, dashes, underscores), press Enter to create. The new behavior starts with one default scenario (ego speed 20, no NPCs, idle weight 1).

### Policy action probabilities in preview

When the currently-selected epoch (from the Runs tab) has a snapshot, the preview pane automatically overlays policy action probabilities on the highway scene — the same arrows and percentage text shown in the Runs tab. This lets you see what the policy would do in each scenario at a given training epoch without switching tabs.

The probabilities are fetched in a background thread via `highjax brain p-by-action` and cached per (behavior, epoch) pair. The cache invalidates when you switch behaviors or select a different epoch. If the selected epoch has no snapshot, the preview renders without probabilities.

The action distribution and text overlays can be toggled via the Graphics dialog (`r` key) using the ActionDistribution and ActionDistributionText controls.

## Drawing behavior scenarios

You can render a behavior scenario to the terminal (or SVG/PNG) without loading a trek:

```bash
# Render first scenario of "hathi" to terminal
octane draw --behavior hathi

# Render a specific scenario
octane draw --behavior hathi --scenario 2

# Output as PNG
octane draw --behavior hathi --png hathi.png --cols 200 --rows 60
```

This works with both preset behaviors (hathi, freebo) and user-defined behaviors in `~/.highjax/behaviors/{env_name}/`. Only state-based scenarios can be drawn — observation-based scenarios don't have enough geometric information for rendering.

Octane looks for behaviors in this order:
1. User behaviors dir (`~/.highjax/behaviors/{env_name}/`)
2. Preset behaviors shipped with the environment package

## Workflow Example

1. Browse training in Octane, navigate to an epoch where the car drives well
2. Find a frame where the car hesitates before merging — press `b`
3. Select SLOWER and type `1` (or press Space), name it "hesitation", press Enter
4. Find another frame showing similar hesitation — press `b` again, same name. Now the behavior has two scenarios.
5. Maybe find a third example from a different epoch. More scenarios = more robust metric.
6. Press `2` to switch to the Behaviors tab, browse your behaviors
7. The behavior metric is tracked in epochia.parquet across training epochs

## Key files

- `octane/src/ui/explorer.rs` — explorer rendering
- `octane/src/app/explorer.rs` — loading, navigation, preview
- `octane/src/app/explorer_editing.rs` — editor fields, CRUD operations
- `octane/src/app/input.rs` — key handling
- `octane/src/app/state.rs` — state types (ExplorerMode, EditorField)
- `octane/src/data/behaviors.rs` — behavior loading, `fetch_action_probs`
