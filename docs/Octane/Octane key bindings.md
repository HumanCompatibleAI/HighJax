# Octane Key Bindings

Related: [[Octane overview]] | [[Octane navigation and playback]] | [[Octane graphics and rendering]] | [[Octane behavior explorer]]

Full reference for all key bindings in Octane. Keys are organized by mode.

## Normal mode

### Quitting

| Key | Action |
|---|---|
| q | Quit |
| Esc | Quit |
| Ctrl-C | Quit |

### Focus

| Key | Action |
|---|---|
| Tab | Cycle focus forward |
| Shift-Tab | Cycle focus backward |
| t | Focus Treks pane |
| a | Focus Parquets pane |
| o | Focus Epochs pane |
| e | Focus Episodes pane |
| s | Focus Highway (scene) pane |

### List navigation (Treks, Parquets, Epochs, Episodes)

| Key | Action |
|---|---|
| Up | Select previous item |
| Down | Select next item |
| Home | First item |
| End | Last item |
| PageUp | Scroll up 80% of visible rows |
| PageDown | Scroll down 80% of visible rows |

### Frame navigation

| Key | Action |
|---|---|
| Left | Previous frame (Highway focused) |
| Right | Next frame (Highway focused) |
| j | 1 frame back (any focus) |
| k | 1 frame forward (any focus) |
| J | 5 frames back |
| K | 5 frames forward |
| h | First frame |
| l | Last frame |
| Home | First frame (Highway focused) |
| End | Last frame (Highway focused) |
| PageUp | Page back in frames (Highway focused) |
| PageDown | Page forward in frames (Highway focused) |

### Playback

| Key | Action |
|---|---|
| p | Toggle play/pause |

### Tab switching

| Key | Action |
|---|---|
| 1 | Switch to Runs tab |
| 2 | Switch to Behaviors tab (Behavior Explorer) |
| B | Switch to Behaviors tab (Behavior Explorer) |

### Modals

| Key | Action |
|---|---|
| ? | Open Help modal |
| r | Open Graphics modal |
| g | Open Jump Epoch modal |
| G | Open Jump Episode modal |
| / | Open Jump to Timestep modal |
| b | Open Behavior Scenario modal (capture a scenario) |
| W | Open Graphics modal focused on sidebar width |

### Utility

| Key | Action |
|---|---|
| d | Toggle debug overlay |
| c | Copy last toast text to clipboard |
| C | Copy octane command to clipboard |
| D | Copy debug info to clipboard |
| y | Dump frame state to JSON file |
| R | Refresh trek from disk |
| Ctrl+L | Copy log file path to clipboard |

## Graphics modal

Opened with `r`. Navigate controls with Up/Down or j/k, adjust values with Left/Right or h/l. Close with Enter, r, Esc, or q.

### Mnemonic jump keys

Each control has a letter that jumps directly to it:

| Key | Control |
|---|---|
| e | Theme (dark / light) |
| f | FPS (1-60) |
| s | Playback speed (0.25x-10.0x in 0.25 steps) |
| z | Zoom (0.1x-10.0x, multiplicative 1.1x steps) |
| w | Sidebar width (30-80, in steps of 2) |
| p | Podium marker (on/off toggle) |
| c | Scala (on/off toggle) |
| x | Sextants (on/off toggle) |
| o | Octants (on/off toggle) |
| v | Velocity arrows (never / on pause / always) |
| i | Action distribution (never / on pause / always) |
| t | Action distribution text (never / on pause / always) |
| a | Attention overlay (on/off toggle) |
| n | NPC text labels (never / on pause / always) |
| d | Debug eye (on/off toggle) |
| b | Light blend mode (normal, screen, lighten, ...) |

## Jump modals

Opened with `g` (epoch), `G` (episode), or `/` (timestep).

| Key | Action |
|---|---|
| 0-9 | Enter digit |
| Backspace | Delete last digit |
| Enter | Confirm jump |
| n | Repeat last successful jump value |
| Esc | Cancel |

## Behavior scenario modal

Opened with `b`. Used to capture action distribution scenarios.

| Key | Action |
|---|---|
| Tab | Cycle fields: Actions -> Name |
| Shift-Tab | Cycle fields backward |
| Enter | Confirm and save scenario |
| Esc | Cancel |

In the **Actions** field:
- Up/Down to navigate the action list (Left, Idle, Right, Faster, Slower)
- Space to quick-toggle weight: 0 becomes 1, nonzero becomes 0
- Digits, minus, dot to type numeric weights
- Backspace to delete last character

In the **Name** field:
- Type alphanumeric characters, underscore, or dash
- Down/Up to navigate autocomplete suggestions (from existing behaviors in `~/.highjax/behaviors/{env_name}/`)
- Enter on a suggestion to accept it, Enter with no suggestion to confirm the scenario

## Behaviors tab (Behavior Explorer)

Full-screen three-pane mode for browsing and editing behavior scenarios. Opened with `2` or `B`. The three panes are Behaviors, Scenarios, and Preview.

### Browse mode

| Key | Action |
|---|---|
| b | Jump to Behaviors pane |
| s | Jump to Scenarios pane |
| p | Jump to Preview pane |
| Tab | Cycle to next pane |
| Shift-Tab | Cycle to previous pane |
| Up/Down | Navigate within pane |
| Home/End | Jump to first/last item in list |
| Enter | Navigate: from Behaviors moves to Scenarios; from Scenarios navigates to source location |
| e | Enter edit mode for selected scenario |
| n | Create new scenario (Scenarios pane) |
| c | Duplicate selected scenario (Scenarios pane) |
| d (twice) | Delete selected scenario or behavior |
| N | Create new behavior (Behaviors pane) |
| 1 / q | Return to Runs tab |

### Editor mode

| Key | Action |
|---|---|
| Up/Down | Navigate fields |
| Home/End | Jump to first/last editor field |
| Left/Right | Adjust value |
| Shift+Left/Right | Fine adjust (0.1x step) |
| a | Add NPC |
| x (twice) | Remove NPC at cursor |
| Enter | Save and exit editor |
| Esc / q | Cancel and exit editor |

See [[Octane behavior explorer]] for full details.
