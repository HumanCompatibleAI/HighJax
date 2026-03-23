# Octane Navigation and Playback

Related: [[Octane overview]] | [[Octane key bindings]] | [[Octane graphics and rendering]]

## Pane focus

Tab cycles focus forward through the panes: Treks -> Parquets -> Epochs -> Episodes -> Highway -> Treks. Shift-Tab goes backward. You can also jump directly to any pane with its mnemonic key: t (Treks), a (Parquets), o (Epochs), e (Episodes), s (Scene/Highway).

The focused pane has a highlighted border. The mnemonic letter in each pane title is shown in gold.

## Navigation by pane

| Pane | Up/Down | Left/Right | Home/End | PageUp/Down |
|---|---|---|---|---|
| Treks | Select trek (loads it) | - | First/last trek | 80% page |
| Parquets | Select parquet source (rebuilds epochs) | - | First/last source | 80% page |
| Epochs | Select epoch (resets episode) | - | First/last epoch | 80% page |
| Episodes | Select episode | - | First/last episode | 80% page |
| Highway | - | Prev/next frame | First/last frame | 80% page |

In list panes (Treks, Parquets, Epochs, Episodes), Up/Down move the selection. In Highway, Left/Right step through frames.

Page size is 80% of the visible rows in the pane, with a minimum of 1. This means you always keep some context when paging.

## Frame navigation shortcuts

These work regardless of which pane is focused:

| Key | Action |
|---|---|
| j | 1 frame back |
| k | 1 frame forward |
| J | 5 frames back |
| K | 5 frames forward |
| h | First frame |
| l | Last frame |

Left/Right arrows also navigate frames when Highway is focused, but j/k work from any pane.

## Frame position memory

When you switch episodes or epochs, Octane saves your current frame position for that (epoch, episode) pair. When you come back, it restores the saved position (clamped to the episode length). This makes it easy to compare the same moment across different episodes or epochs.

If no position was saved for a particular episode, it defaults to frame 0.

## Trek state memory

When you switch treks in the Treks pane, the entire selection state is saved: parquet source, epoch, episode, scroll positions. When you switch back to a previously visited trek, everything is restored. First visits start at the default parquet source (`sample_es`), last epoch, first episode.

## Playback

Press `p` to toggle play/pause.

Playback uses a wall-clock anchor system. When you press play, Octane records the current wall-clock time and the current scene_time. On each frame:

```
scene_time = start_scene_time + elapsed_wall_time * playback_speed
```

This means the playback is smooth and independent of frame rate. The scene_time is a continuous float in seconds from the episode start. Frame index is derived as `floor(scene_time / seconds_per_sub_t)`.

Playback speed ranges from 0.25x to 10.0x in 0.25 steps. Adjust it in the Graphics modal (r key). The default speed is 2.0x.

Playback auto-stops when it reaches the last frame of the episode.

## Scene time

Scene time is a continuous float representing seconds from the start of the episode. It supports smooth interpolation between sub-steps. When navigating manually (j/k, arrows), scene_time snaps to the exact frame boundary. During playback, scene_time flows continuously, and the rendered frame updates whenever it crosses a boundary.

For environments with sub-stepping, the timestep display is fractional. With 5 sub-steps per policy step, frame index 17 displays as "3.40" (policy step 3, sub-step 2 out of 5, fraction = 2/5 = 0.40).

## Jump modals

Three jump modals let you go directly to a specific position:

| Key | Modal | Target |
|---|---|---|
| g | Jump Epoch | Enter epoch number |
| G | Jump Episode | Enter episode index |
| / | Jump to Timestep | Enter timestep index |

In any jump modal:
- Type digits (0-9) to enter the number
- Backspace to delete
- Enter to confirm and jump
- n to repeat the last successful jump value
- Esc to cancel

The jump target is clamped to the valid range. For epochs, Octane first tries to match the epoch_number (which may differ from the index if epochs were skipped), then falls back to using the value as an index.

## Refresh

Press R (Shift-R) to reload the current trek from disk. This re-reads all epoch/episode data and re-discovers the treks list. Useful when training is still running and new epochs have appeared. The current selection is preserved where possible.
