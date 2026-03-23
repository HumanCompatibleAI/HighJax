# HighJax Observations

Related: [[HighJax environment]] | [[HighJax actions and dynamics]]

The observation is a 2D array of shape `(n_observed_vehicles, n_features)`. By default that's `(5, 5)`. Row 0 is the ego vehicle, rows 1-4 are the closest NPCs sorted by longitudinal distance. Absent NPCs are zero-padded.

## Observation layout

### Row 0: Ego vehicle

The ego is always present (presence=1). Position and velocity features are absolute normalized values: x normalized by x_range (200), y by y_range (lane_width * n_lanes), vx/vy by v_range (80). When heading features (cos_h, sin_h) are configured, they are also absolute.

### Rows 1-4: Closest NPCs

NPCs are sorted by longitudinal distance `|npc_x - ego_x|`. Position (x, y) and velocity (vx, vy) are relative to ego and normalized. When configured, heading (cos_h, sin_h) is absolute, not relative. If fewer than 4 NPCs are visible, remaining rows are zero-padded (presence=0).

## Observation parameters

Two parameters control which NPCs appear in the observation:

**`see_behind`** (default: `False`)
- When `False`, NPCs behind the ego (where `npc_x - ego_x < -2 * VEHICLE_LENGTH`) are masked out (presence=0, all features zeroed). This matches highway-env.
- When `True`, all NPCs are visible regardless of position.

**`perception_distance`** (default: `200.0`)
- NPCs with Euclidean distance greater than this value are masked out. Matches highway-env's `PERCEPTION_DISTANCE = 5.0 * MAX_SPEED = 200.0`.
- Set to `float('inf')` to disable the distance limit.

Masked NPCs get sort distance set to infinity, so they sort to the end and appear as zero-padded rows.

## Available features

There are 14 features total. The default set uses 5 of them.

| Feature | Default? | Ego value | NPC value |
|---|---|---|---|
| presence | yes | 1.0 | 1.0 (or 0.0 if padded) |
| x | yes | Absolute x / x_range | Relative x / x_range |
| y | yes | Absolute y / y_range | Relative y / y_range |
| vx | yes | Absolute vx / v_range | Relative vx / v_range |
| vy | yes | Absolute vy / v_range | Relative vy / v_range |
| heading | no | heading / pi | Relative heading / pi |
| cos_h | no | cos(heading) | cos(npc_heading) |
| sin_h | no | sin(heading) | sin(npc_heading) |
| cos_d | no | 0.0 | 0.0 (no destination) |
| sin_d | no | 0.0 | 0.0 (no destination) |
| long_off | no | s / lane_length | s / lane_length |
| lat_off | no | r / y_range | r / y_range |
| ang_off | no | angle_offset / pi | angle_offset / pi |
| lane | no | y / lane_width - 0.5 | y / lane_width - 0.5 |

Note: cos_h and sin_h for NPCs are absolute headings, not relative to ego. The `heading` feature IS relative for NPCs. This is an intentional design choice -- the attention network can learn heading relationships from the absolute values.

## Normalization ranges

The normalized features in the table below are clipped to -1..1. Trigonometric features (cos_h, sin_h) are naturally bounded. The `lane` feature is not clipped and can exceed this range (e.g. 2.5 for lane 3 of a 4-lane highway).

| Feature | Divisor | Default value |
|---|---|---|
| x | x_range | 200 m |
| y | y_range (= lane_width * n_lanes) | 16 m |
| vx, vy | v_range | 80 m/s |
| heading | pi | - |
| long_off | lane_length | 10000 m |
| lat_off | y_range | 16 m |
| ang_off | pi | - |

## Interpreting observations

To convert normalized values back to real units:

```python
# NPC at row i with default features
real_x_distance = obs[i, 1] * 200    # meters ahead/behind
real_y_distance = obs[i, 2] * 16     # meters left/right
real_relative_vx = obs[i, 3] * 80    # m/s longitudinal
real_relative_vy = obs[i, 4] * 80    # m/s lateral
```

Some examples:

- NPC at x=0.1, y=0.0: 20m ahead, same lane
- NPC at x=0.1, y=0.25: 20m ahead, 4m to the side (one lane over)
- NPC at x=-0.05, y=0.0: 10m behind, same lane
- NPC with vx=-0.05: closing at 4 m/s relative

The ego's speed and position are directly available in row 0 as absolute normalized values.

## Observation generation

The `compute_observation()` function at `highjax/observations.py` generates observations from state:

1. Compute all 14 features for ego (absolute normalized position/velocity).
2. Compute all 14 features for each NPC (relative to ego, normalized).
3. Apply visibility mask (behind threshold, perception distance).
4. Sort visible NPCs by longitudinal distance `|npc_x - ego_x|`.
5. Take the closest `n_observed_vehicles - 1` NPCs.
6. Zero-pad if fewer NPCs exist.
7. Stack ego + NPCs into `(n_obs, 14)`.
8. Select only the configured feature columns to get `(n_obs, n_features)`.

See [[Trainer training]] for how observations feed into the ego-attention estimator.
