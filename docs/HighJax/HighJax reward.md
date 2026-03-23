# HighJax Reward

Related: [[HighJax environment]] | [[HighJax actions and dynamics]] | [[HighJax observations]]

## Reward Formula

All reward logic is in `highjax/env.py`, function `_compute_reward`. There is only one formula, matching highway-env's canonical highway-v0 reward.

### Components

**Collision component** (0 or 1):
```
collision = crashed (boolean cast to float)
```

**Speed component**, range 0 to 1:
```
forward_speed = speed * cos(heading)
nz_speed = clip((forward_speed - 20) / 10, 0, 1)
```
At 20 m/s you get 0.0, at 25 m/s you get 0.5, at 30 m/s you get 1.0.

**Right lane component**, range 0 to 1:
```
right_lane = target_lane_idx / (n_lanes - 1)
```
Lane 0 gives 0.0, lane 3 (rightmost in a 4-lane highway) gives 1.0.

### Raw Score

Weighted sum of the three components:
```
collision_reward = -1.0
high_speed_reward = 0.4
right_lane_reward = 0.1

raw_score = collision_reward * collision + high_speed_reward * nz_speed + right_lane_reward * right_lane
```

The raw score ranges from -1 (crash at low speed in lane 0) to 0.5 (no crash, max speed, rightmost lane).

### Normalization

The raw score is normalized to the 0-1 range:
```
reward = (raw_score - collision_reward) / (high_speed_reward + right_lane_reward - collision_reward)
       = (raw_score + 1) / 1.5
```

- Worst-case crash (low speed, lane 0): raw_score = -1, reward = 0
- Perfect driving (no crash, 30 m/s, rightmost lane): raw_score = 0.5, reward = 1

### Post-Crash Zeroing

On post-crash timesteps (the vehicle was already crashed before this step), the reward is set to 0:
```
was_already_crashed = crashed & ~just_crashed
reward = where(was_already_crashed, 0.0, reward)
```

On the crash step itself, `just_crashed` is true so this override does not fire. The reward comes from the formula with `collision=1`, giving `(0.4 * nz_speed + 0.1 * right_lane) / 1.5` -- a low value (0 to 0.333) but not necessarily zero, since the speed and lane terms still contribute. All subsequent post-crash timesteps get reward 0 from this override.

## Design Rationale

The formula gives crash avoidance the dominant learning signal: a crash sharply drops the reward, and the vehicle stays at 0 for the rest of the episode. The remaining 0-to-1 range provides a secondary signal for speed (weight 0.4) and lane preference (weight 0.1). Speed matters more than lane position. The optimal behavior is: go 30 m/s, stay in rightmost lanes, and above all else, don't crash.
