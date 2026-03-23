# HighJax NPCs

Related: [[HighJax environment]] | [[HighJax actions and dynamics]] | [[HighJax observations]]

## Overview

NPC vehicles use two classic traffic models: IDM for longitudinal control (acceleration/braking) and MOBIL for lane change decisions. All logic is in `highjax/idm.py`, with some constants (like `LANE_CHANGE_DELAY`) in `highjax/kinematics.py`.

## IDM (Intelligent Driver Model)

IDM computes acceleration based on desired speed and the gap to the vehicle ahead:

```
a = COMFORT_ACC_MAX * (1 - (v/v0)^delta - (s*/s)^2)
```

Where `s*` is the desired gap:

```
s* = d0 + max(v, 0)*tau + v*dv / (2*sqrt(a_max * |b|))
```

Note: the headway term uses `max(v, 0)` -- negative speeds are clipped to 0 so that a reversing vehicle doesn't produce a negative time-headway contribution.

**Constants**:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| COMFORT_ACC_MAX | 3.0 m/s^2 | Desired max acceleration |
| COMFORT_ACC_MIN | -5.0 m/s^2 | Desired max deceleration |
| ACC_MAX | 6.0 m/s^2 | Physical hard limit |
| DISTANCE_WANTED | 10.0 m | Minimum jam distance |
| TIME_WANTED | 1.5 s | Desired time headway |
| DELTA | 4.0 | Acceleration exponent |

Each NPC has a randomized `IDM_DELTA` drawn from 3.5 to 4.5 at spawn, making behavior slightly different across vehicles.

**Projected velocity difference**: When heading angles are available, `dv` is projected onto the ego's forward direction rather than computed as a simple scalar subtraction. This matters when vehicles have non-zero heading (e.g. mid-lane-change).

## MOBIL (Lane Change Model)

MOBIL decides whether a lane change is worth it based on acceleration gain, including the impact on followers in both the current and target lanes. NPC lane changing can be disabled entirely by setting `enable_npc_lane_change = False` on HighJaxEnv (default True) -- when disabled, NPCs never change lanes and only use IDM longitudinal control.

**Gain formula**:

```
gain = (a'_self - a_self) + POLITENESS * ((a'_new_follower - a_new_follower)
                                         + (a'_old_follower - a_old_follower))
```

Where primed values are the predicted accelerations after the lane change, and "self" refers to the NPC considering the lane change:

- `a_self`: self's acceleration in current lane (following current lane's front vehicle)
- `a'_self`: self's predicted acceleration in target lane (following target lane's front vehicle)
- `a_new_follower`: current acceleration of the rear vehicle in the target lane (following target lane's preceding vehicle, not yet self)
- `a'_new_follower`: predicted acceleration of that follower after self inserts in front of it
- `a_old_follower`: current acceleration of the rear vehicle in the current lane (following self, before self leaves)
- `a'_old_follower`: predicted acceleration of that follower after self vacates (now following self's old front vehicle)

Follower terms are zeroed out when the corresponding follower doesn't exist.

**Constants**:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| POLITENESS | 0.0 | How much to consider others (0 = selfish) |
| LANE_CHANGE_MIN_ACC_GAIN | 0.2 m/s^2 | Minimum gain to bother changing |
| LANE_CHANGE_MAX_BRAKING_IMPOSED | 2.0 m/s^2 | Max braking imposed on new follower |

**Decision process** (`mobil_lane_change`):

1. Don't change lanes if speed < 1.0 m/s
2. Compute full MOBIL gain for left and right lane options
3. Safety check: reject if the new follower in the target lane would need to brake harder than `LANE_CHANGE_MAX_BRAKING_IMPOSED` (2.0 m/s^2) -- gain is set to -inf in this case
4. Each direction passes if its gain exceeds `LANE_CHANGE_MIN_ACC_GAIN` (0.2 m/s^2)
5. Right wins if right passes; else left wins if left passes; else stay (right-priority, not best-gain):
```python
new_lane = jnp.where(right_ok, right_lane,
                      jnp.where(left_ok, left_lane, current_lane))
```

With `POLITENESS=0` (default), the follower terms are multiplied out and NPCs are purely selfish -- they only care about their own acceleration gain. Setting `POLITENESS > 0` makes NPCs also weigh the benefit to the old follower (who gains a gap) and the cost to the new follower (who must now follow a slower vehicle).

## Lane Change Cooldown

`LANE_CHANGE_DELAY = 1.0s`. The timer resets to 0 whenever the cooldown fires (i.e. when `timer >= LANE_CHANGE_DELAY`), regardless of whether MOBIL actually triggers a lane change. This gives a steady ~1 Hz evaluation cadence, matching highway-env. Previously the timer only reset on actual lane change, which meant NPCs would evaluate MOBIL every sub-step once the cooldown expired without changing lanes. The timer is initialized pseudo-randomly at spawn based on position: `((x + y) * pi) % LANE_CHANGE_DELAY`.

## Abort-If-Conflict

During an ongoing lane change (`physical_lane != target_lane`), the change is aborted if a vehicle in the target lane is ahead of the ego (`other_x - ego_x > 0`) and within the desired gap. Only vehicles not already physically in the target lane are considered (`other_physical_lane != target_lane`). The gap check uses the other vehicle's speed in `desired_gap()` rather than self speed. This prevents two NPCs from merging into the same spot simultaneously.

The check runs before MOBIL in `npc_decide` (matching highway-env's order), and only applies to ongoing lane changes -- it does not gate new lane change decisions.

## NPC Sub-Step

The sub-step logic is split into `npc_decide` (computes MOBIL+IDM decisions) and `npc_integrate` (applies the bicycle model); `npc_sub_step` calls both in sequence. When `simultaneous_update=True` (default, matching highway-env), all vehicles decide based on old states before any integration happens. When `False`, the ego integrates first and NPCs see the updated ego position before deciding.

`npc_decide` steps:

1. Check lane change cooldown
2. Apply abort-if-conflict check (ongoing lane changes only)
3. Run MOBIL decision (if cooldown is ready and no ongoing lane change -- MOBIL is masked with `& ~ongoing_change`)
4. Compute IDM acceleration against front vehicle in current lane. `idm_acceleration()` returns unclipped values; clipping to `[-ACC_MAX, ACC_MAX]` happens here in `npc_decide` after step 5.
5. If changing lanes: also check front vehicle in target lane, take the minimum (more conservative) of the two unclipped accelerations, then clip. MOBIL sees the raw unclipped accelerations from `idm_acceleration()`.
6. Compute steering via `steering_control` (same function the ego servo uses)
7. **Crash override**: if the NPC's `CRASHED` flag is set, override acceleration to `-1.0 * speed` (braking to stop), steering to `0.0`, and lock the target lane to the current lane. This bypasses all IDM/MOBIL output.

`npc_integrate` then advances the bicycle model.

## NPC-NPC Collisions

When `npc_npc_collisions=True` (default), `check_npc_npc_collisions_all` in `kinematics.py` runs all-pairs SAT collision detection between NPCs every sub-step inside `sub_step_with_collision`, applying impulse separation to push any overlapping pair apart. The function returns `(impulse_by_npc, colliding_by_npc)` -- the collision mask is also used to set NPC crash flags when `npc_crash_braking` is enabled. NPC-NPC collisions don't affect the ego crash flag -- only NPC-ego contact does.

## NPC Crash Behavior

When `npc_crash_braking=True` (default), NPCs that collide with the ego or with other NPCs get a `CRASHED` flag in their vehicle state (index 9, added in `VEHICLE_STATE_SIZE=10`). The flag is sticky -- once set, it persists for the rest of the episode.

### How crash flags are set

Crash flags are computed per sub-step inside `sub_step_with_collision`, not deferred to `_step_finalize`. Each sub-step iteration:

1. Start from existing crash flags (`npc_already_crashed`)
2. OR in ego-NPC collision mask -- uses the predicted-collision mask when `crash_on_predicted=True`, or current-overlap mask when `False`, following the same `crash_on_predicted` logic as the ego's own crash flag
3. If `npc_npc_collisions=True`, OR in the NPC-NPC collision mask
4. Write the combined boolean mask back to `CRASHED` in each NPC's state

Because flags propagate within the sub-step loop, crash braking kicks in on the very next sub-step after a collision rather than waiting for the next policy step.

The entire block is gated on `npc_crash_braking`. When `False`, NPC states never get crash flags and NPCs drive normally through collisions.

### Braking behavior

The crash override is applied at the end of `npc_decide` in `highjax/idm.py`. When `CRASHED > 0.5`:

- **Acceleration** is set to `-1.0 * speed`, causing the NPC to brake proportionally to its current speed (rapid deceleration that tapers to zero as the vehicle stops)
- **Steering** is set to `0.0` (straight ahead)
- **Target lane** is locked to the current lane (no lane changes)

This matches highway-env's post-crash behavior where crashed vehicles brake to a stop and become static obstacles. The override happens after all IDM/MOBIL computation, so the normal decision pipeline still runs but its output is discarded for crashed NPCs.

### Interaction with other systems

- The crash flag is initialized to `0.0` at spawn in both `make_vehicle_state` (`kinematics.py`) and `make_npc` (`env.py`)
- Crashed NPCs still participate in front/rear vehicle detection and collision checks -- they just stop accelerating and steering
- The `npc_crash_braking` parameter is independent of `npc_npc_collisions`: you can have NPC-NPC impulse separation without crash flags, or crash flags from ego-NPC collisions only (when `npc_npc_collisions=False`)

## Front/Rear Vehicle Detection

Detection uses lateral position, not target lane index. A vehicle is considered "in lane" if `|y - lane_center| <= lane_width/2 + 1.0m` (the 1.0m margin matches highway-env's `on_lane(margin=1)`). This means mid-lane-change vehicles can be detected in both their origin and destination lanes.

A vehicle is never detected as its own front/rear because it's at the same x position, not strictly ahead or behind itself.

## NPC Vision

Each NPC sees the ego plus all other NPCs. The `npc_sub_step_all` function concatenates ego_state and all NPC states, then vmaps over individual NPCs. Self-detection is naturally avoided by the "strictly ahead/behind" logic in `find_front_vehicle` and `find_rear_vehicle`.

## NPC Speeds

- Target speed: each NPC targets its own spawn speed (matching highway-env). The target speed is set directly in the vehicle state at spawn, not through the discrete speed index system that the ego uses.
- Spawn speeds: random uniform in 21-24 m/s (controlled by `npc_speed_min` and `npc_speed_max` in `EnvParams`). NPCs stay near their spawn speed rather than accelerating toward a common target.

## Spawning

Spawning uses sequential placement matching highway-env's `create_random`. Vehicles are placed one by one at increasing x positions. Each vehicle's offset from the previous one is:

```
offset = (1 / vehicles_density) * (12 + speed) * exp(-5/40 * n_lanes)
```

with +/-10% jitter applied. The ego is placed at `3 * ego_offset` where:

```
ego_offset = ego_spacing * (12 + ego_speed) * exp(-5/40 * n_lanes)
```

**Parameters** (in `EnvParams`):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `vehicles_density` | 1.0 | Scales NPC spacing (higher = denser traffic) |
| `ego_spacing` | 2.0 | Multiplier for ego's initial offset |
