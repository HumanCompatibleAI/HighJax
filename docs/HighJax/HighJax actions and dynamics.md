# HighJax Actions and Dynamics

Related: [[HighJax environment]] | [[HighJax observations]] | [[HighJax reward]] | [[HighJax NPCs]]

## Five Discrete Actions

The ego agent picks one of five meta-actions each policy timestep:

| Action | Index | Effect |
|--------|-------|--------|
| LEFT   | 0     | Target lane -= 1 (clamped to 0) |
| IDLE   | 1     | Keep current targets |
| RIGHT  | 2     | Target lane += 1 (clamped to n_lanes-1) |
| FASTER | 3     | Speed index += 1 |
| SLOWER | 4     | Speed index -= 1 |

Target speeds are `linspace(20, 30, 3)`, giving 20, 25, 30 m/s. The speed index is re-derived from actual speed each step, so the agent always picks relative to where it currently is, not where it was aiming.

Source: `highjax/kinematics.py`, constants `ACTION_LANE_LEFT` through `ACTION_SLOWER`.

## Two-Level Control

Actions don't directly set steering or throttle. Instead there are two levels:

**1. Policy level** (`execute_action`): The RL agent picks a meta-action. This updates `TARGET_LANE_IDX` and `TARGET_SPEED` in the vehicle state array. That's it, no physics happens here.

**2. Servo level** (`servo_sub_step`): Proportional controllers track those targets:

- **Lateral**: A cascaded chain. Lateral position error feeds into a heading command (with look-ahead `TAU_PURSUIT=0.1s`), which feeds into a steering angle. The chain goes: lateral offset -> lateral speed command (`KP_LATERAL=1.667`) -> heading command -> heading rate command (`KP_HEADING=5.0`) -> steering angle via inverse bicycle kinematics.
- **Longitudinal**: Simple P-controller, `acceleration = KP_A * (target_speed - speed)` where `KP_A=1.667`.

NPCs don't use the ego servo. They share the lateral `steering_control` function but use [[HighJax NPCs|IDM]] for longitudinal acceleration instead.

## Bicycle Model

The core physics model (`bicycle_forward`):

```python
beta = arctan(0.5 * tan(steering))     # slip angle
vx = speed * cos(heading + beta)
vy = speed * sin(heading + beta)
new_x = x + vx * dt
new_y = y + vy * dt
new_heading = heading + speed * sin(beta) / HALF_LENGTH * dt
new_speed = clip(speed + acceleration * dt, -40, 40)
```

This is a simplified front-wheel-steering bicycle model. `HALF_LENGTH` is 2.5m (half of the 5m vehicle). Speed is hard-clamped to (-40, 40) m/s.

## Sub-Stepping

Each policy timestep (`seconds_per_t=1.0s`) is divided into sub-steps (`seconds_per_sub_t=1/15 s`), giving `n_sub_ts_per_t=15` sub-steps by default.

The flow per policy step:

1. `execute_action` runs once on the ego (sets targets)
2. `fori_loop` over 15 sub-steps, each of which:
   - Ego servo sub-step (or crash braking if already crashed)
   - NPC sub-step for all NPCs (IDM + MOBIL + bicycle model)
   - Collision detection with impulse resolution (accumulated via OR across sub-steps)
3. Finalize: update crashed/just_crashed/time, compute reward

## Crash Braking

If the ego already crashed before this step, the servo is replaced with straight-ahead braking: steering=0, acceleration = `-1.0 * current_speed`. The braking acceleration is recomputed from the current speed at each sub-step, producing exponential decay (matching highway-env). With `dt = seconds_per_sub_t`, each sub-step multiplies speed by `(1 - dt)`, so after all 15 default sub-steps the speed is roughly `0.356 * initial_speed`. Stopping takes multiple policy steps rather than exactly one.

## Collision Detection

Collision uses the Separating Axis Theorem (SAT) on rotated rectangles. Both current-frame and predictive collision are checked and OR'd together.

`check_vehicle_collisions_all` is vmapped over all NPCs, checking each ego-NPC pair. On collision, an impulse resolution pushes vehicles apart by half the SAT minimum translation vector, each direction.

Collisions are checked at every sub-step (not just at the end of the policy step). If any sub-step detects a collision, the entire step counts as a crash.

## Vehicle Constants

| Constant | Value | Unit |
|----------|-------|------|
| VEHICLE_LENGTH | 5.0 | m |
| VEHICLE_WIDTH | 2.0 | m |
| MAX_SPEED | 40.0 | m/s |
| MIN_SPEED | -40.0 | m/s |
| TAU_ACC | 0.6 | s |
| TAU_HEADING | 0.2 | s |
| TAU_LATERAL | 0.6 | s |
| TAU_PURSUIT | 0.1 | s |
| MAX_STEERING_ANGLE | pi/3 (~60 deg) | rad |

## Stepping Logic

The stepping logic lives in `highjax/stepper.py`. It separates physics from state representation:

1. `execute_action` on ego
2. `fori_loop` over `n_sub_ts_per_t` sub-steps
3. Each sub-step: ego servo -> NPC stepping -> collision check + impulse resolution
4. After the loop, `step_env` updates `crashed`/`just_crashed`/`time` and computes [[HighJax reward|reward]]

The stepper defines sub-step closures as inner functions of `step_physics`, which JAX traces once on first call.
