# HighJax Environment

Related: [[HighJax observations]] | [[HighJax actions and dynamics]] | [[HighJax reward]] | [[HighJax NPCs]]

Highway is a JAX-based highway driving environment for HighJax. One ego vehicle navigates a 4-lane highway with configurable NPC traffic. The agent picks from 5 discrete actions at each policy timestep, while a servo controller handles the low-level steering and acceleration.

The name is a portmanteau of "highway" and "JAX." The environment is inspired by highway-env but reimplemented from scratch in pure JAX for JIT compilation and vectorized rollouts.

## Configuration parameters

Parameters are split between two locations. Structural parameters live on the `HighJaxEnv` instance because they determine array shapes and cannot be JIT-traced. Tunable parameters live in `EnvParams` and can be traced through JAX.

### Structural parameters (HighJaxEnv)

| Parameter | Default | Description |
|---|---|---|
| n_lanes | 4 | Number of highway lanes |
| n_npcs | 50 | Number of NPC vehicles |
| lane_length | 10000 m | Length of each lane |
| lane_width | 4 m | Width of each lane |
| n_observed_vehicles | 5 | Ego + 4 closest NPCs |
| features | (presence, x, y, vx, vy) | Which features to include (5 of 14 available) |
| see_behind | False | Whether NPCs behind ego are visible |
| perception_distance | 200.0 m | Max Euclidean distance for NPC visibility |
| enable_npc_lane_change | True | Whether NPCs use MOBIL |
| crash_on_predicted | True | Predicted collisions also trigger crash (matching highway-env) |
| simultaneous_update | True | All vehicles decide then integrate simultaneously |
| npc_npc_collisions | True | NPC-NPC impulse separation enabled |
| npc_crash_braking | True | NPCs involved in collisions brake to a stop |

### Tunable parameters (EnvParams)

| Parameter | Default | Description |
|---|---|---|
| duration | 40 s | Episode time limit |
| seconds_per_t | 1.0 s | Policy timestep duration |
| seconds_per_sub_t | 1/15 s | Servo/physics sub-timestep |
| x_range | 200 m | X normalization range |
| v_range | 80 m/s | Velocity normalization range |
| ego_initial_speed | 25 m/s | Ego starting speed |
| ego_initial_lane | -1 | Ego starting lane index (-1 for random, matching highway-v0) |
| npc_speed_min | 21 m/s | NPC minimum spawn speed |
| npc_speed_max | 24 m/s | NPC maximum spawn speed |
| vehicles_density | 1.0 | NPC density scaling factor (sequential spawn mode) |
| ego_spacing | 2.0 | Ego spacing multiplier (sequential spawn mode) |

### Reward

The reward function is hardcoded in `_compute_reward` using the highway-v0 formula:

```
r = (-1 * collision + 0.4 * high_speed + 0.1 * right_lane + 1.0) / 1.5
```

Where `collision` is 1.0 when crashed, `high_speed` is normalized forward speed clipped to (0, 1), and `right_lane` is the target lane index divided by (n_lanes - 1). The `+ 1.0` offset and `/1.5` divisor normalize the reward into (0, 1). Post-crash timesteps (crashed but not `just_crashed`) get zero reward.

There are no tunable reward parameters. See [[HighJax reward]] for the full formula.

### Observation

Ego features use absolute normalized values (x/y/vx/vy divided by range and clipped). NPC features are relative to ego (position and velocity differences, normalized and clipped). NPCs are sorted by longitudinal distance (abs difference in x). See [[HighJax observations]] for normalization details.

### Collision pipeline

Collision detection and impulse displacement happen every sub-step inside `sub_step_with_collision` in `stepper.py`, not once at the end of the policy step. The loop runs `n_sub_ts_per_t` (default 15) iterations via `jax.lax.fori_loop`. Each iteration:

1. Physics integration runs (`sub_step`).
2. Ego-NPC collision check (`check_vehicle_collisions_all`): computes per-NPC overlap and predicted overlap. Vehicles with a positive impulse mask get pushed apart by half the separation vector each.
3. NPC-NPC collision check (`check_npc_npc_collisions_all`, if `npc_npc_collisions=True`): same impulse logic applied between NPC pairs.
4. Crash flags propagate within the sub-step loop: ego and NPC crash flags are OR-accumulated and written back to the state each iteration. This means crash braking (ego and NPC) kicks in on the very next sub-step after a collision, not deferred to the next policy step.

Because displacement is applied immediately, sub-steps following a collision see the corrected positions. A collision at sub-step 5 of 15 pushes the vehicles apart and sets crash flags before sub-steps 6-15 run.

## Vehicle state layout

Each vehicle (ego and NPCs) is stored as a flat array of size 10 (VEHICLE_STATE_SIZE):

| Index | Name | Description |
|---|---|---|
| 0 | X | Position x (m) |
| 1 | Y | Position y (m) |
| 2 | HEADING | Heading angle (rad) |
| 3 | SPEED | Forward speed (m/s) |
| 4 | TARGET_LANE_IDX | Target lane index |
| 5 | SPEED_INDEX | Index into target speeds array |
| 6 | TARGET_SPEED | Target speed (m/s) |
| 7 | LANE_CHANGE_TIMER | Lane change cooldown timer (s) |
| 8 | IDM_DELTA | IDM acceleration exponent |
| 9 | CRASHED | Crashed flag (0.0 or 1.0) |

## Environment state

`EnvState` is a flax struct dataclass (not a flat array) with these fields:

- `ego_state`: Vehicle state array `(VEHICLE_STATE_SIZE,)` = `(10,)`
- `npc_states`: NPC states array `(n_npcs, VEHICLE_STATE_SIZE)` = `(50, 10)` by default
- `time`: Time counter (scalar)
- `crashed`: Crashed flag (scalar bool)
- `just_crashed`: First crash this frame (scalar bool)
- `previous_ego_lane`: Lane before action that produced this state (scalar)

`EnvState.from_scenario_dict(env, state_dict)` can reconstruct a state from a dict of named fields (as saved by Octane), useful for building test scenarios and behaviors.

## Derived properties

These are derived from the config, not set directly:

- `n_sub_ts_per_t = 15` (hardcoded in constructor; matches `round(seconds_per_t / seconds_per_sub_t)` for the defaults)
- `observation_shape = (n_observed_vehicles, len(features)) = (5, 5)`
- `num_actions = 5` - five discrete actions
- `y_range = lane_width * n_lanes = 16 m`

## CLI usage

```bash
# Basic training run
highjax-trainer train --n-epochs 300 --n-es 128 --n-ts 400 --n-npcs 50 --actor-lr 1e-3
```

The `--n-es` and `--n-ts` flags control episode and timestep counts per epoch. The `--n-npcs` flag sets the NPC count on the environment.

### Matching highway-v0 timing

Highway defaults now match highway-v0 timing: 1 Hz policy, 15 Hz physics, 15 sub-steps per policy step. To use the old Highway timing instead:

```python
params = EnvParams(seconds_per_t=0.5, seconds_per_sub_t=0.1)
```

Note that `n_sub_ts_per_t` is currently hardcoded to 15 on the `HighJaxEnv` instance, so changing the timing params alone does not change the sub-step count. You would also need to adjust `n_sub_ts_per_t` on the env to get the expected 5 sub-steps per policy step at the old timing.

## Estimator

The trainer uses the `ego_attention` estimator pair by default. This is a multi-head attention architecture where the ego embedding queries NPC embeddings.

## Behaviors

The environment ships with a `collision` behavior (in `highjax/behaviors/collision.json`). Behaviors are named measures that score a policy on hand-crafted scenarios: each scenario has a traffic state and action weights, and the score is the weighted average of the policy's action probabilities across all scenarios. Users can add custom behaviors by placing JSON files in `~/.highjax/behaviors/`.

## Source files

The main environment code lives in `highjax/`:

- `env.py` - Environment class, EnvState, EnvParams, reward, done logic
- `stepper.py` - Physics sub-stepping with collision detection
- `observations.py` - Observation generation
- `kinematics.py` - Bicycle model, servo, collision detection (SAT algorithm)
- `idm.py` - IDM/MOBIL for NPCs
- `lanes.py` - Lane geometry
- `behaviors/` - Behavior scenarios (collision.json and user-defined)

## Known limitations

1. **Highway only** — no merge, intersection, or roundabout scenarios.
2. **Discrete actions** — no continuous control.
3. **Fixed NPC count** — set at initialization, cannot change during episode.
4. **Single agent** — one ego vehicle only.

## Divergences from HighwayEnv

Highway is inspired by HighwayEnv but has intentional simplifications for the JAX-vectorized setting:

- **Fixed NPC count** — HighwayEnv dynamically spawns/removes vehicles; Highway keeps a fixed count for JIT compatibility.
- **Straight lanes only** — no curved road geometry.
- **Politeness = 0** — MOBIL's politeness parameter is always zero, simplifying the gain calculation (follower terms cancel out).
- **No `follow_road` behavior** — NPCs don't automatically follow road curvature (irrelevant with straight lanes).
- **No reachability check** — HighwayEnv checks whether a lane change is physically reachable; Highway skips this.
- **Post-crash episode continuation** — HighwayEnv terminates on crash. Highway keeps episodes running (with crash braking and zero reward) because variable-length episodes are impractical in the vectorized setting. `is_terminated` is set on crash (the episode is "over" semantically), `is_truncated` on timeout only, and `is_done` on either. The distinction matters for GAE bootstrapping: truncated episodes bootstrap from the value estimate, terminated episodes don't.
- **Simultaneous NPC updates** — HighwayEnv iterates vehicles sequentially in `road.act()` and `road.step()`, so each vehicle sees already-updated states of earlier vehicles. Highway uses `jax.vmap` for truly simultaneous updates — all vehicles decide and integrate from the same pre-update states. This causes NPC trajectories to diverge slightly (~6m over 10 policy steps) despite identical per-vehicle formulas. Ego physics remain sub-millimeter accurate.
- **NPC sorting by longitudinal distance** — HighwayEnv sorts observed NPCs by Euclidean distance. Highway sorts by `|npc_x - ego_x|` (longitudinal only). On a straight highway these almost always produce the same ordering but can occasionally swap two NPCs at similar x but different y.
- **Reward post-crash zeroing** — Highway explicitly zeros the reward on post-crash timesteps. HighwayEnv continues computing the formula with `collision=1`, yielding a small positive value instead of exactly zero.
- **No on-road reward multiplier** — HighwayEnv multiplies the reward by an `on_road` flag (0 if vehicle is off-road). Highway omits this since vehicles rarely go off-road on a straight highway.
- **MOBIL distance approximation** — HighwayEnv uses exact pairwise lane distances for the MOBIL incentive calculation. Highway approximates follower-to-leader distance as the sum of ego-to-front and ego-to-rear distances, which is exact when vehicles are collinear in x.
# CI test
