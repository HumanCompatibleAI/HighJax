'''
Intelligent Driver Model (IDM) for NPC vehicles in JAX.

IDM Model
---------
The IDM computes acceleration based on:
- Desired speed tracking: accelerate to reach target speed
- Front vehicle following: decelerate to maintain safe gap

    a = a_max * (1 - (v/v0)^delta - (s*/s)^2)

Where:
- a_max = comfort acceleration (COMFORT_ACC_MAX)
- v = current speed
- v0 = target/desired speed
- delta = acceleration exponent (typically 4)
- s* = desired gap (function of speed and relative velocity)
- s = actual gap to front vehicle

Desired Gap Formula
-------------------
    s* = d0 + v*tau + v*dv / (2*sqrt(a_max * |b|))

Where:
- d0 = minimum jam distance (DISTANCE_WANTED)
- tau = desired time headway (TIME_WANTED)
- dv = velocity difference (ego - front, positive when approaching)
- b = comfortable deceleration (COMFORT_ACC_MIN)

MOBIL Lane Change
-----------------
Full MOBIL model: change lane if it improves acceleration, considering
the impact on followers in both lanes.

    gain = (a'_ego - a_ego) + p * ((a'_new_follower - a_new_follower)
                                  + (a'_old_follower - a_old_follower))

Where p is politeness factor and primed values are predicted after
lane change. Safety constraint: reject if new follower must brake
harder than LANE_CHANGE_MAX_BRAKING_IMPOSED.
'''
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import kinematics, lanes
from .kinematics import not_zero

# IDM longitudinal parameters
COMFORT_ACC_MAX = 3.0      # [m/s²] Desired maximum acceleration
COMFORT_ACC_MIN = -5.0     # [m/s²] Desired maximum deceleration
ACC_MAX = 6.0              # [m/s²] Physical maximum acceleration
DISTANCE_WANTED = 10.0     # [m] Minimum jam distance (5 + vehicle length)
TIME_WANTED = 1.5          # [s] Desired time headway
DELTA = 4.0                # [] Acceleration exponent

# MOBIL lateral parameters
POLITENESS = 0.0                    # [] How much to consider others (0 = selfish)
LANE_CHANGE_MIN_ACC_GAIN = 0.2      # [m/s²] Minimum acceleration gain to change lane
LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s²] Max braking imposed on new follower

# NPC target speed fallback (used when rear vehicle doesn't exist)
NPC_TARGET_SPEED = 30.0    # [m/s] matches highway-env lane speed_limit

# Lateral margin for lane membership (matching HE's on_lane margin=1)
ON_LANE_MARGIN = 1.0       # [m]


def desired_gap(ego_speed: jax.Array, front_speed: jax.Array,
                ego_heading: jax.Array = None,
                front_heading: jax.Array = None) -> jax.Array:
    '''
    Compute desired gap to front vehicle.

    Args:
        ego_speed: Ego vehicle speed [m/s]
        front_speed: Front vehicle speed [m/s]
        ego_heading: Ego heading angle [rad] (optional, for projected dv)
        front_heading: Front heading angle [rad] (optional, for projected dv)

    Returns:
        Desired gap [m]
    '''
    # Projected velocity difference (#9): project onto ego direction
    if ego_heading is not None and front_heading is not None:
        ego_vx = ego_speed * jnp.cos(ego_heading)
        ego_vy = ego_speed * jnp.sin(ego_heading)
        front_vx = front_speed * jnp.cos(front_heading)
        front_vy = front_speed * jnp.sin(front_heading)
        dv = ((ego_vx - front_vx) * jnp.cos(ego_heading) +
              (ego_vy - front_vy) * jnp.sin(ego_heading))
    else:
        dv = ego_speed - front_speed  # Positive when approaching

    ab = -COMFORT_ACC_MAX * COMFORT_ACC_MIN  # Product of acc and dec magnitudes

    d_star = (DISTANCE_WANTED +
              jnp.maximum(ego_speed, 0) * TIME_WANTED +
              ego_speed * dv / (2 * jnp.sqrt(ab)))

    return d_star


def idm_acceleration(ego_speed: jax.Array, target_speed: jax.Array,
                     front_distance: jax.Array, front_speed: jax.Array,
                     has_front: jax.Array,
                     ego_heading: jax.Array = None,
                     front_heading: jax.Array = None,
                     delta: jax.Array = None) -> jax.Array:
    '''
    Compute IDM acceleration.

    Args:
        ego_speed: Ego vehicle speed [m/s]
        target_speed: Desired speed [m/s]
        front_distance: Distance to front vehicle [m]
        front_speed: Front vehicle speed [m/s]
        has_front: Boolean, whether there is a front vehicle
        ego_heading: Ego heading angle [rad] (optional, for projected dv)
        front_heading: Front heading angle [rad] (optional, for projected dv)
        delta: IDM acceleration exponent (per-vehicle, defaults to DELTA)

    Returns:
        Acceleration command [m/s²]
    '''
    if delta is None:
        delta = DELTA
    # Free road acceleration: reach target speed
    speed_term = jnp.power(
        jnp.maximum(ego_speed, 0) / not_zero(jnp.abs(target_speed)),
        delta
    )
    free_acc = COMFORT_ACC_MAX * (1 - speed_term)

    # Interaction term: maintain gap to front vehicle
    d_star = desired_gap(ego_speed, front_speed, ego_heading, front_heading)
    gap_term = jnp.power(d_star / not_zero(front_distance), 2)
    interaction_acc = -COMFORT_ACC_MAX * gap_term

    # Total acceleration
    acc = jnp.where(has_front, free_acc + interaction_acc, free_acc)

    return acc


def find_front_vehicle(
    ego_state: jax.Array, all_states: jax.Array, lane_idx: int, lane_width: float = 4.0
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    '''
    Find the front vehicle in a given lane using lateral position check.

    Uses |y - lane_center_y| <= lane_width/2 + margin instead of
    TARGET_LANE_IDX matching, matching HE's on_lane(margin=1).

    Args:
        ego_state: Ego vehicle state
        all_states: All NPC vehicle states, shape (n_vehicles, VEHICLE_STATE_SIZE)
        lane_idx: Lane index to search
        lane_width: Width of each lane [m]

    Returns:
        (front_distance, front_speed, has_front, front_heading)
    '''
    ego_x = ego_state[kinematics.X]

    npc_xs = all_states[:, kinematics.X]
    npc_ys = all_states[:, kinematics.Y]
    npc_speeds = all_states[:, kinematics.SPEED]
    npc_headings = all_states[:, kinematics.HEADING]

    # Lane center y = lane_idx * lane_width
    lane_center_y = lane_idx * lane_width
    # Vehicles in lane: lateral position within lane_width/2 + margin
    in_lane = jnp.abs(npc_ys - lane_center_y) <= (lane_width / 2 + ON_LANE_MARGIN)
    ahead = npc_xs > ego_x
    valid = in_lane & ahead

    # Find closest front vehicle
    distances = jnp.where(valid, npc_xs - ego_x, jnp.inf)
    min_idx = jnp.argmin(distances)
    min_distance = jnp.take(distances, min_idx)

    has_front = jnp.any(valid)
    front_speed = jnp.where(has_front, jnp.take(npc_speeds, min_idx), 0.0)
    front_distance = jnp.where(has_front, min_distance, jnp.inf)
    front_heading = jnp.where(has_front, jnp.take(npc_headings, min_idx), 0.0)

    return front_distance, front_speed, has_front, front_heading


def find_rear_vehicle(ego_state: jax.Array, all_states: jax.Array,
                      lane_idx: int,
                      lane_width: float = 4.0):
    '''
    Find the rear vehicle in a given lane using lateral position check.

    Uses |y - lane_center_y| <= lane_width/2 + margin instead of
    TARGET_LANE_IDX matching, matching HE's on_lane(margin=1).

    Args:
        ego_state: Ego vehicle state
        all_states: All NPC vehicle states, shape (n_vehicles, VEHICLE_STATE_SIZE)
        lane_idx: Lane index to search
        lane_width: Width of each lane [m]

    Returns:
        (rear_distance, rear_speed, has_rear, rear_heading, rear_target_speed, rear_delta)
    '''
    ego_x = ego_state[kinematics.X]

    npc_xs = all_states[:, kinematics.X]
    npc_ys = all_states[:, kinematics.Y]
    npc_speeds = all_states[:, kinematics.SPEED]
    npc_headings = all_states[:, kinematics.HEADING]
    npc_target_speeds = all_states[:, kinematics.TARGET_SPEED]
    npc_deltas = all_states[:, kinematics.IDM_DELTA]

    # Lane center y = lane_idx * lane_width
    lane_center_y = lane_idx * lane_width
    # Vehicles in lane: lateral position within lane_width/2 + margin
    in_lane = jnp.abs(npc_ys - lane_center_y) <= (lane_width / 2 + ON_LANE_MARGIN)
    behind = npc_xs < ego_x
    valid = in_lane & behind

    # Find closest rear vehicle
    distances = jnp.where(valid, ego_x - npc_xs, jnp.inf)
    min_idx = jnp.argmin(distances)
    min_distance = jnp.take(distances, min_idx)

    has_rear = jnp.any(valid)
    rear_speed = jnp.where(has_rear, jnp.take(npc_speeds, min_idx), 0.0)
    rear_distance = jnp.where(has_rear, min_distance, jnp.inf)
    rear_heading = jnp.where(has_rear, jnp.take(npc_headings, min_idx), 0.0)
    rear_target_speed = jnp.where(has_rear, jnp.take(npc_target_speeds, min_idx),
                                   NPC_TARGET_SPEED)
    rear_delta = jnp.where(has_rear, jnp.take(npc_deltas, min_idx), DELTA)

    return rear_distance, rear_speed, has_rear, rear_heading, rear_target_speed, rear_delta


def mobil_gain(ego_state: jax.Array, all_states: jax.Array,
               current_lane: int, target_lane: int,
               lane_width: float = 4.0) -> jax.Array:
    '''
    Compute MOBIL acceleration gain for a potential lane change.

    Args:
        ego_state: Ego vehicle state
        all_states: All vehicle states (excluding ego)
        current_lane: Current lane index
        target_lane: Target lane index
        lane_width: Width of each lane [m]

    Returns:
        Acceleration gain (positive = beneficial lane change)
    '''
    ego_speed = ego_state[kinematics.SPEED]
    ego_heading = ego_state[kinematics.HEADING]
    target_speed = ego_state[kinematics.TARGET_SPEED]
    ego_delta = ego_state[kinematics.IDM_DELTA]

    # Current lane acceleration
    front_dist_cur, front_speed_cur, has_front_cur, front_heading_cur = find_front_vehicle(
        ego_state, all_states, current_lane, lane_width)
    acc_current = idm_acceleration(ego_speed, target_speed,
                                   front_dist_cur, front_speed_cur, has_front_cur,
                                   ego_heading, front_heading_cur, ego_delta)

    # Target lane acceleration
    front_dist_new, front_speed_new, has_front_new, front_heading_new = find_front_vehicle(
        ego_state, all_states, target_lane, lane_width)
    acc_new = idm_acceleration(ego_speed, target_speed,
                               front_dist_new, front_speed_new, has_front_new,
                               ego_heading, front_heading_new, ego_delta)

    # -- New follower (rear vehicle in target lane) --
    rear_dist_new, rear_speed_new, has_rear_new, rear_heading_new, rear_target_speed_new, \
        rear_delta_new = find_rear_vehicle(ego_state, all_states, target_lane, lane_width)

    # New follower's current acc (following new_preceding, before ego arrives)
    new_follower_a = idm_acceleration(
        rear_speed_new, rear_target_speed_new,
        rear_dist_new + front_dist_new, front_speed_new, has_front_new,
        rear_heading_new, front_heading_new, rear_delta_new)

    # New follower's predicted acc (following ego, after ego inserts)
    new_follower_pred_a = idm_acceleration(
        rear_speed_new, rear_target_speed_new,
        rear_dist_new, ego_speed, jnp.array(True),
        rear_heading_new, ego_heading, rear_delta_new)

    # Is the braking too severe?
    unsafe = new_follower_pred_a < -LANE_CHANGE_MAX_BRAKING_IMPOSED

    # -- Old follower (rear vehicle in current lane) --
    rear_dist_old, rear_speed_old, has_rear_old, rear_heading_old, rear_target_speed_old, \
        rear_delta_old = find_rear_vehicle(ego_state, all_states, current_lane, lane_width)

    # Old follower's current acc (following ego, before ego leaves)
    old_follower_a = idm_acceleration(
        rear_speed_old, rear_target_speed_old,
        rear_dist_old, ego_speed, jnp.array(True),
        rear_heading_old, ego_heading, rear_delta_old)

    # Old follower's predicted acc (following old_preceding, after ego leaves)
    old_follower_pred_a = idm_acceleration(
        rear_speed_old, rear_target_speed_old,
        rear_dist_old + front_dist_cur, front_speed_cur, has_front_cur,
        rear_heading_old, front_heading_cur, rear_delta_old)

    # Full MOBIL gain: self improvement + politeness * (follower improvements)
    # Zero out follower terms when the follower doesn't exist
    new_follower_term = jnp.where(has_rear_new,
                                   new_follower_pred_a - new_follower_a, 0.0)
    old_follower_term = jnp.where(has_rear_old,
                                   old_follower_pred_a - old_follower_a, 0.0)
    gain = (acc_new - acc_current
            + POLITENESS * (new_follower_term + old_follower_term))

    return jnp.where(unsafe, -jnp.inf, gain)


def mobil_lane_change(ego_state: jax.Array, all_states: jax.Array,
                      n_lanes: int, lane_width: float = 4.0) -> jax.Array:
    '''
    Decide lane change using simplified MOBIL model.

    Args:
        ego_state: Ego vehicle state
        all_states: All other vehicle states
        n_lanes: Number of lanes
        lane_width: Width of each lane [m]

    Returns:
        New target lane index
    '''
    current_lane = ego_state[kinematics.TARGET_LANE_IDX].astype(jnp.int32)
    ego_speed = ego_state[kinematics.SPEED]

    # Don't change lanes at low speed
    too_slow = jnp.abs(ego_speed) < 1.0

    # Compute gain for left lane change
    left_lane = jnp.clip(current_lane - 1, 0, n_lanes - 1)
    left_valid = (current_lane > 0) & ~too_slow
    left_gain = jnp.where(
        left_valid,
        mobil_gain(ego_state, all_states, current_lane, left_lane, lane_width),
        -jnp.inf
    )

    # Compute gain for right lane change
    right_lane = jnp.clip(current_lane + 1, 0, n_lanes - 1)
    right_valid = (current_lane < n_lanes - 1) & ~too_slow
    right_gain = jnp.where(
        right_valid,
        mobil_gain(ego_state, all_states, current_lane, right_lane, lane_width),
        -jnp.inf
    )

    # Right wins when both pass (matching highway-env's last-overwrite order)
    right_ok = right_gain > LANE_CHANGE_MIN_ACC_GAIN
    left_ok = left_gain > LANE_CHANGE_MIN_ACC_GAIN
    new_lane = jnp.where(right_ok, right_lane,
                          jnp.where(left_ok, left_lane, current_lane))

    return new_lane


def npc_decide(npc_state: jax.Array, all_states: jax.Array,
               highway_lanes: jax.Array,
               enable_lane_change: bool = True
               ) -> tuple[jax.Array, jax.Array, jax.Array]:
    '''Compute NPC decisions (MOBIL + IDM) without integrating physics.

    Returns (acceleration, steering, new_target_lane).
    '''
    n_lanes = highway_lanes.shape[0]
    lane_width = highway_lanes[0, lanes.WIDTH]
    current_lane = npc_state[kinematics.TARGET_LANE_IDX].astype(jnp.int32)

    # Lane change cooldown (#4): only evaluate MOBIL when timer >= LANE_CHANGE_DELAY
    timer = npc_state[kinematics.LANE_CHANGE_TIMER]
    cooldown_ready = timer >= kinematics.LANE_CHANGE_DELAY

    # Abort-if-conflict: only during ongoing lane change (from previous step).
    # Runs BEFORE MOBIL (matching highway-env). Uses signed distance (ahead
    # only), excludes vehicles already in target lane, uses other's speed.
    physical_lane_idx = jnp.round(
        npc_state[kinematics.Y] / lane_width).astype(jnp.int32)
    ongoing_change = physical_lane_idx != current_lane

    other_target_lanes = all_states[:, kinematics.TARGET_LANE_IDX].astype(jnp.int32)
    other_physical_lanes = jnp.round(
        all_states[:, kinematics.Y] / lane_width).astype(jnp.int32)
    other_xs = all_states[:, kinematics.X]
    other_speeds = all_states[:, kinematics.SPEED]
    ego_x = npc_state[kinematics.X]
    ego_speed = npc_state[kinematics.SPEED]

    same_target = other_target_lanes == current_lane
    not_in_target = other_physical_lanes != current_lane
    signed_dist = other_xs - ego_x
    ahead = signed_dist > 0
    d_gap_by_other = jax.vmap(
        lambda other_speed: desired_gap(ego_speed, other_speed))(other_speeds)
    within_gap = signed_dist < d_gap_by_other

    conflict = jnp.any(
        same_target & not_in_target & ahead & within_gap & ongoing_change)
    effective_lane = jnp.where(conflict, physical_lane_idx, current_lane)

    # MOBIL: decide if we should change lanes (only if cooldown ready)
    mobil_target = mobil_lane_change(npc_state, all_states, n_lanes, lane_width)
    new_target_lane = jnp.where(
        enable_lane_change & cooldown_ready & ~ongoing_change,
        mobil_target,
        effective_lane
    )

    # Temporarily set target lane for IDM/steering computation
    state = npc_state.at[kinematics.TARGET_LANE_IDX].set(
        new_target_lane.astype(jnp.float32))

    # IDM: compute acceleration based on front vehicle
    # Check both current and target lanes, use minimum acceleration
    npc_heading = state[kinematics.HEADING]
    npc_delta = state[kinematics.IDM_DELTA]
    front_dist_cur, front_speed_cur, has_front_cur, front_heading_cur = find_front_vehicle(
        state, all_states, current_lane, lane_width)
    acc_current = idm_acceleration(
        state[kinematics.SPEED], state[kinematics.TARGET_SPEED],
        front_dist_cur, front_speed_cur, has_front_cur,
        npc_heading, front_heading_cur, npc_delta)

    # Also check target lane if changing
    front_dist_tgt, front_speed_tgt, has_front_tgt, front_heading_tgt = find_front_vehicle(
        state, all_states, new_target_lane, lane_width)
    acc_target = idm_acceleration(
        state[kinematics.SPEED], state[kinematics.TARGET_SPEED],
        front_dist_tgt, front_speed_tgt, has_front_tgt,
        npc_heading, front_heading_tgt, npc_delta)

    acceleration = jnp.where(
        new_target_lane != current_lane,
        jnp.minimum(acc_current, acc_target),
        acc_current
    )
    acceleration = jnp.clip(acceleration, -ACC_MAX, ACC_MAX)

    # Steering control to follow lane
    target_lane_params = highway_lanes[new_target_lane]
    steering = kinematics.steering_control(state, target_lane_params)

    # Crashed NPCs: override with braking, no steering, no lane change
    is_crashed = npc_state[kinematics.CRASHED] > 0.5
    acceleration = jnp.where(
        is_crashed, -1.0 * npc_state[kinematics.SPEED], acceleration)
    steering = jnp.where(is_crashed, 0.0, steering)
    new_target_lane = jnp.where(is_crashed, current_lane, new_target_lane)

    return acceleration, steering, new_target_lane


def npc_integrate(npc_state: jax.Array, acceleration: jax.Array,
                  steering: jax.Array, new_target_lane: jax.Array,
                  seconds_per_t: float) -> jax.Array:
    '''Integrate NPC physics given pre-computed decisions.'''
    # Timer update (self-contained)
    timer = npc_state[kinematics.LANE_CHANGE_TIMER]
    cooldown_ready = timer >= kinematics.LANE_CHANGE_DELAY
    new_timer = jnp.where(cooldown_ready, 0.0, timer + seconds_per_t)

    # Update target lane and timer
    state = (npc_state
             .at[kinematics.TARGET_LANE_IDX].set(new_target_lane.astype(jnp.float32))
             .at[kinematics.LANE_CHANGE_TIMER].set(new_timer))

    # Apply bicycle model
    new_x, new_y, new_heading, new_speed = kinematics.bicycle_forward(
        state[kinematics.X], state[kinematics.Y],
        state[kinematics.HEADING], state[kinematics.SPEED],
        steering, acceleration, seconds_per_t
    )

    return (state
            .at[kinematics.X].set(new_x)
            .at[kinematics.Y].set(new_y)
            .at[kinematics.HEADING].set(new_heading)
            .at[kinematics.SPEED].set(new_speed))


def npc_sub_step(npc_state: jax.Array, all_states: jax.Array,
             highway_lanes: jax.Array, seconds_per_t: float,
             enable_lane_change: bool = True) -> jax.Array:
    '''Step a single NPC vehicle using IDM + MOBIL (decide + integrate).'''
    acceleration, steering, new_target_lane = npc_decide(
        npc_state, all_states, highway_lanes, enable_lane_change)
    return npc_integrate(npc_state, acceleration, steering, new_target_lane,
                         seconds_per_t)


def npc_decide_all(npc_states: jax.Array, ego_state: jax.Array,
                   highway_lanes: jax.Array,
                   enable_lane_change: bool = True):
    '''Compute decisions for all NPCs based on current (old) states.

    Returns (acceleration_by_npc, steering_by_npc, new_target_lane_by_npc).
    '''
    all_vehicles = jnp.concatenate([ego_state[None, :], npc_states], axis=0)

    def decide_one(npc_state):
        return npc_decide(npc_state, all_vehicles, highway_lanes, enable_lane_change)

    return jax.vmap(decide_one)(npc_states)


def npc_integrate_all(npc_states: jax.Array,
                      acceleration_by_npc: jax.Array,
                      steering_by_npc: jax.Array,
                      new_target_lane_by_npc: jax.Array,
                      seconds_per_t: float) -> jax.Array:
    '''Integrate physics for all NPCs given pre-computed decisions.'''
    def integrate_one(npc_state, acceleration, steering, new_target_lane):
        return npc_integrate(npc_state, acceleration, steering, new_target_lane,
                             seconds_per_t)

    return jax.vmap(integrate_one)(
        npc_states, acceleration_by_npc, steering_by_npc, new_target_lane_by_npc)


# Vectorized NPC step for all NPCs at once
def npc_sub_step_all(npc_states: jax.Array, ego_state: jax.Array,
                 highway_lanes: jax.Array, seconds_per_t: float,
                 enable_lane_change: bool = True) -> jax.Array:
    '''Step all NPC vehicles (decide + integrate in one pass).

    Each NPC sees the ego and all NPCs when computing IDM. An NPC won't
    "see" itself as a front/rear vehicle because it's at the same position
    (not strictly ahead or behind).
    '''
    all_vehicles = jnp.concatenate([ego_state[None, :], npc_states], axis=0)

    def step_one_npc(npc_state: jax.Array) -> jax.Array:
        return npc_sub_step(npc_state, all_vehicles, highway_lanes, seconds_per_t,
                            enable_lane_change)

    # Vectorize over all NPCs
    return jax.vmap(step_one_npc)(npc_states)
