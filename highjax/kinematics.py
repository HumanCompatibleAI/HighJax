'''
Vehicle kinematics using a modified bicycle model in JAX.

Bicycle Model Physics
---------------------
The vehicle uses a simplified bicycle model with front-wheel steering:

    beta = arctan(0.5 * tan(steering))     # Slip angle at center of mass
    velocity = speed * [cos(heading + beta), sin(heading + beta)]
    position += velocity * dt
    heading += speed * sin(beta) / (LENGTH/2) * dt
    speed += acceleration * dt

Servo
-----
The ego vehicle uses a two-level control architecture:

1. **Policy** (RL agent): picks discrete meta-actions (LANE_LEFT, IDLE, etc.)
   at the policy frequency. This sets strategic targets (target lane, target speed).
2. **Servo**: proportional tracking controllers that follow those targets.
   Lateral: cascaded P-controller (position → heading → steering).
   Longitudinal: simple P-controller (target_speed → acceleration).

The servo is analogous to highway-env's ControlledVehicle. NPCs don't use the
servo — they use IDM for longitudinal control and share only the lateral
steering_control with the ego.

Vehicle State Array
-------------------
Vehicle state is a flat array for JIT compatibility:
  [x, y, heading, speed, target_lane_idx, speed_index, target_speed,
   lane_change_timer, idm_delta, crashed]

See VehicleState indices below for accessing individual components.
'''
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import lanes

# Vehicle constants
VEHICLE_LENGTH = 5.0  # [m]
VEHICLE_WIDTH = 2.0   # [m]
HALF_LENGTH = VEHICLE_LENGTH / 2
MAX_SPEED = 40.0      # [m/s]
MIN_SPEED = -40.0     # [m/s]
SLIP_ANGLE_FACTOR = 0.5

# Controller constants (from ControlledVehicle)
TAU_ACC = 0.6         # [s]
TAU_HEADING = 0.2     # [s]
TAU_LATERAL = 0.6     # [s]
TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
KP_A = 1 / TAU_ACC
KP_HEADING = 1 / TAU_HEADING
KP_LATERAL = 1 / TAU_LATERAL
MAX_STEERING_ANGLE = jnp.pi / 3  # [rad]

# Default target speeds for MDPVehicle (linspace(20, 30, 3) = [20, 25, 30])
DEFAULT_TARGET_SPEEDS = jnp.array([20.0, 25.0, 30.0])

# Vehicle state indices
X = 0
Y = 1
HEADING = 2
SPEED = 3
TARGET_LANE_IDX = 4
SPEED_INDEX = 5
TARGET_SPEED = 6
LANE_CHANGE_TIMER = 7
IDM_DELTA = 8
CRASHED = 9
VEHICLE_STATE_SIZE = 10

# Actions
ACTION_LANE_LEFT = 0
ACTION_IDLE = 1
ACTION_LANE_RIGHT = 2
ACTION_FASTER = 3
ACTION_SLOWER = 4


LANE_CHANGE_DELAY = 1.0  # [s] Matching HE's LANE_CHANGE_DELAY


def make_vehicle_state(x: float, y: float, heading: float, speed: float,
                       target_lane_idx: int, target_speeds: jax.Array = DEFAULT_TARGET_SPEEDS,
                       idm_delta: float = 4.0,
                       ) -> jax.Array:
    '''
    Create a vehicle state array.

    Args:
        x, y: Position [m]
        heading: Heading angle [rad]
        speed: Forward speed [m/s]
        target_lane_idx: Index of target lane
        target_speeds: Array of allowed target speeds [m/s]
        idm_delta: IDM acceleration exponent (randomized per vehicle)

    Returns:
        Vehicle state array of shape (VEHICLE_STATE_SIZE,)
    '''
    speed_index = speed_to_index(speed, target_speeds)
    target_speed = target_speeds[speed_index]
    # Initialize timer matching HE: (sum(position) * pi) % LANE_CHANGE_DELAY
    lane_change_timer = ((x + y) * jnp.pi) % LANE_CHANGE_DELAY

    return jnp.array([x, y, heading, speed, target_lane_idx, speed_index,
                       target_speed, lane_change_timer, idm_delta, 0.0])


def speed_to_index(speed: float, target_speeds: jax.Array = DEFAULT_TARGET_SPEEDS) -> jax.Array:
    '''Convert speed to index in target_speeds array.'''
    x = (speed - target_speeds[0]) / (target_speeds[-1] - target_speeds[0] + 1e-8)
    return jnp.clip(jnp.round(x * (target_speeds.size - 1)),
                    0, target_speeds.size - 1).astype(jnp.int32)



def wrap_to_pi(angle: jax.Array) -> jax.Array:
    '''Wrap angle to [-pi, pi].'''
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


def not_zero(x: jax.Array, eps: float = 1e-2) -> jax.Array:
    '''Return x if abs(x) > eps, else eps with same sign.'''
    return jnp.where(jnp.abs(x) > eps, x, jnp.sign(x) * eps + (x == 0) * eps)


def steering_control(vehicle_state: jax.Array, lane_params: jax.Array) -> jax.Array:
    '''
    Compute steering angle to follow lane center (servo lateral component).

    Uses cascaded control: lateral position → lateral speed → heading → heading rate → steering.
    Shared by both ego servo and NPC steering.

    Args:
        vehicle_state: Vehicle state array
        lane_params: Target lane parameters

    Returns:
        Steering angle [rad]
    '''
    x, y = vehicle_state[X], vehicle_state[Y]
    heading = vehicle_state[HEADING]
    speed = vehicle_state[SPEED]

    # Get position in lane coordinates
    long, lat = lanes.lane_local_coordinates(lane_params, x, y)

    # Future lane heading (look-ahead)
    lane_next_coords = long + speed * TAU_PURSUIT
    lane_future_heading = lanes.lane_heading_at(lane_params, lane_next_coords)

    # Lateral position control
    lateral_speed_command = -KP_LATERAL * lat

    # Lateral speed to heading
    heading_command = jnp.arcsin(jnp.clip(lateral_speed_command / not_zero(speed), -1, 1))
    heading_ref = lane_future_heading + jnp.clip(heading_command, -jnp.pi / 4, jnp.pi / 4)

    # Heading control
    heading_rate_command = KP_HEADING * wrap_to_pi(heading_ref - heading)

    # Heading rate to steering angle
    slip_angle = jnp.arcsin(jnp.clip(HALF_LENGTH / not_zero(speed) * heading_rate_command, -1, 1))
    # arctan(2*tan(x)) has a sign discontinuity at x=±pi/2 due to tan overflow.
    # Use copysign to preserve the intended sign from slip_angle.
    steering_angle = jnp.copysign(
        jnp.abs(jnp.arctan(2 * jnp.tan(slip_angle))), slip_angle
    )
    steering_angle = jnp.clip(steering_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

    return steering_angle


def speed_control(vehicle_state: jax.Array) -> jax.Array:
    '''
    Compute acceleration to track target speed (servo longitudinal component).

    Simple proportional control. Only used by the ego servo — NPCs use IDM instead.

    Args:
        vehicle_state: Vehicle state array

    Returns:
        Acceleration [m/s²]
    '''
    speed = vehicle_state[SPEED]
    target_speed = vehicle_state[TARGET_SPEED]
    return KP_A * (target_speed - speed)


def bicycle_forward(x: float, y: float, heading: float, speed: float,
                 steering: float, acceleration: float, dt: float
                 ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    '''
    Single step of bicycle model kinematics.

    Args:
        x, y: Current position [m]
        heading: Current heading [rad]
        speed: Current speed [m/s]
        steering: Steering angle [rad]
        acceleration: Acceleration [m/s²]
        dt: Time delta [s] (seconds_per_t or seconds_per_sub_t)

    Returns:
        new_x, new_y, new_heading, new_speed
    '''
    # Clip speed
    speed = jnp.clip(speed, MIN_SPEED, MAX_SPEED)

    # Compute slip angle
    beta = jnp.arctan(SLIP_ANGLE_FACTOR * jnp.tan(steering))

    # Velocity in world frame
    vx = speed * jnp.cos(heading + beta)
    vy = speed * jnp.sin(heading + beta)

    # Update position
    new_x = x + vx * dt
    new_y = y + vy * dt

    # Update heading
    new_heading = heading + speed * jnp.sin(beta) / HALF_LENGTH * dt

    # Update speed
    new_speed = jnp.clip(speed + acceleration * dt, MIN_SPEED, MAX_SPEED)

    return new_x, new_y, new_heading, new_speed


def execute_action(vehicle_state: jax.Array, action: int, n_lanes: int,
                   target_speeds: jax.Array = DEFAULT_TARGET_SPEEDS) -> jax.Array:
    '''
    Execute a high-level action (LANE_LEFT, LANE_RIGHT, FASTER, SLOWER, IDLE).

    Updates target_lane_idx and speed_index/target_speed based on action.

    Args:
        vehicle_state: Current vehicle state
        action: Action index (0-4)
        n_lanes: Number of lanes on road
        target_speeds: Array of allowed target speeds

    Returns:
        Updated vehicle state with new targets
    '''
    target_lane_idx = vehicle_state[TARGET_LANE_IDX].astype(jnp.int32)
    n_speeds = target_speeds.size

    # Re-derive speed index from actual speed (matching HE, fix #6)
    speed_index = speed_to_index(vehicle_state[SPEED], target_speeds)

    # Update target lane based on action
    new_lane_idx = jnp.where(
        action == ACTION_LANE_LEFT,
        jnp.clip(target_lane_idx - 1, 0, n_lanes - 1),
        jnp.where(
            action == ACTION_LANE_RIGHT,
            jnp.clip(target_lane_idx + 1, 0, n_lanes - 1),
            target_lane_idx
        )
    )

    # Update speed index based on action
    new_speed_index = jnp.where(
        action == ACTION_FASTER,
        jnp.clip(speed_index + 1, 0, n_speeds - 1),
        jnp.where(
            action == ACTION_SLOWER,
            jnp.clip(speed_index - 1, 0, n_speeds - 1),
            speed_index
        )
    )

    new_target_speed = target_speeds[new_speed_index]

    return (
        vehicle_state
        .at[TARGET_LANE_IDX].set(new_lane_idx.astype(jnp.float32))
        .at[SPEED_INDEX].set(new_speed_index.astype(jnp.float32))
        .at[TARGET_SPEED].set(new_target_speed)
    )


def servo_sub_step(vehicle_state: jax.Array, highway_lanes: jax.Array,
                   seconds_per_sub_t: float) -> jax.Array:
    '''
    Single servo sub-step: compute controls from targets and apply kinematics.

    The servo is the ego's low-level tracking controller. Given strategic targets
    (target lane and target speed) already set in the vehicle state, it computes
    steering and acceleration via proportional controllers, then advances the
    bicycle model by one sub_t.

    Args:
        vehicle_state: Vehicle state with targets already set
        highway_lanes: Array of lane parameters, shape (n_lanes, LANE_PARAMS_SIZE)
        seconds_per_sub_t: Time delta for one servo sub_t [s]

    Returns:
        Updated vehicle state with new position/heading/speed
    '''
    target_lane_idx = vehicle_state[TARGET_LANE_IDX].astype(jnp.int32)
    target_lane = highway_lanes[target_lane_idx]

    steering = steering_control(vehicle_state, target_lane)
    acceleration = speed_control(vehicle_state)

    new_x, new_y, new_heading, new_speed = bicycle_forward(
        vehicle_state[X], vehicle_state[Y], vehicle_state[HEADING],
        vehicle_state[SPEED], steering, acceleration, seconds_per_sub_t
    )

    return (vehicle_state
            .at[X].set(new_x).at[Y].set(new_y)
            .at[HEADING].set(new_heading).at[SPEED].set(new_speed))


def vehicle_step(vehicle_state: jax.Array, action: int, highway_lanes: jax.Array,
                 seconds_per_t: float, target_speeds: jax.Array = DEFAULT_TARGET_SPEEDS
                 ) -> jax.Array:
    '''
    Full ego vehicle step for one timestep: execute action, then run servo.

    Two-level control: the policy's action sets strategic targets (lane, speed),
    then the servo tracks those targets.

    Args:
        vehicle_state: Current vehicle state
        action: High-level action (0-4)
        highway_lanes: Array of lane parameters, shape (n_lanes, LANE_PARAMS_SIZE)
        seconds_per_t: Time per timestep [s]
        target_speeds: Array of allowed target speeds

    Returns:
        New vehicle state
    '''
    n_lanes = highway_lanes.shape[0]

    # Policy level: execute high-level action (set targets)
    state = execute_action(vehicle_state, action, n_lanes, target_speeds)

    # Servo level: track targets (currently 1 sub_t per t)
    return servo_sub_step(state, highway_lanes, seconds_per_t)


# =============================================================================
# Rotated Rectangle Collision Detection (SAT Algorithm)
# =============================================================================

def get_vehicle_corners(x: jax.Array, y: jax.Array, heading: jax.Array,
                        length: float = VEHICLE_LENGTH,
                        width: float = VEHICLE_WIDTH) -> jax.Array:
    '''
    Get the 4 corners of a rotated rectangle (vehicle).

    Args:
        x, y: Center position
        heading: Rotation angle in radians
        length, width: Vehicle dimensions

    Returns:
        corners: Array of shape (4, 2) with corner positions
    '''
    cos_h = jnp.cos(heading)
    sin_h = jnp.sin(heading)

    # Half dimensions
    hl = length / 2
    hw = width / 2

    # Local corner offsets (before rotation)
    # Order: front-left, front-right, rear-right, rear-left
    local_corners = jnp.array([
        [hl, hw],
        [hl, -hw],
        [-hl, -hw],
        [-hl, hw],
    ])

    # Rotation matrix
    rot = jnp.array([[cos_h, -sin_h], [sin_h, cos_h]])

    # Rotate and translate
    corners = jnp.dot(local_corners, rot.T) + jnp.array([x, y])
    return corners


def _project_polygon_onto_axis(corners: jax.Array, axis: jax.Array) -> tuple[jax.Array, jax.Array]:
    '''Project polygon corners onto an axis, return (min, max) projections.'''
    projections = jnp.dot(corners, axis)
    return jnp.min(projections), jnp.max(projections)


def _intervals_overlap(min_a: jax.Array, max_a: jax.Array,
                       min_b: jax.Array, max_b: jax.Array) -> jax.Array:
    '''Check if two intervals [min_a, max_a] and [min_b, max_b] overlap.'''
    return (min_a <= max_b) & (min_b <= max_a)


def rotated_rectangles_intersect(
    corners_a: jax.Array, corners_b: jax.Array,
    vel_a: jax.Array = None, vel_b: jax.Array = None,
    dt: float = 0.1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    '''
    Check if two rotated rectangles intersect using SAT algorithm,
    with predictive collision detection via velocity projection.

    Args:
        corners_a: Corners of rectangle A, shape (4, 2)
        corners_b: Corners of rectangle B, shape (4, 2)
        vel_a: Velocity of A, shape (2,) [vx, vy]
        vel_b: Velocity of B, shape (2,) [vx, vy]
        dt: Time delta for prediction [s]

    Returns:
        (intersecting, will_intersect, translation_vector)
        - intersecting: True if currently overlapping
        - will_intersect: True if will overlap within dt
        - translation_vector: minimum translation to separate (shape (2,))
    '''
    if vel_a is None:
        vel_a = jnp.zeros(2)
    if vel_b is None:
        vel_b = jnp.zeros(2)

    # Edge vectors (2 per rectangle)
    edges_a = jnp.array([
        corners_a[1] - corners_a[0],  # Front edge
        corners_a[3] - corners_a[0],  # Left edge
    ])
    edges_b = jnp.array([
        corners_b[1] - corners_b[0],
        corners_b[3] - corners_b[0],
    ])

    # Perpendicular normal axes
    def get_normal(edge):
        return jnp.array([-edge[1], edge[0]])

    axes = jnp.array([
        get_normal(edges_a[0]),
        get_normal(edges_a[1]),
        get_normal(edges_b[0]),
        get_normal(edges_b[1]),
    ])

    # Normalize axes
    norms = jnp.linalg.norm(axes, axis=1, keepdims=True)
    axes = axes / jnp.maximum(norms, 1e-8)

    # Relative displacement for prediction
    displacement = (vel_a - vel_b) * dt

    def check_axis(axis):
        min_a, max_a = _project_polygon_onto_axis(corners_a, axis)
        min_b, max_b = _project_polygon_onto_axis(corners_b, axis)

        # Current overlap check
        current_overlap = _intervals_overlap(min_a, max_a, min_b, max_b)

        # Predictive: extend A's interval by velocity projection
        vel_proj = jnp.dot(displacement, axis)
        ext_min_a = jnp.minimum(min_a, min_a + vel_proj)
        ext_max_a = jnp.maximum(max_a, max_a + vel_proj)
        predicted_overlap = _intervals_overlap(ext_min_a, ext_max_a, min_b, max_b)

        # Overlap amount for translation vector
        # overlap_left: how much A extends past B's min (push A in -axis dir)
        # overlap_right: how much B extends past A's min (push A in +axis dir)
        overlap_left = max_a - min_b
        overlap_right = max_b - min_a
        # Pick the direction with minimum penetration
        push_negative = overlap_left < overlap_right
        min_overlap = jnp.minimum(overlap_left, overlap_right)
        # Sign: negative if pushing A in -axis direction
        signed_overlap = jnp.where(push_negative, -min_overlap, min_overlap)

        return current_overlap, predicted_overlap, signed_overlap

    # Vectorize over axes
    current_overlaps, predicted_overlaps, signed_overlaps = jax.vmap(check_axis)(axes)

    intersecting = jnp.all(current_overlaps)
    will_intersect = jnp.all(predicted_overlaps)

    # Translation vector: axis with minimum overlap, scaled
    abs_overlaps = jnp.abs(signed_overlaps)
    # Only consider axes where there IS overlap for translation
    abs_overlaps_masked = jnp.where(current_overlaps | will_intersect, abs_overlaps, jnp.inf)
    min_axis_idx = jnp.argmin(abs_overlaps_masked)
    translation = axes[min_axis_idx] * signed_overlaps[min_axis_idx]

    # Zero translation if no collision predicted
    translation = jnp.where(intersecting | will_intersect, translation, jnp.zeros(2))

    return intersecting, will_intersect, translation


def _vehicle_velocity(state: jax.Array) -> jax.Array:
    '''Get 2D velocity vector from vehicle state.'''
    return jnp.array([
        state[SPEED] * jnp.cos(state[HEADING]),
        state[SPEED] * jnp.sin(state[HEADING]),
    ])


def check_vehicle_collision(ego_state: jax.Array, npc_state: jax.Array,
                            dt: float = 0.1
                            ) -> tuple[jax.Array, jax.Array, jax.Array]:
    '''
    Check collision between ego vehicle and one NPC using predictive SAT.

    Args:
        ego_state: Ego vehicle state array
        npc_state: NPC vehicle state array
        dt: Time delta for prediction [s]

    Returns:
        (colliding, will_collide, translation_vector)
    '''
    ego_corners = get_vehicle_corners(ego_state[X], ego_state[Y], ego_state[HEADING])
    npc_corners = get_vehicle_corners(npc_state[X], npc_state[Y], npc_state[HEADING])
    ego_vel = _vehicle_velocity(ego_state)
    npc_vel = _vehicle_velocity(npc_state)
    return rotated_rectangles_intersect(ego_corners, npc_corners, ego_vel, npc_vel, dt)


def check_vehicle_collisions_all(ego_state: jax.Array, npc_states: jax.Array,
                                 dt: float = 0.1
                                 ) -> tuple[jax.Array, jax.Array, jax.Array]:
    '''
    Check collision between ego vehicle and all NPCs with impulse.

    Returns colliding and will_collide separately so the caller can decide
    which triggers crash vs impulse.

    Args:
        ego_state: Ego vehicle state array
        npc_states: NPC states array of shape (n_npcs, VEHICLE_STATE_SIZE)
        dt: Time delta for prediction [s]

    Returns:
        (per_npc_colliding, per_npc_will_collide, per_npc_translations)
        - per_npc_colliding: (n_npcs,) bool, currently overlapping
        - per_npc_will_collide: (n_npcs,) bool, predicted overlap
        - per_npc_translations: (n_npcs, 2) translation vectors
    '''
    def check_one(npc):
        return check_vehicle_collision(ego_state, npc, dt)

    per_npc_colliding, per_npc_will_collide, per_npc_translations = (
        jax.vmap(check_one)(npc_states))
    return per_npc_colliding, per_npc_will_collide, per_npc_translations


def check_npc_npc_collisions_all(npc_states: jax.Array,
                                  dt: float = 0.1):
    '''Check all NPC-NPC collisions and compute per-NPC impulse vectors.

    Uses N*N nested vmap with upper-triangle mask to avoid self-collision
    and double-counting.

    Returns (impulse_by_npc, colliding_by_npc):
      impulse_by_npc: shape (n_npcs, 2) — total displacement per NPC
      colliding_by_npc: shape (n_npcs,) — True if NPC is in any collision
    '''
    n_npcs = npc_states.shape[0]

    def check_pair(state_a, state_b):
        return check_vehicle_collision(state_a, state_b, dt)

    # N*N collision check: colliding[i][j] = collision between NPC i and NPC j
    check_one_vs_all = jax.vmap(check_pair, in_axes=(None, 0))
    colliding_by_j_by_i, will_collide_by_j_by_i, translation_by_j_by_i = (
        jax.vmap(check_one_vs_all, in_axes=(0, None))(npc_states, npc_states))

    # Upper triangle mask (exclude self and lower triangle)
    triu_mask = jnp.triu(jnp.ones((n_npcs, n_npcs), dtype=bool), k=1)
    impulse_mask = (colliding_by_j_by_i | will_collide_by_j_by_i) & triu_mask

    # For pair (i, j) with i < j: NPC i gets +translation[i][j]/2,
    # NPC j gets -translation[i][j]/2
    pair_impulses = jnp.where(
        impulse_mask[:, :, None], translation_by_j_by_i / 2, 0.0)

    # Accumulate: positive impulses (NPC i as "ego") + negative (NPC j as "other")
    impulse_by_npc = jnp.sum(pair_impulses, axis=1) - jnp.sum(pair_impulses, axis=0)

    # Which NPCs are involved in any collision (row OR column of impulse_mask)
    colliding_by_npc = jnp.any(impulse_mask, axis=1) | jnp.any(impulse_mask, axis=0)

    return impulse_by_npc, colliding_by_npc
