'''
Lane geometry for straight lanes in JAX.

Lane Local Coordinate System
----------------------------
Each lane has a local coordinate system with two axes:

- Longitudinal (s): distance along the lane centerline
  - Zero at lane start, increases toward lane end
  - Range: 0 to lane length
  - Units: meters [m]

- Lateral (r): perpendicular distance from lane centerline
  - Zero at lane center
  - Positive: to the LEFT of lane direction (90° counterclockwise)
  - Negative: to the RIGHT of lane direction
  - Units: meters [m]

Lane Parameters Array
---------------------
Lanes are represented as a flat array for JIT compatibility:
  [start_x, start_y, end_x, end_y, width, heading, length, direction_x, direction_y,
   direction_lateral_x, direction_lateral_y]

Use `make_lane_params` to create this array from start/end points.
'''
from __future__ import annotations

import jax
import jax.numpy as jnp

# Indices into lane parameter array
START_X = 0
START_Y = 1
END_X = 2
END_Y = 3
WIDTH = 4
HEADING = 5
LENGTH = 6
DIR_X = 7
DIR_Y = 8
DIR_LAT_X = 9
DIR_LAT_Y = 10
LANE_PARAMS_SIZE = 11

DEFAULT_LANE_WIDTH = 4.0
VEHICLE_LENGTH = 5.0


def make_lane_params(start_x: float, start_y: float, end_x: float, end_y: float,
                     width: float = DEFAULT_LANE_WIDTH) -> jax.Array:
    '''
    Create lane parameters array from start and end points.

    Args:
        start_x, start_y: Lane starting position [m]
        end_x, end_y: Lane ending position [m]
        width: Lane width [m]

    Returns:
        Lane parameters array of shape (LANE_PARAMS_SIZE,)
    '''
    start = jnp.array([start_x, start_y])
    end = jnp.array([end_x, end_y])

    heading = jnp.arctan2(end_y - start_y, end_x - start_x)
    length = jnp.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    direction = (end - start) / length
    direction_lateral = jnp.array([-direction[1], direction[0]])

    return jnp.array([
        start_x, start_y, end_x, end_y, width, heading, length,
        direction[0], direction[1], direction_lateral[0], direction_lateral[1]
    ])


def lane_position(lane_params: jax.Array, longitudinal: float, lateral: float) -> jax.Array:
    '''
    Convert local lane coordinates to world position.

    Args:
        lane_params: Lane parameters array
        longitudinal: Distance along lane centerline [m]
        lateral: Perpendicular distance from centerline [m] (positive=left)

    Returns:
        World position [x, y]
    '''
    start = jnp.array([lane_params[START_X], lane_params[START_Y]])
    direction = jnp.array([lane_params[DIR_X], lane_params[DIR_Y]])
    direction_lateral = jnp.array([lane_params[DIR_LAT_X], lane_params[DIR_LAT_Y]])

    return start + longitudinal * direction + lateral * direction_lateral


def lane_local_coordinates(lane_params: jax.Array, world_x: float, world_y: float
                           ) -> tuple[jax.Array, jax.Array]:
    '''
    Convert world position to local lane coordinates.

    Args:
        lane_params: Lane parameters array
        world_x, world_y: World position [m]

    Returns:
        (longitudinal, lateral) lane coordinates [m]
    '''
    start = jnp.array([lane_params[START_X], lane_params[START_Y]])
    direction = jnp.array([lane_params[DIR_X], lane_params[DIR_Y]])
    direction_lateral = jnp.array([lane_params[DIR_LAT_X], lane_params[DIR_LAT_Y]])

    delta = jnp.array([world_x, world_y]) - start
    longitudinal = jnp.dot(delta, direction)
    lateral = jnp.dot(delta, direction_lateral)

    return longitudinal, lateral


def lane_heading_at(lane_params: jax.Array, longitudinal: float) -> jax.Array:
    '''
    Get lane heading at a given longitudinal position.

    For straight lanes, heading is constant.

    Args:
        lane_params: Lane parameters array
        longitudinal: Distance along lane [m] (unused for straight lanes)

    Returns:
        Lane heading [rad]
    '''
    return lane_params[HEADING]


def lane_width_at(lane_params: jax.Array, longitudinal: float) -> jax.Array:
    '''
    Get lane width at a given longitudinal position.

    For straight lanes, width is constant.

    Args:
        lane_params: Lane parameters array
        longitudinal: Distance along lane [m] (unused for straight lanes)

    Returns:
        Lane width [m]
    '''
    return lane_params[WIDTH]


def lane_on_lane(lane_params: jax.Array, world_x: float, world_y: float,
                 margin: float = 0.0) -> jax.Array:
    '''
    Check if a world position is on the lane.

    Args:
        lane_params: Lane parameters array
        world_x, world_y: World position [m]
        margin: Extra margin around lane width [m]

    Returns:
        True if position is on lane
    '''
    longitudinal, lateral = lane_local_coordinates(lane_params, world_x, world_y)
    width = lane_params[WIDTH]
    length = lane_params[LENGTH]

    within_lateral = jnp.abs(lateral) <= width / 2 + margin
    within_longitudinal = (-VEHICLE_LENGTH <= longitudinal) & \
                                                            (longitudinal < length + VEHICLE_LENGTH)

    return within_lateral & within_longitudinal


def lane_distance(lane_params: jax.Array, world_x: float, world_y: float) -> jax.Array:
    '''
    Compute L1 distance from a position to the lane.

    Args:
        lane_params: Lane parameters array
        world_x, world_y: World position [m]

    Returns:
        L1 distance [m]
    '''
    longitudinal, lateral = lane_local_coordinates(lane_params, world_x, world_y)
    length = lane_params[LENGTH]

    return (jnp.abs(lateral) +
            jnp.maximum(longitudinal - length, 0.0) +
            jnp.maximum(-longitudinal, 0.0))


# Vectorized versions for multiple lanes
make_lane_params_batched = jax.vmap(
    lambda start_x, start_y, end_x, end_y, width: make_lane_params(start_x, start_y, end_x, end_y,
                                                                   width),
    in_axes=(0, 0, 0, 0, 0),
)


def get_lane(highway_lanes: jax.Array, lane_index: int) -> jax.Array:
    '''
    Get lane parameters by index.

    Args:
        highway_lanes: Array of lane parameters, shape (n_lanes, LANE_PARAMS_SIZE)
        lane_index: Index of lane to get

    Returns:
        Lane parameters array of shape (LANE_PARAMS_SIZE,)
    '''
    return highway_lanes[lane_index]


def side_lanes(lane_index: int, n_lanes: int) -> tuple[jax.Array, jax.Array]:
    '''
    Get adjacent lane indices (left and right).

    Args:
        lane_index: Current lane index
        n_lanes: Total number of lanes

    Returns:
        (left_lane_index, right_lane_index) - clipped to valid range
        Returns same index if at boundary
    '''
    left = jnp.clip(lane_index - 1, 0, n_lanes - 1)
    right = jnp.clip(lane_index + 1, 0, n_lanes - 1)
    return left, right


def make_highway_lanes(n_lanes: int, lane_length: float, lane_width: float = DEFAULT_LANE_WIDTH,
                       lane_spacing: float = 0.0) -> jax.Array:
    '''
    Create parallel lanes for a highway.

    Lanes run along the x-axis. Lane 0 is at y=0, lane 1 at y=lane_width, etc.

    Args:
        n_lanes: Number of lanes
        lane_length: Length of each lane [m]
        lane_width: Width of each lane [m]
        lane_spacing: Extra spacing between lanes [m] (usually 0)

    Returns:
        Lane parameters array of shape (n_lanes, LANE_PARAMS_SIZE)
    '''
    effective_width = lane_width + lane_spacing

    # Each lane centerline: y = lane_idx * effective_width + lane_width/2
    # But for simplicity, let's put lane centers at y = lane_idx * lane_width
    # So lane 0 center is at y=0, lane 1 center at y=lane_width, etc.
    lane_indices = jnp.arange(n_lanes)
    y_centers = lane_indices * effective_width

    start_x = jnp.zeros(n_lanes)
    start_y = y_centers
    end_x = jnp.full(n_lanes, lane_length)
    end_y = y_centers
    widths = jnp.full(n_lanes, lane_width)

    return make_lane_params_batched(start_x, start_y, end_x, end_y, widths)
