'''Observation construction for HighJax environment.

Computes the kinematic observation: a (n_observed_vehicles, n_features) array
where row 0 is ego and rows 1..N are the closest NPCs sorted by distance.
'''
from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from . import kinematics, lanes
from .env import EnvState, ALL_FEATURES


def compute_observation(state: EnvState, env, params) -> jax.Array:
    '''Compute the kinematic observation from an EnvState.

    Args:
        state: Current environment state
        env: HighJaxEnv instance (structural params)
        params: EnvParams (tunable params)

    Returns:
        Observation array of shape (n_observed_vehicles, n_features)
    '''
    n_obs = env.n_observed_vehicles
    n_all_features = len(ALL_FEATURES)
    highway_lanes = env.highway_lanes

    x_range = params.x_range
    y_range = env.lane_width * env.n_lanes
    v_range = params.v_range

    ego_x = state.ego_state[kinematics.X]
    ego_y = state.ego_state[kinematics.Y]
    ego_heading = state.ego_state[kinematics.HEADING]
    ego_speed = state.ego_state[kinematics.SPEED]

    # Ego velocity
    ego_vx = ego_speed * jnp.cos(ego_heading)
    ego_vy = ego_speed * jnp.sin(ego_heading)

    # Compute all features for ego (in ALL_FEATURES order)
    ego_target_lane_idx = state.ego_state[kinematics.TARGET_LANE_IDX].astype(
        jnp.int32)
    ego_lane_params = highway_lanes[ego_target_lane_idx]
    ego_long, ego_lat = lanes.lane_local_coordinates(
        ego_lane_params, ego_x, ego_y)
    ego_lane_heading = lanes.lane_heading_at(ego_lane_params, ego_long)
    ego_ang_off = ego_heading - ego_lane_heading

    # Ego x/y/vx/vy: absolute normalized
    ego_x_obs = jnp.clip(ego_x / x_range, -1, 1)
    ego_y_obs = jnp.clip(ego_y / y_range, -1, 1)
    ego_vx_obs = jnp.clip(ego_vx / v_range, -1, 1)
    ego_vy_obs = jnp.clip(ego_vy / v_range, -1, 1)

    ego_all_features = jnp.array([
        1.0,  # presence
        ego_x_obs,  # x
        ego_y_obs,  # y
        ego_vx_obs,  # vx
        ego_vy_obs,  # vy
        jnp.clip(ego_heading / jnp.pi, -1, 1),  # heading
        jnp.cos(ego_heading),  # cos_h
        jnp.sin(ego_heading),  # sin_h
        0.0,  # cos_d (no destination in highway)
        0.0,  # sin_d
        jnp.clip(ego_long / env.lane_length, -1, 1),  # long_off
        jnp.clip(ego_lat / y_range, -1, 1),  # lat_off
        jnp.clip(ego_ang_off / jnp.pi, -1, 1),  # ang_off
        ego_y / env.lane_width - 0.5,  # lane
    ])

    # Compute all features for each NPC
    lane_length = env.lane_length
    lane_width = env.lane_width

    def compute_npc_features(npc_state):
        npc_x = npc_state[kinematics.X]
        npc_y = npc_state[kinematics.Y]
        npc_heading = npc_state[kinematics.HEADING]
        npc_speed = npc_state[kinematics.SPEED]
        npc_vx = npc_speed * jnp.cos(npc_heading)
        npc_vy = npc_speed * jnp.sin(npc_heading)

        # Relative position and velocity
        rel_x = npc_x - ego_x
        rel_y = npc_y - ego_y
        rel_vx = npc_vx - ego_vx
        rel_vy = npc_vy - ego_vy

        # Heading relative to ego
        heading_rel = npc_heading - ego_heading

        # Lane offsets
        npc_target_lane_idx = npc_state[
            kinematics.TARGET_LANE_IDX].astype(jnp.int32)
        npc_lane_params = highway_lanes[npc_target_lane_idx]
        npc_long, npc_lat = lanes.lane_local_coordinates(
            npc_lane_params, npc_x, npc_y)
        npc_lane_heading = lanes.lane_heading_at(npc_lane_params, npc_long)
        npc_ang_off = npc_heading - npc_lane_heading

        return jnp.array([
            1.0,  # presence
            jnp.clip(rel_x / x_range, -1, 1),  # x
            jnp.clip(rel_y / y_range, -1, 1),  # y
            jnp.clip(rel_vx / v_range, -1, 1),  # vx
            jnp.clip(rel_vy / v_range, -1, 1),  # vy
            jnp.clip(heading_rel / jnp.pi, -1, 1),  # heading
            jnp.cos(npc_heading),  # cos_h
            jnp.sin(npc_heading),  # sin_h
            0.0,  # cos_d
            0.0,  # sin_d
            jnp.clip(npc_long / lane_length, -1, 1),  # long_off
            jnp.clip(npc_lat / y_range, -1, 1),  # lat_off
            jnp.clip(npc_ang_off / jnp.pi, -1, 1),  # ang_off
            npc_y / lane_width - 0.5,  # lane
        ])

    npc_all_features = jax.vmap(compute_npc_features)(
        state.npc_states)  # (n_npcs, n_all_features)

    # Compute visibility mask
    n_npcs = env.n_npcs
    visible_by_npc = jnp.ones(n_npcs, dtype=jnp.bool_)
    if not env.see_behind:
        behind_threshold = -2 * kinematics.VEHICLE_LENGTH
        npc_rel_x_by_npc = state.npc_states[:, kinematics.X] - ego_x
        visible_by_npc = visible_by_npc & (
            npc_rel_x_by_npc >= behind_threshold)
    if math.isfinite(env.perception_distance):
        dx = state.npc_states[:, kinematics.X] - ego_x
        dy = state.npc_states[:, kinematics.Y] - ego_y
        npc_distances = jnp.sqrt(dx ** 2 + dy ** 2)
        visible_by_npc = visible_by_npc & (
            npc_distances <= env.perception_distance)

    # Zero out invisible NPCs
    npc_all_features = jnp.where(
        visible_by_npc[:, None], npc_all_features, 0.0)

    # Sort NPCs by lane distance (longitudinal)
    npc_lane_distances = jnp.abs(
        state.npc_states[:, kinematics.X] - ego_x)
    sort_distance_by_npc = jnp.where(
        visible_by_npc, npc_lane_distances, jnp.inf)
    sorted_indices = jnp.argsort(sort_distance_by_npc)
    sorted_npc_all_features = npc_all_features[sorted_indices]

    # Take closest n_obs - 1 NPCs
    n_npc_obs = min(n_npcs, n_obs - 1)
    selected_npc_all_features = sorted_npc_all_features[:n_npc_obs]

    # Pad if needed
    padding_needed = n_obs - 1 - n_npc_obs
    padding = jnp.zeros((padding_needed, n_all_features))
    padded_npc_all_features = jnp.concatenate(
        [selected_npc_all_features, padding])[:n_obs - 1]

    # Stack ego + NPCs
    all_vehicles_all_features = jnp.concatenate([
        ego_all_features[None, :],
        padded_npc_all_features
    ])  # (n_obs, n_all_features)

    # Select only the requested features
    feature_idx_array = jnp.array(env.feature_indices)
    observation = all_vehicles_all_features[:, feature_idx_array]

    return observation
