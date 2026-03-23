'''Physics stepping logic for HighJax environment.

Separates stepping concerns (sub-step physics, collision detection) from
state representation (EnvState) and environment config (EnvParams).
'''
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import kinematics, idm


def step_physics(ego_state: jax.Array, npc_states: jax.Array,
                 ego_crashed: jax.Array, highway_lanes: jax.Array,
                 env, params
                 ) -> tuple[jax.Array, jax.Array, jax.Array]:
    '''Run physics sub-steps with collision detection.

    Args:
        ego_state: Ego vehicle state after action execution
        npc_states: NPC states array
        ego_crashed: Whether ego was already crashed
        highway_lanes: Precomputed lane parameters
        env: HighJaxEnv instance (structural params)
        params: EnvParams (tunable params)

    Returns:
        (new_ego_state, new_npc_states, ego_crashed)
    '''
    seconds_per_sub_t = params.seconds_per_sub_t
    n_sub_ts = env.n_sub_ts_per_t
    enable_lane_change = env.enable_npc_lane_change
    simultaneous = env.simultaneous_update
    crash_on_predicted = env.crash_on_predicted
    npc_npc_collisions = env.npc_npc_collisions
    npc_crash_braking = env.npc_crash_braking

    def sub_step(carry):
        ego, npcs, crashed = carry

        # Braking: recompute from current speed each sub-step
        braking_acc = -1.0 * ego[kinematics.SPEED]

        if simultaneous:
            # Decision phase: all vehicles decide based on old states
            npc_acc, npc_steer, npc_lane = idm.npc_decide_all(
                npcs, ego, highway_lanes, enable_lane_change)

            # Integration phase: all vehicles integrate
            servo_ego = kinematics.servo_sub_step(
                ego, highway_lanes, seconds_per_sub_t)
            braked_x, braked_y, braked_h, braked_v = (
                kinematics.bicycle_forward(
                    ego[kinematics.X], ego[kinematics.Y],
                    ego[kinematics.HEADING], ego[kinematics.SPEED],
                    0.0, braking_acc, seconds_per_sub_t))
            braked_ego = (ego
                          .at[kinematics.X].set(braked_x)
                          .at[kinematics.Y].set(braked_y)
                          .at[kinematics.HEADING].set(braked_h)
                          .at[kinematics.SPEED].set(braked_v))
            ego = jnp.where(crashed, braked_ego, servo_ego)
            npcs = idm.npc_integrate_all(
                npcs, npc_acc, npc_steer, npc_lane,
                seconds_per_sub_t)
        else:
            # Sequential: ego integrates first, NPCs see new ego
            servo_ego = kinematics.servo_sub_step(
                ego, highway_lanes, seconds_per_sub_t)
            braked_x, braked_y, braked_h, braked_v = (
                kinematics.bicycle_forward(
                    ego[kinematics.X], ego[kinematics.Y],
                    ego[kinematics.HEADING], ego[kinematics.SPEED],
                    0.0, braking_acc, seconds_per_sub_t))
            braked_ego = (ego
                          .at[kinematics.X].set(braked_x)
                          .at[kinematics.Y].set(braked_y)
                          .at[kinematics.HEADING].set(braked_h)
                          .at[kinematics.SPEED].set(braked_v))
            ego = jnp.where(crashed, braked_ego, servo_ego)
            npcs = idm.npc_sub_step_all(
                npcs, ego, highway_lanes,
                seconds_per_sub_t,
                enable_lane_change=enable_lane_change)

        return (ego, npcs, crashed)

    def sub_step_with_collision(carry):
        ego, npcs, crashed = carry
        ego, npcs, crashed = sub_step((ego, npcs, crashed))

        # Ego-NPC collision check and impulse
        per_npc_colliding, per_npc_will_collide, per_npc_translations = (
            kinematics.check_vehicle_collisions_all(
                ego, npcs, dt=seconds_per_sub_t))
        per_npc_impulse_mask = per_npc_colliding | per_npc_will_collide

        ego_impulse = jnp.sum(
            jnp.where(per_npc_impulse_mask[:, None],
                      per_npc_translations / 2, 0.0),
            axis=0)
        ego = (ego
               .at[kinematics.X].add(ego_impulse[0])
               .at[kinematics.Y].add(ego_impulse[1]))

        npc_ego_impulse_by_npc = jnp.where(
            per_npc_impulse_mask[:, None],
            -per_npc_translations / 2, 0.0)
        npcs = (npcs
                .at[:, kinematics.X].add(npc_ego_impulse_by_npc[:, 0])
                .at[:, kinematics.Y].add(npc_ego_impulse_by_npc[:, 1]))

        # Per-NPC collision tracking
        if crash_on_predicted:
            npc_collided_by_npc = per_npc_impulse_mask
        else:
            npc_collided_by_npc = per_npc_colliding

        # NPC-NPC collision check and impulse
        if npc_npc_collisions:
            npc_npc_impulse_by_npc, npc_npc_colliding_by_npc = (
                kinematics.check_npc_npc_collisions_all(
                    npcs, dt=seconds_per_sub_t))
            npcs = (npcs
                    .at[:, kinematics.X].add(npc_npc_impulse_by_npc[:, 0])
                    .at[:, kinematics.Y].add(npc_npc_impulse_by_npc[:, 1]))
            npc_collided_by_npc = (npc_collided_by_npc
                                   | npc_npc_colliding_by_npc)

        # Propagate ego crash flag per sub-step
        if crash_on_predicted:
            collision = jnp.any(per_npc_impulse_mask)
        else:
            collision = jnp.any(per_npc_colliding)
        crashed = crashed | collision

        # Propagate NPC crash flags per sub-step
        if npc_crash_braking:
            npc_already_crashed = npcs[:, kinematics.CRASHED] > 0.5
            npc_now_crashed = npc_already_crashed | npc_collided_by_npc
            npcs = npcs.at[:, kinematics.CRASHED].set(
                npc_now_crashed.astype(jnp.float32))

        return (ego, npcs, crashed)

    # Run n_sub_ts sub-steps
    init_carry = (ego_state, npc_states, ego_crashed)
    new_ego, new_npcs, new_crashed = jax.lax.fori_loop(
        0, n_sub_ts,
        lambda i, carry: sub_step_with_collision(carry),
        init_carry)

    return new_ego, new_npcs, new_crashed


def step_physics_with_sub_states(
    ego_state: jax.Array, npc_states: jax.Array,
    ego_crashed: jax.Array, highway_lanes: jax.Array,
    env, params,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    '''Like step_physics but also returns intermediate sub-step states.

    Returns:
        (new_ego, new_npcs, crashed, sub_ego_by_sub_t, sub_npcs_by_sub_t)
        where sub arrays have shape (n_sub_ts, ...).
    '''
    seconds_per_sub_t = params.seconds_per_sub_t
    n_sub_ts = env.n_sub_ts_per_t
    enable_lane_change = env.enable_npc_lane_change
    simultaneous = env.simultaneous_update
    crash_on_predicted = env.crash_on_predicted
    npc_npc_collisions = env.npc_npc_collisions
    npc_crash_braking = env.npc_crash_braking

    def sub_step_fn(carry, _):
        ego, npcs, crashed = carry

        braking_acc = -1.0 * ego[kinematics.SPEED]

        if simultaneous:
            npc_acc, npc_steer, npc_lane = idm.npc_decide_all(
                npcs, ego, highway_lanes, enable_lane_change)
            servo_ego = kinematics.servo_sub_step(
                ego, highway_lanes, seconds_per_sub_t)
            braked_x, braked_y, braked_h, braked_v = (
                kinematics.bicycle_forward(
                    ego[kinematics.X], ego[kinematics.Y],
                    ego[kinematics.HEADING], ego[kinematics.SPEED],
                    0.0, braking_acc, seconds_per_sub_t))
            braked_ego = (ego
                          .at[kinematics.X].set(braked_x)
                          .at[kinematics.Y].set(braked_y)
                          .at[kinematics.HEADING].set(braked_h)
                          .at[kinematics.SPEED].set(braked_v))
            ego = jnp.where(crashed, braked_ego, servo_ego)
            npcs = idm.npc_integrate_all(
                npcs, npc_acc, npc_steer, npc_lane,
                seconds_per_sub_t)
        else:
            servo_ego = kinematics.servo_sub_step(
                ego, highway_lanes, seconds_per_sub_t)
            braked_x, braked_y, braked_h, braked_v = (
                kinematics.bicycle_forward(
                    ego[kinematics.X], ego[kinematics.Y],
                    ego[kinematics.HEADING], ego[kinematics.SPEED],
                    0.0, braking_acc, seconds_per_sub_t))
            braked_ego = (ego
                          .at[kinematics.X].set(braked_x)
                          .at[kinematics.Y].set(braked_y)
                          .at[kinematics.HEADING].set(braked_h)
                          .at[kinematics.SPEED].set(braked_v))
            ego = jnp.where(crashed, braked_ego, servo_ego)
            npcs = idm.npc_sub_step_all(
                npcs, ego, highway_lanes,
                seconds_per_sub_t,
                enable_lane_change=enable_lane_change)

        # Collision detection (same as sub_step_with_collision above)
        per_npc_colliding, per_npc_will_collide, per_npc_translations = (
            kinematics.check_vehicle_collisions_all(
                ego, npcs, dt=seconds_per_sub_t))
        per_npc_impulse_mask = per_npc_colliding | per_npc_will_collide

        ego_impulse = jnp.sum(
            jnp.where(per_npc_impulse_mask[:, None],
                      per_npc_translations / 2, 0.0),
            axis=0)
        ego = (ego
               .at[kinematics.X].add(ego_impulse[0])
               .at[kinematics.Y].add(ego_impulse[1]))

        npc_ego_impulse_by_npc = jnp.where(
            per_npc_impulse_mask[:, None],
            -per_npc_translations / 2, 0.0)
        npcs = (npcs
                .at[:, kinematics.X].add(npc_ego_impulse_by_npc[:, 0])
                .at[:, kinematics.Y].add(npc_ego_impulse_by_npc[:, 1]))

        if crash_on_predicted:
            npc_collided_by_npc = per_npc_impulse_mask
        else:
            npc_collided_by_npc = per_npc_colliding

        if npc_npc_collisions:
            npc_npc_impulse_by_npc, npc_npc_colliding_by_npc = (
                kinematics.check_npc_npc_collisions_all(
                    npcs, dt=seconds_per_sub_t))
            npcs = (npcs
                    .at[:, kinematics.X].add(npc_npc_impulse_by_npc[:, 0])
                    .at[:, kinematics.Y].add(npc_npc_impulse_by_npc[:, 1]))
            npc_collided_by_npc = (npc_collided_by_npc
                                   | npc_npc_colliding_by_npc)

        if crash_on_predicted:
            collision = jnp.any(per_npc_impulse_mask)
        else:
            collision = jnp.any(per_npc_colliding)
        crashed = crashed | collision

        if npc_crash_braking:
            npc_already_crashed = npcs[:, kinematics.CRASHED] > 0.5
            npc_now_crashed = npc_already_crashed | npc_collided_by_npc
            npcs = npcs.at[:, kinematics.CRASHED].set(
                npc_now_crashed.astype(jnp.float32))

        new_carry = (ego, npcs, crashed)
        outputs = (ego, npcs, crashed)
        return new_carry, outputs

    init_carry = (ego_state, npc_states, ego_crashed)
    (new_ego, new_npcs, new_crashed), (sub_ego_by_sub_t, sub_npcs_by_sub_t,
                                        sub_crashed_by_sub_t) = (
        jax.lax.scan(sub_step_fn, init_carry, None, length=n_sub_ts))

    return (new_ego, new_npcs, new_crashed,
            sub_ego_by_sub_t, sub_npcs_by_sub_t, sub_crashed_by_sub_t)
