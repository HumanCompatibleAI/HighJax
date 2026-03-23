from __future__ import annotations

import functools

import flax.struct
import jax
from jax import numpy as jnp


@flax.struct.dataclass
class StepCarry:
    seed: jax.Array
    state_by_e: jax.Array
    observation_by_agent_by_e: jax.Array  # (n_es, n_agents, *obs_shape)
    reward_by_agent_by_e: jax.Array       # (n_es, n_agents)


@flax.struct.dataclass
class StepReturn:
    next_state_by_e: jax.Array
    next_observation_by_agent_by_e: jax.Array    # (n_es, n_agents, *obs_shape)
    reward_by_agent_by_e: jax.Array              # (n_es, n_agents)
    action_by_agent_by_e: jax.Array              # (n_es, n_agents, 1)
    v_by_agent_by_e: jax.Array                   # (n_es, n_agents)
    chosen_action_p_by_agent_by_e: jax.Array     # (n_es, n_agents)
    p_by_action_by_agent_by_e: jax.Array         # (n_es, n_agents, action_size)
    done_by_e: jax.Array                         # (n_es,)
    ego_state_by_e: jax.Array
    npc_states_by_e: jax.Array
    crashed_by_e: jax.Array


def step(step_carry: StepCarry, _, population, env, params):
    n_agents = step_carry.observation_by_agent_by_e.shape[1]
    n_es = step_carry.reward_by_agent_by_e.shape[0]

    all_seeds = jax.random.split(step_carry.seed, n_agents + n_es + 1)
    infer_seed_by_agent = all_seeds[:n_agents]
    step_seed_by_e = all_seeds[n_agents:n_agents + n_es]
    next_seed = all_seeds[-1]

    (action_by_agent_by_e, chosen_action_p_by_agent_by_e,
     v_by_agent_by_e, p_by_action_by_agent_by_e) = population.infer(
        infer_seed_by_agent, step_carry.observation_by_agent_by_e,
    )

    # Single-agent env: use agent 0's action for env.step_env
    action_by_e = action_by_agent_by_e[:, 0, 0]

    vmapped_step_env = jax.vmap(
        lambda k, s, a: env.step_env(k, s, a, params)
    )
    (next_observation_by_e, next_state_by_e,
     reward_by_e, done_by_e, _) = vmapped_step_env(
        step_seed_by_e, step_carry.state_by_e, action_by_e,
    )

    # Expand per-episode to per-agent
    next_obs_by_agent_by_e = jnp.expand_dims(
        next_observation_by_e, axis=1,
    )
    reward_by_agent_by_e = reward_by_e[:, None]

    new_carry = StepCarry(
        seed=next_seed,
        state_by_e=next_state_by_e,
        observation_by_agent_by_e=next_obs_by_agent_by_e,
        reward_by_agent_by_e=reward_by_agent_by_e,
    )

    step_return = StepReturn(
        next_state_by_e=next_state_by_e,
        next_observation_by_agent_by_e=next_obs_by_agent_by_e,
        reward_by_agent_by_e=reward_by_agent_by_e,
        action_by_agent_by_e=action_by_agent_by_e,
        v_by_agent_by_e=v_by_agent_by_e,
        chosen_action_p_by_agent_by_e=chosen_action_p_by_agent_by_e,
        p_by_action_by_agent_by_e=p_by_action_by_agent_by_e,
        done_by_e=done_by_e,
        ego_state_by_e=step_carry.state_by_e.ego_state,
        npc_states_by_e=step_carry.state_by_e.npc_states,
        crashed_by_e=next_state_by_e.crashed,
    )

    return new_carry, step_return
