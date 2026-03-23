from __future__ import annotations

import functools

import flax.struct
import jax
from jax import numpy as jnp

from .brain import Brain
from .config import AgentConfig
from .populating import Population
from .stepping import StepCarry, StepReturn, step
from . import objectives


@flax.struct.dataclass
class Rollout:
    n_ts_per_e: int = flax.struct.field(pytree_node=False)
    n_agents: int = flax.struct.field(pytree_node=False)
    observation_shape: tuple[int, ...] = flax.struct.field(pytree_node=False)
    action_size: int = flax.struct.field(pytree_node=False)
    has_termination: bool = flax.struct.field(pytree_node=False)

    # Per-agent arrays: axis order is (n_ts, n_es, n_agents, ...)
    observation_by_agent_by_e_by_t: jax.Array
    epilogue_observation_by_agent_by_e: jax.Array
    reward_by_agent_by_e_by_t: jax.Array
    v_by_agent_by_e_by_t: jax.Array
    epilogue_v_by_agent_by_e: jax.Array
    action_by_agent_by_e_by_t: jax.Array
    chosen_action_p_by_agent_by_e_by_t: jax.Array
    p_by_action_by_agent_by_e_by_t: jax.Array

    # Agent-agnostic arrays: (n_ts, n_es)
    done_by_e_by_t: jax.Array
    crashed_by_e_by_t: jax.Array

    # Replay buffer: which epoch each episode belongs to
    epoch_by_e: jax.Array = None  # (n_es,) or None

    agent_configs: tuple[AgentConfig, ...] = flax.struct.field(
        pytree_node=False, default=None,
    )

    @property
    def n_es(self) -> int:
        return self.reward_by_agent_by_e_by_t.shape[1]

    def __getitem__(self, episode_indices: jax.Array) -> Rollout:
        return self.replace(
            observation_by_agent_by_e_by_t=(
                self.observation_by_agent_by_e_by_t[:, episode_indices]
            ),
            epilogue_observation_by_agent_by_e=(
                self.epilogue_observation_by_agent_by_e[episode_indices]
            ),
            reward_by_agent_by_e_by_t=(
                self.reward_by_agent_by_e_by_t[:, episode_indices]
            ),
            v_by_agent_by_e_by_t=(
                self.v_by_agent_by_e_by_t[:, episode_indices]
            ),
            epilogue_v_by_agent_by_e=(
                self.epilogue_v_by_agent_by_e[episode_indices]
            ),
            action_by_agent_by_e_by_t=(
                self.action_by_agent_by_e_by_t[:, episode_indices]
            ),
            chosen_action_p_by_agent_by_e_by_t=(
                self.chosen_action_p_by_agent_by_e_by_t[
                    :, episode_indices
                ]
            ),
            p_by_action_by_agent_by_e_by_t=(
                self.p_by_action_by_agent_by_e_by_t[:, episode_indices]
            ),
            done_by_e_by_t=self.done_by_e_by_t[:, episode_indices],
            crashed_by_e_by_t=(
                self.crashed_by_e_by_t[:, episode_indices]
            ),
            epoch_by_e=(
                self.epoch_by_e[episode_indices]
                if self.epoch_by_e is not None else None
            ),
        )

    @functools.cached_property
    def tail(self) -> Rollout:
        if self.epoch_by_e is None:
            return self
        max_epoch = self.epoch_by_e.max()
        is_last_by_e = (self.epoch_by_e == max_epoch)
        if is_last_by_e.all():
            return self
        return self[jnp.where(is_last_by_e)[0]]

    def roll_and_extend(self, new_rollout: Rollout) -> Rollout:
        max_es = self.n_es
        def _cat_t(old, new):
            return jnp.concatenate([old, new], axis=1)[:, -max_es:]
        def _cat_e(old, new):
            return jnp.concatenate([old, new], axis=0)[-max_es:]

        new_epoch_by_e = new_rollout.epoch_by_e
        old_epoch_by_e = self.epoch_by_e
        if old_epoch_by_e is not None and new_epoch_by_e is not None:
            combined_epoch_by_e = _cat_e(
                old_epoch_by_e, new_epoch_by_e,
            )
        else:
            combined_epoch_by_e = new_epoch_by_e

        return self.replace(
            observation_by_agent_by_e_by_t=_cat_t(
                self.observation_by_agent_by_e_by_t,
                new_rollout.observation_by_agent_by_e_by_t,
            ),
            epilogue_observation_by_agent_by_e=_cat_e(
                self.epilogue_observation_by_agent_by_e,
                new_rollout.epilogue_observation_by_agent_by_e,
            ),
            reward_by_agent_by_e_by_t=_cat_t(
                self.reward_by_agent_by_e_by_t,
                new_rollout.reward_by_agent_by_e_by_t,
            ),
            v_by_agent_by_e_by_t=_cat_t(
                self.v_by_agent_by_e_by_t,
                new_rollout.v_by_agent_by_e_by_t,
            ),
            epilogue_v_by_agent_by_e=_cat_e(
                self.epilogue_v_by_agent_by_e,
                new_rollout.epilogue_v_by_agent_by_e,
            ),
            action_by_agent_by_e_by_t=_cat_t(
                self.action_by_agent_by_e_by_t,
                new_rollout.action_by_agent_by_e_by_t,
            ),
            chosen_action_p_by_agent_by_e_by_t=_cat_t(
                self.chosen_action_p_by_agent_by_e_by_t,
                new_rollout.chosen_action_p_by_agent_by_e_by_t,
            ),
            p_by_action_by_agent_by_e_by_t=_cat_t(
                self.p_by_action_by_agent_by_e_by_t,
                new_rollout.p_by_action_by_agent_by_e_by_t,
            ),
            done_by_e_by_t=_cat_t(
                self.done_by_e_by_t,
                new_rollout.done_by_e_by_t,
            ),
            crashed_by_e_by_t=_cat_t(
                self.crashed_by_e_by_t,
                new_rollout.crashed_by_e_by_t,
            ),
            epoch_by_e=combined_epoch_by_e,
        )

    @functools.cached_property
    @jax.jit
    def vital_by_e_by_t(self) -> jax.Array:
        if not self.has_termination:
            return jnp.ones(
                (self.n_ts_per_e, self.n_es), dtype=jnp.bool_,
            )
        not_crashed_before = jnp.concatenate([
            jnp.ones((1, self.n_es), dtype=jnp.bool_),
            ~self.crashed_by_e_by_t[:-1],
        ], axis=0)
        return jnp.cumprod(not_crashed_before, axis=0).astype(jnp.bool_)

    @functools.cached_property
    @jax.jit
    def epilogue_vital_by_e(self) -> jax.Array:
        if not self.has_termination:
            return jnp.ones((self.n_es,), dtype=jnp.bool_)
        return ~self.crashed_by_e_by_t[-1]

    def _residual_by_e_by_t(self, agent: int,
                            agent_config: AgentConfig) -> jax.Array:
        discount = agent_config.discount
        v_by_e_by_t = self.v_by_agent_by_e_by_t[:, :, agent]
        epilogue_v_by_e = self.epilogue_v_by_agent_by_e[:, agent]
        reward_by_e_by_t = self.reward_by_agent_by_e_by_t[:, :, agent]

        next_v = jnp.roll(v_by_e_by_t, -1, axis=0).at[-1].set(
            epilogue_v_by_e,
        )
        next_vital = jnp.roll(
            self.vital_by_e_by_t, -1, axis=0,
        ).at[-1].set(self.epilogue_vital_by_e)
        next_v = next_v * next_vital

        residual = reward_by_e_by_t + discount * next_v - v_by_e_by_t
        return residual * self.vital_by_e_by_t

    @jax.jit(static_argnames=('agent', 'agent_config'))
    def calculate_advantage_by_e_by_t(self, agent: int,
                                      agent_config: AgentConfig
                                      ) -> jax.Array:
        return objectives.calculate_advantage_by_e_by_t(
            residual_by_e_by_t=self._residual_by_e_by_t(
                agent, agent_config,
            ),
            discount=agent_config.discount,
            gae_lambda=agent_config.vanilla_gae_lambda,
        )

    def calculate_tendency_by_e_by_t(self, agent: int) -> jax.Array:
        return jnp.log(
            self.chosen_action_p_by_agent_by_e_by_t[:, :, agent] + 1e-8,
        )

    def calculate_v_objective(self, agent: int,
                              agent_config: AgentConfig) -> jax.Array:
        return objectives.calculate_v_objective(
            residual_by_e_by_t=self._residual_by_e_by_t(
                agent, agent_config,
            ),
            discount=agent_config.discount,
            critic_lambda=agent_config.critic_lambda,
            vital_by_e_by_t=self.vital_by_e_by_t,
        )

    def calculate_return_by_e_by_t(self, agent: int,
                                   agent_config: AgentConfig
                                   ) -> jax.Array:
        reward_by_e_by_t = self.reward_by_agent_by_e_by_t[
            :, :, agent
        ]
        discount = agent_config.discount
        vital = self.vital_by_e_by_t
        masked_reward = reward_by_e_by_t * vital
        return objectives.calculate_return_by_e_by_t(
            masked_reward, discount,
        )

    def recompute_v(self, brain: Brain, agent: int) -> Rollout:
        obs_by_e_by_t = self.observation_by_agent_by_e_by_t[:, :, agent]
        epilogue_obs_by_e = self.epilogue_observation_by_agent_by_e[
            :, agent
        ]
        v_by_e_by_t = brain.get_v_by_e_by_t(obs_by_e_by_t)
        epilogue_v_by_e = brain.get_v_by_i_observation(epilogue_obs_by_e)

        new_v = self.v_by_agent_by_e_by_t.at[:, :, agent].set(
            v_by_e_by_t,
        )
        new_epilogue_v = self.epilogue_v_by_agent_by_e.at[:, agent].set(
            epilogue_v_by_e,
        )
        return self.replace(
            v_by_agent_by_e_by_t=new_v,
            epilogue_v_by_agent_by_e=new_epilogue_v,
        )


@functools.partial(jax.jit, static_argnums=(0, 4, 5))
def _collect_rollout_jit(env, params, population: Population,
                         seed: jax.Array, n_es: int, n_ts: int):
    n_agents = population.n_agents

    reset_seeds = jax.random.split(seed, n_es + 1)
    next_seed, env_seeds = reset_seeds[0], reset_seeds[1:]

    vmapped_reset = jax.vmap(lambda k: env.reset(k, params))
    first_obs_by_e, first_state_by_e = vmapped_reset(env_seeds)

    # Expand observation to multi-agent: (n_es, *obs) -> (n_es, 1, *obs)
    first_obs_by_agent_by_e = jnp.expand_dims(first_obs_by_e, axis=1)

    init_carry = StepCarry(
        seed=next_seed,
        state_by_e=first_state_by_e,
        observation_by_agent_by_e=first_obs_by_agent_by_e,
        reward_by_agent_by_e=jnp.zeros((n_es, n_agents)),
    )

    step_fn = functools.partial(
        step, population=population, env=env, params=params,
    )
    final_carry, step_return_stack = jax.lax.scan(
        step_fn, init_carry, None, length=n_ts,
    )

    # Epilogue: value of final state for each agent
    epilogue_obs_by_agent_by_e = final_carry.observation_by_agent_by_e
    epilogue_v_list = []
    for i_agent in population.agents:
        epilogue_v_by_e = population[i_agent].get_v_by_i_observation(
            epilogue_obs_by_agent_by_e[:, i_agent],
        )
        epilogue_v_list.append(epilogue_v_by_e)
    epilogue_v_by_agent_by_e = jnp.stack(epilogue_v_list, axis=1)

    return step_return_stack, epilogue_obs_by_agent_by_e, \
        epilogue_v_by_agent_by_e, first_obs_by_agent_by_e


def collect_rollout(env, params, population: Population,
                    seed: jax.Array, n_es: int, n_ts: int,
                    epoch: float = float('nan'),
                    ) -> tuple[Rollout, dict]:
    (step_return_stack, epilogue_obs_by_agent_by_e,
     epilogue_v_by_agent_by_e, first_obs_by_agent_by_e) = (
        _collect_rollout_jit(env, params, population, seed, n_es, n_ts)
    )

    # step_return_stack fields have time as axis 0 from scan stacking.
    # observation_by_agent_by_e_by_t: prepend first obs, drop last from
    # step_return_stack (which is the obs AFTER the last step).
    obs_by_agent_by_e_by_t = jnp.concatenate([
        first_obs_by_agent_by_e[None, ...],
        step_return_stack.next_observation_by_agent_by_e[:-1],
    ], axis=0)

    agent_configs = tuple(
        population[i].agent_config for i in population.agents
    )

    rollout = Rollout(
        n_ts_per_e=n_ts,
        n_agents=population.n_agents,
        observation_shape=env.observation_shape,
        action_size=env.num_actions,
        has_termination=True,
        agent_configs=agent_configs,
        observation_by_agent_by_e_by_t=obs_by_agent_by_e_by_t,
        epilogue_observation_by_agent_by_e=epilogue_obs_by_agent_by_e,
        reward_by_agent_by_e_by_t=(
            step_return_stack.reward_by_agent_by_e
        ),
        v_by_agent_by_e_by_t=step_return_stack.v_by_agent_by_e,
        epilogue_v_by_agent_by_e=epilogue_v_by_agent_by_e,
        action_by_agent_by_e_by_t=(
            step_return_stack.action_by_agent_by_e
        ),
        chosen_action_p_by_agent_by_e_by_t=(
            step_return_stack.chosen_action_p_by_agent_by_e
        ),
        p_by_action_by_agent_by_e_by_t=(
            step_return_stack.p_by_action_by_agent_by_e
        ),
        done_by_e_by_t=step_return_stack.done_by_e,
        crashed_by_e_by_t=step_return_stack.crashed_by_e,
        epoch_by_e=jnp.full(n_es, epoch),
    )
    state_data = {
        'ego_state_by_e_by_t': step_return_stack.ego_state_by_e,
        'npc_states_by_e_by_t': step_return_stack.npc_states_by_e,
        'crashed_by_e_by_t': step_return_stack.crashed_by_e,
    }
    return rollout, state_data
