from __future__ import annotations

import functools

import flax.struct
import jax
from jax import numpy as jnp

from .ascending import Ascender
from .ascent import Ascent
from .populating import Population
from .rollout import Rollout, collect_rollout


@flax.struct.dataclass
class JointAscent:
    ascent_by_agent: tuple[Ascent, ...]
    rollout: Rollout
    rollout_seed: jax.Array
    evaluation_seed: jax.Array
    next_seed: jax.Array
    replay_buffer_factor: int = flax.struct.field(pytree_node=False)

    def __getitem__(self, agent: int) -> Ascent:
        return self.ascent_by_agent[agent]

    def __len__(self) -> int:
        return len(self.ascent_by_agent)

    @functools.cached_property
    def population(self) -> Population:
        return Population(
            brains=[ascent.brain for ascent in self.ascent_by_agent],
        )

    @functools.cached_property
    def next_population(self) -> Population:
        return Population(
            brains=[
                ascent.next_brain for ascent in self.ascent_by_agent
            ],
        )

    @staticmethod
    def from_population(rollout_seed, evaluation_seed, next_seed,
                        replay_buffer_factor, population, epoch,
                        rollout, env, params, n_es, n_ts):
        ascent_by_agent = []
        for i_agent in population.agents:
            agent_config = population[i_agent].agent_config
            frozen_actor = (
                agent_config.freeze_actor_from_epoch is not None
                and epoch >= agent_config.freeze_actor_from_epoch
            )
            frozen_critic = (
                agent_config.freeze_critic_from_epoch is not None
                and epoch >= agent_config.freeze_critic_from_epoch
            )
            if (agent_config.freeze_from_epoch is not None
                    and epoch >= agent_config.freeze_from_epoch):
                frozen_actor = frozen_critic = True

            ascender = Ascender(
                rollout=rollout, population=population,
                agent=i_agent, agent_config=agent_config,
                frozen_actor=frozen_actor,
                frozen_critic=frozen_critic,
            )
            ascent_by_agent.append(Ascent(
                brain=population[i_agent],
                next_brain=ascender.next_brain,
                v_loss=ascender.v_loss,
                kld=ascender.kld,
                vanilla_objective=ascender.vanilla_objective,
            ))

        return JointAscent(
            ascent_by_agent=tuple(ascent_by_agent),
            rollout=rollout,
            rollout_seed=rollout_seed,
            evaluation_seed=evaluation_seed,
            next_seed=next_seed,
            replay_buffer_factor=replay_buffer_factor,
        )

    def create_next(self, epoch, env, params, n_es, n_ts):
        rollout_seed, evaluation_seed, next_seed = (
            jax.random.split(self.next_seed, 3)
        )
        population = self.next_population
        new_rollout, state_data = collect_rollout(
            env, params, population, rollout_seed, n_es, n_ts,
            epoch=epoch,
        )
        extended_rollout = self.rollout.roll_and_extend(new_rollout)

        joint_ascent = JointAscent.from_population(
            rollout_seed=rollout_seed,
            evaluation_seed=evaluation_seed,
            next_seed=next_seed,
            replay_buffer_factor=self.replay_buffer_factor,
            population=population,
            epoch=epoch,
            rollout=extended_rollout,
            env=env, params=params,
            n_es=n_es, n_ts=n_ts,
        )
        return joint_ascent, state_data
