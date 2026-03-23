from __future__ import annotations

import flax.struct
import jax

from .populating import Population
from .rollout import Rollout, collect_rollout


@flax.struct.dataclass
class Genesis:
    population: Population
    rollout: Rollout
    rollout_seed: jax.Array
    evaluation_seed: jax.Array
    next_seed: jax.Array
    replay_buffer_factor: int = flax.struct.field(pytree_node=False)

    @staticmethod
    def create(population: Population, seed: jax.Array,
               env, params, n_es: int, n_ts: int,
               replay_buffer_factor: int = 1) -> Genesis:
        rollout_seed, evaluation_seed, next_seed = (
            jax.random.split(seed, 3)
        )
        rollout, state_data = collect_rollout(
            env, params, population, rollout_seed,
            n_es * replay_buffer_factor, n_ts, epoch=0,
        )
        return Genesis(
            population=population,
            rollout=rollout,
            rollout_seed=rollout_seed,
            evaluation_seed=evaluation_seed,
            next_seed=next_seed,
            replay_buffer_factor=replay_buffer_factor,
        ), state_data

    def create_first_joint_ascent(self, env, params,
                                  n_es: int, n_ts: int):
        from .joint_ascent import JointAscent
        evaluation_seed, next_seed = jax.random.split(
            self.next_seed, 2,
        )
        return JointAscent.from_population(
            rollout_seed=self.rollout_seed,
            evaluation_seed=evaluation_seed,
            next_seed=next_seed,
            replay_buffer_factor=self.replay_buffer_factor,
            population=self.population,
            epoch=1,
            rollout=self.rollout,
            env=env, params=params,
            n_es=n_es, n_ts=n_ts,
        )
