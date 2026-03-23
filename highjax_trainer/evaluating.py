from __future__ import annotations

import datetime

import jax
from jax import numpy as jnp

from .populating import Population
from .rollout import Rollout


class Evaluator:
    def __init__(self, env, params) -> None:
        self.env = env
        self.params = params

    def evaluate(self, seed: jax.Array, population: Population,
                 rollout: Rollout, *, epoch: int | None = None
                 ) -> dict:
        evaluation = {}
        evaluation['datetime'] = (
            datetime.datetime.now().isoformat().replace('T', ' ')
        )
        if epoch is not None:
            evaluation['epoch'] = epoch

        vital_by_e_by_t = rollout.vital_by_e_by_t.astype(
            jnp.float32,
        )
        n_vital = vital_by_e_by_t.sum()

        for agent in population.agents:
            reward_by_e_by_t = (
                rollout.reward_by_agent_by_e_by_t[:, :, agent]
            )
            avg_reward = float(
                (reward_by_e_by_t * vital_by_e_by_t).sum() / n_vital,
            )
            evaluation['reward.coeval'] = avg_reward

        return evaluation
