from __future__ import annotations

import flax.struct
import jax
from jax import numpy as jnp

from .brain import Brain
from .config import AgentConfig


@flax.struct.dataclass
class Population:
    brains: list[Brain]

    def __getitem__(self, i: int) -> Brain:
        return self.brains[i]

    def __len__(self) -> int:
        return len(self.brains)

    @property
    def agents(self) -> range:
        return range(len(self))

    @property
    def n_agents(self) -> int:
        return len(self)

    @staticmethod
    def create_random(agent_configs: tuple[AgentConfig, ...],
                      seed: jax.Array, env, params,
                      n_es: int) -> Population:
        seeds = jax.random.split(seed, 2 * len(agent_configs))
        seed_pairs = [
            (seeds[2 * i], seeds[2 * i + 1])
            for i in range(len(agent_configs))
        ]
        brains = []
        for i_agent, agent_config in enumerate(agent_configs):
            actor_seed, critic_seed = seed_pairs[i_agent]
            brain = Brain.create_random(
                agent_config=agent_config,
                actor_seed=actor_seed,
                critic_seed=critic_seed,
                observation_shape=env.observation_shape,
                action_size=env.num_actions,
                n_es=n_es,
            )
            brains.append(brain)
        return Population(brains=brains)

    def infer(self, seed_by_agent: jax.Array,
              observation_by_agent_by_e: jax.Array
              ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        action_list = []
        chosen_p_list = []
        v_list = []
        p_by_action_list = []

        for i_agent in self.agents:
            action_by_e, chosen_p_by_e, v_by_e, p_by_action_by_e = (
                self.brains[i_agent].infer(
                    seed_by_agent[i_agent],
                    observation_by_agent_by_e[:, i_agent],
                )
            )
            action_list.append(action_by_e)
            chosen_p_list.append(chosen_p_by_e)
            v_list.append(v_by_e)
            p_by_action_list.append(p_by_action_by_e)

        # Stack along agent axis (axis=1)
        action_by_agent_by_e = jnp.stack(action_list, axis=1)
        chosen_p_by_agent_by_e = jnp.stack(chosen_p_list, axis=1)
        v_by_agent_by_e = jnp.stack(v_list, axis=1)
        p_by_action_by_agent_by_e = jnp.stack(
            p_by_action_list, axis=1,
        )

        return (action_by_agent_by_e, chosen_p_by_agent_by_e,
                v_by_agent_by_e, p_by_action_by_agent_by_e)

    def duplicate(self, brain_by_index: dict[int, Brain] | None = None
                  ) -> Population:
        brains = list(self.brains)
        if brain_by_index is not None:
            for i_agent, brain in brain_by_index.items():
                brains[i_agent] = brain
        return Population(brains=brains)
