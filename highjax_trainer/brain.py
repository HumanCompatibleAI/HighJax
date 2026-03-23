'''Brain: actor + critic lobes for a single agent.'''
from __future__ import annotations

from typing import Optional, Self

import flax.struct
import jax
from jax import numpy as jnp

from .config import AgentConfig
from .lobe import ActorLobe, CriticLobe
from .jax_utils import categorical_choice
from . import attention_estimating


@flax.struct.dataclass
class Brain:
    actor_lobe: ActorLobe
    critic_lobe: CriticLobe
    agent_config: AgentConfig = flax.struct.field(pytree_node=False)

    def duplicate(self, *,
                  actor_theta: Optional[dict] = None,
                  critic_theta: Optional[dict] = None) -> Self:
        return type(self)(
            agent_config=self.agent_config,
            actor_lobe=self.actor_lobe.duplicate(theta=actor_theta)
                        if actor_theta is not None else self.actor_lobe,
            critic_lobe=self.critic_lobe.duplicate(theta=critic_theta)
                        if critic_theta is not None else self.critic_lobe,
        )

    @jax.jit
    def infer(self, seed: jax.Array, observation_by_e: jax.Array
              ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        '''Infer actions, values, and probabilities for a batch of observations.

        Returns:
            (action_by_e, chosen_action_p_by_e, v_by_e, p_by_action_by_e)
        '''
        p_by_action_by_e = self.actor_lobe.get_p_by_action_by_i_observation(observation_by_e)
        action_by_e = categorical_choice(seed, p_by_action_by_e)  # (n_es, 1)
        n_es = p_by_action_by_e.shape[0]
        chosen_action_p_by_e = p_by_action_by_e[jnp.arange(n_es), action_by_e[:, 0]]
        v_by_e = self.critic_lobe.get_v_by_i_observation(observation_by_e)
        return action_by_e, chosen_action_p_by_e, v_by_e, p_by_action_by_e

    def get_v_by_i_observation(self, observations: jax.Array) -> jax.Array:
        return self.critic_lobe.get_v_by_i_observation(observations)

    def get_v_by_e_by_t(self, observation_by_e_by_t: jax.Array) -> jax.Array:
        return jax.vmap(self.get_v_by_i_observation)(observation_by_e_by_t)

    def get_p_by_action_by_i_observation(self, observations: jax.Array) -> jax.Array:
        return self.actor_lobe.get_p_by_action_by_i_observation(observations)

    def get_attention_weights(self, observation_by_e_by_t: jax.Array
                              ) -> jax.Array | None:
        estimator = self.actor_lobe.estimator_type(
            **self.actor_lobe.estimator_kwargs,
        )
        if not hasattr(estimator, 'forward_with_attention'):
            return None

        def _get_attention(obs):
            _, attention = estimator.apply(
                self.actor_lobe.theta, obs,
                method=estimator.forward_with_attention,
            )
            return attention

        return jax.vmap(_get_attention)(observation_by_e_by_t)

    @staticmethod
    def create_random(agent_config: AgentConfig,
                      actor_seed: jax.Array, critic_seed: jax.Array,
                      observation_shape: tuple[int, ...], action_size: int,
                      n_es: int) -> Brain:
        '''Create a brain with random weights.'''

        actor_estimator = attention_estimating.AttentionActorEstimator(
            observation_shape=observation_shape,
            action_size=action_size,
        )
        critic_estimator = attention_estimating.AttentionCriticEstimator(
            observation_shape=observation_shape,
        )

        actor_theta = actor_estimator.get_initial_theta(actor_seed, (n_es,))
        critic_theta = critic_estimator.get_initial_theta(critic_seed, (n_es,))

        return Brain(
            agent_config=agent_config,
            actor_lobe=ActorLobe.create(
                agent_config=agent_config,
                estimator_type=attention_estimating.AttentionActorEstimator,
                estimator_kwargs=attention_estimating.AttentionActorEstimator.get_kwargs(
                    observation_shape, action_size),
                theta=actor_theta,
                optimizer_recipe=agent_config.actor_optimizer_recipe,
            ),
            critic_lobe=CriticLobe.create(
                agent_config=agent_config,
                estimator_type=attention_estimating.AttentionCriticEstimator,
                estimator_kwargs=attention_estimating.AttentionCriticEstimator.get_kwargs(
                    observation_shape),
                theta=critic_theta,
                optimizer_recipe=agent_config.critic_optimizer_recipe,
            ),
        )
