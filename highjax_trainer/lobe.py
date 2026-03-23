'''Lobe: neural network wrapper combining parameters, optimizer, and forward pass.

ActorLobe produces action logits/probabilities. CriticLobe produces value estimates.
Both extend Flax TrainState for gradient management.
'''
from __future__ import annotations

from typing import Optional, Any, Type, Self
import operator

import flax.struct
import flax.training.train_state
import jax
from jax import numpy as jnp
import optax

from .config import AgentConfig
from .optax_tools import OptimizerRecipe


@flax.struct.dataclass
class Lobe(flax.training.train_state.TrainState):
    agent_config: AgentConfig = flax.struct.field(pytree_node=False)
    estimator_type: Type = flax.struct.field(pytree_node=False)
    estimator_kwargs: dict[str, Any] = flax.struct.field(pytree_node=False)
    optimizer_recipe: OptimizerRecipe = flax.struct.field(pytree_node=False)

    @property
    def theta(self) -> dict:
        return self.params

    @property
    def optimizer(self) -> optax.GradientTransformation:
        return self.tx

    @classmethod
    def create(cls, *, agent_config: AgentConfig,
               estimator_type: Type, estimator_kwargs: dict[str, Any],
               theta: dict, optimizer_recipe: OptimizerRecipe,
               step: int = 0, opt_state: Optional[optax.OptState] = None,
               apply_fn=None, optimizer=None) -> Self:
        optimizer = optimizer_recipe.build_optimizer() if optimizer is None else optimizer
        if opt_state is None:
            opt_state = optimizer.init(
                theta['params'] if '_overwrite_with_gradient' in theta else theta
            )
        return cls(
            step=step,
            params=theta,
            agent_config=agent_config,
            estimator_type=estimator_type,
            estimator_kwargs=estimator_kwargs,
            optimizer_recipe=optimizer_recipe,
            apply_fn=(estimator_type(**estimator_kwargs).apply if apply_fn is None
                      else apply_fn),
            tx=optimizer,
            opt_state=opt_state,
        )

    def duplicate(self, *,
                  agent_config: Optional[AgentConfig] = None,
                  theta: Optional[dict] = None,
                  optimizer_recipe: Optional[OptimizerRecipe] = None,
                  reset_opt_state: bool = False) -> Self:
        agent_config = self.agent_config if agent_config is None else agent_config
        optimizer_recipe_ = self.optimizer_recipe if optimizer_recipe is None else optimizer_recipe
        optimizer = self.optimizer if optimizer_recipe is None \
                                   else optimizer_recipe_.build_optimizer()
        theta_ = self.theta if theta is None else theta

        if reset_opt_state:
            opt_state = optimizer.init(
                theta_['params'] if '_overwrite_with_gradient' in theta_ else theta_
            )
        elif optimizer_recipe is None:
            opt_state = self.opt_state
        else:
            opt_state = None

        return type(self).create(
            agent_config=agent_config,
            estimator_type=self.estimator_type,
            estimator_kwargs=self.estimator_kwargs,
            optimizer_recipe=optimizer_recipe_,
            optimizer=optimizer,
            theta=theta_,
            step=self.step,
            opt_state=opt_state,
            apply_fn=self.apply_fn,
        )

    def apply_gradients(self, grads: dict, **kwargs) -> Self:
        raise RuntimeError("Use apply_gradient() instead (gradient ascent).")

    def apply_gradient(self, gradient: dict) -> Self:
        return flax.training.train_state.TrainState.apply_gradients(
            self,
            grads=jax.tree.map(operator.neg, gradient),  # Gradient ascent
        )


@flax.struct.dataclass
class ActorLobe(Lobe):

    @jax.jit
    def get_logit_by_deed_by_i_observation_by_lunge(self, observations: jax.Array
                                                    ) -> tuple[jax.Array, ...]:
        return self.apply_fn(self.theta, observations)

    @jax.jit
    def get_p_by_deed_by_i_observation_by_lunge(self, observations: jax.Array
                                                ) -> tuple[jax.Array, ...]:
        logit_by_deed_by_i_observation_by_lunge = (
            self.get_logit_by_deed_by_i_observation_by_lunge(observations)
        )

        if self.agent_config.actor_logit_clip is not None:
            clip = self.agent_config.actor_logit_clip
            logit_by_deed_by_i_observation_by_lunge = tuple(
                jnp.clip(logits, -clip, clip)
                for logits in logit_by_deed_by_i_observation_by_lunge
            )

        raw_p = tuple(
            jax.nn.softmax(logits, axis=-1)
            for logits in logit_by_deed_by_i_observation_by_lunge
        )

        if self.agent_config.noise:
            noisy_p = []
            for p_by_deed in raw_p:
                n_deeds = p_by_deed.shape[-1]
                noisy = (1 - self.agent_config.noise) * p_by_deed + \
                        self.agent_config.noise / n_deeds
                noisy_p.append(noisy)
            return tuple(noisy_p)
        return raw_p

    @jax.jit
    def get_p_by_action_by_i_observation(self, observations: jax.Array) -> jax.Array:
        '''Get action probabilities. For single-lunge (standard discrete), just returns p.'''
        p_by_lunge = self.get_p_by_deed_by_i_observation_by_lunge(observations)
        # Single action dimension: just return the first (and only) lunge
        return p_by_lunge[0]

    @jax.jit
    def get_p_by_action_by_e_by_t(self, observation_by_e_by_t: jax.Array) -> jax.Array:
        return jax.vmap(self.get_p_by_action_by_i_observation)(observation_by_e_by_t)

    @jax.jit
    def calculate_kld_by_e_by_t(self, old_actor_lobe: ActorLobe,
                                observation_by_e_by_t: jax.Array) -> jax.Array:
        old_p_by_lunge = old_actor_lobe.get_p_by_deed_by_i_observation_by_lunge(
            observation_by_e_by_t
        )
        new_p_by_lunge = self.get_p_by_deed_by_i_observation_by_lunge(
            observation_by_e_by_t
        )
        kld_by_e_by_t = jnp.zeros(observation_by_e_by_t.shape[:2])
        for old_p, new_p in zip(old_p_by_lunge, new_p_by_lunge):
            kld_by_e_by_t = kld_by_e_by_t + (
                old_p * (jnp.log(jnp.maximum(old_p, 1e-8)) -
                         jnp.log(jnp.maximum(new_p, 1e-8)))
            ).sum(axis=-1)
        return kld_by_e_by_t

    @jax.jit
    def calculate_kld(self, old_actor_lobe: ActorLobe,
                      observation_by_e_by_t: jax.Array,
                      vital_by_e_by_t: jax.Array) -> jax.Array:
        kld_by_e_by_t = self.calculate_kld_by_e_by_t(old_actor_lobe, observation_by_e_by_t)
        return (kld_by_e_by_t * vital_by_e_by_t).sum() / vital_by_e_by_t.sum()

    _AMPLIFICATION_CAP = 10000.0
    _BINARY_SEARCH_ITERATIONS = 20

    @jax.jit
    def adjust_for_target_kld(self, old_actor_lobe: ActorLobe,
                              observation_by_e_by_t: jax.Array,
                              vital_by_e_by_t: jax.Array) -> ActorLobe:
        max_kld = self.agent_config.effective_target_max_kld
        min_kld = self.agent_config.effective_target_min_kld

        if max_kld is None and min_kld is None:
            return self

        old_theta = old_actor_lobe.theta
        proposed_theta = self.theta

        if self.agent_config.kld_method == 'binary-search':
            alpha = self._kld_alpha_binary_search(
                old_actor_lobe, observation_by_e_by_t, vital_by_e_by_t,
                max_kld, min_kld,
            )
        else:
            alpha = self._kld_alpha_simple(
                old_actor_lobe, observation_by_e_by_t, vital_by_e_by_t,
                max_kld, min_kld,
            )

        final_theta = jax.tree.map(
            lambda o, n: o + alpha * (n - o), old_theta, proposed_theta,
        )
        return self.replace(params=final_theta)

    def _kld_at_alpha(self, alpha: jax.Array, old_actor_lobe: ActorLobe,
                      observation_by_e_by_t: jax.Array,
                      vital_by_e_by_t: jax.Array) -> jax.Array:
        interpolated_theta = jax.tree.map(
            lambda o, n: o + alpha * (n - o),
            old_actor_lobe.theta, self.theta,
        )
        return self.replace(params=interpolated_theta).calculate_kld(
            old_actor_lobe, observation_by_e_by_t, vital_by_e_by_t,
        )

    @jax.jit
    def _kld_alpha_simple(self, old_actor_lobe: ActorLobe,
                          observation_by_e_by_t: jax.Array,
                          vital_by_e_by_t: jax.Array,
                          max_kld, min_kld) -> jax.Array:
        proposed_kld = self._kld_at_alpha(
            jnp.array(1.0), old_actor_lobe, observation_by_e_by_t, vital_by_e_by_t,
        )
        alpha = jnp.array(1.0)
        if max_kld is not None:
            shrink_alpha = jnp.sqrt(max_kld / jnp.maximum(proposed_kld, 1e-10))
            alpha = jnp.where(proposed_kld > max_kld, shrink_alpha, alpha)
        if min_kld is not None:
            amplify_alpha = jnp.sqrt(min_kld / jnp.maximum(proposed_kld, 1e-10))
            alpha = jnp.where(proposed_kld < min_kld, amplify_alpha, alpha)
        return alpha

    @jax.jit
    def _kld_alpha_binary_search(self, old_actor_lobe: ActorLobe,
                                 observation_by_e_by_t: jax.Array,
                                 vital_by_e_by_t: jax.Array,
                                 max_kld, min_kld) -> jax.Array:
        proposed_kld = self._kld_at_alpha(
            jnp.array(1.0), old_actor_lobe, observation_by_e_by_t, vital_by_e_by_t,
        )

        def binary_search(target, lo, hi):
            def body(_, state):
                lo, hi = state
                mid = (lo + hi) / 2
                mid_kld = self._kld_at_alpha(
                    mid, old_actor_lobe, observation_by_e_by_t, vital_by_e_by_t,
                )
                return (jnp.where(mid_kld < target, mid, lo),
                        jnp.where(mid_kld >= target, mid, hi))
            lo, hi = jax.lax.fori_loop(
                0, self._BINARY_SEARCH_ITERATIONS, body,
                (jnp.array(lo, dtype=jnp.float32),
                 jnp.array(hi, dtype=jnp.float32)),
            )
            return (lo + hi) / 2

        if max_kld is not None and min_kld is not None:
            shrink_alpha = binary_search(max_kld, 0.0, 1.0)
            amplify_alpha = binary_search(min_kld, 1.0, self._AMPLIFICATION_CAP)
            alpha = jnp.where(
                proposed_kld > max_kld, shrink_alpha,
                jnp.where(proposed_kld < min_kld, amplify_alpha, 1.0),
            )
        elif max_kld is not None:
            shrink_alpha = binary_search(max_kld, 0.0, 1.0)
            alpha = jnp.where(proposed_kld > max_kld, shrink_alpha, 1.0)
        else:
            amplify_alpha = binary_search(min_kld, 1.0, self._AMPLIFICATION_CAP)
            alpha = jnp.where(proposed_kld < min_kld, amplify_alpha, 1.0)

        return alpha


@flax.struct.dataclass
class CriticLobe(Lobe):

    def get_v_by_i_observation(self, observations: jax.Array) -> jax.Array:
        return self.apply_fn(self.theta, observations).squeeze(-1)
