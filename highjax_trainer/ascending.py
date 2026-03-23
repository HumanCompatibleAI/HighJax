from __future__ import annotations

import functools

import flax.struct
import jax
from jax import numpy as jnp

from .brain import Brain
from .lobe import ActorLobe, CriticLobe
from .rollout import Rollout
from .config import AgentConfig
from .populating import Population
from . import defaults


# ── Regard: data bundle for gradient engines ─────────────────────────────────

@flax.struct.dataclass
class Regard:
    tendency_by_e_by_t: jax.Array
    p_by_deed_by_e_by_t_by_lunge: tuple[jax.Array, ...]
    logit_by_deed_by_e_by_t_by_lunge: tuple[jax.Array, ...]
    old_tendency_by_e_by_t: jax.Array
    advantage_by_e_by_t: jax.Array


# ── Gradient engines ─────────────────────────────────────────────────────────

def compute_vanilla_objective(regard: Regard,
                              agent_config: AgentConfig) -> jax.Array:
    ratio = jnp.exp(
        regard.tendency_by_e_by_t - regard.old_tendency_by_e_by_t,
    )
    clip_epsilon = agent_config.ppo_clip_epsilon
    if clip_epsilon is None:
        return ratio * regard.advantage_by_e_by_t
    clipped_ratio = jnp.clip(
        ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon,
    )
    return jnp.minimum(
        ratio * regard.advantage_by_e_by_t,
        clipped_ratio * regard.advantage_by_e_by_t,
    )


def compute_entropy_objective(regard: Regard,
                              agent_config: AgentConfig) -> jax.Array:
    if agent_config.logit_based_entropy:
        total = jnp.zeros_like(regard.tendency_by_e_by_t)
        for logits in regard.logit_by_deed_by_e_by_t_by_lunge:
            mean_logit = logits.mean(axis=-1, keepdims=True)
            variance = jnp.mean(
                (logits - mean_logit) ** 2, axis=-1,
            )
            hinge = jnp.maximum(
                0, variance - agent_config.logit_variance_threshold,
            ) ** 2
            total += agent_config.logit_variance_lambda * hinge
        return -agent_config.entropy_temperature * total
    else:
        total = jnp.zeros_like(regard.tendency_by_e_by_t)
        for p in regard.p_by_deed_by_e_by_t_by_lunge:
            total += -jnp.sum(p * jnp.log(p + 1e-8), axis=-1)
        return agent_config.entropy_temperature * total


def compute_composite_objective(regard: Regard,
                                agent_config: AgentConfig) -> jax.Array:
    result = compute_vanilla_objective(regard, agent_config)
    if agent_config.entropy_temperature != 0:
        result = result + compute_entropy_objective(regard, agent_config)
    return result


# ── Minibatcher ──────────────────────────────────────────────────────────────

@flax.struct.dataclass
class Minibatcher:
    advantage_by_mt: jax.Array
    old_tendency_by_mt: jax.Array
    vital_by_mt: jax.Array
    observation_by_mt: jax.Array
    action_by_mt: jax.Array

    def objective(self, actor_theta: dict, agent_config: AgentConfig,
                  actor_apply_fn) -> jax.Array:
        logit_by_lunge = actor_apply_fn(
            actor_theta, self.observation_by_mt,
        )

        if agent_config.actor_logit_clip is not None:
            clip = agent_config.actor_logit_clip
            logit_by_lunge = tuple(
                jnp.clip(l, -clip, clip) for l in logit_by_lunge
            )

        p_by_lunge = tuple(
            jax.nn.softmax(l, axis=-1) for l in logit_by_lunge
        )

        if agent_config.noise:
            p_by_lunge = tuple(
                (1 - agent_config.noise) * p
                + agent_config.noise / p.shape[-1]
                for p in p_by_lunge
            )

        tendency_by_mt = jnp.zeros(self.observation_by_mt.shape[0])
        for i_lunge, p_by_deed in enumerate(p_by_lunge):
            chosen_deed = self.action_by_mt[..., i_lunge]
            chosen_p = jnp.take_along_axis(
                p_by_deed, chosen_deed[..., None], axis=-1,
            )[..., 0]
            tendency_by_mt = tendency_by_mt + jnp.log(
                jnp.maximum(chosen_p, 1e-8),
            )

        regard = Regard(
            tendency_by_e_by_t=tendency_by_mt,
            p_by_deed_by_e_by_t_by_lunge=p_by_lunge,
            logit_by_deed_by_e_by_t_by_lunge=logit_by_lunge,
            old_tendency_by_e_by_t=self.old_tendency_by_mt,
            advantage_by_e_by_t=self.advantage_by_mt,
        )

        objective_by_mt = compute_composite_objective(
            regard, agent_config,
        )
        return (
            (objective_by_mt * self.vital_by_mt).sum()
            / self.vital_by_mt.sum()
        )

    def get_next_actor_lobe(self, actor_lobe: ActorLobe,
                            agent_config: AgentConfig) -> ActorLobe:
        gradient = jax.grad(self.objective)(
            actor_lobe.theta, agent_config, actor_lobe.apply_fn,
        )
        return actor_lobe.apply_gradient(gradient)


# ── Sweeper ──────────────────────────────────────────────────────────────────

@flax.struct.dataclass
class Sweeper:
    advantage_by_mt_by_mb: jax.Array
    old_tendency_by_mt_by_mb: jax.Array
    vital_by_mt_by_mb: jax.Array
    observation_by_mt_by_mb: jax.Array
    action_by_mt_by_mb: jax.Array

    @classmethod
    def create(cls, sweep_master: SweepMaster, seed: jax.Array,
               n_mts_per_minibatch: int) -> Sweeper:
        n_fts = sweep_master.advantage_by_ft.shape[0]
        n_minibatches = n_fts // n_mts_per_minibatch
        perm = jax.random.permutation(seed, n_fts)

        def shuffle_and_batch(x):
            return x[perm].reshape(
                n_minibatches, n_mts_per_minibatch, *x.shape[1:],
            )

        return cls(
            advantage_by_mt_by_mb=shuffle_and_batch(
                sweep_master.advantage_by_ft,
            ),
            old_tendency_by_mt_by_mb=shuffle_and_batch(
                sweep_master.old_tendency_by_ft,
            ),
            vital_by_mt_by_mb=shuffle_and_batch(
                sweep_master.vital_by_ft,
            ),
            observation_by_mt_by_mb=shuffle_and_batch(
                sweep_master.observation_by_ft,
            ),
            action_by_mt_by_mb=shuffle_and_batch(
                sweep_master.action_by_ft,
            ),
        )

    def get_next_actor_lobe(self, actor_lobe: ActorLobe,
                            agent_config: AgentConfig) -> ActorLobe:
        def body(actor_lobe, mb_data):
            mb = Minibatcher(
                advantage_by_mt=mb_data[0],
                old_tendency_by_mt=mb_data[1],
                vital_by_mt=mb_data[2],
                observation_by_mt=mb_data[3],
                action_by_mt=mb_data[4],
            )
            return mb.get_next_actor_lobe(
                actor_lobe, agent_config,
            ), None

        mb_stack = (
            self.advantage_by_mt_by_mb,
            self.old_tendency_by_mt_by_mb,
            self.vital_by_mt_by_mb,
            self.observation_by_mt_by_mb,
            self.action_by_mt_by_mb,
        )
        actor_lobe, _ = jax.lax.scan(body, actor_lobe, mb_stack)
        return actor_lobe


# ── SweepMaster ──────────────────────────────────────────────────────────────

def _pad_to_multiple(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


def _flatten_and_pad(array_by_e_by_t: jax.Array, n_fts: int,
                     pad_value: float | int | bool) -> jax.Array:
    flat = array_by_e_by_t.reshape(-1, *array_by_e_by_t.shape[2:])
    n_pad = n_fts - flat.shape[0]
    if n_pad == 0:
        return flat
    padding = [(0, n_pad)] + [(0, 0)] * (flat.ndim - 1)
    return jnp.pad(flat, padding, constant_values=pad_value)


@flax.struct.dataclass
class SweepMaster:
    agent_config: AgentConfig = flax.struct.field(pytree_node=False)
    n_mts_per_minibatch: int = flax.struct.field(pytree_node=False)

    advantage_by_ft: jax.Array
    old_tendency_by_ft: jax.Array
    vital_by_ft: jax.Array
    observation_by_ft: jax.Array
    action_by_ft: jax.Array

    @classmethod
    def create(cls, rollout: Rollout, agent: int,
               nz_advantage_by_e_by_t: jax.Array,
               agent_config: AgentConfig) -> SweepMaster:
        n_total = rollout.n_ts_per_e * rollout.n_es
        n_mts = agent_config.n_mts_per_minibatch
        if n_mts is None:
            n_mts = n_total
        n_fts = _pad_to_multiple(n_total, n_mts)

        return cls(
            agent_config=agent_config,
            n_mts_per_minibatch=n_mts,
            advantage_by_ft=_flatten_and_pad(
                nz_advantage_by_e_by_t, n_fts, 0.0,
            ),
            old_tendency_by_ft=_flatten_and_pad(
                rollout.calculate_tendency_by_e_by_t(agent),
                n_fts, 0.0,
            ),
            vital_by_ft=_flatten_and_pad(
                rollout.vital_by_e_by_t, n_fts, False,
            ),
            observation_by_ft=_flatten_and_pad(
                rollout.observation_by_agent_by_e_by_t[:, :, agent],
                n_fts, 0.0,
            ),
            action_by_ft=_flatten_and_pad(
                rollout.action_by_agent_by_e_by_t[:, :, agent],
                n_fts, 0,
            ),
        )

    def run(self, actor_lobe: ActorLobe) -> ActorLobe:
        seed_by_sweep = jax.random.split(
            jax.random.PRNGKey(defaults.DEFAULT_MINIBATCH_SEED_NUMBER),
            self.agent_config.n_sweeps_per_epoch,
        )

        def sweep_fn(actor_lobe, seed):
            sweeper = Sweeper.create(
                self, seed, self.n_mts_per_minibatch,
            )
            return sweeper.get_next_actor_lobe(
                actor_lobe, self.agent_config,
            ), None

        actor_lobe, _ = jax.lax.scan(
            sweep_fn, actor_lobe, seed_by_sweep,
        )
        return actor_lobe


# ── Ascender ─────────────────────────────────────────────────────────────────

@flax.struct.dataclass
class Ascender:
    rollout: Rollout
    population: Population
    agent: int = flax.struct.field(pytree_node=False)
    agent_config: AgentConfig = flax.struct.field(pytree_node=False)
    frozen_actor: bool = flax.struct.field(
        pytree_node=False, default=False,
    )
    frozen_critic: bool = flax.struct.field(
        pytree_node=False, default=False,
    )

    @property
    def brain(self) -> Brain:
        return self.population[self.agent]

    @functools.cached_property
    def next_critic_lobe(self) -> CriticLobe:
        if self.frozen_critic:
            return self.brain.critic_lobe
        return self._next_critic_lobe_and_v_loss[0]

    @functools.cached_property
    def v_loss(self) -> jax.Array:
        if self.frozen_critic:
            return jnp.array(float('nan'))
        return self._next_critic_lobe_and_v_loss[1]

    @functools.cached_property
    @jax.jit
    def _next_critic_lobe_and_v_loss(self
                                     ) -> tuple[CriticLobe, jax.Array]:
        def update_step(critic_lobe, _):
            v_objective, gradient = jax.value_and_grad(
                self._get_v_objective, allow_int=True,
            )(critic_lobe.theta)
            return critic_lobe.apply_gradient(gradient), v_objective

        critic_lobe, v_objectives = jax.lax.scan(
            update_step,
            self.brain.critic_lobe,
            None,
            length=self.agent_config.n_critic_iterations,
        )
        return critic_lobe, -v_objectives[-1]

    def _get_v_objective(self, critic_theta: dict) -> jax.Array:
        updated_brain = self.brain.duplicate(critic_theta=critic_theta)
        updated_rollout = self.rollout.recompute_v(
            updated_brain, self.agent,
        )
        return updated_rollout.calculate_v_objective(
            self.agent, self.agent_config,
        )

    @functools.cached_property
    @jax.jit
    def rollout_with_next_critic(self) -> Rollout:
        updated_brain = self.brain.duplicate(
            critic_theta=self.next_critic_lobe.theta,
        )
        return self.rollout.recompute_v(updated_brain, self.agent)

    @functools.cached_property
    @jax.jit
    def nz_advantage_by_e_by_t(self) -> jax.Array:
        raw = self.rollout_with_next_critic.calculate_advantage_by_e_by_t(
            self.agent, self.agent_config,
        )
        vital = self.rollout.vital_by_e_by_t
        n_vital = vital.sum()
        mean = (raw * vital).sum() / n_vital
        var = ((raw - mean) ** 2 * vital).sum() / n_vital
        return (raw - mean) / (jnp.sqrt(var) + 1e-8)

    @functools.cached_property
    def next_actor_lobe(self) -> ActorLobe:
        if self.frozen_actor:
            return self.brain.actor_lobe
        return self._compute_next_actor_lobe

    @functools.cached_property
    @jax.jit
    def _compute_next_actor_lobe(self) -> ActorLobe:
        sweep_master = SweepMaster.create(
            self.rollout, self.agent,
            self.nz_advantage_by_e_by_t, self.agent_config,
        )
        proposed = sweep_master.run(self.brain.actor_lobe)

        observation_by_e_by_t = (
            self.rollout.observation_by_agent_by_e_by_t[:, :, self.agent]
        )
        vital_by_e_by_t = self.rollout.vital_by_e_by_t
        return proposed.adjust_for_target_kld(
            self.brain.actor_lobe,
            observation_by_e_by_t, vital_by_e_by_t,
        )

    @functools.cached_property
    @jax.jit
    def vanilla_objective(self) -> jax.Array:
        obs_by_e_by_t = (
            self.rollout.observation_by_agent_by_e_by_t[:, :, self.agent]
        )
        action_by_e_by_t = (
            self.rollout.action_by_agent_by_e_by_t[:, :, self.agent]
        )
        vital = self.rollout.vital_by_e_by_t

        logit_by_lunge = self.next_actor_lobe.apply_fn(
            self.next_actor_lobe.theta, obs_by_e_by_t,
        )
        if self.agent_config.actor_logit_clip is not None:
            clip = self.agent_config.actor_logit_clip
            logit_by_lunge = tuple(
                jnp.clip(l, -clip, clip) for l in logit_by_lunge
            )
        p_by_lunge = tuple(
            jax.nn.softmax(l, axis=-1) for l in logit_by_lunge
        )
        if self.agent_config.noise:
            p_by_lunge = tuple(
                (1 - self.agent_config.noise) * p
                + self.agent_config.noise / p.shape[-1]
                for p in p_by_lunge
            )
        tendency_by_e_by_t = jnp.zeros(obs_by_e_by_t.shape[:2])
        for i_lunge, p_by_deed in enumerate(p_by_lunge):
            chosen = action_by_e_by_t[..., i_lunge]
            chosen_p = jnp.take_along_axis(
                p_by_deed, chosen[..., None], axis=-1,
            )[..., 0]
            tendency_by_e_by_t = tendency_by_e_by_t + jnp.log(
                jnp.maximum(chosen_p, 1e-8),
            )

        old_tendency = self.rollout.calculate_tendency_by_e_by_t(
            self.agent,
        )
        regard = Regard(
            tendency_by_e_by_t=tendency_by_e_by_t,
            p_by_deed_by_e_by_t_by_lunge=p_by_lunge,
            logit_by_deed_by_e_by_t_by_lunge=logit_by_lunge,
            old_tendency_by_e_by_t=old_tendency,
            advantage_by_e_by_t=self.nz_advantage_by_e_by_t,
        )
        obj_by_e_by_t = compute_vanilla_objective(
            regard, self.agent_config,
        )
        return (obj_by_e_by_t * vital).sum() / vital.sum()

    @functools.cached_property
    @jax.jit
    def kld(self) -> jax.Array:
        return self.next_actor_lobe.calculate_kld(
            self.brain.actor_lobe,
            self.rollout.observation_by_agent_by_e_by_t[
                :, :, self.agent
            ],
            self.rollout.vital_by_e_by_t,
        )

    @functools.cached_property
    def next_brain(self) -> Brain:
        return Brain(
            actor_lobe=self.next_actor_lobe,
            critic_lobe=self.next_critic_lobe,
            agent_config=self.agent_config,
        )
