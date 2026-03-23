from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import highjax
from highjax_trainer.config import AgentConfig
from highjax_trainer.brain import Brain
from highjax_trainer.populating import Population
from highjax_trainer.rollout import Rollout, collect_rollout
from highjax_trainer.ascending import Ascender
from highjax_trainer.training import train


class TestBrain:

    def test_create_random(self):
        env, _ = highjax.make('highjax-v0', n_npcs=5)
        config = AgentConfig()
        actor_seed, critic_seed = jax.random.split(jax.random.PRNGKey(0))
        brain = Brain.create_random(
            config, actor_seed, critic_seed,
            env.observation_shape, env.num_actions, n_es=4,
        )
        assert brain.actor_lobe is not None
        assert brain.critic_lobe is not None

    def test_infer(self):
        env, params = highjax.make('highjax-v0', n_npcs=5)
        config = AgentConfig()
        actor_seed, critic_seed = jax.random.split(jax.random.PRNGKey(0))
        brain = Brain.create_random(
            config, actor_seed, critic_seed,
            env.observation_shape, env.num_actions, n_es=4,
        )
        keys = jax.random.split(jax.random.PRNGKey(1), 4)
        obs_batch, _ = jax.vmap(lambda k: env.reset(k, params))(keys)

        action, chosen_p, v, p_by_action = brain.infer(
            jax.random.PRNGKey(2), obs_batch,
        )
        assert action.shape == (4, 1)
        assert chosen_p.shape == (4,)
        assert v.shape == (4,)
        assert p_by_action.shape == (4, 5)


class TestPopulation:

    def test_create_random(self):
        env, params = highjax.make('highjax-v0', n_npcs=5)
        config = AgentConfig()
        population = Population.create_random(
            (config,), jax.random.PRNGKey(0), env, params, n_es=4,
        )
        assert len(population) == 1
        assert population.n_agents == 1
        assert population[0].actor_lobe is not None

    def test_infer(self):
        env, params = highjax.make('highjax-v0', n_npcs=5)
        config = AgentConfig()
        population = Population.create_random(
            (config,), jax.random.PRNGKey(0), env, params, n_es=4,
        )
        keys = jax.random.split(jax.random.PRNGKey(1), 4)
        obs_batch, _ = jax.vmap(lambda k: env.reset(k, params))(keys)
        # Expand to multi-agent: (4, *obs) -> (4, 1, *obs)
        obs_by_agent_by_e = jnp.expand_dims(obs_batch, axis=1)

        seed_by_agent = jax.random.split(
            jax.random.PRNGKey(2), 1,
        )
        (action, chosen_p, v, p_by_action) = population.infer(
            seed_by_agent, obs_by_agent_by_e,
        )
        assert action.shape == (4, 1, 1)
        assert chosen_p.shape == (4, 1)
        assert v.shape == (4, 1)
        assert p_by_action.shape == (4, 1, 5)

    def test_duplicate(self):
        env, params = highjax.make('highjax-v0', n_npcs=5)
        config = AgentConfig()
        population = Population.create_random(
            (config,), jax.random.PRNGKey(0), env, params, n_es=4,
        )
        dup = population.duplicate()
        assert len(dup) == 1


class TestRollout:

    def setup_method(self):
        self.env, self.params = highjax.make('highjax-v0', n_npcs=5)
        self.config = AgentConfig()
        self.population = Population.create_random(
            (self.config,), jax.random.PRNGKey(0),
            self.env, self.params, n_es=4,
        )

    def test_collect_rollout_shapes(self):
        rollout, state_data = collect_rollout(
            self.env, self.params, self.population,
            jax.random.PRNGKey(1), n_es=4, n_ts=10,
        )
        # (n_ts, n_es, n_agents, ...)
        assert rollout.observation_by_agent_by_e_by_t.shape[:3] == (
            10, 4, 1,
        )
        assert rollout.reward_by_agent_by_e_by_t.shape == (10, 4, 1)
        assert rollout.done_by_e_by_t.shape == (10, 4)
        assert rollout.action_by_agent_by_e_by_t.shape == (
            10, 4, 1, 1,
        )
        assert rollout.v_by_agent_by_e_by_t.shape == (10, 4, 1)

    def test_collect_rollout_state_data(self):
        rollout, state_data = collect_rollout(
            self.env, self.params, self.population,
            jax.random.PRNGKey(1), n_es=4, n_ts=10,
        )
        assert state_data['ego_state_by_e_by_t'].shape == (10, 4, 10)
        assert state_data['npc_states_by_e_by_t'].shape == (
            10, 4, 5, 10,
        )

    def test_vital_mask_shape(self):
        rollout, _ = collect_rollout(
            self.env, self.params, self.population,
            jax.random.PRNGKey(1), n_es=4, n_ts=10,
        )
        vital = rollout.vital_by_e_by_t
        assert vital.shape == (10, 4)
        assert vital.dtype == jnp.bool_
        assert jnp.all(vital[0, :])

    def test_vital_mask_monotonicity(self):
        rollout, _ = collect_rollout(
            self.env, self.params, self.population,
            jax.random.PRNGKey(1), n_es=4, n_ts=10,
        )
        vital = rollout.vital_by_e_by_t
        for e in range(4):
            col = vital[:, e]
            first_false = None
            for t in range(10):
                if not col[t]:
                    first_false = t
                    break
            if first_false is not None:
                assert not jnp.any(col[first_false:])


class TestAscender:

    def setup_method(self):
        self.env, self.params = highjax.make('highjax-v0', n_npcs=5)
        self.config = AgentConfig(n_mts_per_minibatch=None)
        self.population = Population.create_random(
            (self.config,), jax.random.PRNGKey(0),
            self.env, self.params, n_es=4,
        )
        self.rollout, _ = collect_rollout(
            self.env, self.params, self.population,
            jax.random.PRNGKey(1), n_es=4, n_ts=10,
        )

    def test_ascender_produces_next_brain(self):
        ascender = Ascender(
            rollout=self.rollout, population=self.population,
            agent=0, agent_config=self.config,
        )
        next_brain = ascender.next_brain
        assert next_brain.actor_lobe is not None
        assert next_brain.critic_lobe is not None

    def test_ascent_changes_theta(self):
        ascender = Ascender(
            rollout=self.rollout, population=self.population,
            agent=0, agent_config=self.config,
        )
        old_actor_theta = self.population[0].actor_lobe.theta
        new_actor_theta = ascender.next_actor_lobe.theta
        leaves_old = jax.tree.leaves(old_actor_theta)
        leaves_new = jax.tree.leaves(new_actor_theta)
        any_changed = any(
            not jnp.allclose(o, n)
            for o, n in zip(leaves_old, leaves_new)
        )
        assert any_changed, 'Actor theta should change after ascent'

    def test_v_loss_is_scalar(self):
        ascender = Ascender(
            rollout=self.rollout, population=self.population,
            agent=0, agent_config=self.config,
        )
        v_loss = ascender.v_loss
        assert v_loss.shape == ()
        assert not jnp.isnan(v_loss)

    def test_kld_is_nonnegative(self):
        ascender = Ascender(
            rollout=self.rollout, population=self.population,
            agent=0, agent_config=self.config,
        )
        kld = ascender.kld
        assert float(kld) >= 0.0

    def test_frozen_actor(self):
        ascender = Ascender(
            rollout=self.rollout, population=self.population,
            agent=0, agent_config=self.config, frozen_actor=True,
        )
        old_theta = self.population[0].actor_lobe.theta
        new_theta = ascender.next_actor_lobe.theta
        leaves_old = jax.tree.leaves(old_theta)
        leaves_new = jax.tree.leaves(new_theta)
        for o, n in zip(leaves_old, leaves_new):
            assert jnp.allclose(o, n)

    def test_frozen_critic(self):
        ascender = Ascender(
            rollout=self.rollout, population=self.population,
            agent=0, agent_config=self.config, frozen_critic=True,
        )
        old_theta = self.population[0].critic_lobe.theta
        new_theta = ascender.next_critic_lobe.theta
        leaves_old = jax.tree.leaves(old_theta)
        leaves_new = jax.tree.leaves(new_theta)
        for o, n in zip(leaves_old, leaves_new):
            assert jnp.allclose(o, n)

    def test_vanilla_objective_is_scalar(self):
        ascender = Ascender(
            rollout=self.rollout, population=self.population,
            agent=0, agent_config=self.config,
        )
        obj = ascender.vanilla_objective
        assert obj.shape == ()
        assert not jnp.isnan(obj)


class TestTraining:

    def test_train_smoke(self):
        env, params = highjax.make('highjax-v0', n_npcs=5)
        config = AgentConfig(n_mts_per_minibatch=None)
        population = train(
            env, params, config,
            n_epochs=2, n_es=4, n_ts=10, verbose=False, trek=False,
        )
        assert len(population) == 1
        assert population[0].actor_lobe is not None
        assert population[0].critic_lobe is not None

    def test_train_zero_epochs(self):
        env, params = highjax.make('highjax-v0', n_npcs=5)
        config = AgentConfig(n_mts_per_minibatch=None)
        population = train(
            env, params, config,
            n_epochs=0, n_es=4, n_ts=10, verbose=False, trek=False,
        )
        assert population is not None
        assert len(population) == 1
