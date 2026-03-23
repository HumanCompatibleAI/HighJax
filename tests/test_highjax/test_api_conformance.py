'''Tests for JAX API conformance: JIT, vmap, determinism.'''
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import highjax


class TestJitCompatibility:

    def setup_method(self):
        self.env, self.params = highjax.make('highjax-v0', n_npcs=5)
        self.key = jax.random.PRNGKey(42)

    def test_reset_jittable(self):
        jitted_reset = jax.jit(lambda k: self.env.reset(k, self.params))
        obs, state = jitted_reset(self.key)
        assert obs.shape == self.env.observation_shape

    def test_step_env_jittable(self):
        _, state = self.env.reset(self.key, self.params)
        jitted_step = jax.jit(
            lambda k, s, a: self.env.step_env(k, s, a, self.params)
        )
        obs, new_state, reward, done, info = jitted_step(self.key, state, 1)
        assert obs.shape == self.env.observation_shape

    def test_step_jittable(self):
        _, state = self.env.reset(self.key, self.params)
        jitted_step = jax.jit(
            lambda k, s, a: self.env.step(k, s, a, self.params)
        )
        obs, new_state, reward, done, info = jitted_step(self.key, state, 1)
        assert obs.shape == self.env.observation_shape


class TestVmapCompatibility:

    def setup_method(self):
        self.env, self.params = highjax.make('highjax-v0', n_npcs=5)
        self.n_envs = 8

    def test_vmapped_reset(self):
        keys = jax.random.split(jax.random.PRNGKey(0), self.n_envs)
        vmapped_reset = jax.vmap(lambda k: self.env.reset(k, self.params))
        obs_batch, state_batch = vmapped_reset(keys)
        assert obs_batch.shape == (self.n_envs, *self.env.observation_shape)

    def test_vmapped_step_env(self):
        keys = jax.random.split(jax.random.PRNGKey(0), self.n_envs * 2)
        reset_keys, step_keys = keys[:self.n_envs], keys[self.n_envs:]

        vmapped_reset = jax.vmap(lambda k: self.env.reset(k, self.params))
        _, state_batch = vmapped_reset(reset_keys)

        actions = jnp.ones(self.n_envs, dtype=jnp.int32)  # IDLE for all
        vmapped_step = jax.vmap(
            lambda k, s, a: self.env.step_env(k, s, a, self.params)
        )
        obs_batch, new_states, rewards, dones, infos = vmapped_step(
            step_keys, state_batch, actions,
        )
        assert obs_batch.shape == (self.n_envs, *self.env.observation_shape)
        assert rewards.shape == (self.n_envs,)
        assert dones.shape == (self.n_envs,)


class TestDeterminism:

    def test_same_seed_same_reset(self):
        env, params = highjax.make('highjax-v0', n_npcs=5)
        key = jax.random.PRNGKey(42)
        obs1, state1 = env.reset(key, params)
        obs2, state2 = env.reset(key, params)
        assert jnp.allclose(obs1, obs2)
        assert jnp.allclose(state1.ego_state, state2.ego_state)

    def test_same_seed_same_trajectory(self):
        env, params = highjax.make('highjax-v0', n_npcs=5)
        key = jax.random.PRNGKey(42)

        def run_trajectory(key):
            obs, state = env.reset(key, params)
            rewards = []
            for _ in range(10):
                key, subkey = jax.random.split(key)
                obs, state, reward, done, _ = env.step_env(
                    subkey, state, 1, params,
                )
                rewards.append(float(reward))
            return rewards

        r1 = run_trajectory(jax.random.PRNGKey(99))
        r2 = run_trajectory(jax.random.PRNGKey(99))
        assert r1 == r2

    def test_different_seeds_different_observations(self):
        env, params = highjax.make('highjax-v0', n_npcs=5)
        obs1, _ = env.reset(jax.random.PRNGKey(1), params)
        obs2, _ = env.reset(jax.random.PRNGKey(2), params)
        # Different seeds should produce different NPC positions
        assert not jnp.allclose(obs1, obs2)
