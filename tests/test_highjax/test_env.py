'''Tests for HighJax environment.'''
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

import highjax


class TestEnvInit:

    def test_default_values(self):
        env, params = highjax.make('highjax-v0')
        assert env.n_lanes == 4
        assert env.n_npcs == 50
        assert env.num_actions == 5

    def test_custom_values(self):
        env, params = highjax.make('highjax-v0', n_lanes=6, n_npcs=8)
        assert env.n_lanes == 6
        assert env.n_npcs == 8

    def test_observation_shape(self):
        env, _ = highjax.make('highjax-v0')
        assert env.observation_shape == (5, 5)

    def test_invalid_env_id(self):
        with pytest.raises(ValueError):
            highjax.make('invalid-v0')


class TestEnvReset:

    def setup_method(self):
        self.env, self.params = highjax.make('highjax-v0', n_npcs=5)
        self.key = jax.random.PRNGKey(42)

    def test_reset_returns_obs_and_state(self):
        obs, state = self.env.reset(self.key, self.params)
        assert obs.shape == self.env.observation_shape
        assert isinstance(state, highjax.EnvState)

    def test_time_starts_at_zero(self):
        _, state = self.env.reset(self.key, self.params)
        assert float(state.time) == 0.0

    def test_not_crashed_initially(self):
        _, state = self.env.reset(self.key, self.params)
        assert not bool(state.crashed)

    def test_different_seeds_different_states(self):
        obs1, _ = self.env.reset(jax.random.PRNGKey(1), self.params)
        obs2, _ = self.env.reset(jax.random.PRNGKey(2), self.params)
        assert not jnp.allclose(obs1, obs2)


class TestEnvStep:

    def setup_method(self):
        self.env, self.params = highjax.make('highjax-v0', n_npcs=5)
        self.key = jax.random.PRNGKey(42)
        _, self.state = self.env.reset(self.key, self.params)

    def test_step_returns_correct_types(self):
        obs, state, reward, done, info = self.env.step_env(
            self.key, self.state, 1, self.params,
        )
        assert obs.shape == self.env.observation_shape
        assert isinstance(state, highjax.EnvState)
        assert reward.shape == ()
        assert done.shape == ()

    def test_time_advances(self):
        _, state, _, _, _ = self.env.step_env(
            self.key, self.state, 1, self.params,
        )
        assert float(state.time) == pytest.approx(self.params.seconds_per_t)

    def test_ego_moves_forward(self):
        old_x = float(self.state.ego_state[0])
        _, state, _, _, _ = self.env.step_env(
            self.key, self.state, 1, self.params,
        )
        new_x = float(state.ego_state[0])
        assert new_x > old_x

    def test_reward_positive_when_driving(self):
        _, _, reward, _, _ = self.env.step_env(
            self.key, self.state, 1, self.params,
        )
        assert float(reward) > 0

    def test_multiple_steps_no_nan(self):
        state = self.state
        key = self.key
        for i in range(20):
            key, subkey = jax.random.split(key)
            _, state, reward, done, _ = self.env.step_env(
                subkey, state, 1, self.params,
            )
            assert not jnp.any(jnp.isnan(state.ego_state)), f'NaN at step {i}'

    def test_all_actions_valid(self):
        for action in range(5):
            obs, state, reward, done, _ = self.env.step_env(
                self.key, self.state, action, self.params,
            )
            assert obs.shape == self.env.observation_shape


class TestEnvAutoReset:

    def setup_method(self):
        self.env, self.params = highjax.make('highjax-v0', n_npcs=5)
        self.key = jax.random.PRNGKey(42)

    def test_step_with_auto_reset(self):
        _, state = self.env.reset(self.key, self.params)
        obs, state, reward, done, info = self.env.step(
            self.key, state, 1, self.params,
        )
        assert obs.shape == self.env.observation_shape


class TestEnvObservation:

    def setup_method(self):
        self.env, self.params = highjax.make('highjax-v0', n_npcs=5)
        self.key = jax.random.PRNGKey(42)

    def test_observation_shape(self):
        obs, _ = self.env.reset(self.key, self.params)
        assert obs.shape == (5, 5)  # n_observed_vehicles, n_features

    def test_ego_is_first_vehicle(self):
        obs, _ = self.env.reset(self.key, self.params)
        # First vehicle is ego: presence=1
        assert float(obs[0, 0]) == 1.0

    def test_observation_bounded(self):
        obs, _ = self.env.reset(self.key, self.params)
        assert jnp.all(obs >= -2.0)
        assert jnp.all(obs <= 2.0)


class TestEnvTermination:

    def setup_method(self):
        self.env, self.params = highjax.make('highjax-v0', n_npcs=5)

    def test_done_on_timeout(self):
        key = jax.random.PRNGKey(0)
        obs, state = self.env.reset(key, self.params)
        # Run many steps until timeout
        for _ in range(50):
            key, subkey = jax.random.split(key)
            obs, state, reward, done, _ = self.env.step_env(
                subkey, state, 1, self.params,
            )
            if bool(done):
                return
        # Should have terminated by timeout (40s / 1.0s per step = 40 steps)
        assert bool(done), 'Should have terminated by timeout'
