'''Use HighJax with Stoix via the stoa gymnax adapter.

Stoix uses Hydra configs to wire environments to training systems. This example
shows how to create the wrapped environment programmatically. To integrate with
Stoix's config system, add a make function like this to stoix/utils/make_env.py
and register it in ENV_MAKERS.

Requires: pip install stoix
'''
from __future__ import annotations

import jax

# Requires: pip install stoix
from stoa.env_adapters.gymnax import GymnaxToStoa
from stoa.core_wrappers.wrapper import AddRNGKey
from stoa.core_wrappers.auto_reset import AutoResetWrapper
from stoa.utility_wrappers.extras_transforms import NoExtrasWrapper

import highjax


def make_highjax_env():
    # Create the raw gymnax-compatible environment
    env, env_params = highjax.make('highjax-v0')

    # Adapt to stoa's Environment interface
    env = GymnaxToStoa(env, env_params)
    env = NoExtrasWrapper(env)
    env = AddRNGKey(env)
    env = AutoResetWrapper(env, next_obs_in_extras=True)

    return env


def main():
    env = make_highjax_env()

    # Reset
    key = jax.random.PRNGKey(0)
    env_state, timestep = env.reset(key)
    print(f'Observation shape: {timestep.observation.shape}')
    print(f'Reward: {timestep.reward}')

    # Step
    action = 1  # IDLE
    env_state, timestep = env.step(env_state, action)
    print(f'After step — reward: {timestep.reward}')


if __name__ == '__main__':
    main()
