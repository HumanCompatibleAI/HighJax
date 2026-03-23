'''Use HighJax with Rejax for JIT-compiled training.

Rejax accepts gymnax environment objects directly. The entire training loop
is JIT-compiled and can be vmapped across seeds for parallel runs.

Requires: pip install rejax

Note: Rejax checks for gymnax.environments.spaces.Discrete internally. If you
hit a space type error, wrap the env to return gymnax spaces instead of
gymnasium spaces from action_space() / observation_space().
'''
from __future__ import annotations

import jax

# Requires: pip install rejax
from rejax import PPO

import highjax


def main():
    env, env_params = highjax.make('highjax-v0')

    algo = PPO.create(
        env=env,
        env_params=env_params,
        total_timesteps=131_072,
        num_envs=32,
        num_steps=64,
        num_epochs=8,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
    )

    # JIT-compile the full training loop
    train_fn = jax.jit(algo.train)
    key = jax.random.PRNGKey(0)
    train_state, evaluation = train_fn(key)
    print(f'Training complete. Evaluation returns: {evaluation}')

    # Vmap across seeds for parallel independent runs
    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    vmapped_train = jax.vmap(train_fn)
    train_states, evaluations = vmapped_train(keys)
    print(f'Parallel runs complete. Shape: {evaluations.shape}')


if __name__ == '__main__':
    main()
