'''Train a PPO agent on HighJax and print final metrics.'''
from __future__ import annotations

import jax

import highjax
from highjax_trainer.config import AgentConfig
from highjax_trainer.training import train


def main():
    # Create environment
    env, params = highjax.make('highjax-v0', n_npcs=10)

    # Configure PPO hyperparameters
    config = AgentConfig(
        actor_lr=3e-4,
        critic_lr=3e-3,
        discount=0.95,
        ppo_clip_epsilon=0.2,
        entropy_temperature=0.05,
        n_mts_per_minibatch=64,
    )

    # Train for 20 epochs (returns a Population)
    population = train(
        env, params, config,
        n_epochs=20,
        n_es=32,   # 32 parallel episodes
        n_ts=50,   # 50 timesteps per episode
        seed=42,
        verbose=True,
    )

    # Evaluate the trained policy (agent 0)
    brain = population[0]
    print('\nEvaluating trained policy...')
    key = jax.random.PRNGKey(99)
    keys = jax.random.split(key, 8)
    vmapped_reset = jax.vmap(lambda k: env.reset(k, params))
    obs_batch, state_batch = vmapped_reset(keys)

    action, chosen_p, v, p_by_action = brain.infer(
        jax.random.PRNGKey(0), obs_batch,
    )
    print(f'Value estimates: {v}')
    print(f'Action probabilities:\n{p_by_action}')


if __name__ == '__main__':
    main()
