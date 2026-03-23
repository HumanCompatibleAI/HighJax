'''Basic HighJax usage: create an environment, reset, step, and print observations.'''
from __future__ import annotations

import jax
import jax.numpy as jnp

import highjax


def main():
    # Create the highway environment with 4 lanes and 10 NPC vehicles
    env, params = highjax.make('highjax-v0', n_lanes=4, n_npcs=10)

    print(f'Environment created:')
    print(f'  Observation shape: {env.observation_shape}')
    print(f'  Number of actions: {env.num_actions}')
    print(f'  Actions: LEFT=0, IDLE=1, RIGHT=2, FASTER=3, SLOWER=4')

    # Reset the environment
    key = jax.random.PRNGKey(42)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key, params)

    print(f'\nAfter reset:')
    print(f'  Observation (ego + 4 nearest NPCs):\n{obs}')
    print(f'  Ego speed: {float(state.ego_state[3]):.1f} m/s')
    print(f'  Time: {float(state.time):.1f}s')

    # Run a few steps. Note: HighJax uses auto-reset — when done=True (crash),
    # the environment automatically resets on the next step, so done goes back
    # to False and the episode continues from a fresh initial state.
    actions = [1, 3, 3, 1, 1, 2, 1, 1, 0, 1]  # Mix of actions
    action_names = ['LEFT', 'IDLE', 'RIGHT', 'FASTER', 'SLOWER']

    print(f'\nRunning {len(actions)} steps:')
    total_reward = 0.0
    for i, action in enumerate(actions):
        key, step_key = jax.random.split(key)
        obs, state, reward, done, info = env.step(step_key, state, action, params)
        total_reward += float(reward)
        print(f'  Step {i:2d}: action={action_names[action]:7s}  '
              f'reward={float(reward):.3f}  speed={float(state.ego_state[3]):.1f} m/s  '
              f'done={bool(done)}')
        if bool(done):
            print(f'          ^ Episode ended. Next step will auto-reset to a new episode.')

    print(f'\nTotal reward: {total_reward:.3f}')

    # Demonstrate JIT compilation
    print('\nJIT-compiled step:')
    jitted_step = jax.jit(lambda k, s, a: env.step(k, s, a, params))
    key, step_key = jax.random.split(key)
    obs, state, reward, done, info = jitted_step(step_key, state, 1)
    print(f'  reward={float(reward):.3f}  (compiled successfully)')

    # Demonstrate vectorization (vmap)
    print('\nVectorized reset (8 parallel environments):')
    keys = jax.random.split(key, 8)
    vmapped_reset = jax.vmap(lambda k: env.reset(k, params))
    obs_batch, state_batch = vmapped_reset(keys)
    print(f'  Batch observation shape: {obs_batch.shape}')
    print(f'  Ego speeds: {jnp.round(state_batch.ego_state[:, 3], 1)}')


if __name__ == '__main__':
    main()
