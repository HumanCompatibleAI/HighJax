'''
Determinism parity test: verify HighJax produces the same rollout outputs as
Viola's AsphaltEnv given the same seed.

Both environments implement the same highway driving physics (bicycle model,
IDM, MOBIL, collision detection). This test confirms they produce identical
observations, rewards, and done flags for the same PRNG key and action sequence.
'''
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import highjax

viola = pytest.importorskip('viola')


def _create_highjax_env(n_npcs=5, n_lanes=4):
    '''Create a HighJax environment with matching parameters.'''
    env, params = highjax.make('highjax-v0', n_npcs=n_npcs, n_lanes=n_lanes)
    return env, params


def _create_viola_env(n_npcs=5, n_lanes=4):
    '''Create a Viola AsphaltEnv with matching parameters.

    Parameters are chosen to match HighJax defaults: sequential NPC spawning,
    highway-v0 reward preset, absolute ego observation mode, lane_distance
    sorting.
    '''
    from viola.envs.asphalt_env.asphalt_env import AsphaltEnv
    env = AsphaltEnv(
        n_npcs=n_npcs,
        n_lanes=n_lanes,
        npc_spawn_mode='sequential',
        reward_preset='highway-v0',
        ego_obs_mode='absolute',
        npc_sorting='lane_distance',
    )
    return env


def _compare_arrays(name, a, b, rtol=1e-5, atol=1e-5):
    '''Compare two arrays and return a diagnostic dict if they differ.'''
    a_np = np.asarray(a)
    b_np = np.asarray(b)

    if a_np.shape != b_np.shape:
        return {
            'name': name,
            'match': False,
            'reason': f'shape mismatch: highjax={a_np.shape} viola={b_np.shape}',
        }

    close = np.allclose(a_np, b_np, rtol=rtol, atol=atol)
    result = {'name': name, 'match': close}

    if not close:
        diff = np.abs(a_np - b_np)
        max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
        result['max_abs_diff'] = float(np.max(diff))
        result['max_diff_at'] = max_diff_idx
        result['highjax_val'] = float(a_np[max_diff_idx])
        result['viola_val'] = float(b_np[max_diff_idx])
        result['highjax_array'] = a_np
        result['viola_array'] = b_np

    return result


def _print_comparison(result):
    '''Print diagnostic info about a comparison.'''
    status = 'MATCH' if result['match'] else 'MISMATCH'
    print(f'  {result["name"]}: {status}')
    if not result['match']:
        if 'reason' in result:
            print(f'    {result["reason"]}')
        else:
            print(f'    max_abs_diff={result["max_abs_diff"]:.2e} '
                  f'at {result["max_diff_at"]}')
            print(f'    highjax={result["highjax_val"]:.8f} '
                  f'viola={result["viola_val"]:.8f}')
            if result['highjax_array'].size <= 25:
                print(f'    highjax_full={result["highjax_array"]}')
                print(f'    viola_full={result["viola_array"]}')


def _unflatten_viola_state(vi_env, vi_flat_state):
    '''Unflatten a Viola flat state array into an AsphaltState.'''
    from viola.envs.asphalt_env.asphalt_state import AsphaltState
    return AsphaltState.unflatten(vi_env, vi_flat_state)


def test_reset_parity():
    '''Verify that reset produces identical initial states and observations.'''
    n_npcs = 5
    n_lanes = 4
    key = jax.random.PRNGKey(42)

    hj_env, hj_params = _create_highjax_env(n_npcs=n_npcs, n_lanes=n_lanes)
    vi_env = _create_viola_env(n_npcs=n_npcs, n_lanes=n_lanes)

    # Reset both envs with the same key
    hj_obs, hj_state = hj_env.reset(key, hj_params)
    vi_flat_state = vi_env.single_reset(key)

    # Extract Viola observation (remove agent dimension)
    vi_obs = vi_env.state_to_observation_by_agent(vi_flat_state)[0]
    vi_state = _unflatten_viola_state(vi_env, vi_flat_state)

    print('\n=== Reset Parity ===')

    ego_result = _compare_arrays('ego_state', hj_state.ego_state,
                                 vi_state.ego_state)
    _print_comparison(ego_result)

    npc_result = _compare_arrays('npc_states', hj_state.npc_states,
                                 vi_state.npc_states)
    _print_comparison(npc_result)

    time_result = _compare_arrays('time', hj_state.time, vi_state.time)
    _print_comparison(time_result)

    obs_result = _compare_arrays('observation', hj_obs, vi_obs)
    _print_comparison(obs_result)

    assert ego_result['match'], 'Ego state mismatch after reset'
    assert npc_result['match'], 'NPC states mismatch after reset'
    assert time_result['match'], 'Time mismatch after reset'
    assert obs_result['match'], 'Observation mismatch after reset'


def test_step_parity():
    '''Verify that stepping produces identical outputs for a short trajectory.'''
    n_npcs = 5
    n_lanes = 4
    n_steps = 10
    key = jax.random.PRNGKey(42)

    hj_env, hj_params = _create_highjax_env(n_npcs=n_npcs, n_lanes=n_lanes)
    vi_env = _create_viola_env(n_npcs=n_npcs, n_lanes=n_lanes)

    # Reset both
    reset_key, step_keys_source = jax.random.split(key)
    hj_obs, hj_state = hj_env.reset(reset_key, hj_params)
    vi_flat_state = vi_env.single_reset(reset_key)

    # Action sequence exercising all action types
    actions = [1, 3, 1, 2, 1, 4, 1, 0, 1, 1]

    print('\n=== Step Parity ===')
    all_match = True

    for t in range(n_steps):
        action = actions[t]
        step_key = jax.random.fold_in(step_keys_source, t)

        # HighJax step (step_env for no auto-reset)
        hj_obs, hj_state, hj_reward, hj_done, hj_info = hj_env.step_env(
            step_key, hj_state, action, hj_params)

        # Viola step: action as scalar index array
        vi_action = jnp.array([action])
        vi_step_return = vi_env.single_step(step_key, vi_flat_state, vi_action)
        vi_flat_state = vi_step_return.next_state
        vi_obs = vi_env.state_to_observation_by_agent(vi_flat_state)[0]

        vi_state = _unflatten_viola_state(vi_env, vi_flat_state)
        vi_reward = vi_step_return.reward_by_agent[0]
        vi_done = vi_env.is_done(vi_flat_state)

        print(f'\n  Step {t} (action={action}):')

        ego_result = _compare_arrays(f'step{t}_ego', hj_state.ego_state,
                                     vi_state.ego_state)
        _print_comparison(ego_result)

        npc_result = _compare_arrays(f'step{t}_npcs', hj_state.npc_states,
                                     vi_state.npc_states)
        _print_comparison(npc_result)

        obs_result = _compare_arrays(f'step{t}_obs', hj_obs, vi_obs)
        _print_comparison(obs_result)

        reward_result = _compare_arrays(f'step{t}_reward', hj_reward,
                                        vi_reward)
        _print_comparison(reward_result)

        done_result = _compare_arrays(f'step{t}_done', hj_done, vi_done)
        _print_comparison(done_result)

        step_match = all(r['match'] for r in [
            ego_result, npc_result, obs_result, reward_result, done_result,
        ])
        if not step_match:
            all_match = False

    assert all_match, 'Step parity failed -- see diagnostics above'


def test_reward_formula_parity():
    '''Verify that the highway-v0 reward formula produces identical values.'''
    n_npcs = 5
    n_lanes = 4
    key = jax.random.PRNGKey(123)

    hj_env, hj_params = _create_highjax_env(n_npcs=n_npcs, n_lanes=n_lanes)
    vi_env = _create_viola_env(n_npcs=n_npcs, n_lanes=n_lanes)

    hj_obs, hj_state = hj_env.reset(key, hj_params)
    vi_flat_state = vi_env.single_reset(key)

    # Take one IDLE step
    step_key = jax.random.PRNGKey(99)
    action = 1

    hj_obs, hj_state, hj_reward, hj_done, _ = hj_env.step_env(
        step_key, hj_state, action, hj_params)

    vi_action = jnp.array([action])
    vi_step_return = vi_env.single_step(step_key, vi_flat_state, vi_action)

    vi_reward = vi_step_return.reward_by_agent[0]
    vi_score = vi_step_return.score_by_agent[0]

    print('\n=== Reward Formula Parity ===')
    print(f'  HighJax reward: {float(hj_reward):.8f}')
    print(f'  Viola reward:   {float(vi_reward):.8f}')
    print(f'  Viola score:    {float(vi_score):.8f}')

    reward_result = _compare_arrays('reward', hj_reward, vi_reward)
    _print_comparison(reward_result)

    # In highway-v0 mode, reward == score
    score_result = _compare_arrays('score', hj_reward, vi_score)
    _print_comparison(score_result)

    assert reward_result['match'], 'Reward mismatch'
    assert score_result['match'], 'Score mismatch'
