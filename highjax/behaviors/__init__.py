'''Behavior measures for HighJax.

A behavior is a named measure that scores a policy on hand-crafted scenarios.
Each scenario has a traffic state (converted to an observation) and action
weights. The score is the weighted average of the policy's action probabilities
across all scenarios.

Behaviors come from two sources:
  - Preset: JSON files bundled in this package (e.g. collision.json)
  - User: JSON files in ~/.highjax/behaviors/ (created in Octane via 'b')
'''
from __future__ import annotations

import json
import pathlib

import jax.numpy as jnp

from ..env import EnvState
from ..observations import compute_observation

ACTION_NAMES = ('left', 'idle', 'right', 'faster', 'slower')

_PRESETS_DIR = pathlib.Path(__file__).parent


def _get_behaviors_home() -> pathlib.Path:
    import os
    return pathlib.Path(os.environ.get('HIGHJAX_HOME', '~/.highjax')).expanduser() / 'behaviors'


def load_behavior(path: pathlib.Path, env, params) -> dict:
    '''Load a behavior from a JSON file.

    Returns a dict with:
        - name: behavior name
        - observations: stacked (n_scenarios, *obs_shape) array
        - action_weights: (n_scenarios, 5) array
        - abs_weights: absolute values of action_weights
        - sign: sign of action_weights
    '''
    data = json.loads(path.read_text())
    name = data['name']
    scenarios = data.get('scenarios', [])
    action_names = tuple(data.get('action_names', ACTION_NAMES))

    observations = []
    weight_rows = []

    for scenario in scenarios:
        if 'observation' in scenario:
            obs = jnp.array(scenario['observation'])
        elif 'state' in scenario:
            state = EnvState.from_scenario_dict(env, scenario['state'])
            obs = compute_observation(state, env, params)
        else:
            raise ValueError(f'Scenario in {path} has neither observation nor state')
        observations.append(obs)

        weights = jnp.zeros(len(action_names))
        raw_weights = scenario.get('action_weights', {})
        for action_name, w in raw_weights.items():
            idx = action_names.index(action_name)
            weights = weights.at[idx].set(w)
        weight_rows.append(weights)

    return {
        'name': name,
        'observations': jnp.stack(observations),
        'action_weights': jnp.stack(weight_rows),
        'abs_weights': jnp.abs(jnp.stack(weight_rows)),
        'sign': jnp.sign(jnp.stack(weight_rows)),
    }


def _install_presets() -> None:
    '''Copy preset behaviors to the user directory so Octane can find them.

    Octane is a standalone Rust binary that cannot look inside the Python
    package, so presets need to live on disk under ~/.highjax/behaviors/highway/.
    Existing user files with the same name are not overwritten.
    '''
    import shutil
    target_dir = _get_behaviors_home() / 'highway'
    target_dir.mkdir(parents=True, exist_ok=True)
    for src in _PRESETS_DIR.glob('*.json'):
        dst = target_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)


def discover_behaviors(env, params) -> list[dict]:
    '''Load all behaviors: presets first, then user behaviors.

    User behaviors with the same name as a preset override it.
    '''
    _install_presets()

    by_name = {}

    for path in sorted(_PRESETS_DIR.glob('*.json')):
        behavior = load_behavior(path, env, params)
        by_name[behavior['name']] = behavior

    user_dir = _get_behaviors_home()
    # Check both flat dir and highway/ subdir (Octane writes to the subdir)
    for d in (user_dir, user_dir / 'highway'):
        if d.exists():
            for path in sorted(d.glob('*.json')):
                behavior = load_behavior(path, env, params)
                by_name[behavior['name']] = behavior

    return list(by_name.values())


def evaluate_behavior(get_p_by_action, behavior: dict) -> float:
    '''Score a policy on a single behavior.

    Args:
        get_p_by_action: callable that takes (n_scenarios, *obs_shape)
            and returns (n_scenarios, n_actions) action probabilities.
        behavior: loaded behavior dict from load_behavior().

    Returns:
        Scalar score in roughly [0, 1].
    '''
    p_by_action = get_p_by_action(behavior['observations'])
    values = jnp.where(behavior['sign'] >= 0, p_by_action, 1 - p_by_action)
    total_weight = behavior['abs_weights'].sum()
    return float((behavior['abs_weights'] * values).sum() / total_weight)


# Backwards-compatible aliases
def load_collision_behavior():
    '''Load the collision behavior (legacy API, returns old-style dict).'''
    path = _PRESETS_DIR / 'collision.json'
    with open(path) as f:
        data = json.load(f)
    for scenario in data['scenarios']:
        if 'observation' in scenario:
            scenario['observation'] = jnp.array(scenario['observation'])
    return data


def evaluate_collision_behavior(get_action_probs, behavior=None):
    '''Evaluate collision behavior (legacy API).'''
    if behavior is None:
        behavior = load_collision_behavior()
    total_weighted_prob = 0.0
    total_weight = 0.0
    for scenario in behavior['scenarios']:
        obs = scenario.get('observation')
        if obs is None:
            continue
        action_weights = scenario.get('action_weights', {})
        probs = get_action_probs(obs)
        for action_name, weight in action_weights.items():
            action_idx = list(ACTION_NAMES).index(action_name)
            total_weighted_prob += float(probs[action_idx]) * weight
            total_weight += weight
    if total_weight > 0:
        return total_weighted_prob / total_weight
    return 0.0
