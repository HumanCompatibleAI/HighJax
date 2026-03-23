# Trainer Behaviors

Related: [[Trainer training]] | [[Octane behavior explorer]]

Behaviors are named measures that score a policy on hand-crafted traffic scenarios. Each scenario specifies a traffic state and action weights; the score is the weighted average of the policy's action probabilities across all scenarios.

## Two sources

**Preset behaviors** are JSON files bundled in `highjax/behaviors/`. Currently there is one: `collision.json`.

**User behaviors** are JSON files in `~/.highjax/behaviors/` or `~/.highjax/behaviors/highway/`. Octane saves to the `highway/` subdirectory; the trainer checks both locations. These are created in Octane by pressing `b` during episode playback — you capture a traffic moment, mark which actions you want (positive weight) or don't want (negative weight), and name the behavior.

At training start, both sources are discovered. User behaviors with the same name as a preset override it.

## JSON format

```json
{
  "name": "collision",
  "scenarios": [
    {
      "state": {
        "ego_x": 6080.3, "ego_y": 12.0, "ego_heading": 0.0,
        "ego_speed": 29.9, "ego_lane": 3,
        "npc0_x": 6091.6, "npc0_y": 4.0, "npc0_heading": 0.0,
        "npc0_speed": 22.8, "npc0_lane": 1
      },
      "action_weights": {
        "left": 1.0,
        "faster": 0.5
      }
    }
  ]
}
```

Each scenario has:
- **state**: Traffic geometry. `ego_*` fields for the ego vehicle, `npc{i}_*` for NPCs. Converted to an observation via `EnvState.from_scenario_dict()` + `compute_observation()`.
- **action_weights**: Maps action names to signed weights. Positive weight means "this action should have high probability"; negative means "this action should have low probability".

A scenario can also have a raw `observation` array. When both `state` and `observation` are present, `observation` is used directly for scoring and `state` is used for rendering in Octane. When only `state` is present, it is converted to an observation for scoring. `state` is the standard format (what Octane saves).

## Scoring

For each scenario, the policy produces action probabilities. The score is:

```
score = sum(abs(w) * value) / sum(abs(w))
```

where `value = P(action)` for positive weights, `value = 1 - P(action)` for negative weights. This gives a scalar in roughly (0, 1) — higher means the policy aligns more with the specified action preferences.

## Integration with training

Behaviors are discovered once at training startup via `discover_behaviors(env, params)` and evaluated every epoch. Each behavior score is written to epochia as `behavior.{name}` (e.g. `behavior.collision`). The evaluation uses the updated brain from the current epoch.

## API

```python
from highjax.behaviors import discover_behaviors, evaluate_behavior

behaviors = discover_behaviors(env, params)
for b in behaviors:
    score = evaluate_behavior(brain.get_p_by_action_by_i_observation, b)
    print(f'{b["name"]}: {score:.4f}')
```

Note: `discover_behaviors()` loads JSON files and converts them to pre-processed dicts with keys `name`, `observations` (JAX arrays ready for inference), `action_weights`, `abs_weights`, `sign`. The JSON `scenarios` with `state` dicts are converted to observation arrays during loading via `EnvState.from_scenario_dict()` + `compute_observation()`.

## Source files

| File | Role |
|------|------|
| `highjax/behaviors/__init__.py` | Loading, discovery, evaluation |
| `highjax/behaviors/collision.json` | Preset collision behavior (6 scenarios) |
| `highjax/env.py` | `EnvState.from_scenario_dict()` |
| `highjax/observations.py` | `compute_observation()` |
