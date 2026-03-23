'''Shared logic for golden run tests.'''
from __future__ import annotations

import jax.numpy as jnp


def print_golden_data(epoch_metrics: list[dict]) -> None:
    n = len(epoch_metrics)
    keys = sorted(epoch_metrics[0].keys())
    print('golden_data = {')
    print(f"    'epoch': {tuple(range(1, n + 1))},")
    for key in keys:
        vals = tuple(epoch_metrics[i][key] for i in range(n))
        print(f"    '{key}': {vals},")
    print('}')


def check_golden_run(epoch_metrics: list[dict], golden_data: dict) -> None:
    '''Compare per-epoch training metrics against golden data.

    epoch_metrics: list of dicts, one per epoch, with keys like
        'reward_mean', 'v_loss', 'kld'.
    golden_data: dict mapping metric names to tuples of expected values,
        one per epoch. Must include 'epoch' key.
    '''
    n_epochs = len(golden_data['epoch'])
    assert len(epoch_metrics) == n_epochs, (
        f'Expected {n_epochs} epochs, got {len(epoch_metrics)}'
    )

    for i, metrics in enumerate(epoch_metrics):
        epoch = golden_data['epoch'][i]
        bad_fields = []

        for key, expected_values in golden_data.items():
            if key == 'epoch':
                continue
            expected = expected_values[i]
            actual = float(metrics[key])

            if not jnp.isclose(actual, expected, rtol=1e-04, atol=1e-07):
                bad_fields.append((key, actual, expected))

        if bad_fields:
            closest = min(bad_fields, key=lambda x: abs(x[1] - x[2]))
            farthest = max(bad_fields, key=lambda x: abs(x[1] - x[2]))
            field_names = [f[0] for f in bad_fields]
            raise AssertionError(
                f'In epoch {epoch}, these fields didn\'t match golden data: '
                f'{field_names}. The closest is {closest[0]} which is '
                f'{closest[1]} instead of {closest[2]}, the farthest is '
                f'{farthest[0]} which is {farthest[1]} instead of '
                f'{farthest[2]}.'
            )
