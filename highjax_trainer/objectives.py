'''GAE advantage computation and value function objectives.'''
from __future__ import annotations

from typing import Optional

import jax
from jax import numpy as jnp


_AffinePair = tuple[jax.Array, jax.Array]


def _affine_combine(pair1: _AffinePair, pair2: _AffinePair) -> _AffinePair:
    '''Combine two affine transformations (a1, b1) and (a2, b2).

    Each pair represents x -> a*x + b. Composing "apply pair1 first, then pair2"
    gives x -> a2*(a1*x + b1) + b2 = (a2*a1)*x + (a2*b1 + b2).
    '''
    a1, b1 = pair1
    a2, b2 = pair2
    return (a1 * a2, a2 * b1 + b2)


def _backward_cumsum_with_decay(values_by_e_by_t: jax.Array, factor: float) -> jax.Array:
    '''Compute A_t = values_t + factor * A_{t+1} via reverse associative scan.

    O(log n) parallel depth on GPU, numerically stable.
    '''
    a = jnp.full_like(values_by_e_by_t, factor)
    _, result = jax.lax.associative_scan(_affine_combine, (a, values_by_e_by_t), reverse=True)
    return result


@jax.jit(static_argnames=('discount', 'gae_lambda'))
def calculate_advantage_by_e_by_t(residual_by_e_by_t: jax.Array, discount: float,
                                  gae_lambda: float) -> jax.Array:
    factor = (discount * gae_lambda)
    if factor == 0:
        return residual_by_e_by_t
    else:
        return _backward_cumsum_with_decay(residual_by_e_by_t, factor)


@jax.jit(static_argnames=('discount',))
def calculate_return_by_e_by_t(reward_by_e_by_t: jax.Array,
                               discount: float) -> jax.Array:
    return _backward_cumsum_with_decay(reward_by_e_by_t, discount)


@jax.jit(static_argnames=('discount', 'critic_lambda'))
def calculate_v_objective(residual_by_e_by_t: jax.Array, discount: float,
                         critic_lambda: float,
                         vital_by_e_by_t: Optional[jax.Array] = None) -> jax.Array:
    if critic_lambda == 0:
        td_lambda_error_by_e_by_t = residual_by_e_by_t
    else:
        factor = discount * critic_lambda
        td_lambda_error_by_e_by_t = _backward_cumsum_with_decay(
            residual_by_e_by_t, factor)

    td_lambda_loss_by_e_by_t = 0.5 * (td_lambda_error_by_e_by_t ** 2)

    if vital_by_e_by_t is not None:
        return -(td_lambda_loss_by_e_by_t * vital_by_e_by_t).sum() / vital_by_e_by_t.sum()
    return -td_lambda_loss_by_e_by_t.mean()
