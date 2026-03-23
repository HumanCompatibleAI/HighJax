'''JAX utility functions for the trainer.'''
from __future__ import annotations

import jax
from jax import numpy as jnp


def categorical_choice(seed: jax.Array, p: jax.Array) -> jax.Array:
    '''Sample from categorical distribution using Gumbel-max trick.

    Args:
        seed: JAX PRNG key
        p: Action probabilities of shape (..., n_actions)

    Returns:
        Sampled action indices of shape (..., 1)
    '''
    gumbel_noise = jax.random.gumbel(seed, p.shape)
    return jnp.argmax(jnp.log(p) + gumbel_noise, axis=-1, keepdims=True)
