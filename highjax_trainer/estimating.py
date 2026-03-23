'''Neural network estimators for actor and critic.'''
from __future__ import annotations

from typing import Any
import math

import jax
from jax import numpy as jnp
from flax import linen as nn

from . import defaults


class AnnActorEstimator(nn.Module):
    '''Feedforward actor network: observation -> action logits.'''
    observation_shape: tuple[int, ...]
    action_size: int
    n_neurons_by_layer: tuple[int, ...] = defaults.DEFAULT_ANN_N_NEURONS_BY_LAYER

    @property
    def observation_size(self) -> int:
        return math.prod(self.observation_shape)

    def setup(self) -> None:
        self.hidden_layers = tuple(nn.Dense(features=n_neurons)
                                   for n_neurons in self.n_neurons_by_layer)
        self.end_layer = nn.Dense(features=self.action_size)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, ...]:
        x = x.reshape(*x.shape[:-len(self.observation_shape)], self.observation_size)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = nn.relu(x)
        return (self.end_layer(x),)

    def get_initial_theta(self, seed: jax.Array, batch_shape: tuple[int, ...]) -> dict:
        return self.init(seed, jnp.ones([*batch_shape, *self.observation_shape]))

    @staticmethod
    def get_kwargs(observation_shape: tuple[int, ...], action_size: int) -> dict[str, Any]:
        return {
            'observation_shape': observation_shape,
            'action_size': action_size,
        }


class AnnCriticEstimator(nn.Module):
    '''Feedforward critic network: observation -> scalar value.'''
    observation_shape: tuple[int, ...]
    n_neurons_by_layer: tuple[int, ...] = defaults.DEFAULT_ANN_N_NEURONS_BY_LAYER

    @property
    def observation_size(self) -> int:
        return math.prod(self.observation_shape)

    def setup(self) -> None:
        self.hidden_layers = tuple(nn.Dense(features=n_neurons)
                                   for n_neurons in self.n_neurons_by_layer)
        self.end_layer = nn.Dense(features=1)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.reshape(*x.shape[:-len(self.observation_shape)], self.observation_size)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = nn.relu(x)
        return self.end_layer(x)

    def get_initial_theta(self, seed: jax.Array, batch_shape: tuple[int, ...]) -> dict:
        return self.init(seed, jnp.ones([*batch_shape, *self.observation_shape]))

    @staticmethod
    def get_kwargs(observation_shape: tuple[int, ...]) -> dict[str, Any]:
        return {
            'observation_shape': observation_shape,
        }
