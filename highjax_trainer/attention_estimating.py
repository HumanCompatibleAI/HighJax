'''Ego-attention estimator for highway driving.

Uses multi-head attention where the ego vehicle queries all observed vehicles.
This is the default estimator — it handles variable NPC counts naturally
and produces attention weights that can be visualized in Octane.
'''
from __future__ import annotations

from typing import Any

import jax
from jax import numpy as jnp
from flax import linen as nn

from . import defaults


def _scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.shape[-1]
    scores = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) / jnp.sqrt(d_k)
    if mask is not None:
        scores = jnp.where(mask, -1e9, scores)
    p_attention = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(p_attention, value)
    return output, p_attention


class VehicleEmbedding(nn.Module):
    layer_sizes: tuple[int, ...] = defaults.DEFAULT_EGO_ATTENTION_EMBEDDING_SIZES

    @nn.compact
    def __call__(self, x):
        for size in self.layer_sizes:
            x = nn.Dense(size)(x)
            x = nn.relu(x)
        return x


class EgoAttentionLayer(nn.Module):
    feature_size: int = defaults.DEFAULT_EGO_ATTENTION_FEATURE_SIZE
    heads: int = defaults.DEFAULT_EGO_ATTENTION_HEADS

    def setup(self):
        self.value_all = nn.Dense(self.feature_size, use_bias=False)
        self.key_all = nn.Dense(self.feature_size, use_bias=False)
        self.query_ego = nn.Dense(self.feature_size, use_bias=False)
        self.attention_combine = nn.Dense(self.feature_size, use_bias=False)

    @property
    def features_per_head(self):
        return self.feature_size // self.heads

    def __call__(self, ego, others, mask=None):
        n_entities = others.shape[-2] + 1
        fphead = self.features_per_head

        input_all = jnp.concatenate([ego, others], axis=-2)

        key = self.key_all(input_all)
        value = self.value_all(input_all)
        query = self.query_ego(ego)

        key = key.reshape(*key.shape[:-1], self.heads, fphead)
        key = jnp.moveaxis(key, -2, -3)
        value = value.reshape(*value.shape[:-1], self.heads, fphead)
        value = jnp.moveaxis(value, -2, -3)
        query = query.reshape(*query.shape[:-1], self.heads, fphead)
        query = jnp.moveaxis(query, -2, -3)

        if mask is not None:
            mask = mask.reshape(*mask.shape[:-1], 1, 1, mask.shape[-1])
            mask = jnp.broadcast_to(
                mask, mask.shape[:-3] + (self.heads, 1, n_entities))

        attention_out, attention_matrix = _scaled_dot_product_attention(
            query, key, value, mask)

        attention_out = attention_out.reshape(
            *attention_out.shape[:-3], self.feature_size)
        ego_flat = ego.reshape(*ego.shape[:-2], self.feature_size)

        result = (self.attention_combine(attention_out) + ego_flat) / 2
        return result, attention_matrix


class AttentionActorEstimator(nn.Module):
    observation_shape: tuple[int, ...]
    action_size: int
    embedding_sizes: tuple[int, ...] = defaults.DEFAULT_EGO_ATTENTION_EMBEDDING_SIZES
    feature_size: int = defaults.DEFAULT_EGO_ATTENTION_FEATURE_SIZE
    heads: int = defaults.DEFAULT_EGO_ATTENTION_HEADS
    n_attention_layers: int = defaults.DEFAULT_EGO_ATTENTION_N_LAYERS
    output_sizes: tuple[int, ...] = defaults.DEFAULT_EGO_ATTENTION_OUTPUT_SIZES
    presence_feature_idx: int = 0

    def setup(self):
        self.ego_embedding = VehicleEmbedding(layer_sizes=self.embedding_sizes)
        self.other_embedding = VehicleEmbedding(layer_sizes=self.embedding_sizes)
        self.attention_layers = tuple(
            EgoAttentionLayer(feature_size=self.feature_size, heads=self.heads)
            for _ in range(self.n_attention_layers)
        )
        self.output_mlp = tuple(nn.Dense(size) for size in self.output_sizes)
        self.end_layers = tuple(
            nn.Dense(features=self.action_size) for _ in range(1))

    def __call__(self, x):
        ego = x[..., 0:1, :]
        others = x[..., 1:, :]

        presence = x[..., :, self.presence_feature_idx]
        mask = presence < 0.5

        ego_emb = self.ego_embedding(ego)
        others_emb = self.other_embedding(others)

        features = ego_emb
        for attention_layer in self.attention_layers:
            features, _ = attention_layer(features, others_emb, mask)
            features = features[..., jnp.newaxis, :]
        features = features.squeeze(-2)

        for layer in self.output_mlp:
            features = nn.relu(layer(features))

        return tuple(end_layer(features) for end_layer in self.end_layers)

    def forward_with_attention(self, x):
        ego = x[..., 0:1, :]
        others = x[..., 1:, :]

        presence = x[..., :, self.presence_feature_idx]
        mask = presence < 0.5

        ego_emb = self.ego_embedding(ego)
        others_emb = self.other_embedding(others)

        features = ego_emb
        last_attention = None
        for attention_layer in self.attention_layers:
            features, last_attention = attention_layer(
                features, others_emb, mask,
            )
            features = features[..., jnp.newaxis, :]
        features = features.squeeze(-2)

        # last_attention: (..., heads, 1, n_entities)
        # Average across heads, squeeze query dim
        if last_attention is not None:
            attention = last_attention.mean(axis=-3)[..., 0, :]
        else:
            attention = None

        for layer in self.output_mlp:
            features = nn.relu(layer(features))

        return tuple(
            end_layer(features) for end_layer in self.end_layers
        ), attention

    def get_initial_theta(self, seed: jax.Array,
                          batch_shape: tuple[int, ...]) -> dict:
        return self.init(
            seed, jnp.ones([*batch_shape, *self.observation_shape]),
        )

    @staticmethod
    def get_kwargs(observation_shape: tuple[int, ...],
                   action_size: int) -> dict[str, Any]:
        return {
            'observation_shape': observation_shape,
            'action_size': action_size,
        }


class AttentionCriticEstimator(nn.Module):
    observation_shape: tuple[int, ...]
    embedding_sizes: tuple[int, ...] = defaults.DEFAULT_EGO_ATTENTION_EMBEDDING_SIZES
    feature_size: int = defaults.DEFAULT_EGO_ATTENTION_FEATURE_SIZE
    heads: int = defaults.DEFAULT_EGO_ATTENTION_HEADS
    n_attention_layers: int = defaults.DEFAULT_EGO_ATTENTION_N_LAYERS
    output_sizes: tuple[int, ...] = defaults.DEFAULT_EGO_ATTENTION_CRITIC_OUTPUT_SIZES
    presence_feature_idx: int = 0

    def setup(self):
        self.ego_embedding = VehicleEmbedding(layer_sizes=self.embedding_sizes)
        self.other_embedding = VehicleEmbedding(layer_sizes=self.embedding_sizes)
        self.attention_layers = tuple(
            EgoAttentionLayer(feature_size=self.feature_size, heads=self.heads)
            for _ in range(self.n_attention_layers)
        )
        self.output_mlp = tuple(nn.Dense(size) for size in self.output_sizes)
        self.end_layer = nn.Dense(features=1)

    def __call__(self, x):
        ego = x[..., 0:1, :]
        others = x[..., 1:, :]

        presence = x[..., :, self.presence_feature_idx]
        mask = presence < 0.5

        ego_emb = self.ego_embedding(ego)
        others_emb = self.other_embedding(others)

        features = ego_emb
        for attention_layer in self.attention_layers:
            features, _ = attention_layer(features, others_emb, mask)
            features = features[..., jnp.newaxis, :]
        features = features.squeeze(-2)

        for layer in self.output_mlp:
            features = nn.relu(layer(features))

        return self.end_layer(features)

    def get_initial_theta(self, seed: jax.Array, batch_shape: tuple[int, ...]) -> dict:
        return self.init(seed, jnp.ones([*batch_shape, *self.observation_shape]))

    @staticmethod
    def get_kwargs(observation_shape: tuple[int, ...]) -> dict[str, Any]:
        return {
            'observation_shape': observation_shape,
        }
