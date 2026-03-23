'''Train HighJax with PureJaxRL's PPO pipeline.

PureJaxRL (https://github.com/luchris429/purejaxrl) is not a library you
import — it's a collection of reference training scripts for end-to-end
JIT-compiled RL in JAX. This example reproduces PureJaxRL's core PPO loop
with HighJax as a drop-in gymnax environment.

The only HighJax-specific line is the env creation. Everything else is
standard PureJaxRL code.

Requires: pip install distrax
'''
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import NamedTuple

import distrax

import highjax


# --- PureJaxRL network (same as purejaxrl/ppo.py) ---

class ActorCritic(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, x):
        actor = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)),
                         bias_init=constant(0.0))(x)
        actor = nn.tanh(actor)
        actor = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)),
                         bias_init=constant(0.0))(actor)
        actor = nn.tanh(actor)
        actor = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                         bias_init=constant(0.0))(actor)
        pi = distrax.Categorical(logits=actor)

        critic = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)),
                          bias_init=constant(0.0))(x)
        critic = nn.tanh(critic)
        critic = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)),
                          bias_init=constant(0.0))(critic)
        critic = nn.tanh(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0),
                          bias_init=constant(0.0))(critic)
        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


# --- PureJaxRL observation wrapper ---

class FlattenObsWrapper:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def observation_space(self, params):
        import gymnasium
        orig = self._env.observation_space(params)
        flat_size = 1
        for s in orig.shape:
            flat_size *= s
        return gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(flat_size,))

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return jnp.reshape(obs, (-1,)), state

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(
            key, state, action, params)
        return jnp.reshape(obs, (-1,)), state, reward, done, info


# --- PureJaxRL training function (adapted from purejaxrl/ppo.py) ---

def make_train(config, env, env_params):
    config = {**config}
    config['MINIBATCH_SIZE'] = (
        config['NUM_ENVS'] * config['NUM_STEPS'] // config['NUM_MINIBATCHES']
    )

    def train(rng):
        network = ActorCritic(env.action_space(env_params).n)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        tx = optax.chain(
            optax.clip_by_global_norm(config['MAX_GRAD_NORM']),
            optax.adam(config['LR'], eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply, params=network_params, tx=tx,
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config['NUM_ENVS'])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(
            reset_rng, env_params)

        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config['NUM_ENVS'])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs)
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config['NUM_STEPS'])

            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done, transition.value, transition.reward)
                    delta = (reward + config['GAMMA'] * next_value
                             * (1 - done) - value)
                    gae = (delta + config['GAMMA'] * config['GAE_LAMBDA']
                           * (1 - done) * gae)
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch, reverse=True, unroll=16)
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config['CLIP_EPS'], config['CLIP_EPS'])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(
                            value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped).mean()

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(
                            ratio, 1.0 - config['CLIP_EPS'],
                            1.0 + config['CLIP_EPS']) * gae
                        loss_actor = -jnp.minimum(
                            loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()
                        total_loss = (loss_actor
                                      + config['VF_COEF'] * value_loss
                                      - config['ENT_COEF'] * entropy)
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = (
                    update_state)
                rng, _rng = jax.random.split(rng)
                batch_size = (config['MINIBATCH_SIZE']
                              * config['NUM_MINIBATCHES'])
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config['NUM_MINIBATCHES'], -1]
                        + list(x.shape[1:])),
                    shuffled_batch)
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches)
                update_state = (
                    train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (
                train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config['UPDATE_EPOCHS'])
            train_state = update_state[0]
            metric = traj_batch.reward.mean()
            rng = update_state[-1]
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config['NUM_UPDATES'])
        return runner_state, metrics

    return train


def main():
    # This is the only HighJax-specific line. Replace with gymnax.make()
    # for any other gymnax environment — the rest is identical.
    env, env_params = highjax.make('highjax-v0', n_npcs=5)
    env = FlattenObsWrapper(env)

    config = {
        'LR': 2.5e-4,
        'NUM_ENVS': 32,
        'NUM_STEPS': 40,
        'TOTAL_TIMESTEPS': 40_960,
        'UPDATE_EPOCHS': 4,
        'NUM_MINIBATCHES': 4,
        'GAMMA': 0.99,
        'GAE_LAMBDA': 0.95,
        'CLIP_EPS': 0.2,
        'ENT_COEF': 0.01,
        'VF_COEF': 0.5,
        'MAX_GRAD_NORM': 0.5,
    }
    config['NUM_UPDATES'] = (
        config['TOTAL_TIMESTEPS'] // config['NUM_STEPS'] // config['NUM_ENVS']
    )

    print(f'Training with PureJaxRL PPO on HighJax')
    print(f'  {config["NUM_ENVS"]} envs, {config["NUM_STEPS"]} steps, '
          f'{config["NUM_UPDATES"]} updates')

    train_fn = make_train(config, env, env_params)
    train_jit = jax.jit(train_fn)

    rng = jax.random.PRNGKey(0)
    runner_state, metrics = train_jit(rng)

    print(f'  Mean reward per update: {metrics}')
    print(f'  Training steps completed: {runner_state[0].step}')
    print('Done.')


if __name__ == '__main__':
    main()
