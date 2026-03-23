'''CLI entry point for the trainer.'''
from __future__ import annotations

from .ozette import install as _install_ozette
_install_ozette('highjax-trainer', buffer_only=True, show_path=False)

from .filtros import filtros
filtros()

import click

from . import defaults


@click.group()
def cli():
    '''HighJax PPO trainer.'''
    pass


@cli.command()
@click.option('--n-epochs', '-e', default=defaults.DEFAULT_N_EPOCHS,
              help='Number of training epochs')
@click.option('--n-es', default=defaults.DEFAULT_N_ES_PER_EPOCH,
              help='Number of parallel episodes per epoch')
@click.option('--n-ts', default=defaults.DEFAULT_N_TS_PER_E,
              help='Number of timesteps per episode')
@click.option('--seed', '-s', default=defaults.DEFAULT_SEED_NUMBER,
              help='Random seed')
@click.option('--actor-lr', default=defaults.DEFAULT_ACTOR_LR,
              help='Actor learning rate')
@click.option('--critic-lr', default=defaults.DEFAULT_CRITIC_LR,
              help='Critic learning rate')
@click.option('--discount', default=defaults.DEFAULT_DISCOUNT,
              help='Discount factor (gamma)')
@click.option('--ppo-clip-epsilon', default=defaults.DEFAULT_PPO_CLIP_EPSILON,
              help='PPO clipping epsilon')
@click.option('--entropy-temperature', default=defaults.DEFAULT_ENTROPY_TEMPERATURE,
              help='Entropy regularization coefficient')
@click.option('--n-sweeps', default=defaults.DEFAULT_N_SWEEPS_PER_EPOCH,
              help='Number of minibatch sweeps per epoch')
@click.option('--minibatch-size', default=defaults.DEFAULT_N_MTS_PER_MINIBATCH,
              help='Minibatch size (timesteps per minibatch)')
@click.option('--n-critic-iterations', default=defaults.DEFAULT_N_CRITIC_ITERATIONS,
              help='Number of critic update iterations per epoch')
@click.option('--target-kld', default=None, type=float,
              help='Target KLD for policy updates (binary search)')
@click.option('--n-lanes', default=4, help='Number of highway lanes')
@click.option('--n-npcs', default=50, help='Number of NPC vehicles')
@click.option('--no-trek', is_flag=True, help='Disable trek recording')
@click.option('--n-sample-es', default=1,
              help='Number of episodes to sample per epoch for trek')
@click.option('--trek-path', default=None, type=click.Path(),
              help='Custom path for trek directory')
@click.option('--replay-buffer-factor',
              default=defaults.DEFAULT_REPLAY_BUFFER_FACTOR,
              help='Replay buffer size multiplier')
def train(n_epochs, n_es, n_ts, seed, actor_lr, critic_lr, discount,
          ppo_clip_epsilon, entropy_temperature, n_sweeps, minibatch_size,
          n_critic_iterations, target_kld, n_lanes, n_npcs, no_trek,
          n_sample_es, trek_path, replay_buffer_factor):
    '''Train a PPO agent on the HighJax highway environment.'''
    import pathlib

    import highjax
    from .config import AgentConfig
    from .training import train as run_training

    env, params = highjax.make('highjax-v0', n_lanes=n_lanes, n_npcs=n_npcs)

    agent_config = AgentConfig(
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        discount=discount,
        ppo_clip_epsilon=ppo_clip_epsilon,
        entropy_temperature=entropy_temperature,
        n_sweeps_per_epoch=n_sweeps,
        n_mts_per_minibatch=minibatch_size,
        n_critic_iterations=n_critic_iterations,
        target_kld=target_kld,
    )

    run_training(
        env, params, agent_config,
        n_epochs=n_epochs, n_es=n_es, n_ts=n_ts, seed=seed,
        verbose=True,
        trek=not no_trek,
        n_sample_es=n_sample_es,
        trek_path=pathlib.Path(trek_path) if trek_path else None,
        replay_buffer_factor=replay_buffer_factor,
    )
