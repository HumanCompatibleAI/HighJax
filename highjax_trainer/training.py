from __future__ import annotations

import datetime
import pathlib
import sys

import colorama
import jax
from jax import numpy as jnp
import tqdm

from .config import AgentConfig
from .epoch_reporting import EpochReporter
from .evaluating import Evaluator
from .genesis import Genesis
from .joint_ascent import JointAscent
from .populating import Population
from .rollout import collect_rollout
from .trekking import Trek
from . import defaults

from highjax.behaviors import discover_behaviors


def make_training_progress_bar(raw_iterable, n_epochs: int, *,
                               verbose: bool):
    if not verbose or not raw_iterable:
        return raw_iterable
    epoch_n_digits = str(len(str(n_epochs + 1)))
    color = colorama.Fore.YELLOW
    if sys.stdout.isatty():
        return tqdm.tqdm(
            raw_iterable, desc='Training', unit='epoch',
            file=sys.stderr,
            bar_format=(
                '{l_bar} ' + color + '{elapsed}'
                + colorama.Fore.RESET
                + ' {n_fmt:>' + epoch_n_digits
                + '}/{total_fmt:>' + epoch_n_digits
                + '} epochs {bar}'
            ),
            **defaults.DEFAULT_TQDM_KWARGS,
        )
    else:
        return raw_iterable


def train(env, params, agent_config: AgentConfig, *,
          n_epochs: int = defaults.DEFAULT_N_EPOCHS,
          n_es: int = defaults.DEFAULT_N_ES_PER_EPOCH,
          n_ts: int = defaults.DEFAULT_N_TS_PER_E,
          seed: int = defaults.DEFAULT_SEED_NUMBER,
          verbose: bool = True,
          trek: bool = True,
          n_sample_es: int = 1,
          trek_path: pathlib.Path | None = None,
          replay_buffer_factor: int = defaults.DEFAULT_REPLAY_BUFFER_FACTOR,
          ) -> Population:
    startup_time = datetime.datetime.now()
    master_seed = jax.random.PRNGKey(seed)

    agent_configs = (agent_config,)
    population = Population.create_random(
        agent_configs, master_seed, env, params, n_es=n_es,
    )
    behaviors = discover_behaviors(env, params)

    if verbose:
        print(
            f'Created population with {population.n_agents} agent(s), '
            f'obs_shape={env.observation_shape}, '
            f'n_actions={env.num_actions}'
        )
        print(
            f'Training for {n_epochs} epochs, '
            f'{n_es} episodes x {n_ts} timesteps'
        )

    trek_obj = None
    evaluator = None
    if trek:
        trek_obj = Trek.create(
            env=env, params=params,
            agent_configs=agent_configs,
            population=population,
            trek_path=trek_path,
        )
        evaluator = Evaluator(env, params)
        from .ozette import materialize as _materialize_ozette
        _materialize_ozette(trek_obj.folder / 'train.log')
        if verbose:
            print(f'Trek: {trek_obj.folder}')

    # Genesis: initial rollout at epoch 0
    if verbose:
        print('Running initial rollout... ', end='')
        sys.stdout.flush()

    genesis, genesis_state_data = Genesis.create(
        population, master_seed, env, params,
        n_es=n_es, n_ts=n_ts,
        replay_buffer_factor=replay_buffer_factor,
    )
    vital = genesis.rollout.vital_by_e_by_t
    mean_reward = (
        genesis.rollout.reward_by_agent_by_e_by_t[:, :, 0] * vital
    ).sum() / vital.sum()

    if verbose:
        print(
            f'{colorama.Fore.GREEN}Done{colorama.Fore.RESET} '
            f'(mean reward: {float(mean_reward):.4f})'
        )

    if n_epochs == 0:
        return population

    # Epoch 1: first joint ascent from genesis
    joint_ascent = genesis.create_first_joint_ascent(
        env, params, n_es=n_es, n_ts=n_ts,
    )

    if trek_obj is not None:
        evaluation = evaluator.evaluate(
            genesis.evaluation_seed, joint_ascent.next_population,
            genesis.rollout, epoch=1,
        )
        reporter = EpochReporter(
            trek=trek_obj, epoch=1,
            joint_ascent=joint_ascent,
            evaluation=evaluation,
            startup_time=startup_time,
            env=env, params=params,
            behaviors=behaviors,
        )
        reporter.report(
            genesis_state_data,
            n_sample_es_per_epoch=n_sample_es,
        )

    if n_epochs == 1:
        _print_done(verbose)
        return joint_ascent.next_population

    # Epochs 2+
    raw_iterable = range(2, n_epochs + 1)
    iterable = make_training_progress_bar(
        raw_iterable, n_epochs, verbose=verbose,
    )

    for epoch in iterable:
        joint_ascent, state_data = joint_ascent.create_next(
            epoch, env, params, n_es, n_ts,
        )

        if trek_obj is not None:
            evaluation = evaluator.evaluate(
                joint_ascent.evaluation_seed,
                joint_ascent.next_population,
                joint_ascent.rollout.tail, epoch=epoch,
            )
            reporter = EpochReporter(
                trek=trek_obj, epoch=epoch,
                joint_ascent=joint_ascent,
                evaluation=evaluation,
                startup_time=startup_time,
                env=env, params=params,
                behaviors=behaviors,
            )
            reporter.report(
                state_data,
                n_sample_es_per_epoch=n_sample_es,
            )

        if verbose:
            tail = joint_ascent.rollout.tail
            vital = tail.vital_by_e_by_t
            reward_by_e_by_t = (
                tail.reward_by_agent_by_e_by_t[:, :, 0]
            )
            mean_reward = (
                (reward_by_e_by_t * vital).sum() / vital.sum()
            )
            v_loss = joint_ascent[0].v_loss
            kld = joint_ascent[0].kld

            if isinstance(iterable, tqdm.tqdm):
                iterable.set_postfix_str(
                    f'r={float(mean_reward):.3f} '
                    f'vl={float(v_loss):.4f} '
                    f'kld={float(kld):.4f}'
                )
            else:
                print(
                    f'Epoch {epoch:>{len(str(n_epochs))}}'
                    f'/{n_epochs}  '
                    f'reward={float(mean_reward):.4f}  '
                    f'v_loss={float(v_loss):.4f}  '
                    f'kld={float(kld):.4f}'
                )

    _print_done(verbose)
    return joint_ascent.next_population


def _print_done(verbose: bool) -> None:
    if verbose:
        print(
            f'{colorama.Fore.GREEN}{colorama.Style.BRIGHT}'
            f'\u2714 Training done.{colorama.Style.RESET_ALL}'
        )

    from .ozette import uninstall as _uninstall_ozette
    from .ozette import is_installed as _ozette_installed
    if _ozette_installed():
        _uninstall_ozette()
